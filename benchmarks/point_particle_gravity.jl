# Point-particle gravity test on a deep nested 3D hierarchy.
#
# Geometry: each level is a 16^3 patch covering half the parent's extent,
# centered. dx_ℓ = dx_1 / 2^(ℓ-1).
#
# Source: Gaussian point particle, σ = dx_finest, total mass 1, sourcing
# ∇²φ = 4π ρ (i.e. set the solver's ρ = 4π · gaussian).
#
# Particle trajectory: starts 2.5·dx_finest from origin along +x, moves to
# 4.5·dx_finest over 5 steps. We compare PCG-Jacobi cold vs warm start
# convergence and per-step wall time.
#
# (V-cycle alone diverges on 3+ level nested hierarchies — its smoother
# doesn't apply the FAC operator at intermediate-level C/F faces.
# PCG-Jacobi uses the correct composite operator everywhere and converges.)

using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, allocate_phi_rho,
                          MGWorkspace, MGOptions, fill_field!,
                          solve_poisson!,
                          PERIODIC, add_patches!, EulerianFrame,
                          HierarchicalMesh, refine_cells!
using HierarchicalGrids.Overlap: FrameBoundaries, cell_physical_box,
                                  enumerate_leaves
using Printf

function build_nested_hierarchy(::Val{D}, n_levels::Int,
                                  N_per_level::Int) where {D}
    bcs_spec = ntuple(_ -> (PERIODIC, PERIODIC), D)
    bcs = FrameBoundaries(bcs_spec)
    root_lo = ntuple(_ -> -1.0, D)
    root_hi = ntuple(_ ->  1.0, D)
    refines = Int(log2(N_per_level))
    ph = build_uniform_root_hierarchy(Val(D), refines, root_lo, root_hi;
                                       physical_bcs = bcs)
    for ℓ in 2:n_levels
        L = 2.0 / 2.0^(ℓ - 1)
        lo = ntuple(_ -> -L/2, D)
        hi = ntuple(_ ->  L/2, D)
        mesh = HierarchicalMesh{D}()
        for _ in 1:refines
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        add_patches!(ph, ℓ, [EulerianFrame(mesh, lo, hi)])
    end
    return ph, bcs_spec
end

function set_particle_source!(fields, ph, x_p::NTuple{D,Float64},
                              σ::Float64, M::Float64) where {D}
    fill_field!(fields, ph, :rho, x -> begin
        r2 = sum(d -> (x[d] - x_p[d])^2, 1:D)
        4π * M * exp(-r2 / (2 * σ * σ)) / (2π * σ * σ)^(D/2)
    end)
end

# L2 error on the FINEST level, restricted to r > 2σ (outside the
# smoothed source where φ ≈ −1/r applies).
function l2_error_finest(fields, ph, ws, x_p::NTuple{D,Float64},
                          σ::Float64, M::Float64) where {D}
    ℓ_finest = length(ph.levels)
    frame = ph.levels[ℓ_finest][1]
    err_sum = 0.0; n = 0
    @inbounds for c in ws.patch_leaves[ℓ_finest][1]
        p_lo, p_hi = cell_physical_box(frame, c)
        center = ntuple(d -> (p_lo[d] + p_hi[d]) / 2, D)
        r = sqrt(sum(d -> (center[d] - x_p[d])^2, 1:D))
        r > 2 * σ || continue
        φ_num = fields[ℓ_finest][1].phi[c][1]
        φ_exact = -M / r
        err_sum += (φ_num - φ_exact)^2
        n += 1
    end
    return n == 0 ? 0.0 : sqrt(err_sum / n), n
end

const D = 3
const N_per_level = 16
const n_levels = 10

println("="^118)
println("  Point-particle gravity on $(n_levels)-level nested $(N_per_level)^$D 3D hierarchy")
println("="^118)

ph, bcs_spec = build_nested_hierarchy(Val(D), n_levels, N_per_level)
ℓ_finest = n_levels
dx_finest = (ph.levels[ℓ_finest][1].hi[1] - ph.levels[ℓ_finest][1].lo[1]) / N_per_level
σ_source = dx_finest
M = 1.0
@printf("Finest level dx = %.3e (= L_root / %d)\n", dx_finest, Int(2.0 / dx_finest))
@printf("σ = dx_finest = %.3e,  M = %g\n\n", σ_source, M)

x_start = (2.5 * dx_finest, 0.0, 0.0)
x_end   = (4.5 * dx_finest, 0.0, 0.0)
n_steps = 5
positions = [ntuple(d -> x_start[d] + (x_end[d] - x_start[d]) * (i / n_steps), D)
              for i in 0:n_steps]

# Run trajectory: cold OR warm starts.
function run_trajectory(label, ph, bcs_spec, positions, σ_source, M;
                        warm::Bool, tol::Float64, maxiter::Int,
                        cycle::Symbol = :pcg, n_pre::Int = 2, n_post::Int = 2)
    fields = allocate_phi_rho(ph)
    opts = if cycle == :pcg
        MGOptions(tol = tol, maxiter = maxiter, cycle = :pcg,
                   pcg_precond = :jacobi,
                   n_pre = n_pre, n_post = n_post)
    else
        MGOptions(tol = tol, maxiter = maxiter, cycle = :vcycle,
                   n_pre = n_pre, n_post = n_post)
    end
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    # Warm up the JIT once.
    set_particle_source!(fields, ph, positions[1], σ_source, M)
    fill_field!(fields, ph, :phi, x -> 0.0)
    solve_poisson!(ws, fields)

    results = Vector{NamedTuple{(:step, :iters, :r0, :rf, :err, :ms),
                                  Tuple{Int,Int,Float64,Float64,Float64,Float64}}}()
    for (step, x_p) in enumerate(positions)
        if !warm
            fill_field!(fields, ph, :phi, x -> 0.0)
        end
        set_particle_source!(fields, ph, x_p, σ_source, M)
        t0 = time_ns()
        result = solve_poisson!(ws, fields)
        t1 = time_ns()
        l2, _ = l2_error_finest(fields, ph, ws, x_p, σ_source, M)
        push!(results, (step = step - 1, iters = result.iters,
                          r0 = result.res_init, rf = result.res_final,
                          err = l2, ms = (t1 - t0) / 1e6))
    end
    return results
end

function print_table(label, results)
    println(label)
    @printf("  step | iters |    r0       |    rf       |   L2 err    |  wall (ms)\n")
    @printf("  -----+-------+-------------+-------------+-------------+-----------\n")
    for r in results
        @printf("  %4d | %5d | %.3e   | %.3e   | %.3e   | %8.1f\n",
                r.step, r.iters, r.r0, r.rf, r.err, r.ms)
    end
    total_ms = sum(r -> r.ms, results)
    total_iters = sum(r -> r.iters, results)
    @printf("                                                          TOTAL: %.1f ms (%d iters across %d steps)\n\n",
            total_ms, total_iters, length(results))
end

const TOL = 1e-9      # tight target so iteration counts reflect real convergence
const MAXITER = 2000

println(">>> PCG-Jacobi COLD start:")
r_pcg_cold = run_trajectory("PCG COLD", ph, bcs_spec, positions, σ_source, M;
                              warm = false, tol = TOL, maxiter = MAXITER,
                              cycle = :pcg)
print_table("PCG COLD", r_pcg_cold)

println(">>> PCG-Jacobi WARM start:")
r_pcg_warm = run_trajectory("PCG WARM", ph, bcs_spec, positions, σ_source, M;
                              warm = true, tol = TOL, maxiter = MAXITER,
                              cycle = :pcg)
print_table("PCG WARM", r_pcg_warm)

println(">>> V-cycle (n_pre=n_post=50) COLD start:")
r_vc_cold = run_trajectory("V-cycle COLD", ph, bcs_spec, positions, σ_source, M;
                              warm = false, tol = TOL, maxiter = MAXITER,
                              cycle = :vcycle, n_pre = 50, n_post = 50)
print_table("V-cycle COLD", r_vc_cold)

println(">>> V-cycle (n_pre=n_post=50) WARM start:")
r_vc_warm = run_trajectory("V-cycle WARM", ph, bcs_spec, positions, σ_source, M;
                              warm = true, tol = TOL, maxiter = MAXITER,
                              cycle = :vcycle, n_pre = 50, n_post = 50)
print_table("V-cycle WARM", r_vc_warm)

println("="^118)
println("  Cold vs warm comparison:")
for (label, cold, warm) in [("PCG", r_pcg_cold, r_pcg_warm),
                              ("V-cycle", r_vc_cold, r_vc_warm)]
    ct = sum(r -> r.ms, cold); wt = sum(r -> r.ms, warm)
    ci = sum(r -> r.iters, cold); wi = sum(r -> r.iters, warm)
    @printf("  %s: cold=%.0f ms (%d iters), warm=%.0f ms (%d iters), speedup=%.2fx wall, %.2fx iter\n",
            label, ct, ci, wt, wi, ct/wt, ci/wi)
end
