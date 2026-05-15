# Convergence + performance benchmarks for the GeometricMultigrid Poisson
# solver. Sweeps grid sizes and reports L² discretisation error, V-cycle
# count, residual reduction, and wall-clock per solve.
#
# Run with:
#   julia --project benchmarks/poisson_convergence.jl

using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, allocate_phi_rho,
                          MGWorkspace, MGOptions, fill_field!,
                          solve_poisson!,
                          PERIODIC, DIRICHLET, REFLECTING,
                          add_patches!, EulerianFrame,
                          HierarchicalMesh, refine_cells!, enumerate_leaves
using HierarchicalGrids.Overlap: FrameBoundaries, cell_physical_box
using Printf: @printf

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

function l2_error_on_level(fields, ph, ws, ℓ::Int, phi_fun)
    err = 0.0; n = 0
    D = length(ph.levels[1][1].lo)
    for (pi, frame) in enumerate(ph.levels[ℓ])
        for c in ws.patch_leaves[ℓ][pi]
            p_lo, p_hi = cell_physical_box(frame, c)
            center = ntuple(d -> (p_lo[d] + p_hi[d]) / 2, D)
            ex = phi_fun(center)
            ph_v = fields[ℓ][pi].phi[c][1]
            err += (ph_v - ex)^2
            n += 1
        end
    end
    return sqrt(err / max(n, 1))
end

function report(label, N, dim, fft_ok, iters, r0, rf, errs, t_wall_ns)
    @printf("%-22s | N=%4d^%d | fft=%-1s | iters=%2d | r0=%.3e | rf=%.3e | red=%.2e | L2[1]=%.3e",
            label, N, dim, fft_ok ? "Y" : "N", iters, r0, rf,
            r0 == 0 ? 0.0 : rf / r0, errs[1])
    length(errs) >= 2 && @printf(" | L2[2]=%.3e", errs[2])
    @printf(" | %.2f ms\n", t_wall_ns / 1e6)
end

# ----------------------------------------------------------------------------
# Single-level uniform sweeps
# ----------------------------------------------------------------------------

function bench_uniform(label, ::Val{D}, bcs_spec, Ns,
                       rho_fun, phi_fun) where {D}
    bcs = FrameBoundaries(bcs_spec)
    for N in Ns
        refines = Int(log2(N))
        lo = ntuple(_ -> 0.0, D); hi = ntuple(_ -> 1.0, D)
        ph = build_uniform_root_hierarchy(Val(D), refines, lo, hi;
                                           physical_bcs = bcs)
        fields = allocate_phi_rho(ph)
        fill_field!(fields, ph, :rho, rho_fun)
        ws = MGWorkspace(ph, bcs_spec;
                          opts = MGOptions(tol = 1e-11, maxiter = 20))
        # Warm up (FFTW plan / JIT) — first call is slower.
        solve_poisson!(fields, fields, ws)

        # Real timed run on a fresh phi=0.
        for ℓ in 1:length(fields), pi in 1:length(fields[ℓ])
            f = fields[ℓ][pi].phi
            for c in 1:length(f); f[c] = (0.0,); end
        end
        t0 = time_ns()
        result = solve_poisson!(fields, fields, ws)
        t1 = time_ns()

        err1 = l2_error_on_level(fields, ph, ws, 1, phi_fun)
        report(label, N, D, ws.fft_ok, result.iters, result.res_init,
               result.res_final, [err1], t1 - t0)
    end
end

# ----------------------------------------------------------------------------
# 2D AMR sweep
# ----------------------------------------------------------------------------

function bench_amr_2d(base_refines_range, fine_refines)
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    rho_fun(x) = -8π^2 * sin(2π * x[1]) * sin(2π * x[2])
    phi_fun(x) = sin(2π * x[1]) * sin(2π * x[2])

    for br in base_refines_range
        N = 2^br
        ph = build_uniform_root_hierarchy(Val(2), br, (0.0, 0.0), (1.0, 1.0);
                                           physical_bcs = bcs)
        fine_mesh = HierarchicalMesh{2}()
        for _ in 1:fine_refines
            refine_cells!(fine_mesh, enumerate_leaves(fine_mesh))
        end
        add_patches!(ph, 2, [EulerianFrame(fine_mesh, (0.375, 0.375), (0.625, 0.625))])

        fields = allocate_phi_rho(ph)
        fill_field!(fields, ph, :rho, rho_fun)
        ws = MGWorkspace(ph, bcs_spec;
                          opts = MGOptions(tol = 1e-10, maxiter = 20,
                                            n_pre = 2, n_post = 2))
        solve_poisson!(fields, fields, ws)

        for ℓ in 1:length(fields), pi in 1:length(fields[ℓ])
            f = fields[ℓ][pi].phi
            for c in 1:length(f); f[c] = (0.0,); end
        end
        t0 = time_ns()
        result = solve_poisson!(fields, fields, ws)
        t1 = time_ns()

        e1 = l2_error_on_level(fields, ph, ws, 1, phi_fun)
        e2 = l2_error_on_level(fields, ph, ws, 2, phi_fun)
        report("AMR 2D", N, 2, ws.fft_ok, result.iters, result.res_init,
               result.res_final, [e1, e2], t1 - t0)
    end
end

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

println("="^110)
println("  GeometricMultigrid Poisson solver — convergence & performance benchmarks")
println("="^110)

println("\n--- 2D periodic   ρ = −8π² sin(2π x) sin(2π y) -------------------------------------")
bench_uniform("2D periodic", Val(2),
              ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)),
              (32, 64, 128, 256),
              x -> -8π^2 * sin(2π * x[1]) * sin(2π * x[2]),
              x -> sin(2π * x[1]) * sin(2π * x[2]))

println("\n--- 3D periodic   ρ = −12π² sin(2π x) sin(2π y) sin(2π z) ----------------------------")
bench_uniform("3D periodic", Val(3),
              ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC), (PERIODIC, PERIODIC)),
              (16, 32, 64),
              x -> -12π^2 * sin(2π * x[1]) * sin(2π * x[2]) * sin(2π * x[3]),
              x -> sin(2π * x[1]) * sin(2π * x[2]) * sin(2π * x[3]))

println("\n--- 2D Dirichlet  φ = sin(π x) sin(π y) ----------------------------------------------")
bench_uniform("2D Dirichlet", Val(2),
              ((DIRICHLET, DIRICHLET), (DIRICHLET, DIRICHLET)),
              (32, 64, 128, 256),
              x -> -2π^2 * sin(π * x[1]) * sin(π * x[2]),
              x -> sin(π * x[1]) * sin(π * x[2]))

println("\n--- 2D Neumann    φ = cos(π x) cos(π y) ----------------------------------------------")
bench_uniform("2D Neumann", Val(2),
              ((REFLECTING, REFLECTING), (REFLECTING, REFLECTING)),
              (32, 64, 128, 256),
              x -> -2π^2 * cos(π * x[1]) * cos(π * x[2]),
              x -> cos(π * x[1]) * cos(π * x[2]))

println("\n--- 2D AMR (central refined patch, periodic outer domain) ----------------------------")
bench_amr_2d(4:7, 3)
