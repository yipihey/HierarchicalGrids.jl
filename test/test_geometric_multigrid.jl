using Test
using Random
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, allocate_phi_rho,
                          MGWorkspace, MGOptions, fill_field!,
                          solve_poisson!, apply_laplacian!, compute_residual!,
                          residual_l2,
                          PERIODIC, DIRICHLET, REFLECTING,
                          add_patches!, EulerianFrame,
                          HierarchicalMesh, refine_cells!, enumerate_leaves
using HierarchicalGrids.Overlap: FrameBoundaries, cell_physical_box

# Helper: compute L² error against an analytic solution on a per-level basis.
function l2_error(fields, ph, ws, phi_fun)
    err_per_level = Float64[]
    for ℓ in 1:length(fields)
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
        push!(err_per_level, sqrt(err / max(n, 1)))
    end
    return err_per_level
end

# ----------------------------------------------------------------------------
# Test 1: 2D periodic MMS, convergence vs grid size
# ----------------------------------------------------------------------------

@testset "GeometricMultigrid: 2D periodic MMS (FFT bottom)" begin
    rho_fun(x) = -8π^2 * sin(2π * x[1]) * sin(2π * x[2])
    phi_fun(x) = sin(2π * x[1]) * sin(2π * x[2])
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)

    errs = Float64[]
    Ns = (32, 64, 128)
    for N in Ns
        refines = Int(log2(N))
        ph = build_uniform_root_hierarchy(Val(2), refines, (0.0, 0.0), (1.0, 1.0);
                                           physical_bcs = bcs)
        fields = allocate_phi_rho(ph)
        fill_field!(fields, ph, :rho, rho_fun)
        ws = MGWorkspace(ph, bcs_spec; opts = MGOptions(tol = 1e-12, maxiter = 5))
        result = solve_poisson!(fields, fields, ws)
        @test ws.fft_ok
        @test result.converged
        @test result.iters == 1   # FFT bottom is an exact direct solve.
        @test result.res_final < 1e-9 * result.res_init || result.res_final < 1e-9
        push!(errs, l2_error(fields, ph, ws, phi_fun)[1])
    end
    # 2nd-order convergence: each doubling of N halves dx, error ÷4.
    for k in 2:length(Ns)
        ratio = errs[k - 1] / errs[k]
        @test 3.5 <= ratio <= 4.5
    end
end

# ----------------------------------------------------------------------------
# Test 2: 3D periodic MMS
# ----------------------------------------------------------------------------

@testset "GeometricMultigrid: 3D periodic MMS" begin
    rho_fun(x) = -12π^2 * sin(2π * x[1]) * sin(2π * x[2]) * sin(2π * x[3])
    phi_fun(x) = sin(2π * x[1]) * sin(2π * x[2]) * sin(2π * x[3])
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)

    errs = Float64[]
    Ns = (8, 16, 32)
    for N in Ns
        refines = Int(log2(N))
        ph = build_uniform_root_hierarchy(Val(3), refines,
                                           (0.0, 0.0, 0.0), (1.0, 1.0, 1.0);
                                           physical_bcs = bcs)
        fields = allocate_phi_rho(ph)
        fill_field!(fields, ph, :rho, rho_fun)
        ws = MGWorkspace(ph, bcs_spec; opts = MGOptions(tol = 1e-12, maxiter = 5))
        result = solve_poisson!(fields, fields, ws)
        @test ws.fft_ok
        @test result.converged
        @test result.iters == 1
        push!(errs, l2_error(fields, ph, ws, phi_fun)[1])
    end
    for k in 2:length(Ns)
        ratio = errs[k - 1] / errs[k]
        @test 3.5 <= ratio <= 4.5
    end
end

# ----------------------------------------------------------------------------
# Test 3: 2D Dirichlet MMS
# ----------------------------------------------------------------------------

@testset "GeometricMultigrid: 2D Dirichlet MMS (DST bottom)" begin
    rho_fun(x) = -2π^2 * sin(π * x[1]) * sin(π * x[2])
    phi_fun(x) = sin(π * x[1]) * sin(π * x[2])
    bcs_spec = ((DIRICHLET, DIRICHLET), (DIRICHLET, DIRICHLET))
    bcs = FrameBoundaries(bcs_spec)

    errs = Float64[]
    Ns = (32, 64, 128)
    for N in Ns
        refines = Int(log2(N))
        ph = build_uniform_root_hierarchy(Val(2), refines, (0.0, 0.0), (1.0, 1.0);
                                           physical_bcs = bcs)
        fields = allocate_phi_rho(ph)
        fill_field!(fields, ph, :rho, rho_fun)
        ws = MGWorkspace(ph, bcs_spec; opts = MGOptions(tol = 1e-12, maxiter = 5))
        result = solve_poisson!(fields, fields, ws)
        @test ws.fft_ok
        @test result.converged
        @test result.iters == 1
        push!(errs, l2_error(fields, ph, ws, phi_fun)[1])
    end
    for k in 2:length(Ns)
        @test 3.5 <= errs[k - 1] / errs[k] <= 4.5
    end
end

# ----------------------------------------------------------------------------
# Test 4: 2D Neumann MMS
# ----------------------------------------------------------------------------

@testset "GeometricMultigrid: 2D Neumann MMS (DCT bottom)" begin
    # φ = cos(πx)cos(πy) has ∂φ/∂x = 0 at x=0 and x=1.
    rho_fun(x) = -2π^2 * cos(π * x[1]) * cos(π * x[2])
    phi_fun(x) = cos(π * x[1]) * cos(π * x[2])
    bcs_spec = ((REFLECTING, REFLECTING), (REFLECTING, REFLECTING))
    bcs = FrameBoundaries(bcs_spec)

    errs = Float64[]
    Ns = (32, 64, 128)
    for N in Ns
        refines = Int(log2(N))
        ph = build_uniform_root_hierarchy(Val(2), refines, (0.0, 0.0), (1.0, 1.0);
                                           physical_bcs = bcs)
        fields = allocate_phi_rho(ph)
        fill_field!(fields, ph, :rho, rho_fun)
        ws = MGWorkspace(ph, bcs_spec; opts = MGOptions(tol = 1e-12, maxiter = 5))
        result = solve_poisson!(fields, fields, ws)
        @test ws.fft_ok
        @test result.converged
        @test result.iters == 1
        push!(errs, l2_error(fields, ph, ws, phi_fun)[1])
    end
    for k in 2:length(Ns)
        @test 3.5 <= errs[k - 1] / errs[k] <= 4.5
    end
end

# ----------------------------------------------------------------------------
# Test 5: 2D AMR — single central refined patch with periodic outer domain.
#
# The degree-0 ghost prolongation at the C/F interface gives 1st-order
# coupling at the interface, so the V-cycle reduces residual by ~10x in the
# first cycle but stalls at the interface-error level. The composite
# solution still recovers the analytic solution to discretization order on
# the coarse level (and to ~10x worse on the fine level due to the 1st-order
# coupling). Full Martin–Colella linear ghost is a v2 item — see plan §G.
# ----------------------------------------------------------------------------

@testset "GeometricMultigrid: 2D AMR static refinement (periodic)" begin
    rho_fun(x) = -8π^2 * sin(2π * x[1]) * sin(2π * x[2])
    phi_fun(x) = sin(2π * x[1]) * sin(2π * x[2])
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    ph = build_uniform_root_hierarchy(Val(2), 5, (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fine_mesh = HierarchicalMesh{2}()
    for _ in 1:4
        refine_cells!(fine_mesh, enumerate_leaves(fine_mesh))
    end
    add_patches!(ph, 2, [EulerianFrame(fine_mesh, (0.375, 0.375), (0.625, 0.625))])

    fields = allocate_phi_rho(ph)
    fill_field!(fields, ph, :rho, rho_fun)
    ws = MGWorkspace(ph, bcs_spec;
                      opts = MGOptions(tol = 1e-10, maxiter = 10,
                                        n_pre = 2, n_post = 2))
    result = solve_poisson!(fields, fields, ws)

    # V-cycle drops residual by ~10x in the first cycle.
    @test result.history[2] < 0.15 * result.history[1]
    # Then stalls at ~3% relative residual (degree-0 interface limitation).
    @test result.res_final < 0.05 * result.res_init

    errs = l2_error(fields, ph, ws, phi_fun)
    # Level 1 (uncovered + covered): pure 32² discretization error ~ 0.002.
    @test errs[1] < 5e-3
    # Level 2 (fine 16² patch): elevated by C/F interface limitation.
    @test errs[2] < 5e-2
end

# ----------------------------------------------------------------------------
# Test 6: Partial-hierarchy / subcycling
#
# Two-level hierarchy. Initialise both levels with the analytic solution,
# corrupt level 2 with noise, solve only level 2 with frozen level 1.
# Verify level 1 is bit-identical and level 2 is recovered to discretization.
# ----------------------------------------------------------------------------

@testset "GeometricMultigrid: partial-hierarchy / subcycling" begin
    rho_fun(x) = -8π^2 * sin(2π * x[1]) * sin(2π * x[2])
    phi_fun(x) = sin(2π * x[1]) * sin(2π * x[2])
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    ph = build_uniform_root_hierarchy(Val(2), 5, (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fine_mesh = HierarchicalMesh{2}()
    for _ in 1:4
        refine_cells!(fine_mesh, enumerate_leaves(fine_mesh))
    end
    add_patches!(ph, 2, [EulerianFrame(fine_mesh, (0.375, 0.375), (0.625, 0.625))])

    fields = allocate_phi_rho(ph)
    fill_field!(fields, ph, :rho, rho_fun)
    fill_field!(fields, ph, :phi, phi_fun)
    phi1_init = [fields[1][1].phi[c][1] for c in 1:length(fields[1][1].phi)]

    Random.seed!(123)
    for c in 1:length(fields[2][1].phi)
        fields[2][1].phi[c] = (fields[2][1].phi[c][1] + 0.1 * randn(),)
    end

    ws = MGWorkspace(ph, bcs_spec;
                      level_range = 2:2,
                      opts = MGOptions(tol = 1e-9, maxiter = 50,
                                        bottom_smooth_iters = 100))
    result = solve_poisson!(fields, fields, ws; level_range = 2:2)

    # FFT bottom must be disabled (level 2 is a sub-patch, not the full domain).
    @test !ws.fft_ok
    @test result.converged
    # Level 1 must be bit-identical (no touching outside level_range).
    phi1_after = [fields[1][1].phi[c][1] for c in 1:length(fields[1][1].phi)]
    @test maximum(abs.(phi1_after .- phi1_init)) == 0.0
    # Level 2 recovered to ~discretization error.
    errs = l2_error(fields, ph, ws, phi_fun)
    @test errs[2] < 5e-2
end

@testset "GeometricMultigrid: Schur bottom solver (opt-in)" begin
    rho_fun(x) = -8π^2 * sin(2π * x[1]) * sin(2π * x[2])
    phi_fun(x) = sin(2π * x[1]) * sin(2π * x[2])
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    ph = build_uniform_root_hierarchy(Val(2), 5, (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fine_mesh = HierarchicalMesh{2}()
    for _ in 1:4
        refine_cells!(fine_mesh, enumerate_leaves(fine_mesh))
    end
    add_patches!(ph, 2, [EulerianFrame(fine_mesh, (0.375, 0.375), (0.625, 0.625))])

    fields = allocate_phi_rho(ph)
    fill_field!(fields, ph, :rho, rho_fun)
    fill_field!(fields, ph, :phi, phi_fun)
    Random.seed!(7)
    for c in 1:length(fields[2][1].phi)
        fields[2][1].phi[c] = (fields[2][1].phi[c][1] + 0.1 * randn(),)
    end

    ws = MGWorkspace(ph, bcs_spec;
                      level_range = 2:2,
                      opts = MGOptions(tol = 1e-9, maxiter = 5,
                                        bottom_solver = :schur))
    result = solve_poisson!(fields, fields, ws; level_range = 2:2)

    # Schur is a direct solver — should converge in 1 V-cycle to machine
    # residual on the discrete operator with Dirichlet-from-parent.
    @test result.converged
    @test result.iters == 1
    @test result.res_final < 1e-9 * result.res_init

    # L2 error matches the AMR static-refinement test (same operator).
    errs = l2_error(fields, ph, ws, phi_fun)
    @test errs[2] < 1e-2
end

# ----------------------------------------------------------------------------
# Test 7: apply_laplacian! recovers the analytic Laplacian to 2nd order
# ----------------------------------------------------------------------------

@testset "GeometricMultigrid: PCG-on-composite drives residual to machine precision" begin
    # The V-cycle stalls at a prolongation-coupling fixed point on AMR;
    # PCG eliminates it by treating uncovered-coarse + fine cells as one
    # SPD system and running preconditioned CG.
    rho_fun(x) = -8π^2 * sin(2π * x[1]) * sin(2π * x[2])
    phi_fun(x) = sin(2π * x[1]) * sin(2π * x[2])
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    ph = build_uniform_root_hierarchy(Val(2), 5, (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fine_mesh = HierarchicalMesh{2}()
    for _ in 1:4
        refine_cells!(fine_mesh, enumerate_leaves(fine_mesh))
    end
    add_patches!(ph, 2, [EulerianFrame(fine_mesh, (0.375, 0.375), (0.625, 0.625))])
    fields = allocate_phi_rho(ph)
    fill_field!(fields, ph, :rho, rho_fun)
    ws = MGWorkspace(ph, bcs_spec;
                      opts = MGOptions(tol = 1e-8, maxiter = 200,
                                        cycle = :pcg))
    result = solve_poisson!(ws, fields)
    @test result.converged
    # Should drive residual several orders of magnitude below the V-cycle's
    # ~1% relative fixed point. (V-cycle stalls around 0.5; PCG-Jacobi
    # converges to ~1e-7.)
    @test result.res_final < 1e-6
    @test result.res_final / result.res_init < 1e-7
    errs = l2_error(fields, ph, ws, phi_fun)
    # Solution accuracy unchanged from the V-cycle's stalled value —
    # discretization-limited at the C/F interface.
    @test errs[2] < 1e-2
end

@testset "GeometricMultigrid: refinement listener invalidates caches" begin
    using HierarchicalGrids: refine_cells!
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    ph = build_uniform_root_hierarchy(Val(2), 4, (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    ws = MGWorkspace(ph, bcs_spec)
    @test ws.fft_ok          # configured at construction
    # Trigger a refinement event on the root mesh by refining one leaf.
    mesh = ph.levels[1][1].mesh
    refine_cells!(mesh, [enumerate_leaves(mesh)[1]])
    @test !ws.fft_ok         # listener cleared FFT
    @test isempty(ws.schur_factors)
    release!(ws)
end

@testset "GeometricMultigrid: allocation regression (warm solve)" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    ph = build_uniform_root_hierarchy(Val(2), 5, (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    fill_field!(fields, ph, :rho, x -> -8π^2 * sin(2π * x[1]) * sin(2π * x[2]))
    ws = MGWorkspace(ph, bcs_spec; opts = MGOptions(tol = 1e-10, maxiter = 5))
    # Warm up the JIT.
    solve_poisson!(ws, fields)
    # Reset phi to zero for a fair re-solve.
    for ℓ in 1:length(fields), pi in 1:length(fields[ℓ])
        f = fields[ℓ][pi].phi
        for c in 1:length(f); f[c] = (0.0,); end
    end
    # Allocation budget: one warm V-cycle on a 32² periodic problem should
    # allocate well under 64 KB. The exact number changes with Julia and
    # FFTW versions; the bar guards against per-cell allocations creeping
    # back into the hot path.
    bytes = @allocated solve_poisson!(ws, fields)
    @test bytes < 64_000
end

@testset "GeometricMultigrid: apply_laplacian! consistency" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    refines = 6
    ph = build_uniform_root_hierarchy(Val(2), refines, (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    fill_field!(fields, ph, :phi, x -> sin(2π * x[1]) * sin(2π * x[2]))

    ws = MGWorkspace(ph, bcs_spec)
    Lphi = allocate_phi_rho(ph)
    apply_laplacian!(Lphi, fields, ws; level_range = 1:1)

    err = 0.0; n = 0
    for c in ws.patch_leaves[1][1]
        p_lo, p_hi = cell_physical_box(ph.levels[1][1], c)
        x = (p_lo[1] + p_hi[1]) / 2; y = (p_lo[2] + p_hi[2]) / 2
        exact = -8π^2 * sin(2π * x) * sin(2π * y)
        computed = Lphi[1][1].phi[c][1]
        err += (computed - exact)^2; n += 1
    end
    rms_err = sqrt(err / n)
    # 2nd-order finite-volume L²-error: O(h²) on a smooth function.
    @test rms_err < 0.1
end
