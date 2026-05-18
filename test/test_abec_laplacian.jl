# ============================================================================
# Tests for the variable-coefficient ABec operator
# (L φ = A·α·φ - B·∇·(β·∇φ) = f).
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, allocate_phi_rho,
                          MGWorkspace, MGOptions, fill_field!,
                          PERIODIC, ABecCoefs, allocate_abec_coefs,
                          fill_abec_alpha!, fill_abec_beta!,
                          solve_abec!, apply_abec!, compute_abec_residual!,
                          add_patches!, EulerianFrame, HierarchicalMesh,
                          refine_cells!
using HierarchicalGrids.GeometricMultigrid: patches_at
using HierarchicalGrids.Overlap: FrameBoundaries, enumerate_leaves, cell_physical_box

function _abec_l2err(fields, ph, ws, ℓ, pi, phi_exact_fun, ::Val{D}) where {D}
    err2 = 0.0; nc = 0
    frame = patches_at(ph, ℓ)[pi]
    @inbounds for c in ws.patch_leaves[ℓ][pi]
        lo, hi = cell_physical_box(frame, c)
        x = ntuple(d -> 0.5*(lo[d]+hi[d]), Val(D))
        err2 += (fields[ℓ][pi].phi[c][1] - phi_exact_fun(x))^2
        nc += 1
    end
    return sqrt(err2 / nc)
end

@testset "ABec: const-coef reduces to Poisson" begin
    # L φ = -∇²φ = -ρ_p (with B=1, β=1, A=0, f = -ρ_poisson)
    # Use ρ_p = -8π² sin·sin, exact φ = sin·sin.
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 64
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    phi_fun(x) = sin(2π*x[1]) * sin(2π*x[2])
    fill_field!(fields, ph, :rho, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
    fill_field!(fields, ph, :phi, x -> 0.0)

    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg, pcg_precond = :jacobi)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 0.0, B = 1.0, beta_init = 1.0)
    r = solve_abec!(ws, fields, coefs)

    @test r.converged
    @test r.res_final / r.res_init < 1e-9
    err = _abec_l2err(fields, ph, ws, 1, 1, phi_fun, Val(2))
    # 2nd-order discretisation on a 64² grid.
    @test err < 1e-3
end

@testset "ABec: variable-β MMS recovery" begin
    # Apply the discrete operator to the analytic φ to get a consistent
    # discrete RHS, then solve back and verify recovery to machine precision.
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 64
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    β_fun(x) = 1.0 + 0.5 * cos(2π*x[1]) * cos(2π*x[2])
    phi_fun(x) = sin(2π*x[1]) * sin(2π*x[2])

    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg, pcg_precond = :jacobi)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 0.0, B = 1.0, beta_init = 1.0)
    fill_abec_beta!(coefs, ws, β_fun)

    # Set exact φ, compute discrete L φ, store as RHS, then solve.
    fill_field!(fields, ph, :phi, phi_fun)
    Lphi = allocate_phi_rho(ph)
    apply_abec!(Lphi, fields, coefs, ws; level_range = 1:1)
    @inbounds for c in ws.patch_leaves[1][1]
        fields[1][1].rho[c] = (Lphi[1][1].phi[c][1],)
    end
    fill_field!(fields, ph, :phi, x -> 0.0)
    r = solve_abec!(ws, fields, coefs)
    @test r.converged
    err = _abec_l2err(fields, ph, ws, 1, 1, phi_fun, Val(2))
    # Exact recovery — only Krylov tolerance limits the answer.
    @test err < 1e-9
end

@testset "ABec: implicit heat backward Euler" begin
    # (I - Δt·D ∇²) φ = sin·sin.  Analytic answer is sin·sin / (1 + Δt·D·k²).
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 64
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    Δt = 0.01; D_diff = 1.0
    fill_field!(fields, ph, :rho, x -> sin(2π*x[1]) * sin(2π*x[2]))
    fill_field!(fields, ph, :phi, x -> 0.0)

    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg, pcg_precond = :jacobi)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 1.0, B = Δt * D_diff,
                                 alpha_init = 1.0, beta_init = 1.0)
    r = solve_abec!(ws, fields, coefs)

    @test r.converged
    decay = 1.0 / (1.0 + Δt * D_diff * (4π^2 + 4π^2))
    phi_BE(x) = decay * sin(2π*x[1]) * sin(2π*x[2])
    err = _abec_l2err(fields, ph, ws, 1, 1, phi_BE, Val(2))
    # Backward-Euler 2nd-order spatial discretisation.
    @test err < 1e-3
end

@testset "ABec: variable-α and β together (Helmholtz)" begin
    # Solve (α(x) - ∇·(β(x)∇))·φ = f where we manufacture f.
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 64
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    phi_fun(x) = sin(2π*x[1]) * sin(2π*x[2])
    α_fun(x) = 2.0 + cos(2π*x[1])
    β_fun(x) = 1.0 + 0.5 * sin(2π*x[2])^2

    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg, pcg_precond = :jacobi)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 1.0, B = 1.0)
    fill_abec_alpha!(coefs, ws, α_fun)
    fill_abec_beta!(coefs, ws, β_fun)

    # Manufacture RHS from the discrete operator applied to phi_fun.
    fill_field!(fields, ph, :phi, phi_fun)
    Lphi = allocate_phi_rho(ph)
    apply_abec!(Lphi, fields, coefs, ws; level_range = 1:1)
    @inbounds for c in ws.patch_leaves[1][1]
        fields[1][1].rho[c] = (Lphi[1][1].phi[c][1],)
    end
    fill_field!(fields, ph, :phi, x -> 0.0)
    r = solve_abec!(ws, fields, coefs)
    @test r.converged
    err = _abec_l2err(fields, ph, ws, 1, 1, phi_fun, Val(2))
    @test err < 1e-9
end

@testset "ABec: residual computation matches apply" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    fill_field!(fields, ph, :phi, x -> 0.7 * sin(2π*x[1]))
    fill_field!(fields, ph, :rho, x -> 0.3 + cos(4π*x[2]))

    opts = MGOptions(tol = 1e-9, maxiter = 50, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 1.0, B = 0.5)
    fill_abec_alpha!(coefs, ws, x -> 1.0 + 0.3 * cos(2π*x[1]))
    fill_abec_beta!(coefs, ws, x -> 1.0 + 0.5 * sin(2π*x[2]))

    Lphi = allocate_phi_rho(ph)
    apply_abec!(Lphi, fields, coefs, ws; level_range = 1:1)
    rfield = allocate_phi_rho(ph)
    compute_abec_residual!(rfield, fields, fields, coefs, ws; level_range = 1:1)

    # r ≡ ρ - L φ ⇒ Lφ + r should equal ρ at every leaf.
    rmax = 0.0
    @inbounds for c in ws.patch_leaves[1][1]
        v = Lphi[1][1].phi[c][1] + rfield[1][1].phi[c][1] - fields[1][1].rho[c][1]
        rmax = max(rmax, abs(v))
    end
    @test rmax < 1e-12
end
