# ============================================================================
# Tests for the AlgebraicMultigrid bottom-solver wrapper.
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, allocate_phi_rho,
                          MGWorkspace, MGOptions, fill_field!,
                          PERIODIC, allocate_abec_coefs, fill_abec_alpha!,
                          fill_abec_beta!, ABecOp, solve_with_krylov!,
                          amg_preconditioner, amg_precond_callback,
                          assemble_abec_matrix
using HierarchicalGrids.Overlap: FrameBoundaries
using LinearAlgebra: norm

@testset "AMG: matrix assembly matches matvec" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 16
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-9, maxiter = 50, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 1.0, B = 0.5)
    fill_abec_alpha!(coefs, ws, x -> 1.0 + 0.3 * cos(2π*x[1]))
    fill_abec_beta!(coefs, ws, x -> 1.0 + 0.5 * sin(2π*x[2]))

    A = assemble_abec_matrix(ws, coefs)
    @test size(A, 1) == N * N
    @test size(A, 2) == N * N
    # 5-point stencil in 2D: each row has at most 5 nonzeros.
    @test maximum(diff(A.colptr)) <= 5

    # Compare matrix vector product against the matrix-free apply.
    op = ABecOp(ws, coefs; level_range = 1:1)
    x_flat = randn(N * N)
    Ax_dense = A * x_flat
    Ax_mf = similar(x_flat)
    HierarchicalGrids.GeometricMultigrid.mul!(Ax_mf, op, x_flat)
    @test isapprox(Ax_dense, Ax_mf; rtol = 1e-12)
end

@testset "AMG: FGMRES+AMG converges in O(1) iterations" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 64
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    opts = MGOptions(tol = 1e-10, maxiter = 500, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 0.0, B = 1.0)
    fill_abec_beta!(coefs, ws, x -> 1.0 + 0.5 * cos(2π*x[1]) * cos(2π*x[2]))
    op = ABecOp(ws, coefs; level_range = 1:1)
    amg = amg_preconditioner(ws, coefs; method = :rs)
    M = amg_precond_callback(amg)

    fill_field!(fields, ph, :phi, x -> 0.0)
    fill_field!(fields, ph, :rho, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
    r = solve_with_krylov!(fields, fields, op, ws;
                              method = :fgmres, tol = 1e-10, maxiter = 200,
                              precond = M)
    @test r.converged
    @test r.res_final / r.res_init < 1e-9
    # AMG-preconditioned FGMRES should converge in well under 20 iters
    # (Jacobi-preconditioned needed ~50 on this problem).
    @test r.iters < 20
end

@testset "AMG: smoothed-aggregation also works" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    opts = MGOptions(tol = 1e-10, maxiter = 500, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 0.0, B = 1.0)
    fill_abec_beta!(coefs, ws, x -> 1.0 + 0.3 * sin(2π*x[1])^2)
    op = ABecOp(ws, coefs; level_range = 1:1)
    amg = amg_preconditioner(ws, coefs; method = :sa)
    M = amg_precond_callback(amg)

    fill_field!(fields, ph, :phi, x -> 0.0)
    fill_field!(fields, ph, :rho, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
    r = solve_with_krylov!(fields, fields, op, ws;
                              method = :fgmres, tol = 1e-10, maxiter = 200,
                              precond = M)
    @test r.converged
    @test r.iters < 30
end
