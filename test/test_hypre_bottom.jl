# ============================================================================
# Tests for the HYPRE BoomerAMG bottom solver wrapper (Tier 2 #6+).
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, allocate_phi_rho,
                          MGWorkspace, MGOptions, fill_field!,
                          PERIODIC, allocate_abec_coefs, fill_abec_beta!,
                          ABecOp, solve_with_krylov!,
                          hypre_preconditioner, hypre_precond_callback,
                          solve_abec_hypre!
using HierarchicalGrids.Overlap: FrameBoundaries

@testset "HYPRE BoomerAMG: direct solve (variable β)" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 0.0, B = 1.0, beta_init = 1.0)
    fill_abec_beta!(coefs, ws, x -> 1.0 + 0.5 * cos(2π*x[1]) * cos(2π*x[2]))

    fill_field!(fields, ph, :phi, x -> 0.0)
    fill_field!(fields, ph, :rho, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
    r = solve_abec_hypre!(fields, fields, coefs, ws;
                            solver = :boomeramg, tol = 1e-9, maxiter = 50)
    @test r.res_final / r.res_init < 1e-7
end

@testset "HYPRE BoomerAMG: as preconditioner inside FGMRES" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 0.0, B = 1.0, beta_init = 1.0)
    fill_abec_beta!(coefs, ws, x -> 1.0 + 0.5 * cos(2π*x[1]) * cos(2π*x[2]))

    op = ABecOp(ws, coefs; level_range = 1:1)
    hp = hypre_preconditioner(ws, coefs)
    M = hypre_precond_callback(hp)
    fill_field!(fields, ph, :phi, x -> 0.0)
    fill_field!(fields, ph, :rho, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
    r = solve_with_krylov!(fields, fields, op, ws;
                              method = :fgmres, tol = 1e-10, maxiter = 50,
                              precond = M)
    @test r.converged
    # BoomerAMG should make FGMRES converge in single digits of iterations.
    @test r.iters <= 10
    @test r.res_final / r.res_init < 1e-9
end
