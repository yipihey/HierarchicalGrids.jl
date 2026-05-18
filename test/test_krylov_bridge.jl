# ============================================================================
# Tests for the Krylov.jl bridge: flat-vector wrapping of the FAC composite
# and ABec operators so that GMRES / BiCGStab / FGMRES / MINRES can drive
# solves on hierarchical-grid storage.
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, allocate_phi_rho,
                          MGWorkspace, MGOptions, fill_field!,
                          PERIODIC, allocate_abec_coefs, fill_abec_beta!,
                          ABecOp, FACCompositeOp, solve_with_krylov!,
                          abec_jacobi_precond, flat_layout, pack!, unpack!
using HierarchicalGrids.GeometricMultigrid: patches_at
using HierarchicalGrids.Overlap: FrameBoundaries

@testset "Krylov bridge: pack/unpack round-trip" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    ph = build_uniform_root_hierarchy(Val(2), 5, (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    fill_field!(fields, ph, :phi, x -> sin(2π*x[1]) * cos(2π*x[2]))
    opts = MGOptions(tol = 1e-9, maxiter = 50)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    layout = flat_layout(ws; level_range = 1:1)
    @test layout.n == 32 * 32

    flat = Vector{Float64}(undef, layout.n)
    pack!(flat, fields, layout)
    # Re-unpack into a fresh field and check exact agreement.
    fields2 = allocate_phi_rho(ph)
    unpack!(fields2, flat, layout)
    @inbounds for (ℓ, pi, c) in layout.cells
        @test fields[ℓ][pi].phi[c][1] == fields2[ℓ][pi].phi[c][1]
    end
end

@testset "Krylov bridge: const-coef Poisson via CG / GMRES / BiCGStab / MINRES" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 64
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_abec_coefs(ws; A = 0.0, B = 1.0)
    op = ABecOp(ws, coefs; level_range = 1:1)

    for method in (:cg, :gmres, :bicgstab, :minres)
        fill_field!(fields, ph, :phi, x -> 0.0)
        fill_field!(fields, ph, :rho, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
        r = solve_with_krylov!(fields, fields, op, ws;
                                  method = method, tol = 1e-10, maxiter = 200)
        @test r.converged
        @test r.res_final / r.res_init < 1e-9
    end
end

@testset "Krylov bridge: variable-β with all Krylov methods" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    opts = MGOptions(tol = 1e-10, maxiter = 500, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    β_fun(x) = 1.0 + 0.5 * cos(2π*x[1]) * cos(2π*x[2])
    coefs = allocate_abec_coefs(ws; A = 0.0, B = 1.0)
    fill_abec_beta!(coefs, ws, β_fun)
    op = ABecOp(ws, coefs; level_range = 1:1)

    for method in (:cg, :gmres, :bicgstab)
        fill_field!(fields, ph, :phi, x -> 0.0)
        fill_field!(fields, ph, :rho, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
        r = solve_with_krylov!(fields, fields, op, ws;
                                  method = method, tol = 1e-10, maxiter = 500)
        @test r.converged
        @test r.res_final / r.res_init < 1e-9
    end
end

@testset "Krylov bridge: FGMRES + Jacobi preconditioner" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    opts = MGOptions(tol = 1e-10, maxiter = 500, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    β_fun(x) = 1.0 + 0.5 * cos(2π*x[1]) * cos(2π*x[2])
    coefs = allocate_abec_coefs(ws; A = 0.0, B = 1.0)
    fill_abec_beta!(coefs, ws, β_fun)
    op = ABecOp(ws, coefs; level_range = 1:1)
    M = abec_jacobi_precond(ws, coefs; level_range = 1:1)

    fill_field!(fields, ph, :phi, x -> 0.0)
    fill_field!(fields, ph, :rho, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
    r = solve_with_krylov!(fields, fields, op, ws;
                              method = :fgmres, tol = 1e-10, maxiter = 500,
                              precond = M)
    @test r.converged
    @test r.res_final / r.res_init < 1e-9
end

@testset "Krylov bridge: FACCompositeOp on AMR" begin
    # 2-level AMR, check that FACCompositeOp wrapper is consistent with the
    # internal PCG-on-composite path.
    using HierarchicalGrids: add_patches!, EulerianFrame, HierarchicalMesh,
                              refine_cells!
    using HierarchicalGrids.Overlap: enumerate_leaves
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
    fill_field!(fields, ph, :rho, x -> -8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
    fill_field!(fields, ph, :phi, x -> 0.0)

    opts = MGOptions(tol = 1e-8, maxiter = 500, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    op = FACCompositeOp(ws; level_range = 1:2)
    # FACCompositeOp negates internally so A = -L_FAC.  The user RHS must
    # therefore be -ρ to recover the standard convention ∇²φ = ρ.
    r = solve_with_krylov!(fields, fields, op, ws;
                              method = :cg, tol = 1e-8, maxiter = 500,
                              rhs_sign = -1.0)
    @test r.converged
    @test r.res_final / r.res_init < 1e-7
end
