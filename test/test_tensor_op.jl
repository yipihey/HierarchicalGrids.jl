# ============================================================================
# Tests for the full tensor viscosity operator with cross-component
# coupling (Tier 2 #5 full).
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, MGWorkspace, MGOptions,
                          PERIODIC, TensorCoefs, allocate_tensor_coefs,
                          TensorVelocity, allocate_tensor_velocity,
                          fill_tensor_velocity!, apply_tensor!, solve_tensor!
using HierarchicalGrids.Overlap: FrameBoundaries

@testset "TensorOp: apply reproduces analytic operator (incompressible)" begin
    # For divergence-free input, (μ+λ) term should vanish — operator
    # reduces to (I - Δt μ ∇²) per component.
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 64
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-9, maxiter = 50)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    Δt = 0.01; μ = 1.0; λ = 0.0
    coefs = allocate_tensor_coefs(ws; A = 1.0, B = Δt, mu = μ, lambda = λ,
                                     alpha_init = 1.0)
    U = allocate_tensor_velocity(ws)
    # Divergence-free curl-form input: u = (∂_y ψ, -∂_x ψ) with ψ = sin·sin.
    # ∂_y ψ = 2π sin(2πx) cos(2πy);  -∂_x ψ = -2π cos(2πx) sin(2πy).
    ux_fun(x) =  2π * sin(2π*x[1]) * cos(2π*x[2])
    uy_fun(x) = -2π * cos(2π*x[1]) * sin(2π*x[2])
    fill_tensor_velocity!(U, ws, (ux_fun, uy_fun))

    LU = allocate_tensor_velocity(ws)
    apply_tensor!(LU, U, coefs, ws)

    # Sampled center cell.  Expected (L u)_d / u_d = 1 + Δt μ · 8π²
    # because ∇·u ≈ 0 (analytic), ∇² u_d = -8π² u_d.
    sample_c = ws.patch_leaves[1][1][div(end, 2)]
    expected = 1.0 + Δt * μ * 8π^2
    ratio_1 = LU.u[1][1][1].phi[sample_c][1] / U.u[1][1][1].phi[sample_c][1]
    ratio_2 = LU.u[2][1][1].phi[sample_c][1] / U.u[2][1][1].phi[sample_c][1]
    # Discrete operator approximates the analytic factor with O(h²) error.
    @test isapprox(ratio_1, expected; rtol = 1e-2)
    @test isapprox(ratio_2, expected; rtol = 1e-2)
end

@testset "TensorOp: solve incompressible vector Poisson" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-9, maxiter = 200, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    Δt = 0.01; μ = 1.0
    coefs = allocate_tensor_coefs(ws; A = 1.0, B = Δt, mu = μ, lambda = 0.0,
                                     alpha_init = 1.0)
    U = allocate_tensor_velocity(ws)
    RHS = allocate_tensor_velocity(ws)
    fill_tensor_velocity!(RHS, ws, (
        x -> sin(2π*x[1]) * cos(2π*x[2]),
        x -> cos(2π*x[1]) * sin(2π*x[2])
    ))
    fill_tensor_velocity!(U, ws, (x -> 0.0, x -> 0.0))

    r = solve_tensor!(U, RHS, coefs, ws;
                        method = :cg, tol = 1e-9, maxiter = 200)
    @test r.converged
    @test r.res_final / r.res_init < 1e-8
end

@testset "TensorOp: solve compressible variant (λ > 0)" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 16
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-9, maxiter = 200, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    Δt = 0.01; μ = 1.0; λ = 0.5
    coefs = allocate_tensor_coefs(ws; A = 1.0, B = Δt, mu = μ, lambda = λ,
                                     alpha_init = 1.0)
    U = allocate_tensor_velocity(ws)
    RHS = allocate_tensor_velocity(ws)
    fill_tensor_velocity!(RHS, ws, (
        x -> sin(2π*x[1]) * cos(2π*x[2]),
        x -> cos(2π*x[1]) * sin(2π*x[2])
    ))
    fill_tensor_velocity!(U, ws, (x -> 0.0, x -> 0.0))

    r = solve_tensor!(U, RHS, coefs, ws;
                        method = :cg, tol = 1e-9, maxiter = 200)
    @test r.converged
    @test r.res_final / r.res_init < 1e-8
end
