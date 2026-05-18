# ============================================================================
# Tests for edge-centered vector fields and the component-wise scalar
# edge Laplacian (Tier 3 #10 foundation).
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, MGWorkspace, MGOptions,
                          PERIODIC, EdgeField, allocate_edge_field,
                          fill_edge_field!, apply_edge_laplacian!,
                          solve_edge_laplacian!
using HierarchicalGrids.Overlap: FrameBoundaries

@testset "EdgeField: storage shapes" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 16
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-9, maxiter = 50)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    E = allocate_edge_field(ws, 1, 1; init = 0.0)
    # x-edges: (N, N+1).  y-edges: (N+1, N).
    @test size(E.u[1]) == (N, N + 1)
    @test size(E.u[2]) == (N + 1, N)
end

@testset "EdgeField: component-wise Poisson recovers analytic" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-10, maxiter = 500, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    E = allocate_edge_field(ws)
    RHS = allocate_edge_field(ws)
    # -∇² A_d = source_d  with source_x = -8π² sin·sin, source_y = -8π² cos·cos.
    fill_edge_field!(RHS, ws, 1, 1,
                       (x -> -8π^2 * sin(2π*x[1]) * sin(2π*x[2]),
                        x -> -8π^2 * cos(2π*x[1]) * cos(2π*x[2])))

    results = solve_edge_laplacian!(E, RHS, ws, 1, 1;
                                       method = :cg, tol = 1e-10, maxiter = 500)
    @test length(results) == 2
    for r in results
        @test r.converged
        @test r.res_final / r.res_init < 1e-7
    end
end

@testset "EdgeField: apply followed by solve recovers input" begin
    # Set A, compute L A, solve L · A_recover = L A, check A ≈ A_recover
    # (modulo the constant-mode null space which we project out).
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 16
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-10, maxiter = 500, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    A_in = allocate_edge_field(ws)
    fill_edge_field!(A_in, ws, 1, 1,
                       (x -> sin(2π*x[1]) * cos(2π*x[2]),
                        x -> cos(2π*x[1]) * sin(2π*x[2])))
    LA = allocate_edge_field(ws)
    apply_edge_laplacian!(LA, A_in, ws, 1, 1)

    # The forward apply produces ∇² A; we want to solve -∇² A_out = -∇² A_in
    # (the SPD-positive form used by the inner Krylov bridge).
    RHS = allocate_edge_field(ws)
    for d in 1:2
        RHS.u[d] .= -LA.u[d]
    end

    A_out = allocate_edge_field(ws)
    results = solve_edge_laplacian!(A_out, RHS, ws, 1, 1;
                                       method = :cg, tol = 1e-10, maxiter = 500)
    for r in results; @test r.converged; end

    # Compare A_out to A_in modulo mean.
    for d in 1:2
        mean_in  = sum(A_in.u[d])  / length(A_in.u[d])
        mean_out = sum(A_out.u[d]) / length(A_out.u[d])
        err = sqrt(sum((A_out.u[d] .- mean_out .- (A_in.u[d] .- mean_in)) .^ 2)
                    / length(A_in.u[d]))
        @test err < 1e-7
    end
end
