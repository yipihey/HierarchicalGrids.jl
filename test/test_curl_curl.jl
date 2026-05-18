# ============================================================================
# Tests for the 2D edge-centered curl-curl operator (Tier 3 #10).
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, MGWorkspace, MGOptions,
                          PERIODIC, EdgeField, allocate_edge_field,
                          fill_edge_field!, CurlCurlCoefs,
                          allocate_curlcurl_coefs, apply_curl_curl!,
                          solve_curl_curl!
using HierarchicalGrids.Overlap: FrameBoundaries

@testset "CurlCurl: apply on curl-free field reduces to σ·A" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-10, maxiter = 100, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    σ = 1.0
    coefs = allocate_curlcurl_coefs(ws; mu_inv = 1.0, sigma_init = σ)

    # Curl-free field: A = (sin(2πx) cos(2πy), cos(2πx) sin(2πy)).
    # ∇×A = ∂_x A_y - ∂_y A_x = -2π sin·sin - (-2π sin·sin) = 0.
    A = allocate_edge_field(ws)
    fill_edge_field!(A, ws, 1, 1,
                       (x -> sin(2π*x[1]) * cos(2π*x[2]),
                        x -> cos(2π*x[1]) * sin(2π*x[2])))
    LA = allocate_edge_field(ws)
    apply_curl_curl!(LA, A, coefs, ws)
    # For curl-free A, ∇×(∇×A) = 0 so LA = σ·A.
    err1 = sqrt(sum((LA.u[1] .- σ .* A.u[1]) .^ 2) / length(A.u[1]))
    err2 = sqrt(sum((LA.u[2] .- σ .* A.u[2]) .^ 2) / length(A.u[2]))
    @test err1 < 1e-8
    @test err2 < 1e-8
end

@testset "CurlCurl: SPD solve converges via CG" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-10, maxiter = 500, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    coefs = allocate_curlcurl_coefs(ws; mu_inv = 1.0, sigma_init = 1.0)
    A = allocate_edge_field(ws)
    RHS = allocate_edge_field(ws)
    fill_edge_field!(RHS, ws, 1, 1,
                       (x -> sin(2π*x[1]) * cos(2π*x[2]),
                        x -> cos(2π*x[1]) * sin(2π*x[2])))
    r = solve_curl_curl!(A, RHS, coefs, ws;
                            method = :cg, tol = 1e-10, maxiter = 500)
    @test r.converged
    @test r.res_final / r.res_init < 1e-9
end
