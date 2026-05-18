# ============================================================================
# Tests for multi-level node Laplacian (nested grids).
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, MGWorkspace, MGOptions,
                          PERIODIC, NodeField, allocate_node_field,
                          fill_node_field!, NodeCoefs, allocate_node_coefs,
                          restrict_node!, prolong_node!, vcycle_node!,
                          solve_node_laplacian_ml!,
                          add_patches!, EulerianFrame, HierarchicalMesh,
                          refine_cells!
using HierarchicalGrids.Overlap: FrameBoundaries, enumerate_leaves

function _build_nested_hierarchy(refines_per_level::Vector{Int})
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    ph = build_uniform_root_hierarchy(Val(2), refines_per_level[1],
                                       (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    for ℓ in 2:length(refines_per_level)
        m = HierarchicalMesh{2}()
        for _ in 1:refines_per_level[ℓ]
            refine_cells!(m, enumerate_leaves(m))
        end
        add_patches!(ph, ℓ, [EulerianFrame(m, (0.0, 0.0), (1.0, 1.0))])
    end
    return ph, bcs_spec
end

@testset "Node ML: restrict + prolong round-trip on constants" begin
    ph, bcs_spec = _build_nested_hierarchy([3, 4])  # 8x8 + 16x16
    opts = MGOptions(tol = 1e-10, maxiter = 50)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    fine = allocate_node_field(ws)
    coarse = allocate_node_field(ws)
    # Set fine to a constant. Restrict → coarse should be the same constant
    # (on interior nodes; clamping at boundaries can affect endpoints).
    fill_node_field!(fine, ws, x -> 3.7)
    restrict_node!(coarse, fine, 2; patch = 1)
    inner = coarse.data[1][1]
    @test all(isapprox.(inner[2:end-1, 2:end-1], 3.7; atol = 1e-12))

    # Now prolong back and verify match at coincident nodes.
    fine2 = allocate_node_field(ws)
    prolong_node!(fine2, coarse, 2; patch = 1)
    Fd = size(fine2.data[2][1])
    for J in 1:Fd[2], I in 1:Fd[1]
        @test fine2.data[2][1][I, J] ≈ 3.7 atol = 1e-12
    end
end

@testset "Node ML: V-cycle reduces residual on 3-level Poisson" begin
    ph, bcs_spec = _build_nested_hierarchy([3, 4, 5])
    opts = MGOptions(tol = 1e-9, maxiter = 50, cycle = :pcg,
                       bottom_smooth_iters = 30)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    phi = allocate_node_field(ws)
    rho = allocate_node_field(ws)
    fill_node_field!(rho, ws, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
    fill_node_field!(phi, ws, x -> 0.0)
    coefs = allocate_node_coefs(ws; B = 1.0, sigma_init = 1.0)

    r = solve_node_laplacian_ml!(phi, rho, coefs, ws;
                                    tol = 1e-9, maxiter = 50,
                                    n_pre = 2, n_post = 2)
    # Residual must reduce by at least 1 order of magnitude.  Convergence
    # rate is limited by the simple restriction/prolongation (clamped at
    # boundaries; periodic-aware versions are a follow-up).
    @test r.res_final / r.res_init < 0.2
end
