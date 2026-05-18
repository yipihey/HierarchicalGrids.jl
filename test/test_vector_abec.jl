# ============================================================================
# Tests for the multi-component decoupled ABec solver (Tier 2 #5 subset).
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, MGWorkspace, MGOptions,
                          fill_field!, PERIODIC,
                          fill_abec_alpha!, fill_abec_beta!,
                          VectorABecProblem, allocate_vector_abec,
                          solve_vector_abec!
using HierarchicalGrids.Overlap: FrameBoundaries

@testset "VectorABec: K=3 decoupled solves all converge" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    prob = allocate_vector_abec(ws, 3; A = 1.0, B = 0.01)
    # Component 0: diffusivity 1.0
    # Component 1: diffusivity 0.5
    # Component 2: diffusivity 2.0
    for (k, Dk) in enumerate((1.0, 0.5, 2.0))
        fill_abec_alpha!(prob.coefs[k], ws, x -> 1.0)
        fill_abec_beta!(prob.coefs[k], ws, x -> Dk)
        fill_field!(prob.fields[k], ws.ph, :rho,
                     x -> sin(2π*x[1]) * cos(2π*x[2]))
        fill_field!(prob.fields[k], ws.ph, :phi, x -> 0.0)
    end

    results = solve_vector_abec!(prob, ws; tol = 1e-9, maxiter = 200)
    @test length(results) == 3
    for r in results
        @test r.converged
        @test r.res_final / r.res_init < 1e-8
    end
end
