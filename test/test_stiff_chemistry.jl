# ============================================================================
# Tests for the per-cell stiff ODE integrator (Tier 3 #8).
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, MGWorkspace, MGOptions,
                          PERIODIC, ReactionSystem, allocate_species,
                          fill_species!, step_reaction!
using HierarchicalGrids.Overlap: FrameBoundaries

# ---- Test setup -------------------------------------------------------------

function _make_ws(D::Int = 2)
    bcs_spec = D == 2 ?
        ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)) :
        ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    lo = ntuple(_ -> 0.0, D); hi = ntuple(_ -> 1.0, D)
    ph = build_uniform_root_hierarchy(Val(D), 3, lo, hi; physical_bcs = bcs)
    opts = MGOptions(tol = 1e-9, maxiter = 10)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    return ws
end

# ---- Linear scalar decay y' = -k y --------------------------------------

@testset "Stiff chemistry: linear decay analytical agreement" begin
    decay!(dy, y, p, t) = (dy[1] = -10.0 * y[1])
    sys = ReactionSystem(1, decay!; T = Float64)
    ws = _make_ws(2)
    spec = allocate_species(ws, 1)
    fill_species!(spec, ws, x -> (1.0,))

    dt = 0.05
    n_failed = step_reaction!(spec, sys, ws, dt; newton_maxiter = 20)
    @test n_failed == 0

    # Backward Euler: y^{n+1} = y^n / (1 + dt·k).
    expected = 1.0 / (1.0 + dt * 10.0)
    @inbounds for c in ws.patch_leaves[1][1]
        @test isapprox(spec.data[1][1][c, 1], expected; rtol = 1e-10)
    end
end

# ---- Stiff linear 2x2 -------------------------------------------------

@testset "Stiff chemistry: 2x2 stiff linear with analytical Jacobian" begin
    # y1' = -1000·y1 + y2
    # y2' = -y2
    # Stiff (timescales 1e-3 vs 1.0).
    function rhs2!(dy, y, p, t)
        dy[1] = -1000.0 * y[1] + y[2]
        dy[2] = -y[2]
    end
    function jac2!(J, y, p, t)
        J[1, 1] = -1000.0; J[1, 2] = 1.0
        J[2, 1] = 0.0;     J[2, 2] = -1.0
    end
    sys = ReactionSystem(2, rhs2!; jac! = jac2!, T = Float64)
    ws = _make_ws(2)
    spec = allocate_species(ws, 2)
    fill_species!(spec, ws, x -> (1.0, 1.0))

    n_failed = step_reaction!(spec, sys, ws, 0.1; newton_maxiter = 20)
    @test n_failed == 0
    # y2 evolves as 1/(1+0.1) = 0.909... in one BE step.
    # y1 evolves toward y2/1000 ≈ small.
    sample_c = ws.patch_leaves[1][1][1]
    @test spec.data[1][1][sample_c, 2] ≈ 1.0 / 1.1  atol = 1e-8
    @test abs(spec.data[1][1][sample_c, 1]) < 0.02  # decayed (BE not exact)
end

# ---- Robertson stiff (classical) -----------------------------------------

@testset "Stiff chemistry: Robertson kinetics" begin
    function robertson!(dy, y, p, t)
        k1, k2, k3 = 0.04, 3e7, 1e4
        dy[1] = -k1 * y[1] + k3 * y[2] * y[3]
        dy[2] =  k1 * y[1] - k3 * y[2] * y[3] - k2 * y[2]^2
        dy[3] =  k2 * y[2]^2
    end
    sys = ReactionSystem(3, robertson!; T = Float64)
    ws = _make_ws(2)
    spec = allocate_species(ws, 3)
    fill_species!(spec, ws, x -> (1.0, 0.0, 0.0))

    # Walk a sequence of stiff time steps.
    total_failed = 0
    for dt in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
        total_failed += step_reaction!(spec, sys, ws, dt;
                                          newton_maxiter = 80,
                                          newton_tol = 1e-8)
    end
    @test total_failed == 0

    sample_c = ws.patch_leaves[1][1][1]
    y = spec.data[1][1]
    # Mass conservation: y1+y2+y3 = 1 throughout.
    @test sum(y[sample_c, s] for s in 1:3) ≈ 1.0 atol = 1e-6
    # At t≈11, y1 should have decayed a few % from 1.
    @test 0.7 < y[sample_c, 1] < 0.95
    # y3 should be growing.
    @test y[sample_c, 3] > 0.05
end
