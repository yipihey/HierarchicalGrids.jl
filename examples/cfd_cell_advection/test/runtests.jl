using Test
using CFDCellAdvection
using HierarchicalGrids
using HierarchicalGrids: enumerate_leaves, n_cells, level_of

@testset "Cell-based scalar advection" begin

    # ------------------------------------------------------------------
    # Test 1: Construction and IC. The total mass should equal the
    # analytic integral of the Gaussian over the domain (within the
    # cell-mean truncation error of the IC projection).
    # ------------------------------------------------------------------
    @testset "build_state initializes mass + cell layout" begin
        config = AdvectionConfig(n_initial_refines = 3,
                                 n_steps = 0,
                                 amr_every = 0)
        state = CFDCellAdvection.build_state(config)
        @test n_cells(state.mesh) > 1
        @test length(enumerate_leaves(state.mesh)) == 4^3   # 8x8 = 64

        # Initial mass should be positive and reasonably close to the
        # analytic integral of an unnormalized 2D Gaussian (centered well
        # inside the box) = 2π σ² = 2π * 0.01 ≈ 0.0628.
        @test 0.04 < state.initial_mass < 0.08
    end

    # ------------------------------------------------------------------
    # Test 2: Mass conservation under a short run with NO refinement.
    # An exact-mass scheme on periodic BCs should drift only by floating-
    # point round-off accumulated across faces and steps. We allow a
    # very tight tolerance.
    # ------------------------------------------------------------------
    @testset "Mass conservation: short run, no AMR" begin
        config = AdvectionConfig(n_initial_refines = 3,
                                 velocity = (0.5, 0.3),
                                 dt = 0.02,
                                 n_steps = 10,
                                 amr_every = 0)
        state, info = run!(config)
        @test info.mass_drift < 1e-12
        # Diagnostics summary: the cell-mean min/max should remain in a
        # plausible range (upwind preserves [0, 1] for a Gaussian IC ≤ 1).
        @test state.diagnostics.liouville_min >= 0.0
        @test state.diagnostics.liouville_max <= 1.01
    end

    # ------------------------------------------------------------------
    # Test 3: Mass conservation + tracking after one full period.
    # First-order upwind has well-known numerical diffusion; we use
    # generous L¹ tolerance for the qualitative tracking check.
    # ------------------------------------------------------------------
    @testset "Mass conservation + tracking: one full period" begin
        # One full period along both axes: T = 1 / v = 2.0; pick dt so
        # n_steps * dt = 2.
        config = AdvectionConfig(n_initial_refines = 3,
                                 velocity = (0.5, 0.5),
                                 dt = 0.05,
                                 n_steps = 40,
                                 amr_every = 0)
        state, info = run!(config)
        @test info.mass_drift < 1e-10
        # First-order upwind on an 8x8 grid over a full period diffuses
        # heavily; the L¹ error per unit area is in the ~10% ballpark
        # for this resolution.
        @test info.l1_error < 0.15
    end

    # ------------------------------------------------------------------
    # Test 4: AMR-driven refinement should fire and the mass invariant
    # must hold across refinement events too.
    # ------------------------------------------------------------------
    @testset "AMR: step_with_amr! preserves mass through refinement" begin
        config = AdvectionConfig(n_initial_refines = 2,    # start coarse: 4x4
                                 velocity = (0.5, 0.3),
                                 dt = 0.01,
                                 n_steps = 12,
                                 amr_every = 4,
                                 refine_threshold = 0.05,
                                 coarsen_threshold = 0.005,
                                 max_level = 4)
        state, info = run!(config)
        # Mass drift should stay tight even with refinement firing.
        @test info.mass_drift < 1e-10
        # AMR should have refined past the initial uniform layout.
        @test info.n_leaves_final >= 4 * 4
    end

    # ------------------------------------------------------------------
    # Test 5: Internal piece — gradient_indicator is finite and nonneg.
    # ------------------------------------------------------------------
    @testset "gradient_indicator is finite + nonneg" begin
        config = AdvectionConfig(n_initial_refines = 2, n_steps = 0)
        state = CFDCellAdvection.build_state(config)
        ind = CFDCellAdvection.gradient_indicator(parent(state.af), state.mesh)
        @test all(isfinite, ind)
        @test all(>=(0.0), ind)
        # At least some cells should have nonzero indicator (Gaussian peak
        # has high gradient near its center).
        @test any(>(0.0), ind)
    end

end
