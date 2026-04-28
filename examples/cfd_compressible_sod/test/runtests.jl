using Test
using CFDCompressibleSod
using HierarchicalGrids
using HierarchicalGrids: enumerate_leaves, n_cells

@testset "Sod tube: compressible-flow worked example" begin

    # ------------------------------------------------------------------
    # Test 1: Construction. Build the mesh + IC and check the initial
    # density field has the Sod left/right values.
    # ------------------------------------------------------------------
    @testset "build_state initializes IC + mesh" begin
        config = SodConfig(n_initial_refines = 4, t_final = 0.0,
                            amr_every = 0)
        state = CFDCompressibleSod.build_state(config)
        @test n_cells(state.mesh) > 1
        @test length(enumerate_leaves(state.mesh)) == 4^4   # 16x16

        # Initial mass should equal ρL * 0.5 * 0.1 + ρR * 0.5 * 0.1
        # = 0.05 + 0.00625 = 0.05625 (within IC discretization error).
        expected = 1.0 * 0.5 * 0.1 + 0.125 * 0.5 * 0.1
        @test isapprox(state.initial_mass, expected; rtol = 1e-2)
        @test state.initial_energy > 0.0
    end

    # ------------------------------------------------------------------
    # Test 2: Mass + energy conservation under a short run with NO AMR.
    # `t_final = 0.02` keeps the analytic Sod waves well away from the
    # OUTFLOW boundaries; numerical diffusion at the boundary cells
    # only becomes nontrivial after t ≳ 0.04, so we use the shorter
    # window to keep mass / energy drift at machine epsilon.
    # ------------------------------------------------------------------
    @testset "Mass + energy conservation: short run, no AMR" begin
        config = SodConfig(n_initial_refines = 5, t_final = 0.02,
                            amr_every = 0)
        _, info = run!(config)
        @test abs(info.mass_drift)   < 1e-10
        @test abs(info.energy_drift) < 1e-10
        # Momentum drift should still be small at early t (waves far from
        # boundary). Generous bound.
        @test abs(info.momentum_drift) < 0.1
    end

    # ------------------------------------------------------------------
    # Test 3: Shock position at t=0.2. The standard Sod tube places the
    # right-going shock at x ≈ 0.85 at this time.
    # ------------------------------------------------------------------
    @testset "Shock position at t=0.2" begin
        config = SodConfig(n_initial_refines = 5, t_final = 0.2,
                            amr_every = 0)
        _, info = run!(config)
        @test 0.80 < info.shock_position < 0.92
    end

    # ------------------------------------------------------------------
    # Test 4: AMR engages around the shock. Start from a moderately
    # refined base; allow refinement up to two levels deeper, so the
    # indicator has somewhere to push without immediately running
    # into `max_level`.
    # ------------------------------------------------------------------
    @testset "AMR engages around shock" begin
        config = SodConfig(n_initial_refines = 4,
                            t_final = 0.1,
                            amr_every = 5,
                            max_level = 6,
                            refine_threshold = 0.1,
                            coarsen_threshold = 0.01)
        _, info = run!(config)
        @test info.n_leaves_final > info.n_leaves_initial
    end

    # ------------------------------------------------------------------
    # Test 5: L¹ error against the analytic Riemann solution.
    # First-order HLL on degree-0 cell means is heavily diffusive at the
    # discontinuities — L¹ ~ O(h^{1/2}) at the shock. We use a generous
    # bound on a moderately resolved run.
    # ------------------------------------------------------------------
    @testset "L¹ error vs analytic Riemann" begin
        config = SodConfig(n_initial_refines = 5, t_final = 0.2,
                            amr_every = 0)
        _, info = run!(config)
        @test info.l1_error < 0.10
    end

    # ------------------------------------------------------------------
    # Test 6: Internal piece — the analytic Riemann sampler at t=0
    # reproduces the IC.
    # ------------------------------------------------------------------
    @testset "exact_riemann_at(t=0) matches IC" begin
        ρL, uL, vL, pL = CFDCompressibleSod.exact_riemann_at(0.25, 0.0)
        ρR, uR, vR, pR = CFDCompressibleSod.exact_riemann_at(0.75, 0.0)
        @test ρL == 1.0 && pL == 1.0 && uL == 0.0
        @test ρR == 0.125 && pR == 0.1 && uR == 0.0
    end

    # ------------------------------------------------------------------
    # Test 7: Internal piece — primitive ↔ conservative round-trip.
    # ------------------------------------------------------------------
    @testset "cons ↔ prim round-trip" begin
        U = CFDCompressibleSod.prim_to_cons(1.5, 0.3, -0.2, 0.7)
        ρ, u, v, p = CFDCompressibleSod.cons_to_prim(U)
        @test isapprox(ρ, 1.5)
        @test isapprox(u, 0.3)
        @test isapprox(v, -0.2)
        @test isapprox(p, 0.7)
    end
end
