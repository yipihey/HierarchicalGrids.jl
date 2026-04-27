using Test
using CFDBlockDiffusion
using HierarchicalGrids

@testset "CFDBlockDiffusion (block-based scalar diffusion)" begin

    @testset "Initial Gaussian: peak ≈ 1, mass ≈ 2π σ²" begin
        # Just initialize, then check sanity. n_levels=2 ⇒ 4×4 = 16 blocks,
        # each 8×8 ⇒ 32×32 grid effectively (with shared boundary nodes).
        sigma = 0.1
        config = DiffusionConfig(domain_lo = (0.0, 0.0),
                                  domain_hi = (1.0, 1.0),
                                  n_levels = 2,
                                  diffusivity = 0.01,
                                  dt = 1.0e-3,
                                  n_steps = 0)         # zero-step run
        final_field, diags = run!(config; sigma = sigma)

        # Initial peak is at the centre, value 1.
        @test isapprox(diags.peak_initial, 1.0; atol = 1e-12)
        # Zero steps ⇒ final equals initial.
        @test diags.peak_final == diags.peak_initial
        @test diags.mass_drift == 0.0
        # Trapezoidal integral of the Gaussian: with σ = 0.1 ≪ 1, the box
        # is ~5σ wide on each side from the centre, so the trapezoidal
        # estimate of ∫ G dV ≈ 2π σ² to ~0.1% (boundary tail is tiny).
        analytic = 2π * sigma^2
        @test isapprox(diags.mass_initial, analytic; rtol = 5e-3)
    end

    @testset "Mass conservation (drift ≤ 1e-10)" begin
        # 16 blocks × 8×8 = 32×32 effective grid. 50 RK2 steps.
        # dx = 1/(4*7) ≈ 0.0357; dt < dx²/(4D) ≈ 0.0319; dt = 5e-3 is safe.
        config = DiffusionConfig(n_levels = 2,
                                  diffusivity = 0.01,
                                  dt = 5.0e-3,
                                  n_steps = 50)
        final_field, diags = run!(config; sigma = 0.1)

        # The trapezoidal rule on the equispaced N-node grid with periodic
        # BCs is mass-exact for a discrete Laplacian whose stencil at the
        # block boundary couples (N-1, j) ↔ (N, j) ↔ neighbour-(2, j).
        # Drift should be at the round-off level for ~10^4 stencil ops × 50 steps.
        @test diags.mass_drift < 1e-10
        @test diags.mass_drift > -1e-10
    end

    @testset "Peak decay tracks σ²/(σ² + 2Dt) within 10%" begin
        # Same numerical setup; check that the centre value matches the
        # analytic 2-D Gaussian decay law on the unbounded plane.
        config = DiffusionConfig(n_levels = 2,
                                  diffusivity = 0.01,
                                  dt = 5.0e-3,
                                  n_steps = 50)
        final_field, diags = run!(config; sigma = 0.1)

        # Predicted peak after t = 50 * 5e-3 = 0.25:
        #   σ² / (σ² + 2 D t) = 0.01 / (0.01 + 2 * 0.01 * 0.25) = 0.01/0.015 ≈ 0.667
        # Slack 10 % per the PR-15 acceptance criterion.
        @test diags.peak_decay_error < 0.10
    end

    @testset "DiagnosticsResult has expected fields" begin
        config = DiffusionConfig(n_levels = 1,
                                  diffusivity = 0.01,
                                  dt = 1.0e-3,
                                  n_steps = 5)
        _, diags = run!(config; sigma = 0.1)
        @test diags.n_blocks == 4
        @test diags.dx > 0
        @test diags.wall_time_seconds >= 0
        @test diags.peak_predicted > 0
    end

end
