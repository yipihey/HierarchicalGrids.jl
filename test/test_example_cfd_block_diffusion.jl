# Smoke test for the `examples/cfd_block_diffusion/` worked example
# (PR-15). Loads the example as a stand-alone module via `include` of its
# entry-point file — the example's own `Project.toml` only declares
# HierarchicalGrids as a dep, which we already have in scope here.
#
# Targets a small problem (4 blocks, 8×8 = ~16×16 effective grid, 25 RK2
# steps) so the smoke test is well under 60 s wall time. The mass-drift
# / peak-decay assertions match the PR-15 acceptance criteria.

using Test

# Load the example module without colliding with any prior `using` —
# the file uses `module CFDBlockDiffusion ... end`, so `include` defines
# it in the test scope.
const _CFD_DIFF_DIR = joinpath(@__DIR__, "..", "examples", "cfd_block_diffusion")
include(joinpath(_CFD_DIFF_DIR, "src", "CFDBlockDiffusion.jl"))
using .CFDBlockDiffusion: DiffusionConfig, run!

@testset "Smoke test: 1 level, 25 steps" begin
    # 4 blocks × 8×8 = ~16×16 effective grid. dx = 0.5/7 ≈ 0.0714,
    # dt_max = dx²/(4D) ≈ 0.127 — dt = 5e-3 is comfortably stable.
    config = DiffusionConfig(n_levels    = 1,
                              diffusivity = 0.01,
                              dt          = 5.0e-3,
                              n_steps     = 25)
    final_field, diags = run!(config; sigma = 0.1)

    # Mass conservation: trapezoidal rule with periodic BCs is mass-exact
    # for the heat equation; drift should be at the round-off level.
    @test diags.mass_drift < 1e-10
    @test diags.mass_drift > -1e-10

    # Peak decay should match σ²/(σ² + 2 D t) within 10 % slack.
    @test diags.peak_decay_error < 0.10

    # Wall time guard — keep the smoke test honest.
    @test diags.wall_time_seconds < 30.0
end
