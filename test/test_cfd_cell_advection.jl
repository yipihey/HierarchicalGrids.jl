# Phase 2 worked example: cell-based scalar advection (PR-14).
#
# Smoke-test wrapper that runs a subset of the example's tests under the
# parent package's `Pkg.test()`. Rather than activate the example's own
# project (which is fragile inside a Pkg.test() session), we include the
# example's source directly — its only dependency is HierarchicalGrids,
# which is already loaded.
#
# The example carries its own full test suite under its project; CI on
# the parent package only needs a fast smoke confirmation that the
# orchestration stack still composes correctly. Wall time: ≤ 5 seconds
# (excluding precompile, which is amortized into the parent test run).

using Test

const _example_root = abspath(joinpath(@__DIR__, "..", "examples", "cfd_cell_advection"))
isdir(_example_root) ||
    error("test_cfd_cell_advection: expected example at $_example_root")

# Load the example module by include — no project activation needed since
# its only dep (HierarchicalGrids) is already in scope.
module _CFDCellAdvectionSmoke
    using HierarchicalGrids
    include(joinpath(@__DIR__, "..", "examples", "cfd_cell_advection",
                     "src", "CFDCellAdvection.jl"))
    using .CFDCellAdvection
    export AdvectionConfig, AdvectionState, run!, build_state, gaussian_ic
end

using ._CFDCellAdvectionSmoke

# Run a handful of conservation/tracking checks. The example's own
# `test/runtests.jl` covers the full set; here we just confirm the
# orchestration plumbing is still wired correctly.

@testset "Construction + initial mass" begin
    config = AdvectionConfig(n_initial_refines = 3, n_steps = 0)
    state = _CFDCellAdvectionSmoke.CFDCellAdvection.build_state(config)
    @test n_cells(state.mesh) > 1
    @test 0.04 < state.initial_mass < 0.08
end

@testset "Mass conservation: short run, no AMR" begin
    config = AdvectionConfig(n_initial_refines = 3,
                             velocity = (0.5, 0.3),
                             dt = 0.02,
                             n_steps = 8,
                             amr_every = 0)
    _, info = run!(config)
    @test info.mass_drift < 1e-12
end

@testset "Mass conservation + tracking: one period" begin
    # n_steps * dt * v = 1 (one full domain crossing) on both axes.
    config = AdvectionConfig(n_initial_refines = 3,
                             velocity = (0.5, 0.5),
                             dt = 0.05,
                             n_steps = 40,
                             amr_every = 0)
    _, info = run!(config)
    @test info.mass_drift < 1e-10
    @test info.l1_error < 0.15
end

@testset "AMR: step_with_amr! + AdaptiveField stay consistent" begin
    config = AdvectionConfig(n_initial_refines = 2,
                             velocity = (0.5, 0.3),
                             dt = 0.01,
                             n_steps = 12,
                             amr_every = 4,
                             refine_threshold = 0.05,
                             coarsen_threshold = 0.005,
                             max_level = 4)
    _, info = run!(config)
    @test info.mass_drift < 1e-10
    @test info.n_leaves_final >= 4 * 4
end
