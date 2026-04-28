# Phase 2 worked example: Sod-tube compressible flow (PR-3).
#
# Smoke-test wrapper that runs a subset of the example's tests under the
# parent package's `Pkg.test()`. Rather than activate the example's own
# project (fragile inside a Pkg.test() session), we include the example's
# source directly — its only HG-internal dependency is HierarchicalGrids,
# which is already loaded. (LinearAlgebra is in stdlib.)
#
# The smoke uses a smaller mesh and a shorter t_final than the example's
# own runtests.jl to keep CI under 30 s wall time.

using Test

const _example_root = abspath(joinpath(@__DIR__, "..", "examples", "cfd_compressible_sod"))
isdir(_example_root) ||
    error("test_cfd_compressible_sod: expected example at $_example_root")

# Load the example module by include — no project activation.
module _CFDCompressibleSodSmoke
    using HierarchicalGrids
    include(joinpath(@__DIR__, "..", "examples", "cfd_compressible_sod",
                     "src", "CFDCompressibleSod.jl"))
    using .CFDCompressibleSod
    export SodConfig, SodState, run!, build_state
end

using ._CFDCompressibleSodSmoke

# Run a handful of conservation/sanity checks at low resolution.
# The example's full test/runtests.jl covers the larger-mesh L¹ check.

@testset "Construction + initial mass" begin
    config = SodConfig(n_initial_refines = 3, t_final = 0.0,
                        amr_every = 0)
    state = _CFDCompressibleSodSmoke.CFDCompressibleSod.build_state(config)
    @test n_cells(state.mesh) > 1
    expected = 1.0 * 0.5 * 0.1 + 0.125 * 0.5 * 0.1
    @test isapprox(state.initial_mass, expected; rtol = 5e-2)
end

@testset "Mass + energy conservation: short run, no AMR" begin
    # `t_final = 0.02` keeps the rarefaction & shock fronts well away
    # from the OUTFLOW boundaries (numerical diffusion preconditions
    # the boundary cells with u != 0 by t ≈ 0.04 even though the
    # analytic waves don't reach there until t ≈ 0.42 / 0.29). The
    # short window keeps mass / energy drift at machine epsilon.
    config = SodConfig(n_initial_refines = 4, t_final = 0.02,
                        amr_every = 0)
    _, info = _CFDCompressibleSodSmoke.CFDCompressibleSod.run!(config)
    @test abs(info.mass_drift)   < 1e-10
    @test abs(info.energy_drift) < 1e-10
end

@testset "Shock develops" begin
    # Use a shorter time than the analytic t=0.2 reference to keep
    # smoke wall time well under the 30 s budget. By t=0.1 the
    # right-going shock has reached x ≈ 0.67 in the analytic
    # solution; first-order HLL on this mesh smears it but the
    # detector should still pick a position past the IC interface.
    config = SodConfig(n_initial_refines = 3, t_final = 0.1,
                        amr_every = 0)
    _, info = _CFDCompressibleSodSmoke.CFDCompressibleSod.run!(config)
    @test info.shock_position > 0.55
end

@testset "AMR engages around shock" begin
    # Start at level 3 (8x8 = 64 leaves); allow refinement up to
    # level 4 around the discontinuity. This gives the indicator
    # somewhere to refine without immediately running into max_level.
    config = SodConfig(n_initial_refines = 3,
                        t_final = 0.05,
                        amr_every = 5,
                        max_level = 4,
                        refine_threshold = 0.1,
                        coarsen_threshold = 0.01)
    _, info = _CFDCompressibleSodSmoke.CFDCompressibleSod.run!(config)
    @test info.n_leaves_final > info.n_leaves_initial
end

@testset "exact_riemann_at(t=0) matches IC" begin
    ρL, _, _, pL =
        _CFDCompressibleSodSmoke.CFDCompressibleSod.exact_riemann_at(0.25, 0.0)
    ρR, _, _, pR =
        _CFDCompressibleSodSmoke.CFDCompressibleSod.exact_riemann_at(0.75, 0.0)
    @test ρL == 1.0 && pL == 1.0
    @test ρR == 0.125 && pR == 0.1
end
