# Cell-mode incompressible-NS scaffold, step 1: passive 2nd-order vector
# advection across coarse-fine interfaces.

using Test

const _example_root = abspath(joinpath(@__DIR__, "..", "examples",
                                         "cfd_incompressible_ns_cell"))
isdir(_example_root) ||
    error("test_cfd_incompressible_ns_cell: expected example at $_example_root")

module _CFDINSCellSmoke
    using HierarchicalGrids
    include(joinpath(@__DIR__, "..", "examples", "cfd_incompressible_ns_cell",
                       "src", "CFDIncompressibleNSCell.jl"))
    using .CFDIncompressibleNSCell
end

const _C = _CFDINSCellSmoke.CFDIncompressibleNSCell

_order(e_coarse, e_fine) = log2(e_coarse / e_fine)

@testset "INSCell: build + Taylor-Green IC sanity" begin
    cfg = _C.NSCellConfig(n_base_refines = 4, t_final = 0.0, dt = 1e-3,
                              time_scheme = :rk2)
    state = _C.build_state(cfg)
    @test length(state.leaves) == 16 * 16
    # Taylor-Green has zero mean momentum on the unit square.
    mu, mv = _C.total_momentum(state)
    @test abs(mu) < 1e-12
    @test abs(mv) < 1e-12
    # <(1/2)(u² + v²)> for U0=1, k=2π: <sin² · cos²> + <cos² · sin²> = 1/4 each,
    # → KE = 1/2 · (1/4 + 1/4) = 1/4.
    @test isapprox(_C.kinetic_energy(state), 0.25; atol = 1e-3)
end

@testset "INSCell: momentum conservation under pure advection (uniform)" begin
    cfg = _C.NSCellConfig(n_base_refines = 4, t_final = 0.1, dt = 1e-3,
                              time_scheme = :rk2,
                              carrier_velocity = (1.0, 0.5))
    state = _C.build_state(cfg)
    mu0, mv0 = _C.total_momentum(state)
    while state.t < cfg.t_final - 1e-12
        dt = min(cfg.dt, cfg.t_final - state.t)
        _C.step!(state, dt)
    end
    mu1, mv1 = _C.total_momentum(state)
    @test abs(mu1 - mu0) < 1e-11
    @test abs(mv1 - mv0) < 1e-11
end

@testset "INSCell: momentum conservation under pure advection (AMR)" begin
    cfg = _C.NSCellConfig(n_base_refines = 4, refine_center = true,
                              t_final = 0.1, dt = 1e-3,
                              time_scheme = :rk2,
                              carrier_velocity = (1.0, 0.5))
    state = _C.build_state(cfg)
    mu0, mv0 = _C.total_momentum(state)
    while state.t < cfg.t_final - 1e-12
        dt = min(cfg.dt, cfg.t_final - state.t)
        _C.step!(state, dt)
    end
    mu1, mv1 = _C.total_momentum(state)
    @test abs(mu1 - mu0) < 1e-11
    @test abs(mv1 - mv0) < 1e-11
end

@testset "INSCell: single-level spatial convergence ≥ 2" begin
    errs = Float64[]
    for n in 3:6
        cfg = _C.NSCellConfig(n_base_refines = n, t_final = 0.05, dt = 1e-3,
                                  time_scheme = :rk2,
                                  carrier_velocity = (1.0, 0.5))
        state = _C.run!(cfg)
        push!(errs, _C.l2_error(state))
    end
    @test _order(errs[1], errs[2]) > 1.7
    @test _order(errs[2], errs[3]) > 1.85
    @test _order(errs[3], errs[4]) > 1.9
end

@testset "INSCell: AMR (centered refined block) convergence ≥ 2" begin
    errs = Float64[]
    for n in 3:6
        cfg = _C.NSCellConfig(n_base_refines = n, refine_center = true,
                                  t_final = 0.05, dt = 1e-3,
                                  time_scheme = :rk2,
                                  carrier_velocity = (1.0, 0.5))
        state = _C.run!(cfg)
        push!(errs, _C.l2_error(state))
    end
    @test _order(errs[1], errs[2]) > 1.5
    @test _order(errs[2], errs[3]) > 1.85
    @test _order(errs[3], errs[4]) > 1.9
end
