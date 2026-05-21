# 2nd-order AMR scalar advection mini-app: convergence + mass-conservation tests.

using Test

const _example_root = abspath(joinpath(@__DIR__, "..", "examples",
                                         "cfd_amr_advection_2o"))
isdir(_example_root) ||
    error("test_cfd_amr_advection_2o: expected example at $_example_root")

module _CFDAMRAdvSmoke
    using HierarchicalGrids
    include(joinpath(@__DIR__, "..", "examples", "cfd_amr_advection_2o",
                       "src", "CFDAMRAdvection2O.jl"))
    using .CFDAMRAdvection2O
end

const _A = _CFDAMRAdvSmoke.CFDAMRAdvection2O

# Slope (order) estimator: log₂(err_coarse / err_fine).
_order(e_coarse, e_fine) = log2(e_coarse / e_fine)

@testset "CFDAMRAdvection2O: build + IC sanity" begin
    cfg = _A.AdvectionConfig(n_base_refines = 4, t_final = 0.0,
                                dt = 1e-3, time_scheme = :rk2)
    state = _A.build_state(cfg)
    @test length(state.leaves) == 16 * 16
    # Mass at t=0 should equal the integrated IC — for the asymmetric
    # sinusoidal it integrates to zero over the unit square.
    @test abs(_A.total_mass(state)) < 1e-12
end

@testset "CFDAMRAdvection2O: mass conservation" begin
    cfg = _A.AdvectionConfig(n_base_refines = 4, t_final = 0.1,
                                dt = 1e-3, time_scheme = :rk2)
    state = _A.build_state(cfg)
    m0 = _A.total_mass(state)
    while state.t < cfg.t_final - 1e-12
        dt = min(cfg.dt, cfg.t_final - state.t)
        _A.step!(state, dt)
    end
    m1 = _A.total_mass(state)
    # Centered + RK2 is exactly conservative (sum of fluxes through
    # cancelling face contributions); machine-precision check.
    @test abs(m1 - m0) < 1e-11
end

@testset "CFDAMRAdvection2O: single-level spatial convergence ≥ 2" begin
    # Run on N ∈ {8, 16, 32, 64}, fixed dt small enough that the time
    # contribution to the L² error is dominated by space.
    Ns = [8, 16, 32, 64]
    errs = Float64[]
    for n in 3:6
        cfg = _A.AdvectionConfig(n_base_refines = n,
                                    t_final = 0.05, dt = 1e-3,
                                    time_scheme = :rk2,
                                    velocity = (1.0, 0.5))
        state = _A.run!(cfg)
        push!(errs, _A.l2_error(state))
    end
    # Slope between successive pairs should be ~ 2.
    s_8_16  = _order(errs[1], errs[2])
    s_16_32 = _order(errs[2], errs[3])
    s_32_64 = _order(errs[3], errs[4])
    @test s_8_16  > 1.7
    @test s_16_32 > 1.85
    @test s_32_64 > 1.9
end

@testset "CFDAMRAdvection2O: AMR (centered refined block) convergence ≥ 2" begin
    # Same setup but with the central [0.25, 0.75]² refined one extra
    # level.  The smooth profile is advected through the C/F interfaces.
    # The Martin-Colella fine-ghost reconstruction must hold 2nd-order
    # convergence end-to-end.
    Ns = [8, 16, 32, 64]
    errs = Float64[]
    for n in 3:6
        cfg = _A.AdvectionConfig(n_base_refines = n,
                                    refine_center = true,
                                    t_final = 0.05, dt = 1e-3,
                                    time_scheme = :rk2,
                                    velocity = (1.0, 0.5))
        state = _A.run!(cfg)
        push!(errs, _A.l2_error(state))
    end
    s_8_16  = _order(errs[1], errs[2])
    s_16_32 = _order(errs[2], errs[3])
    s_32_64 = _order(errs[3], errs[4])
    # Allow slight degradation from coarsest pair (under-resolved fine
    # block on N=8 vs N=16); demand ≥ 1.9 from the asymptotic pairs.
    @test s_8_16  > 1.5
    @test s_16_32 > 1.85
    @test s_32_64 > 1.9
end

@testset "CFDAMRAdvection2O: AMR mass conservation" begin
    cfg = _A.AdvectionConfig(n_base_refines = 4, refine_center = true,
                                t_final = 0.1, dt = 1e-3,
                                time_scheme = :rk2)
    state = _A.build_state(cfg)
    m0 = _A.total_mass(state)
    while state.t < cfg.t_final - 1e-12
        dt = min(cfg.dt, cfg.t_final - state.t)
        _A.step!(state, dt)
    end
    m1 = _A.total_mass(state)
    # Refluxing via per-fine-sub-face dispatch gives exact conservation.
    @test abs(m1 - m0) < 1e-11
end
