# Incompressible NS mini-app smoke + Taylor-Green verification tests.
#
# Mirrors the convention from cfd_compressible_sod / cfd_implicit_ns: include
# the example's source directly under the parent project's Pkg.test() rather
# than activating the example's own Project.toml.

using Test

const _example_root = abspath(joinpath(@__DIR__, "..", "examples",
                                         "cfd_incompressible_ns"))
isdir(_example_root) ||
    error("test_cfd_incompressible_ns: expected example at $_example_root")

module _CFDIncompressibleNSSmoke
    using HierarchicalGrids
    include(joinpath(@__DIR__, "..", "examples", "cfd_incompressible_ns",
                       "src", "CFDIncompressibleNS.jl"))
    using .CFDIncompressibleNS
end

const _INS = _CFDIncompressibleNSSmoke.CFDIncompressibleNS

@testset "Incompressible NS: build_state + Taylor-Green IC" begin
    cfg = _INS.IncompressibleNSConfig(n_initial_refines = 4,
                                         t_final = 0.0, dt = 1e-3,
                                         μ = 1e-2)
    state = _INS.build_state(cfg)
    @test state.Nx == 16
    @test state.Ny == 16
    @test length(state.leaves) == 16 * 16
    # Taylor-Green is analytically divergence-free; centered differences
    # on a periodic uniform grid recover that to machine precision.
    @test _INS.l2_divergence(state) < 1e-12
    # KE at t=0 with U0=1 and k=2π is 1/4 of <u² + v²> = 1/4.
    # <(1/2)(u² + v²)> = (1/2)·(1/4 + 1/4) = 1/4 for U0 = 1, k = 2π on [0,1]².
    @test isapprox(_INS.kinetic_energy(state), 0.25; atol = 1e-3)
end

@testset "Incompressible NS (:cn): one step preserves divergence ~ 0" begin
    cfg = _INS.IncompressibleNSConfig(n_initial_refines = 4,
                                         t_final = 5e-3, dt = 5e-3,
                                         μ = 1e-2,
                                         time_scheme = :cn,
                                         helmholtz_tol = 1e-12,
                                         poisson_tol = 1e-12)
    state = _INS.build_state(cfg)
    _INS.step!(state, cfg.dt)
    @test _INS.l2_divergence(state) < 1e-7
end

@testset "Incompressible NS (:cn): Taylor-Green energy decay" begin
    # KE(t) decays as KE(0) · exp(-4 k² ν t).
    cfg = _INS.IncompressibleNSConfig(n_initial_refines = 5,
                                         t_final = 0.02, dt = 5e-3,
                                         μ = 5e-2,
                                         time_scheme = :cn,
                                         helmholtz_tol = 1e-12,
                                         poisson_tol = 1e-12)
    state = _INS.build_state(cfg)
    ke0 = _INS.kinetic_energy(state)
    while state.t < cfg.t_final - 1e-12
        dt = min(cfg.dt, cfg.t_final - state.t)
        _INS.step!(state, dt)
    end
    ke_final = _INS.kinetic_energy(state)
    # Analytic decay: exp(-4 (2π)² ν t_final).
    decay_factor = exp(-4 * (2π)^2 * cfg.μ * cfg.t_final)
    ke_expected = ke0 * decay_factor
    @test isapprox(ke_final, ke_expected; rtol = 0.10)
    # Sanity: divergence small after the run.
    @test _INS.l2_divergence(state) < 1e-6
end

@testset "Incompressible NS (:sdirk2): one step converges, divergence ~ 0" begin
    cfg = _INS.IncompressibleNSConfig(n_initial_refines = 4,
                                         t_final = 5e-3, dt = 5e-3,
                                         μ = 1e-2,
                                         time_scheme = :sdirk2,
                                         newton_tol = 1e-6,
                                         newton_maxiter = 8,
                                         gmres_tol = 1e-4,
                                         gmres_maxiter = 50,
                                         helmholtz_tol = 1e-10,
                                         poisson_tol = 1e-10)
    state = _INS.build_state(cfg)
    info = _INS.step!(state, cfg.dt)
    iters1, res1, iters2, res2 = info
    @test iters1 <= cfg.newton_maxiter
    @test iters2 <= cfg.newton_maxiter
    @test _INS.l2_divergence(state) < 1e-5
end

@testset "Incompressible NS (:sdirk2): Taylor-Green energy decay" begin
    cfg = _INS.IncompressibleNSConfig(n_initial_refines = 5,
                                         t_final = 0.02, dt = 1e-2,
                                         μ = 5e-2,
                                         time_scheme = :sdirk2,
                                         newton_tol = 1e-6,
                                         newton_maxiter = 8,
                                         gmres_tol = 1e-4,
                                         gmres_maxiter = 50,
                                         helmholtz_tol = 1e-10,
                                         poisson_tol = 1e-10)
    state = _INS.build_state(cfg)
    ke0 = _INS.kinetic_energy(state)
    while state.t < cfg.t_final - 1e-12
        dt = min(cfg.dt, cfg.t_final - state.t)
        _INS.step!(state, dt)
    end
    ke_final = _INS.kinetic_energy(state)
    decay_factor = exp(-4 * (2π)^2 * cfg.μ * cfg.t_final)
    ke_expected = ke0 * decay_factor
    # SDIRK2 with γ = 1 - 1/√2 takes a single large step here; allow a
    # bit more slack than the CN test.
    @test isapprox(ke_final, ke_expected; rtol = 0.15)
    @test _INS.l2_divergence(state) < 1e-5
end
