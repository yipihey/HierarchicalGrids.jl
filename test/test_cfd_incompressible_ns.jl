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
    # Taylor-Green is analytically divergence-free; both face and cell
    # divergence collapse to machine precision on a periodic grid.
    @test _INS.l2_divergence(state) < 1e-12
    @test _INS.l2_face_divergence(state) < 1e-12
    # <(1/2)(u² + v²)> = 1/4 for U0 = 1, k = 2π on [0,1]².
    @test isapprox(_INS.kinetic_energy(state), 0.25; atol = 1e-3)
end

@testset "Incompressible NS (:cn): one step drives face divergence ~ tol" begin
    cfg = _INS.IncompressibleNSConfig(n_initial_refines = 4,
                                         t_final = 5e-3, dt = 5e-3,
                                         μ = 1e-2,
                                         time_scheme = :cn,
                                         helmholtz_tol = 1e-12,
                                         poisson_tol = 1e-12)
    state = _INS.build_state(cfg)
    _INS.step!(state, cfg.dt)
    # MAC projection drives the face divergence to MG tolerance.
    @test _INS.l2_face_divergence(state) < 1e-7
    # Cell divergence is O(h²) by design (approximate projection); for
    # Taylor-Green at N=16 that's a few × 1e-3.
    @test _INS.l2_divergence(state) < 1e-2
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
    @test isapprox(ke_final, ke_expected; rtol = 0.15)
    # Sanity: face divergence still small after the run.
    @test _INS.l2_face_divergence(state) < 1e-6
end

@testset "Incompressible NS (:sdirk2): Newton + GMRES + projection converge" begin
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
    @test iters1 < cfg.newton_maxiter           # actually converged
    @test iters2 < cfg.newton_maxiter
    @test res1 < cfg.newton_tol
    @test res2 < cfg.newton_tol
    # Approximate-projection face divergence is bounded — see README's
    # "Limitations" section: cell-centered SDIRK2 + averaging-based
    # projection leaves O(h²)-ish residual face divergence on a step
    # whose JFNK has introduced unbalanced rotational modes.
    @test _INS.l2_face_divergence(state) < 1e-3
end

@testset "Incompressible NS (:sdirk2): Taylor-Green energy still decays" begin
    # The two-stage SDIRK2 + operator-split projection has order reduction
    # (it's roughly 1st-order in time, NOT the 2nd order of "pure" SDIRK2
    # on a saddle-point system). Verify the qualitative decay: the energy
    # decays monotonically and ends in a reasonable neighbourhood of the
    # analytic value. A tighter convergence-rate test belongs to v2,
    # which would couple pressure into the JFNK saddle-point system.
    cfg = _INS.IncompressibleNSConfig(n_initial_refines = 5,
                                         t_final = 0.01, dt = 5e-3,
                                         μ = 5e-2,
                                         time_scheme = :sdirk2,
                                         newton_tol = 1e-7,
                                         newton_maxiter = 10,
                                         gmres_tol = 1e-5,
                                         gmres_maxiter = 80,
                                         helmholtz_tol = 1e-12,
                                         poisson_tol = 1e-12)
    state = _INS.build_state(cfg)
    ke0 = _INS.kinetic_energy(state)
    while state.t < cfg.t_final - 1e-12
        dt = min(cfg.dt, cfg.t_final - state.t)
        _INS.step!(state, dt)
    end
    ke_final = _INS.kinetic_energy(state)
    decay_factor = exp(-4 * (2π)^2 * cfg.μ * cfg.t_final)
    ke_expected = ke0 * decay_factor
    # Decayed (didn't blow up).
    @test ke_final < ke0
    # Within the order-reduced bound (CN gets ~5%, SDIRK2 ~15% over this
    # short integration; the analytic decay factor is ~0.92 so 15% rel
    # error means we landed roughly between 0.78·ke0 and 1.06·ke0).
    @test isapprox(ke_final, ke_expected; rtol = 0.20)
end
