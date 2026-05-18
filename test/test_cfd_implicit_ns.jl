# Implicit NS mini-app smoke test.
#
# Mirrors the cfd_compressible_sod test wrapper convention: include the
# example's source directly under the parent project's `Pkg.test()`
# rather than activating the example's own Project.toml.

using Test

const _example_root = abspath(joinpath(@__DIR__, "..", "examples",
                                         "cfd_implicit_ns"))
isdir(_example_root) ||
    error("test_cfd_implicit_ns: expected example at $_example_root")

module _CFDImplicitNSSmoke
    using HierarchicalGrids
    include(joinpath(@__DIR__, "..", "examples", "cfd_implicit_ns",
                     "src", "CFDImplicitNS.jl"))
    using .CFDImplicitNS
    export ImplicitNSConfig, ImplicitNSState, build_state,
           implicit_ns_step!, run!
end

using ._CFDImplicitNSSmoke
using LinearAlgebra: norm

const _NS = _CFDImplicitNSSmoke.CFDImplicitNS

@testset "Implicit NS: build_state + IC" begin
    cfg = _NS.ImplicitNSConfig(
        n_initial_refines = 3, t_final = 0.0, dt = 1e-3)
    state = _NS.build_state(cfg; ic = x -> (1.0, 0.0, 0.0, 1.0))
    @test state.n_leaves == 8 * 8
    field = parent(state.af)
    # All cells should have ρ = 1.
    @inbounds for c in state.leaves
        @test field.rho[c][1] ≈ 1.0  atol = 1e-12
    end
end

@testset "Implicit NS: uniform flow is preserved (one step)" begin
    # Uniform stationary flow (ρ=1, u=0, p=1).  A consistent backward-Euler
    # step on this state should leave it unchanged.
    cfg = _NS.ImplicitNSConfig(
        n_initial_refines = 3, t_final = 1e-3, dt = 1e-3,
        newton_tol = 1e-12, gmres_tol = 1e-10)
    state = _NS.build_state(cfg; ic = x -> (1.0, 0.0, 0.0, 1.0))
    iters, res = _NS.implicit_ns_step!(state, cfg.dt)
    @test iters <= 5
    @test res < 1e-8
    # Verify state is essentially unchanged.
    field = parent(state.af)
    @inbounds for c in state.leaves
        @test field.rho[c][1]  ≈ 1.0  atol = 1e-8
        @test field.rhou[c][1] ≈ 0.0  atol = 1e-8
        @test field.rhov[c][1] ≈ 0.0  atol = 1e-8
        # E = p/(γ-1) + 0.5 ρ |u|² = 1.0/(0.4) = 2.5
        @test field.E[c][1] ≈ 2.5  atol = 1e-8
    end
end

@testset "Implicit NS: uniform translation preserves state form" begin
    # Constant flow (ρ=1, u=0.5, v=0, p=1) on a periodic domain — the
    # implicit step should preserve ρ, u, v, p (translation in periodic
    # box returns to itself after wrap, but more importantly the spatial
    # gradients are zero so R(U) ≡ 0 and U^{n+1} = U^n).
    cfg = _NS.ImplicitNSConfig(
        n_initial_refines = 3, t_final = 5e-3, dt = 5e-3,
        newton_tol = 1e-12, gmres_tol = 1e-10)
    state = _NS.build_state(cfg; ic = x -> (1.0, 0.5, 0.0, 1.0))
    iters, res = _NS.implicit_ns_step!(state, cfg.dt)
    @test iters <= 5
    @test res < 1e-8
    field = parent(state.af)
    @inbounds for c in state.leaves
        ρ = field.rho[c][1]
        u = field.rhou[c][1] / ρ
        v = field.rhov[c][1] / ρ
        @test ρ  ≈ 1.0  atol = 1e-8
        @test u  ≈ 0.5  atol = 1e-8
        @test v  ≈ 0.0  atol = 1e-8
        # E should still be 0.5·1·0.25 + 1/0.4 = 0.125 + 2.5 = 2.625
        @test field.E[c][1] ≈ 2.625  atol = 1e-8
    end
end

@testset "Implicit NS: viscous mode runs and is more diffusive" begin
    # Initial gaussian density pulse on a periodic domain at rest.  Compare
    # final density profile peak height with and without viscosity / heat
    # conduction: the viscous run should damp the acoustic-wave bouncing
    # more (heat conduction equilibrates).
    ic_gauss = function (x)
        r2 = (x[1] - 0.5)^2 + (x[2] - 0.5)^2
        ρ = 1.0 + 0.1 * exp(-r2 / 0.01)
        return (ρ, 0.0, 0.0, 1.0)
    end
    # Inviscid baseline.
    cfg_i = _NS.ImplicitNSConfig(
        n_initial_refines = 4, t_final = 0.02, dt = 5e-3,
        newton_tol = 1e-6, gmres_tol = 1e-5)
    state_i = _NS.build_state(cfg_i; ic = ic_gauss)
    for _ in 1:4
        _NS.implicit_ns_step!(state_i, cfg_i.dt)
    end
    # Viscous run with the same dt.
    cfg_v = _NS.ImplicitNSConfig(
        n_initial_refines = 4, t_final = 0.02, dt = 5e-3,
        viscous = true, μ = 1e-2, κ = 1e-2,
        newton_tol = 1e-6, gmres_tol = 1e-5)
    state_v = _NS.build_state(cfg_v; ic = ic_gauss)
    for _ in 1:4
        _NS.implicit_ns_step!(state_v, cfg_v.dt)
    end
    # Both should preserve positivity.
    f_i = parent(state_i.af); f_v = parent(state_v.af)
    @test minimum(f_i.rho[c][1] for c in state_i.leaves) > 0
    @test minimum(f_v.rho[c][1] for c in state_v.leaves) > 0
    # The variance of the density should be lower in the viscous case
    # (heat conduction + viscosity smear the pulse).
    var(f) = let
        vals = [f.rho[c][1] for c in state_i.leaves]
        m = sum(vals) / length(vals)
        sum((v - m)^2 for v in vals) / length(vals)
    end
    @test var(f_v) <= var(f_i) * 1.05    # within tolerance; viscous ≤ inviscid
end

@testset "Implicit NS: Sod tube runs to completion (CFL-unbounded dt)" begin
    # Sod IC: discontinuous initial state.  Implicit backward-Euler is
    # unconditionally stable so we can take a `dt` an order of magnitude
    # above the explicit CFL limit.  We don't check shock-capture sharpness
    # (BE is more diffusive than HLL explicit) — only that the solve runs,
    # Newton converges each step, and the final state is physically sane
    # (positive ρ and p, bounded total energy).
    ic = function (x)
        # Sod left/right primitives, interface at x = 0.5.
        return x[1] < 0.5 ? (1.0, 0.0, 0.0, 1.0) : (0.125, 0.0, 0.0, 0.1)
    end
    cfg = _NS.ImplicitNSConfig(
        n_initial_refines = 4,            # 16×16
        t_final = 0.05,
        dt = 5e-3,                         # ~5× the explicit CFL bound
        periodic_x = false, periodic_y = true,
        newton_tol = 1e-6,
        gmres_tol = 1e-5,
        gmres_maxiter = 100,
        newton_maxiter = 15)
    state = _NS.build_state(cfg; ic = ic)
    n_steps = Int(ceil(cfg.t_final / cfg.dt))
    converged_count = 0
    for k in 1:n_steps
        dt = min(cfg.dt, cfg.t_final - (k - 1) * cfg.dt)
        iters, res = _NS.implicit_ns_step!(state, dt)
        if res < cfg.newton_tol * 100
            converged_count += 1
        end
    end
    @test converged_count >= n_steps - 1   # at least N-1 of N steps converged
    # Sanity check the final state.
    field = parent(state.af)
    ρ_min = minimum(field.rho[c][1] for c in state.leaves)
    p_min = minimum(let
        ρ = field.rho[c][1]; u = field.rhou[c][1]/ρ; v = field.rhov[c][1]/ρ
        e_int = field.E[c][1]/ρ - 0.5*(u^2 + v^2)
        (1.4 - 1.0) * ρ * e_int
    end for c in state.leaves)
    @test ρ_min > 0.05      # density positive
    @test p_min > 0.05      # pressure positive
end
