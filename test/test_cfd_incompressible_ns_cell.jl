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

# ============================================================================
# Step 2: implicit viscous step
# ============================================================================

@testset "INSCell viscous: stationary Taylor-Green decay (uniform)" begin
    # Backward-Euler one step of the heat equation on Taylor-Green:
    #   u^{n+1} = u^n + Δt ν L u^{n+1}        (= explicit form: u^n / (1 - Δt ν λ))
    # For TG, L u = -2 k² u (eigenfunction), so the backward-Euler answer is
    #   u^{n+1} = u^n / (1 + 2 Δt ν k²).
    # The kinetic energy then scales as (1/(1 + 2 Δt ν k²))² ≈ exp(-4 ν k² Δt) for
    # small Δt — match within the 1st-order BE truncation.
    cfg = _C.NSCellConfig(n_base_refines = 5, t_final = 0.0, dt = 0.0,
                              μ = 5e-2,
                              helmholtz_tol = 1e-12,
                              helmholtz_maxiter = 500)
    state = _C.build_state(cfg)
    ke0 = _C.kinetic_energy(state)

    dt = 1e-2
    stats_u, stats_v = _C.viscous_step!(state, dt)
    @test stats_u.solved
    @test stats_v.solved

    ke = _C.kinetic_energy(state)
    # Backward-Euler analytic factor for the TG eigenmode:
    factor = 1.0 / (1.0 + 2.0 * dt * cfg.μ * (2π)^2)^2
    @test isapprox(ke, ke0 * factor; rtol = 1e-3)
end

@testset "INSCell viscous: multi-step Taylor-Green energy decay" begin
    # Repeated backward-Euler steps — the kinetic energy should track the
    # analytic exp(-4 (2π)² ν t) decay closely (BE has 1st-order time
    # truncation but the eigenmode is exact, so the multi-step factor is
    # 1/(1+2 dt ν k²)^(2N) which equals exp(-4 ν k² t) in the limit).
    cfg = _C.NSCellConfig(n_base_refines = 5, t_final = 0.0, dt = 0.0,
                              μ = 5e-2,
                              helmholtz_tol = 1e-12,
                              helmholtz_maxiter = 500)
    state = _C.build_state(cfg)
    ke0 = _C.kinetic_energy(state)
    dt = 5e-3
    t_final = 0.02
    nsteps = Int(round(t_final / dt))
    for _ in 1:nsteps
        _C.viscous_step!(state, dt)
    end
    ke = _C.kinetic_energy(state)
    factor_per_step = 1.0 / (1.0 + 2.0 * dt * cfg.μ * (2π)^2)^2
    expected = ke0 * factor_per_step^nsteps
    @test isapprox(ke, expected; rtol = 5e-3)
    # And it should be reasonably close to the analytic decay too.
    analytic = ke0 * exp(-4 * (2π)^2 * cfg.μ * (dt * nsteps))
    @test isapprox(ke, analytic; rtol = 0.05)
end

@testset "INSCell viscous: AMR (one step runs, CG converges)" begin
    # On an AMR mesh the Laplacian is 1st-order at C/F faces (documented
    # limitation; step 3 upgrades to Martin-Colella). Verify the solve
    # still runs and energy decreases.
    cfg = _C.NSCellConfig(n_base_refines = 4, refine_center = true,
                              t_final = 0.0, dt = 0.0,
                              μ = 5e-2,
                              helmholtz_tol = 1e-10,
                              helmholtz_maxiter = 500)
    state = _C.build_state(cfg)
    ke0 = _C.kinetic_energy(state)
    dt = 5e-3
    stats_u, stats_v = _C.viscous_step!(state, dt)
    @test stats_u.solved
    @test stats_v.solved
    ke = _C.kinetic_energy(state)
    @test ke < ke0
    # Bounded — not blowing up.
    @test ke > 0.5 * ke0
end
