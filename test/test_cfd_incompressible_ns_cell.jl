# Cell-mode incompressible-NS scaffold, step 1: passive 2nd-order vector
# advection across coarse-fine interfaces.

using Test
using HierarchicalGrids: cell_physical_box

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

# ============================================================================
# Step 3: cell-mode Poisson solver
# ============================================================================

# Manufactured Taylor-Green-style solution:
#     φ(x, y)        = sin(2π x) cos(2π y)
#     −∇² φ          = 8 π² sin(2π x) cos(2π y) = 8 π² · φ
# So the RHS is just `8 π² · φ_exact`.
function _poisson_mms_data(state)
    n_leaves = length(state.leaves)
    rhs = Vector{Float64}(undef, n_leaves)
    phi_exact = Vector{Float64}(undef, n_leaves)
    @inbounds for (k, c) in enumerate(state.leaves)
        lo, hi = cell_physical_box(state.frame, c)
        x = (0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
        φ = sin(2π * x[1]) * cos(2π * x[2])
        phi_exact[k] = φ
        rhs[k] = 8 * π^2 * φ
    end
    # Anchor the exact solution to mean zero (matches what solve does).
    num = 0.0; vol = 0.0
    @inbounds for (k, c) in enumerate(state.leaves)
        V = (cell_physical_box(state.frame, c)[2][1] -
             cell_physical_box(state.frame, c)[1][1]) *
            (cell_physical_box(state.frame, c)[2][2] -
             cell_physical_box(state.frame, c)[1][2])
        num += phi_exact[k] * V
        vol += V
    end
    mean_phi = num / vol
    @inbounds for k in 1:length(phi_exact); phi_exact[k] -= mean_phi; end
    return (rhs, phi_exact)
end

# L² error between solver output and the analytic reference.
function _poisson_l2_error(phi::Vector{Float64}, phi_exact::Vector{Float64},
                            state)
    num = 0.0; vol = 0.0
    @inbounds for (k, c) in enumerate(state.leaves)
        lo, hi = cell_physical_box(state.frame, c)
        V = (hi[1] - lo[1]) * (hi[2] - lo[2])
        d = phi[k] - phi_exact[k]
        num += d * d * V
        vol += V
    end
    return sqrt(num / vol)
end

@testset "INSCell Poisson: MMS recovers analytic φ (uniform)" begin
    cfg = _C.NSCellConfig(n_base_refines = 5,
                              poisson_tol = 1e-12, poisson_maxiter = 4000)
    state = _C.build_state(cfg)
    rhs, phi_exact = _poisson_mms_data(state)
    phi = zeros(Float64, length(state.leaves))
    stats = _C.solve_cell_poisson!(phi, rhs, state)
    @test stats.solved
    # Recovery error: O(h²) for the 5-point Laplacian on a 32² grid.
    err = _poisson_l2_error(phi, phi_exact, state)
    @test err < 5e-3
end

@testset "INSCell Poisson: spatial convergence ≥ 2 (uniform)" begin
    errs = Float64[]
    for n in 3:6
        cfg = _C.NSCellConfig(n_base_refines = n,
                                  poisson_tol = 1e-12, poisson_maxiter = 8000)
        state = _C.build_state(cfg)
        rhs, phi_exact = _poisson_mms_data(state)
        phi = zeros(Float64, length(state.leaves))
        _C.solve_cell_poisson!(phi, rhs, state)
        push!(errs, _poisson_l2_error(phi, phi_exact, state))
    end
    @test _order(errs[1], errs[2]) > 1.7
    @test _order(errs[2], errs[3]) > 1.85
    @test _order(errs[3], errs[4]) > 1.9
end

@testset "INSCell Poisson: AMR (CG converges, error bounded)" begin
    cfg = _C.NSCellConfig(n_base_refines = 4, refine_center = true,
                              poisson_tol = 1e-10, poisson_maxiter = 5000)
    state = _C.build_state(cfg)
    rhs, phi_exact = _poisson_mms_data(state)
    phi = zeros(Float64, length(state.leaves))
    stats = _C.solve_cell_poisson!(phi, rhs, state)
    @test stats.solved
    err = _poisson_l2_error(phi, phi_exact, state)
    # 1st-order at C/F → coarser asymptotic, but error should still be
    # small at N=16-equivalent. Sanity-check it's not catastrophic.
    @test err < 5e-2
end
