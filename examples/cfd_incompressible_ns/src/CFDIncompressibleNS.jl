"""
    CFDIncompressibleNS

2D incompressible Navier-Stokes mini-app over a single uniform patch,
implementing both `time_scheme = :cn` and `time_scheme = :sdirk2`:

* `:cn` — Crank-Nicolson viscous + explicit centered advection +
  approximate cell-centered projection. The classical fractional-step
  scheme (Chorin / Bell-Colella-Glaz core), constant density.
* `:sdirk2` — 2-stage L-stable SDIRK2 backward implicit on the coupled
  (u, v, p) system. Each stage solves the saddle-point system by JFNK
  (Jacobian-free Newton-Krylov) with a projection-as-preconditioner.
  Unconditionally stable; takes large `Δt` on stiff or steady problems.

Scope (v1): single patch, single level, 2D, periodic BCs, constant
viscosity, constant density. Two-level AMR + Godunov upwinding + lid /
inflow BCs are explicit follow-ups; see README.

Verification: 2D Taylor-Green vortex (analytic exponential decay) on
`[0,1]²`, both schemes; SDIRK2 path additionally verified by temporal
order convergence on the same problem.
"""
module CFDIncompressibleNS

using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, MGWorkspace, MGOptions,
                          MGResult,
                          allocate_phi_rho, allocate_abec_coefs, solve_abec!,
                          allocate_vector_abec, solve_vector_abec!,
                          fill_abec_alpha!, fill_abec_beta!,
                          MonomialBasis, allocate_polynomial_fields, SoA,
                          enumerate_leaves, cell_physical_box,
                          PERIODIC, n_cells
using HierarchicalGrids.Solver: n_levels, patches_at
using HierarchicalGrids.Overlap: FrameBoundaries
using LinearAlgebra: norm
using Krylov: gmres

export IncompressibleNSConfig, IncompressibleNSState
export build_state, step!, run!
export taylor_green_ic, taylor_green_solution
export l2_divergence, kinetic_energy

# ============================================================================
# Configuration
# ============================================================================

"""
    IncompressibleNSConfig

Numerical-experiment knobs.

# Time stepping
- `t_final::Float64` — final time.
- `dt::Float64` — fixed time step.
- `time_scheme::Symbol` — `:cn` (default) or `:sdirk2`.

# Physics
- `μ::Float64` — kinematic viscosity (constant-density, so this is ν).
- `domain_lo::NTuple{2,Float64}`, `domain_hi::NTuple{2,Float64}` — box.

# Resolution
- `n_initial_refines::Int` — domain is `2^n_initial_refines` cells per axis.

# BCs
- `periodic_x::Bool`, `periodic_y::Bool` — v1 only supports periodic.

# Elliptic-solver tolerances
- `helmholtz_tol::Float64`, `helmholtz_maxiter::Int` — CN viscous Helmholtz.
- `poisson_tol::Float64`, `poisson_maxiter::Int` — projection Poisson.

# JFNK (SDIRK2 path)
- `newton_tol::Float64`, `newton_maxiter::Int`.
- `gmres_tol::Float64`, `gmres_maxiter::Int`.
- `linesearch::Bool` — Armijo line search inside Newton.

# Diagnostics
- `verbose::Bool` — print step diagnostics.
"""
Base.@kwdef struct IncompressibleNSConfig
    t_final::Float64 = 0.05
    dt::Float64 = 5e-3
    time_scheme::Symbol = :cn

    μ::Float64 = 1e-2
    domain_lo::NTuple{2, Float64} = (0.0, 0.0)
    domain_hi::NTuple{2, Float64} = (1.0, 1.0)

    n_initial_refines::Int = 5

    periodic_x::Bool = true
    periodic_y::Bool = true

    helmholtz_tol::Float64 = 1e-10
    helmholtz_maxiter::Int = 200
    poisson_tol::Float64 = 1e-10
    poisson_maxiter::Int = 200

    newton_tol::Float64 = 1e-8
    newton_maxiter::Int = 10
    gmres_tol::Float64 = 1e-6
    gmres_maxiter::Int = 80
    linesearch::Bool = true

    verbose::Bool = false
end

# ============================================================================
# State
# ============================================================================

"""
    IncompressibleNSState{T}

Holds the patch hierarchy, MG workspace, cell-centered velocity / pressure
storage, and the elliptic-solver coefficient containers reused across
steps.

`cells.u[c]` and `cells.v[c]` are degree-0 polynomial views (read with
`[1]`). The cell-centered pressure scratch `p_field` is in `phi/rho`
shape so it can be passed directly to `solve_abec!`.
"""
mutable struct IncompressibleNSState{T}
    config::IncompressibleNSConfig
    ph::Any                     # PatchHierarchy{2,T}
    ws::MGWorkspace{2, T}
    bcs_spec::Any
    Nx::Int
    Ny::Int
    hx::T
    hy::T
    leaves::Vector{Int}
    c2c::Array{Int, 2}          # cart_to_cell at (level=1, patch=1)

    # Cell-centered velocity (uses :u and :v slots in a polynomial fieldset).
    cells::Any

    # Pressure scratch (Vector{Vector{NamedTuple}} via allocate_phi_rho).
    # `p_field[1][1].phi` is the cell-centered pressure; `.rho` is the RHS.
    p_field::Vector{Vector{NamedTuple}}
    p_coefs::Any                 # ABecCoefs (A=0, B=1) for the Poisson
    helm_problem::Any            # VectorABecProblem for K=2 CN viscous

    # Time tracking
    t::T
    n_steps::Int
end

# ============================================================================
# Initial-condition utilities
# ============================================================================

"""
    taylor_green_ic(x, ν=0; t=0, U0=1, k=2π) -> (u, v)

Analytic 2D Taylor-Green vortex on `[0,1]²`:

    u(x,y,t) =  U0 sin(k x) cos(k y) exp(-2 k² ν t)
    v(x,y,t) = -U0 cos(k x) sin(k y) exp(-2 k² ν t)
    p(x,y,t) = -(U0²/4) (cos(2 k x) + cos(2 k y)) exp(-4 k² ν t)

`ν = 0` recovers the static profile.
"""
function taylor_green_ic(x::NTuple{2, Float64};
                          ν::Float64 = 0.0, t::Float64 = 0.0,
                          U0::Float64 = 1.0, k::Float64 = 2π)
    decay = exp(-2 * k^2 * ν * t)
    u =  U0 * sin(k * x[1]) * cos(k * x[2]) * decay
    v = -U0 * cos(k * x[1]) * sin(k * x[2]) * decay
    return (u, v)
end

"""
    taylor_green_solution(state, t) -> (u_arr, v_arr)

Build (Nx, Ny) arrays of the analytic Taylor-Green velocity at time `t`.
Useful for L² error reporting.
"""
function taylor_green_solution(state::IncompressibleNSState{T}, t::T) where {T}
    cfg = state.config
    u_arr = zeros(T, state.Nx, state.Ny)
    v_arr = zeros(T, state.Nx, state.Ny)
    @inbounds for j in 1:state.Ny, i in 1:state.Nx
        x = (cfg.domain_lo[1] + (i - 0.5) * state.hx,
             cfg.domain_lo[2] + (j - 0.5) * state.hy)
        u, v = taylor_green_ic(x; ν = cfg.μ, t = t)
        u_arr[i, j] = u
        v_arr[i, j] = v
    end
    return (u_arr, v_arr)
end

# ============================================================================
# Builder
# ============================================================================

"""
    build_state(cfg; ic) -> IncompressibleNSState

Construct the patch hierarchy, MG workspace, velocity / pressure
storage, and elliptic-solver workspaces. `ic(x) -> (u, v)` initialises
the cell-centered velocity. The default `ic` is the Taylor-Green vortex.
"""
function build_state(cfg::IncompressibleNSConfig;
                      ic = x -> taylor_green_ic(x))
    T = Float64
    cfg.periodic_x || error("v1: only periodic BCs supported.")
    cfg.periodic_y || error("v1: only periodic BCs supported.")
    cfg.time_scheme in (:cn, :sdirk2) ||
        error("Unknown time_scheme = $(cfg.time_scheme); use :cn or :sdirk2.")

    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)

    ph = build_uniform_root_hierarchy(Val(2), cfg.n_initial_refines,
                                       cfg.domain_lo, cfg.domain_hi;
                                       physical_bcs = bcs)
    opts = MGOptions(tol = cfg.poisson_tol, maxiter = cfg.poisson_maxiter,
                     cycle = :pcg, pcg_precond = :jacobi)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    Nx, Ny = ws.patch_N[1][1]
    hx, hy = ws.patch_dx[1][1]
    leaves = ws.patch_leaves[1][1]
    c2c = ws.cart_to_cell[1][1]

    # Cell-centered velocity storage (degree-0 monomial, fields :u and :v).
    # Allocate by `n_cells` (total mesh including tree internals) so that
    # `cells.u[c]` is indexable for every leaf cell index in `leaves`.
    basis = MonomialBasis{2, 0}()
    n_storage = n_cells(patches_at(ph, 1)[1].mesh)
    cells = allocate_polynomial_fields(SoA(), basis, n_storage;
                                        u = T, v = T)

    # Apply the initial condition at every leaf cell.
    frame = patches_at(ph, 1)[1]
    @inbounds for c in leaves
        lo, hi = cell_physical_box(frame, c)
        x = ((lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
        u, v = ic(x)
        cells.u[c] = (T(u),)
        cells.v[c] = (T(v),)
    end

    # Pressure scratch (phi/rho fields wired for solve_abec!).
    p_field = allocate_phi_rho(ph)
    p_coefs = allocate_abec_coefs(ws; A = zero(T), B = one(T))

    # Helmholtz problem (K=2 decoupled) for the CN viscous step. A and B
    # are reset per step (they depend on dt and ν).
    helm_problem = allocate_vector_abec(ws, 2; A = one(T), B = one(T))

    return IncompressibleNSState{T}(cfg, ph, ws, bcs_spec, Nx, Ny, hx, hy,
                                     leaves, c2c, cells, p_field, p_coefs,
                                     helm_problem, zero(T), 0)
end

# ============================================================================
# Discrete operators on the single uniform patch
#
# All operators are 2D, centered, periodic. Cell (i, j) has its center
# at `((i - 0.5) hx, (j - 0.5) hy)` from `domain_lo`. Index arithmetic
# wraps with `mod1` to honour periodic BCs.
# ============================================================================

@inline _wrap(i::Int, N::Int) = mod1(i, N)

"""
    cell_at(state, i, j) -> Int

Cell index in the leaves array for Cartesian (i, j) with periodic wrap.
"""
@inline cell_at(state::IncompressibleNSState, i::Int, j::Int) =
    @inbounds state.c2c[_wrap(i, state.Nx), _wrap(j, state.Ny)]

"""
    compute_cell_divergence!(div, u, v, state) -> div

Centered 2nd-order ∇·u at cell centers (periodic). `div` is a length-`n`
`Vector{T}` indexed by cell.
"""
function compute_cell_divergence!(div::Vector{T}, u, v,
                                   state::IncompressibleNSState{T}) where {T}
    invhx = one(T) / (2 * state.hx)
    invhy = one(T) / (2 * state.hy)
    @inbounds for j in 1:state.Ny, i in 1:state.Nx
        c   = cell_at(state, i,   j)
        cE  = cell_at(state, i+1, j)
        cW  = cell_at(state, i-1, j)
        cN  = cell_at(state, i,   j+1)
        cS  = cell_at(state, i,   j-1)
        div[c] = invhx * (_get(u, cE) - _get(u, cW)) +
                 invhy * (_get(v, cN) - _get(v, cS))
    end
    return div
end

"""
    compute_cell_gradient!(gx, gy, phi_raw, state)

Centered 2nd-order ∇φ at cell centers (periodic), writing to per-cell
vectors `gx, gy`.
"""
function compute_cell_gradient!(gx::Vector{T}, gy::Vector{T},
                                 phi_raw::Vector{T},
                                 state::IncompressibleNSState{T}) where {T}
    invhx = one(T) / (2 * state.hx)
    invhy = one(T) / (2 * state.hy)
    @inbounds for j in 1:state.Ny, i in 1:state.Nx
        c  = cell_at(state, i,   j)
        cE = cell_at(state, i+1, j)
        cW = cell_at(state, i-1, j)
        cN = cell_at(state, i,   j+1)
        cS = cell_at(state, i,   j-1)
        gx[c] = invhx * (phi_raw[cE] - phi_raw[cW])
        gy[c] = invhy * (phi_raw[cN] - phi_raw[cS])
    end
    return (gx, gy)
end

"""
    compute_advection!(adv_u, adv_v, u, v, state)

Convective form (u · ∇u) at cell centers via centered differences.
Writes into per-cell scratch vectors.
"""
function compute_advection!(adv_u::Vector{T}, adv_v::Vector{T},
                             u, v,
                             state::IncompressibleNSState{T}) where {T}
    invhx = one(T) / (2 * state.hx)
    invhy = one(T) / (2 * state.hy)
    @inbounds for j in 1:state.Ny, i in 1:state.Nx
        c  = cell_at(state, i,   j)
        cE = cell_at(state, i+1, j)
        cW = cell_at(state, i-1, j)
        cN = cell_at(state, i,   j+1)
        cS = cell_at(state, i,   j-1)
        uc = _get(u, c); vc = _get(v, c)
        ∂u_∂x = invhx * (_get(u, cE) - _get(u, cW))
        ∂u_∂y = invhy * (_get(u, cN) - _get(u, cS))
        ∂v_∂x = invhx * (_get(v, cE) - _get(v, cW))
        ∂v_∂y = invhy * (_get(v, cN) - _get(v, cS))
        adv_u[c] = uc * ∂u_∂x + vc * ∂u_∂y
        adv_v[c] = uc * ∂v_∂x + vc * ∂v_∂y
    end
    return (adv_u, adv_v)
end

"""
    compute_laplacian!(lap, f, state)

Cell-centered 5-point Laplacian (periodic). `f` is read via `_get`,
`lap` is a per-cell vector.
"""
function compute_laplacian!(lap::Vector{T}, f,
                             state::IncompressibleNSState{T}) where {T}
    invhx2 = one(T) / (state.hx * state.hx)
    invhy2 = one(T) / (state.hy * state.hy)
    @inbounds for j in 1:state.Ny, i in 1:state.Nx
        c  = cell_at(state, i,   j)
        cE = cell_at(state, i+1, j)
        cW = cell_at(state, i-1, j)
        cN = cell_at(state, i,   j+1)
        cS = cell_at(state, i,   j-1)
        fc = _get(f, c)
        lap[c] = invhx2 * (_get(f, cE) - 2 * fc + _get(f, cW)) +
                 invhy2 * (_get(f, cN) - 2 * fc + _get(f, cS))
    end
    return lap
end

# Polymorphic getter — supports both a Vector{T} (raw) and a
# PolynomialFieldView (returns the [1] coefficient = the degree-0 value).
@inline _get(f::AbstractVector{T}, c::Int) where {T <: Real} = @inbounds f[c]
@inline _get(f, c::Int) = @inbounds f[c][1]

# Bulk read / write between the (.u, .v) polynomial views and a raw
# Vector{T}. The CN and JFNK paths repeatedly copy in and out; doing it
# field-at-a-time keeps the hot loops simple.
function _read_uv_into_raw!(u_raw::Vector{T}, v_raw::Vector{T},
                             cells, leaves) where {T}
    @inbounds for c in leaves
        u_raw[c] = cells.u[c][1]
        v_raw[c] = cells.v[c][1]
    end
    return nothing
end

function _write_raw_into_uv!(cells, u_raw::Vector{T}, v_raw::Vector{T},
                              leaves) where {T}
    @inbounds for c in leaves
        cells.u[c] = (u_raw[c],)
        cells.v[c] = (v_raw[c],)
    end
    return nothing
end

# ============================================================================
# Approximate cell-centered projection.
#
# Given a tentative velocity (u*, v*) that is not divergence-free, find a
# scalar φ such that
#     u = u* - Δt ∇φ          satisfies     ∇·u = 0.
# This is the cell-centered analogue of the IAMR approximate projection
# (Lai 1993 / IAMR-Lite). Discretely:
#     ∇²φ = (∇·u*) / Δt              (centered cell Laplacian + divergence)
# Solved via solve_abec! with A = 0, B = 1, β = 1. The continuous and
# discrete Laplacians have a nullspace of constants; we anchor by
# subtracting the mean RHS, which is consistent with periodic BCs.
# ============================================================================

"""
    project_cell_velocity!(state, dt) -> (mg_result, l2div_pre, l2div_post)

Mutates `state.cells.u`, `.v` to be discretely divergence-free at cell
centers. Returns the inner MG result and the L² divergence before/after.
"""
function project_cell_velocity!(state::IncompressibleNSState{T},
                                  dt::T) where {T}
    Ncell = length(state.leaves)
    div_pre = Vector{T}(undef, Ncell)
    gx = Vector{T}(undef, Ncell)
    gy = Vector{T}(undef, Ncell)

    compute_cell_divergence!(div_pre, state.cells.u, state.cells.v, state)
    l2_pre = sqrt(sum(d -> d * d, view(div_pre, state.leaves)) /
                  length(state.leaves))

    # Reset φ and load -∇·u* into the RHS slot. ABec convention is
    #     A α φ - B ∇·(β ∇φ) = rho_rhs
    # with A=0, B=1, β=1, so the solve is  -∇²φ = rho_rhs  →  rho_rhs = -(∇·u*)/Δt.
    invdt = one(T) / dt
    rho_storage = state.p_field[1][1].rho
    phi_storage = state.p_field[1][1].phi
    @inbounds for c in state.leaves
        rho_storage[c] = (-invdt * div_pre[c],)
        phi_storage[c] = (zero(T),)
    end

    # Subtract the mean RHS to remove the nullspace (periodic).
    mean_rhs = zero(T)
    @inbounds for c in state.leaves
        mean_rhs += rho_storage[c][1]
    end
    mean_rhs /= length(state.leaves)
    @inbounds for c in state.leaves
        rho_storage[c] = (rho_storage[c][1] - mean_rhs,)
    end

    mgr = solve_abec!(state.p_field, state.p_field, state.p_coefs, state.ws;
                       tol = state.config.poisson_tol,
                       maxiter = state.config.poisson_maxiter)

    # Subtract Δt ∇φ from u*, v*.
    phi_raw = phi_storage isa Vector ? error("phi storage shape") : nothing
    # Actual storage path: state.p_field[1][1].phi is a PolynomialFieldView;
    # its underlying scalar buffer is `phi.pfs.storage.phi :: Vector{T}`.
    phi_raw_vec = state.p_field[1][1].phi.pfs.storage.phi
    compute_cell_gradient!(gx, gy, phi_raw_vec, state)
    @inbounds for c in state.leaves
        u_new = state.cells.u[c][1] - dt * gx[c]
        v_new = state.cells.v[c][1] - dt * gy[c]
        state.cells.u[c] = (u_new,)
        state.cells.v[c] = (v_new,)
    end

    div_post = Vector{T}(undef, Ncell)
    compute_cell_divergence!(div_post, state.cells.u, state.cells.v, state)
    l2_post = sqrt(sum(d -> d * d, view(div_post, state.leaves)) /
                   length(state.leaves))

    return (mgr, l2_pre, l2_post)
end

# ============================================================================
# CN time scheme (Option A): explicit advection + CN viscous + projection
# ============================================================================

"""
    cn_viscous_step!(state, dt, rhs_u, rhs_v) -> NTuple{2, MGResult}

Solve the per-component implicit viscous step

    (I - Δt ν / 2 · Δ) u^{**} = rhs_u
    (I - Δt ν / 2 · Δ) v^{**} = rhs_v

using `solve_vector_abec!`. The ABec operator with A = 1, B = Δt ν / 2,
α = 1, β = 1 matches this exactly. The right-hand sides are the
explicit half-step velocities. Results are written back into
`state.cells.u, .v`.
"""
function cn_viscous_step!(state::IncompressibleNSState{T}, dt::T,
                            rhs_u::Vector{T}, rhs_v::Vector{T}) where {T}
    ν = state.config.μ
    A_coef = one(T)
    B_coef = T(dt * ν / 2)

    helm = state.helm_problem
    for k in 1:2
        # Reset α to 1 everywhere (ABec α is cell-storage-sized).
        αk = helm.coefs[k].alpha[1][1]
        @inbounds for i in eachindex(αk); αk[i] = one(T); end
        # Reset β to 1 everywhere (NTuple of face arrays).
        for d in 1:2
            βkd = helm.coefs[k].beta[1][1][d]
            @inbounds for i in eachindex(βkd); βkd[i] = one(T); end
        end
        # ABec stores A and B as Float64 fields in the struct; we replace
        # the whole `ABecCoefs` to set them (the struct is immutable wrt
        # those fields).
    end
    helm = HierarchicalGrids.VectorABecProblem{2, T, 2}(
        ntuple(k -> HierarchicalGrids.ABecCoefs{2, T}(A_coef, B_coef,
                                                       helm.coefs[k].alpha,
                                                       helm.coefs[k].beta),
                2),
        helm.fields)
    state.helm_problem = helm

    # Push the right-hand sides into fields[k].rho and zero phi as
    # initial guess.
    phi1 = helm.fields[1][1][1].phi
    rho1 = helm.fields[1][1][1].rho
    phi2 = helm.fields[2][1][1].phi
    rho2 = helm.fields[2][1][1].rho
    @inbounds for c in state.leaves
        phi1[c] = (state.cells.u[c][1],)   # warm start
        rho1[c] = (rhs_u[c],)
        phi2[c] = (state.cells.v[c][1],)
        rho2[c] = (rhs_v[c],)
    end

    results = solve_vector_abec!(helm, state.ws;
                                   tol = state.config.helmholtz_tol,
                                   maxiter = state.config.helmholtz_maxiter)

    @inbounds for c in state.leaves
        state.cells.u[c] = (phi1[c][1],)
        state.cells.v[c] = (phi2[c][1],)
    end
    return results
end

"""
    step_cn!(state, dt) -> Tuple

One Crank-Nicolson + explicit-advection + projection step. Returns
`(helm_results, mg_result, l2_div_pre, l2_div_post)`.

Algorithm (per step):

    u^* = u^n - Δt · advec(u^n) + Δt · ν · Δu^n / 2     (explicit half)
    (I - Δt ν / 2 · Δ) u^{**} = u^*                     (implicit half)
    Solve  ∇²φ = ∇·u^{**} / Δt                          (projection)
    u^{n+1} = u^{**} - Δt ∇φ
"""
function step_cn!(state::IncompressibleNSState{T}, dt::T) where {T}
    Ncell = length(state.leaves)
    ν = state.config.μ

    adv_u = Vector{T}(undef, Ncell)
    adv_v = Vector{T}(undef, Ncell)
    lap_u = Vector{T}(undef, Ncell)
    lap_v = Vector{T}(undef, Ncell)
    rhs_u = Vector{T}(undef, Ncell)
    rhs_v = Vector{T}(undef, Ncell)

    compute_advection!(adv_u, adv_v, state.cells.u, state.cells.v, state)
    compute_laplacian!(lap_u, state.cells.u, state)
    compute_laplacian!(lap_v, state.cells.v, state)

    @inbounds for c in state.leaves
        uc = state.cells.u[c][1]
        vc = state.cells.v[c][1]
        # Explicit half-step (advection + half of viscous diffusion).
        rhs_u[c] = uc - dt * adv_u[c] + (dt * ν / 2) * lap_u[c]
        rhs_v[c] = vc - dt * adv_v[c] + (dt * ν / 2) * lap_v[c]
    end

    helm_results = cn_viscous_step!(state, dt, rhs_u, rhs_v)
    mg_result, l2_pre, l2_post = project_cell_velocity!(state, dt)

    state.t += dt
    state.n_steps += 1
    return (helm_results, mg_result, l2_pre, l2_post)
end

# ============================================================================
# SDIRK2 + JFNK time scheme (Option B)
#
# Implicit residual per stage (with stage matrix γ = 1 - 1/√2):
#     F(W) = [ u - γΔt (-advec(u) + ν Δu - ∇p) - rhs_u
#            , v - γΔt (-advec(v) + ν Δv - ∇p_y) - rhs_v
#            , ∇·u ]
# where W = stacked (u_cells, v_cells, p_cells). Newton iterates W via
# J⁻¹ F with J·v applied matrix-free (Pernice-Walker ε directional FD).
# GMRES solves J δ = -F with a projection-as-preconditioner.
# ============================================================================

const SDIRK2_GAMMA = 1.0 - 1.0 / sqrt(2.0)

"""
    JFNKMatvec{T}

Wraps the current Newton iterate W and the stage parameters so that
`mul!(y, op, v)` evaluates `J · v ≈ (F(W + ε v) - F(W)) / ε`. Used as a
linear operator argument to `Krylov.gmres`.
"""
mutable struct JFNKMatvec{T}
    state::IncompressibleNSState{T}
    γdt::T            # γ · dt for the current stage
    W::Vector{T}      # current iterate (flattened)
    F_W::Vector{T}    # F(W), reused
    W_pert::Vector{T} # scratch
    F_pert::Vector{T} # scratch
    rhs_u::Vector{T}
    rhs_v::Vector{T}
end

Base.size(op::JFNKMatvec) = (length(op.W), length(op.W))
Base.size(op::JFNKMatvec, k::Int) = length(op.W)
Base.eltype(::JFNKMatvec{T}) where {T} = T

import LinearAlgebra: mul!
function mul!(y::AbstractVector{T}, op::JFNKMatvec{T},
              v::AbstractVector{T}) where {T}
    # Pernice-Walker ε: ε = sqrt(eps) * (1 + |W|) / |v|.
    nW = norm(op.W) + one(T)
    nv = norm(v) + eps(T)
    ε = sqrt(eps(T)) * nW / nv
    @inbounds @simd for i in eachindex(op.W_pert)
        op.W_pert[i] = op.W[i] + ε * v[i]
    end
    _eval_F!(op.F_pert, op.W_pert, op)
    invε = one(T) / ε
    @inbounds @simd for i in eachindex(y)
        y[i] = (op.F_pert[i] - op.F_W[i]) * invε
    end
    return y
end

"""
    BlockProjectionPrecond{T}

Approximate inverse of the saddle-point Jacobian. Applied as:

    1. Solve (I + γΔt ν L) δu = b_u, (I + γΔt ν L) δv = b_v
       (per-component velocity Helmholtz, frozen advection).
    2. Compute d = ∇·δu + b_p.
    3. Solve ∇²δp = d / (γΔt).
    4. Correct δu -= γΔt · ∂_x δp ;  δv -= γΔt · ∂_y δp.

This is the IAMR/Chorin projection algorithm used as a left-precond.
"""
mutable struct BlockProjectionPrecond{T}
    state::IncompressibleNSState{T}
    γdt::T
end

function ldiv!(y::AbstractVector{T}, P::BlockProjectionPrecond{T},
                b::AbstractVector{T}) where {T}
    s = P.state
    N = length(s.leaves)
    ν = s.config.μ
    γdt = P.γdt

    # Unpack b → (b_u, b_v, b_p).
    b_u = view(b, 1:N)
    b_v = view(b, N+1:2N)
    b_p = view(b, 2N+1:3N)

    # Step 1: per-component Helmholtz solve  (I + γΔt ν L) δu = b_u.
    # Reuse the helm_problem container but with A = 1, B = γΔt ν.
    A_coef = one(T)
    B_coef = γdt * T(ν)
    helm = s.helm_problem
    for k in 1:2
        αk = helm.coefs[k].alpha[1][1]
        @inbounds for i in eachindex(αk); αk[i] = one(T); end
        for d in 1:2
            βkd = helm.coefs[k].beta[1][1][d]
            @inbounds for i in eachindex(βkd); βkd[i] = one(T); end
        end
    end
    helm = HierarchicalGrids.VectorABecProblem{2, T, 2}(
        ntuple(k -> HierarchicalGrids.ABecCoefs{2, T}(A_coef, B_coef,
                                                       helm.coefs[k].alpha,
                                                       helm.coefs[k].beta),
                2),
        helm.fields)
    s.helm_problem = helm

    phi_u = helm.fields[1][1][1].phi
    rho_u = helm.fields[1][1][1].rho
    phi_v = helm.fields[2][1][1].phi
    rho_v = helm.fields[2][1][1].rho

    @inbounds for (k, c) in enumerate(s.leaves)
        rho_u[c] = (b_u[k],)
        phi_u[c] = (zero(T),)
        rho_v[c] = (b_v[k],)
        phi_v[c] = (zero(T),)
    end

    solve_vector_abec!(helm, s.ws;
                        tol = s.config.helmholtz_tol,
                        maxiter = s.config.helmholtz_maxiter)

    # Step 2: write δu, δv into raw vectors and compute ∇·δu.
    du_raw = Vector{T}(undef, N)
    dv_raw = Vector{T}(undef, N)
    @inbounds for (k, c) in enumerate(s.leaves)
        du_raw[k] = phi_u[c][1]
        dv_raw[k] = phi_v[c][1]
    end
    # Compute ∇·(δu, δv) but in storage-cell order. Re-pack into a
    # per-cell raw buffer indexed by `c` so cell_at-based stencil works.
    u_storage = Vector{T}(undef, length(s.cells.u))
    v_storage = similar(u_storage)
    @inbounds for (k, c) in enumerate(s.leaves)
        u_storage[c] = du_raw[k]
        v_storage[c] = dv_raw[k]
    end
    div_du = Vector{T}(undef, length(u_storage))
    compute_cell_divergence!(div_du, u_storage, v_storage, s)

    # Step 3: solve ∇²δp = (∇·δu + b_p) / γdt.
    invγdt = one(T) / γdt
    rho_p = s.p_field[1][1].rho
    phi_p = s.p_field[1][1].phi
    @inbounds for (k, c) in enumerate(s.leaves)
        rho_p[c] = (-invγdt * (div_du[c] + b_p[k]),)
        phi_p[c] = (zero(T),)
    end
    # Remove the mean RHS (periodic nullspace).
    mean_rhs = zero(T)
    @inbounds for c in s.leaves
        mean_rhs += rho_p[c][1]
    end
    mean_rhs /= length(s.leaves)
    @inbounds for c in s.leaves
        rho_p[c] = (rho_p[c][1] - mean_rhs,)
    end

    solve_abec!(s.p_field, s.p_field, s.p_coefs, s.ws;
                  tol = s.config.poisson_tol,
                  maxiter = s.config.poisson_maxiter)

    # Step 4: correct δu, δv by gradient.
    phi_raw = s.p_field[1][1].phi.pfs.storage.phi
    gx = Vector{T}(undef, length(u_storage))
    gy = Vector{T}(undef, length(u_storage))
    compute_cell_gradient!(gx, gy, phi_raw, s)

    @inbounds for (k, c) in enumerate(s.leaves)
        y[k]       = du_raw[k] - γdt * gx[c]
        y[k + N]   = dv_raw[k] - γdt * gy[c]
        y[k + 2N]  = phi_raw[c]
    end
    return y
end

"""
    _eval_F!(out, W, op)

Compute the SDIRK2 stage residual into `out`. `W` is the flattened
iterate `[u_1..u_N, v_1..v_N, p_1..p_N]`. Mutates `state.cells.u, .v`
through the per-cell scratch so cell_at-based stencils work. After
returning, `state.cells` is left holding `W`'s u, v values (caller
must restore if needed).
"""
function _eval_F!(out::Vector{T}, W::AbstractVector{T},
                  op::JFNKMatvec{T}) where {T}
    s = op.state
    N = length(s.leaves)
    ν = s.config.μ
    γdt = op.γdt

    # Unpack into per-cell storage buffers indexed by `c` (so cell_at works).
    u_storage = Vector{T}(undef, length(s.cells.u))
    v_storage = similar(u_storage)
    p_storage = similar(u_storage)
    @inbounds for (k, c) in enumerate(s.leaves)
        u_storage[c] = W[k]
        v_storage[c] = W[k + N]
        p_storage[c] = W[k + 2N]
    end

    adv_u = Vector{T}(undef, length(u_storage))
    adv_v = similar(adv_u)
    lap_u = similar(adv_u)
    lap_v = similar(adv_u)
    gx    = similar(adv_u)
    gy    = similar(adv_u)
    divu  = similar(adv_u)

    compute_advection!(adv_u, adv_v, u_storage, v_storage, s)
    compute_laplacian!(lap_u, u_storage, s)
    compute_laplacian!(lap_v, v_storage, s)
    compute_cell_gradient!(gx, gy, p_storage, s)
    compute_cell_divergence!(divu, u_storage, v_storage, s)

    @inbounds for (k, c) in enumerate(s.leaves)
        # F_u = u - γΔt (-advec_u + ν Δu - ∂p/∂x) - rhs_u
        out[k] = u_storage[c] -
                 γdt * (-adv_u[c] + ν * lap_u[c] - gx[c]) -
                 op.rhs_u[k]
        # F_v = v - γΔt (-advec_v + ν Δv - ∂p/∂y) - rhs_v
        out[k + N] = v_storage[c] -
                      γdt * (-adv_v[c] + ν * lap_v[c] - gy[c]) -
                      op.rhs_v[k]
        # F_p = ∇·u  (we want this driven to zero — divergence constraint)
        out[k + 2N] = divu[c]
    end
    return out
end

"""
    sdirk2_stage!(state, dt, rhs_u, rhs_v, γdt; warm_start) -> (iters, residual)

Solve one implicit SDIRK2 stage using Newton-Krylov. `rhs_u, rhs_v` are
the per-cell explicit-side right-hand sides; `γdt = γ * dt` for the
current stage. Writes the converged (u, v, p) back into `state.cells`.
"""
function sdirk2_stage!(state::IncompressibleNSState{T}, dt::T,
                        rhs_u::Vector{T}, rhs_v::Vector{T},
                        γdt::T) where {T}
    cfg = state.config
    N = length(state.leaves)
    n_total = 3 * N

    # Initial guess: u, v from state.cells; p = 0.
    W = Vector{T}(undef, n_total)
    @inbounds for (k, c) in enumerate(state.leaves)
        W[k]      = state.cells.u[c][1]
        W[k + N]  = state.cells.v[c][1]
        W[k + 2N] = zero(T)
    end

    F_W   = Vector{T}(undef, n_total)
    W_per = Vector{T}(undef, n_total)
    F_per = Vector{T}(undef, n_total)

    op = JFNKMatvec{T}(state, γdt, W, F_W, W_per, F_per, rhs_u, rhs_v)
    precond = BlockProjectionPrecond{T}(state, γdt)

    res = zero(T)
    iters = 0
    for newton_iter in 1:cfg.newton_maxiter
        iters = newton_iter
        _eval_F!(F_W, W, op)
        res = norm(F_W)
        if res < cfg.newton_tol
            break
        end

        # Build -F as RHS for GMRES.
        b = Vector{T}(undef, n_total)
        @inbounds @simd for i in eachindex(b); b[i] = -F_W[i]; end

        # Apply right preconditioner: solve M^{-1} J δ = M^{-1} b, then
        # δ = M^{-1} z is the Newton update where z = b.
        # We embed precond as left-precond on the operator action: define
        # PJ with `mul!(y, PJ, v)` returning M^{-1} (J v). Krylov.jl
        # supports either; here we use simple right-preconditioning via
        # the helper below.
        δ = _gmres_with_precond(op, precond, b, cfg)

        # Optional Armijo line search.
        α = one(T)
        if cfg.linesearch
            α = _armijo_linesearch!(W, δ, F_W, op, res)
        end
        @inbounds @simd for i in eachindex(W)
            W[i] += α * δ[i]
        end
    end

    # Unpack final W back into state.cells.
    @inbounds for (k, c) in enumerate(state.leaves)
        state.cells.u[c] = (W[k],)
        state.cells.v[c] = (W[k + N],)
    end
    return (iters, res)
end

# Solve J δ = b via Krylov.gmres with the BlockProjectionPrecond applied
# as a right-preconditioner: we form a "preconditioned operator" P_J
# such that (P_J) z = J (M^{-1} z) and recover δ = M^{-1} z.
function _gmres_with_precond(op::JFNKMatvec{T},
                              precond::BlockProjectionPrecond{T},
                              b::Vector{T},
                              cfg::IncompressibleNSConfig) where {T}
    n = length(b)
    # Wrap as a struct so Krylov can mul! it.
    Pop = _RightPrecondOp{T}(op, precond, Vector{T}(undef, n),
                              Vector{T}(undef, n))
    z, stats = gmres(Pop, b;
                       rtol = cfg.gmres_tol, atol = 0.0,
                       itmax = cfg.gmres_maxiter)
    # Recover δ = M^{-1} z.
    δ = Vector{T}(undef, n)
    ldiv!(δ, precond, z)
    return δ
end

mutable struct _RightPrecondOp{T}
    op::JFNKMatvec{T}
    precond::BlockProjectionPrecond{T}
    scratch_in::Vector{T}
    scratch_out::Vector{T}
end

Base.size(o::_RightPrecondOp) = (length(o.scratch_in), length(o.scratch_in))
Base.size(o::_RightPrecondOp, k::Int) = length(o.scratch_in)
Base.eltype(::_RightPrecondOp{T}) where {T} = T

function mul!(y::AbstractVector{T}, P::_RightPrecondOp{T},
               z::AbstractVector{T}) where {T}
    # Apply M^{-1} to z, then J to the result.
    ldiv!(P.scratch_out, P.precond, z)
    mul!(y, P.op, P.scratch_out)
    return y
end

# Cheap Armijo backtracking, accepts the full step if it reduces |F|.
function _armijo_linesearch!(W::Vector{T}, δ::Vector{T}, F_W::Vector{T},
                              op::JFNKMatvec{T}, res::T) where {T}
    α = one(T)
    F_try = Vector{T}(undef, length(W))
    W_try = Vector{T}(undef, length(W))
    for _ in 1:6
        @inbounds @simd for i in eachindex(W)
            W_try[i] = W[i] + α * δ[i]
        end
        _eval_F!(F_try, W_try, op)
        res_try = norm(F_try)
        if res_try < (1.0 - 1e-4 * α) * res
            return α
        end
        α *= 0.5
    end
    return α
end

"""
    step_sdirk2!(state, dt) -> (stage1_iters, stage1_res, stage2_iters, stage2_res)

Two-stage L-stable SDIRK2:

    γ = 1 - 1/√2
    Stage 1: solve  u_1 - γΔt L(u_1, p_1) = u^n,  ∇·u_1 = 0.
    Stage 2: solve  u_2 - γΔt L(u_2, p_2) = u^n + (1-γ)Δt k_1,  ∇·u_2 = 0
             where k_1 = (u_1 - u^n)/(γΔt) is the first stage slope.
    u^{n+1} = u_2.

Both stages share the same JFNK + projection-precond infrastructure.
"""
function step_sdirk2!(state::IncompressibleNSState{T}, dt::T) where {T}
    N = length(state.leaves)
    γ = T(SDIRK2_GAMMA)
    γdt = γ * dt

    # Save u^n.
    u_n = Vector{T}(undef, N)
    v_n = Vector{T}(undef, N)
    @inbounds for (k, c) in enumerate(state.leaves)
        u_n[k] = state.cells.u[c][1]
        v_n[k] = state.cells.v[c][1]
    end

    # --- Stage 1 ---  rhs = u^n
    iters1, res1 = sdirk2_stage!(state, dt, u_n, v_n, γdt)

    # k_1 = (u_1 - u^n) / γdt (stage slope).
    u_1 = Vector{T}(undef, N)
    v_1 = Vector{T}(undef, N)
    @inbounds for (k, c) in enumerate(state.leaves)
        u_1[k] = state.cells.u[c][1]
        v_1[k] = state.cells.v[c][1]
    end
    invγdt = one(T) / γdt
    rhs_u = Vector{T}(undef, N)
    rhs_v = Vector{T}(undef, N)
    @inbounds for k in 1:N
        k1u = (u_1[k] - u_n[k]) * invγdt
        k1v = (v_1[k] - v_n[k]) * invγdt
        rhs_u[k] = u_n[k] + (one(T) - γ) * dt * k1u
        rhs_v[k] = v_n[k] + (one(T) - γ) * dt * k1v
    end

    # --- Stage 2 ---
    iters2, res2 = sdirk2_stage!(state, dt, rhs_u, rhs_v, γdt)

    state.t += dt
    state.n_steps += 1
    return (iters1, res1, iters2, res2)
end

# ============================================================================
# Public step / run interface
# ============================================================================

"""
    step!(state, dt) -> info

Take one time step using the configured `time_scheme`. Returns the
per-scheme diagnostic tuple from `step_cn!` or `step_sdirk2!`.
"""
function step!(state::IncompressibleNSState{T}, dt::T) where {T}
    if state.config.time_scheme === :cn
        return step_cn!(state, dt)
    elseif state.config.time_scheme === :sdirk2
        return step_sdirk2!(state, dt)
    else
        error("Unknown time_scheme = $(state.config.time_scheme)")
    end
end

"""
    run!(cfg; ic) -> state

Build a fresh state and step until `t >= t_final`. Returns the state.
"""
function run!(cfg::IncompressibleNSConfig;
                ic = x -> taylor_green_ic(x))
    state = build_state(cfg; ic = ic)
    while state.t < cfg.t_final - 1e-12
        dt = min(cfg.dt, cfg.t_final - state.t)
        step!(state, dt)
        if cfg.verbose
            ke = kinetic_energy(state)
            l2 = l2_divergence(state)
            println("step $(state.n_steps)  t=$(round(state.t, sigdigits=4))",
                    "  KE=$(round(ke, sigdigits=4))  |div|=$(round(l2, sigdigits=3))")
        end
    end
    return state
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    l2_divergence(state) -> Float64

L² norm of the cell-centered divergence (centered differences), useful
for verifying the projection step.
"""
function l2_divergence(state::IncompressibleNSState{T}) where {T}
    div = Vector{T}(undef, length(state.cells.u))
    compute_cell_divergence!(div, state.cells.u, state.cells.v, state)
    s = zero(T)
    @inbounds for c in state.leaves
        s += div[c] * div[c]
    end
    return sqrt(s / length(state.leaves))
end

"""
    kinetic_energy(state) -> Float64

Domain-averaged kinetic energy `(1/2) <u·u>`.
"""
function kinetic_energy(state::IncompressibleNSState{T}) where {T}
    e = zero(T)
    @inbounds for c in state.leaves
        u = state.cells.u[c][1]
        v = state.cells.v[c][1]
        e += 0.5 * (u * u + v * v)
    end
    return e / length(state.leaves)
end

end # module CFDIncompressibleNS
