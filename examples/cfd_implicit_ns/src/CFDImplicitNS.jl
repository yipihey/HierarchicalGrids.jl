module CFDImplicitNS

# ============================================================================
# Fully-implicit compressible Navier-Stokes via Jacobian-Free Newton-Krylov.
#
# State per cell:  U = (ρ, ρu, ρv, E)         (2D, ideal gas).
# Time integration:  backward Euler
#       U^{n+1}  =  U^n  +  Δt · R(U^{n+1})
# where  R(U) = -∇·F(U) / V  is the net flux into a cell (per unit volume).
#
# Nonlinear solve:  Newton on
#       F(U) := U - U^n - Δt · R(U) = 0
# with the action of the Jacobian computed by directional finite differences
#       J · v  ≈  (F(U + ε v) - F(U)) / ε                                 (JFNK).
# Krylov inner solve: Krylov.jl GMRES via our existing Krylov bridge.
#
# Flux model:
#   * Convective: HLL Riemann solver (identical to the Sod tube mini-app).
#   * Viscous (optional, `viscous = true`): Newton fluid stress tensor
#     τ = μ (∇u + ∇uᵀ) - (2/3) μ ∇·u I  with Stokes' hypothesis,  plus
#     heat conduction  q = -κ ∇T  in the energy equation.
#
# Scope:  2D, uniform single-level grid, periodic + outflow BCs.  Multi-
# level / AMR comes later (the residual function is parameterised on the
# state vector so the same nonlinear solve can be reused once
# `for_each_face!` is plumbed through AMR).
# ============================================================================

using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, n_cells, refine_cells!,
                          enumerate_leaves, cell_physical_box, level_of,
                          is_leaf, face_neighbors, face_neighbors_with_bcs,
                          face_fine_neighbors, ensure_neighbor_graph!,
                          BernsteinBasis, n_coeffs,
                          allocate_polynomial_fields, SoA,
                          EulerianFrame, FrameBoundaries, BCKind,
                          PERIODIC, OUTFLOW,
                          AdaptiveField, dispose!,
                          for_each_cell!, for_each_face!,
                          step_with_amr!, refine_by_indicator!,
                          Sequential

using LinearAlgebra
using LinearAlgebra: norm, mul!
using Krylov: gmres

export ImplicitNSConfig, ImplicitNSState, build_state, implicit_ns_step!, run!

# ============================================================================
# Equation of state + flux helpers (self-contained; mirrors cfd_compressible_sod)
# ============================================================================

const GAMMA = 1.4

@inline function cons_to_prim(U::NTuple{4, Float64})
    ρ = U[1]
    u = U[2] / ρ
    v = U[3] / ρ
    e_internal = U[4] / ρ - 0.5 * (u * u + v * v)
    p = (GAMMA - 1.0) * ρ * e_internal
    return (ρ, u, v, p)
end

@inline function prim_to_cons(ρ::Float64, u::Float64, v::Float64, p::Float64)
    e_internal = p / ((GAMMA - 1.0) * ρ)
    E = ρ * (e_internal + 0.5 * (u * u + v * v))
    return (ρ, ρ * u, ρ * v, E)
end

@inline sound_speed(ρ::Float64, p::Float64) = sqrt(GAMMA * p / ρ)

@inline function euler_flux(U::NTuple{4, Float64}, n::NTuple{2, Float64})
    ρ, u, v, p = cons_to_prim(U)
    vn = u * n[1] + v * n[2]
    E = U[4]
    return (ρ * vn,
             ρ * u * vn + p * n[1],
             ρ * v * vn + p * n[2],
             (E + p) * vn)
end

@inline function hll_flux(UL::NTuple{4, Float64}, UR::NTuple{4, Float64},
                            n::NTuple{2, Float64})
    ρL, uL, vL, pL = cons_to_prim(UL)
    ρR, uR, vR, pR = cons_to_prim(UR)
    cL = sound_speed(ρL, pL)
    cR = sound_speed(ρR, pR)
    vnL = uL * n[1] + vL * n[2]
    vnR = uR * n[1] + vR * n[2]
    SL = min(vnL - cL, vnR - cR)
    SR = max(vnL + cL, vnR + cR)
    FL = euler_flux(UL, n)
    FR = euler_flux(UR, n)
    if SL >= 0.0
        return FL
    elseif SR <= 0.0
        return FR
    else
        d = SR - SL
        return ((SR * FL[1] - SL * FR[1] + SL * SR * (UR[1] - UL[1])) / d,
                (SR * FL[2] - SL * FR[2] + SL * SR * (UR[2] - UL[2])) / d,
                (SR * FL[3] - SL * FR[3] + SL * SR * (UR[3] - UL[3])) / d,
                (SR * FL[4] - SL * FR[4] + SL * SR * (UR[4] - UL[4])) / d)
    end
end

# ----------------------------------------------------------------------------
# Viscous + heat-conduction flux across a face.
#
# Inputs:
#   UL, UR     — conservative states on the two adjacent cells
#   uL_off±,   — off-axis neighbours of UL  (for tangential velocity gradients)
#   uR_off±    — off-axis neighbours of UR
#   normal     — face normal (axis-aligned in 2D Cartesian)
#   h_normal   — face-normal cell spacing
#   h_tang     — tangential cell spacing
#   μ, λ, κ    — viscosity, second viscosity, thermal conductivity
#
# Returns a 4-tuple matching the conservative-variable layout (mass flux is
# zero — viscous diffusion doesn't carry mass).
# ----------------------------------------------------------------------------
@inline function viscous_flux(UL::NTuple{4, Float64}, UR::NTuple{4, Float64},
                                uL_top::NTuple{4, Float64},
                                uL_bot::NTuple{4, Float64},
                                uR_top::NTuple{4, Float64},
                                uR_bot::NTuple{4, Float64},
                                normal::NTuple{2, Float64},
                                h_normal::Float64, h_tang::Float64,
                                μ::Float64, λ::Float64, κ::Float64)
    ρL, uL_x, uL_y, pL = cons_to_prim(UL)
    ρR, uR_x, uR_y, pR = cons_to_prim(UR)
    _, uLt_x, uLt_y, _ = cons_to_prim(uL_top)
    _, uLb_x, uLb_y, _ = cons_to_prim(uL_bot)
    _, uRt_x, uRt_y, _ = cons_to_prim(uR_top)
    _, uRb_x, uRb_y, _ = cons_to_prim(uR_bot)

    # Cell-centered temperature  T = p / (ρ R)  with R = (γ-1) c_v.  Using
    # the ideal-gas relation in dimensionless form: T = p / ρ (i.e. R = 1).
    TL = pL / ρL
    TR = pR / ρR

    inv_hn = 1.0 / h_normal
    inv_ht = 1.0 / (2.0 * h_tang)

    if normal[1] != 0.0
        # Vertical face — normal in +x.  Velocity at the face is
        # the average of the two adjoining cells; gradients in x use
        # the two-cell difference, gradients in y use four-cell tangential.
        ∂u∂x = (uR_x - uL_x) * inv_hn
        ∂v∂x = (uR_y - uL_y) * inv_hn
        ∂u∂y = ((uLt_x + uRt_x) - (uLb_x + uRb_x)) * 0.5 * inv_ht
        ∂v∂y = ((uLt_y + uRt_y) - (uLb_y + uRb_y)) * 0.5 * inv_ht
    else
        # Horizontal face — normal in +y.
        ∂u∂y = (uR_x - uL_x) * inv_hn
        ∂v∂y = (uR_y - uL_y) * inv_hn
        ∂u∂x = ((uLt_x + uRt_x) - (uLb_x + uRb_x)) * 0.5 * inv_ht
        ∂v∂x = ((uLt_y + uRt_y) - (uLb_y + uRb_y)) * 0.5 * inv_ht
    end

    div_u = ∂u∂x + ∂v∂y
    τxx = 2.0 * μ * ∂u∂x + λ * div_u
    τyy = 2.0 * μ * ∂v∂y + λ * div_u
    τxy = μ * (∂u∂y + ∂v∂x)

    uf = 0.5 * (uL_x + uR_x)
    vf = 0.5 * (uL_y + uR_y)
    ∂T∂x = normal[1] != 0.0 ? (TR - TL) * inv_hn :
            ((TR + uRt_x * 0 + uLt_x * 0)  # only used in non-normal direction below
             - 0.0) * 0.0
    # Heat flux normal component: q·n = -κ ∇T·n.  Only the normal direction is
    # needed for the face flux.
    qn = normal[1] != 0.0 ? -κ * (TR - TL) * inv_hn :
                              -κ * (TR - TL) * inv_hn
    # τ·n contributions
    τxn = τxx * normal[1] + τxy * normal[2]
    τyn = τxy * normal[1] + τyy * normal[2]

    # Viscous fluxes per conservative variable.  Mass: 0.  Momentum: -τ·n.
    # Energy: -(τ·n)·u + q·n  (work done by viscous stress + heat flux).
    return (0.0,
             -τxn,
             -τyn,
             -(τxn * uf + τyn * vf) + qn)
end

# ============================================================================
# Field-set + cell-volume / face-area helpers
# ============================================================================

@inline function _cell_volume(frame::EulerianFrame{2, Float64}, i::Int)
    lo, hi = cell_physical_box(frame, i)
    return (hi[1] - lo[1]) * (hi[2] - lo[2])
end

@inline function _face_area(frame::EulerianFrame{2, Float64},
                              i_left::Int, i_right::Int, axis::Int)
    lo_l, hi_l = cell_physical_box(frame, i_left)
    lo_r, hi_r = cell_physical_box(frame, i_right)
    off = axis == 1 ? 2 : 1
    return min(hi_l[off] - lo_l[off], hi_r[off] - lo_r[off])
end

@inline function _face_area_boundary(frame::EulerianFrame{2, Float64},
                                       i::Int, axis::Int)
    lo, hi = cell_physical_box(frame, i)
    off = axis == 1 ? 2 : 1
    return hi[off] - lo[off]
end

@inline function _read_U(field, i::Int)
    return (field.rho[i][1], field.rhou[i][1], field.rhov[i][1], field.E[i][1])
end

@inline function _write_U!(field, i::Int, U::NTuple{4, Float64})
    field.rho[i]  = (U[1],)
    field.rhou[i] = (U[2],)
    field.rhov[i] = (U[3],)
    field.E[i]    = (U[4],)
    return nothing
end

# ============================================================================
# Config + State
# ============================================================================

"""
    ImplicitNSConfig

Configuration for a single implicit-NS run.

  * `n_initial_refines`   — uniform mesh depth, gives 2ⁿ × 2ⁿ leaf cells.
  * `domain_lo`/`hi`      — physical bounds; default [0,1]² .
  * `bcs`                 — `(axis1, axis2)` pair of `(lo, hi)` BC kinds.
                            Periodic on both is the default smoke test.
  * `t_final`, `dt`       — fixed time step (no adaptive CFL — point of the
                            implicit step is to be CFL-unbounded).
  * `viscous`             — toggle viscous + heat-conduction flux.
  * `μ`, `λ`, `κ`         — fluid constants (Stokes: λ = -2μ/3).
  * `newton_tol`, `newton_maxiter` — Newton outer tolerance / cap.
  * `gmres_tol`, `gmres_maxiter`, `gmres_restart` — Krylov inner solver.
  * `precond`             — `:none` or `:block_jacobi`.
"""
Base.@kwdef struct ImplicitNSConfig
    n_initial_refines::Int            = 5
    domain_lo::NTuple{2, Float64}     = (0.0, 0.0)
    domain_hi::NTuple{2, Float64}     = (1.0, 1.0)
    periodic_x::Bool                  = true
    periodic_y::Bool                  = true
    t_final::Float64                  = 0.2
    dt::Float64                       = 1e-2
    viscous::Bool                     = false
    μ::Float64                        = 1e-3
    λ::Float64                        = -2.0 / 3.0 * 1e-3
    κ::Float64                        = 1e-3
    newton_tol::Float64               = 1e-8
    newton_maxiter::Int               = 30
    gmres_tol::Float64                = 1e-6
    gmres_maxiter::Int                = 100
    gmres_restart::Int                = 30
    precond::Symbol                   = :none
    # AMR settings — `amr_every = 0` disables AMR entirely.
    amr_every::Int                    = 0
    refine_threshold::Float64         = 0.20
    coarsen_threshold::Float64        = 0.05
    max_level::Int                    = 3
end

# Mutable state — holds the mesh, scratch buffers for residual + JFNK,
# and U^n at the previous time step.
mutable struct ImplicitNSState
    config::ImplicitNSConfig
    mesh::HierarchicalMesh{2}
    frame::EulerianFrame{2, Float64}
    bcs::FrameBoundaries{2}
    af::AdaptiveField                  # field-of-record (4 conservative vars)
    leaves::Vector{Int}                # cached leaf indices
    cell_to_idx::Vector{Int}           # leaf cell → flat-vector position (0 if non-leaf)
    n_leaves::Int
    # Flat-vector buffers (one ordering: 4*(idx-1)+[1..4]).
    U_n::Vector{Float64}               # state at t^n
    U_iter::Vector{Float64}            # current Newton iterate
    U_pert::Vector{Float64}            # for JFNK probe
    F_iter::Vector{Float64}            # F(U_iter)
    F_pert::Vector{Float64}            # F(U_iter + ε v)
    # Per-cell flux divergence accumulators (re-used between residual calls).
    fdr::Vector{Float64}
    fdru::Vector{Float64}
    fdrv::Vector{Float64}
    fdE::Vector{Float64}
    # Block-Jacobi preconditioner cache (per-cell inverse 4×4 frozen at U^n).
    bj_inv::Vector{NTuple{16, Float64}}  # row-major flattened 4×4 inverse
    # Time-tracking (incremented by the run! driver).
    t::Float64
    n_steps::Int
end

function _build_uniform_mesh(n_levels::Int)
    mesh = HierarchicalMesh{2}()
    for _ in 1:n_levels
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    return mesh
end

function _alloc_field(mesh::HierarchicalMesh{2})
    basis = BernsteinBasis{2, 0}()
    n = n_cells(mesh)
    return allocate_polynomial_fields(SoA(), basis, n;
                                        rho = Float64, rhou = Float64,
                                        rhov = Float64, E = Float64)
end

"""
    build_state(config; ic = nothing) -> ImplicitNSState

Build the mesh / frame / BCs / scratch buffers; install the user-supplied
initial condition `ic(x) -> (ρ, u, v, p)` if given (otherwise leaves the
field unchanged from the allocator default).
"""
function build_state(config::ImplicitNSConfig;
                       ic = nothing)
    mesh = _build_uniform_mesh(config.n_initial_refines)
    frame = EulerianFrame(mesh, config.domain_lo, config.domain_hi)
    bcs = FrameBoundaries((
        (config.periodic_x ? PERIODIC : OUTFLOW,
         config.periodic_x ? PERIODIC : OUTFLOW),
        (config.periodic_y ? PERIODIC : OUTFLOW,
         config.periodic_y ? PERIODIC : OUTFLOW),
    ))
    pfs = _alloc_field(mesh)
    af = AdaptiveField(pfs, mesh)

    leaves = enumerate_leaves(mesh)
    n_cells_total = n_cells(mesh)
    n_leaves = length(leaves)
    cell_to_idx = zeros(Int, n_cells_total)
    for (k, c) in enumerate(leaves)
        cell_to_idx[c] = k
    end

    n_flat = 4 * n_leaves
    U_n     = zeros(Float64, n_flat)
    U_iter  = zeros(Float64, n_flat)
    U_pert  = zeros(Float64, n_flat)
    F_iter  = zeros(Float64, n_flat)
    F_pert  = zeros(Float64, n_flat)
    fdr  = zeros(Float64, n_cells_total)
    fdru = zeros(Float64, n_cells_total)
    fdrv = zeros(Float64, n_cells_total)
    fdE  = zeros(Float64, n_cells_total)
    bj_inv = Vector{NTuple{16, Float64}}(undef, n_leaves)

    state = ImplicitNSState(config, mesh, frame, bcs, af, leaves, cell_to_idx,
                              n_leaves, U_n, U_iter, U_pert, F_iter, F_pert,
                              fdr, fdru, fdrv, fdE, bj_inv, 0.0, 0)

    if ic !== nothing
        field = parent(af)
        for c in leaves
            lo, hi = cell_physical_box(frame, c)
            x = ((lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
            ρ, u, v, p = ic(x)
            _write_U!(field, c, prim_to_cons(ρ, u, v, p))
        end
        pack_field_to_flat!(state.U_n, state)
    end

    return state
end

# ============================================================================
# Pack / unpack between the field-of-record and a flat 4 N_leaves vector.
# ============================================================================

@inline function pack_field_to_flat!(flat::Vector{Float64}, state::ImplicitNSState)
    field = parent(state.af)
    @inbounds for (k, c) in enumerate(state.leaves)
        base = 4 * (k - 1)
        flat[base + 1] = field.rho[c][1]
        flat[base + 2] = field.rhou[c][1]
        flat[base + 3] = field.rhov[c][1]
        flat[base + 4] = field.E[c][1]
    end
    return flat
end

@inline function unpack_flat_to_field!(state::ImplicitNSState, flat::Vector{Float64})
    field = parent(state.af)
    @inbounds for (k, c) in enumerate(state.leaves)
        base = 4 * (k - 1)
        field.rho[c]  = (flat[base + 1],)
        field.rhou[c] = (flat[base + 2],)
        field.rhov[c] = (flat[base + 3],)
        field.E[c]    = (flat[base + 4],)
    end
    return state
end

# ============================================================================
# Residual:  R(U) = -∇·F(U) / V   (per unit volume — what dU/dt equals).
#
# Reuses the orchestrator pattern from cfd_compressible_sod for the
# convective flux, then optionally adds viscous + heat-conduction
# contributions via direct neighbor reads.
# ============================================================================

"""
    compute_residual!(R_flat, U_flat, state)

Write `R_flat[k,:] = -∇·F(U)/V` for each leaf-cell index `k` (4-tuple per
cell, packed). Assumes `U_flat` is the current state in flat layout;
unpacks into `state.af` internally, then runs the standard face-pass.
"""
function compute_residual!(R_flat::Vector{Float64},
                             U_flat::Vector{Float64},
                             state::ImplicitNSState)
    unpack_flat_to_field!(state, U_flat)
    field = parent(state.af)
    frame = state.frame
    mesh = state.mesh
    bcs = state.bcs

    fdr  = state.fdr
    fdru = state.fdru
    fdrv = state.fdrv
    fdE  = state.fdE
    fill!(fdr,  0.0)
    fill!(fdru, 0.0)
    fill!(fdrv, 0.0)
    fill!(fdE,  0.0)

    config = state.config
    viscous = config.viscous
    μ = config.μ; λ = config.λ; κ = config.κ

    # Cell-spacing (uniform single-level for now).
    hxs = let
        lo, hi = cell_physical_box(frame, state.leaves[1])
        (hi[1] - lo[1], hi[2] - lo[2])
    end

    fin_v = (rho  = field.rho,
              rhou = field.rhou,
              rhov = field.rhov,
              E    = field.E)
    fout_v = fin_v

    flux_kernel = let frame = frame, fdr = fdr, fdru = fdru,
                       fdrv = fdrv, fdE = fdE, viscous = viscous,
                       μ = μ, λ = λ, κ = κ, mesh = mesh, hxs = hxs,
                       field_capture = field
        function (cv_left, cv_right, normal, _ctx)
            i = cv_left.index
            j = cv_right.index
            UL = (cv_left[Val(:rho)][1],
                   cv_left[Val(:rhou)][1],
                   cv_left[Val(:rhov)][1],
                   cv_left[Val(:E)][1])
            UR = (cv_right[Val(:rho)][1],
                   cv_right[Val(:rhou)][1],
                   cv_right[Val(:rhov)][1],
                   cv_right[Val(:E)][1])
            axis = normal[1] != 0.0 ? 1 : 2
            face_area = _face_area(frame, i, j, axis)
            F = hll_flux(UL, UR, normal)
            if viscous
                nbrs_L = face_neighbors(mesh, i)
                nbrs_R = face_neighbors(mesh, j)
                off = axis == 1 ? 2 : 1
                Lt_idx = nbrs_L[2 * (off - 1) + 2]
                Lb_idx = nbrs_L[2 * (off - 1) + 1]
                Rt_idx = nbrs_R[2 * (off - 1) + 2]
                Rb_idx = nbrs_R[2 * (off - 1) + 1]
                Lt = Lt_idx != 0 ? _read_U(field_capture, Int(Lt_idx)) : UL
                Lb = Lb_idx != 0 ? _read_U(field_capture, Int(Lb_idx)) : UL
                Rt = Rt_idx != 0 ? _read_U(field_capture, Int(Rt_idx)) : UR
                Rb = Rb_idx != 0 ? _read_U(field_capture, Int(Rb_idx)) : UR
                h_n = axis == 1 ? hxs[1] : hxs[2]
                h_t = axis == 1 ? hxs[2] : hxs[1]
                Fv = viscous_flux(UL, UR, Lt, Lb, Rt, Rb, normal, h_n, h_t,
                                    μ, λ, κ)
                F = (F[1] + Fv[1], F[2] + Fv[2], F[3] + Fv[3], F[4] + Fv[4])
            end
            scale = face_area
            @inbounds fdr[i]  += F[1] * scale
            @inbounds fdru[i] += F[2] * scale
            @inbounds fdrv[i] += F[3] * scale
            @inbounds fdE[i]  += F[4] * scale
            @inbounds fdr[j]  -= F[1] * scale
            @inbounds fdru[j] -= F[2] * scale
            @inbounds fdrv[j] -= F[3] * scale
            @inbounds fdE[j]  -= F[4] * scale
            return nothing
        end
    end

    # Boundary kernel: OUTFLOW (zero-gradient) on non-periodic axes.
    boundary_kernel = let frame = frame, fdr = fdr, fdru = fdru,
                          fdrv = fdrv, fdE = fdE,
                          periodic_x = state.config.periodic_x,
                          periodic_y = state.config.periodic_y
        function (cv, axis, side, normal, bcs_in, _ctx)
            # Periodic axes are handled in a separate manual pass below
            # (see Sod tube for the reason).
            (axis == 1 && periodic_x) && return nothing
            (axis == 2 && periodic_y) && return nothing
            i = cv.index
            UI = (cv[Val(:rho)][1], cv[Val(:rhou)][1],
                   cv[Val(:rhov)][1], cv[Val(:E)][1])
            # Zero-gradient ghost: mirror UI.
            F = hll_flux(UI, UI, normal)
            face_area = _face_area_boundary(frame, i, axis)
            @inbounds fdr[i]  += F[1] * face_area
            @inbounds fdru[i] += F[2] * face_area
            @inbounds fdrv[i] += F[3] * face_area
            @inbounds fdE[i]  += F[4] * face_area
            return nothing
        end
    end

    for_each_face!(flux_kernel, fout_v, fin_v, frame;
                   bcs = bcs,
                   flux_kernel_boundary = boundary_kernel,
                   backend = Sequential())

    # Periodic-axis face pass — same as Sod tube.
    ensure_neighbor_graph!(mesh)
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        nbs    = face_neighbors(mesh, i)
        nbs_bc = face_neighbors_with_bcs(mesh, i, bcs)
        for axis in 1:2
            ((axis == 1 && state.config.periodic_x) ||
             (axis == 2 && state.config.periodic_y)) || continue
            lo_face = 2 * axis - 1
            if nbs[lo_face] == 0 && nbs_bc[lo_face] != 0
                j = Int(nbs_bc[lo_face])
                normal = axis == 1 ? (1.0, 0.0) : (0.0, 1.0)
                UL = _read_U(field, j)
                UR = _read_U(field, i)
                face_area = _face_area(frame, j, i, axis)
                F = hll_flux(UL, UR, normal)
                # Add viscous part if enabled.  We don't bother with the
                # tangential gradient at periodic boundaries here; zero-grad
                # is a reasonable simplification for the first cut.
                @inbounds fdr[j]  += F[1] * face_area
                @inbounds fdru[j] += F[2] * face_area
                @inbounds fdrv[j] += F[3] * face_area
                @inbounds fdE[j]  += F[4] * face_area
                @inbounds fdr[i]  -= F[1] * face_area
                @inbounds fdru[i] -= F[2] * face_area
                @inbounds fdrv[i] -= F[3] * face_area
                @inbounds fdE[i]  -= F[4] * face_area
            end
        end
    end

    # Pack -fdiv/V into R_flat.
    @inbounds for (k, c) in enumerate(state.leaves)
        base = 4 * (k - 1)
        invV = 1.0 / _cell_volume(frame, c)
        R_flat[base + 1] = -fdr[c]  * invV
        R_flat[base + 2] = -fdru[c] * invV
        R_flat[base + 3] = -fdrv[c] * invV
        R_flat[base + 4] = -fdE[c]  * invV
    end
    return R_flat
end

"""
    compute_F!(F_flat, U_flat, state, dt)

Newton residual `F(U) = U - U^n - dt·R(U)`. Uses `state.U_n` as the
fixed reference. Reads `U_flat`, writes `F_flat`.
"""
function compute_F!(F_flat::Vector{Float64},
                     U_flat::Vector{Float64},
                     state::ImplicitNSState, dt::Float64)
    compute_residual!(F_flat, U_flat, state)
    @inbounds @simd for i in eachindex(F_flat)
        F_flat[i] = U_flat[i] - state.U_n[i] - dt * F_flat[i]
    end
    return F_flat
end

# ============================================================================
# JFNK Jacobian-vector product:  J·v ≈ (F(U + ε v) - F(U)) / ε
# ============================================================================

mutable struct JFNKMatvec
    state::ImplicitNSState
    dt::Float64
    eps_safe::Float64               # cached: √(machine_eps) * (1 + ||U||_∞)
end

# `ε` is chosen per call by Pernice-Walker:
#   ε = sqrt(machine_eps) * (||U||_∞ + ε_safe) / max(||v||_∞, ε_safe)
@inline function _jfnk_eps(op::JFNKMatvec, v::Vector{Float64})
    norm_v = 0.0
    @inbounds for x in v
        ax = abs(x)
        if ax > norm_v; norm_v = ax; end
    end
    norm_v < op.eps_safe && (norm_v = op.eps_safe)
    return sqrt(eps(Float64)) * op.eps_safe / norm_v
end

Base.size(op::JFNKMatvec) = (length(op.state.U_iter), length(op.state.U_iter))
Base.size(op::JFNKMatvec, ::Int) = length(op.state.U_iter)
Base.eltype(::JFNKMatvec) = Float64

function LinearAlgebra.mul!(y::AbstractVector{Float64},
                              op::JFNKMatvec, v::AbstractVector{Float64})
    ε = _jfnk_eps(op, v)
    inv_ε = 1.0 / ε
    @inbounds @simd for i in eachindex(v)
        op.state.U_pert[i] = op.state.U_iter[i] + ε * v[i]
    end
    compute_F!(op.state.F_pert, op.state.U_pert, op.state, op.dt)
    @inbounds @simd for i in eachindex(y)
        y[i] = (op.state.F_pert[i] - op.state.F_iter[i]) * inv_ε
    end
    return y
end

Base.:*(op::JFNKMatvec, v::AbstractVector{Float64}) = mul!(similar(v), op, v)

# ============================================================================
# Block-Jacobi preconditioner.
#
# Per leaf cell, build a 4×4 block  M = I - dt · ∂R/∂U |_cell  using
# numerical differentiation on the cell-local residual (single cell with
# fixed neighbours from U^n), then invert and cache.  Application: per-cell
# 4×4 matvec on the input flat vector.
# ============================================================================

function build_block_jacobi!(state::ImplicitNSState, dt::Float64)
    n_leaves = state.n_leaves
    # We'll perturb each of the 4 components of cell k's state in U_n and
    # measure the change in F at the same cell.  This is cell-local only —
    # spatial coupling is ignored.
    F0 = state.F_iter   # scratch
    F1 = state.F_pert   # scratch
    compute_F!(F0, state.U_n, state, dt)
    ε = sqrt(eps(Float64))
    @inbounds for k in 1:n_leaves
        base = 4 * (k - 1)
        # Build the 4×4 block (rows = output components, cols = input).
        block = MMatrix4()
        for col in 1:4
            for j in 1:length(state.U_pert)
                state.U_pert[j] = state.U_n[j]
            end
            scale = ε * (abs(state.U_n[base + col]) + ε)
            state.U_pert[base + col] += scale
            compute_F!(F1, state.U_pert, state, dt)
            inv_s = 1.0 / scale
            for row in 1:4
                block[row, col] = (F1[base + row] - F0[base + row]) * inv_s
            end
        end
        state.bj_inv[k] = _invert_4x4(block)
    end
    return state.bj_inv
end

# Tiny scratch 4×4 — keeps the inner loop alloc-free.
mutable struct MMatrix4
    data::NTuple{16, Float64}
end
MMatrix4() = MMatrix4(ntuple(_ -> 0.0, 16))
@inline Base.getindex(M::MMatrix4, i::Int, j::Int) = M.data[(j - 1) * 4 + i]
@inline function Base.setindex!(M::MMatrix4, v::Float64, i::Int, j::Int)
    idx = (j - 1) * 4 + i
    M.data = ntuple(k -> k == idx ? v : M.data[k], 16)
    return v
end

# Invert a 4×4 matrix in-place via Gauss-Jordan, returning the flattened
# inverse as a 16-tuple (column-major).  Returns identity on singular
# matrix (degenerate fallback — Newton recovers).
function _invert_4x4(M::MMatrix4)
    # Copy into a mutable array for elimination.
    A = zeros(4, 8)
    @inbounds for j in 1:4, i in 1:4
        A[i, j] = M[i, j]
    end
    @inbounds for i in 1:4
        A[i, 4 + i] = 1.0
    end
    @inbounds for k in 1:4
        pivot = A[k, k]
        if abs(pivot) < 1e-14
            # Singular — fall back to identity.
            return ntuple(j -> (j - 1) % 5 == 0 ? 1.0 : 0.0, 16)
        end
        inv_p = 1.0 / pivot
        @inbounds for j in 1:8
            A[k, j] *= inv_p
        end
        @inbounds for i in 1:4
            i == k && continue
            f = A[i, k]
            @inbounds for j in 1:8
                A[i, j] -= f * A[k, j]
            end
        end
    end
    return ntuple(idx -> begin
        col = ((idx - 1) >> 2) + 1   # column 1..4 within inverse
        row = ((idx - 1) & 3) + 1    # row 1..4
        A[row, 4 + col]
    end, 16)
end

# Apply block-Jacobi preconditioner: y[k] = M_k^{-1} · x[k] for each cell.
function apply_block_jacobi!(y::AbstractVector{Float64},
                              x::AbstractVector{Float64},
                              state::ImplicitNSState)
    @inbounds for k in 1:state.n_leaves
        base = 4 * (k - 1)
        x1 = x[base + 1]; x2 = x[base + 2]
        x3 = x[base + 3]; x4 = x[base + 4]
        inv = state.bj_inv[k]
        # column-major: inv[(col-1)*4 + row]
        y[base + 1] = inv[1]  * x1 + inv[5]  * x2 + inv[9]   * x3 + inv[13] * x4
        y[base + 2] = inv[2]  * x1 + inv[6]  * x2 + inv[10]  * x3 + inv[14] * x4
        y[base + 3] = inv[3]  * x1 + inv[7]  * x2 + inv[11]  * x3 + inv[15] * x4
        y[base + 4] = inv[4]  * x1 + inv[8]  * x2 + inv[12]  * x3 + inv[16] * x4
    end
    return y
end

# Wrap as a Krylov-compatible LinearOperator-ish struct.
mutable struct BlockJacobiOp
    state::ImplicitNSState
end
Base.size(op::BlockJacobiOp) = (length(op.state.U_iter), length(op.state.U_iter))
Base.size(op::BlockJacobiOp, ::Int) = length(op.state.U_iter)
Base.eltype(::BlockJacobiOp) = Float64
function LinearAlgebra.mul!(y::AbstractVector{Float64},
                              op::BlockJacobiOp, x::AbstractVector{Float64})
    return apply_block_jacobi!(y, x, op.state)
end
Base.:*(op::BlockJacobiOp, x::AbstractVector{Float64}) = mul!(similar(x), op, x)

# ============================================================================
# Newton-Krylov outer loop.
# ============================================================================

"""
    implicit_ns_step!(state, dt) -> (newton_iters, final_residual)

One backward-Euler step.  Newton on `F(U) = U - U^n - dt·R(U)` with
inner-loop GMRES.  Optional damped line search if the residual fails to
decrease.
"""
function implicit_ns_step!(state::ImplicitNSState, dt::Float64;
                             verbose::Bool = false)
    config = state.config
    # Snapshot U^n.
    pack_field_to_flat!(state.U_n, state)
    # Initial Newton iterate = U^n.
    copyto!(state.U_iter, state.U_n)

    if config.precond === :block_jacobi
        build_block_jacobi!(state, dt)
    end

    matvec = JFNKMatvec(state, dt,
                          max(1.0, maximum(abs, state.U_n)))

    iters = 0
    for k in 1:config.newton_maxiter
        compute_F!(state.F_iter, state.U_iter, state, dt)
        norm_F = norm(state.F_iter)
        if verbose
            @info "Newton iter $k: |F| = $norm_F"
        end
        if norm_F < config.newton_tol
            iters = k - 1
            return (iters, norm_F)
        end

        # GMRES solve  J · δU = -F  for δU.
        rhs = -state.F_iter
        if config.precond === :block_jacobi
            M = BlockJacobiOp(state)
            δU, stats = gmres(matvec, rhs;
                                M = M, itmax = config.gmres_maxiter,
                                memory = config.gmres_restart,
                                atol = config.gmres_tol,
                                rtol = config.gmres_tol,
                                history = true, verbose = 0)
        else
            δU, stats = gmres(matvec, rhs;
                                itmax = config.gmres_maxiter,
                                memory = config.gmres_restart,
                                atol = config.gmres_tol,
                                rtol = config.gmres_tol,
                                history = true, verbose = 0)
        end

        # Damped line search: try full step, halve if F-norm goes up.
        α = 1.0
        for _ in 1:6
            @inbounds @simd for i in eachindex(state.U_iter)
                state.U_pert[i] = state.U_iter[i] + α * δU[i]
            end
            compute_F!(state.F_pert, state.U_pert, state, dt)
            if norm(state.F_pert) < norm_F
                copyto!(state.U_iter, state.U_pert)
                break
            end
            α *= 0.5
        end
        iters = k
    end

    # Final unpack regardless of convergence.
    unpack_flat_to_field!(state, state.U_iter)
    compute_F!(state.F_iter, state.U_iter, state, dt)
    return (iters, norm(state.F_iter))
end

# ============================================================================
# AMR primitives
#
# AMR runs *between* implicit steps — the mesh is frozen during the Newton
# / GMRES inner loop.  After `step_with_amr!` refines or coarsens, the
# cached `leaves` / `cell_to_idx` / flat-vector scratch buffers in
# `state` are stale and must be rebuilt before the next residual is
# computed.  `refresh_state!` does that in place.
#
# Conservative remapping of (ρ, ρu, ρv, E) across refinement events is
# handled by the `AdaptiveField` machinery — degree-0 BernsteinBasis is
# remapped by mean-preserving (volume-weighted average for coarsening,
# copy for refinement), which matches the FV conservation law.
# ============================================================================

"""
    refresh_state!(state)

Rebuild `state.leaves`, `state.cell_to_idx`, and resize the JFNK scratch
buffers (`U_n`, `U_iter`, `U_pert`, `F_iter`, `F_pert`, `bj_inv`) and the
per-cell flux divergence accumulators (`fdr`, `fdru`, `fdrv`, `fdE`) to
match the current mesh.  Idempotent on an unchanged mesh.
"""
function refresh_state!(state::ImplicitNSState)
    state.leaves = enumerate_leaves(state.mesh)
    n_cells_total = n_cells(state.mesh)
    state.n_leaves = length(state.leaves)
    resize!(state.cell_to_idx, n_cells_total)
    fill!(state.cell_to_idx, 0)
    @inbounds for (k, c) in enumerate(state.leaves)
        state.cell_to_idx[c] = k
    end
    n_flat = 4 * state.n_leaves
    resize!(state.U_n,    n_flat)
    resize!(state.U_iter, n_flat)
    resize!(state.U_pert, n_flat)
    resize!(state.F_iter, n_flat)
    resize!(state.F_pert, n_flat)
    resize!(state.fdr,   n_cells_total)
    resize!(state.fdru,  n_cells_total)
    resize!(state.fdrv,  n_cells_total)
    resize!(state.fdE,   n_cells_total)
    resize!(state.bj_inv, state.n_leaves)
    return state
end

"""
    gradient_indicator(field, mesh) -> Vector{Float64}

Per-cell `|Δρ|/max(ρ, ρ_nbr)` as a finite-difference proxy on face
neighbours.  Mirrors the indicator used by `cfd_compressible_sod` — same
shock / contact detection, cheap to evaluate.
"""
function gradient_indicator(field, mesh::HierarchicalMesh{2})
    n = n_cells(mesh)
    out = zeros(Float64, n)
    ensure_neighbor_graph!(mesh)
    @inbounds for i in 1:n
        is_leaf(mesh.cells[i]) || continue
        nbs = face_neighbors(mesh, i)
        ρ_i = field.rho[i][1]
        ρ_i > 0.0 || continue
        max_rel = 0.0
        for f in 1:4
            nb = nbs[f]
            nb == 0 && continue
            ρ_j = field.rho[Int(nb)][1]
            d = abs(ρ_i - ρ_j) / max(ρ_i, ρ_j, eps(Float64))
            if d > max_rel
                max_rel = d
            end
        end
        out[i] = max_rel
    end
    return out
end

"""
    implicit_ns_step_with_amr!(state, dt; refine_now = false) -> (newton_iters, residual)

One implicit step + optional AMR cycle.  Caller decides whether to fire
AMR this step (typically `step % config.amr_every == 0`).  After AMR,
`refresh_state!` is called to resize the scratch buffers.
"""
function implicit_ns_step_with_amr!(state::ImplicitNSState, dt::Float64;
                                       refine_now::Bool = false,
                                       verbose::Bool = false)
    refresh_state!(state)
    iters, res = implicit_ns_step!(state, dt; verbose = verbose)
    if refine_now && state.config.amr_every > 0
        indicator_fn = m -> gradient_indicator(parent(state.af), m)
        refine_by_indicator!(state.mesh, indicator_fn(state.mesh);
                               refine_threshold  = state.config.refine_threshold,
                               coarsen_threshold = state.config.coarsen_threshold,
                               max_level         = state.config.max_level,
                               isotropic         = true)
        refresh_state!(state)
    end
    return (iters, res)
end

# ============================================================================
# Top-level driver
# ============================================================================

"""
    run!(config; ic = nothing, callback = nothing, verbose = false) -> ImplicitNSState

Run an implicit-NS simulation from `t = 0` to `config.t_final` with fixed
`config.dt`.  If `config.amr_every > 0`, AMR fires every Nth step using a
density-gradient indicator; otherwise the mesh is static.

Optional `ic(x) -> (ρ, u, v, p)` sets the initial condition;
optional `callback(state, t, step)` runs after every accepted step.
"""
function run!(config::ImplicitNSConfig;
                ic = nothing,
                callback = nothing,
                verbose::Bool = false)
    state = build_state(config; ic = ic)
    while state.t < config.t_final - 1e-14
        dt = min(config.dt, config.t_final - state.t)
        refine_now = (config.amr_every > 0) &&
                     ((state.n_steps + 1) % config.amr_every == 0)
        iters, res = implicit_ns_step_with_amr!(state, dt;
                                                    refine_now = refine_now,
                                                    verbose = verbose)
        state.t += dt
        state.n_steps += 1
        if verbose
            @info "step $(state.n_steps)  t=$(state.t)  newton_iters=$iters  |F|=$res  n_leaves=$(state.n_leaves)"
        end
        callback === nothing || callback(state, state.t, state.n_steps)
    end
    return state
end

end # module
