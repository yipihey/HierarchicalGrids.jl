"""
    CFDIncompressibleNSCell

Cell-based AMR scaffolding for a 2D incompressible Navier-Stokes solver,
mirroring the `cfd_implicit_ns` architecture (HierarchicalMesh +
AdaptiveField + for_each_face!). Step 1 of the cell-mode punch list:
**scaffold + 2nd-order C/F-aware advection of a vector field**.

What this provides:

* `HierarchicalMesh{2}` + optional centered-block refinement.
* `AdaptiveField` holding cell-centered `(:u, :v)`.
* `for_each_face!` flux kernel that calls the same Martin-Colella
  transverse correction shipped in `cfd_amr_advection_2o`, but applied
  per-component (once for u, once for v).
* RK2 (Heun) time integration; mass/momentum exactly conserved across
  same-level and C/F faces.
* Passive vector advection by a frozen carrier velocity (the carrier
  becomes the field's own velocity in later steps once viscous +
  projection are wired in).

Verification: Taylor-Green-shaped IC advected by a constant carrier;
analytic solution is `u(x, t) = u₀(x − u_carrier · t)`. Convergence
sweep gives ≥ 2nd-order in space on both uniform and AMR-refined meshes.

What's NOT in this step (deferred to steps 2–4):

* Implicit viscous step (JFNK + GMRES + Helmholtz precond).
* Cell-native multigrid Poisson for projection.
* MAC-on-averaged-faces or nodal projection.
* AMR re-refinement during a run (`step_with_amr!`) — only static AMR.
"""
module CFDIncompressibleNSCell

using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells, is_leaf,
                          level_of, enumerate_leaves, cell_physical_box,
                          EulerianFrame, FrameBoundaries, BCKind, PERIODIC,
                          face_neighbors, face_neighbors_with_bcs,
                          ensure_neighbor_graph!,
                          BernsteinBasis, allocate_polynomial_fields, SoA,
                          AdaptiveField, dispose!,
                          for_each_face!, Sequential
import LinearAlgebra: mul!
using Krylov: cg

export NSCellConfig, NSCellState
export build_state, step!, run!, viscous_step!
export taylor_green_ic, taylor_green_solution
export l2_error, total_momentum, kinetic_energy

# ============================================================================
# Configuration
# ============================================================================

"""
    NSCellConfig

# Mesh
- `n_base_refines::Int` — `2^n_base_refines` cells per axis on the base grid.
- `refine_center::Bool` — refine cells in the centered box once more.
- `center_box_lo`, `center_box_hi` — extent (physical, fraction of the
  unit square by default) of the refined block.

# Physics
- `carrier_velocity::NTuple{2, Float64}` — constant background velocity
  that advects the (u, v) field. In step 1 this is *not* the field
  itself; once projection + viscous are wired in (steps 2–4), the
  carrier becomes the field's own velocity (self-advection).
- `domain_lo`, `domain_hi` — physical domain extent.

# Time
- `t_final::Float64`, `dt::Float64`.
- `time_scheme::Symbol` — `:euler` (1st order) or `:rk2` (2nd-order Heun).
"""
Base.@kwdef struct NSCellConfig
    n_base_refines::Int = 5
    refine_center::Bool = false
    center_box_lo::NTuple{2, Float64} = (0.25, 0.25)
    center_box_hi::NTuple{2, Float64} = (0.75, 0.75)

    carrier_velocity::NTuple{2, Float64} = (1.0, 0.5)
    domain_lo::NTuple{2, Float64} = (0.0, 0.0)
    domain_hi::NTuple{2, Float64} = (1.0, 1.0)

    t_final::Float64 = 0.05
    dt::Float64 = 1e-3
    time_scheme::Symbol = :rk2

    # Step 2: implicit viscous step.
    μ::Float64 = 1e-2                 # kinematic viscosity ν
    helmholtz_tol::Float64 = 1e-10
    helmholtz_maxiter::Int = 200
end

# ============================================================================
# State
# ============================================================================

mutable struct NSCellState
    config::NSCellConfig
    mesh::HierarchicalMesh{2}
    frame::EulerianFrame{2, Float64}
    bcs::FrameBoundaries{2}
    af::AdaptiveField               # field-of-record carrying (:u, :v)
    leaves::Vector{Int}
    t::Float64
    n_steps::Int
end

# ============================================================================
# Analytic initial / reference solutions
# ============================================================================

"""
    taylor_green_ic(x; k = (2π, 2π), U0 = 1.0) -> (u, v)

Standard Taylor-Green vortex profile on `[0, 1]²`. Divergence-free —
useful both as IC for the eventual full incompressible solver and as a
smooth periodic test profile here.
"""
function taylor_green_ic(x::NTuple{2, Float64};
                          k::NTuple{2, Float64} = (2π, 2π),
                          U0::Float64 = 1.0)
    u =  U0 * sin(k[1] * x[1]) * cos(k[2] * x[2])
    v = -U0 * cos(k[1] * x[1]) * sin(k[2] * x[2])
    return (u, v)
end

"""
    taylor_green_solution(x, t, carrier; k = (2π, 2π), U0 = 1.0) -> (u, v)

Analytic solution under pure advection by constant `carrier`:
`u(x, t) = u₀(x − carrier · t)`. Periodic wrap is enforced through the
sin/cos arguments.
"""
function taylor_green_solution(x::NTuple{2, Float64}, t::Float64,
                                carrier::NTuple{2, Float64};
                                k::NTuple{2, Float64} = (2π, 2π),
                                U0::Float64 = 1.0)
    xs = (x[1] - carrier[1] * t, x[2] - carrier[2] * t)
    return taylor_green_ic(xs; k = k, U0 = U0)
end

# ============================================================================
# Mesh construction + IC
# ============================================================================

function _build_uniform_mesh(n::Int)
    mesh = HierarchicalMesh{2}()
    for _ in 1:n
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    return mesh
end

function _alloc_field(mesh::HierarchicalMesh{2})
    basis = BernsteinBasis{2, 0}()
    n = n_cells(mesh)
    pfs = allocate_polynomial_fields(SoA(), basis, n;
                                       u = Float64, v = Float64)
    @inbounds for i in 1:n
        pfs.u[i] = (0.0,)
        pfs.v[i] = (0.0,)
    end
    return pfs
end

"""
    build_state(cfg; ic = taylor_green_ic) -> NSCellState

Build mesh (with optional centered refinement), allocate `(:u, :v)`,
apply IC by cell-center evaluation. The `ic` callable must return
`(u, v)` at a given `(x, y)`.
"""
function build_state(cfg::NSCellConfig; ic = taylor_green_ic)
    cfg.time_scheme in (:euler, :rk2) ||
        error("time_scheme must be :euler or :rk2")

    mesh = _build_uniform_mesh(cfg.n_base_refines)
    frame = EulerianFrame(mesh, cfg.domain_lo, cfg.domain_hi)
    bcs = FrameBoundaries(((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)))

    if cfg.refine_center
        leaves = enumerate_leaves(mesh)
        to_refine = Int[]
        @inbounds for c in leaves
            lo, hi = cell_physical_box(frame, c)
            x = (0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
            if cfg.center_box_lo[1] <= x[1] <= cfg.center_box_hi[1] &&
               cfg.center_box_lo[2] <= x[2] <= cfg.center_box_hi[2]
                push!(to_refine, c)
            end
        end
        isempty(to_refine) || refine_cells!(mesh, to_refine)
    end

    pfs = _alloc_field(mesh)
    af = AdaptiveField(pfs, mesh)

    leaves = enumerate_leaves(mesh)
    field = parent(af)
    @inbounds for c in leaves
        lo, hi = cell_physical_box(frame, c)
        x = (0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
        u, v = ic(x)
        field.u[c] = (u,)
        field.v[c] = (v,)
    end

    return NSCellState(cfg, mesh, frame, bcs, af, leaves, 0.0, 0)
end

# ============================================================================
# Geometry helpers
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

# ============================================================================
# Martin-Colella 2nd-order C/F face reconstruction (same formula as
# cfd_amr_advection_2o, applied per scalar component)
#
# Closed form for a 3-point linear fit through ρ_C (coarse), ρ_F (fine),
# ρ_T (coarse transverse neighbor in the direction of the fine cell):
#
#   ρ_face = (2/3)·ρ_F + (1/3)·ρ_C + (1/3)·slope_t·t_offset
#
# Exact for linear fields; recovers 2nd-order at sub-face centers.
# ============================================================================

@inline function _nbr_index(nbrs, axis::Int, side::Int)
    return nbrs[2 * (axis - 1) + (side > 0 ? 2 : 1)]
end

function _cf_face_value(mesh::HierarchicalMesh{2},
                         frame::EulerianFrame{2, Float64},
                         field, field_name::Symbol,
                         bcs::FrameBoundaries{2},
                         i_coarse::Int, i_fine::Int,
                         rho_coarse::Float64, rho_fine::Float64,
                         axis::Int)
    transverse = axis == 1 ? 2 : 1

    lo_c, hi_c = cell_physical_box(frame, i_coarse)
    lo_f, hi_f = cell_physical_box(frame, i_fine)
    coarse_center_t = 0.5 * (lo_c[transverse] + hi_c[transverse])
    fine_center_t   = 0.5 * (lo_f[transverse] + hi_f[transverse])
    t_offset = fine_center_t - coarse_center_t

    if abs(t_offset) < 1e-12 * (hi_c[transverse] - lo_c[transverse])
        return (2 * rho_fine + rho_coarse) / 3
    end

    nbrs = face_neighbors_with_bcs(mesh, i_coarse, bcs)
    t_side = t_offset > 0 ? +1 : -1
    nbr_idx = _nbr_index(nbrs, transverse, t_side)

    if nbr_idx == 0
        return (2 * rho_fine + rho_coarse) / 3
    end

    nbr_idx_i = Int(nbr_idx)
    lo_n, hi_n = cell_physical_box(frame, nbr_idx_i)
    nbr_center_t = 0.5 * (lo_n[transverse] + hi_n[transverse])
    domain_extent = frame.hi[transverse] - frame.lo[transverse]
    raw_dist = nbr_center_t - coarse_center_t
    if abs(raw_dist) > domain_extent / 2
        raw_dist -= sign(raw_dist) * domain_extent
    end
    arr = getproperty(field, field_name)
    rho_nbr = arr[nbr_idx_i][1]
    slope_t = (rho_nbr - rho_coarse) / raw_dist

    return (2 * rho_fine + rho_coarse + slope_t * t_offset) / 3
end

# ============================================================================
# Vector flux divergence: passive advection of (u, v) by a frozen carrier.
# Same kernel structure as cfd_amr_advection_2o, applied per component.
# ============================================================================

# Compute per-cell flux divergence buffers fdu, fdv = ∑_faces F · A,
# i.e. the negative of d(u, v)/dt before dividing by V.
function _accumulate_flux_divergence!(fdu::Vector{Float64},
                                       fdv::Vector{Float64},
                                       state::NSCellState,
                                       field)
    mesh = state.mesh
    frame = state.frame
    bcs   = state.bcs
    carrier = state.config.carrier_velocity

    @inbounds for i in 1:length(fdu); fdu[i] = 0.0; fdv[i] = 0.0; end

    flux_kernel = let frame = frame, mesh = mesh, bcs = bcs,
                       carrier = carrier, fdu = fdu, fdv = fdv,
                       field = field
        function (cv_L, cv_R, normal, _ctx)
            i = cv_L.index
            j = cv_R.index
            u_L = cv_L[Val(:u)][1]; v_L = cv_L[Val(:v)][1]
            u_R = cv_R[Val(:u)][1]; v_R = cv_R[Val(:v)][1]
            lvl_L = cv_L.level
            lvl_R = cv_R.level
            axis = normal[1] != 0.0 ? 1 : 2
            u_n  = carrier[1] * normal[1] + carrier[2] * normal[2]

            if lvl_L == lvl_R
                u_face = 0.5 * (u_L + u_R)
                v_face = 0.5 * (v_L + v_R)
            elseif lvl_L < lvl_R          # left coarse, right fine
                u_face = _cf_face_value(mesh, frame, field, :u, bcs,
                                          i, j, u_L, u_R, axis)
                v_face = _cf_face_value(mesh, frame, field, :v, bcs,
                                          i, j, v_L, v_R, axis)
            else                            # left fine, right coarse
                u_face = _cf_face_value(mesh, frame, field, :u, bcs,
                                          j, i, u_R, u_L, axis)
                v_face = _cf_face_value(mesh, frame, field, :v, bcs,
                                          j, i, v_R, v_L, axis)
            end

            face_area = _face_area(frame, i, j, axis)
            Fu = u_n * u_face * face_area
            Fv = u_n * v_face * face_area
            fdu[i] += Fu; fdu[j] -= Fu
            fdv[i] += Fv; fdv[j] -= Fv
            return nothing
        end
    end

    fin_v  = (u = field.u, v = field.v)
    fout_v = fin_v
    for_each_face!(flux_kernel, fout_v, fin_v, frame;
                   bcs = bcs,
                   backend = Sequential())

    # Periodic-axis face pass (orchestrator skips periodic outer faces;
    # we re-create them here). Same pattern as cfd_implicit_ns.
    ensure_neighbor_graph!(mesh)
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        nbs    = face_neighbors(mesh, i)
        nbs_bc = face_neighbors_with_bcs(mesh, i, bcs)
        for axis in 1:2
            lo_face = 2 * axis - 1
            if nbs[lo_face] == 0 && nbs_bc[lo_face] != 0
                j = Int(nbs_bc[lo_face])
                normal = axis == 1 ? (1.0, 0.0) : (0.0, 1.0)
                u_L = field.u[j][1]; v_L = field.v[j][1]
                u_R = field.u[i][1]; v_R = field.v[i][1]
                lvl_L = Int(level_of(mesh, j))
                lvl_R = Int(level_of(mesh, i))
                if lvl_L == lvl_R
                    u_face = 0.5 * (u_L + u_R)
                    v_face = 0.5 * (v_L + v_R)
                elseif lvl_L < lvl_R
                    u_face = _cf_face_value(mesh, frame, field, :u, bcs,
                                              j, i, u_L, u_R, axis)
                    v_face = _cf_face_value(mesh, frame, field, :v, bcs,
                                              j, i, v_L, v_R, axis)
                else
                    u_face = _cf_face_value(mesh, frame, field, :u, bcs,
                                              i, j, u_R, u_L, axis)
                    v_face = _cf_face_value(mesh, frame, field, :v, bcs,
                                              i, j, v_R, v_L, axis)
                end
                u_n = carrier[1] * normal[1] + carrier[2] * normal[2]
                face_area = _face_area(frame, j, i, axis)
                Fu = u_n * u_face * face_area
                Fv = u_n * v_face * face_area
                fdu[j] += Fu; fdu[i] -= Fu
                fdv[j] += Fv; fdv[i] -= Fv
            end
        end
    end

    return (fdu, fdv)
end

# ============================================================================
# Discrete Laplacian on the AMR mesh
#
# Conservative form: (L u)_c = (1/V_c) · ∑_faces β · (u_neighbour − u_c) ·
#                                                   face_area / center_distance
# with β ≡ 1 here (variable-coefficient extension is a one-liner).
#
# To avoid double-counting on the cell sweep, we only process each leaf's
# *low* face along each axis (face_idx ∈ {1, 3}), and add the equal-and-
# opposite contribution to the neighbour. Every face is then visited
# exactly once: same-level faces by the cell on the high side, C/F faces
# by each fine cell (each of which has the C/F as its low face — coarse
# never iterates a C/F face from its high side).
#
# At hanging-node faces the gradient is approximated by the simple
# centre-to-centre formula `(u_j − u_i) / dist_ij`. Discretely this is
# 2nd-order along the centreline between cells but 1st-order at the
# sub-face centre in 2D (the same transverse-offset issue the advection
# fixes with Martin-Colella). Step 3 (cell-native multigrid Poisson)
# upgrades this to a fully 2nd-order Martin-Colella Laplacian; for the
# step-2 implicit viscous solve the 1st-order C/F treatment is
# acceptable (and the AMR test below just checks the solve runs and
# the field decays — the 2nd-order verification is on uniform meshes).
# ============================================================================

function _apply_laplacian!(Lv::Vector{T}, v::Vector{T},
                            state::NSCellState) where {T}
    mesh = state.mesh
    frame = state.frame
    bcs = state.bcs
    @inbounds for c in state.leaves; Lv[c] = zero(T); end

    ensure_neighbor_graph!(mesh)
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        nbrs = face_neighbors_with_bcs(mesh, i, bcs)
        lo_i, hi_i = cell_physical_box(frame, i)
        V_i = (hi_i[1] - lo_i[1]) * (hi_i[2] - lo_i[2])
        v_i = v[i]

        # Low faces only: face_idx 1 (axis 1 lo) and 3 (axis 2 lo).
        for axis in 1:2
            face_idx = 2 * (axis - 1) + 1
            j_raw = nbrs[face_idx]
            j_raw == 0 && continue
            j = Int(j_raw)

            lo_j, hi_j = cell_physical_box(frame, j)
            V_j = (hi_j[1] - lo_j[1]) * (hi_j[2] - lo_j[2])
            transverse = axis == 1 ? 2 : 1
            face_area = min(hi_i[transverse] - lo_i[transverse],
                              hi_j[transverse] - lo_j[transverse])

            # Centre-to-centre distance along the face normal, with
            # periodic-wrap correction.
            c_i_n = 0.5 * (lo_i[axis] + hi_i[axis])
            c_j_n = 0.5 * (lo_j[axis] + hi_j[axis])
            raw = c_i_n - c_j_n
            domain_extent = frame.hi[axis] - frame.lo[axis]
            if abs(raw) > domain_extent / 2
                raw -= sign(raw) * domain_extent
            end
            dist = abs(raw)

            F = (v[j] - v_i) * face_area / dist
            Lv[i] += F / V_i
            Lv[j] -= F / V_j
        end
    end
    return Lv
end

# ============================================================================
# Helmholtz operator (I − Δt · ν · L) as a Krylov matrix-free operator
# ============================================================================

mutable struct HelmholtzOp{T}
    state::NSCellState
    dt::T
    ν::T
    Lv_scratch::Vector{T}     # mesh-indexed scratch for L·v
    v_scratch::Vector{T}      # mesh-indexed scratch for v
end

Base.size(op::HelmholtzOp) = (length(op.state.leaves), length(op.state.leaves))
Base.size(op::HelmholtzOp, k::Int) = length(op.state.leaves)
Base.eltype(::HelmholtzOp{T}) where {T} = T

function mul!(y::AbstractVector{T}, op::HelmholtzOp{T},
              v::AbstractVector{T}) where {T}
    s = op.state
    # Scatter v (leaf-indexed, length n_leaves) into mesh-indexed scratch.
    @inbounds for i in eachindex(op.v_scratch); op.v_scratch[i] = zero(T); end
    @inbounds for (k, c) in enumerate(s.leaves)
        op.v_scratch[c] = v[k]
    end
    _apply_laplacian!(op.Lv_scratch, op.v_scratch, s)
    @inbounds for (k, c) in enumerate(s.leaves)
        y[k] = v[k] - op.dt * op.ν * op.Lv_scratch[c]
    end
    return y
end

"""
    viscous_step!(state, dt) -> (cg_stats_u, cg_stats_v)

Solve `(I − Δt · ν · L) u = u^n` per component using Krylov.cg with a
matrix-free SPD operator. Writes the new (u, v) back into the field of
record. Returns the per-component Krylov stats for diagnostics.
"""
function viscous_step!(state::NSCellState, dt::Float64)
    cfg = state.config
    field = parent(state.af)
    raw_u = field.u.pfs.storage.u
    raw_v = field.v.pfs.storage.v
    n_leaves = length(state.leaves)
    n_storage = length(raw_u)

    Lv_scratch = Vector{Float64}(undef, n_storage)
    v_scratch  = Vector{Float64}(undef, n_storage)
    op = HelmholtzOp{Float64}(state, dt, cfg.μ, Lv_scratch, v_scratch)

    # Component u: RHS = current u (leaf-indexed flat vector).
    rhs = Vector{Float64}(undef, n_leaves)
    @inbounds for (k, c) in enumerate(state.leaves); rhs[k] = raw_u[c]; end
    u_new, stats_u = cg(op, rhs;
                          rtol = cfg.helmholtz_tol, atol = 0.0,
                          itmax = cfg.helmholtz_maxiter)
    @inbounds for (k, c) in enumerate(state.leaves); raw_u[c] = u_new[k]; end

    # Component v: RHS = current v.
    @inbounds for (k, c) in enumerate(state.leaves); rhs[k] = raw_v[c]; end
    v_new, stats_v = cg(op, rhs;
                          rtol = cfg.helmholtz_tol, atol = 0.0,
                          itmax = cfg.helmholtz_maxiter)
    @inbounds for (k, c) in enumerate(state.leaves); raw_v[c] = v_new[k]; end

    return (stats_u, stats_v)
end

# ============================================================================
# Time integration: forward Euler or RK2 (Heun)
# ============================================================================

function step!(state::NSCellState, dt::Float64)
    field = parent(state.af)
    raw_u = field.u.pfs.storage.u
    raw_v = field.v.pfs.storage.v
    n = length(raw_u)

    if state.config.time_scheme === :euler
        fdu = Vector{Float64}(undef, n)
        fdv = Vector{Float64}(undef, n)
        _accumulate_flux_divergence!(fdu, fdv, state, field)
        @inbounds for c in state.leaves
            invV = 1.0 / _cell_volume(state.frame, c)
            raw_u[c] = raw_u[c] - dt * fdu[c] * invV
            raw_v[c] = raw_v[c] - dt * fdv[c] * invV
        end
    else  # :rk2 (Heun)
        u_n = Vector{Float64}(undef, n)
        v_n = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            u_n[i] = raw_u[i]; v_n[i] = raw_v[i]
        end

        fdu = Vector{Float64}(undef, n)
        fdv = Vector{Float64}(undef, n)
        _accumulate_flux_divergence!(fdu, fdv, state, field)

        @inbounds for c in state.leaves
            invV = 1.0 / _cell_volume(state.frame, c)
            raw_u[c] = u_n[c] - dt * fdu[c] * invV
            raw_v[c] = v_n[c] - dt * fdv[c] * invV
        end

        fdu2 = Vector{Float64}(undef, n)
        fdv2 = Vector{Float64}(undef, n)
        _accumulate_flux_divergence!(fdu2, fdv2, state, field)

        @inbounds for c in state.leaves
            invV = 1.0 / _cell_volume(state.frame, c)
            raw_u[c] = u_n[c] - 0.5 * dt * invV * (fdu[c] + fdu2[c])
            raw_v[c] = v_n[c] - 0.5 * dt * invV * (fdv[c] + fdv2[c])
        end
    end

    state.t += dt
    state.n_steps += 1
    return state
end

"""
    run!(cfg; ic = taylor_green_ic) -> state
"""
function run!(cfg::NSCellConfig; ic = taylor_green_ic)
    state = build_state(cfg; ic = ic)
    while state.t < cfg.t_final - 1e-12
        dt = min(cfg.dt, cfg.t_final - state.t)
        step!(state, dt)
    end
    return state
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    total_momentum(state) -> (mu, mv)

Volume-weighted (u, v) integrals. Conserved exactly under pure
advection on a periodic domain (mass conservation in each component).
"""
function total_momentum(state::NSCellState)
    field = parent(state.af)
    mu = 0.0; mv = 0.0
    @inbounds for c in state.leaves
        V = _cell_volume(state.frame, c)
        mu += field.u[c][1] * V
        mv += field.v[c][1] * V
    end
    return (mu, mv)
end

"""
    kinetic_energy(state) -> Float64

Volume-weighted `(1/2)·<u² + v²>`. Bounded above by the initial value
under pure advection of a divergence-free field by a constant carrier
(actually it equals the initial value up to discretization error).
"""
function kinetic_energy(state::NSCellState)
    field = parent(state.af)
    e = 0.0; vol = 0.0
    @inbounds for c in state.leaves
        V = _cell_volume(state.frame, c)
        u = field.u[c][1]; v = field.v[c][1]
        e += 0.5 * (u * u + v * v) * V
        vol += V
    end
    return e / vol
end

"""
    l2_error(state; analytic = ...) -> Float64

Volume-weighted L² error of (u, v) vs an analytic reference at the
current state time. Default uses `taylor_green_solution` with the
configured carrier velocity.
"""
function l2_error(state::NSCellState;
                   analytic = (x, t) ->
                       taylor_green_solution(x, t, state.config.carrier_velocity))
    field = parent(state.af)
    num = 0.0; vol = 0.0
    @inbounds for c in state.leaves
        lo, hi = cell_physical_box(state.frame, c)
        x = (0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
        V = _cell_volume(state.frame, c)
        u_ref, v_ref = analytic(x, state.t)
        du = field.u[c][1] - u_ref
        dv = field.v[c][1] - v_ref
        num += (du * du + dv * dv) * V
        vol += V
    end
    return sqrt(num / vol)
end

end # module CFDIncompressibleNSCell
