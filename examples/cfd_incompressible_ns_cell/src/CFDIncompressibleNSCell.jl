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
export build_state, step!, run!, viscous_step!, solve_cell_poisson!
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

    # Step 3: Poisson solver (for projection).
    poisson_tol::Float64 = 1e-10
    poisson_maxiter::Int = 2000
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

# Variant of `_cf_face_value` that reads the transverse neighbour value
# from a raw `Vector{Float64}` indexed by mesh-cell id (rather than from
# a named field on a `PolynomialFieldSet`). Used by the cell-mode
# Laplacian where the operand is a flat vector.
function _cf_face_value_arr(mesh::HierarchicalMesh{2},
                              frame::EulerianFrame{2, Float64},
                              v::Vector{Float64},
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
    rho_nbr = v[nbr_idx_i]
    slope_t = (rho_nbr - rho_coarse) / raw_dist
    return (2 * rho_fine + rho_coarse + slope_t * t_offset) / 3
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
# Discrete Laplacian on the AMR mesh — 2nd-order at C/F via Martin-Colella
#
# Conservative form: (L u)_c = (1/V_c) · ∑_faces β · ∂u/∂n · face_area
#                              with β ≡ 1 here.
#
# Same-level faces: ∂u/∂n at the face midpoint ≈ (u_R − u_L) / h. 2nd-order.
#
# C/F sub-faces: the simple centre-to-centre formula `(u_j − u_i) / dist`
# is 2nd-order along the centreline between cells but only 1st-order at
# the sub-face centre (transverse offset h_fine/2). Recovering 2nd
# order requires the same Martin-Colella linear reconstruction we use
# for the advection: compute a 2nd-order face value via the
# transverse-corrected linear fit, then take the fine-side gradient
# (u_fine − u_face)/(h_fine/2) as the flux gradient. This is the
# "fine-ghost" construction; the resulting operator is consistent
# between the two cells and conservative (per-fine-sub-face flux
# accumulates symmetrically into fine cell and the coarse parent).
#
# Iteration ordering: each leaf processes only its *low* faces
# (face_idx ∈ {1, 3}). Each face is then visited exactly once:
# same-level faces by the cell on the high side, C/F faces by each
# fine cell (each fine cell sees the C/F as its low face; coarse cells
# never iterate a C/F face from their high side).
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
        lvl_i = Int(level_of(mesh, i))

        # Low faces only: face_idx 1 (axis 1 lo) and 3 (axis 2 lo).
        for axis in 1:2
            face_idx = 2 * (axis - 1) + 1
            j_raw = nbrs[face_idx]
            j_raw == 0 && continue
            j = Int(j_raw)

            lo_j, hi_j = cell_physical_box(frame, j)
            V_j = (hi_j[1] - lo_j[1]) * (hi_j[2] - lo_j[2])
            v_j = v[j]
            lvl_j = Int(level_of(mesh, j))
            transverse = axis == 1 ? 2 : 1
            face_area = min(hi_i[transverse] - lo_i[transverse],
                              hi_j[transverse] - lo_j[transverse])

            # Distance from each cell centre to the face centre,
            # which equals half the cell extent along the normal axis.
            d_i = (hi_i[axis] - lo_i[axis]) / 2
            d_j = (hi_j[axis] - lo_j[axis]) / 2

            # The 2nd-order Martin-Colella face value (with transverse
            # correction) gives a NON-SPD operator on AMR — its coupling
            # to the coarse cell's transverse neighbour is one-sided
            # (L[i, T] ≠ L[T, i] = 0), so CG diverges. The standard
            # AMReX cell-centred MLABec uses the simple
            # centre-to-centre formula `(v_j − v_i) / (d_i + d_j)`
            # which is V-symmetric and SPD; it's 1st-order at sub-face
            # centres in 2D. We adopt that here.
            # A genuine 2nd-order, SPD-preserving AMR Laplacian is a
            # more involved construction (see Almgren-Bell-Crutchfield
            # 2000 for the "stencil-symmetric" version) and is left as
            # a v2 follow-on.
            dist = d_i + d_j
            F = (v_j - v_i) * face_area / dist
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
# Step 3: cell-mode Poisson solver for projection.
#
# Solves `−L · φ = ρ` on the AMR mesh under periodic BCs.  The operator
# `−L` is symmetric positive *semi-definite*: it has a one-dimensional
# nullspace of constants. We handle this by (a) subtracting the
# volume-weighted mean of the RHS to make it compatible with the range
# of the operator and (b) anchoring the solution by subtracting its
# volume-weighted mean at the end.
#
# Implementation reuses `_apply_laplacian!` for the matvec; CG without
# a preconditioner is adequate at moderate N (≤ 64²). A multigrid
# preconditioner (cell-native FAC V-cycle) is the natural v2
# acceleration but is not required for the step-4 projection
# verification at the resolutions we care about — CG converges in a
# few hundred iterations at the tolerances we set.
# ============================================================================

mutable struct NegLaplacianOp
    state::NSCellState
    Lv_scratch::Vector{Float64}
    v_scratch::Vector{Float64}
end

Base.size(op::NegLaplacianOp) = (length(op.state.leaves), length(op.state.leaves))
Base.size(op::NegLaplacianOp, k::Int) = length(op.state.leaves)
Base.eltype(::NegLaplacianOp) = Float64

function mul!(y::AbstractVector{Float64}, op::NegLaplacianOp,
              v::AbstractVector{Float64})
    s = op.state
    @inbounds for i in eachindex(op.v_scratch); op.v_scratch[i] = 0.0; end
    @inbounds for (k, c) in enumerate(s.leaves)
        op.v_scratch[c] = v[k]
    end
    _apply_laplacian!(op.Lv_scratch, op.v_scratch, s)
    @inbounds for (k, c) in enumerate(s.leaves)
        y[k] = -op.Lv_scratch[c]
    end
    return y
end

# Volume-weighted mean of a leaf-indexed vector.
function _leaf_volume_mean(v::Vector{Float64}, state::NSCellState)
    num = 0.0; vol = 0.0
    @inbounds for (k, c) in enumerate(state.leaves)
        V = _cell_volume(state.frame, c)
        num += v[k] * V
        vol += V
    end
    return num / vol
end

"""
    solve_cell_poisson!(phi, rhs, state; tol, maxiter) -> stats

Solve `−L · φ = ρ` on the leaf cells of the AMR mesh under periodic
BCs. `phi` and `rhs` are length-`n_leaves` flat vectors. Returns the
Krylov.cg stats.

The RHS is automatically projected to the range of the operator
(zero-mean) and the solution is anchored to zero-mean — i.e. the
returned φ has volume-weighted mean equal to zero.
"""
function solve_cell_poisson!(phi::Vector{Float64}, rhs::Vector{Float64},
                              state::NSCellState;
                              tol::Float64 = state.config.poisson_tol,
                              maxiter::Int = state.config.poisson_maxiter)
    n_leaves = length(state.leaves)
    n_storage = length(parent(state.af).u)
    @assert length(phi) == n_leaves
    @assert length(rhs) == n_leaves

    # Compatibility: project RHS to mean-zero so it lives in range(−L).
    mean_rhs = _leaf_volume_mean(rhs, state)
    rhs_zm = Vector{Float64}(undef, n_leaves)
    @inbounds for k in 1:n_leaves
        rhs_zm[k] = rhs[k] - mean_rhs
    end

    Lv_scratch = Vector{Float64}(undef, n_storage)
    v_scratch  = Vector{Float64}(undef, n_storage)
    op = NegLaplacianOp(state, Lv_scratch, v_scratch)

    phi_new, stats = cg(op, rhs_zm;
                          rtol = tol, atol = 0.0,
                          itmax = maxiter)

    # Anchor the constant nullspace.
    mean_phi = _leaf_volume_mean(phi_new, state)
    @inbounds for k in 1:n_leaves
        phi[k] = phi_new[k] - mean_phi
    end
    return stats
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
