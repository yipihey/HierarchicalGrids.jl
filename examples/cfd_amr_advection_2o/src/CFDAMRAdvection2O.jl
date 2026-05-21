"""
    CFDAMRAdvection2O

2D constant-velocity scalar advection on a HierarchicalMesh + AdaptiveField,
with 2nd-order centered fluxes that remain 2nd-order across coarse-fine
(C/F) interfaces.

The flux at same-level faces is the standard centered average. At C/F
faces, a Martin-Colella style "fine ghost" reconstruction is used: a
virtual fine-sized ghost cell on the coarse side is built from the
coarse cell value plus the transverse coarse-cell gradient, then the
face value is the centered average of the fine cell and the fine
ghost — same formula as same-level, just with the right ghost.

The `for_each_face!` orchestrator dispatches per-fine-sub-face at every
C/F interface, so the coarse cell sees the area-weighted sum of fine
sub-fluxes automatically — i.e., flux refluxing is for free.

Time integration: RK2 (Heun) for 2nd-order in time, so the overall
scheme is 2nd-order in space and time across same-level AND C/F faces.
"""
module CFDAMRAdvection2O

using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells, is_leaf,
                          level_of, enumerate_leaves, cell_physical_box,
                          EulerianFrame, FrameBoundaries, BCKind, PERIODIC,
                          face_neighbors, face_neighbors_with_bcs,
                          ensure_neighbor_graph!,
                          BernsteinBasis, allocate_polynomial_fields, SoA,
                          AdaptiveField, dispose!,
                          for_each_face!, Sequential

export AdvectionConfig, AdvectionState
export build_state, step!, run!
export sinusoidal_ic, sinusoidal_solution
export l2_error, total_mass

# ============================================================================
# Configuration
# ============================================================================

"""
    AdvectionConfig

# Mesh
- `n_base_refines::Int` — base grid has `2^n_base_refines` cells per axis.
- `refine_center::Bool` — refine a centered block once more (one extra level).
- `center_box_lo`, `center_box_hi` — extent (fraction of unit square) of the
  refined block, e.g. `(0.25, 0.25)` to `(0.75, 0.75)`. Cells whose centers
  lie in `[lo, hi]²` are refined.

# Physics
- `velocity::NTuple{2,Float64}` — constant advection velocity.
- `domain_lo`, `domain_hi` — domain extent (default unit square).

# Time
- `t_final::Float64`, `dt::Float64`.
- `time_scheme::Symbol` — `:euler` (1st order) or `:rk2` (2nd order Heun).
"""
Base.@kwdef struct AdvectionConfig
    n_base_refines::Int = 5
    refine_center::Bool = false
    center_box_lo::NTuple{2, Float64} = (0.25, 0.25)
    center_box_hi::NTuple{2, Float64} = (0.75, 0.75)

    velocity::NTuple{2, Float64} = (1.0, 0.5)
    domain_lo::NTuple{2, Float64} = (0.0, 0.0)
    domain_hi::NTuple{2, Float64} = (1.0, 1.0)

    t_final::Float64 = 1.0
    dt::Float64 = 1e-3
    time_scheme::Symbol = :rk2
end

# ============================================================================
# State
# ============================================================================

mutable struct AdvectionState
    config::AdvectionConfig
    mesh::HierarchicalMesh{2}
    frame::EulerianFrame{2, Float64}
    bcs::FrameBoundaries{2}
    af::AdaptiveField                # holds field-of-record (.rho)
    leaves::Vector{Int}
    t::Float64
    n_steps::Int
end

# ============================================================================
# Initial conditions / analytic reference
# ============================================================================

"""
    sinusoidal_ic(x; k=(2π, 2π)) -> Float64

A smooth periodic profile suitable for spatial convergence testing:
    ρ₀(x, y) = sin(k_x · x) · cos(k_y · y) + 0.5 cos(k_x · x) · sin(k_y · y)
The two terms keep the field from being separable (so the AMR transverse
correction matters for 2nd order in 2D).
"""
function sinusoidal_ic(x::NTuple{2, Float64};
                         k::NTuple{2, Float64} = (2π, 2π))
    s = sin(k[1] * x[1]) * cos(k[2] * x[2]) +
        0.5 * cos(k[1] * x[1]) * sin(k[2] * x[2])
    return s
end

"""
    sinusoidal_solution(x, t, u; k=(2π, 2π)) -> Float64

Analytic solution: ρ(x, t) = ρ₀(x − u · t).
"""
function sinusoidal_solution(x::NTuple{2, Float64}, t::Float64,
                              u::NTuple{2, Float64};
                              k::NTuple{2, Float64} = (2π, 2π))
    return sinusoidal_ic((x[1] - u[1] * t, x[2] - u[2] * t); k = k)
end

# ============================================================================
# State construction
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
    pfs = allocate_polynomial_fields(SoA(), basis, n; rho = Float64)
    @inbounds for i in 1:n; pfs.rho[i] = (0.0,); end
    return pfs
end

"""
    build_state(cfg; ic = sinusoidal_ic) -> AdvectionState

Construct the mesh (with optional central refinement), allocate the
adaptive field, apply the initial condition `ic`.
"""
function build_state(cfg::AdvectionConfig; ic = sinusoidal_ic)
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
        if !isempty(to_refine)
            refine_cells!(mesh, to_refine)
        end
    end

    pfs = _alloc_field(mesh)
    af = AdaptiveField(pfs, mesh)

    # Apply IC: cell-mean is well-approximated by point evaluation at the
    # cell center (Bernstein degree-0 stores the cell average).
    leaves = enumerate_leaves(mesh)
    field = parent(af)
    @inbounds for c in leaves
        lo, hi = cell_physical_box(frame, c)
        x = (0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
        field.rho[c] = (ic(x),)
    end

    return AdvectionState(cfg, mesh, frame, bcs, af, leaves, 0.0, 0)
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
    # Hanging-node face area is the smaller of left/right cell extents
    # (the fine sub-face area). The orchestrator fires once per sub-face.
    return min(hi_l[off] - lo_l[off], hi_r[off] - lo_r[off])
end

# ============================================================================
# Martin-Colella C/F face reconstruction
#
# At a C/F face the coarse cell's "natural" face value sits at the coarse
# face midpoint, offset transversely from each fine sub-face center by
# h_fine/2.  A naive 1D linear interpolant (2/3, 1/3) is 2nd-order along
# the centerline between fine and coarse cells, but loses one order at
# the sub-face center due to that transverse offset.  Recovering 2nd
# order requires accounting for the transverse coarse-cell gradient.
#
# Fit a linear function f(x, y) = a + b·x + c·y through three values:
#   - ρ_C at coarse cell center,
#   - ρ_F at fine cell center,
#   - ρ_T at the coarse transverse neighbour (in the direction of the
#     fine cell's transverse offset),
# then evaluate it at the sub-face center.  The closed form is
#
#   face_value = (2/3)·ρ_F + (1/3)·ρ_C + (1/3)·slope_t·t_offset
#
# where slope_t is the coarse-cell transverse slope and t_offset is the
# fine cell's transverse offset from the coarse cell center (±h_fine/2).
# Exact for linear fields.
# ============================================================================

# Convention for `face_neighbors(mesh, i)` indices:
#   axis 1: lo=1, hi=2
#   axis 2: lo=3, hi=4
@inline function _nbr_index(nbrs, axis::Int, side::Int)
    # axis ∈ {1, 2}, side ∈ {-1, +1}
    return nbrs[2 * (axis - 1) + (side > 0 ? 2 : 1)]
end

# Reconstruct rho at the C/F sub-face given the coarse cell, the fine
# cell, and the face axis. `field` is the field-of-record (PolynomialFieldSet).
function _cf_face_value(mesh::HierarchicalMesh{2},
                          frame::EulerianFrame{2, Float64},
                          field,
                          bcs::FrameBoundaries{2},
                          i_coarse::Int, i_fine::Int,
                          rho_coarse::Float64, rho_fine::Float64,
                          axis::Int)
    transverse = axis == 1 ? 2 : 1

    # Transverse offset of the fine cell from the coarse cell center.
    lo_c, hi_c = cell_physical_box(frame, i_coarse)
    lo_f, hi_f = cell_physical_box(frame, i_fine)
    coarse_center_t = 0.5 * (lo_c[transverse] + hi_c[transverse])
    fine_center_t   = 0.5 * (lo_f[transverse] + hi_f[transverse])
    t_offset = fine_center_t - coarse_center_t

    # If the transverse offset is essentially zero (e.g., same-level
    # rounding artifact), fall back to 1D linear: face = (2 ρ_fine + ρ_coarse)/3.
    if abs(t_offset) < 1e-12 * (hi_c[transverse] - lo_c[transverse])
        return (2 * rho_fine + rho_coarse) / 3
    end

    # Find the coarse transverse neighbor in the direction of the fine cell.
    # Periodic BC-aware lookup (so cells at the periodic boundary still
    # find their wrapped neighbor).
    nbrs = face_neighbors_with_bcs(mesh, i_coarse, bcs)
    t_side = t_offset > 0 ? +1 : -1
    nbr_idx = _nbr_index(nbrs, transverse, t_side)

    h_coarse_t = hi_c[transverse] - lo_c[transverse]
    if nbr_idx == 0
        # No transverse neighbor (non-periodic boundary): use 1D linear.
        return (2 * rho_fine + rho_coarse) / 3
    end

    # Transverse slope from the coarse cell to its neighbor.  We use a
    # one-sided difference toward the fine cell.  For the orientation
    # we picked (`t_side` = direction of fine cell), the neighbor sits
    # at coarse_center_t + t_side * (h_coarse_t/2 + h_neighbor_t/2).
    nbr_idx_i = Int(nbr_idx)
    lo_n, hi_n = cell_physical_box(frame, nbr_idx_i)
    nbr_center_t = 0.5 * (lo_n[transverse] + hi_n[transverse])
    # Periodic-wrap distance: handle the case where lo_n is on the other
    # side of the domain due to a periodic neighbour.
    domain_extent = frame.hi[transverse] - frame.lo[transverse]
    raw_dist = nbr_center_t - coarse_center_t
    if abs(raw_dist) > domain_extent / 2
        raw_dist -= sign(raw_dist) * domain_extent
    end
    rho_nbr = field.rho[nbr_idx_i][1]
    slope_t = (rho_nbr - rho_coarse) / raw_dist

    # 2nd-order face value at sub-face center (closed-form solution to
    # the 3-point linear fit; see the section header for derivation).
    return (2 * rho_fine + rho_coarse + slope_t * t_offset) / 3
end

# ============================================================================
# Flux-divergence kernel and one-stage update
# ============================================================================

# Compute the per-cell flux divergence  fdiv[c] = ∑_{faces of c} F · A,
# i.e. the negative of dρ/dt before dividing by V.
function _accumulate_flux_divergence!(fdiv::Vector{Float64},
                                       state::AdvectionState,
                                       field)
    mesh = state.mesh
    frame = state.frame
    bcs   = state.bcs
    velocity = state.config.velocity

    @inbounds for i in 1:length(fdiv); fdiv[i] = 0.0; end

    flux_kernel = let frame = frame, mesh = mesh, bcs = bcs,
                       velocity = velocity, fdiv = fdiv, field = field
        function (cv_L, cv_R, normal, _ctx)
            i = cv_L.index
            j = cv_R.index
            rho_L = cv_L[Val(:rho)][1]
            rho_R = cv_R[Val(:rho)][1]
            lvl_L = cv_L.level
            lvl_R = cv_R.level
            axis = normal[1] != 0.0 ? 1 : 2
            u_n = velocity[1] * normal[1] + velocity[2] * normal[2]

            if lvl_L == lvl_R
                rho_face = 0.5 * (rho_L + rho_R)
            elseif lvl_L < lvl_R          # left coarse, right fine
                rho_face = _cf_face_value(mesh, frame, field, bcs,
                                            i, j, rho_L, rho_R, axis)
            else                           # left fine, right coarse
                rho_face = _cf_face_value(mesh, frame, field, bcs,
                                            j, i, rho_R, rho_L, axis)
            end

            F = u_n * rho_face
            scale = F * _face_area(frame, i, j, axis)
            fdiv[i] += scale
            fdiv[j] -= scale
            return nothing
        end
    end

    fin_v  = (rho = field.rho,)
    fout_v = fin_v
    for_each_face!(flux_kernel, fout_v, fin_v, frame;
                   bcs = bcs,
                   backend = Sequential())

    # Periodic-axis face pass (the orchestrator skips periodic boundary
    # faces; we handle them manually here).  Same construction as the
    # cfd_implicit_ns example.
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
                rho_L = field.rho[j][1]    # other-side cell (periodic-wrap leftward)
                rho_R = field.rho[i][1]
                lvl_L = Int(level_of(mesh, j))
                lvl_R = Int(level_of(mesh, i))
                axis_idx = axis
                if lvl_L == lvl_R
                    rho_face = 0.5 * (rho_L + rho_R)
                elseif lvl_L < lvl_R
                    rho_face = _cf_face_value(mesh, frame, field, bcs,
                                                j, i, rho_L, rho_R, axis_idx)
                else
                    rho_face = _cf_face_value(mesh, frame, field, bcs,
                                                i, j, rho_R, rho_L, axis_idx)
                end
                u_n = velocity[1] * normal[1] + velocity[2] * normal[2]
                F = u_n * rho_face
                face_area = _face_area(frame, j, i, axis)
                fdiv[j] += F * face_area
                fdiv[i] -= F * face_area
            end
        end
    end

    return fdiv
end

# One forward-Euler-style sub-step: ρ_out = ρ_in − dt · ∇·F(ρ_in) / V.
function _apply_rhs!(rho_out, rho_in, fdiv::Vector{Float64},
                     state::AdvectionState, dt::Float64)
    @inbounds for c in state.leaves
        invV = 1.0 / _cell_volume(state.frame, c)
        rho_out[c] = (rho_in[c][1] - dt * fdiv[c] * invV,)
    end
    return rho_out
end

"""
    step!(state, dt) -> state

Take one time step.  RK2 (Heun) for `time_scheme = :rk2`, forward Euler
for `:euler`.  Operates directly on the field-of-record's raw storage
buffer to avoid allocating PolynomialFieldView clones.
"""
function step!(state::AdvectionState, dt::Float64)
    field = parent(state.af)
    raw_rho = field.rho.pfs.storage.rho   # Vector{Float64}
    n = length(raw_rho)

    if state.config.time_scheme === :euler
        fdiv = Vector{Float64}(undef, n)
        _accumulate_flux_divergence!(fdiv, state, field)
        @inbounds for c in state.leaves
            invV = 1.0 / _cell_volume(state.frame, c)
            raw_rho[c] = raw_rho[c] - dt * fdiv[c] * invV
        end
    else  # :rk2 (Heun)
        # Save ρ^n into a flat scratch.
        rho_n = Vector{Float64}(undef, n)
        @inbounds for i in 1:n; rho_n[i] = raw_rho[i]; end

        # Stage 1 divergence on ρ^n.
        fdiv = Vector{Float64}(undef, n)
        _accumulate_flux_divergence!(fdiv, state, field)

        # Write ρ_pred = ρ^n − dt · ∇·F(ρ^n)/V into raw_rho (in place).
        @inbounds for c in state.leaves
            invV = 1.0 / _cell_volume(state.frame, c)
            raw_rho[c] = rho_n[c] - dt * fdiv[c] * invV
        end

        # Stage 2 divergence on ρ_pred (which is now in raw_rho).
        fdiv2 = Vector{Float64}(undef, n)
        _accumulate_flux_divergence!(fdiv2, state, field)

        # Final update: ρ^{n+1} = ρ^n − dt/2 (∇·F(ρ^n) + ∇·F(ρ_pred)) / V.
        @inbounds for c in state.leaves
            invV = 1.0 / _cell_volume(state.frame, c)
            raw_rho[c] = rho_n[c] - 0.5 * dt * invV * (fdiv[c] + fdiv2[c])
        end
    end

    state.t += dt
    state.n_steps += 1
    return state
end

"""
    run!(cfg; ic = sinusoidal_ic) -> state

Build a fresh state and step until `t >= t_final`.
"""
function run!(cfg::AdvectionConfig; ic = sinusoidal_ic)
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
    total_mass(state) -> Float64

Volume-weighted integral of ρ over leaves. Should be conserved across
arbitrary mesh topology.
"""
function total_mass(state::AdvectionState)
    field = parent(state.af)
    s = 0.0
    @inbounds for c in state.leaves
        s += field.rho[c][1] * _cell_volume(state.frame, c)
    end
    return s
end

"""
    l2_error(state; analytic = (x, t) -> ...) -> Float64

Volume-weighted L² error of ρ vs an analytic reference at the current
state time.  Defaults to the sinusoidal_solution with the configured
velocity.
"""
function l2_error(state::AdvectionState;
                   analytic = (x, t) ->
                       sinusoidal_solution(x, t, state.config.velocity))
    field = parent(state.af)
    num = 0.0; vol = 0.0
    @inbounds for c in state.leaves
        lo, hi = cell_physical_box(state.frame, c)
        x = (0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
        v = _cell_volume(state.frame, c)
        ref = analytic(x, state.t)
        d = field.rho[c][1] - ref
        num += d * d * v
        vol += v
    end
    return sqrt(num / vol)
end

end # module CFDAMRAdvection2O
