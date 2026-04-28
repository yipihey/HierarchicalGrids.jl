"""
    CFDCompressibleSod

A 2D compressible-flow worked example for HierarchicalGrids.jl that
exercises the Phase-2 cell-based AMR stack on the standard Sod shock-
tube problem (Toro §4.3).

The setup is the canonical 1D Sod problem extruded to 2D:

- Domain `[0, 1] × [0, 0.1]`.
- `OUTFLOW` BCs on the x-axes; `PERIODIC` on the y-axes.
- `t = 0`: `(ρ, u, v, p) = (1.0, 0.0, 0.0, 1.0)` for x < 0.5, and
  `(0.125, 0.0, 0.0, 0.1)` for x ≥ 0.5.
- Final time `t = 0.2`.
- Ideal-gas equation of state with γ = 1.4.

The exact (analytic) Riemann solution at `t = 0.2` consists of, from
left to right: a left-going rarefaction, a contact discontinuity moving
to the right, and a right-going shock. We ship a closed-form exact-
Riemann sampler (Toro Algorithm 4.1) so the L¹-error test against the
analytic solution is meaningful.

Numerics:

- 4 conserved variables stored as `BernsteinBasis{2, 0}` cell means:
  `(ρ, ρu, ρv, E)`. (Degree-0 makes the single coefficient exactly the
  cell mean — the natural FV state — and avoids the higher-degree
  AdaptiveField warning. Slope-limited reconstruction is left as a
  follow-up; this PR ships first-order godunov + HLL.)
- HLL approximate Riemann solver at faces.
- First-order forward-Euler in time with a CFL-controlled adaptive `dt`.
- AMR refinement on `|∇ρ|/ρ` (max-neighbor finite difference).

Mass and energy conservation are exact to round-off. Momentum is
exactly conserved across PERIODIC faces but NOT across OUTFLOW faces
(by construction: the OUTFLOW boundary acts as a flux sink), so we
report momentum drift only as a sanity bound.
"""
module CFDCompressibleSod

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
                          RemapDiagnostics, reset!,
                          Sequential

import HierarchicalGrids.Mesh

export SodConfig, SodState, run!, build_state

# ============================================================================
# Physical constants
# ============================================================================

"""
    GAMMA

Adiabatic index for the ideal-gas equation of state. The standard value
for the Sod test is γ = 1.4 (diatomic gas).
"""
const GAMMA = 1.4

# ============================================================================
# Configuration
# ============================================================================

"""
    SodConfig

Parameters for the Sod-tube example.

# Fields
- `n_initial_refines` — number of uniform isotropic refinements of the
  square root mesh. Each refine doubles cells per axis, so e.g. 5 gives
  a 32×32 effective mesh on the unit square root. Note the physical
  domain is `[0, 1] × [0, 0.1]` so the y-extent is squashed by 10×.
- `cfl`  — CFL safety factor for the adaptive time step (≤ 0.5 for HLL).
- `t_final` — physical end-time for the run.
- `refine_threshold`, `coarsen_threshold` — `|∇ρ|/ρ` thresholds.
- `amr_every` — number of steps between AMR cycles. Set to 0 to disable.
- `max_level` — refinement-depth ceiling.
"""
struct SodConfig
    n_initial_refines::Int
    cfl::Float64
    t_final::Float64
    refine_threshold::Float64
    coarsen_threshold::Float64
    amr_every::Int
    max_level::Int
end

function SodConfig(; n_initial_refines::Int = 5,
                     cfl::Float64 = 0.4,
                     t_final::Float64 = 0.2,
                     refine_threshold::Float64 = 0.1,
                     coarsen_threshold::Float64 = 0.01,
                     amr_every::Int = 5,
                     max_level::Int = 3)
    return SodConfig(n_initial_refines, cfl, t_final,
                      refine_threshold, coarsen_threshold,
                      amr_every, max_level)
end

# ============================================================================
# State conversions (4 conserved variables ↔ primitives)
# ============================================================================

"""
    cons_to_prim(U) -> (ρ, u, v, p)

Convert a conservative-variable tuple `U = (ρ, ρu, ρv, E)` to primitives.
Uses the ideal-gas relation `p = (γ-1) ρ e_internal` with the kinetic-
energy subtracted total energy.
"""
@inline function cons_to_prim(U::NTuple{4, Float64})
    ρ = U[1]
    u = U[2] / ρ
    v = U[3] / ρ
    e_internal = U[4] / ρ - 0.5 * (u * u + v * v)
    p = (GAMMA - 1.0) * ρ * e_internal
    return (ρ, u, v, p)
end

"""
    prim_to_cons(ρ, u, v, p) -> (ρ, ρu, ρv, E)

Inverse of [`cons_to_prim`](@ref).
"""
@inline function prim_to_cons(ρ::Float64, u::Float64, v::Float64, p::Float64)
    e_internal = p / ((GAMMA - 1.0) * ρ)
    E = ρ * (e_internal + 0.5 * (u * u + v * v))
    return (ρ, ρ * u, ρ * v, E)
end

"""
    sound_speed(ρ, p) -> Float64

`c = sqrt(γ p / ρ)` — ideal-gas sound speed.
"""
@inline sound_speed(ρ::Float64, p::Float64) = sqrt(GAMMA * p / ρ)

# ============================================================================
# Euler-equation flux in a given direction
# ============================================================================

# Physical Euler flux F·n in the direction of the unit normal `n`.
# Returns a 4-tuple matching the conservative-variable layout.
@inline function euler_flux(U::NTuple{4, Float64}, n::NTuple{2, Float64})
    ρ, u, v, p = cons_to_prim(U)
    vn = u * n[1] + v * n[2]
    E = U[4]
    return (ρ * vn,
             ρ * u * vn + p * n[1],
             ρ * v * vn + p * n[2],
             (E + p) * vn)
end

# ============================================================================
# HLL Riemann solver (2-wave estimate)
# ============================================================================

"""
    hll_flux(UL, UR, normal) -> NTuple{4, Float64}

Single-state HLL flux estimate (Harten-Lax-van Leer; Toro §10.3) for
the Euler equations across a face with unit `normal` pointing from
`UL` (left state) to `UR` (right state). Two-wave estimate using the
direct-DDB wave-speed bounds:

    SL = min(uL · n - cL, uR · n - cR)
    SR = max(uL · n + cL, uR · n + cR)

with the closed-form HLL flux

    F = (SR FL - SL FR + SL SR (UR - UL)) / (SR - SL)

The function falls back to upwind FL or FR for supersonic cases
(`SL ≥ 0` or `SR ≤ 0`) for numerical robustness.
"""
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
        denom = SR - SL
        return ((SR * FL[1] - SL * FR[1] + SL * SR * (UR[1] - UL[1])) / denom,
                (SR * FL[2] - SL * FR[2] + SL * SR * (UR[2] - UL[2])) / denom,
                (SR * FL[3] - SL * FR[3] + SL * SR * (UR[3] - UL[3])) / denom,
                (SR * FL[4] - SL * FR[4] + SL * SR * (UR[4] - UL[4])) / denom)
    end
end

# ============================================================================
# Initial condition (Sod left/right states)
# ============================================================================

# Standard Sod left/right primitives.
const SOD_LEFT_PRIM  = (1.0, 0.0, 0.0, 1.0)
const SOD_RIGHT_PRIM = (0.125, 0.0, 0.0, 0.1)
const SOD_INTERFACE_X = 0.5

# Cell-mean conservative state of the IC: a Sod tube initialized via point
# evaluation at the cell centroid. The IC is piecewise-constant in x, so the
# cell-mean coincides with the centroid value EXCEPT for the cell straddling
# the interface — which we accept as a 1-cell IC error (standard convention).
@inline function sod_ic(x::NTuple{2, Float64})
    ρ, u, v, p = x[1] < SOD_INTERFACE_X ? SOD_LEFT_PRIM : SOD_RIGHT_PRIM
    return prim_to_cons(ρ, u, v, p)
end

# ============================================================================
# State
# ============================================================================

"""
    SodState

Per-run state assembled by [`build_state`](@ref). Owns the mesh, frame,
BC spec, the active 4-component `AdaptiveField`, a scratch output
field-set, and bookkeeping for conservation-drift diagnostics.

The two field-sets are double-buffered: every step computes new
coefficients into `field_out`, then copies them back into the
`AdaptiveField`-tracked input. `AdaptiveField` listens on the input
field and remaps coefficients on refinement events.
"""
mutable struct SodState
    config::SodConfig
    mesh::HierarchicalMesh{2}
    frame::EulerianFrame{2, Float64}
    bcs::FrameBoundaries{2}
    af::AdaptiveField
    field_out::Any                 # PolynomialFieldSet, write target
    # Per-cell flux-divergence accumulator (shared by all 4 components,
    # SoA-style). The face kernel writes face_flux × area × dt into these
    # buffers without dividing by cell volume; the per-cell update pass
    # then computes Δρ = -flux_div / V at the end. This recovers exact
    # mass / momentum / energy conservation up to FP roundoff per cell
    # (rather than per face, which is sensitive to volume-asymmetry round-
    # off on a uniform mesh whose extents aren't representable in Float64).
    flux_div_rho::Vector{Float64}
    flux_div_rhou::Vector{Float64}
    flux_div_rhov::Vector{Float64}
    flux_div_E::Vector{Float64}
    diagnostics::RemapDiagnostics{Float64}
    initial_mass::Float64
    initial_momentum_x::Float64
    initial_momentum_y::Float64
    initial_energy::Float64
    n_leaves_initial::Int
    t::Float64                      # current sim time
    n_steps::Int                    # step counter
end

# ----------------------------------------------------------------------------
# Build helpers
# ----------------------------------------------------------------------------

# Build a mesh refined uniformly to `n_levels` levels.
function _build_uniform_mesh(n_levels::Int)
    mesh = HierarchicalMesh{2}()
    for _ in 1:n_levels
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    return mesh
end

# Allocate a 4-field PolynomialFieldSet matching the mesh's current cell count.
# Fields: rho, rhou, rhov, E (all Float64, BernsteinBasis{2, 0}).
function _alloc_field(mesh::HierarchicalMesh{2})
    basis = BernsteinBasis{2, 0}()
    pfs = allocate_polynomial_fields(SoA(), basis, n_cells(mesh);
                                       rho  = Float64,
                                       rhou = Float64,
                                       rhov = Float64,
                                       E    = Float64)
    @inbounds for i in 1:n_cells(mesh)
        pfs.rho[i]  = (0.0,)
        pfs.rhou[i] = (0.0,)
        pfs.rhov[i] = (0.0,)
        pfs.E[i]    = (0.0,)
    end
    return pfs
end

# Number of elements in a PolynomialFieldSet (authoritative size for
# storage indexing — distinct from `HierarchicalGrids.n_cells`).
@inline _field_size(pfs) = pfs.n

# Resize an output field-set + flux-div buffers to match the current
# mesh's cell count, by rebuilding. Both are zeroed each step, so
# tossing old storage is fine.
function _resize_output_field!(state::SodState)
    nc = n_cells(state.mesh)
    if nc != _field_size(state.field_out)
        state.field_out = _alloc_field(state.mesh)
    end
    if nc != length(state.flux_div_rho)
        state.flux_div_rho  = zeros(Float64, nc)
        state.flux_div_rhou = zeros(Float64, nc)
        state.flux_div_rhov = zeros(Float64, nc)
        state.flux_div_E    = zeros(Float64, nc)
    end
    return state
end

# Initialize the field-set from the IC by sampling at cell centroids.
# Degree-0 Bernstein basis: the single coefficient IS the cell mean.
function _init_from_ic!(field, frame::EulerianFrame{2, Float64})
    @inbounds for i in 1:n_cells(frame.mesh)
        is_leaf(frame.mesh.cells[i]) || continue
        lo, hi = cell_physical_box(frame, i)
        cx = 0.5 * (lo[1] + hi[1])
        cy = 0.5 * (lo[2] + hi[2])
        U = sod_ic((cx, cy))
        field.rho[i]  = (U[1],)
        field.rhou[i] = (U[2],)
        field.rhov[i] = (U[3],)
        field.E[i]    = (U[4],)
    end
    return field
end

# Total mass / momentum / energy = sum over leaves of (cell_mean * volume).
function _conserved_totals(field, frame::EulerianFrame{2, Float64})
    m  = 0.0
    px = 0.0
    py = 0.0
    e  = 0.0
    @inbounds for i in 1:n_cells(frame.mesh)
        is_leaf(frame.mesh.cells[i]) || continue
        lo, hi = cell_physical_box(frame, i)
        vol = (hi[1] - lo[1]) * (hi[2] - lo[2])
        m  += field.rho[i][1]  * vol
        px += field.rhou[i][1] * vol
        py += field.rhov[i][1] * vol
        e  += field.E[i][1]    * vol
    end
    return (m, px, py, e)
end

# Cell volume in 2D.
@inline function _cell_volume(frame::EulerianFrame{2, Float64}, i::Int)
    lo, hi = cell_physical_box(frame, i)
    return (hi[1] - lo[1]) * (hi[2] - lo[2])
end

# Face area in 2D = orthogonal extent of the smaller of the two adjoining
# cells. For conforming faces these are equal; for hanging-node faces the
# fine cell's extent is the actual face area, ensuring conservative book-
# keeping when the coarse cell sees `n` fine neighbors on one face.
@inline function _face_area(frame::EulerianFrame{2, Float64},
                              i_left::Int, i_right::Int, axis::Int)
    lo_l, hi_l = cell_physical_box(frame, i_left)
    lo_r, hi_r = cell_physical_box(frame, i_right)
    off = axis == 1 ? 2 : 1
    ext_l = hi_l[off] - lo_l[off]
    ext_r = hi_r[off] - lo_r[off]
    return min(ext_l, ext_r)
end

# Read a 4-tuple conservative state from the input field-set at cell `i`.
@inline function _read_U(field, i::Int)
    return (field.rho[i][1], field.rhou[i][1], field.rhov[i][1], field.E[i][1])
end

# Write a 4-tuple conservative state into the output field-set at cell `i`.
@inline function _write_U!(field, i::Int, U::NTuple{4, Float64})
    field.rho[i]  = (U[1],)
    field.rhou[i] = (U[2],)
    field.rhov[i] = (U[3],)
    field.E[i]    = (U[4],)
    return nothing
end

# ============================================================================
# State construction + IC
# ============================================================================

"""
    build_state(config::SodConfig) -> SodState

Build the mesh, frame, BCs, adaptive field, and diagnostics for one run.
"""
function build_state(config::SodConfig)
    mesh = _build_uniform_mesh(config.n_initial_refines)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 0.1))
    bcs = FrameBoundaries(((OUTFLOW, OUTFLOW), (PERIODIC, PERIODIC)))

    field_in = _alloc_field(mesh)
    _init_from_ic!(field_in, frame)

    af = AdaptiveField(field_in, mesh)
    field_out = _alloc_field(mesh)

    diag = RemapDiagnostics{Float64}()
    m0, px0, py0, e0 = _conserved_totals(field_in, frame)
    n_leaves = length(enumerate_leaves(mesh))

    nc = n_cells(mesh)
    flux_div_rho  = zeros(Float64, nc)
    flux_div_rhou = zeros(Float64, nc)
    flux_div_rhov = zeros(Float64, nc)
    flux_div_E    = zeros(Float64, nc)

    return SodState(config, mesh, frame, bcs, af, field_out,
                     flux_div_rho, flux_div_rhou, flux_div_rhov, flux_div_E,
                     diag, m0, px0, py0, e0, n_leaves, 0.0, 0)
end

# ============================================================================
# Adaptive time-step (CFL on max signal speed)
# ============================================================================

# Compute max wave speed |u·n̂| + c over all leaves and both axes.
function _max_wave_speed(field, frame::EulerianFrame{2, Float64})
    s_max = 0.0
    @inbounds for i in 1:n_cells(frame.mesh)
        is_leaf(frame.mesh.cells[i]) || continue
        U = _read_U(field, i)
        ρ, u, v, p = cons_to_prim(U)
        # If the IC has a tiny pressure or density, guard against NaN.
        if !(ρ > 0.0) || !(p > 0.0)
            continue
        end
        c = sound_speed(ρ, p)
        s = max(abs(u), abs(v)) + c
        if s > s_max
            s_max = s
        end
    end
    return s_max
end

# Compute min cell extent (h_min) over all leaves to set the stable dt.
function _min_cell_extent(frame::EulerianFrame{2, Float64})
    h = Inf
    @inbounds for i in 1:n_cells(frame.mesh)
        is_leaf(frame.mesh.cells[i]) || continue
        lo, hi = cell_physical_box(frame, i)
        ext = min(hi[1] - lo[1], hi[2] - lo[2])
        if ext < h
            h = ext
        end
    end
    return h
end

# CFL-controlled stable dt.
function _stable_dt(state::SodState)
    field = parent(state.af)
    s = _max_wave_speed(field, state.frame)
    h = _min_cell_extent(state.frame)
    s > 0.0 || return state.config.cfl * h
    return state.config.cfl * h / s
end

# ============================================================================
# Single time step
# ============================================================================

"""
    sod_step!(state::SodState, dt::Float64)

One forward-Euler finite-volume update with HLL fluxes. Writes
`U_new = U_old - dt/Vol * sum_faces(F · n * face_area)`.

The face pass accumulates `F·face_area·dt` into per-cell
`flux_div_*` buffers WITHOUT dividing by cell volume; the per-cell
update at the end then divides by Vᵢ. This factoring guarantees
exact (round-off) conservation: each face contributes `+F·δ` to one
cell and `-F·δ` to the other, so `Σᵢ flux_div_i = 0` independent of
the volume distribution.

Uses `Sequential()` on the flux pass to avoid a read-modify-write
race on the flux-divergence buffers. Same pattern as
`cfd_cell_advection`; a face-buffer scheme could parallelize this,
deferred to a follow-up.
"""
function sod_step!(state::SodState, dt::Float64)
    _resize_output_field!(state)
    field_in  = parent(state.af)
    frame     = state.frame
    mesh      = state.mesh
    bcs       = state.bcs

    fdr  = state.flux_div_rho
    fdru = state.flux_div_rhou
    fdrv = state.flux_div_rhov
    fdE  = state.flux_div_E

    # Zero the flux-divergence accumulators.
    fill!(fdr,  0.0)
    fill!(fdru, 0.0)
    fill!(fdrv, 0.0)
    fill!(fdE,  0.0)

    # Bind FieldViews for the orchestrator.
    fin_v = (rho  = field_in.rho,
              rhou = field_in.rhou,
              rhov = field_in.rhov,
              E    = field_in.E)
    fout_v = fin_v   # face pass reads only; writes go to flux-div buffers.

    flux_kernel = let dt = dt, frame = frame, fdr = fdr, fdru = fdru,
                       fdrv = fdrv, fdE = fdE
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
            scale = face_area * dt
            d1 = F[1] * scale
            d2 = F[2] * scale
            d3 = F[3] * scale
            d4 = F[4] * scale
            # `fdr[i]` accumulates the NET FLUX OUT of cell i. The
            # per-cell update at the end is `U_new = U_old - fdr/V`.
            # For face i→j with normal +n̂, flux F leaves i and enters
            # j: cell i's flux-out += +F·δ, cell j's flux-out -= F·δ.
            @inbounds fdr[i]  += d1
            @inbounds fdru[i] += d2
            @inbounds fdrv[i] += d3
            @inbounds fdE[i]  += d4
            @inbounds fdr[j]  -= d1
            @inbounds fdru[j] -= d2
            @inbounds fdrv[j] -= d3
            @inbounds fdE[j]  -= d4
            return nothing
        end
    end

    # Boundary kernel: dispatched by the orchestrator for every face
    # whose `face_neighbors` entry is 0 (i.e. anything not interior).
    # We branch on the BC kind:
    # - OUTFLOW (x-axes): zero-gradient zeroth-order extrapolation.
    #   The interior cell's state is mirrored to the ghost, so HLL
    #   collapses to the physical Euler flux. Mass / energy fluxes
    #   vanish when u·n=0 (the IC is at rest), but pressure-driven
    #   momentum flux is nonzero, so x-momentum drifts away through
    #   OUTFLOW: this is the physically correct behavior.
    # - PERIODIC (y-axes): handled in a SEPARATE manual pass (Step 3
    #   below) using `face_neighbors_with_bcs`, mirroring the
    #   `cfd_cell_advection` pattern. We skip it here.
    boundary_kernel = let dt = dt, frame = frame, fdr = fdr, fdru = fdru,
                          fdrv = fdrv, fdE = fdE
        function (cv, axis, side, normal, bcs_in, _ctx)
            # Skip periodic axes — their flux is handled in the manual
            # pass below.
            if bcs_in !== nothing && bcs_in.spec[axis][side] === PERIODIC
                return nothing
            end
            i = cv.index
            UI = (cv[Val(:rho)][1],
                   cv[Val(:rhou)][1],
                   cv[Val(:rhov)][1],
                   cv[Val(:E)][1])
            F = hll_flux(UI, UI, normal)
            face_area = _face_area_boundary(frame, i, axis)
            scale = face_area * dt
            # `normal` is the OUTWARD normal at this boundary, so F·δ
            # is the net flux leaving the cell (= +contribution to fdr).
            @inbounds fdr[i]  += F[1] * scale
            @inbounds fdru[i] += F[2] * scale
            @inbounds fdrv[i] += F[3] * scale
            @inbounds fdE[i]  += F[4] * scale
            return nothing
        end
    end

    for_each_face!(flux_kernel, fout_v, fin_v, frame;
                   bcs = bcs,
                   flux_kernel_boundary = boundary_kernel,
                   backend = Sequential())

    # Step 3 (PERIODIC y-axis): the orchestrator's interior list
    # excludes domain-boundary faces (whose `face_neighbors` is 0); for
    # PERIODIC y-axes we resolve those via `face_neighbors_with_bcs`
    # and dispatch each periodic pair exactly once (visit from the lo
    # side only).
    ensure_neighbor_graph!(mesh)
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        nbs    = face_neighbors(mesh, i)
        nbs_bc = face_neighbors_with_bcs(mesh, i, bcs)
        for axis in 1:2
            # Only PERIODIC axes are wired through `face_neighbors_with_bcs`.
            (axis == 2) || continue   # only y is periodic in this run
            lo_face = 2 * axis - 1
            if nbs[lo_face] == 0 && nbs_bc[lo_face] != 0
                j = Int(nbs_bc[lo_face])
                # Lo cell is `i` (lo wall); partner is `j` (hi wall).
                # Dispatch with j as "left" and i as "right" so the
                # +axis normal points from j to i (matches the
                # interior-face convention).
                normal = (0.0, 1.0)
                UL = _read_U(field_in, j)
                UR = _read_U(field_in, i)
                face_area = _face_area(frame, j, i, axis)
                F = hll_flux(UL, UR, normal)
                scale = face_area * dt
                # Same sign convention as the interior face pass:
                # `fdr` accumulates net flux OUT of each cell. With
                # normal pointing from j to i, flux F leaves cell j
                # and enters cell i.
                @inbounds fdr[j]  += F[1] * scale
                @inbounds fdru[j] += F[2] * scale
                @inbounds fdrv[j] += F[3] * scale
                @inbounds fdE[j]  += F[4] * scale
                @inbounds fdr[i]  -= F[1] * scale
                @inbounds fdru[i] -= F[2] * scale
                @inbounds fdrv[i] -= F[3] * scale
                @inbounds fdE[i]  -= F[4] * scale
            end
        end
    end

    # Per-cell update: U_new = U_old - flux_div / V.
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        Vi = _cell_volume(frame, i)
        invV = 1.0 / Vi
        field_in.rho[i]  = (field_in.rho[i][1]  - fdr[i]  * invV,)
        field_in.rhou[i] = (field_in.rhou[i][1] - fdru[i] * invV,)
        field_in.rhov[i] = (field_in.rhov[i][1] - fdrv[i] * invV,)
        field_in.E[i]    = (field_in.E[i][1]    - fdE[i]  * invV,)
    end

    return state
end

# Boundary face area: the off-axis extent of cell `i` (= face area in 2D).
@inline function _face_area_boundary(frame::EulerianFrame{2, Float64},
                                       i::Int, axis::Int)
    lo, hi = cell_physical_box(frame, i)
    off = axis == 1 ? 2 : 1
    return hi[off] - lo[off]
end

# ============================================================================
# Refinement indicator (gradient-magnitude proxy)
# ============================================================================

"""
    gradient_indicator(field, mesh) -> Vector{Float64}

Per-cell `|Δρ|/ρ` as a finite-difference proxy on neighbors. Cell `i`
gets the maximum absolute relative density difference among its leaf
face-neighbors. This captures the shock and contact discontinuity (both
have steep ρ jumps) and is cheap to evaluate.
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

# ============================================================================
# Exact Riemann solution (Toro Algorithm 4.1)
#
# This is a textbook implementation: solve for the star-region pressure
# `p*` by Newton iteration on the wave-strength function `f(p) = fL(p) + fR(p)`
# where `fK` is the rarefaction or shock relation for the K-th wave. Then
# sample (x, t) by classifying which wave region the ray `x/t` falls into.
# ============================================================================

# The wave-strength function and its derivative for one side (Toro eq.
# 4.6-4.7). `K = L` or `R`; rarefaction if p ≤ pK, shock otherwise.
@inline function _f_K(p::Float64, ρK::Float64, pK::Float64, cK::Float64)
    if p > pK
        # Shock branch.
        AK = 2.0 / ((GAMMA + 1.0) * ρK)
        BK = (GAMMA - 1.0) / (GAMMA + 1.0) * pK
        return (p - pK) * sqrt(AK / (p + BK))
    else
        # Rarefaction branch.
        return 2.0 * cK / (GAMMA - 1.0) *
               ((p / pK) ^ ((GAMMA - 1.0) / (2.0 * GAMMA)) - 1.0)
    end
end

@inline function _df_K(p::Float64, ρK::Float64, pK::Float64, cK::Float64)
    if p > pK
        AK = 2.0 / ((GAMMA + 1.0) * ρK)
        BK = (GAMMA - 1.0) / (GAMMA + 1.0) * pK
        return sqrt(AK / (BK + p)) * (1.0 - 0.5 * (p - pK) / (BK + p))
    else
        return 1.0 / (ρK * cK) * (p / pK) ^ (-(GAMMA + 1.0) / (2.0 * GAMMA))
    end
end

# Initial guess: two-rarefaction approximation (Toro §4.5.1).
@inline function _p_star_guess(ρL, uL, pL, cL, ρR, uR, pR, cR)
    pPV = max(0.5 * (pL + pR) - 0.125 * (uR - uL) * (ρL + ρR) * (cL + cR), 1e-6)
    return pPV
end

# Newton solve for p* (Toro Algorithm 4.1).
function _solve_p_star(ρL::Float64, uL::Float64, pL::Float64, cL::Float64,
                        ρR::Float64, uR::Float64, pR::Float64, cR::Float64)
    p = _p_star_guess(ρL, uL, pL, cL, ρR, uR, pR, cR)
    Δu = uR - uL
    tol = 1e-10
    @inbounds for _ in 1:50
        f  = _f_K(p, ρL, pL, cL) + _f_K(p, ρR, pR, cR) + Δu
        df = _df_K(p, ρL, pL, cL) + _df_K(p, ρR, pR, cR)
        Δp = f / df
        p_new = p - Δp
        if p_new < tol
            p_new = 0.5 * p
        end
        if abs(Δp) / (0.5 * (p + p_new + tol)) < tol
            return p_new
        end
        p = p_new
    end
    return p
end

"""
    exact_riemann_at(x::Float64, t::Float64) -> (ρ, u, v, p)

Sample the analytic Sod-tube Riemann solution at physical position `x`
(the y-coordinate is irrelevant; the IC is x-only) at time `t`. Returns
the primitive-variable 4-tuple `(ρ, u, v=0, p)`. Cf. Toro §4.4-4.5.

The function returns the post-rarefaction or post-shock state by
classifying `s = (x - 0.5) / t` against the wave speeds.
"""
function exact_riemann_at(x::Float64, t::Float64)
    if t <= 0.0
        ρ, u, v, p = x < SOD_INTERFACE_X ? SOD_LEFT_PRIM : SOD_RIGHT_PRIM
        return (ρ, u, v, p)
    end
    ρL, uL, _, pL = SOD_LEFT_PRIM
    ρR, uR, _, pR = SOD_RIGHT_PRIM
    cL = sound_speed(ρL, pL)
    cR = sound_speed(ρR, pR)

    p_star = _solve_p_star(ρL, uL, pL, cL, ρR, uR, pR, cR)
    u_star = 0.5 * (uL + uR) +
             0.5 * (_f_K(p_star, ρR, pR, cR) - _f_K(p_star, ρL, pL, cL))

    s = (x - SOD_INTERFACE_X) / t

    if s <= u_star
        # Left wave (rarefaction or shock).
        if p_star > pL
            # Left shock.
            SL = uL - cL * sqrt((GAMMA + 1.0) / (2.0 * GAMMA) * p_star / pL +
                                  (GAMMA - 1.0) / (2.0 * GAMMA))
            if s <= SL
                return (ρL, uL, 0.0, pL)
            else
                ρ_starL = ρL * ((p_star / pL +
                                  (GAMMA - 1.0) / (GAMMA + 1.0)) /
                                 ((GAMMA - 1.0) / (GAMMA + 1.0) *
                                  p_star / pL + 1.0))
                return (ρ_starL, u_star, 0.0, p_star)
            end
        else
            # Left rarefaction (head + tail).
            c_starL = cL * (p_star / pL) ^ ((GAMMA - 1.0) / (2.0 * GAMMA))
            SHL = uL - cL
            STL = u_star - c_starL
            if s <= SHL
                return (ρL, uL, 0.0, pL)
            elseif s <= STL
                # Inside the rarefaction fan.
                fac1 = 2.0 / (GAMMA + 1.0) +
                       (GAMMA - 1.0) / ((GAMMA + 1.0) * cL) * (uL - s)
                ρ_fan = ρL * fac1 ^ (2.0 / (GAMMA - 1.0))
                u_fan = 2.0 / (GAMMA + 1.0) *
                        (cL + (GAMMA - 1.0) / 2.0 * uL + s)
                p_fan = pL * fac1 ^ (2.0 * GAMMA / (GAMMA - 1.0))
                return (ρ_fan, u_fan, 0.0, p_fan)
            else
                ρ_starL = ρL * (p_star / pL) ^ (1.0 / GAMMA)
                return (ρ_starL, u_star, 0.0, p_star)
            end
        end
    else
        # Right wave (rarefaction or shock).
        if p_star > pR
            # Right shock.
            SR = uR + cR * sqrt((GAMMA + 1.0) / (2.0 * GAMMA) * p_star / pR +
                                  (GAMMA - 1.0) / (2.0 * GAMMA))
            if s >= SR
                return (ρR, uR, 0.0, pR)
            else
                ρ_starR = ρR * ((p_star / pR +
                                  (GAMMA - 1.0) / (GAMMA + 1.0)) /
                                 ((GAMMA - 1.0) / (GAMMA + 1.0) *
                                  p_star / pR + 1.0))
                return (ρ_starR, u_star, 0.0, p_star)
            end
        else
            # Right rarefaction (head + tail).
            c_starR = cR * (p_star / pR) ^ ((GAMMA - 1.0) / (2.0 * GAMMA))
            SHR = uR + cR
            STR = u_star + c_starR
            if s >= SHR
                return (ρR, uR, 0.0, pR)
            elseif s >= STR
                fac1 = 2.0 / (GAMMA + 1.0) -
                       (GAMMA - 1.0) / ((GAMMA + 1.0) * cR) * (uR - s)
                ρ_fan = ρR * fac1 ^ (2.0 / (GAMMA - 1.0))
                u_fan = 2.0 / (GAMMA + 1.0) *
                        (-cR + (GAMMA - 1.0) / 2.0 * uR + s)
                p_fan = pR * fac1 ^ (2.0 * GAMMA / (GAMMA - 1.0))
                return (ρ_fan, u_fan, 0.0, p_fan)
            else
                ρ_starR = ρR * (p_star / pR) ^ (1.0 / GAMMA)
                return (ρ_starR, u_star, 0.0, p_star)
            end
        end
    end
end

# ============================================================================
# Diagnostics: shock position + L¹ error
# ============================================================================

# Estimate the post-shock density-jump location. The Sod tube at t=0.2
# has a moving right shock at x ≈ 0.85, with ρ dropping from the star-
# state plateau to the right pre-shock value. We pick the RIGHT-MOST
# cell whose +x density drop exceeds half the maximum drop on the
# right-half — this rejects the rarefaction's smooth gradient and
# identifies the steep shock front.
function _shock_position(field, frame::EulerianFrame{2, Float64})
    mesh = frame.mesh
    ensure_neighbor_graph!(mesh)

    # Pass 1: find the maximum +x density drop on the right half.
    max_jump = 0.0
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        lo, hi = cell_physical_box(frame, i)
        cx = 0.5 * (lo[1] + hi[1])
        cx > 0.5 || continue
        nbs = face_neighbors(mesh, i)
        right_nb = nbs[2]
        right_nb == 0 && continue
        ρ_i = field.rho[i][1]
        ρ_j = field.rho[Int(right_nb)][1]
        jump = ρ_i - ρ_j
        if jump > max_jump
            max_jump = jump
        end
    end

    # Pass 2: scan right-to-left and return the first cell with a
    # +x density drop ≥ 0.5 · max_jump. This is the steep shock front
    # rather than any of the smoother rarefaction-tail cells.
    threshold = 0.5 * max_jump
    x_shock = NaN
    best_cx = -Inf
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        lo, hi = cell_physical_box(frame, i)
        cx = 0.5 * (lo[1] + hi[1])
        cx > 0.5 || continue
        nbs = face_neighbors(mesh, i)
        right_nb = nbs[2]
        right_nb == 0 && continue
        ρ_i = field.rho[i][1]
        ρ_j = field.rho[Int(right_nb)][1]
        jump = ρ_i - ρ_j
        if jump >= threshold && cx > best_cx
            best_cx = cx
            x_shock = cx
        end
    end
    return x_shock
end

# L¹ error per unit volume against the analytic Riemann solution.
# Compares the cell-mean density to the pointwise analytic density at
# the cell centroid (a cell-centered single-point quadrature). The
# analytic solution is x-only; integrating over y is exact.
function _l1_error(field, frame::EulerianFrame{2, Float64}, t::Float64)
    err = 0.0
    domain_vol = 1.0 * 0.1
    @inbounds for i in 1:n_cells(frame.mesh)
        is_leaf(frame.mesh.cells[i]) || continue
        lo, hi = cell_physical_box(frame, i)
        cx = 0.5 * (lo[1] + hi[1])
        ρ_a, _, _, _ = exact_riemann_at(cx, t)
        cell_val = field.rho[i][1]
        cell_vol = (hi[1] - lo[1]) * (hi[2] - lo[2])
        err += abs(cell_val - ρ_a) * cell_vol
    end
    return err / domain_vol
end

# ============================================================================
# Top-level run
# ============================================================================

"""
    run!(config::SodConfig) -> (state, info)

Build a fresh state and integrate to `config.t_final` with adaptive CFL
time-stepping. Optionally fires AMR every `config.amr_every` steps.

Returns:
- `state::SodState` — the final state.
- `info::NamedTuple` with keys:
    * `mass_drift`     — `final_mass - initial_mass` (signed).
    * `energy_drift`   — `final_energy - initial_energy` (signed).
    * `momentum_drift` — `final_momentum_x - initial_momentum_x`.
                          OUTFLOW BCs allow x-momentum drift; we report
                          it for diagnostics.
    * `final_mass`, `final_energy`, `final_momentum_x` — totals.
    * `initial_mass`, `initial_energy` — initial totals.
    * `t_final`         — the simulation time reached.
    * `n_steps`         — number of forward-Euler steps taken.
    * `n_cells_final`   — final mesh cell count.
    * `n_leaves_final`  — final leaf count.
    * `n_leaves_initial` — initial leaf count (for AMR-engages tests).
    * `shock_position`  — estimated x-coordinate of the right-going
                          shock (peak +x density gradient on the right
                          half).
    * `l1_error`        — L¹ error vs the analytic Riemann ρ at `t_final`.
"""
function run!(config::SodConfig)
    state = build_state(config)

    # Indicator closure: snapshots the live AdaptiveField each call.
    indicator = let state = state
        m -> gradient_indicator(parent(state.af), m)
    end

    # The AMR driver expects step_fn(state, frame) to advance one step.
    # We compute dt inside the closure (so it's adaptive every step).
    step_fn = let state = state
        function (st, fr)
            dt = _stable_dt(st)
            # Don't overshoot t_final.
            dt = min(dt, st.config.t_final - st.t)
            sod_step!(st, dt)
            st.t += dt
            st.n_steps += 1
            return nothing
        end
    end

    # Drive the loop until `t_final`. We can't predict n_steps in advance
    # because dt is adaptive, so we hand the AMR driver a 1-step window
    # and loop ourselves. (`step_with_amr!` fires AMR on iteration k where
    # k % hysteresis_steps == 0.)
    safety_max_steps = 100_000
    if config.amr_every > 0
        local_step = 0
        while state.t < config.t_final && local_step < safety_max_steps
            step_with_amr!(state, state.frame, step_fn, indicator,
                            1;
                            refine_threshold = config.refine_threshold,
                            coarsen_threshold = config.coarsen_threshold,
                            hysteresis_steps =
                                local_step % config.amr_every == 0 ? 1 : 0,
                            max_level = config.max_level,
                            isotropic = true)
            local_step += 1
        end
    else
        while state.t < config.t_final && state.n_steps < safety_max_steps
            step_fn(state, state.frame)
        end
    end

    # Final diagnostics.
    field = parent(state.af)
    m, px, py, e = _conserved_totals(field, state.frame)
    shock_x = _shock_position(field, state.frame)
    l1 = _l1_error(field, state.frame, state.t)

    # Repurpose RemapDiagnostics as a small summary container.
    reset!(state.diagnostics)
    state.diagnostics.total_volume_in  = state.initial_mass
    state.diagnostics.total_volume_out = m
    @inbounds for i in 1:n_cells(state.mesh)
        is_leaf(state.mesh.cells[i]) || continue
        ρ = field.rho[i][1]
        state.diagnostics.liouville_min = min(state.diagnostics.liouville_min, ρ)
        state.diagnostics.liouville_max = max(state.diagnostics.liouville_max, ρ)
    end

    info = (
        mass_drift       = m  - state.initial_mass,
        energy_drift     = e  - state.initial_energy,
        momentum_drift   = px - state.initial_momentum_x,
        final_mass       = m,
        final_energy     = e,
        final_momentum_x = px,
        initial_mass     = state.initial_mass,
        initial_energy   = state.initial_energy,
        t_final          = state.t,
        n_steps          = state.n_steps,
        n_cells_final    = n_cells(state.mesh),
        n_leaves_final   = length(enumerate_leaves(state.mesh)),
        n_leaves_initial = state.n_leaves_initial,
        shock_position   = shock_x,
        l1_error         = l1,
    )
    return state, info
end

end # module CFDCompressibleSod
