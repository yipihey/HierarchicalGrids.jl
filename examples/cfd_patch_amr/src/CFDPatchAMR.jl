"""
    CFDPatchAMR

A 2D scalar-advection demonstration of the Berger-Oliger style
patch-based AMR stack provided by `HierarchicalGrids.PatchHierarchy`.

# Physics

Linear advection on the unit square `[0, 1]^2` with constant velocity
`u = (0.5, 0.3)` and periodic boundary conditions. A Gaussian peak
initialised at `(0.3, 0.3)` is transported by the flow; after one
period (`t = 1`) it returns to its starting position (the velocity is
chosen so each component's period over the unit-cell is 1).

# AMR pipeline

A two-level `PatchHierarchy{2, Float64}`:

- Level 1: a uniform base patch covering `[0, 1]^2` (16x16 = 256 leaves
  by default).
- Level 2: a single rectangular patch of half-extent
  `config.fine_box_size` centred on the current peak (cell-aligned to
  the coarse grid; clamped to the domain). Re-created every
  `config.refresh_every` steps to follow the moving feature.

Per time step (see `step!`):

1. Update level 1 with `for_each_patch!(level=1)`.
2. `prolong_from_parents!` → fill level 2 from level 1.
3. Update level 2 with `for_each_patch!(level=2)` (parent halos via
   `PatchBoundaryBC`).
4. `restrict_to_parents!` → push volume-weighted fine averages back to
   the covered coarse cells.

The numerics are first-order upwind finite-volume on degree-0
polynomials (cell averages). On a uniform periodic grid the coarse-
level update is exactly mass-conservative; the fine-level update is
locally conservative on the fine mesh interior but reads parent
values at the patch boundary (a known Berger-Oliger refluxing gap —
see `README.md`).

# Public API

- `PatchAMRConfig` — solver configuration.
- `run!(config)` — runs to completion; returns `(state, diagnostics)`.

The returned `Diagnostics` carries mass drift (vs initial total mass)
and the L∞ peak-position error vs the analytic moving-Gaussian centre.
"""
module CFDPatchAMR

using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells,
    EulerianFrame, FrameBoundaries, enumerate_leaves,
    BCKind, PERIODIC, is_leaf, cell_physical_box,
    MonomialBasis, n_coeffs, allocate_polynomial_fields, SoA,
    Sequential,
    PatchHierarchy, PatchBoundaryBC, PatchView, PatchHaloView,
    add_patches!, for_each_patch!,
    restrict_to_parents!, prolong_from_parents!

export PatchAMRConfig, Diagnostics, State
export run!, build_initial_state, step!
export gaussian_ic, analytic_peak_position

# ============================================================================
# Configuration
# ============================================================================

"""
    PatchAMRConfig

Numerical-experiment knobs for the patch-AMR advection demo.

# Fields

- `velocity::NTuple{2, Float64}` — constant flow velocity. Both
  components must be ≥ 0 for the upwind kernel as written.
- `dt::Float64` — time step. Should respect the CFL bound on the
  finer of the two patches: `dt * (|ux|/dx + |uy|/dy) ≤ 1`.
- `n_steps::Int` — number of steps to run.
- `n_levels::Int` — patch-hierarchy depth (2 supported; 3 is
  documented but not shipped).
- `base_refines::Int` — uniform refinement count for the base patch.
  `levels=k` produces a `2^k × 2^k` leaf grid on `[0, 1]^2`.
- `fine_refines::Int` — additional refinements per coarse cell on
  the fine patch. The fine patch ends up with `2^fine_refines` fine
  cells per coarse cell per axis (and total
  `(2^(fine_refines + log2(n_coarse_cells_per_axis)))^2` leaves).
- `fine_box_size::Float64` — half-extent of the level-2 patch in
  physical units. The actual box width is rounded up to a power-of-2
  multiple of the coarse cell size so the fine patch exactly tiles
  whole coarse cells (see `fine_patch_box`).
- `refresh_every::Int` — how often (in steps) to re-create the level-2
  patch around the current peak. `0` disables refresh (patch stays
  put).
- `peak_init::NTuple{2, Float64}` — initial Gaussian centre.
- `gauss_sigma::Float64` — Gaussian standard deviation.
"""
struct PatchAMRConfig
    velocity::NTuple{2, Float64}
    dt::Float64
    n_steps::Int
    n_levels::Int
    base_refines::Int
    fine_refines::Int
    fine_box_size::Float64
    refresh_every::Int
    peak_init::NTuple{2, Float64}
    gauss_sigma::Float64
end

function PatchAMRConfig(; velocity = (0.5, 0.3),
                          dt = 0.02,
                          n_steps = 50,
                          n_levels = 2,
                          base_refines = 4,
                          fine_refines = 1,
                          fine_box_size = 0.2,
                          refresh_every = 5,
                          peak_init = (0.3, 0.3),
                          gauss_sigma = 0.06)
    n_levels in (2,) ||
        throw(ArgumentError("PatchAMRConfig: only n_levels=2 is shipped " *
                            "(see README for extending to 3)"))
    velocity[1] >= 0 && velocity[2] >= 0 ||
        throw(ArgumentError("PatchAMRConfig: velocity components must be ≥ 0 " *
                            "(upwind kernel assumes positive flow)"))
    dt > 0 || throw(ArgumentError("dt must be > 0"))
    base_refines >= 1 || throw(ArgumentError("base_refines must be ≥ 1"))
    fine_refines >= 1 || throw(ArgumentError("fine_refines must be ≥ 1"))
    fine_box_size > 0 || throw(ArgumentError("fine_box_size must be > 0"))
    return PatchAMRConfig(velocity, dt, n_steps, n_levels,
                           base_refines, fine_refines, fine_box_size,
                           refresh_every, peak_init, gauss_sigma)
end

# ============================================================================
# Initial condition + analytic reference
# ============================================================================

"""
    gaussian_ic(x, peak, sigma) -> Float64

Unit-amplitude 2D Gaussian centred at `peak` with isotropic stddev
`sigma`. Used as the initial condition.
"""
@inline function gaussian_ic(x::NTuple{2, Float64},
                              peak::NTuple{2, Float64},
                              sigma::Float64)
    dx = x[1] - peak[1]
    dy = x[2] - peak[2]
    return exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
end

"""
    analytic_peak_position(config, t) -> NTuple{2, Float64}

Closed-form peak position at time `t` under periodic linear advection:
`peak(t) = (peak_init + velocity * t) mod 1`.
"""
@inline function analytic_peak_position(config::PatchAMRConfig, t::Float64)
    p = config.peak_init
    u = config.velocity
    return (mod(p[1] + u[1] * t, 1.0),
            mod(p[2] + u[2] * t, 1.0))
end

# ============================================================================
# State container
# ============================================================================

"""
    State

Bundle of mutable per-step state owned by the solver: the patch
hierarchy, double-buffered field-sets per patch (input/output), and
per-step bookkeeping.

Fields are private; the orchestration in `step!`/`run!` is the
public interface.
"""
mutable struct State
    config::PatchAMRConfig
    ph::PatchHierarchy{2, Float64}

    # Per-patch field-sets: index 1 = base, index 2 = current fine.
    # Field layout is SoA, single :rho field, MonomialBasis{2, 0}.
    coarse_in::Any
    coarse_out::Any
    fine_in::Any
    fine_out::Any

    # Cached views (NamedTuple{(:rho,)}) over the field-sets above.
    coarse_in_v::NamedTuple
    coarse_out_v::NamedTuple
    fine_in_v::NamedTuple
    fine_out_v::NamedTuple

    # Periodic outer-domain BC (assigned to the hierarchy).
    bcs::FrameBoundaries{2}

    # Time and step.
    t::Float64
    step_index::Int
end

# ============================================================================
# Mesh / patch construction helpers
# ============================================================================

# Build a uniform 2D HierarchicalMesh by repeated full-tree refinement.
function _uniform_2d_mesh(refines::Int)
    mesh = HierarchicalMesh{2}()
    for _ in 1:refines
        leaves = enumerate_leaves(mesh)
        refine_cells!(mesh, leaves)
    end
    return mesh
end

# Build a uniform-refined EulerianFrame on the box [lo, hi].
function _uniform_2d_frame(refines::Int,
                            lo::NTuple{2, Float64},
                            hi::NTuple{2, Float64})
    mesh = _uniform_2d_mesh(refines)
    return EulerianFrame(mesh, lo, hi)
end

# Cell-align (snap) a value to the nearest multiple of `dx_coarse`.
@inline _snap(x, dx) = round(x / dx) * dx

# Integer log2 of a power-of-2 positive integer. Returns -1 if not a
# power of 2 (caller is expected to pass a verified power of 2).
@inline function _ilog2(n::Integer)
    n > 0 || throw(ArgumentError("_ilog2: n must be positive"))
    (n & (n - 1)) == 0 || throw(ArgumentError("_ilog2: $n is not a power of 2"))
    k = 0
    while n > 1
        n >>= 1
        k += 1
    end
    return k
end

"""
    fine_patch_box(peak, half_size, dx_coarse) -> (lo, hi, n_coarse)

Compute a coarse-cell-aligned bounding box around `peak` whose width
is a power-of-2 number of coarse cells. The returned `n_coarse` is
that number per axis (same on both axes).

The box is then `lo .+ n_coarse * dx_coarse` per axis, snapped to a
multiple of `dx_coarse` so the box endpoints lie on coarse-cell
boundaries — this is what makes `restrict_to_parents!` recover the
fine patch's mass exactly. Box is clamped to `[0, 1]^2`; if the peak
sits too close to a boundary, the box may be off-centre.
"""
function fine_patch_box(peak::NTuple{2, Float64}, half_size::Float64,
                          dx_coarse::Float64)
    # Total number of coarse cells per axis on the unit interval.
    n_total = round(Int, 1 / dx_coarse)
    # Desired box width in coarse cells (rounded up to a power of 2).
    n_target = max(1, ceil(Int, 2 * half_size / dx_coarse))
    n_pow2 = 1
    while n_pow2 < n_target
        n_pow2 *= 2
    end
    n_pow2 = min(n_pow2, n_total)   # don't exceed the domain

    # Centre on the coarse cell that contains the peak; then shift so
    # that the n_pow2 × n_pow2 block lies inside the domain.
    function _axis(p::Float64)
        # Coarse cell index (0-based) containing the peak.
        ci = clamp(floor(Int, p / dx_coarse), 0, n_total - 1)
        # Half-block in cells (round down so the block is centred on
        # the peak's cell when n_pow2 is even).
        half = n_pow2 ÷ 2
        # Initial lo cell index: ci - half + 1 (so that ci is roughly mid).
        lo_cell = ci - half + 1
        # Clamp.
        if lo_cell < 0
            lo_cell = 0
        end
        if lo_cell + n_pow2 > n_total
            lo_cell = n_total - n_pow2
        end
        return (lo_cell * dx_coarse, (lo_cell + n_pow2) * dx_coarse)
    end
    lo_x, hi_x = _axis(peak[1])
    lo_y, hi_y = _axis(peak[2])
    return ((lo_x, lo_y), (hi_x, hi_y), n_pow2)
end

# Allocate a degree-0 SoA field-set with a single :rho scalar.
function _allocate_field(n_cells_total::Int)
    basis = MonomialBasis{2, 0}()
    return allocate_polynomial_fields(SoA(), basis, n_cells_total; rho = Float64)
end

# Zero every coefficient of the field-set.
function _zero!(pfs)
    nc = n_coeffs(pfs.basis)
    field = pfs.rho
    for i in 1:length(field)
        field[i] = ntuple(_ -> 0.0, nc)
    end
    return pfs
end

# Read coarse cell extent from a uniform-refined frame.
function _coarse_dx(config::PatchAMRConfig)
    return 1.0 / (1 << config.base_refines)
end

# Build the wrapped views (NamedTuple{(:rho,)}) over a field-set.
@inline _rho_views(pfs) = (rho = pfs.rho,)

# ============================================================================
# Initial state
# ============================================================================

"""
    build_initial_state(config) -> State

Build the patch hierarchy + field-sets and project the Gaussian IC
onto every leaf cell of every patch (cell-average via centre-point
evaluation; quadrature is unnecessary for a smooth IC sampled at
degree 0).
"""
function build_initial_state(config::PatchAMRConfig)
    bcs = FrameBoundaries(((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)))

    # Level 1: base patch covering the full domain.
    base_frame = _uniform_2d_frame(config.base_refines, (0.0, 0.0), (1.0, 1.0))
    base_in  = _zero!(_allocate_field(n_cells(base_frame.mesh)))
    base_out = _zero!(_allocate_field(n_cells(base_frame.mesh)))

    # Initialise base coarse cells: cell-centre evaluation of the
    # Gaussian gives the cell-average to second-order in dx, which is
    # plenty for a smoke test.
    _project_gaussian!(base_in, base_frame, config)

    ph = PatchHierarchy(base_frame; physical_bcs = bcs)

    # Level 2: first fine patch around peak.
    dx_c = _coarse_dx(config)
    flo, fhi, n_pow2 = fine_patch_box(config.peak_init, config.fine_box_size, dx_c)
    total_refines = config.fine_refines + _ilog2(n_pow2)
    fine_frame = _uniform_2d_frame(total_refines, flo, fhi)
    fine_in  = _zero!(_allocate_field(n_cells(fine_frame.mesh)))
    fine_out = _zero!(_allocate_field(n_cells(fine_frame.mesh)))
    _project_gaussian!(fine_in, fine_frame, config)
    add_patches!(ph, 2, [fine_frame])

    return State(config, ph,
                  base_in, base_out, fine_in, fine_out,
                  _rho_views(base_in),  _rho_views(base_out),
                  _rho_views(fine_in),  _rho_views(fine_out),
                  bcs,
                  0.0, 0)
end

# Cell-centre projection of the Gaussian IC.
function _project_gaussian!(pfs, frame::EulerianFrame{2, Float64},
                              config::PatchAMRConfig)
    nc = n_coeffs(pfs.basis)
    field = pfs.rho
    @inbounds for ci in 1:n_cells(frame.mesh)
        if !is_leaf(frame.mesh.cells[ci])
            continue
        end
        lo, hi = cell_physical_box(frame, ci)
        cx = (lo[1] + hi[1]) / 2
        cy = (lo[2] + hi[2]) / 2
        v  = gaussian_ic((cx, cy), config.peak_init, config.gauss_sigma)
        field[ci] = ntuple(k -> k == 1 ? v : 0.0, nc)
    end
    return pfs
end

# ============================================================================
# Advection kernel (degree-0 upwind FV)
# ============================================================================

# Small struct passed via `ctx` so the kernel stays a pure top-level
# function that the closure-converter can compile cleanly.
struct AdvectCtx
    ux::Float64
    uy::Float64
    dt::Float64
end

"""
    advect_patch_kernel(pv, hv, ctx::AdvectCtx)

First-order upwind finite-volume update on a degree-0 polynomial
field. Reads `:rho` from the central cell (`pv[:rho]`) and from the
upwind neighbours via the halo view (`hv[:rho, off]`); writes the
updated cell average to `pv[:rho]`.

Only the constant coefficient (index 1) is touched. The kernel works
identically on level-1 and level-2 patches: at level 2 the halo view
falls through to parent values when offsets walk off the patch.
"""
function advect_patch_kernel(pv::PatchView, hv::PatchHaloView,
                                ctx::AdvectCtx)
    # Cell extent on this patch (uniform leaves → vol = dx*dy is per-cell).
    # We need dx and dy separately for the upwind flux; recover them from
    # the cell's metadata. Each leaf has identical extents on a uniform
    # patch, so we derive dx, dy from `pv.volume` and the aspect ratio.
    # On a patch built from a square HierarchicalMesh{2} with rectangular
    # frame box, dx may differ from dy if the box is not square. Instead
    # of guessing, read it from the inner CellView's coords + nbrs:
    # cleanest is to ask the patch directly. We approximate from
    # `volume = dx * dy` and `dx == dy` only for square cells. To be safe
    # we recover dx, dy from the patch frame stored in the halo view.
    frame = hv.frame
    # Frame extent / leaves-per-axis. The mesh is a uniform-refined
    # quadtree, so leaves-per-axis = 2^level on a square-root basis. The
    # cleanest portable formula uses the cell's physical box.
    p_lo, p_hi = cell_physical_box(frame, pv.cv.index)
    dx = p_hi[1] - p_lo[1]
    dy = p_hi[2] - p_lo[2]

    rho_c = pv[:rho][1]
    # Upwind reads (positive velocity → look at -x and -y neighbours).
    rho_xm = hv[:rho, (-1, 0)]
    rho_ym = hv[:rho, (0, -1)]
    # `rho_xm`/`rho_ym` are PolynomialView-likes with [1] = constant.
    rho_left   = rho_xm === nothing ? rho_c : rho_xm[1]
    rho_bottom = rho_ym === nothing ? rho_c : rho_ym[1]

    a_x = ctx.ux * ctx.dt / dx
    a_y = ctx.uy * ctx.dt / dy
    # Conservative upwind:
    #   rho_new = rho - a_x*(rho - rho_left) - a_y*(rho - rho_bottom)
    rho_new = rho_c - a_x * (rho_c - rho_left) - a_y * (rho_c - rho_bottom)
    pv[:rho] = (rho_new,)
    return nothing
end

# ============================================================================
# Per-step orchestration
# ============================================================================

# Swap the in/out halves of a State's coarse OR fine field-set after a
# kernel invocation, so the next read sees the just-written data.
function _swap_coarse!(state::State)
    state.coarse_in,   state.coarse_out   = state.coarse_out,   state.coarse_in
    state.coarse_in_v, state.coarse_out_v = state.coarse_out_v, state.coarse_in_v
    return state
end

function _swap_fine!(state::State)
    state.fine_in,   state.fine_out   = state.fine_out,   state.fine_in
    state.fine_in_v, state.fine_out_v = state.fine_out_v, state.fine_in_v
    return state
end

"""
    step!(state) -> state

Advance the solution by one time step using the patch-AMR pipeline
(prolong → fine update → coarse update → restrict). Mutates `state`
in place; returns it.

The prolong/fine-update happen *before* the coarse update so that
the fine boundary's halo reads use pre-step coarse values. The coarse
update then runs on the full domain (covered cells will be
overwritten by the subsequent restrict). With this ordering the
cross-level flux mismatch is bounded by O(dt²) per step rather than
O(dt) per step (the latter is what naïve "coarse first" produces);
see `README.md` for the conservation analysis.
"""
function step!(state::State)
    cfg = state.config
    ctx = AdvectCtx(cfg.velocity[1], cfg.velocity[2], cfg.dt)

    # Stage 1: prolong pre-step coarse → fine. The fine patch's
    # interior is filled from the pre-step coarse field; the fine
    # boundary halos in stage 2 will read pre-step coarse values too.
    prolong_from_parents!([state.fine_in_v], [state.coarse_in_v],
                           state.ph; level = 2, fieldname = :rho)

    # Stage 2: update fine (level 2) using pre-step coarse parent halos.
    for_each_patch!(advect_patch_kernel,
                     [state.fine_out_v], [state.fine_in_v],
                     state.ph; level = 2, ghost_depth = 1,
                     fields_in_parent = [state.coarse_in_v],
                     ctx = ctx, backend = Sequential())
    _swap_fine!(state)

    # Stage 3: update coarse (level 1) on the full domain. Mass-flux
    # exchange between covered and uncovered coarse cells uses the
    # same pre-step covered values that the fine update saw at its
    # boundary, so the cross-level flux balance is consistent.
    for_each_patch!(advect_patch_kernel,
                     [state.coarse_out_v], [state.coarse_in_v],
                     state.ph; level = 1, ghost_depth = 1,
                     ctx = ctx, backend = Sequential())
    _swap_coarse!(state)

    # Stage 4: restrict fine → coarse (volume-weighted average) on
    # cells covered by the fine patch. This overwrites the covered
    # coarse cells' post-coarse values with the volume-weighted fine
    # averages.
    restrict_to_parents!([state.coarse_in_v], [state.fine_in_v],
                          state.ph; level = 2, fieldname = :rho)

    state.t += cfg.dt
    state.step_index += 1

    # Optional: re-create the fine patch around the current peak.
    if cfg.refresh_every > 0 && state.step_index % cfg.refresh_every == 0
        _refresh_fine_patch!(state)
    end
    return state
end

# Locate the current peak by scanning the *fine* patch's cell-average
# field for its maximum cell.
function _current_peak(state::State)
    fine_frame = state.ph.levels[2][1]
    field = state.fine_in.rho
    best_v = -Inf
    best_pos = (0.0, 0.0)
    @inbounds for ci in 1:n_cells(fine_frame.mesh)
        is_leaf(fine_frame.mesh.cells[ci]) || continue
        v = field[ci][1]
        if v > best_v
            best_v = v
            lo, hi = cell_physical_box(fine_frame, ci)
            best_pos = ((lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
        end
    end
    return best_pos
end

# Replace the level-2 patch with a fresh one re-centred on the current
# peak. Conservation note: anything in the OLD fine patch's footprint
# that lay outside the new fine patch's footprint has already been
# pushed back to the coarse via `restrict_to_parents!` at the end of
# the previous step, so the destruction is not lossy. The new fine
# patch's interior is filled by `prolong_from_parents!` (constant
# prolongation = exact for degree-0).
function _refresh_fine_patch!(state::State)
    cfg = state.config
    peak = _current_peak(state)
    dx_c = _coarse_dx(cfg)
    flo, fhi, n_pow2 = fine_patch_box(peak, cfg.fine_box_size, dx_c)

    # Skip rebuild if the box is unchanged (avoids needless work).
    old_frame = state.ph.levels[2][1]
    if old_frame.lo == flo && old_frame.hi == fhi
        return state
    end

    total_refines = cfg.fine_refines + _ilog2(n_pow2)
    fine_frame = _uniform_2d_frame(total_refines, flo, fhi)
    fine_in  = _zero!(_allocate_field(n_cells(fine_frame.mesh)))
    fine_out = _zero!(_allocate_field(n_cells(fine_frame.mesh)))

    state.ph.levels[2] = [fine_frame]
    state.fine_in  = fine_in
    state.fine_out = fine_out
    state.fine_in_v  = _rho_views(fine_in)
    state.fine_out_v = _rho_views(fine_out)

    # Initialize the new fine patch's interior from the coarse.
    prolong_from_parents!([state.fine_in_v], [state.coarse_in_v],
                           state.ph; level = 2, fieldname = :rho)
    return state
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    Diagnostics

Per-run summary used by tests and the worked-example output.

# Fields

- `mass_initial::Float64` — `∫ ρ dx` over the coarse base patch at
  step 0 (sum of cell-average × cell-volume).
- `mass_final::Float64` — same after `n_steps` steps.
- `mass_drift::Float64` — `mass_final - mass_initial`. Sign is kept.
- `peak_position::NTuple{2, Float64}` — argmax of the fine field at
  the final step.
- `peak_position_analytic::NTuple{2, Float64}` — analytic position.
- `peak_position_error::Float64` — Euclidean distance between the two
  (mod-1 wrapped on each axis to handle periodic returns).
"""
struct Diagnostics
    mass_initial::Float64
    mass_final::Float64
    mass_drift::Float64
    peak_position::NTuple{2, Float64}
    peak_position_analytic::NTuple{2, Float64}
    peak_position_error::Float64
end

# Total mass on the coarse (base) patch: sum over leaves of ρ × V.
function _coarse_total_mass(state::State)
    base_frame = state.ph.levels[1][1]
    field = state.coarse_in.rho
    s = 0.0
    @inbounds for ci in 1:n_cells(base_frame.mesh)
        is_leaf(base_frame.mesh.cells[ci]) || continue
        lo, hi = cell_physical_box(base_frame, ci)
        v = (hi[1] - lo[1]) * (hi[2] - lo[2])
        s += field[ci][1] * v
    end
    return s
end

# Periodic-wrapped Euclidean distance between two points on [0, 1]^2.
@inline function _periodic_distance(a::NTuple{2, Float64},
                                      b::NTuple{2, Float64})
    function _wrap(d)
        d = mod(d, 1.0)
        d > 0.5 && (d -= 1.0)
        return d
    end
    dx = _wrap(a[1] - b[1])
    dy = _wrap(a[2] - b[2])
    return sqrt(dx * dx + dy * dy)
end

# ============================================================================
# Top-level driver
# ============================================================================

"""
    run!(config) -> (state::State, diagnostics::Diagnostics)

Build initial state and time-step `config.n_steps` times. Returns the
final state and a summary of the conservation / tracking diagnostics.
"""
function run!(config::PatchAMRConfig)
    state = build_initial_state(config)
    mass_init = _coarse_total_mass(state)

    for _ in 1:config.n_steps
        step!(state)
    end

    mass_final = _coarse_total_mass(state)
    peak       = _current_peak(state)
    peak_an    = analytic_peak_position(config, state.t)
    err        = _periodic_distance(peak, peak_an)

    diagnostics = Diagnostics(mass_init,
                                mass_final,
                                mass_final - mass_init,
                                peak,
                                peak_an,
                                err)
    return state, diagnostics
end

end # module CFDPatchAMR
