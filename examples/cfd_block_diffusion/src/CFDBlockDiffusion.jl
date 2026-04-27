"""
    CFDBlockDiffusion

A 2D scalar-diffusion solver demonstrating the block-based AMR stack with
explicit point-sample blocks (Path B of the Phase-2 abstractions):

  - `PointSampleFieldSet{2, N, Float64}` carries an N×N grid of point
    samples per leaf cell (block).
  - `for_each_block!` drives the explicit RK2 time step over per-block
    point-stencil kernels.
  - `BlockHaloView` provides cross-block boundary access (with periodic
    BCs handled transparently).

The PDE is the heat equation on `[0, 1]^2` with periodic BCs:

    ∂u/∂t = D ∇²u

initialized to a centred Gaussian. Discretization uses the equispaced
N×N Lagrange-node grid per cell — node `i=1` is the left/lower edge,
node `i=N` is the right/upper edge — so adjacent blocks SHARE the
boundary nodes. The centred-difference Laplacian therefore treats the
neighbour at `i=1` as the LEFT block's point `(N-1, j)` (one grid
spacing to the left, skipping the shared edge), and similarly at
`i=N` as the right block's `(2, j)`.

Two diagnostics are tracked:
  - Mass conservation: ∫u dV is preserved by the heat equation.
  - Peak decay: for an isolated Gaussian on the unbounded plane the
    centre value evolves as u_max(t) = σ² / (σ² + 2 D t). On a finite
    periodic box the prediction holds while the diffused tail is well
    below machine accuracy at the boundary.
"""
module CFDBlockDiffusion

using HierarchicalGrids
using HierarchicalGrids: PERIODIC, is_leaf, refine_cells!, n_cells,
                          enumerate_leaves

# ----------------------------------------------------------------------------
# Compile-time block parameters
# ----------------------------------------------------------------------------

const N      = 8        # 8 × 8 point samples per block
const NGHOST = 1        # ghost depth for centred Laplacian

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

"""
    DiffusionConfig(; domain_lo, domain_hi, n_levels, diffusivity, dt, n_steps)

Configuration record for `run!`.

# Fields
- `domain_lo::NTuple{2, Float64}` — lower corner of the periodic box.
- `domain_hi::NTuple{2, Float64}` — upper corner.
- `n_levels::Int`                 — uniform refinements applied to the
                                     root cell (block count = 2^(2*n_levels)).
- `diffusivity::Float64`          — D in the heat equation.
- `dt::Float64`                   — RK2 timestep. Must satisfy
                                     dt < dx² / (4 D), where
                                     dx = domain_size / (n_blocks * (N-1)).
- `n_steps::Int`                  — number of RK2 steps to take.
"""
struct DiffusionConfig
    domain_lo::NTuple{2, Float64}
    domain_hi::NTuple{2, Float64}
    n_levels::Int
    diffusivity::Float64
    dt::Float64
    n_steps::Int
end

function DiffusionConfig(; domain_lo = (0.0, 0.0),
                          domain_hi = (1.0, 1.0),
                          n_levels::Integer = 2,
                          diffusivity::Real = 0.01,
                          dt::Real,
                          n_steps::Integer)
    return DiffusionConfig(NTuple{2, Float64}(domain_lo),
                            NTuple{2, Float64}(domain_hi),
                            Int(n_levels),
                            Float64(diffusivity),
                            Float64(dt),
                            Int(n_steps))
end

# ----------------------------------------------------------------------------
# Initial condition
# ----------------------------------------------------------------------------

"""
    gaussian_ic(x; sigma=0.1, x0=(0.5, 0.5)) -> Float64

Centred isotropic 2-D Gaussian. The initial condition for the test
problem.
"""
@inline gaussian_ic(x; sigma::Real = 0.1, x0 = (0.5, 0.5)) =
    exp(-((x[1] - x0[1])^2 + (x[2] - x0[2])^2) / (2 * sigma^2))

"""
    gaussian_peak_decay(t, sigma, D) -> Float64

Analytic centre value of an isolated 2-D Gaussian heat kernel started
from amplitude 1, width σ, evolved for time `t` under diffusivity `D`:

    u_max(t) = σ² / (σ² + 2 D t)

Valid for the unbounded plane; on a periodic box it is asymptotic so
long as the diffused tail at the boundary is small.
"""
@inline gaussian_peak_decay(t, sigma, D) =
    sigma^2 / (sigma^2 + 2 * D * t)

# ----------------------------------------------------------------------------
# Block kernel: in-block centred Laplacian + RK2 stage
# ----------------------------------------------------------------------------
#
# The kernel is parameterized by the input field name (`:u`, the state
# being read), the output field name (where the right-hand side is
# accumulated), the per-axis grid spacing `h`, the diffusivity `D`,
# and a time-step-style scale `α` which multiplies the RHS:
#
#     fields_out[name_out][p] = base_field[p] + α * D * Δ u[p]
#
# `base_field` is whichever buffer the RK2 stage starts from. We pass
# its NamedTuple via `ctx` so the kernel can index it with the
# block's cell index.

struct StageCtx{B}
    base_in_views::B      # NamedTuple of PointSampleFieldViews — RK2 base state
    inv_h2::Float64       # 1 / h² for one block (uniform mesh ⇒ same for all)
    D::Float64            # diffusivity
    alpha_dt::Float64     # α * dt — the stage's effective time scale
end

StageCtx(base, inv_h2, D, αdt) =
    StageCtx{typeof(base)}(base, Float64(inv_h2), Float64(D), Float64(αdt))

# Inline helper: read u at (i, j) from the block's :u field.
@inline _u_in(bv, i::Int, j::Int) = bv[Val(:u), (i, j)]

# Inline helper: read u from a neighbour at offset `off` and node `(i, j)`.
@inline function _u_nb(bhv, off::NTuple{2, Int}, i::Int, j::Int)
    return bhv[Val(:u), off, (i, j)]
end

# ----------------------------------------------------------------------------
# RK2 stage kernel.
#
# Stencil convention with N nodes at ξ = (k-1)/(N-1):
#   - node i = 1   sits at the LEFT cell-face  (shared with the (-1, 0)
#     neighbour at its node N)
#   - node i = N   sits at the RIGHT cell-face (shared with the (+1, 0)
#     neighbour at its node 1)
#
# For a centred-difference Laplacian with spacing h between neighbouring
# nodes, the left-of-i=1 neighbour in physical space is the LEFT block's
# node (N-1, j) — one grid spacing to the LEFT of the shared boundary
# node, skipping the duplicated edge.
# Analogously: right-of-(i=N) is the RIGHT block's (2, j).
# ----------------------------------------------------------------------------
function rk2_stage_kernel(bv, bhv, ctx::StageCtx)
    inv_h2 = ctx.inv_h2
    D      = ctx.D
    αdt    = ctx.alpha_dt
    base   = ctx.base_in_views     # NamedTuple{(:u,)} of PointSampleFieldViews
    cell_i = bv.index

    # Snap the base state for this cell — we'll add α*dt*RHS to it.
    base_u_view = base.u[cell_i]

    @inbounds for j in 1:N
        # Lookup neighbours along y (compute outside the inner loop —
        # depend only on j, not i).
        for i in 1:N
            uc = _u_in(bv, i, j)

            # x neighbours
            if i == 1
                u_left  = _u_nb(bhv, (-1, 0), N - 1, j)
            else
                u_left  = _u_in(bv, i - 1, j)
            end
            if i == N
                u_right = _u_nb(bhv, (+1, 0), 2, j)
            else
                u_right = _u_in(bv, i + 1, j)
            end

            # y neighbours
            if j == 1
                u_down  = _u_nb(bhv, (0, -1), i, N - 1)
            else
                u_down  = _u_in(bv, i, j - 1)
            end
            if j == N
                u_up    = _u_nb(bhv, (0, +1), i, 2)
            else
                u_up    = _u_in(bv, i, j + 1)
            end

            lap = (u_left + u_right + u_down + u_up - 4 * uc) * inv_h2
            new_val = base_u_view[(i, j)] + αdt * D * lap
            bv[Val(:u), (i, j)] = new_val
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Mesh setup: uniformly refine the root `n_levels` times.
# ----------------------------------------------------------------------------

function build_uniform_mesh(n_levels::Int)
    mesh = HierarchicalMesh{2}()
    for _ in 1:n_levels
        leaves = enumerate_leaves(mesh)
        refine_cells!(mesh, leaves)
    end
    return mesh
end

# ----------------------------------------------------------------------------
# Field initialization: evaluate `gaussian_ic` at the N×N Lagrange nodes
# of every leaf cell.
# ----------------------------------------------------------------------------

# Equispaced node coordinate in [0, 1] for index k ∈ 1..N.
@inline _ref_node(k::Int) = Float64(k - 1) / Float64(N - 1)

function initialize_field!(field_view, frame, mesh; sigma = 0.1)
    npts = N * N
    for c in enumerate_leaves(mesh)
        p_lo, p_hi = cell_physical_box(frame, c)
        ext1 = p_hi[1] - p_lo[1]
        ext2 = p_hi[2] - p_lo[2]
        # Build the N^2 sample tuple in column-major order (matching
        # `point_multi_to_flat` convention).
        vals = ntuple(npts) do flat
            mi = point_flat_to_multi(Val(2), Val(N), flat)
            x = (p_lo[1] + _ref_node(mi[1]) * ext1,
                 p_lo[2] + _ref_node(mi[2]) * ext2)
            gaussian_ic(x; sigma = sigma)
        end
        field_view[c] = vals
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

# Composite trapezoidal weight on the equispaced N-node Lagrange grid:
# w_1 = w_N = 1/2, w_2..w_{N-1} = 1, scaled so that ∑ w_k = N - 1.
# (The interval [0, 1] has length 1; the spacing is h_ref = 1/(N-1) in
# the reference cube; so the integral ≈ h_ref * ∑ w_k * f_k.)
@inline function _trapz_weight_1d(k::Int)
    return (k == 1 || k == N) ? 0.5 : 1.0
end

@inline function _trapz_weight_2d(i::Int, j::Int)
    return _trapz_weight_1d(i) * _trapz_weight_1d(j)
end

"""
    integrate_u(field, mesh, frame) -> Float64

Trapezoidal-rule estimate of ∫ u dV over the whole domain. Uses the
N×N Lagrange grid in each leaf block.

This is the "mass" of the field; for the heat equation with periodic
BCs it should be conserved exactly under the trapezoidal rule (the
weights of the interior of one cell PLUS the half-weights of the
shared boundary nodes recombine to a single full weight at the
shared edge).
"""
function integrate_u(field_view, mesh, frame)
    total = 0.0
    h_ref_2 = (1.0 / Float64(N - 1))^2     # reference-cube cell area
    for c in enumerate_leaves(mesh)
        p_lo, p_hi = cell_physical_box(frame, c)
        ext1 = p_hi[1] - p_lo[1]
        ext2 = p_hi[2] - p_lo[2]
        cell_jac = ext1 * ext2             # physical area of one block
        s = 0.0
        pv = field_view[c]
        for j in 1:N, i in 1:N
            s += _trapz_weight_2d(i, j) * pv[(i, j)]
        end
        total += cell_jac * h_ref_2 * s
    end
    return total
end

"""
    peak_value(field, mesh) -> Float64

Maximum point-sample value across all blocks — proxies the centre
value of the Gaussian.
"""
function peak_value(field_view, mesh)
    m = -Inf
    for c in enumerate_leaves(mesh)
        pv = field_view[c]
        for k in 1:length(pv)
            v = pv[k]
            v > m && (m = v)
        end
    end
    return m
end

# ----------------------------------------------------------------------------
# RK2 driver
# ----------------------------------------------------------------------------

"""
    Diagnostics

Diagnostic record returned by `run!`.

# Fields
- `mass_initial::Float64`        — ∫ u dV at t = 0.
- `mass_final::Float64`          — ∫ u dV at the final time.
- `mass_drift::Float64`          — `mass_final - mass_initial` (should be 0
                                    to round-off for the periodic heat
                                    equation under the trapezoidal rule).
- `peak_initial::Float64`        — initial centre value (≈ 1 for σ = 0.1).
- `peak_final::Float64`          — measured final centre value.
- `peak_predicted::Float64`      — analytic σ² / (σ² + 2 D t).
- `peak_decay_error::Float64`    — `|peak_final - peak_predicted| / peak_predicted`.
- `wall_time_seconds::Float64`   — total wall time of the time-stepping loop.
- `n_blocks::Int`                — leaf count after uniform refinement.
- `dx::Float64`                  — physical grid spacing on the N-node grid.
"""
struct DiagnosticsResult
    mass_initial::Float64
    mass_final::Float64
    mass_drift::Float64
    peak_initial::Float64
    peak_final::Float64
    peak_predicted::Float64
    peak_decay_error::Float64
    wall_time_seconds::Float64
    n_blocks::Int
    dx::Float64
end

"""
    run!(config::DiffusionConfig; sigma=0.1, backend=Sequential())
        -> (final_field, diagnostics)

Build the mesh, allocate two `PointSampleFieldSet` buffers (current /
next), initialize from a centred Gaussian, then take `n_steps` explicit
RK2 steps. Returns the final field-set and a `DiagnosticsResult`.

The RK2 scheme uses two stages:

    u^* = u^n + (dt/2) RHS(u^n)
    u^{n+1} = u^n + dt * RHS(u^*)

executed by two `for_each_block!` passes per step. Buffers are swapped
between steps to avoid repeated allocation.
"""
function run!(config::DiffusionConfig; sigma::Real = 0.1,
              backend::HierarchicalGrids.AbstractParallelBackend =
                                              HierarchicalGrids.Sequential())
    # 1. Mesh
    mesh = build_uniform_mesh(config.n_levels)
    n_leaves = length(enumerate_leaves(mesh))

    frame = EulerianFrame(mesh, config.domain_lo, config.domain_hi)

    # All blocks have identical extent in our uniform-refinement setup,
    # so `dx` is a single number we can compute once.
    n_blocks_per_axis = 1 << config.n_levels
    cell_extent = (config.domain_hi[1] - config.domain_lo[1]) /
                    n_blocks_per_axis
    dx = cell_extent / Float64(N - 1)
    inv_h2 = 1.0 / (dx * dx)

    # CFL safety check (warn but proceed; the test wires a safe dt).
    dt_max = (dx * dx) / (4 * config.diffusivity)
    if config.dt >= dt_max
        @warn "CFD diffusion: dt = $(config.dt) ≥ dx²/(4D) = $dt_max — " *
              "explicit RK2 may be unstable" maxlog = 1
    end

    # 2. Allocate two PointSampleFieldSet buffers.
    n_cells_total = n_cells(mesh)
    fs_curr = allocate_point_sample_fields(SoA(), Val(2), Val(N),
                                              n_cells_total; u = Float64)
    fs_next = allocate_point_sample_fields(SoA(), Val(2), Val(N),
                                              n_cells_total; u = Float64)

    # Zero the storage for non-leaf slots so diagnostics don't read junk.
    for i in 1:n_cells_total
        fs_curr.u[i] = ntuple(_ -> 0.0, N * N)
        fs_next.u[i] = ntuple(_ -> 0.0, N * N)
    end

    # 3. Initial condition.
    initialize_field!(fs_curr.u, frame, mesh; sigma = sigma)

    bcs = FrameBoundaries(((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)))

    # 4. Diagnostics (initial).
    mass0 = integrate_u(fs_curr.u, mesh, frame)
    peak0 = peak_value(fs_curr.u, mesh)

    # 5. RK2 loop.
    #
    # Roles of the two buffers across one step:
    #   stage 1: read = fs_curr,  write = fs_next  (u* = u^n + dt/2 * RHS(u^n))
    #   stage 2: read = fs_next,  write = fs_curr  (u^{n+1} = u^n + dt * RHS(u*))
    #              base = fs_curr (held over from before stage 1)
    #
    # We pass the *base* state through `ctx.base_in_views` so the kernel
    # can compute  base + α dt RHS(read).
    fs_a = fs_curr     # holds u^n at start of step
    fs_b = fs_next     # scratch

    t_start = time_ns()
    for _step in 1:config.n_steps
        base_views_a = (u = fs_a.u,)
        in_views_a   = (u = fs_a.u,)
        out_views_b  = (u = fs_b.u,)

        ctx_stage1 = StageCtx(base_views_a, inv_h2, config.diffusivity,
                                0.5 * config.dt)
        for_each_block!(rk2_stage_kernel, out_views_b, in_views_a, frame;
                          ghost_depth = NGHOST, bcs = bcs, ctx = ctx_stage1,
                          backend = backend)

        # Stage 2: read fs_b (= u*), still base = fs_a, write back to fs_a.
        in_views_b   = (u = fs_b.u,)
        out_views_a  = (u = fs_a.u,)

        ctx_stage2 = StageCtx(base_views_a, inv_h2, config.diffusivity,
                                config.dt)
        for_each_block!(rk2_stage_kernel, out_views_a, in_views_b, frame;
                          ghost_depth = NGHOST, bcs = bcs, ctx = ctx_stage2,
                          backend = backend)
        # fs_a now holds u^{n+1}; fs_b is junk scratch (will be reused in
        # the next step's stage 1).
    end
    wall = (time_ns() - t_start) / 1e9

    # 6. Diagnostics (final).
    final_t = config.n_steps * config.dt
    mass1 = integrate_u(fs_a.u, mesh, frame)
    peak1 = peak_value(fs_a.u, mesh)
    peak_pred = gaussian_peak_decay(final_t, sigma, config.diffusivity)

    diags = DiagnosticsResult(mass0, mass1, mass1 - mass0,
                                peak0, peak1, peak_pred,
                                abs(peak1 - peak_pred) / peak_pred,
                                wall, n_leaves, dx)

    return fs_a, diags
end

# ----------------------------------------------------------------------------
# Re-exports for the test driver
# ----------------------------------------------------------------------------

export DiffusionConfig, DiagnosticsResult, run!
export gaussian_ic, gaussian_peak_decay
export N, NGHOST

end # module
