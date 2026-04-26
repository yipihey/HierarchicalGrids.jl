# ============================================================================
# Pixelization: AMR-driven Eulerian quadtree fitting to the Lagrangian mesh
# ============================================================================
#
# Goal: build an Eulerian `HierarchicalMesh{2}` whose leaves overlap each
# Lagrangian simplex with approximately a target count. Where a Lagrangian
# simplex is so small that fewer than `target` Eulerian leaves overlap it,
# we refine those leaves. Iterate until stable.
#
# The criterion is purely topological — it doesn't need any field data on
# the Lagrangian mesh — and is the natural visual analog of "make sure
# every Lagrangian patch has enough Eulerian elements under it."

"""
    PixelizationParams(; target_eul_per_lag=4, max_depth=8, max_iters=12,
                         leaf_size=8)

Parameters controlling the AMR pixelization loop.

- `target_eul_per_lag::Int` — desired number of Eulerian leaves overlapping
  each Lagrangian simplex. The loop refines wherever this falls short. A
  value of 4 means "each Lagrangian triangle should be cut into roughly 4
  pieces by the Eulerian grid." Higher values give finer pixelization at
  the cost of more cells.
- `max_depth::Int` — hard cap on Eulerian quadtree depth. With `max_depth = 8`
  the finest leaves are 1 / 256 of the unit square per side.
- `max_iters::Int` — maximum AMR refinement iterations. The loop stops
  early if a pass refines zero cells.
- `leaf_size::Int` — BVH leaf granularity for `compute_overlap`.
"""
Base.@kwdef struct PixelizationParams
    target_eul_per_lag::Int = 4
    max_depth::Int = 8
    max_iters::Int = 12
    leaf_size::Int = 8
end

"""
    n_eulerian_per_lagrangian(ov::GeometricOverlap) -> Vector{Int}

For each Lagrangian simplex, the count of Eulerian leaves overlapping it.
Useful as a diagnostic for AMR convergence.
"""
function n_eulerian_per_lagrangian(ov::GeometricOverlap)
    return [length(ov.lag_to_entries[s]) for s in 1:ov.n_lag]
end

"""
    n_lagrangian_per_eulerian(ov::GeometricOverlap) -> Vector{Int}

For each Eulerian cell index (leaf or otherwise), the count of Lagrangian
simplices overlapping it. Non-leaf cells get 0.
"""
function n_lagrangian_per_eulerian(ov::GeometricOverlap)
    return [length(ov.eul_to_entries[i]) for i in 1:ov.n_eul]
end

"""
    lagrangian_density_factors(lag::SimplicialMesh{2, Float64}) -> Vector{Float64}

Per-Lagrangian-simplex density factor

    ρ_s = |V_ref(s)| / |V_now(s)|

where `V_ref(s)` is the simplex's reference (initial) area and `V_now(s)`
is its current (deformed) area. The absolute value handles the rare case
of inverted simplices, treating them as ordinary mass-bearing cells with
a density derived from their physical area regardless of orientation.

Physical interpretation: each Lagrangian simplex carries a fixed mass
proportional to its reference area (uniform mass-per-reference-area is
the standard assumption for cosmological dark-matter sheet tessellations
and for any incompressible-flow Lagrangian discretization). When the
flow compresses a simplex (current area < reference area), its density
goes up by the same factor; when it stretches, density goes down.

Used by `eulerian_density!` to deposit Lagrangian mass onto the Eulerian
mesh.
"""
function lagrangian_density_factors(lag::SimplicialMesh{2, Float64})
    n_s = n_simplices(lag)
    ρ = Vector{Float64}(undef, n_s)
    @inbounds for s in 1:n_s
        v_ref = abs(simplex_reference_volume(lag, s))
        v_now = abs(simplex_volume(lag, s))
        # Floor to avoid Inf in nearly-degenerate triangles. A simplex
        # that's compressed to ~zero area has effectively infinite density;
        # we cap by clipping at a tiny minimum area.
        ρ[s] = v_ref / max(v_now, 1e-14)
    end
    return ρ
end

"""
    eulerian_density(ov::GeometricOverlap, lag::SimplicialMesh{2, Float64},
                     frame::EulerianFrame{2, Float64}) -> Vector{Float64}

Compute physical mass density on each Eulerian cell from the geometric
overlap and the Lagrangian deformation:

1. Each Lagrangian simplex `s` carries mass `m_s = V_ref(s)` (uniform
   mass-per-reference-area; total Lagrangian mass = 1 for a unit-square
   reference mesh).
2. Each overlap entry `(s, i, V_overlap)` deposits mass

       Δm = ρ_s · V_overlap = (V_ref(s) / V_now(s)) · V_overlap

   from simplex `s` into Eulerian cell `i`. (Equivalently: of the
   simplex's mass `V_ref(s)`, the fraction `V_overlap / V_now(s)` lives
   in cell `i`, so the deposited mass is `V_ref(s) · V_overlap /
   V_now(s)`.)
3. Density on cell `i` is `mass[i] / V_eulerian(i)`.

For non-leaf or non-overlapping cells, density is 0.

The resulting `ρ_eul[i]` is the physical density a continuum solver
would see at the Eulerian cell scale. It's the natural quantity to
visualize: regions of compressed Lagrangian flow show up bright,
stretched regions show up dim.
"""
function eulerian_density(ov::GeometricOverlap,
                           lag::SimplicialMesh{2, Float64},
                           frame::EulerianFrame{2, Float64})
    ρ_lag = lagrangian_density_factors(lag)
    n_eul = ov.n_eul
    ρ_eul = zeros(Float64, n_eul)
    # Deposit mass: walk all entries, accumulate per Eulerian cell.
    @inbounds for entry in ov.entries
        s = Int(entry.lag_idx)
        i = Int(entry.eul_idx)
        ρ_eul[i] += ρ_lag[s] * Float64(entry.volume)
    end
    # Convert mass to density.
    @inbounds for i in 1:n_eul
        ρ_eul[i] == 0 && continue
        lo, hi = cell_physical_box(frame, i)
        cell_area = (hi[1] - lo[1]) * (hi[2] - lo[2])
        ρ_eul[i] /= cell_area
    end
    return ρ_eul
end

# ----------------------------------------------------------------------------
# Build the per-Eulerian-cell indicator
# ----------------------------------------------------------------------------
#
# For each Eulerian leaf, the indicator value is the maximum "deficit"
# (target - n_eul_per_lag[s]) over all Lagrangian simplices `s` overlapping
# that leaf. Cells with no Lagrangian overlap, or whose covered simplices
# all already meet target, get value 0. Refinement threshold is 0.5 (so
# any positive deficit triggers refinement); coarsening is suppressed by
# using a coarsen threshold below the minimum (e.g. -1.0) — since values
# are always ≥ 0, no cell ever falls below it.

function _build_indicator(ov::GeometricOverlap, target::Int)
    n_eul = ov.n_eul
    n_per_lag = n_eulerian_per_lagrangian(ov)
    indicator = zeros(Float64, n_eul)
    @inbounds for ci in 1:n_eul
        idxs = ov.eul_to_entries[ci]
        isempty(idxs) && continue        # leave at 0 (no signal)
        worst_deficit = 0
        for k in idxs
            s = ov.entries[k].lag_idx
            deficit = target - n_per_lag[s]
            if deficit > worst_deficit
                worst_deficit = deficit
            end
        end
        indicator[ci] = Float64(worst_deficit)
    end
    return indicator
end

"""
    pixelize!(eul::HierarchicalMesh{2}, frame::EulerianFrame{2, Float64},
                lag::SimplicialMesh{2, Float64};
                params::PixelizationParams = PixelizationParams())
        -> (overlap::GeometricOverlap, n_iters::Int, refined_per_iter::Vector{Int})

Run the AMR loop: repeatedly compute the geometric overlap between `lag`
and `eul`, derive the per-Eulerian-cell indicator, refine, and recompute.
Stops when an iteration refines zero cells, when a leaf would be refined
past `params.max_depth`, or after `params.max_iters` iterations.

Returns the final overlap (so callers can inspect counts, plot density,
etc.), the number of AMR iterations actually used, and the count of
cells refined in each iteration (useful for diagnostics).

# Arguments

- `eul` — Eulerian mesh; modified in place. The starting state matters:
  pass a uniformly-refined mesh of moderate depth (3–4) to avoid the
  first iteration being dominated by under-refinement.
- `frame` — physical-coordinate wrapper around `eul`. Built once at sim
  startup; doesn't change as the mesh refines (only the cell list does).
- `lag` — Lagrangian mesh in its current configuration. Not modified.

# Notes

The criterion is "any covered simplex needs more leaves" — this is a
high-water-mark trigger and may produce isolated refined cells. For a
visually smoother result, run several extra iterations beyond convergence
to let neighboring cells equalize; or post-process by 2:1 balancing.
"""
function pixelize!(eul::HierarchicalMesh{2}, frame::EulerianFrame{2, Float64},
                    lag::SimplicialMesh{2, Float64};
                    params::PixelizationParams = PixelizationParams())
    refined_per_iter = Int[]

    for iter in 1:params.max_iters
        ov = compute_overlap(lag, frame;
                              moment_order = 0,
                              leaf_size = params.leaf_size)

        indicator = _build_indicator(ov, params.target_eul_per_lag)

        # Refine cells with positive indicator (any covered Lagrangian
        # simplex is below target). Coarsen threshold of -1 never fires
        # since the indicator is always ≥ 0.
        result = refine_by_indicator!(eul, indicator;
                                       refine_threshold = 0.5,
                                       coarsen_threshold = -1.0,
                                       max_level = params.max_depth,
                                       isotropic = true)
        push!(refined_per_iter, result.refined)

        if result.refined == 0
            # Return the overlap consistent with the FINAL mesh state.
            # If we refined this iteration, the overlap above is stale;
            # if we didn't refine, it's already current.
            return (ov, iter, refined_per_iter)
        end
    end

    # Out of iterations: recompute one more time so the returned overlap
    # matches the final mesh.
    ov_final = compute_overlap(lag, frame;
                                moment_order = 0,
                                leaf_size = params.leaf_size)
    return (ov_final, params.max_iters, refined_per_iter)
end
