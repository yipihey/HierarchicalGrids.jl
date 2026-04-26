"""
    Geometry

Layer 2: integer-exact geometric operations on the hierarchical mesh.

Operations work in **LCA-relative frames** rather than absolute coordinates.
This means the bit width of arithmetic depends on the depth between cells
and their LCA, not on the absolute depth of either cell — so deep zoom
doesn't widen integer types here.

Vertex coordinates within a cell are signed offsets from the cell center,
in the range [-VERTEX_HALF_RANGE, VERTEX_HALF_RANGE-1] for the chosen
integer type (e.g., [-32768, 32767] for Int16).

# Conventions

- Cell extent at the canonical reference level is `2^VERTEX_BITS` in cell-
  local units (so the interior covers [-2^(VERTEX_BITS-1), 2^(VERTEX_BITS-1)) ).
- A cell's children at the next finer level have half the parent's extent
  along each split axis.
- Volumes are computed in cell-local lattice units (canonical reference
  cell = 1 volume unit).

# Note on Voronoi geometry

This module currently provides axis-aligned cell geometry only. General
polytope clipping (Voronoi-style, r3d-equivalent) is planned for a future
phase. The integer-exact predicates here are the foundation for that future
work.
"""
module Geometry

using StaticArrays
using ..BitPrimitives
using ..Mesh

export cell_extent, cell_center_in_parent
export cell_volume_at_level, cell_volume
export relative_position, distance_squared
export sign_of_axis, in_box, axis_aligned_box_volume

# ============================================================================
# Cell extent and center
# ============================================================================

"""
    cell_extent(mesh::HierarchicalMesh, i::Integer)

Per-axis extent of cell `i` in canonical-reference-cell units.

Returns an SVector of length D with each entry being a fraction (numerator
and denominator stored as a Rational-like pair, or as exact powers of 2).

For a cell at scalar level L below canonical, the extent is 2^(-L) along
each split axis (and 1 along non-split axes — but our current refinement
makes all-or-nothing splits along split axes).

In our integer-exact representation, we return the extent as a tuple of
Int32 values representing the denominator of 2^k for axis k. The actual
fractional extent is 1/2^k.
"""
function cell_extent(mesh::HierarchicalMesh{D, M}, i::Integer) where {D, M}
    # The cell's per-axis level relative to canonical reference.
    # For our current scalar-level cache, we return a tuple where each axis
    # has the same level (true under fully-isotropic refinement).
    #
    # For anisotropic refinement, we'd need to walk the parent chain and
    # accumulate per-axis level changes based on each ancestor's split_mask.
    # That walk is in `per_axis_level_of` below.

    levels = per_axis_level_of(mesh, i)
    return ntuple(d -> Int32(1) << levels[d], Val(D))  # denominator of extent
end

"""
    per_axis_level_of(mesh::HierarchicalMesh, i::Integer)

Per-axis level of cell `i` (number of refinement steps along each axis from
the canonical reference). Walks the parent chain and accumulates level
changes based on each ancestor's split_mask.

Returns an NTuple{D, Int16}.
"""
function per_axis_level_of(mesh::HierarchicalMesh{D, M}, i::Integer) where {D, M}
    levels = zeros(MVector{D, Int16})

    current = UInt32(i)
    parents = mesh._parents
    Mesh.ensure_caches!(mesh)

    while current != Mesh.ROOT_PARENT && parents[current] != Mesh.ROOT_PARENT
        cell = mesh.cells[current]
        # Each split axis gets +1 level
        for axis in 0:(D-1)
            if (cell.split_mask >> axis) & M(1) == M(1)
                levels[axis + 1] += Int16(1)
            end
        end
        current = parents[current]
    end

    return Tuple(levels)
end

"""
    cell_center_in_parent(c::CellMeta)

Center of cell `c` within its parent's local frame, in units where the
parent's extent is `VERTEX_RANGE` along each axis. Returns SVector of length D.

For a cell at position 0 along a split axis, its center is at -VERTEX_RANGE/4.
For position 1, at +VERTEX_RANGE/4. For non-split axes, the cell occupies
the full parent extent and its center is at the parent's center (0).
"""
@inline function cell_center_in_parent(c::CellMeta{D, M}, ::Type{T}=Int32) where {D, M, T}
    # PDEP scatters sibling_index across the mask
    expanded = pdep(UInt32(c.sibling_index), UInt32(c.split_mask))
    # Each split axis: position 0 -> center at -1/4, position 1 -> center at +1/4
    # Non-split axes: center at 0
    # We work in units where the parent extent = 2 (so half-extents are ±1)
    # and child centers along split axes are at ±1/2.
    # In integer units, multiply by VERTEX_HALF_RANGE for storage.
    # For now, return integer offsets in units where parent half-extent = 2.
    return ntuple(Val(D)) do d
        if (c.split_mask >> (d-1)) & M(1) == M(1)
            T((Int(expanded >> (d-1)) & 1) * 2 - 1)  # -1 or +1
        else
            T(0)
        end
    end
end

# ============================================================================
# Volume computation
# ============================================================================

"""
    cell_volume_at_level(::Val{D}, level::NTuple{D, Integer})

Compute the exact integer volume of a cell at the given per-axis level,
expressed as numerator and denominator (since volume = 1/(2^sum(level))
in canonical-reference units).

For a fully axis-aligned cell, volume = ∏ 1/2^level[d] = 1/2^sum(level).

Returns (numerator=1, denominator=2^sum(level)) as a tuple. If sum(level)
exceeds typemax(Int) bits (extremely deep refinement), returns BigInt.
"""
@inline function cell_volume_at_level(::Val{D}, level::NTuple{D, T}) where {D, T<:Integer}
    total_level = sum(level)
    if total_level <= 62
        return (1, Int64(1) << total_level)
    elseif total_level <= 126
        return (Int128(1), Int128(1) << total_level)
    else
        return (BigInt(1), BigInt(1) << total_level)
    end
end

"""
    cell_volume(mesh::HierarchicalMesh, i::Integer)

Exact volume of cell `i` in canonical-reference-cell units. Returns a tuple
(numerator, denominator). For axis-aligned cells, numerator is always 1 and
denominator is 2^total_level.
"""
function cell_volume(mesh::HierarchicalMesh{D, M}, i::Integer) where {D, M}
    levels = per_axis_level_of(mesh, i)
    return cell_volume_at_level(Val(D), levels)
end

"""
    axis_aligned_box_volume(extent::NTuple{D, Integer})

Volume of an axis-aligned box with the given per-axis extent. Returns the
exact integer volume in the chosen unit type.
"""
@inline function axis_aligned_box_volume(extent::NTuple{D, T}) where {D, T<:Integer}
    return prod(extent)
end

# ============================================================================
# Relative positioning between cells
# ============================================================================

"""
    relative_position(mesh, cell_a, cell_b)

Compute the relative position of cell_a's center with respect to cell_b's
center, in the LCA's local frame.

Returns a tuple of integer offsets along each axis. The bit width of the
result depends on the depth-difference from the LCA, not on absolute depth.

This is the core operation for cell-to-cell geometric calculations:
- For nearby cells (small depth-difference from LCA), result is small.
- For distant cells (large depth-difference), result is larger but still
  bounded.
- For deep zoom-ins, this stays bounded by the *local* depth, not the
  absolute zoom level.
"""
function relative_position(mesh::HierarchicalMesh{D, M}, a::Integer, b::Integer) where {D, M}
    Mesh.ensure_caches!(mesh)
    lca = find_lca(mesh, a, b)

    pos_a = position_relative_to_ancestor(mesh, a, lca)
    pos_b = position_relative_to_ancestor(mesh, b, lca)

    return ntuple(d -> pos_a[d] - pos_b[d], Val(D))
end

"""
    position_relative_to_ancestor(mesh, cell, ancestor)

Compute the position of `cell`'s center in `ancestor`'s local frame.
Walks from `cell` up to `ancestor`, accumulating per-axis position bits.

Returned coordinates are in units where the LCA's extent along each axis
is 2^(depth_below_LCA along that axis). This keeps everything as exact
integers.
"""
function position_relative_to_ancestor(mesh::HierarchicalMesh{D, M},
                                       cell::Integer, ancestor::Integer) where {D, M}
    if cell == ancestor
        return ntuple(_ -> Int64(0), Val(D))
    end

    Mesh.ensure_caches!(mesh)
    parents = mesh._parents

    # Walk from cell to ancestor, collecting per-axis position bits at each step
    # Position contribution at each step depends on which axes were split

    # We need to track per-axis depth below ancestor as we walk
    position = zeros(MVector{D, Int64})
    per_axis_depth = zeros(MVector{D, Int64})

    # First pass: collect path from cell to ancestor
    path_indices = UInt32[]
    current = UInt32(cell)
    while current != ancestor && current != Mesh.ROOT_PARENT
        push!(path_indices, current)
        current = parents[current]
    end

    if current != ancestor
        error("Cell $cell is not a descendant of cell $ancestor")
    end

    # Now walk from ancestor down toward cell, building position
    # We process path_indices in reverse (root-toward-leaf)
    for k in length(path_indices):-1:1
        idx = path_indices[k]
        c = mesh.cells[idx]
        mask = c.split_mask

        # Determine per-axis position (0 or 1) for this step
        expanded = pdep(UInt32(c.sibling_index), UInt32(mask))

        # For each split axis, add the bit at the current per-axis depth
        for axis in 1:D
            if (mask >> (axis-1)) & M(1) == M(1)
                bit = Int64((expanded >> (axis-1)) & UInt32(1))
                # Increase depth and shift position accordingly
                # Position is in the ancestor's frame; each refinement step doubles resolution
                position[axis] = position[axis] * 2 + bit
                per_axis_depth[axis] += 1
            end
        end
    end

    # Convert position to "centered" coordinates: cell center, not lower corner
    # Cell center at depth d along axis is at (position * 2 + 1) - 2^d
    # in units where ancestor extent = 2^(d+1)
    # Actually, simpler: position is the lower-corner index in [0, 2^d - 1]
    # at per-axis depth d. Center = 2*position + 1, in units of 2^(d+1) total.
    # We just return the position; centering is the caller's choice.
    return Tuple(position)
end

"""
    distance_squared(mesh, cell_a, cell_b)

Squared distance between centers of cells a and b in LCA-frame integer units.
Useful as an exact integer predicate for spatial queries.

Note: the units depend on the LCA's depth and the per-axis depths of a and b
below it. For comparisons between distance_squared values from the same query,
they must be in the same frame.
"""
function distance_squared(mesh::HierarchicalMesh{D, M}, a::Integer, b::Integer) where {D, M}
    rel = relative_position(mesh, a, b)
    return sum(Int128(r)^2 for r in rel)
end

# ============================================================================
# Predicates
# ============================================================================

"""
    sign_of_axis(value::Integer)

Returns -1, 0, or +1 based on the sign of `value`. Exact predicate for
"which side of the midplane is this point on" type queries.
"""
@inline sign_of_axis(value::Integer) = sign(value)

"""
    in_box(point::NTuple{D}, lower::NTuple{D}, upper::NTuple{D})

Whether `point` is contained in the axis-aligned box [lower, upper] (closed).
Exact integer predicate.
"""
@inline function in_box(point::NTuple{D, T1}, lower::NTuple{D, T2}, upper::NTuple{D, T3}) where {D, T1, T2, T3}
    for d in 1:D
        if point[d] < lower[d] || point[d] > upper[d]
            return false
        end
    end
    return true
end

# ============================================================================
# Continuous 1D interval primitives (for remap support)
# ============================================================================
#
# These are floating-point, NOT integer-exact: they live in the same module
# as the integer predicates because they are the natural 1D specialization of
# what r3d-style polytope clipping does in higher dimensions. The 2D/3D
# polytope-clipping primitives are deferred to a separate r3d.jl package
# providing integer-exact polygon/polyhedron intersection with polynomial
# moment integration.

"""
    Interval{T}(lo::T, hi::T)

Closed real interval `[lo, hi]` with `lo <= hi`. The empty interval is
represented by `lo > hi` (canonically `Interval(T(NaN), T(NaN))` after
intersection of disjoint intervals).
"""
struct Interval{T}
    lo::T
    hi::T
end

@inline function Interval(lo::Real, hi::Real)
    T = promote_type(typeof(lo), typeof(hi))
    return Interval{T}(T(lo), T(hi))
end

"""
    is_empty(I::Interval) :: Bool

Whether the interval has no positive measure: `hi <= lo`. (Strict inequality
is required for a non-empty interval; equal endpoints are treated as a
zero-measure point set, hence empty for integration purposes.)

`Base.isempty` is also defined as an alias, so the standard Julia idiom
`isempty(interval)` works.
"""
@inline is_empty(I::Interval{T}) where T = !(I.hi > I.lo) || isnan(I.lo) || isnan(I.hi)

# Standard alias so callers can use the conventional `isempty`.
@inline Base.isempty(I::Interval) = is_empty(I)

"""
    interval_length(I::Interval) :: T

Length `hi - lo`, or zero for an empty interval.
"""
@inline interval_length(I::Interval{T}) where T = is_empty(I) ? zero(T) : (I.hi - I.lo)

"""
    interval_intersection(a::Interval, b::Interval) :: Interval

Intersection of two intervals. If disjoint, returns an empty interval
(canonical `Interval(T(NaN), T(NaN))`).

This is the 1D specialization of polytope clipping. In 1D it is trivial;
the higher-dimensional analogue (Sutherland–Hodgman / r3d) lives in the
companion r3d.jl package.
"""
@inline function interval_intersection(a::Interval{T}, b::Interval{T}) where T
    lo = max(a.lo, b.lo)
    hi = min(a.hi, b.hi)
    if hi <= lo
        return Interval{T}(T(NaN), T(NaN))
    end
    return Interval{T}(lo, hi)
end

@inline function interval_intersection(a::Interval, b::Interval)
    T = promote_type(typeof(a.lo), typeof(b.lo))
    return interval_intersection(Interval{T}(T(a.lo), T(a.hi)),
                                  Interval{T}(T(b.lo), T(b.hi)))
end

"""
    interval_contains(I::Interval, x::Real) :: Bool

Whether `x ∈ [lo, hi]` (closed).
"""
@inline interval_contains(I::Interval, x::Real) = !isnan(I.lo) && !isnan(I.hi) && (I.lo <= x <= I.hi)

"""
    affine_map_to_reference(I::Interval, x::Real) :: Real

Map `x` from the interval `I = [lo, hi]` into the reference interval `[0, 1]`.
Returns `(x - lo) / (hi - lo)`. Throws if `I` is empty.
"""
@inline function affine_map_to_reference(I::Interval{T}, x::Real) where T
    is_empty(I) && throw(ArgumentError("cannot affine-map from an empty interval"))
    return (T(x) - I.lo) / (I.hi - I.lo)
end

"""
    affine_map_from_reference(I::Interval, t::Real) :: Real

Inverse of `affine_map_to_reference`: map `t ∈ [0, 1]` to a point in `I`.
Returns `lo + t * (hi - lo)`.
"""
@inline function affine_map_from_reference(I::Interval{T}, t::Real) where T
    is_empty(I) && throw(ArgumentError("cannot affine-map from an empty interval"))
    return I.lo + T(t) * (I.hi - I.lo)
end

export Interval, is_empty, interval_length, interval_intersection
export interval_contains, affine_map_to_reference, affine_map_from_reference

end # module Geometry
