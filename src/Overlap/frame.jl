"""
    EulerianFrame{D, T}

Pairs a `HierarchicalMesh{D}` with the physical-space bounds of its
canonical reference cell. The mesh itself only knows integer-relative
geometry; the frame is what gives those coordinates physical meaning.

# Fields

- `mesh::HierarchicalMesh{D}` — the underlying tree mesh.
- `lo::NTuple{D, T}` — lower corner of the root cell in physical units.
- `hi::NTuple{D, T}` — upper corner of the root cell in physical units.

The root cell `mesh.cells[1]` covers the box `[lo, hi]` in physical
coordinates. Refinement subdivides this box hierarchically; each leaf
cell has a physical AABB given by `cell_physical_box(frame, i)`.

The frame is intentionally a thin lightweight wrapper, separate from
HierarchicalMesh itself, so the core mesh layer stays pure
integer-relative geometry. Different applications can attach different
physical frames to the same mesh.
"""
struct EulerianFrame{D, T}
    mesh::HierarchicalMesh{D}
    lo::NTuple{D, T}
    hi::NTuple{D, T}

    function EulerianFrame{D, T}(mesh::HierarchicalMesh{D}, lo::NTuple{D, T}, hi::NTuple{D, T}) where {D, T}
        for d in 1:D
            hi[d] > lo[d] || throw(ArgumentError("hi[$d] must exceed lo[$d] (got hi=$(hi[d]), lo=$(lo[d]))"))
        end
        return new{D, T}(mesh, lo, hi)
    end
end

# Convenience outer constructors: type inference for tuple inputs, and
# vector/iterable input with promotion.
EulerianFrame(mesh::HierarchicalMesh{D}, lo::NTuple{D, T}, hi::NTuple{D, T}) where {D, T} =
    EulerianFrame{D, T}(mesh, lo, hi)

function EulerianFrame(mesh::HierarchicalMesh{D}, lo, hi) where D
    length(lo) == D || throw(ArgumentError("lo must have length $D, got $(length(lo))"))
    length(hi) == D || throw(ArgumentError("hi must have length $D, got $(length(hi))"))
    T = promote_type(eltype(lo), eltype(hi))
    lo_t = ntuple(d -> T(lo[d]), D)
    hi_t = ntuple(d -> T(hi[d]), D)
    return EulerianFrame{D, T}(mesh, lo_t, hi_t)
end

@inline spatial_dimension(::EulerianFrame{D, T}) where {D, T} = D
@inline scalar_type(::EulerianFrame{D, T}) where {D, T} = T

"""
    root_box(frame::EulerianFrame) -> (lo, hi)

The physical AABB of the root cell. Trivial accessor.
"""
@inline root_box(frame::EulerianFrame) = (frame.lo, frame.hi)

"""
    cell_unit_box(mesh::HierarchicalMesh{D}, i::Integer) -> (lo, hi)

Compute the AABB of cell `i` in the unit reference cube `[0, 1]^D` by
walking from root to `i` and accumulating relative offsets.

This depends only on the mesh topology and is independent of any
physical-coordinate frame. Use `cell_physical_box` to get coordinates
in a particular `EulerianFrame`.

Returns `(lo, hi)` where each is an `NTuple{D, Float64}` in `[0, 1]`.
"""
function cell_unit_box(mesh::HierarchicalMesh{D, M}, i::Integer) where {D, M}
    Mesh.ensure_caches!(mesh)
    parents = mesh._parents

    # Collect the path from root to `i` (parent chain)
    chain = UInt32[]
    current = UInt32(i)
    while current != Mesh.ROOT_PARENT && parents[current] != Mesh.ROOT_PARENT
        push!(chain, current)
        current = parents[current]
    end
    reverse!(chain)  # root → leaf order

    # Start at the unit cube and walk down
    lo = ntuple(_ -> 0.0, Val(D))
    hi = ntuple(_ -> 1.0, Val(D))
    for ci in chain
        cell = mesh.cells[ci]
        sib = cell.sibling_index
        mask = cell.split_mask
        # For each split axis, determine whether this cell is in the lower
        # or upper half of its parent along that axis. Non-split axes
        # inherit the parent's full extent.
        new_lo = MVector{D, Float64}(undef)
        new_hi = MVector{D, Float64}(undef)
        # Need the bit position in `sib` corresponding to each split axis.
        # split_mask bits enumerate which axes are split; the sib bits index
        # into those split axes from low to high.
        bit_pos = 0
        for axis in 1:D
            axis_bit = M(1) << (axis - 1)
            if (mask & axis_bit) != 0
                # This axis is split. Look at sib's `bit_pos`-th bit.
                lower_half = (sib >> bit_pos) & M(1) == 0
                mid = (lo[axis] + hi[axis]) / 2
                if lower_half
                    new_lo[axis] = lo[axis]
                    new_hi[axis] = mid
                else
                    new_lo[axis] = mid
                    new_hi[axis] = hi[axis]
                end
                bit_pos += 1
            else
                # Axis not split; inherit parent's extent
                new_lo[axis] = lo[axis]
                new_hi[axis] = hi[axis]
            end
        end
        lo = Tuple(new_lo)
        hi = Tuple(new_hi)
    end
    return (lo, hi)
end

"""
    cell_physical_box(frame::EulerianFrame{D, T}, i::Integer) -> (lo, hi)

Physical AABB of cell `i` in the given Eulerian frame. Returns
`(lo, hi)` where each is an `NTuple{D, T}` in physical coordinates.
"""
function cell_physical_box(frame::EulerianFrame{D, T}, i::Integer) where {D, T}
    u_lo, u_hi = cell_unit_box(frame.mesh, i)
    extent = ntuple(d -> frame.hi[d] - frame.lo[d], Val(D))
    p_lo = ntuple(d -> frame.lo[d] + T(u_lo[d]) * extent[d], Val(D))
    p_hi = ntuple(d -> frame.lo[d] + T(u_hi[d]) * extent[d], Val(D))
    return (p_lo, p_hi)
end

"""
    enumerate_leaves(mesh::HierarchicalMesh) -> Vector{Int}

Indices of all leaf cells in the mesh, in mesh-storage order.
Convenience helper for traversal.
"""
function enumerate_leaves(mesh::HierarchicalMesh)
    out = Int[]
    @inbounds for i in 1:n_cells(mesh)
        if is_leaf(mesh.cells[i])
            push!(out, i)
        end
    end
    return out
end

"""
    aabbs_overlap(a_lo, a_hi, b_lo, b_hi) -> Bool

Whether two axis-aligned bounding boxes intersect with positive measure.
Uses strict inequality: touching at a face is treated as non-overlapping
(consistent with the empty-interval convention in 1D).
"""
@inline function aabbs_overlap(a_lo::NTuple{D, T}, a_hi::NTuple{D, T},
                                b_lo::NTuple{D, T}, b_hi::NTuple{D, T}) where {D, T}
    @inbounds for d in 1:D
        (a_hi[d] > b_lo[d] && b_hi[d] > a_lo[d]) || return false
    end
    return true
end
