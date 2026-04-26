"""
    HaloView

Lazy view that wraps a per-cell `PolynomialFieldView` plus the mesh's cached
`NeighborGraph` (built lazily on first use). Supports indexing of the form

```
hv[i, off::NTuple{D, Int}]   # coefficients of the cell at offset `off`
                              # from cell `i` (in cell-hops along each axis)
                              # or `nothing` if out-of-domain (boundary)
```

Implementation notes:

- The wrapper is a small immutable struct. Constructing it allocates only
  the struct itself; `getindex` is allocation-free in the no-listener fast
  path (the neighbor graph is built once and reused).
- For `depth > 1`, the magnitude of `off` (sum of `abs.(off)`) must not
  exceed `depth`; out-of-range offsets raise an `ArgumentError`.
- Out-of-domain offsets (a face hop that hits the boundary mid-walk) return
  `nothing`. Once PR-D lands the boundary fill, this same path will return
  ghost values.
"""

# This file is included from src/Storage/Storage.jl, INSIDE the Storage
# module. It needs `PolynomialFieldView` (defined in PolynomialFieldSet.jl)
# and the mesh's `NeighborGraph` (qualified via the parent module).

# We deliberately keep the parameter list typed-but-not-too-specific so the
# wrapper survives any of the layout/basis combinations PolynomialFieldSet
# supports.

struct HaloView{FV, MeshT, D}
    field::FV
    mesh::MeshT
    depth::Int

    function HaloView{FV, MeshT, D}(field::FV, mesh::MeshT, depth::Int) where {FV, MeshT, D}
        depth >= 1 || throw(ArgumentError("HaloView depth must be ≥ 1, got $depth"))
        new{FV, MeshT, D}(field, mesh, depth)
    end
end

"""
    halo_view(field::PolynomialFieldView, mesh::HierarchicalMesh{D}, depth=1)

Construct a `HaloView`. The mesh argument is required because a
`PolynomialFieldView` does not know its mesh; the indexing convention
assumes `field`'s element index `i` matches `mesh.cells[i]`.

Building the view is O(1); the underlying `NeighborGraph` is built lazily
on first index access via the mesh's standard cache mechanism.
"""
function halo_view(field::PolynomialFieldView, mesh::HierarchicalMesh{D},
                   depth::Integer=1) where {D}
    return HaloView{typeof(field), typeof(mesh), D}(field, mesh, Int(depth))
end

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

# Walk along cell hops to find the cell at a given offset from cell `i`.
# Each axis contribution moves abs(off[d]) face-hops in the (sign(off[d]))
# direction along axis d. Returns 0 if at any step we hit the domain
# boundary or run out of neighbor coverage.
@inline function _walk_offset(mesh, i::Integer, off::NTuple{D, Int}) where {D}
    cur = UInt32(i)
    @inbounds for d in 1:D
        steps = off[d]
        if steps == 0
            continue
        end
        face_idx = steps > 0 ? 2*d : 2*d - 1
        s = abs(steps)
        for _ in 1:s
            tup = face_neighbors(mesh, Int(cur))
            nxt = tup[face_idx]
            nxt == 0 && return UInt32(0)
            cur = nxt
        end
    end
    return cur
end

@inline function _check_offset(off::NTuple{D, Int}, depth::Int) where {D}
    s = 0
    @inbounds for d in 1:D
        s += abs(off[d])
    end
    if s > depth
        throw(ArgumentError("HaloView offset $off exceeds depth $depth"))
    end
    return nothing
end

"""
    hv[i, off::NTuple{D, Int}] -> coefficient column or `nothing`

For `off == ntuple(_->0, D)` (the cell itself) returns a `PolynomialView`.
For non-zero offsets, walks the neighbor graph; if any hop hits the domain
boundary, returns `nothing`. The coefficient column is returned as a
`Tuple` of length `n_coeffs(basis)` for zero allocation.
"""
@inline function Base.getindex(hv::HaloView{FV, MeshT, D}, i::Integer,
                                off::NTuple{D, Int}) where {FV, MeshT, D}
    _check_offset(off, hv.depth)
    nb = _walk_offset(hv.mesh, i, off)
    nb == 0 && return nothing
    return hv.field[Int(nb)]
end

# Convenience: scalar offset for D=1
@inline function Base.getindex(hv::HaloView{FV, MeshT, 1}, i::Integer,
                                off::Int) where {FV, MeshT}
    return Base.getindex(hv, i, (off,))
end
