# ============================================================================
# FrameFaceCache — face-list cache for `for_each_face!` (PR-2)
#
# `for_each_face!` (Solver/Orchestrators.jl) used to re-enumerate interior +
# boundary faces on every call. On a 64-leaf 2D mesh that costs ~44 KB of
# transient allocations per invocation (two `Vector{Tuple{...}}` builds with
# per-cell `face_neighbors` and `face_fine_neighbors` calls). For a CFD
# pipeline that calls `for_each_face!` thousands of times between refinement
# events, that's pure waste.
#
# This file caches the enumeration on the `EulerianFrame`. The cache is
# built lazily on first use and invalidated by a one-shot refinement
# listener on the underlying `HierarchicalMesh`, mirroring the
# `_cached_neighbor_graph` pattern in `Mesh.Neighbors`.
#
# Storage is SoA (parallel `Vector`s) rather than `Vector{Tuple{...}}` so
# the cached payload is GC-friendly: a few small contiguous arrays instead
# of N small heap-allocated tuples.
# ============================================================================

"""
    FrameFaceCache{D}

Caches `for_each_face!`'s interior and boundary face enumeration on a
`(frame, frame_bcs)` pair. Built lazily on first use; invalidated by a
one-shot refinement listener on the underlying `HierarchicalMesh`,
mirroring the `_cached_neighbor_graph` pattern in `Mesh.Neighbors`.

# Fields

Interior faces (one entry per `(i, j, axis, hanging)` tuple from
`_enumerate_interior_faces`):

- `interior_left_idx::Vector{Int32}` — lower-indexed cell `i`.
- `interior_right_idx::Vector{Int32}` — neighbor cell `j`.
- `interior_axis::Vector{Int8}` — axis `1..D`.
- `interior_hanging::BitVector` — `true` for coarse-fine contact.

Boundary faces (one entry per `(i, axis, side)` tuple from
`_enumerate_boundary_faces`):

- `boundary_cell_idx::Vector{Int32}` — boundary cell.
- `boundary_axis::Vector{Int8}` — axis `1..D`.
- `boundary_side::Vector{Int8}` — `1` (lo) or `2` (hi).

- `_listener_handle::ListenerHandle` — handle of the one-shot listener
  registered on `frame.mesh`. Used by the listener to unregister itself
  on first invalidation.
"""
mutable struct FrameFaceCache{D}
    interior_left_idx::Vector{Int32}
    interior_right_idx::Vector{Int32}
    interior_axis::Vector{Int8}
    interior_hanging::BitVector

    boundary_cell_idx::Vector{Int32}
    boundary_axis::Vector{Int8}
    boundary_side::Vector{Int8}

    _listener_handle::Mesh.ListenerHandle
end

# ----------------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------------

# Build a fresh `FrameFaceCache` from the current mesh topology. The
# `_listener_handle` is filled in by `ensure_face_cache!` AFTER the
# listener is registered (we need a handle to a cache that exists, and
# the listener needs the handle to unregister itself).
# Inline replicas of `Solver._enumerate_interior_faces` and
# `Solver._enumerate_boundary_faces`. We can't call those from here:
# `Overlap` is loaded before `Solver` (Solver depends on Overlap), so
# the symbols don't exist at the time this module is compiled. The
# originals stay in place — they may have other uses, and we want the
# behavior to be a byte-equal mirror, not a delete-and-rewrite.
#
# Each interior face is `(i, j, axis, hanging::Bool)` where `i` is the
# lower-indexed cell and we visit each face from `i`'s +axis side only.
# Hanging-node faces (coarse `i` with multiple fine neighbors) emit one
# entry per fine neighbor with `hanging=true`.
function _enumerate_interior_faces_for_cache(mesh::HierarchicalMesh{D}) where {D}
    faces = Tuple{Int, Int, Int, Bool}[]
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        nbs = Mesh.face_neighbors(mesh, i)
        for axis in 1:D
            face_idx = 2 * axis            # +axis side
            rep = nbs[face_idx]
            rep == 0 && continue
            fines = Mesh.face_fine_neighbors(mesh, i, face_idx)
            if length(fines) >= 2
                for f in fines
                    j = Int(f)
                    j > i || continue
                    push!(faces, (i, j, axis, true))
                end
            else
                j = Int(rep)
                j > i || continue
                push!(faces, (i, j, axis, false))
            end
        end
    end
    return faces
end

# Boundary faces: `(i, axis, side)` where `side ∈ {1=lo, 2=hi}`.
function _enumerate_boundary_faces_for_cache(mesh::HierarchicalMesh{D}) where {D}
    out = Tuple{Int, Int, Int}[]
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        nbs = Mesh.face_neighbors(mesh, i)
        for axis in 1:D
            if nbs[2 * axis - 1] == 0
                push!(out, (i, axis, 1))
            end
            if nbs[2 * axis] == 0
                push!(out, (i, axis, 2))
            end
        end
    end
    return out
end

function _build_face_cache(frame::EulerianFrame{D, T}) where {D, T}
    mesh = frame.mesh

    interior = _enumerate_interior_faces_for_cache(mesh)
    boundary = _enumerate_boundary_faces_for_cache(mesh)

    n_int = length(interior)
    interior_left_idx  = Vector{Int32}(undef, n_int)
    interior_right_idx = Vector{Int32}(undef, n_int)
    interior_axis      = Vector{Int8}(undef, n_int)
    interior_hanging   = falses(n_int)
    @inbounds for k in 1:n_int
        (i, j, axis, hanging) = interior[k]
        interior_left_idx[k]  = Int32(i)
        interior_right_idx[k] = Int32(j)
        interior_axis[k]      = Int8(axis)
        interior_hanging[k]   = hanging
    end

    n_bnd = length(boundary)
    boundary_cell_idx = Vector{Int32}(undef, n_bnd)
    boundary_axis     = Vector{Int8}(undef, n_bnd)
    boundary_side     = Vector{Int8}(undef, n_bnd)
    @inbounds for k in 1:n_bnd
        (i, axis, side) = boundary[k]
        boundary_cell_idx[k] = Int32(i)
        boundary_axis[k]     = Int8(axis)
        boundary_side[k]     = Int8(side)
    end

    return FrameFaceCache{D}(
        interior_left_idx, interior_right_idx, interior_axis, interior_hanging,
        boundary_cell_idx, boundary_axis, boundary_side,
        Mesh.ListenerHandle(0),  # placeholder; filled by ensure_face_cache!
    )
end

# ----------------------------------------------------------------------------
# Cache slot management on EulerianFrame
# ----------------------------------------------------------------------------

# Internal: clear the cache slot. Called by the one-shot listener AFTER
# it fires; the listener also unregisters itself in the same step.
function _invalidate_face_cache!(frame::EulerianFrame{D, T}) where {D, T}
    frame._cached_face_cache = nothing
    return frame
end

"""
    ensure_face_cache!(frame::EulerianFrame{D, T},
                        frame_bcs::Union{Nothing, FrameBoundaries{D}}
                        ) -> FrameFaceCache{D}

Return the cached face enumeration for `frame`, building it lazily on
first call. On the first build, registers a one-shot refinement
listener on `frame.mesh` that clears the cache slot and unregisters
itself on the next refine/coarsen event. Subsequent calls after
invalidation rebuild and re-register.

The `frame_bcs` argument is currently unused (the enumeration is purely
topological — boundary detection comes from `face_neighbors == 0`, not
from BC kind), but it's accepted in the signature so callers can pass
it through and so future cache flavors that DO depend on BC can extend
the keying without changing the call site.
"""
function ensure_face_cache!(frame::EulerianFrame{D, T},
                              frame_bcs::Union{Nothing, FrameBoundaries{D}} = nothing
                              ) where {D, T}
    cached = frame._cached_face_cache
    if cached === nothing
        new_cache = _build_face_cache(frame)
        # Register a one-shot listener BEFORE storing the cache so the
        # cache always carries a live handle. The listener captures
        # `frame` by closure; on fire it nulls the slot and removes
        # itself from the mesh's listener list.
        local_handle = Ref{Mesh.ListenerHandle}(Mesh.ListenerHandle(0))
        cb = function (_event::Mesh.RefinementEvent)
            _invalidate_face_cache!(frame)
            Mesh.unregister_refinement_listener!(frame.mesh, local_handle[])
        end
        local_handle[] = Mesh.register_refinement_listener!(frame.mesh, cb)
        new_cache._listener_handle = local_handle[]
        frame._cached_face_cache = new_cache
        return new_cache::FrameFaceCache{D}
    else
        return cached::FrameFaceCache{D}
    end
end

# ----------------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------------

function Base.show(io::IO, fc::FrameFaceCache{D}) where {D}
    n_int = length(fc.interior_left_idx)
    n_bnd = length(fc.boundary_cell_idx)
    print(io, "FrameFaceCache{", D, "}(",
              "n_interior=", n_int,
              ", n_boundary=", n_bnd, ")")
end
