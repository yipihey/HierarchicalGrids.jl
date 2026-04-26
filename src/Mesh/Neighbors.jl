"""
    Neighbors

Face-neighbor adjacency for `HierarchicalMesh`.

# Algorithm

Each leaf cell `i` lives in a unit reference box `[0,1]^D` whose AABB is
recovered by walking the parent chain of `i` from the root and applying the
sibling/split-mask coordinates of every step. Two leaves are face-neighbors
along axis `d`, side `s ∈ {lo, hi}`, iff their unit-cube AABBs share a
`(D-1)`-face: they agree on axes other than `d` (with positive overlap), and
they touch on axis `d` (one cell's hi face equals the other cell's lo face,
within a tolerance scaled to the deepest split level).

For the first cut (PR-C, correctness-first), the implementation computes
unit-cube AABBs once per leaf in `O(N * depth_avg)` and then performs
`O(2D * N_leaves)` face lookups against an axis-bucketed leaf list. This is
quadratic in the worst case but well below the cost of a full polytope
clip; the data structure is a pure `Vector{NTuple{2D, UInt32}}` for
representative neighbors plus a `Dict` of fine-neighbor lists for
hanging-node faces (only populated when the mesh is unbalanced).

# Coarse-fine face contract

Under `mesh.balanced == false`, a coarse cell C may have many fine
neighbors on a single face. `face_neighbors(mesh, i)[f]` returns *one*
representative leaf neighbor (deterministically, the lowest-indexed).
`face_fine_neighbors(mesh, i, f)` enumerates every fine leaf that
shares the corresponding face with C.

Under `mesh.balanced == true`, the level gap on any face is at most 1.
For 2D this means at most 2 fine neighbors per face; for 3D at most 4.

# Caching

The graph is built lazily on first access (`face_neighbors`,
`face_fine_neighbors`, `cell_adjacency_sparsity`, or `halo_view`). The first
build registers a refinement listener that invalidates the cached graph on
the next mesh modification. The graph is rebuilt on the next access.
"""

# This file is `include`d at the bottom of src/Mesh/Mesh.jl, INSIDE the Mesh
# module, so it has direct access to `HierarchicalMesh` and friends.

# ---------------------------------------------------------------------------
# Internal helper: per-leaf unit-cube AABB
#
# Mirrors src/Overlap/frame.jl::cell_unit_box but without depending on the
# Overlap submodule (we live in Mesh, which is below Overlap). Returns
# `(lo, hi)` as `NTuple{D, Float64}`.
# ---------------------------------------------------------------------------

function _unit_box(mesh::HierarchicalMesh{D, M}, i::Integer) where {D, M}
    ensure_caches!(mesh)
    parents = mesh._parents

    # Collect path from root to i (root excluded — it's the unit cube)
    chain = UInt32[]
    current = UInt32(i)
    while current != ROOT_PARENT && parents[current] != ROOT_PARENT
        push!(chain, current)
        current = parents[current]
    end
    reverse!(chain)

    lo = ntuple(_ -> 0.0, Val(D))
    hi = ntuple(_ -> 1.0, Val(D))
    for ci in chain
        cell = mesh.cells[ci]
        sib  = cell.sibling_index
        mask = cell.split_mask

        bit_pos = 0
        new_lo_t = ntuple(Val(D)) do axis
            axis_bit = M(1) << (axis - 1)
            if (mask & axis_bit) != 0
                lower_half = ((sib >> bit_pos) & M(1)) == M(0)
                bit_pos += 1
                lower_half ? lo[axis] : (lo[axis] + hi[axis]) / 2
            else
                lo[axis]
            end
        end
        bit_pos = 0
        new_hi_t = ntuple(Val(D)) do axis
            axis_bit = M(1) << (axis - 1)
            if (mask & axis_bit) != 0
                lower_half = ((sib >> bit_pos) & M(1)) == M(0)
                bit_pos += 1
                lower_half ? (lo[axis] + hi[axis]) / 2 : hi[axis]
            else
                hi[axis]
            end
        end
        lo = new_lo_t
        hi = new_hi_t
    end
    return (lo, hi)
end

# ---------------------------------------------------------------------------
# NeighborGraph
# ---------------------------------------------------------------------------

"""
    NeighborGraph{D, M}

Face-neighbor adjacency for a `HierarchicalMesh{D, M}`.

# Layout

- `representatives::Vector{NTuple{2D, UInt32}}`: indexed by *cell* index `i`
  (1..n_cells). For non-leaf cells the entry is all zeros. For each face
  `f ∈ 1..2D` (axis `(f+1) ÷ 2`, side lo if `f` is odd, hi if even) the
  entry is the index of the lowest-indexed leaf that touches that face,
  or 0 for a domain-boundary face.
- `fine::Dict{Tuple{UInt32, UInt8}, Vector{UInt32}}`: only populated for
  faces with multiple fine-leaf neighbors. Keys are `(cell, face_index)`;
  values are sorted leaf-index vectors. Empty under balanced=true with
  uniform refinement; small under balanced=false.
"""
struct NeighborGraph{D, M, F}
    # F == 2*D enforced via inner constructor.
    representatives::Vector{NTuple{F, UInt32}}
    fine::Dict{Tuple{UInt32, UInt8}, Vector{UInt32}}

    function NeighborGraph{D, M, F}(reps::Vector{NTuple{F, UInt32}},
                                     fine::Dict{Tuple{UInt32, UInt8}, Vector{UInt32}}
                                     ) where {D, M, F}
        F == 2 * D || throw(ArgumentError("NeighborGraph F must be 2D, got F=$F D=$D"))
        new{D, M, F}(reps, fine)
    end
end

# Convenience constructor that infers F from D.
function NeighborGraph{D, M}(reps::Vector{NTuple{F, UInt32}},
                              fine::Dict{Tuple{UInt32, UInt8}, Vector{UInt32}}
                              ) where {D, M, F}
    return NeighborGraph{D, M, F}(reps, fine)
end

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

# Floating-point tolerance scaled to representable-precision at the deepest
# uniform level. The coordinates are halvings of [0,1], so values are
# exactly representable down to depth ~52; the tolerance protects against
# accumulated rounding only.
const _NEIGHBOR_EPS = 1e-12

@inline _approx_eq(a::Float64, b::Float64) = abs(a - b) <= _NEIGHBOR_EPS

# Two ranges [a_lo, a_hi] and [b_lo, b_hi] overlap with positive measure
# (strict, like `aabbs_overlap` in Overlap/frame.jl).
@inline function _open_overlap(a_lo, a_hi, b_lo, b_hi)
    return (a_hi > b_lo + _NEIGHBOR_EPS) && (b_hi > a_lo + _NEIGHBOR_EPS)
end

"""
    build_neighbor_graph(mesh::HierarchicalMesh{D, M}) -> NeighborGraph{D, M}

Build the face-neighbor adjacency from scratch by computing unit-cube AABBs
for every leaf and matching faces. O(N_leaves^2) in the worst case but with
small constants and early termination on axis disagreement.
"""
function build_neighbor_graph(mesh::HierarchicalMesh{D, M}) where {D, M}
    n = n_cells(mesh)
    F = 2 * D
    representatives = fill(ntuple(_ -> UInt32(0), Val(2*D)), n)
    fine = Dict{Tuple{UInt32, UInt8}, Vector{UInt32}}()

    if n == 0
        return NeighborGraph{D, M}(representatives, fine)
    end

    # Collect leaves and their unit-cube boxes.
    leaf_indices = UInt32[]
    leaf_lo  = NTuple{D, Float64}[]
    leaf_hi  = NTuple{D, Float64}[]
    sizehint!(leaf_indices, n)
    sizehint!(leaf_lo, n)
    sizehint!(leaf_hi, n)
    @inbounds for i in 1:n
        if is_leaf(mesh.cells[i])
            (lo, hi) = _unit_box(mesh, i)
            push!(leaf_indices, UInt32(i))
            push!(leaf_lo, lo)
            push!(leaf_hi, hi)
        end
    end
    L = length(leaf_indices)

    # Bucket leaves by axis-d coordinate of their lo / hi face. We use a
    # dictionary keyed by a discretization of the coordinate, so that
    # candidate faces share an O(1) lookup. The granularity is the smallest
    # representable splitting at depth 32 — well below practical depths.
    quantize(x) = round(Int64, x / _NEIGHBOR_EPS)

    # For each axis d, two dicts: cells whose `lo`-face is at coordinate q,
    # cells whose `hi`-face is at coordinate q.
    lo_buckets = ntuple(_ -> Dict{Int64, Vector{Int}}(), Val(D))
    hi_buckets = ntuple(_ -> Dict{Int64, Vector{Int}}(), Val(D))
    @inbounds for li in 1:L
        for d in 1:D
            push!(get!(lo_buckets[d], quantize(leaf_lo[li][d]), Int[]), li)
            push!(get!(hi_buckets[d], quantize(leaf_hi[li][d]), Int[]), li)
        end
    end

    # For each leaf i, for each face (axis d, side s), collect candidate
    # neighbors from the opposite-bucket at the matching coordinate, then
    # filter by overlap on the OTHER axes.
    @inbounds for li in 1:L
        i_idx = leaf_indices[li]
        a_lo = leaf_lo[li]
        a_hi = leaf_hi[li]

        face_arr = Vector{UInt32}(undef, F)
        for f in 1:F
            face_arr[f] = UInt32(0)
        end

        for d in 1:D
            # face f = 2d-1 (lo side). Our lo[d] face touches neighbors
            # whose hi[d] equals our lo[d].
            face_lo_idx = 2*d - 1
            face_hi_idx = 2*d
            qlo = quantize(a_lo[d])
            qhi = quantize(a_hi[d])

            # Skip immediately if our face is at the domain boundary
            # (lo[d] == 0 or hi[d] == 1). Boundary check is cheap and
            # avoids walking the whole bucket of all cells on the boundary.
            on_lo_boundary = a_lo[d] <= _NEIGHBOR_EPS
            on_hi_boundary = a_hi[d] >= 1.0 - _NEIGHBOR_EPS

            # --- LO side ---
            if !on_lo_boundary
                cands = get(hi_buckets[d], qlo, nothing)
                if cands !== nothing
                    fine_list = UInt32[]
                    rep::UInt32 = UInt32(0)
                    for cj in cands
                        cj == li && continue
                        b_lo = leaf_lo[cj]
                        b_hi = leaf_hi[cj]
                        # axis-d hi of cj must touch our lo
                        # (already guaranteed by bucket)
                        # All OTHER axes must overlap with positive measure
                        ok = true
                        for d2 in 1:D
                            d2 == d && continue
                            if !_open_overlap(a_lo[d2], a_hi[d2], b_lo[d2], b_hi[d2])
                                ok = false
                                break
                            end
                        end
                        ok || continue
                        j_idx = leaf_indices[cj]
                        push!(fine_list, j_idx)
                        if rep == 0 || j_idx < rep
                            rep = j_idx
                        end
                    end
                    if rep != 0
                        face_arr[face_lo_idx] = rep
                        if length(fine_list) > 1
                            sort!(fine_list)
                            fine[(i_idx, UInt8(face_lo_idx))] = fine_list
                        end
                    end
                end
            end

            # --- HI side ---
            if !on_hi_boundary
                cands = get(lo_buckets[d], qhi, nothing)
                if cands !== nothing
                    fine_list = UInt32[]
                    rep = UInt32(0)
                    for cj in cands
                        cj == li && continue
                        b_lo = leaf_lo[cj]
                        b_hi = leaf_hi[cj]
                        ok = true
                        for d2 in 1:D
                            d2 == d && continue
                            if !_open_overlap(a_lo[d2], a_hi[d2], b_lo[d2], b_hi[d2])
                                ok = false
                                break
                            end
                        end
                        ok || continue
                        j_idx = leaf_indices[cj]
                        push!(fine_list, j_idx)
                        if rep == 0 || j_idx < rep
                            rep = j_idx
                        end
                    end
                    if rep != 0
                        face_arr[face_hi_idx] = rep
                        if length(fine_list) > 1
                            sort!(fine_list)
                            fine[(i_idx, UInt8(face_hi_idx))] = fine_list
                        end
                    end
                end
            end
        end

        representatives[i_idx] = ntuple(f -> face_arr[f], Val(2*D))
    end

    return NeighborGraph{D, M}(representatives, fine)
end

# ---------------------------------------------------------------------------
# Cache + invalidation
# ---------------------------------------------------------------------------

"""
    ensure_neighbor_graph!(mesh::HierarchicalMesh{D, M}) -> NeighborGraph{D, M}

Build the neighbor graph if absent (or if invalidated by refinement) and
return it. Registers a refinement listener on first build that invalidates
the cache on every subsequent mesh modification.
"""
function ensure_neighbor_graph!(mesh::HierarchicalMesh{D, M}) where {D, M}
    g = mesh._cached_neighbor_graph
    F = 2 * D
    if g === nothing
        new_g = build_neighbor_graph(mesh)
        mesh._cached_neighbor_graph = new_g
        # Register a one-shot-style listener: zeroes the slot. Subsequent
        # accesses rebuild and re-register. (Cheaper than a permanent
        # listener that would fire on every refine even if no one cares
        # about the graph.)
        local_handle = Ref{ListenerHandle}(UInt64(0))
        cb = function (_event::RefinementEvent)
            mesh._cached_neighbor_graph = nothing
            unregister_refinement_listener!(mesh, local_handle[])
        end
        local_handle[] = register_refinement_listener!(mesh, cb)
        return new_g::NeighborGraph{D, M, F}
    else
        return g::NeighborGraph{D, M, F}
    end
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    face_neighbors(mesh::HierarchicalMesh{D,M}, i::Integer) -> NTuple{2D, UInt32}

For each axis-aligned face of cell `i` in the order
`(axis 1 lo, axis 1 hi, axis 2 lo, axis 2 hi, …, axis D lo, axis D hi)`,
return the leaf-neighbor's cell index, or `0` if on the domain boundary.

For a coarse cell with multiple fine neighbors (only possible when
`mesh.balanced == false`), one representative is returned; see
[`face_fine_neighbors`](@ref).
"""
function face_neighbors(mesh::HierarchicalMesh{D, M}, i::Integer) where {D, M}
    g = ensure_neighbor_graph!(mesh)
    return g.representatives[Int(i)]
end

"""
    face_fine_neighbors(mesh::HierarchicalMesh{D,M}, i::Integer, face::Integer)
        -> Vector{UInt32}

All fine-leaf neighbors of cell `i` on the given face index (1-based, in the
order described in [`face_neighbors`](@ref)). For a face with a single
neighbor or a domain boundary, returns at most a one-element vector. For a
face on the domain boundary, returns an empty vector.

Always returns a fresh vector that the caller may modify.
"""
function face_fine_neighbors(mesh::HierarchicalMesh{D, M}, i::Integer, face::Integer) where {D, M}
    g = ensure_neighbor_graph!(mesh)
    F = 2 * D
    1 <= face <= F || throw(BoundsError("face index $face out of range 1..$F"))
    key = (UInt32(i), UInt8(face))
    list = get(g.fine, key, nothing)
    if list !== nothing
        return copy(list)
    end
    rep = g.representatives[Int(i)][face]
    return rep == 0 ? UInt32[] : UInt32[rep]
end

# ---------------------------------------------------------------------------
# Sparsity API
# ---------------------------------------------------------------------------

using SparseArrays: SparseMatrixCSC, sparse, spzeros

"""
    cell_adjacency_sparsity(mesh::HierarchicalMesh; depth::Int=1, leaves_only::Bool=true)
        -> SparseMatrixCSC{Bool, Int32}

Boolean adjacency matrix where row/col `i` corresponds to leaf cell `i`
(when `leaves_only=true`) or to ALL cells (when `leaves_only=false`).
`M[i,j] == true` iff cells `i` and `j` are within `depth` face-hops of each
other. Symmetric, with the diagonal set to `true`.

Useful for downstream sparse-AD Jacobian sparsity patterns.
"""
function cell_adjacency_sparsity(mesh::HierarchicalMesh{D, M};
                                  depth::Int=1,
                                  leaves_only::Bool=true) where {D, M}
    depth >= 1 || throw(ArgumentError("depth must be ≥ 1"))
    g = ensure_neighbor_graph!(mesh)
    F = 2 * D

    # Determine the index set
    if leaves_only
        nodes = UInt32[i for i in 1:n_cells(mesh) if is_leaf(mesh.cells[i])]
    else
        nodes = UInt32[i for i in 1:n_cells(mesh)]
    end
    N = length(nodes)
    # Map cell-index -> 1..N row position
    pos = Dict{UInt32, Int32}()
    for (k, c) in enumerate(nodes)
        pos[c] = Int32(k)
    end

    rows = Int32[]
    cols = Int32[]
    sizehint!(rows, N)
    sizehint!(cols, N)

    # BFS up to `depth` from each seed node
    visited = Vector{Bool}(undef, N)
    for k in 1:N
        seed = nodes[k]
        # reset
        @inbounds for q in 1:N
            visited[q] = false
        end
        # frontier as (cell, hops)
        frontier = Tuple{UInt32, Int}[(seed, 0)]
        head = 1
        visited[k] = true
        push!(rows, Int32(k)); push!(cols, Int32(k))  # diagonal
        while head <= length(frontier)
            (c, h) = frontier[head]; head += 1
            h >= depth && continue
            # Walk all face-neighbors
            for f in 1:F
                # representative
                rep = g.representatives[Int(c)][f]
                if rep != 0
                    _adj_visit!(rep, k, visited, pos, rows, cols, frontier, h+1)
                end
                # any extra fine neighbors
                key = (c, UInt8(f))
                fl = get(g.fine, key, nothing)
                if fl !== nothing
                    for nb in fl
                        nb == rep && continue
                        _adj_visit!(nb, k, visited, pos, rows, cols, frontier, h+1)
                    end
                end
                # If the cell is non-leaf or face was a coarse cell looking
                # in, we may need to include the coarse-side neighbour too.
                # When `leaves_only=false` and the seed is non-leaf, we still
                # walk its (empty) neighbor entries — non-leaf cells in our
                # graph have all-zero rows, so no edges are produced. That
                # is consistent with the "treat the leaf cells as the graph"
                # contract; non-leaf cells appear only as isolated diagonal
                # entries plus whatever transient connections come from
                # subtree-internal hops.
            end
        end
    end

    # Symmetrize (coarse cell saw fine neighbours but fine cell, if it's a
    # neighbour-of-neighbour, may not have seen the coarse one through a
    # representative; the fine dict already captures all fine sides).
    nnz = length(rows)
    sym_rows = Vector{Int32}(undef, 2*nnz)
    sym_cols = Vector{Int32}(undef, 2*nnz)
    @inbounds for q in 1:nnz
        sym_rows[q]      = rows[q]
        sym_cols[q]      = cols[q]
        sym_rows[q+nnz]  = cols[q]
        sym_cols[q+nnz]  = rows[q]
    end
    vals = trues(length(sym_rows))
    return sparse(sym_rows, sym_cols, vals, N, N, |)
end

@inline function _adj_visit!(neighbor::UInt32, seed_k::Int, visited, pos,
                              rows::Vector{Int32}, cols::Vector{Int32},
                              frontier::Vector{Tuple{UInt32, Int}}, new_h::Int)
    p = get(pos, neighbor, Int32(0))
    p == 0 && return
    if !visited[p]
        visited[p] = true
        push!(rows, Int32(seed_k))
        push!(cols, p)
        push!(frontier, (neighbor, new_h))
    end
    return
end

# ---------------------------------------------------------------------------
# Periodic / boundary-condition aware neighbor wiring (PR-D)
# ---------------------------------------------------------------------------

"""
    face_neighbors_with_bcs(mesh::HierarchicalMesh{D,M}, i::Integer,
                             frame_bcs) -> NTuple{2D, UInt32}

Boundary-condition-aware variant of [`face_neighbors`](@ref). Builds the
underlying neighbor graph (lazily, like the plain `face_neighbors`) and
post-processes the result for each periodic axis declared in
`frame_bcs::FrameBoundaries{D}`:

- Lo-side boundary entries on a periodic axis are replaced with the leaf
  on the opposite (hi) side whose other-axes unit-cube extent overlaps
  with `i`'s.
- Hi-side boundary entries are replaced symmetrically.
- Non-periodic-axis boundary entries are left as `0`.

Non-periodic BC kinds (`INFLOW`, `OUTFLOW`, `REFLECTING`, `DIRICHLET`)
are not consumed here — those are handled by PDE-level code that reads
the `FrameBoundaries` directly.

The `frame_bcs` argument is typed as `Any` to avoid an upward dependency
from `Mesh` on the `Overlap` submodule where `FrameBoundaries` lives.
The function looks for `is_periodic_axis(frame_bcs, axis)` via duck
typing; either `BoundaryConditions.is_periodic_axis` (on a
`BoundarySpec`) or a `FrameBoundaries` instance will work.
"""
function face_neighbors_with_bcs(mesh::HierarchicalMesh{D, M}, i::Integer,
                                  frame_bcs) where {D, M}
    g = ensure_neighbor_graph!(mesh)
    base = g.representatives[Int(i)]

    # Identify which axes are periodic. Use the public is_periodic_axis
    # method that BoundaryConditions and FrameBoundaries both export.
    any_periodic = false
    @inbounds for d in 1:D
        if _bc_is_periodic_axis(frame_bcs, d)
            any_periodic = true
            break
        end
    end
    any_periodic || return base

    # Get this leaf's unit-cube box. If it's not a leaf, we have nothing
    # sensible to wire; return the (zero) entry.
    is_leaf(mesh.cells[Int(i)]) || return base

    (a_lo, a_hi) = _unit_box(mesh, Int(i))

    # Collect leaves once; we need to scan opposite-wall leaves for matches.
    # For typical mesh sizes the cost is O(N_leaves * D), in line with the
    # rest of the neighbor-graph pipeline.
    leaves = UInt32[]
    leaf_lo = NTuple{D, Float64}[]
    leaf_hi = NTuple{D, Float64}[]
    @inbounds for k in 1:n_cells(mesh)
        if is_leaf(mesh.cells[k])
            (lo, hi) = _unit_box(mesh, k)
            push!(leaves, UInt32(k))
            push!(leaf_lo, lo)
            push!(leaf_hi, hi)
        end
    end

    # Output tuple as a mutable array; we'll re-tuple at the end.
    out = collect(base)

    @inbounds for d in 1:D
        _bc_is_periodic_axis(frame_bcs, d) || continue
        face_lo_idx = 2 * d - 1
        face_hi_idx = 2 * d

        on_lo_boundary = a_lo[d] <= _NEIGHBOR_EPS
        on_hi_boundary = a_hi[d] >= 1.0 - _NEIGHBOR_EPS

        if on_lo_boundary && out[face_lo_idx] == 0
            # Find the leaf on the hi-wall that overlaps in the other axes.
            best = UInt32(0)
            for k in eachindex(leaves)
                # Candidate must touch the hi wall on axis d.
                leaf_hi[k][d] >= 1.0 - _NEIGHBOR_EPS || continue
                ok = true
                for d2 in 1:D
                    d2 == d && continue
                    if !_open_overlap(a_lo[d2], a_hi[d2],
                                       leaf_lo[k][d2], leaf_hi[k][d2])
                        ok = false
                        break
                    end
                end
                ok || continue
                idx = leaves[k]
                if best == 0 || idx < best
                    best = idx
                end
            end
            best != 0 && (out[face_lo_idx] = best)
        end

        if on_hi_boundary && out[face_hi_idx] == 0
            best = UInt32(0)
            for k in eachindex(leaves)
                leaf_lo[k][d] <= _NEIGHBOR_EPS || continue
                ok = true
                for d2 in 1:D
                    d2 == d && continue
                    if !_open_overlap(a_lo[d2], a_hi[d2],
                                       leaf_lo[k][d2], leaf_hi[k][d2])
                        ok = false
                        break
                    end
                end
                ok || continue
                idx = leaves[k]
                if best == 0 || idx < best
                    best = idx
                end
            end
            best != 0 && (out[face_hi_idx] = best)
        end
    end

    return ntuple(f -> out[f], Val(2*D))
end

# Duck-typed dispatcher: works for both BoundarySpec tuples (via the
# BoundaryConditions function we imported as `_is_periodic_axis_spec`) and
# `FrameBoundaries`-like objects exposing `is_periodic_axis(fb, axis)`.
@inline _bc_is_periodic_axis(spec::NTuple{D, NTuple{2, BCKind}},
                              axis::Integer) where {D} =
    _is_periodic_axis_spec(spec, axis)

# Generic fallback: any object with a `.spec` field that is a BoundarySpec.
@inline _bc_is_periodic_axis(fb, axis::Integer) =
    _is_periodic_axis_spec(fb.spec, axis)

