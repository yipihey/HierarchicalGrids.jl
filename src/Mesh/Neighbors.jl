"""
    Neighbors

Face-neighbor adjacency for `HierarchicalMesh`.

# Algorithm: path-walking face-neighbor finder

For each leaf cell `i` and each face `(axis d, side s ∈ {lo, hi})`:

1. Walk up the parent chain from `i`. At each step, examine the cell's own
   `position_in_parent` on axis `d`. The first ancestor `e*` whose split
   mask includes axis `d` AND whose own `pos_d == 1 - s` marks the level
   at which the path crosses axis `d` on side `s`.
2. The mirror sibling `M` of `e*` (same parent, axis-`d` position flipped
   to `s`) is the root of the subtree containing the face-neighbor.
3. Descend `M`, mirroring the upward path's axis-`d` positions but
   flipping side `s ↔ 1 - s` at every step where axis `d` is split. At
   each step, if the current node is a leaf, that's the (potentially
   coarse) face-neighbor. If we exhaust the path while still at a non-leaf
   interior node, enumerate ALL descendant leaves on the matching face —
   those are the fine neighbors (multi-many on unbalanced meshes).
4. If the upward walk reaches the root without finding such an ancestor,
   the face is on the domain boundary (returns 0).

This runs in `O(N · D · depth_max)` worst case, asymptotically replacing
the previous axis-bucket strategy that was `O(N²)` worst-case (when many
leaves quantized to the same bucket key on degenerate axis-aligned cuts).

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

# ---------------------------------------------------------------------------
# Path-walking helpers
# ---------------------------------------------------------------------------

# Position-in-parent on axis d for a cell with given (sibling_index, mask).
# Returns 0 (lo half) or 1 (hi half). If axis d is NOT in the mask, returns
# 0 — but in that case axis d is "passed through" (the cell occupies its
# parent's full axis-d extent), so the caller must check `_axis_in_mask`.
@inline function _pos_on_axis(sib::Unsigned, mask::Unsigned, d::Integer)
    expanded = pdep(UInt32(sib), UInt32(mask))
    return Int((expanded >> (d - 1)) & UInt32(1))
end

@inline function _axis_in_mask(mask::Unsigned, d::Integer)
    return (UInt32(mask) >> (d - 1)) & UInt32(1) != 0
end

# Flip the axis-d bit of a sibling index, given the cell's mask. Used to
# produce the "mirror" sibling that is the same as `sib` on every other
# split axis but on the opposite half of axis d.
@inline function _flip_axis(sib::Unsigned, mask::Unsigned, d::Integer)
    M_ = typeof(sib)
    # The PEXT/PDEP roundtrip would re-pack bits; cheaper to xor the
    # appropriate bit position directly.
    # The bit position of axis d within `sib` is the popcount of mask bits
    # below d.
    bit_pos = count_ones(UInt32(mask) & ((UInt32(1) << (d - 1)) - UInt32(1)))
    return sib ⊻ (M_(1) << bit_pos)
end

# Find the child of `parent_idx` (a non-leaf cell) whose sibling_index
# matches `target_sib`. Returns 0 if no such child (e.g., split mask
# disagreement).
function _find_child_with_sib(mesh::HierarchicalMesh{D, M},
                              parent_idx::UInt32, target_sib::M) where {D, M}
    children = find_children(mesh, parent_idx)
    for ci in children
        if mesh.cells[ci].sibling_index == target_sib
            return ci
        end
    end
    return UInt32(0)
end

# Enumerate all leaves of `subtree_root`'s subtree that touch axis-d on the
# `target_side` (0 = lo, 1 = hi) face of `subtree_root`. At each non-leaf
# encountered, descend only into children whose pos on axis d is the
# target side (when axis d is split there) or all children (when axis d
# is unsplit at that level — the children pass through axis d).
#
# `out` accumulates the leaf cell indices.
function _collect_face_leaves!(out::Vector{UInt32},
                                mesh::HierarchicalMesh{D, M},
                                subtree_root::UInt32,
                                d::Integer, target_side::Int) where {D, M}
    cell = mesh.cells[subtree_root]
    if is_leaf(cell)
        push!(out, subtree_root)
        return out
    end
    children = find_children(mesh, subtree_root)
    mask = mesh.cells[children[1]].split_mask
    if _axis_in_mask(mask, d)
        # Only follow children with pos_d == target_side
        for ci in children
            child_cell = mesh.cells[ci]
            if _pos_on_axis(child_cell.sibling_index, mask, d) == target_side
                _collect_face_leaves!(out, mesh, ci, d, target_side)
            end
        end
    else
        # Axis d not split here; recurse into all children.
        for ci in children
            _collect_face_leaves!(out, mesh, ci, d, target_side)
        end
    end
    return out
end

# Locate the face-neighbor (or neighbors) of leaf `i` on face (axis d,
# side s). Returns a tuple (representative, fine_list_or_nothing) where
# `representative` is 0 if the face is on the domain boundary, otherwise
# the lowest-indexed neighbor leaf. `fine_list_or_nothing` is `nothing`
# unless there are >1 fine neighbors, in which case it is a sorted
# Vector{UInt32}.
function _path_face_neighbor(mesh::HierarchicalMesh{D, M},
                              i::UInt32, d::Integer, s::Int) where {D, M}
    parents = mesh._parents

    # ---- Phase 1: walk up to find e*. We collect the path from leaf
    # toward root, recording (sib, mask) at each step. The walk stops when
    # we encounter an ancestor whose own pos on axis d is `1 - s`. That
    # ancestor exists iff at some point our path crossed axis d on side s.
    chain_sibs  = M[]      # sibling_indices encountered, leaf-first
    chain_masks = M[]      # split_masks encountered, leaf-first

    current = i
    e_star::UInt32 = UInt32(0)        # the ancestor cell (cell id)
    parent_of_e_star::UInt32 = UInt32(0)
    other_side = 1 - s
    while current != ROOT_PARENT && parents[current] != ROOT_PARENT
        cell = mesh.cells[current]
        sib  = cell.sibling_index
        mask = cell.split_mask
        if _axis_in_mask(mask, d) && _pos_on_axis(sib, mask, d) == other_side
            e_star = current
            parent_of_e_star = parents[current]
            break
        end
        push!(chain_sibs, sib)
        push!(chain_masks, mask)
        current = parents[current]
    end

    if e_star == 0
        # Walked to root without crossing — domain boundary.
        return (UInt32(0), nothing)
    end

    # ---- Phase 2: navigate to the mirror sibling M of e*.
    e_cell = mesh.cells[e_star]
    e_mask = e_cell.split_mask
    e_sib  = e_cell.sibling_index
    mirror_sib = _flip_axis(e_sib, e_mask, d)
    M_idx = _find_child_with_sib(mesh, parent_of_e_star, mirror_sib)
    if M_idx == 0
        # Should not happen: siblings are always present when the parent
        # was refined. Defensive bail-out.
        return (UInt32(0), nothing)
    end

    # ---- Phase 3: descend M, consuming chain entries from the END
    # (closest to e*) down toward i (closest to leaf). At each entry we
    # flip axis d.
    cur = M_idx
    # The chain is leaf-first; we want to consume from the entry just
    # below e*, which is at the END of the chain (the last pushed entry
    # was the cell directly below e*; the first pushed was i itself).
    # Wait: chain_sibs[1] = i's own sib/mask, chain_sibs[end] = the cell
    # directly below e* (i.e., e*'s child on the i-path). To descend from
    # M, we want to start with the entry that is e*'s child on i's path,
    # then go deeper. So iterate end -> 1.
    k = length(chain_sibs)
    while k >= 1
        cur_cell = mesh.cells[cur]
        if is_leaf(cur_cell)
            # Coarse neighbor; we found a leaf before exhausting the path.
            return (cur, nothing)
        end
        # Compute the target sibling for the descent step.
        path_sib  = chain_sibs[k]
        path_mask = chain_masks[k]
        cur_mask  = cur_cell.split_mask
        if cur_mask == path_mask && _axis_in_mask(cur_mask, d)
            # Same split pattern: just flip axis d in the path's sibling.
            target_sib = _flip_axis(path_sib, cur_mask, d)
            child_idx = _find_child_with_sib(mesh, cur, target_sib)
            if child_idx == 0
                # Mismatch (shouldn't normally happen with same mask, but
                # fall back to face enumeration).
                break
            end
            cur = child_idx
            k -= 1
        else
            # Split-mask divergence between path and current subtree
            # (anisotropic refinement variation). Fall back to face
            # enumeration of the matching face from `cur`.
            break
        end
    end

    # ---- Phase 4: at `cur`, collect all leaf descendants on the matching
    # face. The matching face on `cur` is the (1 - s)-side of axis d (the
    # face that touches our cell's side-s face).
    cur_cell = mesh.cells[cur]
    if is_leaf(cur_cell)
        return (cur, nothing)
    end
    matching_side = 1 - s
    collected = UInt32[]
    _collect_face_leaves!(collected, mesh, cur, d, matching_side)
    if isempty(collected)
        return (UInt32(0), nothing)
    elseif length(collected) == 1
        return (collected[1], nothing)
    else
        sort!(collected)
        rep = collected[1]
        return (rep, collected)
    end
end

"""
    build_neighbor_graph(mesh::HierarchicalMesh{D, M};
                          backend = nothing) -> NeighborGraph{D, M}

Build the face-neighbor adjacency by walking the parent chain of each leaf.
For each leaf and each face `(axis d, side s)`, locate the lowest ancestor
whose path crosses axis `d` on side `s`, navigate to its mirror sibling,
and descend the mirror subtree to find the face-neighbor (or enumerate
multiple fine neighbors at hanging-node faces).

Worst-case cost: `O(N · D · depth_max)` for the per-leaf path walks plus
`O(adjacency)` for fine-leaf enumeration. The previous bucket-based
implementation was `O(N²)` worst-case.

# Parallelism (PR-2)

The per-leaf path walks are independent and embarrassingly parallel. The
build is parallelized with a two-phase pattern:

1. **Phase 1 (parallel):** the leaf range is partitioned into chunks; each
   task walks its chunk, writing its `representatives[i]` slot directly
   (per-`i` writes are race-free) and accumulating any hanging-node `fine`
   entries into a per-task `Vector{Pair{...}}`.
2. **Phase 2 (sequential):** the per-task vectors are walked on the
   calling thread and inserted into the single shared `fine::Dict`. The
   merge cost is `O(adjacency)`, which is asymptotically dominated by
   Phase 1, so a serial merge keeps the implementation simple without
   sacrificing scalability.

`ensure_caches!(mesh)` is forced before the parallel fan-out so per-task
accesses see a populated, read-only cache (lazy first-access on the
parent/level caches is not thread-safe).

The `backend` kwarg accepts an `AbstractParallelBackend`; when `nothing`,
the process-global `default_backend()` is consulted. Pass `Sequential()`
to get the original byte-identical serial loop.
"""
function build_neighbor_graph(mesh::HierarchicalMesh{D, M};
                                backend = nothing) where {D, M}
    ensure_caches!(mesh)
    n = n_cells(mesh)
    F = 2 * D
    representatives = Vector{NTuple{2*D, UInt32}}(undef, n)
    fine = Dict{Tuple{UInt32, UInt8}, Vector{UInt32}}()

    if n == 0
        return NeighborGraph{D, M}(representatives, fine)
    end

    # Resolve the backend lazily — Threading is loaded AFTER Mesh, so we
    # can't statically reference its types here. We grab the verbs from
    # the parent module at call time.
    HG = parentmodule(@__MODULE__)
    eff_backend = backend === nothing ? HG.Threading.default_backend() : backend

    return _build_neighbor_graph_impl(mesh, representatives, fine, n, F,
                                       eff_backend, HG)
end

# Per-chunk worker: writes `representatives[i]` slots directly and returns
# a vector of (key, value) pairs for the fine-neighbor dict. Each task
# gets its own per-task buffer for the inner face_buf.
function _neighbor_graph_chunk!(mesh::HierarchicalMesh{D, M},
                                  representatives::Vector{NTuple{F, UInt32}},
                                  cell_range,
                                  ) where {D, M, F}
    fine_pairs = Pair{Tuple{UInt32, UInt8}, Vector{UInt32}}[]
    face_buf = Vector{UInt32}(undef, F)
    @inbounds for i in cell_range
        i_int = Int(i)
        if !is_leaf(mesh.cells[i_int])
            representatives[i_int] = ntuple(_ -> UInt32(0), Val(F))
            continue
        end
        i32 = UInt32(i_int)
        for f in 1:F
            face_buf[f] = UInt32(0)
        end
        for d in 1:D
            # lo side (s = 0), face index 2d - 1
            (rep_lo, fl_lo) = _path_face_neighbor(mesh, i32, d, 0)
            face_buf[2*d - 1] = rep_lo
            if fl_lo !== nothing
                push!(fine_pairs, (i32, UInt8(2*d - 1)) => fl_lo)
            end
            # hi side (s = 1), face index 2d
            (rep_hi, fl_hi) = _path_face_neighbor(mesh, i32, d, 1)
            face_buf[2*d] = rep_hi
            if fl_hi !== nothing
                push!(fine_pairs, (i32, UInt8(2*d)) => fl_hi)
            end
        end
        representatives[i_int] = ntuple(f -> face_buf[f], Val(F))
    end
    return fine_pairs
end

# Sequential serial path: identical byte-for-byte to the legacy loop.
function _build_neighbor_graph_serial!(mesh::HierarchicalMesh{D, M},
                                         representatives::Vector{NTuple{F, UInt32}},
                                         fine::Dict{Tuple{UInt32, UInt8}, Vector{UInt32}},
                                         n::Integer,
                                         ) where {D, M, F}
    pairs = _neighbor_graph_chunk!(mesh, representatives, 1:n)
    @inbounds for kv in pairs
        fine[kv.first] = kv.second
    end
    return nothing
end

# Phase-2 merge of an iterable of per-task Pair vectors into the shared dict.
function _merge_fine_pairs!(fine::Dict{Tuple{UInt32, UInt8}, Vector{UInt32}},
                              part_lists)
    @inbounds for part in part_lists
        for kv in part
            fine[kv.first] = kv.second
        end
    end
    return nothing
end

function _build_neighbor_graph_impl(mesh::HierarchicalMesh{D, M},
                                      representatives::Vector{NTuple{F, UInt32}},
                                      fine::Dict{Tuple{UInt32, UInt8}, Vector{UInt32}},
                                      n::Integer, ::Integer,
                                      backend, HG) where {D, M, F}
    # Sequential fast-path: avoid any task plumbing, byte-equal to legacy.
    if backend isa HG.Threading.Sequential
        _build_neighbor_graph_serial!(mesh, representatives, fine, n)
        return NeighborGraph{D, M}(representatives, fine)
    end

    # Parallel path. Partition the cell range into chunks (one task per
    # chunk) and run the inner kernel under the supplied backend. We use
    # `parallel_chunked` so that each task gets a contiguous range and a
    # private `face_buf` allocation; per-task fine-pair vectors are
    # collected via a thread-safe push under a lock (chunk count is small,
    # so contention is negligible — this beats reaching for tmapreduce
    # since we're collecting Vectors, not reducing).
    n_chunks = max(1, min(Int(n), Threads.nthreads()))
    # Manually partition (avoid depending on Mesh + Threading both being
    # loaded for the chunked verb).
    chunks = HG.Threading.partition_for_threads(mesh, n_chunks)
    parts = Vector{Vector{Pair{Tuple{UInt32, UInt8}, Vector{UInt32}}}}(undef,
                                                                       length(chunks))
    HG.Threading.parallel_chunked(backend,
        (m, chunk) -> begin
            parts[Int(chunk.chunk_id)] =
                _neighbor_graph_chunk!(m, representatives, chunk.cell_range)
        end,
        mesh, length(chunks))

    # Phase 2: merge per-task fine-pair vectors into the shared dict.
    _merge_fine_pairs!(fine, parts)

    return NeighborGraph{D, M}(representatives, fine)
end

# ---------------------------------------------------------------------------
# Cache + invalidation
# ---------------------------------------------------------------------------

"""
    ensure_neighbor_graph!(mesh::HierarchicalMesh{D, M};
                            backend = nothing) -> NeighborGraph{D, M}

Build the neighbor graph if absent (or if invalidated by refinement) and
return it. Registers a refinement listener on first build that invalidates
the cache on every subsequent mesh modification.

The `backend` kwarg is forwarded to [`build_neighbor_graph`](@ref). The
listener wiring itself is single-threaded — only the build is parallel.
"""
function ensure_neighbor_graph!(mesh::HierarchicalMesh{D, M};
                                  backend = nothing) where {D, M}
    g = mesh._cached_neighbor_graph
    F = 2 * D
    if g === nothing
        new_g = build_neighbor_graph(mesh; backend = backend)
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

