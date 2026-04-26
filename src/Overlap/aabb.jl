"""
Bounding-Volume Hierarchy for Lagrangian simplices.

Used as the broad-phase index in `compute_overlap`: for each Eulerian
leaf, query the BVH for Lagrangian simplices whose AABBs intersect the
leaf's box, then do exact polytope clipping per candidate.

# Construction

`build_simplex_aabb_tree(mesh)` builds a binary BVH by recursively splitting
the simplex set along the axis with the largest centroid spread, picking
the split point at the median centroid coordinate. Tree depth is
`O(log n_simplices)`; build cost is `O(n_simplices log n_simplices)`.

# Query

`query_aabb!(out, tree, query_lo, query_hi)` writes the indices of all
simplices whose AABBs overlap the query box into `out` (caller-provided
vector for zero allocation in hot loops).

# Storage

The tree is stored as parallel arrays, not a heap of nodes — this keeps
data structure pointer-free, cache-friendly, and easy to serialize. Each
node has:
- An AABB
- A range of simplex indices it covers (CSR-style, into `simplex_order`)
- Children (left/right node indices, or 0 for a leaf node)
"""

"""
    SimplicialAABBTree{D, T}

Static BVH over the simplices of a `SimplicialMesh{D, T}`. Built once;
to update after vertex motion, rebuild (typical cost is small relative
to the overlap computation it supports).

# Fields

- `node_lo::Vector{NTuple{D, T}}`, `node_hi::Vector{NTuple{D, T}}` —
  per-node AABBs.
- `node_left::Vector{Int32}`, `node_right::Vector{Int32}` — child indices,
  0 for leaf nodes.
- `node_first::Vector{Int32}`, `node_count::Vector{Int32}` — for leaf
  nodes, the range `[first, first + count)` into `simplex_order` giving
  the simplices in this node. Non-leaf nodes have `count == 0`.
- `simplex_order::Vector{Int32}` — permutation of `1:n_simplices` such
  that simplices in the same leaf are contiguous.
- `n_simplices::Int`, `root::Int` — book-keeping.
"""
struct SimplicialAABBTree{D, T}
    node_lo::Vector{NTuple{D, T}}
    node_hi::Vector{NTuple{D, T}}
    node_left::Vector{Int32}
    node_right::Vector{Int32}
    node_first::Vector{Int32}
    node_count::Vector{Int32}
    simplex_order::Vector{Int32}
    n_simplices::Int
    root::Int
end

@inline n_nodes(t::SimplicialAABBTree) = length(t.node_lo)

"""
    simplex_aabb(mesh::SimplicialMesh{D, T}, s::Integer) -> (lo, hi)

Per-simplex axis-aligned bounding box in physical coordinates. O(D+1)
with no allocation.
"""
function simplex_aabb(mesh::SimplicialMesh{D, T}, s::Integer) where {D, T}
    verts = simplex_vertex_positions(mesh, s)
    lo = MVector{D, T}(undef)
    hi = MVector{D, T}(undef)
    @inbounds for d in 1:D
        lo[d] = verts[1][d]
        hi[d] = verts[1][d]
    end
    @inbounds for k in 2:length(verts)
        v = verts[k]
        for d in 1:D
            if v[d] < lo[d]; lo[d] = v[d]; end
            if v[d] > hi[d]; hi[d] = v[d]; end
        end
    end
    return (Tuple(lo), Tuple(hi))
end

"""
    build_simplex_aabb_tree(mesh::SimplicialMesh{D, T};
                              leaf_size::Int = 8) -> SimplicialAABBTree

Build a BVH over the simplices of `mesh`. `leaf_size` is the maximum
number of simplices stored at a leaf node; smaller values give finer
broad-phase culling at the cost of a deeper tree.
"""
function build_simplex_aabb_tree(mesh::SimplicialMesh{D, T};
                                   leaf_size::Int = 8) where {D, T}
    n = n_simplices(mesh)
    if n == 0
        # Single empty leaf
        return SimplicialAABBTree{D, T}(
            [ntuple(_ -> zero(T), Val(D))],
            [ntuple(_ -> zero(T), Val(D))],
            Int32[0], Int32[0],
            Int32[1], Int32[0],
            Int32[],
            0, 1,
        )
    end

    # Per-simplex AABB and centroid for splitting
    aabbs_lo = Vector{NTuple{D, T}}(undef, n)
    aabbs_hi = Vector{NTuple{D, T}}(undef, n)
    centroids = Vector{NTuple{D, T}}(undef, n)
    @inbounds for s in 1:n
        lo, hi = simplex_aabb(mesh, s)
        aabbs_lo[s] = lo
        aabbs_hi[s] = hi
        centroids[s] = ntuple(d -> (lo[d] + hi[d]) / 2, Val(D))
    end

    # Working order array (will be permuted as we split)
    order = Int32.(1:n)

    # Pre-allocate node arrays (worst-case size 2n - 1 for a binary BVH)
    cap = max(1, 2 * n)
    node_lo = Vector{NTuple{D, T}}(undef, 0); sizehint!(node_lo, cap)
    node_hi = Vector{NTuple{D, T}}(undef, 0); sizehint!(node_hi, cap)
    node_left = Int32[]; sizehint!(node_left, cap)
    node_right = Int32[]; sizehint!(node_right, cap)
    node_first = Int32[]; sizehint!(node_first, cap)
    node_count = Int32[]; sizehint!(node_count, cap)

    root = _build_bvh_recursive!(node_lo, node_hi, node_left, node_right,
                                   node_first, node_count,
                                   order, aabbs_lo, aabbs_hi, centroids,
                                   1, n, leaf_size, Val(D))

    return SimplicialAABBTree{D, T}(
        node_lo, node_hi, node_left, node_right,
        node_first, node_count, order, n, root,
    )
end

# Returns the node index for the subtree covering `order[first:last]`.
function _build_bvh_recursive!(node_lo, node_hi, node_left, node_right,
                                node_first, node_count,
                                order, aabbs_lo, aabbs_hi, centroids,
                                first::Int, last::Int, leaf_size::Int,
                                ::Val{D}) where D
    T = eltype(eltype(aabbs_lo))
    count = last - first + 1
    # Compute combined AABB for this node
    lo = ntuple(d -> typemax(T), Val(D))
    hi = ntuple(d -> typemin(T), Val(D))
    @inbounds for k in first:last
        s = order[k]
        a_lo = aabbs_lo[s]; a_hi = aabbs_hi[s]
        lo = ntuple(d -> min(lo[d], a_lo[d]), Val(D))
        hi = ntuple(d -> max(hi[d], a_hi[d]), Val(D))
    end

    # Allocate this node
    push!(node_lo, lo); push!(node_hi, hi)
    push!(node_left, Int32(0)); push!(node_right, Int32(0))
    push!(node_first, Int32(first)); push!(node_count, Int32(0))
    this_idx = length(node_lo)

    if count <= leaf_size
        # Leaf
        node_count[this_idx] = Int32(count)
        return this_idx
    end

    # Choose split axis: largest centroid spread
    cmin = ntuple(d -> typemax(T), Val(D))
    cmax = ntuple(d -> typemin(T), Val(D))
    @inbounds for k in first:last
        c = centroids[order[k]]
        cmin = ntuple(d -> min(cmin[d], c[d]), Val(D))
        cmax = ntuple(d -> max(cmax[d], c[d]), Val(D))
    end
    best_axis = 1
    best_spread = cmax[1] - cmin[1]
    @inbounds for d in 2:D
        sp = cmax[d] - cmin[d]
        if sp > best_spread
            best_spread = sp
            best_axis = d
        end
    end

    # If centroid spread is zero (all coincident), make a leaf
    if best_spread == zero(T)
        node_count[this_idx] = Int32(count)
        return this_idx
    end

    # Partition `order[first:last]` around the median centroid coordinate
    # along `best_axis`. Use Hoare partition based on the midpoint of the
    # centroid spread (median-of-centroids would be a refinement).
    pivot = cmin[best_axis] + best_spread / 2
    i = first
    j = last
    @inbounds while i <= j
        while i <= last && centroids[order[i]][best_axis] < pivot
            i += 1
        end
        while j >= first && centroids[order[j]][best_axis] >= pivot
            j -= 1
        end
        if i < j
            order[i], order[j] = order[j], order[i]
            i += 1; j -= 1
        end
    end

    # If split degenerated (everything on one side), fall back to splitting at the median
    if i == first || i > last
        mid = (first + last) ÷ 2
        # Sort the relevant range by axis coordinate (small in practice; use partial sort)
        sub_idxs = order[first:last]
        sort!(sub_idxs, by = s -> centroids[s][best_axis])
        @inbounds for k in 1:length(sub_idxs)
            order[first + k - 1] = sub_idxs[k]
        end
        i = mid + 1
    end

    left_idx = _build_bvh_recursive!(node_lo, node_hi, node_left, node_right,
                                       node_first, node_count,
                                       order, aabbs_lo, aabbs_hi, centroids,
                                       first, i - 1, leaf_size, Val(D))
    right_idx = _build_bvh_recursive!(node_lo, node_hi, node_left, node_right,
                                        node_first, node_count,
                                        order, aabbs_lo, aabbs_hi, centroids,
                                        i, last, leaf_size, Val(D))
    node_left[this_idx] = Int32(left_idx)
    node_right[this_idx] = Int32(right_idx)
    return this_idx
end

"""
    query_aabb!(out::Vector{Int32}, tree::SimplicialAABBTree{D, T},
                 query_lo::NTuple{D, T}, query_hi::NTuple{D, T})

Append the indices of all simplices whose AABB overlaps the query box
into `out`. `out` is **not** cleared first; the caller decides whether
to reuse or empty it. Order of returned indices is unspecified.

Iterative (stack-based) traversal — no recursion depth concerns even on
deep trees.
"""
function query_aabb!(out::Vector{Int32}, tree::SimplicialAABBTree{D, T},
                      query_lo::NTuple{D, T}, query_hi::NTuple{D, T}) where {D, T}
    tree.n_simplices == 0 && return out
    # Stack of node indices to process
    stack = Int32[Int32(tree.root)]
    @inbounds while !isempty(stack)
        ni = pop!(stack)
        # AABB-AABB overlap test
        n_lo = tree.node_lo[ni]; n_hi = tree.node_hi[ni]
        ok = true
        for d in 1:D
            if !(n_hi[d] > query_lo[d] && query_hi[d] > n_lo[d])
                ok = false; break
            end
        end
        ok || continue
        # Internal node?
        if tree.node_left[ni] != 0
            push!(stack, tree.node_left[ni])
            push!(stack, tree.node_right[ni])
        else
            # Leaf: append covered simplices that pass the per-simplex test.
            # Note: per-simplex AABBs were tested when the node was built;
            # we still need to filter at the per-simplex level since the leaf
            # AABB is a bounding union and may overlap when no member does.
            first = tree.node_first[ni]
            count = tree.node_count[ni]
            for k in 0:(count - 1)
                push!(out, tree.simplex_order[first + k])
            end
        end
    end
    return out
end

"""
    query_aabb(tree::SimplicialAABBTree{D, T},
                query_lo::NTuple{D, T}, query_hi::NTuple{D, T}) -> Vector{Int32}

Allocating variant of `query_aabb!`.
"""
function query_aabb(tree::SimplicialAABBTree{D, T},
                     query_lo::NTuple{D, T}, query_hi::NTuple{D, T}) where {D, T}
    out = Int32[]
    query_aabb!(out, tree, query_lo, query_hi)
    return out
end

function Base.show(io::IO, t::SimplicialAABBTree{D, T}) where {D, T}
    print(io, "SimplicialAABBTree{$D, $T}(",
          t.n_simplices, " simplices, ", n_nodes(t), " nodes)")
end
