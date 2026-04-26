"""
    Mesh

Layer 1: cell metadata and hierarchical tree structure for adaptive grids.

The mesh is **purely structural** — it contains only geometric information
(cell hierarchy, refinement metadata) and no physics, no fields, no particles.
Field data and physics-specific state live in separate modules and are
associated with the mesh by parallel-array indexing.

# Canonical per-cell storage

Each cell carries the **minimum irreducible information**:

- `sibling_index`: which child this is among its parent's children (0..2^k-1
  where k = number of split axes).
- `split_mask`: which axes were split when this cell's siblings were created.
- `flags`: leaf/boundary/dirty bits.

Total: 2-3 bytes per cell depending on dimension. Compare to typical AMR
codes that carry 100+ bytes per cell of mixed metadata, gravity state,
buffering, etc.

# Caches

Derived data (per-axis level, parent index, subtree size) is stored as
parallel arrays alongside the cells, but is **lazy** — invalidated when the
mesh changes, rebuilt on first access. Cached fields are not part of the
canonical cell representation; they're an optimization.

# Anisotropic refinement

A cell can be split along any subset of axes (encoded in split_mask). For
fully-isotropic refinement (the common case), all axes are split together.
The framework handles both with the same code; the fully-isotropic case
gets fast-path treatment when detected.

# Hierarchical relative coordinates

A cell's identity is its position in the tree (sibling indices walking from
the canonical reference). Absolute coordinates are derivable by walking but
rarely needed — operations between cells use their LCA-relative frame, which
keeps bit widths bounded by depth-difference rather than absolute depth.

This means deep zoom doesn't widen integer types anywhere. Adding a coarser
boundary or finer refined region is non-disruptive.
"""
module Mesh

using ..BitPrimitives

export CellMeta, HierarchicalMesh
export FLAG_LEAF, FLAG_BOUNDARY, FLAG_DIRTY
export is_leaf, is_boundary, is_dirty, set_flag, clear_flag
export sibling_index, split_mask, flags
export FULLY_ISOTROPIC_MASK, isotropic_mask
export n_cells, root_cell_index
export find_parent, find_children, find_lca, children_count
export level_of, position_in_parent
export rebuild_caches!, invalidate_caches!, caches_valid
export refine_cells!, coarsen_cells!
export cell_path, CellPath, find_at_path
export ROOT_PARENT  # sentinel value
export RefinementEvent, ListenerHandle
export register_refinement_listener!, unregister_refinement_listener!

# SimplicialMesh exports (added below)
export SimplicialMesh
export n_vertices, spatial_dimension
export vertex_position, reference_position, set_vertex_position!, set_reference_to_current!
export simplex_vertex_indices, simplex_vertex_positions, simplex_reference_positions
export simplex_neighbor, is_boundary_face
export simplex_volume, simplex_reference_volume
export deformation_gradient, volume_jacobian, distortion_metric
export has_inverted_simplex, max_distortion, enumerate_edges
# Note: n_simplices is exported via the type (don't shadow Mesh's n_cells)
export n_simplices

# CompositeMesh / PairedMesh exports (defined later)
export CompositeMesh, PairedMesh
export ensure_overlap!, invalidate_overlap!, overlap_cache
export set_overlap_compute_function!
export update_lagrangian_positions!

# Generic refinement
export refine_by_indicator!

# ============================================================================
# Flag constants
# ============================================================================

const FLAG_LEAF     = UInt8(0x01)
const FLAG_BOUNDARY = UInt8(0x02)
const FLAG_DIRTY    = UInt8(0x04)
# Bits 3-7 reserved for future use (active, ghost, owned, etc.)

# Sentinel for "no parent" (used for the canonical reference cell)
const ROOT_PARENT = typemax(UInt32)

# ============================================================================
# CellMeta — the canonical per-cell data
# ============================================================================

"""
    CellMeta{D, M}

Canonical per-cell metadata for a D-dimensional mesh. Holds only the
irreducible structural information:

- `sibling_index::M`: position among siblings (0..2^popcount(split_mask)-1)
- `split_mask::M`: which axes were split when this cell's siblings were created
- `flags::UInt8`: leaf/boundary/dirty bits

`M` is selected by `BitPrimitives.sibling_index_type(Val(D))`.

Per-cell size: 3 bytes for D ≤ 7. Compare to traditional AMR cell metadata
that often runs into hundreds of bytes.
"""
struct CellMeta{D, M<:Unsigned}
    sibling_index::M
    split_mask::M
    flags::UInt8
end

"""
    CellMeta{D}(sibling_index, split_mask, flags) where D

Convenience constructor that infers the integer type M from D.
"""
function CellMeta{D}(sibling_index::Integer, split_mask::Integer, flags::Integer=UInt8(0)) where D
    M = sibling_index_type(Val(D))
    return CellMeta{D, M}(M(sibling_index), M(split_mask), UInt8(flags))
end

# Accessors (mostly for clarity; users can also use `.sibling_index` etc.)
@inline sibling_index(c::CellMeta) = c.sibling_index
@inline split_mask(c::CellMeta) = c.split_mask
@inline flags(c::CellMeta) = c.flags

# Flag queries
@inline is_leaf(c::CellMeta) = (c.flags & FLAG_LEAF) != 0
@inline is_boundary(c::CellMeta) = (c.flags & FLAG_BOUNDARY) != 0
@inline is_dirty(c::CellMeta) = (c.flags & FLAG_DIRTY) != 0

# Flag setters return new CellMeta (immutable)
@inline function set_flag(c::CellMeta{D, M}, flag::UInt8) where {D, M}
    return CellMeta{D, M}(c.sibling_index, c.split_mask, c.flags | flag)
end

@inline function clear_flag(c::CellMeta{D, M}, flag::UInt8) where {D, M}
    return CellMeta{D, M}(c.sibling_index, c.split_mask, c.flags & ~flag)
end

# ============================================================================
# Mask helpers
# ============================================================================

"""
    FULLY_ISOTROPIC_MASK(::Val{D}) where D

The split_mask value representing a fully isotropic refinement (all D axes
split). For D=3, returns 0b111 = 7.
"""
@inline function FULLY_ISOTROPIC_MASK(::Val{D}) where D
    M = split_mask_type(Val(D))
    return (M(1) << D) - M(1)
end

"""
    isotropic_mask(D::Integer)

Runtime version of FULLY_ISOTROPIC_MASK.
"""
@inline isotropic_mask(D::Integer) = (UInt32(1) << D) - UInt32(1)

"""
    children_count(c::CellMeta)

Number of children this cell would have if refined according to its split_mask.
For a leaf cell, returns the count from its current split_mask (the count
that *would* result if it were refined further with the same pattern).
"""
@inline children_count(c::CellMeta) = 1 << count_ones(c.split_mask)

@inline children_count(mask::Unsigned) = 1 << count_ones(mask)

# ============================================================================
# HierarchicalMesh — the canonical mesh type
# ============================================================================

"""
    HierarchicalMesh{D, M}

A D-dimensional hierarchical adaptive mesh. Stores only structural data;
fields and physics are separate (associated by parallel-array indexing).

The cells are stored in DFS order, which means subtrees are contiguous —
critical for cache-friendly traversal and for future MPI domain decomposition.

# Fields

- `cells::Vector{CellMeta{D,M}}`: canonical per-cell data, in DFS order.
- `canonical_reference_level::Int16`: which logical level is "level 0"
  (the canonical reference). Cells coarser than this have negative levels;
  finer cells have positive levels.

# Caches (lazy)

- `_levels::Vector{Int16}`: per-cell scalar level (negative = coarser than
  canonical, positive = finer).
- `_parents::Vector{UInt32}`: index in `cells` of each cell's parent
  (ROOT_PARENT for the root).
- `_subtree_sizes::Vector{UInt32}`: number of cells in each subtree
  (including the cell itself).
- `_cache_valid::Bool`: whether caches reflect the current cells array.

Caches are invalidated by any structural change and rebuilt on first access
of a cached function.

# Construction

```julia
# A single-cell mesh (the root)
mesh = HierarchicalMesh{3}()

# Refine the root into 8 children (isotropic)
refine_cells!(mesh, [1])  # refine cell 1
```
"""
mutable struct HierarchicalMesh{D, M<:Unsigned}
    cells::Vector{CellMeta{D, M}}
    canonical_reference_level::Int16

    # Lazy caches
    _levels::Vector{Int16}
    _parents::Vector{UInt32}
    _subtree_sizes::Vector{UInt32}
    _cache_valid::Bool

    # Refinement-event observers. Each entry is a (handle, callback) tuple.
    # `Any` is used so that closures (do…end blocks) can be stored without
    # forcing all callers to share a callable type.
    _listeners::Vector{Any}
    _next_listener_handle::UInt64
end

"""
    HierarchicalMesh{D}() where D

Construct a single-cell mesh containing just the root cell. The root is a
leaf with no siblings (split_mask = 0, sibling_index = 0).
"""
function HierarchicalMesh{D}() where D
    M = sibling_index_type(Val(D))
    cells = [CellMeta{D, M}(M(0), M(0), FLAG_LEAF)]
    return HierarchicalMesh{D, M}(
        cells,
        Int16(0),                    # root is at canonical reference
        Int16[],                     # empty caches
        UInt32[],
        UInt32[],
        false,                       # caches invalid
        Any[],                       # no listeners yet
        UInt64(1)                    # next handle starts at 1
    )
end

"""
    HierarchicalMesh{D}(initial_capacity::Integer) where D

Construct a single-cell mesh with reserved capacity for `initial_capacity`
cells, useful for avoiding reallocation as the mesh grows during refinement.
"""
function HierarchicalMesh{D}(initial_capacity::Integer) where D
    mesh = HierarchicalMesh{D}()
    sizehint!(mesh.cells, initial_capacity)
    return mesh
end

# Basic queries

"""
    n_cells(mesh::HierarchicalMesh)

Total number of cells in the mesh, including non-leaves.
"""
@inline n_cells(mesh::HierarchicalMesh) = length(mesh.cells)

"""
    root_cell_index(mesh::HierarchicalMesh)

Index of the root cell in the cells array. Always 1 for our DFS layout.
"""
@inline root_cell_index(mesh::HierarchicalMesh) = UInt32(1)

# Indexing
@inline Base.getindex(mesh::HierarchicalMesh, i::Integer) = mesh.cells[i]
@inline Base.length(mesh::HierarchicalMesh) = length(mesh.cells)
@inline Base.eachindex(mesh::HierarchicalMesh) = eachindex(mesh.cells)

# ============================================================================
# Cache management
# ============================================================================

"""
    caches_valid(mesh::HierarchicalMesh)

Whether the lazy caches (levels, parents, subtree_sizes) are valid for the
current cells array.
"""
@inline caches_valid(mesh::HierarchicalMesh) = mesh._cache_valid

"""
    invalidate_caches!(mesh::HierarchicalMesh)

Mark all caches as invalid. Called after structural changes (refine, coarsen).
The next call to a cache-dependent function will trigger rebuild.
"""
function invalidate_caches!(mesh::HierarchicalMesh)
    mesh._cache_valid = false
    return mesh
end

"""
    rebuild_caches!(mesh::HierarchicalMesh)

Rebuild all caches from the canonical cells array. Single-pass O(N) traversal
of the cells in DFS order.

Called automatically by accessor functions when caches are invalid; can be
called explicitly to amortize rebuild cost outside hot paths.
"""
function rebuild_caches!(mesh::HierarchicalMesh{D, M}) where {D, M}
    n = length(mesh.cells)
    resize!(mesh._levels, n)
    resize!(mesh._parents, n)
    resize!(mesh._subtree_sizes, n)

    if n == 0
        mesh._cache_valid = true
        return mesh
    end

    # First pass: compute levels and parents using a stack
    # Cell 1 is root; parents come before children in DFS order
    # We track: for each level of nesting, the index of the open parent
    # and the count of children seen so far for it.

    mesh._parents[1] = ROOT_PARENT
    mesh._levels[1] = mesh.canonical_reference_level

    # Stack: (parent_index, children_remaining_to_assign)
    # When a non-leaf cell is encountered, push it with its children_count.
    # Each subsequent cell consumes one slot; when slots reach 0, pop.
    stack = Tuple{UInt32, Int}[]

    if !is_leaf(mesh.cells[1])
        push!(stack, (UInt32(1), children_count(mesh.cells[1])))
    end

    for i in 2:n
        @assert !isempty(stack) "DFS order violated: cell $i has no parent"

        parent_idx, remaining = stack[end]
        mesh._parents[i] = parent_idx
        # Level: parent's level + 1 (scalar level; per-axis would be more nuanced)
        mesh._levels[i] = mesh._levels[parent_idx] + Int16(1)

        # Decrement remaining children for the current parent
        stack[end] = (parent_idx, remaining - 1)
        if stack[end][2] == 0
            pop!(stack)
        end

        # If this cell is a non-leaf, push it as a new parent
        if !is_leaf(mesh.cells[i])
            push!(stack, (UInt32(i), children_count(mesh.cells[i])))
        end
    end

    @assert isempty(stack) "Tree structure inconsistent: $(length(stack)) parents still open after traversal"

    # Second pass: compute subtree_sizes (reverse iteration)
    for i in n:-1:1
        if is_leaf(mesh.cells[i])
            mesh._subtree_sizes[i] = UInt32(1)
        else
            # Sum subtree_sizes of all children
            total = UInt32(1)
            child_idx = UInt32(i + 1)
            for _ in 1:children_count(mesh.cells[i])
                total += mesh._subtree_sizes[child_idx]
                child_idx += mesh._subtree_sizes[child_idx]
            end
            mesh._subtree_sizes[i] = total
        end
    end

    mesh._cache_valid = true
    return mesh
end

"""
    ensure_caches!(mesh::HierarchicalMesh)

Internal helper: rebuild caches if they're invalid. Called by accessors.
"""
@inline function ensure_caches!(mesh::HierarchicalMesh)
    if !mesh._cache_valid
        rebuild_caches!(mesh)
    end
end

# ============================================================================
# Tree navigation (using caches)
# ============================================================================

"""
    find_parent(mesh::HierarchicalMesh, i::Integer)

Index of the parent of cell `i`. Returns ROOT_PARENT for the root cell.
Triggers cache rebuild if needed.
"""
function find_parent(mesh::HierarchicalMesh, i::Integer)
    ensure_caches!(mesh)
    return mesh._parents[i]
end

"""
    level_of(mesh::HierarchicalMesh, i::Integer)

Scalar level of cell `i` relative to the canonical reference. Negative for
coarser cells, positive for finer.
"""
function level_of(mesh::HierarchicalMesh, i::Integer)
    ensure_caches!(mesh)
    return mesh._levels[i]
end

"""
    find_children(mesh::HierarchicalMesh, i::Integer)

Indices of the children of cell `i`. Returns an empty range if `i` is a leaf.
For a non-leaf cell, returns the contiguous range of children indices in the
DFS-ordered array.
"""
function find_children(mesh::HierarchicalMesh, i::Integer)
    cell = mesh.cells[i]
    if is_leaf(cell)
        return UInt32(i+1):UInt32(i)  # empty range
    end
    n_children = children_count(cell)
    # Children are at indices i+1, i+1+subtree_size(i+1), i+1+sub+sub, ...
    # For DFS layout, only the first child is at i+1; others are scattered
    # by their subtrees. We need subtree_sizes for this.
    ensure_caches!(mesh)

    # Build a list of child indices
    # (could return an iterator for zero allocation, but list is small)
    child_indices = Vector{UInt32}(undef, n_children)
    child_idx = UInt32(i + 1)
    for c in 1:n_children
        child_indices[c] = child_idx
        child_idx += mesh._subtree_sizes[child_idx]
    end
    return child_indices
end

"""
    find_lca(mesh::HierarchicalMesh, a::Integer, b::Integer)

Lowest common ancestor of cells `a` and `b`. Walks up from the deeper of
the two until levels match, then walks both up together until they meet.
"""
function find_lca(mesh::HierarchicalMesh, a::Integer, b::Integer)
    ensure_caches!(mesh)
    a, b = UInt32(a), UInt32(b)

    # Lift the deeper one to match the shallower
    while mesh._levels[a] > mesh._levels[b]
        a = mesh._parents[a]
    end
    while mesh._levels[b] > mesh._levels[a]
        b = mesh._parents[b]
    end

    # Walk both up together
    while a != b
        if a == ROOT_PARENT || b == ROOT_PARENT
            return ROOT_PARENT  # no common ancestor (shouldn't happen in well-formed mesh)
        end
        a = mesh._parents[a]
        b = mesh._parents[b]
    end

    return a
end

"""
    position_in_parent(c::CellMeta)

Per-axis position (0 or 1 for each split axis) of this cell within its
parent. Returns a tuple of length D, with non-split axes having position 0
(the cell occupies the full extent of those axes).

For sibling_index = 0b101, split_mask = 0b111 (D=3): returns (1, 0, 1).
"""
@inline function position_in_parent(c::CellMeta{D, M}) where {D, M}
    # Use PDEP to scatter sibling_index bits across the mask positions
    expanded = pdep(UInt32(c.sibling_index), UInt32(c.split_mask))
    return ntuple(i -> Int((expanded >> (i-1)) & UInt32(1)), Val(D))
end

# ============================================================================
# Cell paths — global identity for cells (MPI-friendly)
# ============================================================================

"""
    CellPath{D}

A cell's path from the root: a sequence of (sibling_index, split_mask) pairs
describing how to walk from the root to this cell. Used as a process-
independent cell identity (useful for MPI distribution and for I/O).

Stored as variable-length sequences. For typical depths (<10), small.
"""
struct CellPath{D, M<:Unsigned}
    # Each entry: which child of the previous cell we are.
    # Stored as parallel vectors of (sibling_index, split_mask) for clarity.
    sibling_indices::Vector{M}
    split_masks::Vector{M}
end

"""
    cell_path(mesh::HierarchicalMesh, i::Integer)

Compute the path from root to cell `i`. Walks the parent chain and
collects sibling_index/split_mask at each step.
"""
function cell_path(mesh::HierarchicalMesh{D, M}, i::Integer) where {D, M}
    ensure_caches!(mesh)
    sibs = M[]
    masks = M[]

    current = UInt32(i)
    while current != ROOT_PARENT && mesh._parents[current] != ROOT_PARENT
        c = mesh.cells[current]
        push!(sibs, c.sibling_index)
        push!(masks, c.split_mask)
        current = mesh._parents[current]
    end

    # We've collected from leaf to root; reverse for root-to-leaf order
    reverse!(sibs)
    reverse!(masks)
    return CellPath{D, M}(sibs, masks)
end

"""
    find_at_path(mesh::HierarchicalMesh, path::CellPath)

Find the cell index in the mesh that corresponds to the given path.
Returns 0 if no such cell exists (path goes beyond what's been refined).
"""
function find_at_path(mesh::HierarchicalMesh{D, M}, path::CellPath{D, M}) where {D, M}
    current = root_cell_index(mesh)

    for k in eachindex(path.sibling_indices)
        target_sib = path.sibling_indices[k]
        target_mask = path.split_masks[k]

        cell = mesh.cells[current]
        if is_leaf(cell)
            return UInt32(0)  # path requires further refinement that doesn't exist
        end
        if cell.split_mask != target_mask
            return UInt32(0)  # split pattern mismatch
        end

        # Find the child with the target sibling_index
        children = find_children(mesh, current)
        found = false
        for child_idx in children
            if mesh.cells[child_idx].sibling_index == target_sib
                current = child_idx
                found = true
                break
            end
        end
        if !found
            return UInt32(0)
        end
    end

    return current
end

# ============================================================================
# Refinement events and listener API
# ============================================================================

"""
    RefinementEvent

Describes a single batched mesh modification, carrying enough information for
downstream per-cell storage to resize and permute its arrays so cell indices
stay aligned with the mesh after the event.

# Fields
- `refined_parents::Vector{UInt32}` — OLD indices of cells that became non-leaf
  parents in this batch. Empty for a pure-coarsen event.
- `new_children::Vector{UnitRange{UInt32}}` — parallel to `refined_parents`;
  `new_children[k]` is the NEW-index range of children produced by refining
  `refined_parents[k]`.
- `coarsened_parents::Vector{UInt32}` — NEW indices of cells that became leaf
  again in this batch. Empty for a pure-refine event.
- `removed_old_indices::Vector{UInt32}` — OLD indices of cells that were
  removed by coarsening (children of `coarsened_parents`).
- `index_remap::Vector{UInt32}` — length = old `n_cells`. `index_remap[old_i]
  == 0` means the old cell was removed; otherwise it is the new index.
"""
struct RefinementEvent
    refined_parents::Vector{UInt32}
    new_children::Vector{UnitRange{UInt32}}
    coarsened_parents::Vector{UInt32}
    removed_old_indices::Vector{UInt32}
    index_remap::Vector{UInt32}
end

"""
    ListenerHandle

Opaque handle returned by [`register_refinement_listener!`](@ref) and accepted
by [`unregister_refinement_listener!`](@ref).
"""
const ListenerHandle = UInt64

"""
    register_refinement_listener!(mesh::HierarchicalMesh, callback) -> ListenerHandle

Register `callback(event::RefinementEvent)` to be invoked after every
`refine_cells!`/`coarsen_cells!` call on `mesh`. Returns an opaque handle
usable with [`unregister_refinement_listener!`](@ref).

Listener exceptions are caught and rethrown AFTER cache invalidation so the
mesh is left in a consistent state. Order of invocation matches registration
order. Listeners registered during a callback are not fired for the current
event (the listener list is snapshotted before iteration).
"""
function register_refinement_listener!(mesh::HierarchicalMesh, callback)::ListenerHandle
    handle = mesh._next_listener_handle
    mesh._next_listener_handle = handle + UInt64(1)
    push!(mesh._listeners, (handle, callback))
    return handle
end

"""
    unregister_refinement_listener!(mesh::HierarchicalMesh, handle::ListenerHandle) -> Bool

Remove the listener identified by `handle`. Returns `true` if found and
removed, `false` otherwise.
"""
function unregister_refinement_listener!(mesh::HierarchicalMesh, handle::ListenerHandle)::Bool
    listeners = mesh._listeners
    for k in eachindex(listeners)
        h, _ = listeners[k]::Tuple{UInt64, Any}
        if h == handle
            deleteat!(listeners, k)
            return true
        end
    end
    return false
end

# Internal: dispatch an event to all currently-registered listeners.
# Snapshots the listener list so registrations made by callbacks do not see
# the in-flight event. Catches exceptions, allowing every listener to run,
# and rethrows the first exception only after all listeners have been called.
function _fire_refinement_event!(mesh::HierarchicalMesh, event::RefinementEvent)
    # Snapshot so listeners that re-register / unregister don't disturb iteration.
    snapshot = copy(mesh._listeners)
    first_err = nothing
    first_bt = nothing
    for entry in snapshot
        _, cb = entry::Tuple{UInt64, Any}
        try
            cb(event)
        catch err
            if first_err === nothing
                first_err = err
                first_bt = catch_backtrace()
            end
        end
    end
    if first_err !== nothing
        # Rethrow with the original backtrace so the caller sees the original site.
        Base.rethrow(first_err)
    end
    return mesh
end

# ============================================================================
# Refinement and coarsening
# ============================================================================

"""
    refine_cells!(mesh::HierarchicalMesh, cell_indices, split_masks=nothing)

Refine the specified cells. Each cell in `cell_indices` (which must currently
be a leaf) is replaced by 2^popcount(mask) children, where mask is taken from
`split_masks` (one entry per cell) or defaults to fully-isotropic if not
specified.

This is a batch operation: all refinements are applied together with a
single rebuild of the cells array. The mesh's caches are invalidated.

# Arguments
- `mesh`: the mesh to modify
- `cell_indices`: indices of cells to refine (must be currently leaves)
- `split_masks`: optional per-cell split masks; defaults to fully isotropic

# Implementation note
The cells array is rebuilt to maintain DFS order. Cell indices change after
refinement. If you need stable identities across refinement, use cell paths
(see `cell_path` / `find_at_path`).
"""
function refine_cells!(mesh::HierarchicalMesh{D, M}, cell_indices,
                      split_masks=nothing) where {D, M}
    if isempty(cell_indices)
        return mesh
    end

    iso_mask = FULLY_ISOTROPIC_MASK(Val(D))

    # Validate inputs
    for (k, ci) in enumerate(cell_indices)
        is_leaf(mesh.cells[ci]) ||
            throw(ArgumentError("refine_cells!: cell $ci is not a leaf; cannot refine"))
        if split_masks !== nothing
            mask = M(split_masks[k])
            count_ones(mask) > 0 ||
                throw(ArgumentError("refine_cells!: split mask for cell $ci must have at least one bit set"))
        end
    end

    # Sort cell indices in increasing order so we can rebuild in one pass
    sorted_order = sortperm(collect(cell_indices))
    sorted_indices = collect(cell_indices)[sorted_order]
    sorted_masks = if split_masks === nothing
        fill(iso_mask, length(sorted_indices))
    else
        [M(split_masks[i]) for i in sorted_order]
    end

    # Build the new cells array
    n_old = length(mesh.cells)
    n_new_cells = sum(children_count(m) for m in sorted_masks)
    new_n = n_old + n_new_cells
    new_cells = Vector{CellMeta{D, M}}(undef, new_n)

    # Walk through old cells, inserting children after refined cells
    refine_set = Set(sorted_indices)
    refine_mask_dict = Dict(sorted_indices[i] => sorted_masks[i] for i in eachindex(sorted_indices))

    # Track event data only if there are listeners (event construction
    # allocates; the no-listener fast path keeps refine_cells! at its
    # original allocation cost).
    fire_event = !isempty(mesh._listeners)
    index_remap = fire_event ? Vector{UInt32}(undef, n_old) : UInt32[]
    new_children_ranges = fire_event ?
        Vector{UnitRange{UInt32}}(undef, length(sorted_indices)) :
        UnitRange{UInt32}[]
    # Map old refined-cell index -> position in sorted_indices, so we know
    # which slot of `new_children_ranges` to fill. Built only when needed.
    refined_pos = if fire_event
        Dict{UInt32, Int}(UInt32(sorted_indices[i]) => i
                          for i in eachindex(sorted_indices))
    else
        Dict{UInt32, Int}()
    end

    write_idx = 1
    for read_idx in 1:n_old
        old_cell = mesh.cells[read_idx]

        if read_idx in refine_set
            # Mark this cell as non-leaf, with the chosen split_mask
            mask = refine_mask_dict[read_idx]
            parent_new_idx = UInt32(write_idx)
            new_cells[write_idx] = CellMeta{D, M}(
                old_cell.sibling_index,
                mask,                       # update split_mask to indicate refinement pattern
                old_cell.flags & ~FLAG_LEAF  # clear leaf flag
            )
            if fire_event
                @inbounds index_remap[read_idx] = parent_new_idx
            end
            write_idx += 1

            # Add children
            n_children = children_count(mask)
            child_start = UInt32(write_idx)
            for child_sib in 0:(n_children - 1)
                new_cells[write_idx] = CellMeta{D, M}(
                    M(child_sib),
                    mask,                   # children inherit the split pattern
                    FLAG_LEAF
                )
                write_idx += 1
            end
            if fire_event
                child_end = UInt32(write_idx - 1)
                @inbounds new_children_ranges[refined_pos[UInt32(read_idx)]] = child_start:child_end
            end
        else
            # Copy unchanged
            new_cells[write_idx] = old_cell
            if fire_event
                @inbounds index_remap[read_idx] = UInt32(write_idx)
            end
            write_idx += 1
        end
    end

    @assert write_idx - 1 == new_n "Refinement bookkeeping error: wrote $(write_idx-1) of $new_n cells"

    mesh.cells = new_cells
    invalidate_caches!(mesh)

    if fire_event
        event = RefinementEvent(
            UInt32.(sorted_indices),
            new_children_ranges,
            UInt32[],
            UInt32[],
            index_remap,
        )
        _fire_refinement_event!(mesh, event)
    end

    return mesh
end

"""
    coarsen_cells!(mesh::HierarchicalMesh, parent_indices)

Coarsen the specified parent cells: remove all children of each parent in
`parent_indices`, making the parent a leaf again. All children must currently
be leaves (no recursive coarsening in one step).

Like refine, this is a batch operation that rebuilds the cells array.
"""
function coarsen_cells!(mesh::HierarchicalMesh{D, M}, parent_indices) where {D, M}
    if isempty(parent_indices)
        return mesh
    end

    ensure_caches!(mesh)
    sorted = sort(collect(parent_indices))

    # Validate: each parent must be non-leaf, and all its children must be leaves
    for pi in sorted
        is_leaf(mesh.cells[pi]) &&
            throw(ArgumentError("coarsen_cells!: cell $pi is a leaf; cannot coarsen"))
        for ci in find_children(mesh, pi)
            is_leaf(mesh.cells[ci]) ||
                throw(ArgumentError("coarsen_cells!: cell $ci (child of $pi) is not a leaf; " *
                                      "cannot coarsen non-leaf children"))
        end
    end

    # Determine cells to remove: for each parent, all cells in its subtree except itself
    to_remove = Set{UInt32}()
    for pi in sorted
        sub_size = mesh._subtree_sizes[pi]
        for offset in 1:(sub_size - 1)
            push!(to_remove, UInt32(pi) + UInt32(offset))
        end
    end

    n_old = length(mesh.cells)
    new_n = n_old - length(to_remove)
    new_cells = Vector{CellMeta{D, M}}(undef, new_n)

    coarsen_set = Set(sorted)

    # Event bookkeeping only when there's a listener.
    fire_event = !isempty(mesh._listeners)
    index_remap = fire_event ? Vector{UInt32}(undef, n_old) : UInt32[]
    coarsened_new = fire_event ? Vector{UInt32}(undef, length(sorted)) : UInt32[]
    # Build coarsened_new in the order of `sorted`. We fill via a dict for
    # deterministic order matching the sorted parent indices.
    coarsened_pos = if fire_event
        Dict{UInt32, Int}(UInt32(sorted[i]) => i for i in eachindex(sorted))
    else
        Dict{UInt32, Int}()
    end
    removed_old = fire_event ? Vector{UInt32}(undef, length(to_remove)) : UInt32[]
    removed_count = 0

    write_idx = 1
    for read_idx in 1:n_old
        if read_idx in coarsen_set
            # Mark as leaf (keep sibling_index and split_mask)
            old_cell = mesh.cells[read_idx]
            new_cells[write_idx] = CellMeta{D, M}(
                old_cell.sibling_index,
                old_cell.split_mask,
                old_cell.flags | FLAG_LEAF
            )
            if fire_event
                @inbounds index_remap[read_idx] = UInt32(write_idx)
                @inbounds coarsened_new[coarsened_pos[UInt32(read_idx)]] = UInt32(write_idx)
            end
            write_idx += 1
        elseif read_idx in to_remove
            # Skip
            if fire_event
                @inbounds index_remap[read_idx] = UInt32(0)
                removed_count += 1
                @inbounds removed_old[removed_count] = UInt32(read_idx)
            end
            continue
        else
            new_cells[write_idx] = mesh.cells[read_idx]
            if fire_event
                @inbounds index_remap[read_idx] = UInt32(write_idx)
            end
            write_idx += 1
        end
    end

    @assert write_idx - 1 == new_n "Coarsening bookkeeping error"

    mesh.cells = new_cells
    invalidate_caches!(mesh)

    if fire_event
        event = RefinementEvent(
            UInt32[],
            UnitRange{UInt32}[],
            coarsened_new,
            removed_old,
            index_remap,
        )
        _fire_refinement_event!(mesh, event)
    end

    return mesh
end

# ============================================================================
# Simplicial mesh (Lagrangian, moving vertices)
# ============================================================================
include("SimplicialMesh.jl")

# ============================================================================
# Composite and paired meshes
# ============================================================================
include("CompositeMesh.jl")

# ============================================================================
# Generic indicator-driven refinement
# ============================================================================
include("RefineByIndicator.jl")

end # module Mesh
