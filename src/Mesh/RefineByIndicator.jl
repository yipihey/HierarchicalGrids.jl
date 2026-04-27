"""
    RefineByIndicator

Generic indicator-driven adaptive mesh refinement loop. Lives as a
top-level submodule (rather than inside `Mesh`) because it depends on
the `Threading` backend trait + verbs to parallelize the per-cell
indicator evaluation. The mesh-mutation calls (`refine_cells!` /
`coarsen_cells!`) themselves remain sequential — mesh rebuild is
inherently serial.

Public API (re-exported at the top level):

- [`refine_by_indicator!`](@ref) — single-pass refine/coarsen driver.
"""
module RefineByIndicator

using ..Mesh
using ..Mesh: HierarchicalMesh, n_cells, level_of, is_leaf, find_parent,
              find_children, refine_cells!, coarsen_cells!,
              FULLY_ISOTROPIC_MASK, ROOT_PARENT
using ..Threading: AbstractParallelBackend, Sequential, OhMyThreadsBackend,
                   default_backend, parallel_chunked, ThreadChunk,
                   partition_for_threads

export refine_by_indicator!

"""
    refine_by_indicator!(mesh::HierarchicalMesh, indicator;
                          refine_threshold,
                          coarsen_threshold = refine_threshold / 4,
                          max_level = typemax(Int),
                          isotropic = true,
                          backend = default_backend())

Generic indicator-driven adaptive mesh refinement loop. Examines a per-cell
scalar `indicator` and refines cells where the indicator exceeds
`refine_threshold`, coarsens sibling groups where all members fall below
`coarsen_threshold`. The hysteresis between thresholds prevents oscillation
between refining and coarsening the same cells.

# Arguments

- `mesh::HierarchicalMesh{D, M}` — the mesh to modify in place.
- `indicator` — an iterable or function:
  - If iterable: indicator[i] is the scalar value for cell i (must have
    length == n_cells(mesh)).
  - If callable: indicator(i) returns the scalar value for cell i.
- `refine_threshold` — refine any leaf cell with indicator > threshold.
- `coarsen_threshold` — coarsen sibling groups where all members have
  indicator < threshold. Defaults to `refine_threshold / 4` for hysteresis.
- `max_level` — don't refine cells already at this level.
- `isotropic` — if true (default), refine all D axes simultaneously.
- `backend` — parallel backend used for the per-cell indicator-evaluation
  passes. The actual mesh-mutation calls remain sequential. Defaults to
  `default_backend()`.

# Returns

A NamedTuple `(refined=n_refined, coarsened=n_coarsened)` reporting the
counts. The mesh is modified in place.

# Notes

- Coarsening only succeeds if all 2^D children of a parent cell are leaves
  AND all their indicator values are below the coarsen threshold.
- This is a "one pass" refinement; for multi-level adaptation, call in a
  loop until counts stabilize.
- The mesh's caches are invalidated by `refine_cells!` / `coarsen_cells!`,
  so any cached level information is rebuilt on next access.
- The candidate-list ordering is identical to the sequential reference;
  per-task partial lists are concatenated in chunk order and then sorted,
  so refinement / coarsening targets are deterministic across backends.
"""
function refine_by_indicator!(mesh::HierarchicalMesh{D, M}, indicator;
                               refine_threshold::Real,
                               coarsen_threshold::Real = refine_threshold / 4,
                               max_level::Integer = typemax(Int),
                               isotropic::Bool = true,
                               backend::AbstractParallelBackend = default_backend()) where {D, M}
    n = n_cells(mesh)
    # Materialize indicator as a function for uniform access
    ind_fn = _normalize_indicator(indicator, n)

    # Pre-build caches before fanning out — `level_of` reads cached
    # per-axis levels, and the lazy-cache build is not thread-safe under
    # concurrent first access.
    Mesh.ensure_caches!(mesh)

    # Identify cells to refine (leaves above threshold and below max level).
    to_refine = _gather_refine_candidates(mesh, ind_fn, refine_threshold,
                                          max_level, backend)

    n_refined = length(to_refine)
    if !isempty(to_refine)
        if isotropic
            iso = FULLY_ISOTROPIC_MASK(Val(D))
            refine_cells!(mesh, to_refine, fill(iso, n_refined))
        else
            refine_cells!(mesh, to_refine)
        end
    end

    # After refinement, cell indices have changed. We need a fresh indicator
    # for coarsening — but we don't have one for the new cells. So coarsening
    # is based on the pre-refinement state: only consider parents whose
    # children all existed and were below the coarsen threshold pre-refinement.
    # The simplest contract is: refine_by_indicator! does NOT coarsen in the
    # same pass. The user can call it again with the new indicator.
    #
    # However, often the user wants both in one pass with the SAME indicator
    # (computed before any refinement). Support that by computing coarsening
    # candidates BEFORE refinement.

    # If we already refined, the indicator (which was over the OLD indices)
    # is no longer aligned. Skip coarsening in that case unless explicitly
    # requested via a separate path.
    n_coarsened = 0
    if isempty(to_refine)
        # No refinement happened, indices are stable, look for coarsening
        candidates = _find_coarsen_candidates(mesh, ind_fn, coarsen_threshold,
                                              backend)
        if !isempty(candidates)
            coarsen_cells!(mesh, candidates)
            n_coarsened = length(candidates)
        end
    end

    return (refined=n_refined, coarsened=n_coarsened)
end

# ============================================================================
# Internal: parallel candidate gathering
# ----------------------------------------------------------------------------
# Each chunk produces a `Vector{Int}` of candidates (cell indices that pass
# the per-cell predicate). The partial results are concatenated in chunk
# order and then sorted to recover the same ordering as the sequential
# reference implementation. Sorting after the merge is what makes the
# pipeline deterministic across schedulers — partial lists may arrive out
# of chunk order under :dynamic / :greedy schedulers, but a sort closes
# the gap.
# ============================================================================

# Gather refine candidates: leaves with `level_of(...) < max_level` whose
# indicator exceeds the refine threshold.
function _gather_refine_candidates(mesh::HierarchicalMesh{D, M}, ind_fn,
                                    refine_threshold::Real,
                                    max_level::Integer,
                                    backend::AbstractParallelBackend) where {D, M}
    n = n_cells(mesh)
    return _gather_candidates(mesh, backend, n) do partial, ci
        @inbounds if is_leaf(mesh.cells[ci]) && level_of(mesh, ci) < max_level
            if ind_fn(ci) > refine_threshold
                push!(partial, ci)
            end
        end
        return nothing
    end
end

# Sequential fast-path: fall back to a single in-order loop.
@inline function _gather_candidates(per_cell, mesh::HierarchicalMesh,
                                     ::Sequential, n::Integer)
    out = Int[]
    @inbounds for ci in 1:Int(n)
        per_cell(out, ci)
    end
    return out
end

# Parallel path: per-chunk scratch vectors, populated in parallel via
# `parallel_chunked`, then concatenated and sorted to preserve the
# sequential ordering.
function _gather_candidates(per_cell, mesh::HierarchicalMesh,
                             backend::AbstractParallelBackend, n::Integer)
    n = Int(n)
    # No work — short-circuit (also keeps `partition_for_threads` happy
    # for an empty mesh, though that path shouldn't normally arise).
    n == 0 && return Int[]
    n_chunks = max(1, min(Int(Threads.nthreads()), n))
    partials = Vector{Vector{Int}}(undef, n_chunks)
    for k in 1:n_chunks
        partials[k] = Int[]
    end
    parallel_chunked(backend, function (_mesh, chunk::ThreadChunk)
        local_buf = partials[Int(chunk.chunk_id)]
        @inbounds for ci in Int(first(chunk.cell_range)):Int(last(chunk.cell_range))
            per_cell(local_buf, ci)
        end
        return nothing
    end, mesh, n_chunks)
    # Concatenate in chunk order, then sort to match the sequential reference
    # (cell indices appear in increasing order because the sequential loop
    # walks `1:n`). Partition is contiguous and chunk-id ordered, so the
    # merged list is already monotone — we sort defensively to be robust to
    # any future change in partitioning strategy.
    total = 0
    for v in partials
        total += length(v)
    end
    out = Vector{Int}(undef, total)
    pos = 1
    for v in partials
        copyto!(out, pos, v, 1, length(v))
        pos += length(v)
    end
    sort!(out)
    return out
end

# Find all parent cells whose direct children are all leaves AND all have
# indicator below the coarsen threshold.
#
# The parallel pass enumerates leaves and emits each leaf's parent index
# (after de-duplication within the task). The merge step removes
# cross-task duplicates, then verifies the all-leaves-and-below-threshold
# predicate sequentially. The final candidate list is sorted to match the
# sequential reference ordering.
function _find_coarsen_candidates(mesh::HierarchicalMesh{D, M}, ind_fn,
                                   coarsen_threshold::Real,
                                   backend::AbstractParallelBackend) where {D, M}
    n = n_cells(mesh)
    n == 0 && return Int[]

    # Step 1: parallel-collect candidate parent indices (one per leaf, with
    # per-task deduplication). Indices may repeat across tasks — that's
    # fixed in the sequential merge below.
    parent_lists = _gather_parent_candidates(mesh, backend, n)

    # Step 2: merge + dedupe (sequential).
    seen = Set{Int}()
    parents = Int[]
    for vlist in parent_lists
        for p in vlist
            if !(p in seen)
                push!(seen, p)
                push!(parents, p)
            end
        end
    end

    # Step 3: verify the all-leaves-and-below-threshold predicate. Cheap
    # per parent, no allocation; left sequential for clarity.
    candidates = Int[]
    @inbounds for parent in parents
        children = find_children(mesh, parent)
        isempty(children) && continue
        all_leaf_below = true
        for c in children
            if !is_leaf(mesh.cells[c])
                all_leaf_below = false; break
            end
            if ind_fn(c) >= coarsen_threshold
                all_leaf_below = false; break
            end
        end
        all_leaf_below && push!(candidates, parent)
    end

    # Sort to match the sequential reference ordering (parents are emitted
    # in the order their first leaf-child was encountered in the sequential
    # walk of `1:n`, which under our partition equals the natural
    # parent-index order).
    sort!(candidates)
    return candidates
end

# Sequential fast-path mirror for parent gathering.
@inline function _gather_parent_candidates(mesh::HierarchicalMesh,
                                            ::Sequential, n::Integer)
    n = Int(n)
    out = Int[]
    seen = Set{Int}()
    @inbounds for ci in 1:n
        if !is_leaf(mesh.cells[ci])
            continue
        end
        parent = Int(find_parent(mesh, ci))
        # Skip the root cell (UInt32 ROOT_PARENT) and already-seen parents
        if parent == Int(typemax(UInt32)) || parent in seen
            continue
        end
        push!(seen, parent)
        push!(out, parent)
    end
    return [out]
end

# Parallel: one parent-list per chunk, with per-chunk dedup.
function _gather_parent_candidates(mesh::HierarchicalMesh,
                                    backend::AbstractParallelBackend,
                                    n::Integer)
    n = Int(n)
    n_chunks = max(1, min(Int(Threads.nthreads()), n))
    parent_lists = Vector{Vector{Int}}(undef, n_chunks)
    seen_per_chunk = Vector{Set{Int}}(undef, n_chunks)
    for k in 1:n_chunks
        parent_lists[k] = Int[]
        seen_per_chunk[k] = Set{Int}()
    end
    parallel_chunked(backend, function (_mesh, chunk::ThreadChunk)
        k = Int(chunk.chunk_id)
        local_out = parent_lists[k]
        local_seen = seen_per_chunk[k]
        @inbounds for ci in Int(first(chunk.cell_range)):Int(last(chunk.cell_range))
            if !is_leaf(mesh.cells[ci])
                continue
            end
            parent = Int(find_parent(mesh, ci))
            if parent == Int(typemax(UInt32)) || parent in local_seen
                continue
            end
            push!(local_seen, parent)
            push!(local_out, parent)
        end
        return nothing
    end, mesh, n_chunks)
    return parent_lists
end

# ============================================================================
# Indicator normalization (callable vs iterable)
# ============================================================================

@inline _normalize_indicator(ind::Function, ::Int) = ind
@inline function _normalize_indicator(ind, n::Int)
    length(ind) == n || throw(ArgumentError("indicator length $(length(ind)) doesn't match n_cells $n"))
    return i -> ind[i]
end

end # module RefineByIndicator
