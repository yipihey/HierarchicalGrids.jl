"""
    refine_by_indicator!(mesh::HierarchicalMesh, indicator;
                          refine_threshold,
                          coarsen_threshold = refine_threshold / 4,
                          max_level = typemax(Int),
                          isotropic = true)

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
"""
function refine_by_indicator!(mesh::HierarchicalMesh{D, M}, indicator;
                               refine_threshold::Real,
                               coarsen_threshold::Real = refine_threshold / 4,
                               max_level::Integer = typemax(Int),
                               isotropic::Bool = true) where {D, M}
    n = n_cells(mesh)
    # Materialize indicator as a function for uniform access
    ind_fn = _normalize_indicator(indicator, n)

    # Identify cells to refine (leaves above threshold and below max level)
    to_refine = Int[]
    @inbounds for ci in 1:n
        if is_leaf(mesh.cells[ci]) && level_of(mesh, ci) < max_level
            if ind_fn(ci) > refine_threshold
                push!(to_refine, ci)
            end
        end
    end

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
        candidates = _find_coarsen_candidates(mesh, ind_fn, coarsen_threshold)
        if !isempty(candidates)
            coarsen_cells!(mesh, candidates)
            n_coarsened = length(candidates)
        end
    end

    return (refined=n_refined, coarsened=n_coarsened)
end

# Find all parent cells whose direct children are all leaves AND all have
# indicator below the coarsen threshold.
function _find_coarsen_candidates(mesh::HierarchicalMesh{D, M}, ind_fn,
                                   coarsen_threshold::Real) where {D, M}
    candidates = Int[]
    n = n_cells(mesh)
    # Build a map: parent -> list of children
    # Use the existing find_children helper
    seen_parents = Set{Int}()
    @inbounds for ci in 1:n
        if !is_leaf(mesh.cells[ci])
            continue
        end
        parent = find_parent(mesh, ci)
        # Skip root and already-processed parents
        (parent <= 0 || parent in seen_parents) && continue
        push!(seen_parents, parent)

        # Get all children of this parent
        children = find_children(mesh, parent)
        if isempty(children)
            continue
        end
        # All children must be leaves AND below threshold
        all_leaf_below = true
        for c in children
            if !is_leaf(mesh.cells[c])
                all_leaf_below = false; break
            end
            if ind_fn(c) >= coarsen_threshold
                all_leaf_below = false; break
            end
        end
        if all_leaf_below
            push!(candidates, parent)
        end
    end
    return candidates
end

# Normalize indicator into a callable indicator(i)
@inline _normalize_indicator(ind::Function, ::Int) = ind
@inline function _normalize_indicator(ind, n::Int)
    length(ind) == n || throw(ArgumentError("indicator length $(length(ind)) doesn't match n_cells $n"))
    return i -> ind[i]
end
