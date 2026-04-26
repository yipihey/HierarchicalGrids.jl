"""
    compute_overlap(lag::SimplicialMesh{D, T}, frame::EulerianFrame{D, T};
                     moment_order::Integer = 3,
                     edge_kind::Symbol = :linear,
                     leaf_size::Integer = 8,
                     parallel::Bool = false,
                     scheduler::Symbol = :dynamic) -> GeometricOverlap{D, T}

Compute the full geometric overlap between a Lagrangian simplicial mesh
and an Eulerian hierarchical mesh (wrapped in an `EulerianFrame` for
physical coordinates).

# Algorithm

1. Build a BVH over the Lagrangian simplices for broad-phase pruning.
2. Enumerate Eulerian leaves; for each leaf, compute its physical AABB
   and query the BVH for candidate Lagrangian simplices.
3. For each `(simplex, leaf)` candidate, call `overlap_simplex_box!` to
   compute the exact overlap polygon, its volume, centroid, and full
   moment vector up to `moment_order`.
4. Collect all nonzero entries and finalize into a `GeometricOverlap`.

# Arguments

- `lag` — Lagrangian simplicial mesh with current vertex positions.
- `frame` — Eulerian frame providing physical bounds for the tree mesh.
- `moment_order` — order of polynomial moments to integrate per overlap.
  Default `3` matches dfmm's cubic-reconstruction Bayesian remap.
- `edge_kind` — `:linear` (the default) treats Lagrangian simplex edges
  as straight; `:cubic` (NOT YET SUPPORTED) would use the dimension-
  lifting trick to handle cubic edges exactly.
- `leaf_size` — BVH leaf granularity; smaller is more pruning, more nodes.
- `parallel` — when `true`, dispatch the per-Eulerian-leaf clipping
  across threads via OhMyThreads. Auto-falls-through to sequential for
  small problems (`n_simplices × n_eul_leaves < 10_000`) where the
  task overhead would exceed the gain. Default `false` so single-thread
  callers see no behavior change.
- `scheduler` — OhMyThreads scheduler symbol (`:dynamic`, `:static`,
  `:greedy`, `:serial`). Only consulted when `parallel = true`. Default
  `:dynamic` for load balancing and composability with nested parallel
  calls.

# Returns

A `GeometricOverlap{D, T}` with all nonzero pairs and CSR-style indexing
both ways. Iterate with `entries_for_lag` or `entries_for_eul`.

# Performance

The BVH build itself is sequential and dominates wall time for large
Lagrangian meshes (60–100% of total at `n_simplices ≥ 10⁴`). When
`parallel = true`, only the per-leaf clipping loop is threaded; the BVH
build remains a serial cost. For smaller problems, where the leaf loop
is the bulk of work, parallel `compute_overlap` gives close to linear
speedup over the leaf loop on the available threads.

# Determinism

The parallel and sequential paths produce identical `GeometricOverlap`
objects: same number of entries, same per-entry `(lag_idx, eul_idx,
volume, centroid, moments)`, in the same order. Entries are sorted by
`(lag_idx, eul_idx)` after collection, so task interleaving doesn't
affect the result.
"""
function compute_overlap(lag::SimplicialMesh{D, T},
                          frame::EulerianFrame{D, T};
                          moment_order::Integer = 3,
                          edge_kind::Symbol = :linear,
                          leaf_size::Integer = 8,
                          parallel::Bool = false,
                          scheduler::Symbol = :dynamic,
                          frame_bcs::Union{Nothing, FrameBoundaries{D}} = nothing
                          ) where {D, T}
    edge_kind === :linear ||
        throw(ArgumentError("edge_kind=$edge_kind not yet supported (:linear only). " *
                             "Cubic-edge dimension lifting volume-only path is unblocked, " *
                             "but full polynomial remap requires r3djl higher-order moments " *
                             "(P ≥ 1) at D ≥ 4 (still pending). " *
                             "See src/Overlap/lifting.jl for status."))

    # PR-D: periodic ghost-overlap entries (Lagrangian simplices wrapped
    # to the opposite side of a periodic axis) are deferred to a follow-up
    # PR. Non-periodic BC kinds (INFLOW / OUTFLOW / REFLECTING / DIRICHLET)
    # are advisory at this layer — they're consumed by PDE-level code
    # (face fluxes, boundary integrals, Lagrangian motion clamping) rather
    # than by the geometric overlap computation. When `frame_bcs` is
    # `nothing` (the default), behavior is identical to pre-PR-D code.
    # When non-`nothing`, this implementation accepts the argument for
    # forward-compatibility but does not yet emit wrap-around ghost
    # entries; callers that need periodic halos should consult
    # `face_neighbors_with_bcs` for neighbor-graph wiring.
    frame_bcs  # silence the unused-binding hint; the validation happened
              # at FrameBoundaries construction time.

    # Build the BVH over Lagrangian simplices (sequential — small).
    tree = build_simplex_aabb_tree(lag; leaf_size = Int(leaf_size))
    eul = frame.mesh

    if !parallel || Threads.nthreads() == 1
        return _compute_overlap_sequential(lag, frame, tree, Int(moment_order))
    end

    # Heuristic threshold: for small problems, parallel overhead exceeds
    # the gain. Count cells × simplices crudely; below ~10k pair-candidates
    # we stay sequential. This matches the rule used by the framework's
    # other parallel primitives.
    n_eul_leaves = 0
    @inbounds for ci in 1:n_cells(frame.mesh)
        is_leaf(frame.mesh.cells[ci]) && (n_eul_leaves += 1)
    end
    if n_simplices(lag) * n_eul_leaves < 10_000
        return _compute_overlap_sequential(lag, frame, tree, Int(moment_order))
    end

    return _compute_overlap_parallel(lag, frame, tree, Int(moment_order); scheduler)
end

# Sequential implementation — used directly when parallel=false or
# nthreads()==1, and as the per-task body in the parallel implementation.
function _compute_overlap_sequential(lag::SimplicialMesh{D, T},
                                       frame::EulerianFrame{D, T},
                                       tree::SimplicialAABBTree{D, T},
                                       moment_order::Int) where {D, T}
    builder = OverlapBuilder{D, T}(moment_order)
    scratch = PairScratch(Val(D), T)
    moments_buf = Vector{T}(undef, moments_length(D, moment_order))
    candidates = Int32[]; sizehint!(candidates, 64)

    eul = frame.mesh
    @inbounds for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        leaf_lo, leaf_hi = cell_physical_box(frame, ci)
        empty!(candidates)
        query_aabb!(candidates, tree, leaf_lo, leaf_hi)
        for s in candidates
            verts = simplex_vertex_positions(lag, Int(s))
            vol, centroid, _ = overlap_simplex_box!(moments_buf, scratch,
                                                      verts, leaf_lo, leaf_hi,
                                                      moment_order)
            vol > zero(T) || continue
            push_overlap!(builder, s, ci, vol, centroid, moments_buf)
        end
    end

    return finalize_overlap(builder, n_simplices(lag), n_cells(eul))
end

# Parallel implementation — chunks the Eulerian leaves across tasks, each
# with its own per-task `OverlapBuilder` + `PairScratch` + buffers, then
# merges the per-chunk builders before finalizing.
function _compute_overlap_parallel(lag::SimplicialMesh{D, T},
                                     frame::EulerianFrame{D, T},
                                     tree::SimplicialAABBTree{D, T},
                                     moment_order::Int;
                                     scheduler::Symbol = :dynamic) where {D, T}
    eul = frame.mesh

    # IMPORTANT: force the lazy parent/level caches to build NOW, on the
    # main thread, before the parallel section. `cell_physical_box` →
    # `cell_unit_box` → `Mesh.ensure_caches!` lazily resizes
    # `mesh._parents` on first call; concurrent task entry races on the
    # `resize!` and raises `ConcurrencyViolationError("Vector can not be
    # resized concurrently")`. A pre-build pass here is cheap (the cache
    # is keyed by mesh topology, not by leaf, so once built it's stable
    # for the duration of the call) and turns every per-leaf access into
    # a pure read.
    Mesh.ensure_caches!(eul)

    # Pre-collect leaf indices: every iteration is real work, no
    # is_leaf-skip noise inside the parallel section.
    leaf_indices = Int[]
    sizehint!(leaf_indices, n_cells(eul))
    @inbounds for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) && push!(leaf_indices, ci)
    end
    isempty(leaf_indices) &&
        return finalize_overlap(OverlapBuilder{D, T}(moment_order),
                                  n_simplices(lag), n_cells(eul))

    n_phys = moments_length(D, moment_order)

    # Chunk the leaves manually. Each chunk → one task → one per-chunk
    # builder. We use ChunkSplitters via OhMyThreads (re-exported) for
    # the chunk math.
    n_chunks = min(length(leaf_indices), max(1, Threads.nthreads()))
    chunk_ranges = OhMyThreads.index_chunks(leaf_indices; n = n_chunks)

    # `tmapreduce` over the chunk ranges: each task processes one chunk,
    # produces one builder, then `merge_builder!` reduces across them.
    #
    # IMPORTANT: we do NOT pass `init = OverlapBuilder(...)` here even
    # though it would seem natural. OhMyThreads passes the SAME `init`
    # object to all tasks (verified empirically); since `merge_builder!`
    # mutates its first argument in place, multiple tasks would all be
    # pushing/appending into the same shared builder concurrently —
    # which raises `ConcurrencyViolationError("Vector can not be resized
    # concurrently")` under `:static` scheduling and is racy under
    # `:dynamic`. Without `init`, OhMyThreads seeds the reduction with
    # one of the mapped values (a per-task fresh builder), which is
    # exactly what we want.
    #
    # Per-task scratch (builder, scratch, moments_buf, candidates) is
    # allocated once at the start of each task body — one allocation
    # per task, not per leaf.
    final_builder = OhMyThreads.tmapreduce(
        merge_builder!,
        chunk_ranges;
        scheduler = scheduler,
    ) do chunk_range
        builder = OverlapBuilder{D, T}(moment_order)
        scratch = PairScratch(Val(D), T)
        moments_buf = Vector{T}(undef, n_phys)
        candidates = Int32[]; sizehint!(candidates, 64)

        @inbounds for k in chunk_range
            ci = leaf_indices[k]
            leaf_lo, leaf_hi = cell_physical_box(frame, ci)
            empty!(candidates)
            query_aabb!(candidates, tree, leaf_lo, leaf_hi)
            for s in candidates
                verts = simplex_vertex_positions(lag, Int(s))
                vol, centroid, _ = overlap_simplex_box!(moments_buf, scratch,
                                                          verts, leaf_lo, leaf_hi,
                                                          moment_order)
                vol > zero(T) || continue
                push_overlap!(builder, s, ci, vol, centroid, moments_buf)
            end
        end
        builder
    end

    return finalize_overlap(final_builder, n_simplices(lag), n_cells(eul))
end

"""
    install_r3d_overlap!(paired::PairedMesh, frame::EulerianFrame;
                          moment_order = 3, edge_kind = :linear, leaf_size = 8)

Configure an existing `PairedMesh` so that `ensure_overlap!(paired)`
computes the overlap via this layer. The `frame` provides the physical
coordinates for the Eulerian half; the Lagrangian half is taken from
`paired.lagrangian` (must be a `SimplicialMesh{D, T}` matching `frame`).

After this, `overlap_cache(paired)` returns the cached `GeometricOverlap`.
"""
function install_r3d_overlap!(paired::PairedMesh,
                                frame::EulerianFrame{D, T};
                                moment_order::Integer = 3,
                                edge_kind::Symbol = :linear,
                                leaf_size::Integer = 8) where {D, T}
    paired.eulerian === frame.mesh ||
        throw(ArgumentError("frame.mesh must be the same object as paired.eulerian"))
    set_overlap_compute_function!(paired) do pm
        return compute_overlap(pm.lagrangian, frame;
                                moment_order = moment_order,
                                edge_kind = edge_kind,
                                leaf_size = leaf_size)
    end
    return paired
end
