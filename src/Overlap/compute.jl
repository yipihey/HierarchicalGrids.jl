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

    # PR-D + follow-up: when `frame_bcs` is `nothing` (the default), the
    # geometric path is identical to pre-PR-D code (single branch, zero
    # overhead). When `frame_bcs` is supplied, periodic axes generate
    # ghost-image overlap entries: a Lagrangian simplex partially or
    # wholly outside the frame on a periodic axis is wrapped to overlap
    # with the frame's interior via translation by ±(hi - lo). Multiple
    # periodic axes produce corner ghosts via every combination of ±period
    # shifts. Non-periodic BC kinds (INFLOW / OUTFLOW / REFLECTING /
    # DIRICHLET) are advisory at this layer — they're consumed by PDE-
    # level code (face fluxes, boundary integrals, Lagrangian motion
    # clamping) rather than by the geometric overlap computation.
    ghost_shifts = _ghost_shift_table(frame, frame_bcs)

    # Build the BVH over Lagrangian simplices (sequential — small).
    tree = build_simplex_aabb_tree(lag; leaf_size = Int(leaf_size))
    eul = frame.mesh

    if !parallel || Threads.nthreads() == 1
        return _compute_overlap_sequential(lag, frame, tree, Int(moment_order),
                                             ghost_shifts)
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
        return _compute_overlap_sequential(lag, frame, tree, Int(moment_order),
                                             ghost_shifts)
    end

    return _compute_overlap_parallel(lag, frame, tree, Int(moment_order),
                                       ghost_shifts; scheduler)
end

# Build the table of periodic ghost shifts. Returns `nothing` when no BCs
# are attached (so the hot path can take a single, predictable branch on
# `ghost_shifts === nothing`). Otherwise returns a Vector of NTuple{D, T}
# shift vectors, EXCLUDING the zero shift (the zero-shift overlap is
# computed by the regular per-leaf path). For k periodic axes the table
# has 3^k - 1 entries, covering every combination of `{-period, 0,
# +period}` along each periodic axis with at least one nonzero component.
@inline function _ghost_shift_table(frame::EulerianFrame{D, T},
                                     ::Nothing) where {D, T}
    return nothing
end

function _ghost_shift_table(frame::EulerianFrame{D, T},
                              fb::FrameBoundaries{D}) where {D, T}
    # Determine periodic axes and their periods.
    periodic_axes = Int[]
    @inbounds for d in 1:D
        if is_periodic_axis(fb, d)
            push!(periodic_axes, d)
        end
    end
    isempty(periodic_axes) && return nothing

    periods = ntuple(d -> frame.hi[d] - frame.lo[d], Val(D))

    # Enumerate all 3^k combinations of {-1, 0, +1} on the periodic axes,
    # skipping the all-zero one. Non-periodic axes get a 0 shift.
    k = length(periodic_axes)
    n_combos = 3^k
    shifts = Vector{NTuple{D, T}}()
    sizehint!(shifts, n_combos - 1)
    @inbounds for code in 0:(n_combos - 1)
        # Decode `code` in base 3, mapping digit ∈ {0, 1, 2} → sign ∈ {0, +1, -1}.
        # We store the sign per periodic axis, zero for non-periodic axes.
        c = code
        all_zero = true
        signs = MVector{D, Int}(ntuple(_ -> 0, Val(D)))
        for ax in periodic_axes
            digit = c % 3
            c = c ÷ 3
            s = digit == 0 ? 0 : (digit == 1 ? 1 : -1)
            signs[ax] = s
            if s != 0
                all_zero = false
            end
        end
        all_zero && continue
        push!(shifts, ntuple(d -> T(signs[d]) * periods[d], Val(D)))
    end
    return shifts
end

# Sequential implementation — used directly when parallel=false or
# nthreads()==1, and as the per-task body in the parallel implementation.
function _compute_overlap_sequential(lag::SimplicialMesh{D, T},
                                       frame::EulerianFrame{D, T},
                                       tree::SimplicialAABBTree{D, T},
                                       moment_order::Int,
                                       ghost_shifts::Union{Nothing,
                                                           Vector{NTuple{D, T}}}
                                       ) where {D, T}
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

        # Periodic ghost path: for each non-zero ghost shift, query the
        # BVH against the leaf box translated by `-shift` (equivalent to
        # finding simplices whose AABB, when shifted by `+shift`,
        # intersects this leaf), then clip the shifted simplex against
        # the leaf box. Ghost overlap entries are attributed to the
        # original simplex index `s`, NOT a synthetic ghost index, so the
        # downstream remap sees them as additional contributions to
        # simplex s's mass balance.
        if ghost_shifts !== nothing
            for shift in ghost_shifts
                q_lo = ntuple(d -> leaf_lo[d] - shift[d], Val(D))
                q_hi = ntuple(d -> leaf_hi[d] - shift[d], Val(D))
                empty!(candidates)
                query_aabb!(candidates, tree, q_lo, q_hi)
                for s in candidates
                    verts = simplex_vertex_positions(lag, Int(s))
                    shifted = ntuple(k -> ntuple(d -> verts[k][d] + shift[d],
                                                    Val(D)),
                                      Val(D + 1))
                    vol, centroid, _ = overlap_simplex_box!(moments_buf,
                                                              scratch,
                                                              shifted,
                                                              leaf_lo, leaf_hi,
                                                              moment_order)
                    vol > zero(T) || continue
                    push_overlap!(builder, s, ci, vol, centroid, moments_buf)
                end
            end
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
                                     moment_order::Int,
                                     ghost_shifts::Union{Nothing,
                                                         Vector{NTuple{D, T}}};
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

            # Periodic ghost path (mirror of the sequential implementation).
            if ghost_shifts !== nothing
                for shift in ghost_shifts
                    q_lo = ntuple(d -> leaf_lo[d] - shift[d], Val(D))
                    q_hi = ntuple(d -> leaf_hi[d] - shift[d], Val(D))
                    empty!(candidates)
                    query_aabb!(candidates, tree, q_lo, q_hi)
                    for s in candidates
                        verts = simplex_vertex_positions(lag, Int(s))
                        shifted = ntuple(j -> ntuple(d -> verts[j][d] + shift[d],
                                                        Val(D)),
                                          Val(D + 1))
                        vol, centroid, _ = overlap_simplex_box!(moments_buf,
                                                                  scratch,
                                                                  shifted,
                                                                  leaf_lo, leaf_hi,
                                                                  moment_order)
                        vol > zero(T) || continue
                        push_overlap!(builder, s, ci, vol, centroid, moments_buf)
                    end
                end
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
