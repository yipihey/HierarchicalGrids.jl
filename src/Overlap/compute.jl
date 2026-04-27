"""
    compute_overlap(lag::SimplicialMesh{D, T}, frame::EulerianFrame{D, T};
                     moment_order::Integer = 3,
                     edge_kind::Symbol = :linear,
                     leaf_size::Integer = 8,
                     parallel::Bool = false,
                     scheduler::Symbol = :dynamic,
                     backend::Symbol = :float,
                     lattice::Union{Nothing, IntegerLattice{D}} = nothing,
                     accumulator::Union{Type{<:Signed}, Nothing} = nothing,
                     ) -> GeometricOverlap{D, T}

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
- `backend` — `:float` (default) routes the per-pair clip through r3djl's
  Float64 backend (existing behavior). `:exact` routes through
  `R3D.IntExact` via the `IntegerLattice` quantization helpers, returning
  a `GeometricOverlap{D, Float64}` whose volumes/moments are dequantized
  from exact-rational arithmetic. The `:exact` path requires
  `T == Float64`. `D ∈ {2, 3, 4}` are supported with full polynomial
  moments. At `D = 1` the `:float` path is already exact (closed-form
  interval intersection), so `:exact` is rejected with a recommendation
  to use `:float`.
- `lattice` — optional `IntegerLattice{D}` controlling the quantization
  grid for `backend = :exact`. When `nothing` (the default), an internal
  lattice is auto-derived from the frame with `bits = 16` and
  `int_type = (D == 4 ? Int64 : Int32)`. Must be `nothing` when
  `backend = :float`.
- `accumulator` — optional integer type for the exact-rational
  accumulator passed to `overlap_simplex_box_exact!`. When `nothing`
  (the default), `_default_accumulator(lattice.int_type, Val(D))`
  decides (`Int128` for D=2/3 with `Int16`/`Int32` lattice, `BigInt`
  for D=4 / wider lattices). Must be `nothing` when `backend = :float`.
- `audit_drops` — diagnostic switch valid only with `backend = :exact`.
  When `false` (the default), `compute_overlap` returns a single
  `GeometricOverlap{D, T}` (existing behavior; bit-for-bit identical
  to the old return path). When `true`, the per-pair loop tracks every
  contribution that the upstream `R3D.IntExact` backend dropped — either
  by returning a non-positive volume (a known D = 2 bug, classed as
  `:negative_volume`), by throwing inside `moments_exact!` on a
  degenerate clip (`:moments_throw`), or by producing an empty overlap
  (`:empty`, informational only) — and `compute_overlap` returns the
  TUPLE `(overlap::GeometricOverlap{D, T}, drops::OverlapDropReport)`.
  The drop report counts negative-volume + moments-throw drops as
  actual data loss; the empty count is recorded for completeness.
  See [`HierarchicalGrids.OverlapDropReport`](@ref). Has no effect on
  the float backend (where drops do not occur and the kwarg is
  rejected with an `ArgumentError`).

# Returns

A `GeometricOverlap{D, T}` with all nonzero pairs and CSR-style indexing
both ways. Iterate with `entries_for_lag` or `entries_for_eul`. When
`audit_drops = true` and `backend = :exact`, the return shape changes
to a 2-tuple `(GeometricOverlap{D, T}, OverlapDropReport)`.

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
                          frame_bcs::Union{Nothing, FrameBoundaries{D}} = nothing,
                          backend::Symbol = :float,
                          lattice::Union{Nothing, IntegerLattice{D}} = nothing,
                          accumulator::Union{Type{<:Signed}, Nothing} = nothing,
                          audit_drops::Bool = false,
                          ) where {D, T}
    edge_kind === :linear ||
        throw(ArgumentError("edge_kind=$edge_kind not yet supported (:linear only). " *
                             "Cubic-edge dimension lifting volume-only path is unblocked, " *
                             "but full polynomial remap requires r3djl higher-order moments " *
                             "(P ≥ 1) at D ≥ 4 (still pending). " *
                             "See src/Overlap/lifting.jl for status."))

    # ------------------------------------------------------------------
    # Backend validation
    # ------------------------------------------------------------------
    backend === :float || backend === :exact ||
        throw(ArgumentError(
            "compute_overlap: backend must be :float or :exact (got $(backend))"))

    if backend === :float
        lattice === nothing ||
            throw(ArgumentError(
                "compute_overlap: `lattice` is only valid when backend = :exact"))
        accumulator === nothing ||
            throw(ArgumentError(
                "compute_overlap: `accumulator` is only valid when backend = :exact"))
        audit_drops === false ||
            throw(ArgumentError(
                "compute_overlap: `audit_drops = true` is only valid when " *
                "backend = :exact (the float backend does not drop pairs)."))
    else  # :exact
        T === Float64 ||
            throw(ArgumentError(
                "compute_overlap: backend = :exact requires Float64 meshes/frame " *
                "(got T = $(T)). The :exact path quantizes Float64 → integer " *
                "lattice internally."))
        D >= 2 ||
            throw(ArgumentError(
                "compute_overlap: backend = :exact is not supported at D = 1. " *
                "The :float backend already uses closed-form interval " *
                "intersection at D = 1 (bit-exact); use backend = :float."))
        D <= 4 ||
            throw(ArgumentError(
                "compute_overlap: backend = :exact is not supported at D = $D. " *
                "Currently supported: D ∈ {2, 3, 4} with full polynomial moments."))
        # Periodic-ghost interaction: defer for now. Quantizing a vertex
        # that has been ghost-shifted by ±period can land outside the
        # default lattice's representable range (the lattice is derived
        # from the frame's interior). Rather than bake in lattice-padding
        # logic that would silently change behavior, we error out and
        # recommend :float — which already handles periodic ghosts
        # correctly.
        if frame_bcs !== nothing
            for d in 1:D
                if is_periodic_axis(frame_bcs, d)
                    throw(ArgumentError(
                        "compute_overlap: backend = :exact with periodic " *
                        "frame_bcs is not yet supported. Periodic ghost " *
                        "shifts can push vertices outside the auto-derived " *
                        "lattice's representable range. Use backend = :float " *
                        "for periodic ghost-overlap; both backends produce " *
                        "the same result up to ~1e-16 relative error on " *
                        "canonical polytopes (see test_exact_audit.jl)."))
                end
            end
        end
    end

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

    # ------------------------------------------------------------------
    # :exact backend dispatch — auto-derive lattice/accumulator and run
    # the integer pipeline. Always sequential for now; the exact path is
    # not yet thread-safe for the Rational{BigInt} accumulator (BigInt
    # allocations per call don't compose cleanly with the per-task
    # builders). PR-D introduced a parallel float path; PR-3 keeps the
    # exact path single-threaded until a follow-up profiles the threading
    # overhead vs the BigInt allocation cost.
    # ------------------------------------------------------------------
    if backend === :exact
        lat = lattice !== nothing ? lattice :
              IntegerLattice(frame; bits = 16,
                             int_type = (D == 4 ? Int64 : Int32))
        # Default-accumulator pick: at the per-pair level
        # `_default_accumulator(T, Val(D))` would suffice for the simple
        # canonical polytopes audited by PR-4 (Int128 for D=2/3 with
        # Int16/Int32 lattice). In the broader compute_overlap setting
        # — where Lagrangian simplices clip against many leaf boxes
        # whose corners can sit anywhere on the lattice — the polygon
        # clipper accumulates positions_num × positions_den products
        # whose subsequent moment-integration in Rational{Int128}
        # arithmetic overflows for many clip configurations even at
        # moment_order = 1. We therefore promote the default to
        # `BigInt` when `moment_order >= 1`, which is bulletproof at the
        # cost of ~5-10x slowdown vs Int128. Callers who know their
        # geometry stays small can pass `accumulator = Int128`
        # explicitly.
        # `moment_order = 0` (volume-only) integrates only `twoa // 2`
        # per polygon edge — the simpler 0-order recursion fits
        # comfortably in `_default_accumulator` 's Int128 default and
        # we keep that default in the volume-only case.
        acc = accumulator !== nothing ? accumulator :
              (moment_order == 0 ?
               _default_accumulator(lat.int_type, Val(D)) :
               BigInt)
        return _compute_overlap_sequential_exact(lag, frame, tree,
                                                  Int(moment_order),
                                                  ghost_shifts, lat, acc;
                                                  audit_drops = audit_drops)
    end

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

# ----------------------------------------------------------------------------
# :exact backend — sequential integer-rational pipeline.
#
# The per-pair flow:
#
#   1. Quantize the leaf box and the simplex's physical vertices onto
#      the integer lattice.
#   2. Call `overlap_simplex_box_exact!` → `Rational{R}` moments in
#      the LATTICE-INTEGER coordinate frame. The 0th moment (volume)
#      is translation-invariant; higher-degree moments are "in lattice
#      coordinates", i.e. with origin at `lat.lo`.
#   3. Convert each lattice-frame Rational moment back to Float64:
#        - 0th moment   → `unscale_volume(m, lat)`             (physical
#                          volume, scale-only undo).
#        - degree-k ≥ 1 → `unscale_moment(m, lat, k)`          (lattice-
#                          frame physical units, scale-only undo).
#   4. Translate the resulting Float64 moments from the lattice frame
#      (origin at `lat.lo`) to the physical frame (origin at 0) using
#      `shift_moments!` with `Δ = -lat.lo`.
#
#      Why `Δ = -lat.lo`? `shift_moments!` implements
#         M_new(a) = ∫ (x - x_new)^a dV = Σ binomial(a, b) (-Δ)^(a-b) M_old(b)
#      where `Δ = x_new - x_old` is the shift of the origin. The lattice
#      coordinate is `x_lat = x_phys - lat.lo`; the lattice-frame moments
#      take `x_old = lat.lo` as their origin. To get physical-frame
#      moments (origin at 0) we set `x_new = 0`, hence
#         Δ = x_new - x_old = 0 - lat.lo = -lat.lo
#
# The centroid is recomputed from the shifted physical-frame moments
# (entries 2..D+1 of the moment vector divided by the volume) when
# `moment_order >= 1`; otherwise the centroid is a zero placeholder
# (consistent with the float adapter's convention at order 0).
# ----------------------------------------------------------------------------
function _compute_overlap_sequential_exact(
        lag::SimplicialMesh{D, Float64},
        frame::EulerianFrame{D, Float64},
        tree::SimplicialAABBTree{D, Float64},
        moment_order::Int,
        ghost_shifts::Union{Nothing, Vector{NTuple{D, Float64}}},
        lat::IntegerLattice{D},
        ::Type{R};
        audit_drops::Bool = false) where {D, R<:Signed}
    builder = OverlapBuilder{D, Float64}(moment_order)
    scratch = IntPairScratch(Val(D), lat.int_type; capacity = 64)
    n_phys = moments_length(D, moment_order)
    int_moment_buf = Vector{Rational{R}}(undef, n_phys)
    # Lattice-frame Float64 moments (after unscale, before shift).
    lat_phys_buf = Vector{Float64}(undef, n_phys)
    # Final physical-frame Float64 moments (after shift_moments!).
    phys_moments = Vector{Float64}(undef, n_phys)
    candidates = Int32[]; sizehint!(candidates, 64)

    # Drop bookkeeping (only populated when audit_drops = true).
    n_neg = 0
    n_throw = 0
    n_empty_cnt = 0
    drops = Vector{NamedTuple{(:lag_idx, :eul_idx, :kind),
                              Tuple{Int32, Int32, Symbol}}}()

    # Cache of per-flat-index total degrees for unscale_moment dispatch.
    # `moment_multiindices` is cached at the moments.jl layer; we just
    # precompute total degrees once.
    multi = moment_multiindices(Int(D), moment_order)
    total_degrees = ntuple_or_vec_degrees(multi)

    # `Δ` in `shift_moments!`: lattice origin is `lat.lo` in physical
    # coordinates; physical origin is 0. See block comment above.
    delta = ntuple(d -> -lat.lo[d], Val(D))

    eul = frame.mesh
    @inbounds for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        leaf_lo, leaf_hi = cell_physical_box(frame, ci)
        empty!(candidates)
        query_aabb!(candidates, tree, leaf_lo, leaf_hi)
        if !isempty(candidates)
            q_box_lo = quantize(leaf_lo, lat)
            q_box_hi = quantize(leaf_hi, lat)
            for s in candidates
                verts = simplex_vertex_positions(lag, Int(s))
                q_verts = ntuple(k -> quantize(verts[k], lat), Val(D + 1))

                # Audit-mode dispatch: use the drop-kind variant so we
                # can distinguish empty-overlap from upstream-bug drops.
                # Wrap in try/catch to surface `:moments_throw` drops
                # (R3D.IntExact.moments_exact! can raise on degenerate
                # clip polytopes — known D = 2 upstream issue).
                vol_rat, _centroid_rat, _out_rats, drop_kind =
                if audit_drops
                    try
                        overlap_simplex_box_exact_with_drop_kind!(
                            int_moment_buf, scratch,
                            q_verts, q_box_lo, q_box_hi,
                            moment_order; accumulator = R)
                    catch err
                        # Capacity overflow is a programmer error
                        # (scratch buffer too small) — do NOT mask
                        # those as drops. Everything else (including
                        # the upstream `ArgumentError("0//0")` from
                        # `R3D.IntExact._vertex_rational_d2`) is
                        # treated as a `:moments_throw` drop.
                        if err isa OverflowError
                            rethrow(err)
                        end
                        n_throw += 1
                        push!(drops, (lag_idx = Int32(s),
                                       eul_idx = Int32(ci),
                                       kind = :moments_throw))
                        # Synthesize an empty result so downstream stays
                        # consistent — the entry is dropped, no push.
                        fill!(int_moment_buf, zero(Rational{R}))
                        (zero(Rational{R}),
                         ntuple(_ -> zero(Rational{R}), Val(D)),
                         int_moment_buf, :moments_throw)
                    end
                else
                    vol, cent, out, _ =
                        overlap_simplex_box_exact_with_drop_kind!(
                            int_moment_buf, scratch,
                            q_verts, q_box_lo, q_box_hi,
                            moment_order; accumulator = R)
                    (vol, cent, out, :none)
                end

                if iszero(vol_rat)
                    if audit_drops
                        if drop_kind === :negative_volume
                            n_neg += 1
                            push!(drops, (lag_idx = Int32(s),
                                           eul_idx = Int32(ci),
                                           kind = :negative_volume))
                        elseif drop_kind === :empty
                            n_empty_cnt += 1
                            push!(drops, (lag_idx = Int32(s),
                                           eul_idx = Int32(ci),
                                           kind = :empty))
                        end
                        # :moments_throw was already recorded above.
                    end
                    continue
                end
                vol_phys = unscale_volume(vol_rat, lat)
                if vol_phys <= 0.0
                    # Lattice-quantization rounded a non-zero rational down
                    # to zero physical volume — informational; record as
                    # :empty (not data loss).
                    if audit_drops
                        n_empty_cnt += 1
                        push!(drops, (lag_idx = Int32(s),
                                       eul_idx = Int32(ci),
                                       kind = :empty))
                    end
                    continue
                end

                if moment_order == 0
                    # Volume-only path: no shift needed; centroid is a
                    # zero placeholder (consistent with the int adapter
                    # and float-order-0 convention).
                    phys_moments[1] = vol_phys
                    centroid = ntuple(_ -> 0.0, Val(D))
                    push_overlap!(builder, s, ci, vol_phys, centroid,
                                  phys_moments)
                else
                    # 1) Unscale every Rational moment to Float64 in the
                    #    LATTICE-coordinate frame.
                    for k in 1:n_phys
                        kdeg = total_degrees[k]
                        lat_phys_buf[k] = unscale_moment(int_moment_buf[k],
                                                          lat, kdeg)
                    end
                    # 2) Translate to the PHYSICAL-coordinate frame.
                    shift_moments!(phys_moments, lat_phys_buf,
                                   D, moment_order, delta)
                    # 3) Centroid from physical-frame moments.
                    centroid = ntuple(d -> phys_moments[d + 1] / vol_phys,
                                      Val(D))
                    push_overlap!(builder, s, ci, vol_phys, centroid,
                                  phys_moments)
                end
            end
        end

        # No periodic-ghost path here: validation upstream rejects
        # backend = :exact when any periodic axis is present in
        # `frame_bcs`. `ghost_shifts` is therefore guaranteed to be
        # `nothing` on this code path.
    end

    overlap = finalize_overlap(builder, n_simplices(lag), n_cells(eul))
    if audit_drops
        # Reach across to Diagnostics via the parent module — Overlap
        # imports Diagnostics's `RemapDiagnostics` already, but
        # `OverlapDropReport` is added in this batch and isn't on the
        # `using ..Diagnostics` import list. The parent-module path
        # avoids touching the Overlap module's import surface.
        DropReportT = parentmodule(@__MODULE__).Diagnostics.OverlapDropReport
        report = DropReportT(n_neg, n_throw, n_empty_cnt, drops)
        return (overlap, report)
    end
    return overlap
end

# Helper: extract the total degree of every multi-index in `multi` as a
# plain `Vector{Int}` (faster than re-summing inside the hot loop).
@inline function ntuple_or_vec_degrees(multi::Vector{Vector{Int}})
    out = Vector{Int}(undef, length(multi))
    @inbounds for k in eachindex(multi)
        s = 0
        for e in multi[k]
            s += e
        end
        out[k] = s
    end
    return out
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
