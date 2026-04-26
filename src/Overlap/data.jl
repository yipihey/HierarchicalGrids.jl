"""
    OverlapEntry{D, T}

One nonzero pair in the geometric overlap between a Lagrangian simplicial
mesh and an Eulerian hierarchical mesh.

# Fields

- `lag_idx::Int32` — Lagrangian simplex index (1-based).
- `eul_idx::Int32` — Eulerian leaf cell index in the mesh's storage order.
- `volume::T` — overlap measure (length in 1D, area in 2D, volume in 3D).
- `centroid::NTuple{D, T}` — overlap centroid in physical coordinates.
- `moments::Vector{T}` — full moment vector up to the configured order,
  in graded-lex order, with origin at the **physical origin** (not the
  centroid). Use `shift_moments!` to translate to a different origin
  if needed.

The choice of origin for `moments` matches r3djl's convention: moments
are computed in the same coordinate frame the polytope is initialized in.
For the dfmm Bayesian remap, this is convenient because both source and
destination cells live in the same physical frame.
"""
struct OverlapEntry{D, T}
    lag_idx::Int32
    eul_idx::Int32
    volume::T
    centroid::NTuple{D, T}
    moments::Vector{T}
end

"""
    GeometricOverlap{D, T}

Sparse geometric overlap between a Lagrangian simplicial mesh and an
Eulerian hierarchical mesh, with CSR-like dual indexing for cache-friendly
traversal in either direction (per Lagrangian simplex or per Eulerian
leaf).

# Fields

- `entries::Vector{OverlapEntry{D, T}}` — all nonzero pairs.
- `lag_to_entries::Vector{UnitRange{Int}}` — for each Lagrangian simplex,
  the contiguous range of indices in `entries` that belong to it. Length
  equals `n_simplices(lag)`. Empty range for simplices with no overlap.
- `eul_to_entries::Vector{Vector{Int}}` — for each Eulerian leaf, the list
  of indices in `entries` covering it. Vector-of-vectors rather than CSR
  because the per-leaf entries are not contiguous in `entries`. Length
  equals `n_cells(eul)`; non-leaf cells get empty vectors.
- `moment_order::Int` — order of polynomial moments stored per entry.
- `n_lag::Int`, `n_eul::Int` — saved sizes for validation.

# Invariants

- `entries` is sorted by `lag_idx` ascending; ties broken by `eul_idx`.
  This makes `lag_to_entries` a valid CSR row-pointer structure.
- All `volume` values are strictly positive.
- All `moments` vectors have the same length `moments_length(D, moment_order)`.
"""
struct GeometricOverlap{D, T}
    entries::Vector{OverlapEntry{D, T}}
    lag_to_entries::Vector{UnitRange{Int}}
    eul_to_entries::Vector{Vector{Int}}
    moment_order::Int
    n_lag::Int
    n_eul::Int
end

@inline n_entries(o::GeometricOverlap) = length(o.entries)
@inline moment_order(o::GeometricOverlap) = o.moment_order
@inline spatial_dimension(::GeometricOverlap{D, T}) where {D, T} = D

"""
    entries_for_lag(o::GeometricOverlap, lag_idx::Integer) -> SubArray

View into `o.entries` for all pairs whose Lagrangian simplex is `lag_idx`.
Empty if that simplex has no overlap.
"""
@inline function entries_for_lag(o::GeometricOverlap, lag_idx::Integer)
    rng = o.lag_to_entries[lag_idx]
    return view(o.entries, rng)
end

"""
    entries_for_eul(o::GeometricOverlap, eul_idx::Integer) -> Vector{OverlapEntry}

The overlap entries covering Eulerian cell `eul_idx`. Returns an array
of entries (copy of references; the entries themselves are shared).
Empty if `eul_idx` is not a leaf or has no overlap.
"""
@inline function entries_for_eul(o::GeometricOverlap{D, T}, eul_idx::Integer) where {D, T}
    idxs = o.eul_to_entries[eul_idx]
    return [o.entries[k] for k in idxs]
end

"""
    total_overlap_volume(o::GeometricOverlap) -> T

Sum of all entry volumes. Useful for conservation diagnostics: equals
the total measure of (Lagrangian domain) ∩ (Eulerian domain) restricted
to the parts both meshes cover.
"""
function total_overlap_volume(o::GeometricOverlap{D, T}) where {D, T}
    s = zero(T)
    @inbounds for e in o.entries
        s += e.volume
    end
    return s
end

function Base.show(io::IO, o::GeometricOverlap{D, T}) where {D, T}
    print(io, "GeometricOverlap{$D, $T}(",
          n_entries(o), " entries; ",
          o.n_lag, " Lagrangian simplices, ",
          o.n_eul, " Eulerian cells; moment_order=", o.moment_order,
          ")")
end

# ============================================================================
# Builder — accumulates entries during compute_overlap
# ============================================================================

"""
    OverlapBuilder{D, T}

Mutable accumulator used during `compute_overlap`. Internal type — most
users won't touch it directly, but it's exposed for parallel (per-thread)
construction.

The builder collects unsorted entries; `finalize!` sorts and constructs
the CSR-style index. For parallel use, allocate one builder per thread,
collect entries independently, then `merge!` into a primary builder
before finalizing.
"""
mutable struct OverlapBuilder{D, T}
    entries::Vector{OverlapEntry{D, T}}
    moment_order::Int
end

OverlapBuilder{D, T}(moment_order::Integer = 3) where {D, T} =
    OverlapBuilder{D, T}(OverlapEntry{D, T}[], Int(moment_order))

OverlapBuilder(::Val{D}, ::Type{T}, moment_order::Integer = 3) where {D, T} =
    OverlapBuilder{D, T}(moment_order)

@inline n_pending(b::OverlapBuilder) = length(b.entries)

"""
    push_overlap!(b::OverlapBuilder{D, T}, lag_idx, eul_idx, volume, centroid, moments)

Append a single nonzero overlap entry. The `moments` vector is **copied**
into the entry; the caller may safely reuse its buffer.
"""
function push_overlap!(b::OverlapBuilder{D, T},
                        lag_idx::Integer, eul_idx::Integer,
                        volume::Real, centroid::NTuple{D},
                        moments::AbstractVector) where {D, T}
    expected = moments_length(D, b.moment_order)
    length(moments) == expected ||
        throw(DimensionMismatch("moments has length $(length(moments)), expected $expected"))
    entry = OverlapEntry{D, T}(
        Int32(lag_idx), Int32(eul_idx),
        T(volume),
        ntuple(d -> T(centroid[d]), D),
        Vector{T}(moments),
    )
    push!(b.entries, entry)
    return b
end

"""
    merge_builder!(dest::OverlapBuilder, src::OverlapBuilder)

Append all entries from `src` into `dest`. After this call, `src` retains
its entries (no clearing); call `empty!(src.entries)` if you want to
reuse it.

Used to combine per-thread builders before finalization.
"""
function merge_builder!(dest::OverlapBuilder{D, T}, src::OverlapBuilder{D, T}) where {D, T}
    dest.moment_order == src.moment_order ||
        throw(ArgumentError("cannot merge builders with different moment_order"))
    append!(dest.entries, src.entries)
    return dest
end

"""
    finalize_overlap(b::OverlapBuilder, n_lag::Integer, n_eul::Integer) -> GeometricOverlap

Sort entries and build the CSR-style indices. `n_lag` and `n_eul` are
the number of Lagrangian simplices and total Eulerian cells (leaf and
non-leaf), used to size the index vectors.
"""
function finalize_overlap(b::OverlapBuilder{D, T}, n_lag::Integer, n_eul::Integer) where {D, T}
    sort!(b.entries, by = e -> (e.lag_idx, e.eul_idx))
    n_l = Int(n_lag)
    n_e = Int(n_eul)

    # CSR row pointers for Lagrangian indexing
    lag_to_entries = Vector{UnitRange{Int}}(undef, n_l)
    fill!(lag_to_entries, 1:0)  # empty range as default
    if !isempty(b.entries)
        # Sweep once
        cur_lag = Int(b.entries[1].lag_idx)
        run_start = 1
        for k in 2:length(b.entries)
            l = Int(b.entries[k].lag_idx)
            if l != cur_lag
                lag_to_entries[cur_lag] = run_start:(k - 1)
                cur_lag = l
                run_start = k
            end
        end
        lag_to_entries[cur_lag] = run_start:length(b.entries)
    end

    # Eulerian inverse index (vector-of-vectors)
    eul_to_entries = [Int[] for _ in 1:n_e]
    for k in 1:length(b.entries)
        push!(eul_to_entries[Int(b.entries[k].eul_idx)], k)
    end

    return GeometricOverlap{D, T}(
        b.entries, lag_to_entries, eul_to_entries,
        b.moment_order, n_l, n_e,
    )
end
