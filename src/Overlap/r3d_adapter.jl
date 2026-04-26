"""
Adapter between the overlap layer's needs and the r3djl polytope library.

This is the only file in the Overlap module that knows about r3djl
specifics. Everything else uses the abstract API:

    overlap_simplex_box!(out_moments, scratch, simplex_vertices,
                          box_lo, box_hi, moment_order)
        -> (volume, centroid, out_moments)

Returns `(0, (0, ...))` if the overlap is empty.

# Backend

Real R3D backend via the r3djl package. Supports D=2 and D=3 directly
through `R3D.Flat.init_simplex!` → `R3D.Flat.box_planes!` →
`R3D.Flat.clip!` → `R3D.Flat.moments!`. D=1 uses a closed-form interval
intersection (no r3djl call) — intervals are trivial polytopes and the
moments are elementary integrals. D ≥ 4 throws a clean error:
r3djl now provides correct `clip!` (single + sequential, simplex bug
fixed in r3djl f704cff) and `moments!` order = 0 at D ≥ 4, but
`compute_overlap` consumes higher-order moments (centroid +
polynomial-remap moments) which are still pending at D ≥ 4. Cubic-
edge dimension lifting (see `lifting.jl`) needs the same higher-order
moments before it can be plumbed through.

# Convention agreement

R3D uses the half-space convention `n·x + d ≥ 0` for "kept", and graded-lex
moment ordering with the first index most significant
(`1, x, y, z, x², xy, xz, y², yz, z², ...`). Both match the rest of the
overlap layer; the moment ordering is verified at module load via the
`_verify_moment_ordering` call at the bottom of this file.

# Scratch workspace

The hot loop wants zero allocations per pair. The adapter exposes a
`PairScratch` type holding the persistent FlatBuffer and plane buffer.
Allocate once per thread and reuse across many `overlap_simplex_box!`
calls.
"""

using R3D

# ============================================================================
# Per-thread scratch workspace
# ============================================================================

"""
    PairScratch{D, T}

Persistent workspace for the per-pair overlap computation. Allocate once
per thread; reuse across many `overlap_simplex_box!` calls.

# Fields

- `poly::R3D.Flat.FlatPolytope{D, T}` — working polytope buffer reused for
  every (simplex, box) pair. Re-initialized via `init_simplex!` on each
  call. Capacity is set generously (default 64 vertices) since clipping
  a simplex against a box can at most produce a handful of vertices in
  practice.
- `plane_buf::Vector{R3D.Plane{D, T}}` — plane buffer pre-sized to `2D`.
  Refilled from each leaf box via `box_planes!`.
"""
mutable struct PairScratch{D, T}
    poly::R3D.Flat.FlatPolytope{D, T}
    plane_buf::Vector{R3D.Plane{D, T}}
end

function PairScratch(::Val{D}, ::Type{T}; capacity::Int = 64) where {D, T}
    poly = R3D.Flat.FlatPolytope{D, T}(capacity)
    # Pre-size plane buffer with placeholder values; box_planes! will
    # overwrite them on each call.
    placeholder_n = R3D.Vec{D, T}(ntuple(d -> d == 1 ? one(T) : zero(T), Val(D))...)
    placeholder = R3D.Plane{D, T}(placeholder_n, zero(T))
    return PairScratch{D, T}(poly, fill(placeholder, 2 * D))
end

# D=1 specialization: the closed-form interval path doesn't use the
# polytope or plane buffers. We still allocate a minimal FlatPolytope
# (size 1, zero capacity is fine since we never touch it) so the type
# signature `PairScratch{1, T}` is satisfied; this keeps the type stable
# across D without burdening the 1D path with a real polytope buffer.
function PairScratch(::Val{1}, ::Type{T}; capacity::Int = 64) where {T}
    poly = R3D.Flat.FlatPolytope{1, T}(1)
    placeholder_n = R3D.Vec{1, T}(one(T))
    placeholder = R3D.Plane{1, T}(placeholder_n, zero(T))
    return PairScratch{1, T}(poly, R3D.Plane{1, T}[placeholder, placeholder])
end

# ============================================================================
# Public adapter entry point
# ============================================================================

"""
    overlap_simplex_box!(out_moments::AbstractVector{T},
                          scratch::PairScratch{D, T},
                          simplex_vertices,
                          box_lo::NTuple{D, T}, box_hi::NTuple{D, T},
                          moment_order::Integer) where {D, T}
    -> (volume::T, centroid::NTuple{D, T}, out_moments)

Compute the overlap polytope of a D-simplex (with `D+1` vertices) against
an axis-aligned box, returning its volume, centroid, and full moment
vector up to `moment_order` in graded-lex order. The moments are written
to `out_moments` (which must have length `moments_length(D, moment_order)`).

Returns `volume = 0` and `centroid = (0, ..., 0)` if the overlap is
empty (in which case `out_moments` is left filled with zeros).

The scratch workspace is reused across calls; no per-call heap allocation
once the polytope buffers are sized.
"""
function overlap_simplex_box!(out_moments::AbstractVector{T},
                                scratch::PairScratch{D, T},
                                simplex_vertices,
                                box_lo::NTuple{D, T},
                                box_hi::NTuple{D, T},
                                moment_order::Integer) where {D, T}
    expected_len = moments_length(D, moment_order)
    length(out_moments) == expected_len ||
        throw(DimensionMismatch("out_moments has length $(length(out_moments)), expected $expected_len"))
    fill!(out_moments, zero(T))
    return _overlap_dispatch(out_moments, scratch, simplex_vertices,
                              box_lo, box_hi, Int(moment_order), Val(D))
end

# Dispatch by dimension. Specific methods for D=2 and D=3 use R3D directly;
# the generic catch-all throws for D ≥ 4: r3djl's higher-order moments
# (P ≥ 1) at D ≥ 4 aren't yet implemented, which `compute_overlap` needs
# for centroids and polynomial-remap moments. (The previous D ≥ 4 simplex
# sequential-clip bug was fixed in r3djl f704cff.) See lifting.jl for
# the full status note.
function _overlap_dispatch(out_moments::AbstractVector{T},
                            scratch::PairScratch{D, T},
                            simplex_vertices,
                            box_lo::NTuple{D, T},
                            box_hi::NTuple{D, T},
                            moment_order::Int,
                            ::Val{D}) where {D, T}
    throw(ErrorException(
        "overlap_simplex_box! at D=$D not supported. r3djl provides correct " *
        "clip + 0th-moment at D ≥ 4, but `compute_overlap` consumes higher-" *
        "order moments (centroid + polynomial-remap moments) which aren't " *
        "yet implemented in r3djl at D ≥ 4. " *
        "See src/Overlap/lifting.jl for status."))
end

_overlap_dispatch(out::AbstractVector{T}, scratch::PairScratch{1, T},
                   verts, lo::NTuple{1, T}, hi::NTuple{1, T},
                   order::Int, ::Val{1}) where {T} =
    _overlap_1d!(out, verts, lo, hi, order)

_overlap_dispatch(out::AbstractVector{T}, scratch::PairScratch{2, T},
                   verts, lo::NTuple{2, T}, hi::NTuple{2, T},
                   order::Int, ::Val{2}) where {T} =
    _overlap_via_r3d!(out, scratch, verts, lo, hi, order)

_overlap_dispatch(out::AbstractVector{T}, scratch::PairScratch{3, T},
                   verts, lo::NTuple{3, T}, hi::NTuple{3, T},
                   order::Int, ::Val{3}) where {T} =
    _overlap_via_r3d!(out, scratch, verts, lo, hi, order)

# ============================================================================
# Closed-form 1D implementation
# ============================================================================

# In 1D, a "simplex" is a segment with two vertices (a, b), and the
# Eulerian "box" is the interval [box_lo[1], box_hi[1]]. The overlap is
# an interval [lo, hi] computed by interval intersection. The k-th moment
# integrated about the origin (matching the D=2/3 r3djl convention — see
# `_verify_moment_ordering` and `moments.jl`) is
#
#     M_k = ∫_lo^hi x^k dx = (hi^{k+1} - lo^{k+1}) / (k + 1)
#
# Graded-lex ordering for D=1 collapses to ascending degree:
#     out_moments[k+1] = M_k   for k = 0, ..., moment_order
#
# The 0th moment is the length (volume); the 1st divided by length is
# the centroid x-coordinate. Returns `(0, (0,))` for empty overlaps.
@inline function _overlap_1d!(out_moments::AbstractVector{T},
                                simplex_vertices,
                                box_lo::NTuple{1, T},
                                box_hi::NTuple{1, T},
                                moment_order::Int) where {T}
    # Simplex vertices is a 2-element collection of NTuple{1, T}.
    a = simplex_vertices[1][1]
    b = simplex_vertices[2][1]
    seg_lo = a < b ? a : b
    seg_hi = a < b ? b : a

    lo = max(seg_lo, box_lo[1])
    hi = min(seg_hi, box_hi[1])

    if hi <= lo
        # Empty / degenerate overlap.
        return (zero(T), (zero(T),), out_moments)
    end

    vol = hi - lo
    out_moments[1] = vol
    # Higher-order moments in graded-lex (== ascending degree in 1D).
    @inbounds for k in 1:moment_order
        out_moments[k + 1] = (hi^(k + 1) - lo^(k + 1)) / T(k + 1)
    end

    # Centroid requires order >= 1; otherwise sentinel zero.
    centroid = if moment_order >= 1
        (out_moments[2] / vol,)
    else
        (zero(T),)
    end
    return (vol, centroid, out_moments)
end

# ============================================================================
# R3D-backed implementation (D=2, D=3)
# ============================================================================

@inline function _overlap_via_r3d!(out_moments::AbstractVector{T},
                                     scratch::PairScratch{D, T},
                                     simplex_vertices,
                                     box_lo::NTuple{D, T},
                                     box_hi::NTuple{D, T},
                                     moment_order::Int) where {D, T}
    poly = scratch.poly
    planes = scratch.plane_buf

    # Initialize working polytope to the simplex.
    R3D.Flat.init_simplex!(poly, simplex_vertices)

    # Refill plane buffer with this leaf's box planes (zero-alloc).
    R3D.Flat.box_planes!(planes, box_lo, box_hi)

    # Clip in-place.
    R3D.Flat.clip!(poly, planes)

    if R3D.Flat.is_empty(poly)
        return (zero(T), ntuple(_ -> zero(T), Val(D)), out_moments)
    end

    # Integrate moments to required order.
    R3D.Flat.moments!(out_moments, poly, moment_order)
    vol = out_moments[1]
    if vol <= zero(T)
        # Numerically-degenerate overlap; treat as empty.
        fill!(out_moments, zero(T))
        return (zero(T), ntuple(_ -> zero(T), Val(D)), out_moments)
    end
    # Centroid requires order ≥ 1 (which means out_moments has length ≥ D+1).
    # When the caller asked for moment_order == 0, no first moments are
    # available; return a sentinel zero centroid in that case.
    centroid = if moment_order >= 1
        ntuple(d -> out_moments[d + 1] / vol, Val(D))
    else
        ntuple(_ -> zero(T), Val(D))
    end
    return (vol, centroid, out_moments)
end

# ============================================================================
# Convention check (runs once at module load)
# ============================================================================

# Verify that r3djl's moment ordering matches ours by integrating moments
# of a simple known polytope and comparing against the expected graded-lex
# order with first-index-most-significant. This catches a future r3djl
# update that silently changes the convention.
function _verify_moment_ordering()
    # 2D unit triangle (0,0)-(1,0)-(0,1):
    #   Order 2 moments in our ordering [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]:
    #     (1/2, 1/6, 1/6, 1/12, 1/24, 1/12)
    poly = R3D.Flat.FlatPolytope{2, Float64}(32)
    R3D.Flat.init_simplex!(poly, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    m = R3D.Flat.moments(poly, 2)
    expected = [0.5, 1/6, 1/6, 1/12, 1/24, 1/12]
    for k in eachindex(expected)
        if !isapprox(m[k], expected[k]; atol = 1e-12)
            error("r3djl moment ordering disagreement at index $k: got $(m[k]), expected $(expected[k]). " *
                  "This means r3djl's moment ordering convention has changed and the adapter needs " *
                  "to be updated to match.")
        end
    end
    return true
end

# Run once at module-load time.
_verify_moment_ordering()
