"""
Integer-exact adapter between the overlap layer and r3djl's
`R3D.IntExact` submodule. Mirrors the float-only `r3d_adapter.jl`, but
clips simplices against axis-aligned boxes in pure shared-denominator
integer arithmetic and integrates polynomial moments as exact rationals.

This file is the only place in HierarchicalGrids that knows about
`R3D.IntExact`. Every other Overlap-module consumer of the integer
backend goes through

    overlap_simplex_box_exact!(out_moments, scratch,
                               simplex_vertices, box_lo, box_hi,
                               moment_order;
                               accumulator = _default_accumulator(T, D))
        -> (volume::Rational{R},
            centroid::NTuple{D, Rational{R}},
            out_moments)

Returns `volume = 0 // 1` and zero moments when the overlap is empty
(simplex outside box, fully clipped, or numerically degenerate).

The centroid is returned as `Rational{R}`, NOT `Float64`. PR-3 will
dequantize at the call boundary (lattice scale + degree-aware
rescaling); keeping the boundary exact here means the float boundary
is the only place rounding enters.

# Plane convention

`R3D.IntExact` uses the same half-space convention as `R3D.Flat`:
`{x : n · x + d ≥ 0}` is kept. We construct the box's `2D` clip planes
manually since `R3D.IntExact` does not ship a `box_planes!` helper —
the integer encoding makes the construction trivial (axis-aligned ±1
normals, integer offsets `-lo[k]` and `+hi[k]`). A startup check
(`_verify_intexact_plane_convention()`) clips a known polytope and
asserts the convention matches.

# Moment ordering

`R3D.IntExact.moments_exact!` writes moments in the SAME canonical
lex-by-degree order as `R3D.Flat.moments!` (verified at the upstream
level via `R3D.num_moments(D, P)` and the per-D Koehl recursion in
`intexact.jl`). The float adapter already verifies that ordering
matches HG's graded-lex via `_verify_moment_ordering` at module load;
the integer path inherits that agreement and adds an exact-rational
cross-check below for defense in depth.

# Scratch workspace

The hot loop wants zero allocations per pair on the IntExact `clip!`
path itself. The `IntPairScratch` type holds the persistent
`IntFlatPolytope` and pre-sized `Plane` buffer. The `moments_exact!`
call still allocates internally (per-call polynomial scratch +
`Rational{R}` outputs); that's a property of the upstream Koehl
recursion in exact-rational form, not of this adapter.
"""

using R3D

# ============================================================================
# Per-thread scratch workspace
# ============================================================================

"""
    IntPairScratch{D, T<:Signed}

Persistent workspace for the per-pair integer-exact overlap
computation. Allocate once per thread; reuse across many
`overlap_simplex_box_exact!` calls.

# Fields

- `poly::R3D.IntExact.IntFlatPolytope{D, T}` — working polytope
  reused for every (simplex, box) pair. Re-initialized via the
  appropriate `init_simplex!` / `init_tet!` on each call.
- `plane_buf::Vector{R3D.Plane{D, T}}` — plane buffer pre-sized to
  `2D`. Refilled with the leaf box's clip planes via the internal
  `_fill_box_planes_int!` helper (zero alloc).
- `moment_buf::Vector{Rational{T}}` — placeholder small buffer reserved
  for callers that need a per-scratch landing zone for moments at the
  storage type `T`. The public adapter writes into the caller-supplied
  output vector, NOT into this buffer; it's exposed solely for tests
  that want a `Rational{T}`-typed scratch with the same lifetime as the
  rest of the workspace.

# Capacity

`capacity` (default 64) bounds the number of vertices the working
polytope can hold after clipping. A simplex against an axis-aligned
box clips to at most a small constant in practice (≤ 4 for D=2, ≤ 8
for D=3). 64 leaves comfortable headroom for repeated reuse.
"""
mutable struct IntPairScratch{D, T<:Signed}
    poly::R3D.IntExact.IntFlatPolytope{D, T}
    plane_buf::Vector{R3D.Plane{D, T}}
    moment_buf::Vector{Rational{T}}
end

function IntPairScratch(::Val{D}, ::Type{T}; capacity::Int = 64) where {D, T<:Signed}
    poly = R3D.IntExact.IntFlatPolytope{D, T}(capacity)
    placeholder_n = R3D.Vec{D, T}(ntuple(d -> d == 1 ? one(T) : zero(T), Val(D))...)
    placeholder = R3D.Plane{D, T}(placeholder_n, zero(T))
    plane_buf = fill(placeholder, 2 * D)
    # `moment_buf` is reserved scratch (see docstring); 1 entry is enough
    # to hold the volume in the storage type if a caller wants it.
    moment_buf = Vector{Rational{T}}(undef, 0)
    return IntPairScratch{D, T}(poly, plane_buf, moment_buf)
end

# ============================================================================
# Default-accumulator selection
# ============================================================================

"""
    _default_accumulator(::Type{T}, ::Val{D}) -> Type{<:Signed}

Recommended accumulator type `R` for `moments_exact!` and the
exact-volume fan triangulation given input coordinate type `T` and
polytope dimension `D`. Mirrors the type-promotion guidance in
`R3D.IntExact`'s docstring (`intexact.jl` lines 22-34):

| input `T` | D ∈ {2, 3} | D = 4 |
|-----------|------------|-------|
| `Int16`   | `Int128`   | `BigInt` |
| `Int32`   | `Int128`   | `BigInt` |
| `Int64`   | `BigInt`   | `BigInt` |
| `Int128`  | `BigInt`   | `BigInt` |
| `BigInt`  | `BigInt`   | `BigInt` |

These defaults are conservative — they assume up to a few oblique
clips on top of the 2D axis-aligned box clips. Callers that know
their geometry stays axis-aligned can pass a tighter accumulator
(`R = Int64` for `T = Int16`, e.g.) via the `accumulator` kwarg.

D = 4 always gets `BigInt` regardless of `T`: the per-simplex `det(M)`
in the exact-volume fan triangulation has 4-fold integer products of
positions_num that exceed `Int128` even for `T = Int16` after a few
clips.
"""
@inline function _default_accumulator(::Type{T}, ::Val{D}) where {T<:Signed, D}
    D >= 4 && return BigInt
    if T === Int16 || T === Int32
        return Int128
    elseif T === Int64
        return BigInt
    elseif T === Int128
        return BigInt
    elseif T === BigInt
        return BigInt
    else
        # Unknown signed integer type — be conservative.
        return BigInt
    end
end

# Keep the default-accumulator function silent on `BigInt` input (avoids
# the `T === BigInt` chain re-promoting to BigInt when it's already BigInt).
@inline _default_accumulator(::Type{BigInt}, ::Val{D}) where {D} = BigInt

# ============================================================================
# Box-plane construction (no upstream `box_planes!` for IntExact)
# ============================================================================

# Fill `out` with the `2D` axis-aligned-box clip planes for `[lo, hi]`
# in the same order as `R3D.Flat.box_planes!`: `(+axis_k, -axis_k)`
# pairs for `k = 1, …, D`. Plane orientation is `n · x + d ≥ 0`
# (kept). Integer encoding: normals are unit vectors with `±1`
# components, offsets are `-lo[k]` (positive face) and `+hi[k]`
# (negative face), so the plane test reduces to a single-axis
# integer comparison.
@inline function _fill_box_planes_int!(out::AbstractVector{R3D.Plane{2, T}},
                                        lo::NTuple{2, T},
                                        hi::NTuple{2, T}) where {T<:Signed}
    @inbounds begin
        out[1] = R3D.Plane{2, T}(R3D.Vec{2, T}(T( 1), T(0)), -lo[1])
        out[2] = R3D.Plane{2, T}(R3D.Vec{2, T}(T(-1), T(0)),  hi[1])
        out[3] = R3D.Plane{2, T}(R3D.Vec{2, T}(T(0),  T( 1)), -lo[2])
        out[4] = R3D.Plane{2, T}(R3D.Vec{2, T}(T(0),  T(-1)),  hi[2])
    end
    return out
end

@inline function _fill_box_planes_int!(out::AbstractVector{R3D.Plane{3, T}},
                                        lo::NTuple{3, T},
                                        hi::NTuple{3, T}) where {T<:Signed}
    @inbounds begin
        out[1] = R3D.Plane{3, T}(R3D.Vec{3, T}(T( 1), T(0), T(0)), -lo[1])
        out[2] = R3D.Plane{3, T}(R3D.Vec{3, T}(T(-1), T(0), T(0)),  hi[1])
        out[3] = R3D.Plane{3, T}(R3D.Vec{3, T}(T(0),  T( 1), T(0)), -lo[2])
        out[4] = R3D.Plane{3, T}(R3D.Vec{3, T}(T(0),  T(-1), T(0)),  hi[2])
        out[5] = R3D.Plane{3, T}(R3D.Vec{3, T}(T(0),  T(0),  T( 1)), -lo[3])
        out[6] = R3D.Plane{3, T}(R3D.Vec{3, T}(T(0),  T(0),  T(-1)),  hi[3])
    end
    return out
end

@inline function _fill_box_planes_int!(out::AbstractVector{R3D.Plane{4, T}},
                                        lo::NTuple{4, T},
                                        hi::NTuple{4, T}) where {T<:Signed}
    @inbounds begin
        out[1] = R3D.Plane{4, T}(R3D.Vec{4, T}(T( 1), T(0), T(0), T(0)), -lo[1])
        out[2] = R3D.Plane{4, T}(R3D.Vec{4, T}(T(-1), T(0), T(0), T(0)),  hi[1])
        out[3] = R3D.Plane{4, T}(R3D.Vec{4, T}(T(0),  T( 1), T(0), T(0)), -lo[2])
        out[4] = R3D.Plane{4, T}(R3D.Vec{4, T}(T(0),  T(-1), T(0), T(0)),  hi[2])
        out[5] = R3D.Plane{4, T}(R3D.Vec{4, T}(T(0),  T(0),  T( 1), T(0)), -lo[3])
        out[6] = R3D.Plane{4, T}(R3D.Vec{4, T}(T(0),  T(0),  T(-1), T(0)),  hi[3])
        out[7] = R3D.Plane{4, T}(R3D.Vec{4, T}(T(0),  T(0),  T(0),  T( 1)), -lo[4])
        out[8] = R3D.Plane{4, T}(R3D.Vec{4, T}(T(0),  T(0),  T(0),  T(-1)),  hi[4])
    end
    return out
end

# ============================================================================
# Public adapter entry point
# ============================================================================

"""
    overlap_simplex_box_exact!(out_moments::AbstractVector{Rational{R}},
                                scratch::IntPairScratch{D, T},
                                simplex_vertices,
                                box_lo::NTuple{D, T},
                                box_hi::NTuple{D, T},
                                moment_order::Integer;
                                accumulator::Type{<:Signed} =
                                    _default_accumulator(T, Val(D)))
        -> (volume::Rational{R},
            centroid::NTuple{D, Rational{R}},
            out_moments)

Compute the overlap polytope of an integer-coordinate D-simplex (with
`D + 1` vertices) against an integer-coordinate axis-aligned box,
returning its volume, centroid, and full moment vector up to
`moment_order` in graded-lex order — all as `Rational{R}` for an
accumulator type `R`.

`simplex_vertices` is a `D + 1`-tuple of `NTuple{D, T}` integer
coordinates. `box_lo` and `box_hi` are integer-coordinate corners.

The `accumulator` keyword controls the integer width used by
`R3D.IntExact.moments_exact!`. The default
(`_default_accumulator(T, Val(D))`) follows IntExact's documented
type-promotion guidance; pass a tighter type (e.g. `Int64`) if you
know the clip count is bounded, or `BigInt` for guaranteed safety.

`out_moments` must be of type `AbstractVector{Rational{R}}` matching
the `accumulator` parameter, with length
`moments_length(D, moment_order)`. The vector is overwritten on every
call (filled with zeros for empty overlaps).

Returns the empty triple `(0 // 1, ntuple(_ -> 0 // 1, Val(D)),
out_moments)` when the simplex is outside the box, fully clipped
away, or numerically degenerate (signed area / volume ≤ 0 — same
convention as the float path's `vol > 0` guard).

D = 2, D = 3, and D = 4 are supported with full polynomial moments via
`R3D.IntExact.moments_exact!` (D=4 polynomial-moments support landed
upstream in r3djl commit `943135f1`, dispatched through
`_moments_exact_dgeneric_4plus!`). Higher D (D = 5, D = 6) is
research-grade upstream and intentionally not exposed here.

# Centroid convention

The returned centroid is `Rational{R}`, NOT `Float64`. The PR-3
backend integration dequantizes at the boundary; keeping the
adapter's output exact preserves bit-exact reproducibility through
the per-pair computation.
"""
function overlap_simplex_box_exact!(
        out_moments::AbstractVector{Rational{R}},
        scratch::IntPairScratch{D, T},
        simplex_vertices,
        box_lo::NTuple{D, T},
        box_hi::NTuple{D, T},
        moment_order::Integer;
        accumulator::Type{<:Signed} = _default_accumulator(T, Val(D))
    ) where {D, T<:Signed, R<:Signed}
    accumulator === R ||
        throw(ArgumentError(
            "overlap_simplex_box_exact!: accumulator type ($accumulator) " *
            "must match the element type of out_moments (Rational{$R}). " *
            "Allocate `out_moments` with the same accumulator you pass " *
            "as the kwarg, or call with `accumulator = R`."))
    expected_len = moments_length(D, Int(moment_order))
    length(out_moments) == expected_len ||
        throw(DimensionMismatch(
            "out_moments has length $(length(out_moments)), expected $expected_len"))
    fill!(out_moments, zero(Rational{R}))
    vol, cent, moms, _kind = _overlap_dispatch_int!(out_moments, scratch,
                                                     simplex_vertices,
                                                     box_lo, box_hi,
                                                     Int(moment_order), Val(D))
    return (vol, cent, moms)
end

"""
    overlap_simplex_box_exact_with_drop_kind!(out_moments, scratch,
                                                simplex_vertices,
                                                box_lo, box_hi,
                                                moment_order;
                                                accumulator = ...)
        -> (volume, centroid, out_moments, drop_kind::Symbol)

Internal-API variant of [`overlap_simplex_box_exact!`](@ref) that ALSO
returns a `drop_kind ∈ (:none, :empty, :negative_volume)` indicating
whether the per-pair clip was empty (no overlap), produced a negative
or zero-area polytope (a known upstream bug at D = 2 — the adapter
zeros out_moments and reports the drop), or completed normally.

`drop_kind = :none` means a non-empty positively-oriented overlap was
produced. `:empty` means the simplex is outside the box / fully
clipped / degenerate (not a bug — counted for completeness). The
`:negative_volume` kind is the actual data-loss case: r3djl's
`R3D.IntExact.moments_exact!` returned a non-positive volume for a
non-degenerate clip polytope, which the adapter treats as empty
(volume == 0) on the outward-facing path. Used by
[`compute_overlap`](@ref)'s `audit_drops = true` mode and the
`OverlapDropReport` it returns.

This function does NOT surface `:moments_throw` drops (where
`moments_exact!` itself raises an exception on a degenerate
polytope). Those are caught in the caller (`compute_overlap`) via a
try/catch, since they manifest as a Julia exception rather than a
post-call inspection.
"""
function overlap_simplex_box_exact_with_drop_kind!(
        out_moments::AbstractVector{Rational{R}},
        scratch::IntPairScratch{D, T},
        simplex_vertices,
        box_lo::NTuple{D, T},
        box_hi::NTuple{D, T},
        moment_order::Integer;
        accumulator::Type{<:Signed} = _default_accumulator(T, Val(D))
    ) where {D, T<:Signed, R<:Signed}
    accumulator === R ||
        throw(ArgumentError(
            "overlap_simplex_box_exact_with_drop_kind!: accumulator type " *
            "($accumulator) must match the element type of out_moments " *
            "(Rational{$R})."))
    expected_len = moments_length(D, Int(moment_order))
    length(out_moments) == expected_len ||
        throw(DimensionMismatch(
            "out_moments has length $(length(out_moments)), expected $expected_len"))
    fill!(out_moments, zero(Rational{R}))
    return _overlap_dispatch_int!(out_moments, scratch, simplex_vertices,
                                   box_lo, box_hi, Int(moment_order), Val(D))
end

# ============================================================================
# Per-D dispatch
# ============================================================================

# Generic catch-all: D ∉ {2, 3, 4} not supported. Upstream R3D.IntExact
# has experimental D=5/D=6 paths but HG does not surface them — the
# downstream geometry / lattice / quantization stack is targeted at
# D ∈ {1, 2, 3, 4}.
function _overlap_dispatch_int!(out_moments::AbstractVector{Rational{R}},
                                  scratch::IntPairScratch{D, T},
                                  simplex_vertices,
                                  box_lo::NTuple{D, T},
                                  box_hi::NTuple{D, T},
                                  moment_order::Int,
                                  ::Val{D}) where {D, T<:Signed, R<:Signed}
    throw(ErrorException(
        "overlap_simplex_box_exact! at D=$D not supported. " *
        "Currently supported: D=2, D=3, D=4 (full polynomial moments). " *
        "Higher D (D=5, D=6) is research-grade upstream and not surfaced " *
        "in HierarchicalGrids. See src/Overlap/r3d_int_adapter.jl for the " *
        "dispatch table."))
end

# Return shape for the per-D dispatch tuple: every leaf returns the
# 4-tuple `(volume::Rational{R}, centroid::NTuple{D, Rational{R}},
# out_moments, drop_kind::Symbol)`. The `drop_kind` is one of:
#   :none             — non-empty positive overlap.
#   :empty            — simplex outside / fully clipped / degenerate.
#   :negative_volume  — r3djl returned a non-positive volume for a
#                       non-degenerate clip polygon (upstream bug at
#                       D = 2). Adapter zeros the moments and treats
#                       as empty externally.

# ----------------------------------------------------------------------------
# D = 2
# ----------------------------------------------------------------------------

function _overlap_dispatch_int!(out_moments::AbstractVector{Rational{R}},
                                  scratch::IntPairScratch{2, T},
                                  simplex_vertices,
                                  box_lo::NTuple{2, T},
                                  box_hi::NTuple{2, T},
                                  moment_order::Int,
                                  ::Val{2}) where {T<:Signed, R<:Signed}
    # Quick AABB reject: degenerate / inverted box ⇒ empty.
    if box_lo[1] >= box_hi[1] || box_lo[2] >= box_hi[2]
        return _empty_result(out_moments, Val(2), R, :empty)
    end
    poly = scratch.poly
    planes = scratch.plane_buf

    # CCW-orient the triangle: `R3D.IntExact.init_simplex!` D=2 demands
    # a CCW boundary so the shoelace area is positive. If the user
    # supplied a CW triangle, we transparently swap the last two
    # vertices to CCW-orient — same convention as the float path's
    # `_kuhn_unit_cube_offset` helper, which post-flips negative-volume
    # tetrahedra so r3djl sees positively oriented simplices.
    v1 = simplex_vertices[1]
    v2 = simplex_vertices[2]
    v3 = simplex_vertices[3]
    # 2 × signed area (integer):
    twoa = R(v2[1] - v1[1]) * R(v3[2] - v1[2]) -
           R(v2[2] - v1[2]) * R(v3[1] - v1[1])
    if twoa < zero(R)
        v2, v3 = v3, v2
    elseif twoa == zero(R)
        # Degenerate triangle (collinear) — overlap is measure zero.
        return _empty_result(out_moments, Val(2), R, :empty)
    end

    # Stack-allocated 2-element vectors per vertex (init_simplex! takes
    # AbstractVector). We use length-2 SVectors for zero allocation.
    vv1 = SVector{2, T}(v1[1], v1[2])
    vv2 = SVector{2, T}(v2[1], v2[2])
    vv3 = SVector{2, T}(v3[1], v3[2])
    R3D.IntExact.init_simplex!(poly, vv1, vv2, vv3)

    _fill_box_planes_int!(planes, box_lo, box_hi)

    ok = R3D.IntExact.clip!(poly, planes)
    if !ok
        # Capacity overflow — surface loudly. Recovery requires a larger
        # `IntPairScratch` capacity (default 64 ⇒ comfortable for
        # axis-aligned clips of a triangle, but defensive).
        throw(OverflowError(
            "R3D.IntExact.clip! reported capacity overflow during 2D " *
            "simplex-box clipping (scratch.poly.capacity = $(poly.capacity)). " *
            "Allocate IntPairScratch with a larger capacity if this is " *
            "expected geometry."))
    end
    if poly.nverts == 0
        return _empty_result(out_moments, Val(2), R, :empty)
    end

    R3D.IntExact.moments_exact!(out_moments, poly, moment_order)
    vol = out_moments[1]
    if vol <= zero(Rational{R})
        # Numerically degenerate (zero-area sliver) OR upstream IntExact
        # produced a NEGATIVE-volume polytope (a known D = 2 bug —
        # orientation flip after clip!). The adapter zeros out_moments
        # and treats the pair as empty externally; we surface the
        # distinction via `drop_kind = :negative_volume` for the audit
        # path. `vol == 0` with a non-empty poly is also classed as
        # `:negative_volume` since it indicates upstream returned a
        # numerically-zero result for a polygon that geometrically
        # had positive area (the `poly.nverts == 0` empty case is
        # handled above).
        fill!(out_moments, zero(Rational{R}))
        return _empty_result(out_moments, Val(2), R, :negative_volume)
    end

    centroid = if moment_order >= 1
        (out_moments[2] // vol, out_moments[3] // vol)
    else
        (zero(Rational{R}), zero(Rational{R}))
    end
    return (vol, centroid, out_moments, :none)
end

# ----------------------------------------------------------------------------
# D = 3
# ----------------------------------------------------------------------------

function _overlap_dispatch_int!(out_moments::AbstractVector{Rational{R}},
                                  scratch::IntPairScratch{3, T},
                                  simplex_vertices,
                                  box_lo::NTuple{3, T},
                                  box_hi::NTuple{3, T},
                                  moment_order::Int,
                                  ::Val{3}) where {T<:Signed, R<:Signed}
    if box_lo[1] >= box_hi[1] || box_lo[2] >= box_hi[2] || box_lo[3] >= box_hi[3]
        return _empty_result(out_moments, Val(3), R, :empty)
    end
    poly = scratch.poly
    planes = scratch.plane_buf

    v1 = simplex_vertices[1]
    v2 = simplex_vertices[2]
    v3 = simplex_vertices[3]
    v4 = simplex_vertices[4]
    # 6 × signed volume (integer): det of the 3×3 matrix of (vk - v1).
    a1 = R(v2[1] - v1[1]); a2 = R(v2[2] - v1[2]); a3 = R(v2[3] - v1[3])
    b1 = R(v3[1] - v1[1]); b2 = R(v3[2] - v1[2]); b3 = R(v3[3] - v1[3])
    c1 = R(v4[1] - v1[1]); c2 = R(v4[2] - v1[2]); c3 = R(v4[3] - v1[3])
    sixv = a1 * (b2 * c3 - b3 * c2) -
           a2 * (b1 * c3 - b3 * c1) +
           a3 * (b1 * c2 - b2 * c1)
    if sixv < zero(R)
        v3, v4 = v4, v3
    elseif sixv == zero(R)
        return _empty_result(out_moments, Val(3), R, :empty)
    end

    # `init_tet!` D=3 takes an `NTuple{4, <:AbstractVector}`.
    vv1 = SVector{3, T}(v1[1], v1[2], v1[3])
    vv2 = SVector{3, T}(v2[1], v2[2], v2[3])
    vv3 = SVector{3, T}(v3[1], v3[2], v3[3])
    vv4 = SVector{3, T}(v4[1], v4[2], v4[3])
    R3D.IntExact.init_tet!(poly, (vv1, vv2, vv3, vv4))

    _fill_box_planes_int!(planes, box_lo, box_hi)

    ok = R3D.IntExact.clip!(poly, planes)
    if !ok
        throw(OverflowError(
            "R3D.IntExact.clip! reported capacity overflow during 3D " *
            "simplex-box clipping (scratch.poly.capacity = $(poly.capacity)). " *
            "Allocate IntPairScratch with a larger capacity if this is " *
            "expected geometry."))
    end
    if poly.nverts == 0
        return _empty_result(out_moments, Val(3), R, :empty)
    end

    R3D.IntExact.moments_exact!(out_moments, poly, moment_order)
    vol = out_moments[1]
    if vol <= zero(Rational{R})
        fill!(out_moments, zero(Rational{R}))
        return _empty_result(out_moments, Val(3), R, :negative_volume)
    end

    centroid = if moment_order >= 1
        (out_moments[2] // vol, out_moments[3] // vol, out_moments[4] // vol)
    else
        (zero(Rational{R}), zero(Rational{R}), zero(Rational{R}))
    end
    return (vol, centroid, out_moments, :none)
end

# ----------------------------------------------------------------------------
# D = 4
#
# As of r3djl commit 943135f1 (`IntExact polynomial moments at D ∈ {4, 5,
# 6} via simplex decomposition`), `R3D.IntExact.moments_exact!` ships
# full polynomial moments for D = 4. The dispatch in `intexact.jl` routes
# `D == 4` to `_moments_exact_dgeneric_4plus!`, so this method now mirrors
# the D=2 / D=3 paths exactly: clip the pentachoron against the 8
# axis-aligned box planes, then call `moments_exact!` to fill `out_moments`.
#
# Vertex orientation (positive 4-volume): the 5! / 2 = 60 even
# permutations on 5 vertices give a positive `det(V)`, the 60 odd ones
# negative. We compute the signed 4-volume (det of the 4x4 matrix of
# `vk - v1`, k = 2..5) and swap two trailing vertices to flip parity if
# negative — same convention as the D = 2, 3 paths.
# ----------------------------------------------------------------------------

# Signed 24·volume of the integer pentachoron (v1, v2, v3, v4, v5) via the
# 4×4 determinant of (vk - v1) columns. Returned as `R` to keep the
# accumulator clean; for `T = Int16` and modest coords the intermediate
# expansion fits in `Int128` after a few clips, but we promote to `R`
# (the moment-buffer accumulator) to match the D = 4 default `BigInt`.
@inline function _signed_24vol_d4(::Type{R}, v1, v2, v3, v4, v5) where {R<:Signed}
    a1 = R(v2[1] - v1[1]); a2 = R(v2[2] - v1[2]); a3 = R(v2[3] - v1[3]); a4 = R(v2[4] - v1[4])
    b1 = R(v3[1] - v1[1]); b2 = R(v3[2] - v1[2]); b3 = R(v3[3] - v1[3]); b4 = R(v3[4] - v1[4])
    c1 = R(v4[1] - v1[1]); c2 = R(v4[2] - v1[2]); c3 = R(v4[3] - v1[3]); c4 = R(v4[4] - v1[4])
    d1 = R(v5[1] - v1[1]); d2 = R(v5[2] - v1[2]); d3 = R(v5[3] - v1[3]); d4 = R(v5[4] - v1[4])
    # 4×4 determinant by cofactor expansion along the first row.
    # M_ij = 3×3 minor with row 1 and column j removed.
    m11 = b2*(c3*d4 - c4*d3) - b3*(c2*d4 - c4*d2) + b4*(c2*d3 - c3*d2)
    m12 = b1*(c3*d4 - c4*d3) - b3*(c1*d4 - c4*d1) + b4*(c1*d3 - c3*d1)
    m13 = b1*(c2*d4 - c4*d2) - b2*(c1*d4 - c4*d1) + b4*(c1*d2 - c2*d1)
    m14 = b1*(c2*d3 - c3*d2) - b2*(c1*d3 - c3*d1) + b3*(c1*d2 - c2*d1)
    return a1*m11 - a2*m12 + a3*m13 - a4*m14
end

function _overlap_dispatch_int!(out_moments::AbstractVector{Rational{R}},
                                  scratch::IntPairScratch{4, T},
                                  simplex_vertices,
                                  box_lo::NTuple{4, T},
                                  box_hi::NTuple{4, T},
                                  moment_order::Int,
                                  ::Val{4}) where {T<:Signed, R<:Signed}
    if box_lo[1] >= box_hi[1] || box_lo[2] >= box_hi[2] ||
       box_lo[3] >= box_hi[3] || box_lo[4] >= box_hi[4]
        return _empty_result(out_moments, Val(4), R, :empty)
    end

    poly = scratch.poly
    planes = scratch.plane_buf

    v1 = simplex_vertices[1]
    v2 = simplex_vertices[2]
    v3 = simplex_vertices[3]
    v4 = simplex_vertices[4]
    v5 = simplex_vertices[5]
    sv = _signed_24vol_d4(R, v1, v2, v3, v4, v5)
    if sv < zero(R)
        v4, v5 = v5, v4
    elseif sv == zero(R)
        # Degenerate pentachoron (coplanar in 4D) — measure zero.
        return _empty_result(out_moments, Val(4), R, :empty)
    end

    # `init_simplex!` D=4 takes a length-5 indexable collection of
    # length-D indexable vertex positions. A 5-tuple of `SVector{4, T}`
    # is the zero-allocation form (stack-allocated, supports indexing).
    vv1 = SVector{4, T}(v1[1], v1[2], v1[3], v1[4])
    vv2 = SVector{4, T}(v2[1], v2[2], v2[3], v2[4])
    vv3 = SVector{4, T}(v3[1], v3[2], v3[3], v3[4])
    vv4 = SVector{4, T}(v4[1], v4[2], v4[3], v4[4])
    vv5 = SVector{4, T}(v5[1], v5[2], v5[3], v5[4])
    R3D.IntExact.init_simplex!(poly, (vv1, vv2, vv3, vv4, vv5))

    _fill_box_planes_int!(planes, box_lo, box_hi)

    ok = R3D.IntExact.clip!(poly, planes)
    if !ok
        throw(OverflowError(
            "R3D.IntExact.clip! reported capacity overflow during 4D " *
            "pentachoron-box clipping (scratch.poly.capacity = $(poly.capacity)). " *
            "Allocate IntPairScratch with a larger capacity if this is " *
            "expected geometry."))
    end
    if poly.nverts == 0
        return _empty_result(out_moments, Val(4), R, :empty)
    end

    R3D.IntExact.moments_exact!(out_moments, poly, moment_order)
    vol = out_moments[1]
    if vol <= zero(Rational{R})
        fill!(out_moments, zero(Rational{R}))
        return _empty_result(out_moments, Val(4), R, :negative_volume)
    end

    centroid = if moment_order >= 1
        (out_moments[2] // vol, out_moments[3] // vol,
         out_moments[4] // vol, out_moments[5] // vol)
    else
        (zero(Rational{R}), zero(Rational{R}),
         zero(Rational{R}), zero(Rational{R}))
    end
    return (vol, centroid, out_moments, :none)
end

# ----------------------------------------------------------------------------
# Empty-result helper
# ----------------------------------------------------------------------------

@inline function _empty_result(out_moments::AbstractVector{Rational{R}},
                                ::Val{D}, ::Type{R},
                                kind::Symbol = :empty) where {D, R<:Signed}
    return (zero(Rational{R}),
            ntuple(_ -> zero(Rational{R}), Val(D)),
            out_moments,
            kind)
end

# ============================================================================
# Module-load convention check
# ============================================================================

# Verify that `R3D.IntExact`'s plane convention matches `{x : n · x + d ≥ 0}`
# is kept (same convention as `R3D.Flat`, used by the rest of the overlap
# layer). We construct a known polytope (axis-aligned integer triangle),
# clip it with a single plane that retains exactly half the area, and
# assert the exact-rational area matches the analytic answer. If r3djl
# ever flips the convention this load-time check fails loudly rather
# than silently mis-clipping every overlap pair.
#
# The known polytope: triangle (0,0)-(2,0)-(0,2), area 2. Plane n=(-1,0),
# d=1 keeps the half-space `-x + 1 ≥ 0` ⇔ `x ≤ 1`. The kept region is
# the trapezoid (0,0)-(1,0)-(1,1)-(0,2), area `1·1 + (1·1)/2·1 = ...`
# Actually let's use a simpler bisection: plane n=(1,0), d=-1 keeps
# `x - 1 ≥ 0` ⇔ `x ≥ 1`. The kept region is the small triangle
# (1,0)-(2,0)-(0,2)∩{x≥1} which is (1,0)-(2,0)-(1,1), area = 1/2.
function _verify_intexact_plane_convention()
    poly = R3D.IntExact.IntFlatPolytope{2, Int64}(32)
    R3D.IntExact.init_simplex!(poly, [0, 0], [2, 0], [0, 2])
    # `x ≥ 1` plane in integer encoding: n = (1, 0), d = -1.
    plane = R3D.Plane{2, Int64}(R3D.Vec{2, Int64}(1, 0), -1)
    R3D.IntExact.clip!(poly, plane)
    a = R3D.IntExact.area_exact(poly, Int128)
    expected = 1 // 2
    if a != expected
        error("R3D.IntExact plane convention check failed: clipping " *
              "triangle (0,0)-(2,0)-(0,2) with plane {x : x - 1 ≥ 0} " *
              "should keep area $(expected), got $(a). " *
              "This means the IntExact half-space convention has changed " *
              "from `n·x + d ≥ 0`-kept and the integer adapter needs to " *
              "be updated to match.")
    end

    # Also verify moment ordering matches the float path: integrate
    # moments through order 2 of the canonical unit triangle scaled by
    # an integer factor and assert the exact rationals match the float
    # convention's analytic values (already cross-checked by the float
    # adapter's `_verify_moment_ordering`). Triangle (0,0)-(s,0)-(0,s)
    # has moments (in graded-lex (0,0),(1,0),(0,1),(2,0),(1,1),(0,2)):
    #   M(0,0) = s² / 2
    #   M(1,0) = s³ / 6
    #   M(0,1) = s³ / 6
    #   M(2,0) = s⁴ / 12
    #   M(1,1) = s⁴ / 24
    #   M(0,2) = s⁴ / 12
    let s = 6
        p2 = R3D.IntExact.IntFlatPolytope{2, Int64}(32)
        R3D.IntExact.init_simplex!(p2, [0, 0], [s, 0], [0, s])
        m = R3D.IntExact.moments_exact(p2, 2; R = Int128)
        exp = Rational{Int128}[s^2 // 2, s^3 // 6, s^3 // 6,
                                s^4 // 12, s^4 // 24, s^4 // 12]
        for k in eachindex(exp)
            if m[k] != exp[k]
                error("R3D.IntExact moment-ordering check failed at index $k: " *
                      "got $(m[k]), expected $(exp[k]). " *
                      "IntExact's moment ordering may have diverged from " *
                      "the float path; the integer adapter needs an update.")
            end
        end
    end
    return true
end

# Run once at module-load time.
_verify_intexact_plane_convention()
