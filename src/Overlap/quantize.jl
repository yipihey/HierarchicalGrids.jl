"""
Float-vertex → integer-lattice quantization helpers for the IntExact
exact-rational overlap path.

Users continue to construct `SimplicialMesh{D, Float64}` and
`EulerianFrame{D, Float64}` as today. When the exact-rational backend
is requested (PR-3), an `IntegerLattice` quantizes both the leaf-box
corners and the simplex vertices onto a common integer grid before
calling `overlap_simplex_box_exact!`. This file is the only place in
the Overlap layer that knows how float coordinates relate to the
lattice; PR-3 will compose the helpers below at the per-pair boundary.

# Lattice convention

- **Equal scale across all axes**: avoids subtle moment rescaling
  under anisotropic lattices. `scale = (2^bits - 1) /
  max_d (hi[d] - lo[d])` is computed from the frame's longest physical
  axis and applied uniformly to every axis.
- **Origin at `lat.lo`**: integer coordinate `p` along axis `d`
  represents the physical point `lat.lo[d] + p / lat.scale`.
- **Round-to-nearest**: `quantize` uses `round(Int)` rounding (default
  RoundNearest, ties to even). Round-trip error is bounded by
  `lat_resolution(lat) / 2` per axis.
- **Box-frame moments**: `R3D.IntExact.moments_exact!` integrates over
  the polytope in its OWN coordinate frame — the lattice-INTEGER frame.
  The 0th moment (volume / area / length) scales by `(1/scale)^D` only;
  higher-degree moments require both a scale factor AND an offset shift
  (because lattice-frame `x` is offset from physical `x` by `lat.lo`).
  This file ships `unscale_volume` for the 0th-moment case (the only
  one without offset bookkeeping); higher moments must be composed
  with `shift_moments!` from `moments.jl` by callers (PR-3).
"""

# ============================================================================
# Type
# ============================================================================

"""
    IntegerLattice{D, T<:Signed}

Equal-scale lattice over physical box `[lo, hi]` with `2^bits - 1`
integer steps along the longest physical axis. Used to quantize float
vertices for exact-rational overlap computation.

# Fields

- `lo::NTuple{D, Float64}` — physical lower corner (lattice origin).
- `hi::NTuple{D, Float64}` — physical upper corner.
- `bits::Int` — bit budget; the longest physical axis spans
  `2^bits - 1` integer steps.
- `scale::Float64` — `(2^bits - 1) / maximum(hi[d] - lo[d])`. Multiply a
  physical offset (relative to `lo`) by `scale` to get the lattice-integer
  coordinate.
- `int_type::Type{T}` — integer type used by `quantize` outputs.

The lattice is constructed via `IntegerLattice(frame; bits, int_type)`
or `IntegerLattice(lo, hi; bits, int_type)`. Both constructors enforce
that `2^bits - 1` fits in `int_type` (otherwise an `ArgumentError` is
thrown at construction).
"""
struct IntegerLattice{D, T<:Signed}
    lo::NTuple{D, Float64}
    hi::NTuple{D, Float64}
    bits::Int
    scale::Float64
    int_type::Type{T}
end

# ============================================================================
# Constructors
# ============================================================================

"""
    IntegerLattice(frame::EulerianFrame{D, Float64};
                   bits::Int = 16,
                   int_type::Type{<:Signed} = Int32) -> IntegerLattice{D, int_type}

Derive an `IntegerLattice` from a `EulerianFrame`'s physical bounds.
Convenience wrapper around `IntegerLattice(frame.lo, frame.hi; ...)`.
"""
function IntegerLattice(frame::EulerianFrame{D, Float64};
                        bits::Int = 16,
                        int_type::Type{<:Signed} = Int32) where {D}
    return IntegerLattice(frame.lo, frame.hi; bits = bits, int_type = int_type)
end

"""
    IntegerLattice(lo::NTuple{D, Float64}, hi::NTuple{D, Float64};
                   bits::Int = 16,
                   int_type::Type{<:Signed} = Int32) -> IntegerLattice{D, int_type}

Construct an `IntegerLattice` directly from physical bounds.

Validation:
- Each axis must satisfy `hi[d] > lo[d]`.
- `bits ≥ 1`.
- `2^bits - 1` must fit in `int_type` (the lattice spans this many
  integer steps along its longest axis; the `quantize` output range
  must therefore be representable in `int_type`).
"""
function IntegerLattice(lo::NTuple{D, Float64}, hi::NTuple{D, Float64};
                        bits::Int = 16,
                        int_type::Type{<:Signed} = Int32) where {D}
    bits >= 1 || throw(ArgumentError(
        "IntegerLattice: bits must be ≥ 1 (got bits=$bits)"))
    for d in 1:D
        hi[d] > lo[d] || throw(ArgumentError(
            "IntegerLattice: hi[$d] ($(hi[d])) must exceed lo[$d] ($(lo[d]))"))
    end
    # Overflow guard: (2^bits - 1) must fit in int_type with a sign bit.
    # `bits == 8 * sizeof(int_type)` would land on the sign bit (typemax + 1
    # ≡ 2^bits, so 2^bits - 1 == typemax). We require strict fit:
    # 2^bits - 1 ≤ typemax(int_type).
    if int_type !== BigInt
        max_steps_widened = (BigInt(1) << bits) - BigInt(1)
        tmax = BigInt(typemax(int_type))
        if max_steps_widened > tmax
            throw(ArgumentError(
                "IntegerLattice: bits=$bits requires $(max_steps_widened) " *
                "lattice steps, which overflows int_type=$(int_type) " *
                "(typemax = $(typemax(int_type))). Use a wider integer type " *
                "(e.g. Int64 / Int128 / BigInt) or fewer bits."))
        end
    end
    # Compute scale from the longest physical axis. Equal scale across all
    # axes (anisotropic lattices would complicate moment rescaling).
    longest = zero(Float64)
    for d in 1:D
        ext = hi[d] - lo[d]
        ext > longest && (longest = ext)
    end
    # `longest > 0` is guaranteed by the per-axis hi > lo check above.
    max_steps = Float64((BigInt(1) << bits) - BigInt(1))
    scale = max_steps / longest
    return IntegerLattice{D, int_type}(lo, hi, bits, scale, int_type)
end

# ============================================================================
# Accessors
# ============================================================================

"""
    lat_resolution(lat::IntegerLattice) -> Float64

Physical distance between adjacent lattice points along ANY axis (the
lattice is equal-scale): `1 / lat.scale`.

Round-trip error from `dequantize ∘ quantize` is bounded by
`lat_resolution(lat) / 2` per axis (round-to-nearest convention).
"""
@inline lat_resolution(lat::IntegerLattice) = 1.0 / lat.scale

# ============================================================================
# quantize / dequantize
# ============================================================================

"""
    quantize(v::NTuple{D, Float64}, lat::IntegerLattice{D, T}) -> NTuple{D, T}

Round each component of `v` to its nearest integer on the lattice.
Coordinates are taken relative to `lat.lo` and scaled by `lat.scale`,
then rounded to `T` (round-to-nearest, ties to even).

Out-of-range values are clamped to the lattice's representable
integer range `[typemin(T), typemax(T)]` so the caller never sees an
`InexactError`. Callers that want to detect off-lattice / out-of-range
inputs should use `quantize_strict` instead.

Round-trip error is bounded: `|dequantize(quantize(v, lat), lat) - v|
≤ lat_resolution(lat) / 2` per axis (for inputs inside the lattice
range).
"""
@inline function quantize(v::NTuple{D, Float64},
                          lat::IntegerLattice{D, T}) where {D, T}
    return ntuple(Val(D)) do d
        x = (v[d] - lat.lo[d]) * lat.scale
        # Clamp before rounding to avoid InexactError for very-out-of-range
        # values. Using floats for the clamp bounds keeps the code path
        # branch-light; the final round/convert lands in T cleanly.
        if T === BigInt
            return T(round(x))
        else
            tmin = Float64(typemin(T))
            tmax = Float64(typemax(T))
            xc = x < tmin ? tmin : (x > tmax ? tmax : x)
            return round(T, xc)
        end
    end
end

"""
    quantize_strict(v::NTuple{D, Float64}, lat::IntegerLattice{D, T};
                    atol::Float64 = lat_resolution(lat) / 2) -> NTuple{D, T}

Like `quantize`, but throws `ArgumentError` if the snap distance
(absolute Euclidean offset between `v[d]` and the dequantized integer
coordinate) exceeds `atol` for any axis. Use when `v` is expected to
already lie on the lattice (e.g. synthesized from integer geometry).

The default tolerance equals the maximum rounding error of plain
`quantize`; tighten `atol` to assert exact lattice alignment.
"""
function quantize_strict(v::NTuple{D, Float64},
                         lat::IntegerLattice{D, T};
                         atol::Float64 = lat_resolution(lat) / 2) where {D, T}
    p = quantize(v, lat)
    back = dequantize(p, lat)
    @inbounds for d in 1:D
        delta = abs(back[d] - v[d])
        if delta > atol
            throw(ArgumentError(
                "quantize_strict: axis $d snap distance $(delta) " *
                "exceeds atol=$(atol) (lattice resolution = " *
                "$(lat_resolution(lat))). Vertex $v is not on the " *
                "lattice within the requested tolerance."))
        end
    end
    return p
end

"""
    dequantize(p::NTuple{D, T}, lat::IntegerLattice{D, T}) -> NTuple{D, Float64}

Inverse of `quantize`. Maps lattice-integer coordinates back to
physical Float64 coordinates: `lat.lo[d] + p[d] / lat.scale`.

`dequantize ∘ quantize` is identity up to `lat_resolution(lat) / 2`
per axis (for in-range inputs).
"""
@inline function dequantize(p::NTuple{D, T},
                            lat::IntegerLattice{D, T}) where {D, T}
    inv_scale = 1.0 / lat.scale
    return ntuple(Val(D)) do d
        lat.lo[d] + Float64(p[d]) * inv_scale
    end
end

# ============================================================================
# Moment unscaling
# ============================================================================

"""
    unscale_volume(m::Rational{R}, lat::IntegerLattice{D}) -> Float64

Convert a 0th moment (D-dimensional volume / area / length) from
lattice-integer units back to physical units. The 0th moment is the
ONLY moment whose unscaling needs no offset bookkeeping — it is
translation-invariant — so this helper is exact and self-contained.

The conversion is `Float64(m) * (1 / scale)^D`.

# Notes

- For D=1, returns physical length.
- For D=2, returns physical area.
- For D=3, returns physical volume.
- For higher moments (degree ≥ 1), use `unscale_moment` (lattice-frame
  output) and compose with `shift_moments!` to handle the lattice
  origin offset. See `unscale_moment` for the convention.
"""
@inline function unscale_volume(m::Rational{R},
                                lat::IntegerLattice{D}) where {R, D}
    inv_scale = 1.0 / lat.scale
    factor = inv_scale ^ D
    return Float64(m) * factor
end

"""
    unscale_moment(m::Rational{R}, lat::IntegerLattice{D},
                   monomial_total_degree::Int) -> Float64

Convert a moment of total monomial degree `monomial_total_degree`
from lattice-integer units back to *lattice-frame* physical units.
A degree-`k` monomial integrated over a D-dimensional region scales
by `(1/scale)^(D + k)` under the inverse-scale map.

# Important: lattice-frame, NOT physical-frame for k ≥ 1

For `monomial_total_degree == 0` (volume) the returned value is
the physical volume — translation-invariant. This case is also
exposed as `unscale_volume(m, lat)`.

For `monomial_total_degree ≥ 1`, the returned value is the moment
in the **lattice's coordinate frame**, i.e. with origin at `lat.lo`.
To recover physical-frame moments the caller must additionally apply
the offset shift `Δ = lat.lo` via `shift_moments!` (see
`src/Overlap/moments.jl`). PR-3 composes these two steps in the
per-pair flow:

1. Quantize simplex + box vertices → integer coords.
2. Call `overlap_simplex_box_exact!` → `Rational` moments in
   lattice-integer frame.
3. Map each moment through `unscale_moment` with its multi-index's
   total degree → Float64 moments in lattice-frame physical units.
4. Apply `shift_moments!` with `Δ = lat.lo` to translate the moment
   vector into the physical (frame) coordinate system.

Keeping this helper purely scale-aware (no offset arithmetic) lets
callers reuse the existing `shift_moments!` machinery for the offset
step without double-counting.
"""
@inline function unscale_moment(m::Rational{R}, lat::IntegerLattice{D},
                                monomial_total_degree::Int) where {R, D}
    inv_scale = 1.0 / lat.scale
    factor = inv_scale ^ (D + monomial_total_degree)
    return Float64(m) * factor
end
