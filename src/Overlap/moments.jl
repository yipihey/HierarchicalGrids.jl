"""
Moment-vector helpers for the overlap layer.

A moment vector for a polytope `P` and order `O` in `D` dimensions is a
flat `Vector{T}` whose `k`th entry is the integral

    M_k = ∫_P x_1^a_1 · x_2^a_2 · ... · x_D^a_D dV

for the multi-index `(a_1, ..., a_D)` mapped to `k` by `moment_index`.

# Ordering convention (graded-lex)

Multi-indices are sorted first by total degree `|a| = a_1 + ... + a_D`,
then within each degree group by lexicographic order on `(a_1, a_2, ..., a_D)`
where `a_1` is most significant. Examples (D=2, O=2):

    k=1: (0, 0)         # deg 0; the volume
    k=2: (1, 0)
    k=3: (0, 1)         # deg 1
    k=4: (2, 0)
    k=5: (1, 1)
    k=6: (0, 2)         # deg 2

Total length = `binomial(D + O, D)`.

This matches the convention used by r3djl. The adapter performs an
explicit cross-check on first call to verify ordering agreement at
runtime; if it ever disagrees we'd see a clean error there rather than
silent corruption.
"""

"""
    moments_length(D::Integer, order::Integer) -> Int

Number of monomials of degree at most `order` in `D` variables, i.e. the
length of a moment vector. Equals `binomial(D + order, D)`.
"""
@inline moments_length(D::Integer, order::Integer) = binomial(D + order, D)

"""
    moment_multiindices(D::Integer, order::Integer) -> Vector{NTuple{D, Int}}

All multi-indices `(a_1, ..., a_D)` with `0 ≤ Σa_i ≤ order`, in graded-lex
order. Cached per (D, order) pair; safe to call repeatedly.

This is the canonical mapping `flat_index → multi_index`.
"""
function moment_multiindices end

const _MOMENT_INDEX_CACHE = Dict{Tuple{Int, Int}, Vector{Vector{Int}}}()

function moment_multiindices(D::Integer, order::Integer)
    key = (Int(D), Int(order))
    haskey(_MOMENT_INDEX_CACHE, key) || (_MOMENT_INDEX_CACHE[key] = _build_moment_multiindices(Int(D), Int(order)))
    return _MOMENT_INDEX_CACHE[key]
end

# Build the list of multi-indices for the given dimension and order.
# Returns a Vector{Vector{Int}}, each inner vector of length D.
function _build_moment_multiindices(D::Int, order::Int)
    D >= 1   || throw(ArgumentError("D must be ≥ 1"))
    order >= 0 || throw(ArgumentError("order must be ≥ 0"))
    out = Vector{Vector{Int}}()
    for total in 0:order
        # Enumerate compositions of `total` into D non-negative parts in
        # lex order with first index most significant. We use a recursive
        # generator implemented iteratively via a stack of partial tuples.
        _enumerate_compositions!(out, Int[], total, D)
    end
    return out
end

# Recursively build all D-element non-negative integer tuples summing to `total`,
# in lex order with first index most significant. Append each to `out`.
function _enumerate_compositions!(out::Vector{Vector{Int}}, prefix::Vector{Int},
                                    remaining::Int, slots_left::Int)
    if slots_left == 1
        push!(prefix, remaining)
        push!(out, copy(prefix))
        pop!(prefix)
        return
    end
    # First index goes from `remaining` down to 0 (lex order with first index
    # most significant ⇒ larger first values come earlier).
    for first in remaining:-1:0
        push!(prefix, first)
        _enumerate_compositions!(out, prefix, remaining - first, slots_left - 1)
        pop!(prefix)
    end
    return
end

"""
    moment_index(D::Integer, order::Integer, exponents) -> Int

Flat index of the multi-index `exponents` in the moment vector for the
given `D` and `order`. Throws `ArgumentError` if `exponents` is out of
range or has the wrong length.

`exponents` may be an `NTuple{D, Int}` or any indexable collection of
length `D` containing non-negative integers with `sum ≤ order`.
"""
function moment_index(D::Integer, order::Integer, exponents)
    length(exponents) == D ||
        throw(ArgumentError("exponents must have length $D, got $(length(exponents))"))
    s = 0
    for e in exponents
        e >= 0 || throw(ArgumentError("exponents must be non-negative; got $e"))
        s += e
    end
    s <= order || throw(ArgumentError("sum(exponents) = $s exceeds order $order"))
    multi = moment_multiindices(Int(D), Int(order))
    @inbounds for (k, m) in enumerate(multi)
        ok = true
        for d in 1:D
            if m[d] != exponents[d]
                ok = false; break
            end
        end
        ok && return k
    end
    error("moment_index: should be unreachable")
end

"""
    moment_volume(moments::AbstractVector) -> T

The 0th moment (volume / area), which is always the first entry.
"""
@inline moment_volume(moments::AbstractVector) = moments[1]

"""
    moment_centroid(moments::AbstractVector, D::Integer) -> NTuple{D, T}

Centroid extracted from a moment vector with order ≥ 1: returns
`(M_{e_d} / M_0)` for each axis `d`, where `e_d` is the unit basis vector.
Assumes the standard graded-lex ordering: indices `2..D+1` are the first
moments in axis order.
"""
function moment_centroid(moments::AbstractVector{T}, D::Integer) where T
    length(moments) >= D + 1 || error("moment vector too short for centroid in $(D)D")
    vol = moments[1]
    return ntuple(d -> moments[d + 1] / vol, Int(D))
end

# ============================================================================
# Frame shift via Pascal-triangle multinomial expansion
# ============================================================================

"""
    shift_moments!(out::AbstractVector{T}, src::AbstractVector{T},
                    D::Integer, order::Integer, Δ::NTuple{D, T}) where T

Translate a moment vector from one origin to another:

    M_new(a) = ∫_P (x - x_new)^a dV
            = Σ_{b ≤ a} binomial(a, b) (-Δ)^(a - b) M_old(b)

where `Δ = x_new - x_old` is the shift of the origin. `binomial(a, b)` is
the multinomial product `∏_d binomial(a_d, b_d)` and `(-Δ)^(a-b)` is the
componentwise product `∏_d (-Δ_d)^(a_d - b_d)`.

`out` and `src` must each have length `moments_length(D, order)`. They
may alias only if `out === src` is intentional and the loop ordering
(small `|a|` to large) is acceptable — for safety, prefer non-aliasing
buffers.

This is the discrete law-of-total-cumulants building block: the
contribution of one source cell's moments to the new (coarser) cell's
moments at the new centroid is precisely a moment shift.
"""
function shift_moments!(out::AbstractVector{T}, src::AbstractVector{T},
                          D::Integer, order::Integer, Δ::NTuple) where T
    length(out) == moments_length(D, order) ||
        throw(DimensionMismatch("out length"))
    length(src) == moments_length(D, order) ||
        throw(DimensionMismatch("src length"))
    multi = moment_multiindices(Int(D), Int(order))
    n = length(multi)
    Δt = ntuple(d -> T(Δ[d]), Int(D))
    @inbounds for ka in 1:n
        a = multi[ka]
        s = zero(T)
        # Iterate over all b ≤ a (componentwise)
        for kb in 1:n
            b = multi[kb]
            # Check b ≤ a componentwise
            ok = true
            for d in 1:Int(D)
                if b[d] > a[d]
                    ok = false; break
                end
            end
            ok || continue
            coeff = one(T)
            for d in 1:Int(D)
                coeff *= T(binomial(a[d], b[d])) * (-Δt[d])^(a[d] - b[d])
            end
            s += coeff * src[kb]
        end
        out[ka] = s
    end
    return out
end

"""
    shift_moments(src::AbstractVector{T}, D, order, Δ) where T

Allocating variant of `shift_moments!`. Returns a new vector.
"""
function shift_moments(src::AbstractVector{T}, D::Integer, order::Integer,
                         Δ::NTuple) where T
    out = similar(src)
    shift_moments!(out, src, D, order, Δ)
    return out
end
