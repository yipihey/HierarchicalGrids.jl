"""
    Bases

Polynomial bases for high-order field reconstruction. Supports three
common families:

- `MonomialBasis{D, P}` — total-degree monomials in D variables, degree ≤ P.
  Convenient for cell-tree (quad/hex) cells using reference coordinates.
- `BernsteinBasis{D, P}` — Bernstein polynomials on the D-simplex, degree P.
  Natural for triangle/tetrahedron cells; the convex-hull property gives
  cheap positivity certificates (`all_bernstein_coeffs_positive`).
- `LagrangeBasis{D, P}` — nodal Lagrange polynomials on a chosen point set.
  Convenient when fields are stored as values at nodes rather than abstract
  coefficients.

All bases share a common interface:

```julia
n_coeffs(::AbstractBasis{D, P}) :: Int
evaluate(basis, coeffs, point) :: T
gradient(basis, coeffs, point) :: NTuple{D, T}
```

Coefficient layouts use a canonical ordering documented per basis. The
conversion between bases (e.g., monomial → Bernstein for a positivity
check) is provided by `change_basis(target, source, source_coeffs)`.

Reference domains:
- 1D interval: [0, 1]
- 2D simplex (triangle): {(x, y) : x ≥ 0, y ≥ 0, x + y ≤ 1}
- 2D quad: [0, 1]²
- 3D simplex (tetrahedron): {(x, y, z) : x ≥ 0, y ≥ 0, z ≥ 0, x + y + z ≤ 1}
- 3D cube: [0, 1]³
"""
module Bases

export AbstractBasis
export MonomialBasis, BernsteinBasis, LagrangeBasis
export n_coeffs, evaluate, gradient
export all_bernstein_coeffs_positive, is_positive_certificate
export bernstein_positivity_certificate
export change_basis

# ============================================================================
# Abstract type
# ============================================================================

"""
    AbstractBasis{D, P}

A polynomial basis in `D` spatial dimensions of degree (or maximum total
degree) `P`. Concrete subtypes implement the common interface.
"""
abstract type AbstractBasis{D, P} end

# Default queries (must be defined concretely)
"""
    n_coeffs(basis) :: Int

Number of coefficients in the basis. For total-degree-P polynomials in D
variables: binomial(D+P, P). For tensor-product bases on quads/cubes:
(P+1)^D.
"""
function n_coeffs end

"""
    evaluate(basis, coeffs, point) :: T

Evaluate the polynomial defined by `coeffs` at `point` in reference
coordinates. `coeffs` is an iterable of length `n_coeffs(basis)`; `point`
is an NTuple{D, T} or AbstractVector of length D.
"""
function evaluate end

"""
    gradient(basis, coeffs, point) :: NTuple{D, T}

Gradient of the polynomial at `point`, as an NTuple{D, T} of partial
derivatives. Computed in reference coordinates; users responsible for
chain rule to physical coordinates.
"""
function gradient end

# ============================================================================
# MonomialBasis
# ============================================================================

"""
    MonomialBasis{D, P}() <: AbstractBasis{D, P}

Total-degree monomials in `D` variables, total degree ≤ `P`.

# Coefficient ordering (canonical)

For `D=1`, `P=3`: `[1, x, x², x³]` — coefficient `c[k+1]` multiplies `x^k`.

For `D=2`, `P=2`: `[1, x, y, x², xy, y²]` — graded lexicographic, lower
total degree first; within a degree, lex order by exponent tuple.

For `D=3`, `P=2`: `[1, x, y, z, x², xy, xz, y², yz, z²]`.

The ordering is provided by `monomial_exponents(D, P)` returning the list
of exponent tuples in canonical order.
"""
struct MonomialBasis{D, P} <: AbstractBasis{D, P}
    function MonomialBasis{D, P}() where {D, P}
        D >= 1 || throw(ArgumentError("dimension D must be ≥ 1"))
        P >= 0 || throw(ArgumentError("degree P must be ≥ 0"))
        new{D, P}()
    end
end

# Number of total-degree monomials in D variables of degree ≤ P
@inline n_coeffs(::MonomialBasis{D, P}) where {D, P} = binomial(D + P, P)

"""
    monomial_exponents(D, P) :: Vector{NTuple{D, Int}}

Generate the canonical ordering of exponent tuples for total-degree
monomials in D variables, degree ≤ P. Graded lex: lower total degree
first; within a degree, lex order on exponents.
"""
function monomial_exponents(D::Int, P::Int)
    exponents = NTuple{D, Int}[]
    # For each total degree d = 0:P, list all D-tuples summing to d
    for d in 0:P
        _push_tuples_summing_to!(exponents, D, d)
    end
    return exponents
end

# Helper: push to `out` all N-tuples of nonneg ints summing to d, in lex order.
# The tuple width N is fixed by the eltype of `out`.
function _push_tuples_summing_to!(out::Vector{NTuple{N, Int}}, N_arg::Int, d::Int) where N
    @assert N_arg == N "N_arg must match tuple width parameter N"
    if N == 1
        push!(out, (d,))
        return
    end
    # First slot takes value k from d down to 0; recurse on remaining (N-1) slots summing to d-k
    for k in d:-1:0
        rest = NTuple{N-1, Int}[]
        _push_tuples_summing_to!(rest, N-1, d-k)
        for r in rest
            push!(out, (k, r...))
        end
    end
end

# Specialized monomial_exponents for common cases — generated once, cached
const _monomial_exponent_cache = Dict{Tuple{Int, Int}, Vector}()
function _cached_monomial_exponents(D::Int, P::Int)
    key = (D, P)
    if !haskey(_monomial_exponent_cache, key)
        _monomial_exponent_cache[key] = monomial_exponents(D, P)
    end
    return _monomial_exponent_cache[key]::Vector{NTuple{D, Int}}
end

# Evaluate monomial polynomial at a point
@inline function evaluate(::MonomialBasis{D, P}, coeffs, point) where {D, P}
    length(coeffs) == binomial(D + P, P) || throw(DimensionMismatch("coeffs length doesn't match basis"))
    exps = _cached_monomial_exponents(D, P)
    T = promote_type(eltype(coeffs), eltype(point))
    result = zero(T)
    @inbounds for k in eachindex(exps)
        e = exps[k]
        m = one(T)
        for i in 1:D
            m *= point[i]^e[i]
        end
        result += coeffs[k] * m
    end
    return result
end

# Gradient — partial derivative in each variable
@inline function gradient(::MonomialBasis{D, P}, coeffs, point) where {D, P}
    length(coeffs) == binomial(D + P, P) || throw(DimensionMismatch("coeffs length doesn't match basis"))
    exps = _cached_monomial_exponents(D, P)
    T = promote_type(eltype(coeffs), eltype(point))
    grad = ntuple(_ -> zero(T), Val(D))
    grad_arr = collect(grad)  # mutable for accumulation
    @inbounds for k in eachindex(exps)
        e = exps[k]
        c = coeffs[k]
        # Partial wrt variable d: derivative of x_1^e[1] * ... * x_D^e[D] wrt x_d is e[d] * x_d^(e[d]-1) * (rest)
        for d in 1:D
            if e[d] > 0
                m = c * e[d]
                for i in 1:D
                    if i == d
                        m *= point[i]^(e[i] - 1)
                    else
                        m *= point[i]^e[i]
                    end
                end
                grad_arr[d] += m
            end
        end
    end
    return ntuple(d -> grad_arr[d], Val(D))
end

# ============================================================================
# BernsteinBasis
# ============================================================================

"""
    BernsteinBasis{D, P}() <: AbstractBasis{D, P}

Bernstein polynomials on the D-simplex of degree exactly P. The basis has
binomial(D+P, P) functions, one per multi-index α = (α_0, α_1, ..., α_D)
with sum α_i = P.

# Reference simplex

The D-simplex is parameterized by barycentric coordinates λ_0, λ_1, ..., λ_D
with sum λ_i = 1 and λ_i ≥ 0. Equivalently, in Cartesian reference
coordinates (x_1, ..., x_D), λ_0 = 1 - x_1 - ... - x_D and λ_i = x_i.

Basis functions: B_α^P(λ) = (P!/(α_0! α_1! ... α_D!)) λ_0^α_0 λ_1^α_1 ... λ_D^α_D.

# Coefficient ordering

Multi-indices α with sum P are listed in the same graded-lex order as
monomial exponents for `(D+1)` variables of degree exactly P:
α = (α_0, α_1, ..., α_D), sum = P.

# Convex-hull property

For coefficients c_α and a point with barycentric coordinates λ_i ≥ 0:
min(c_α) ≤ p(λ) ≤ max(c_α). Hence: if all coefficients are positive,
the polynomial is positive everywhere on the simplex. This gives a cheap
sufficient (not necessary) positivity certificate.
"""
struct BernsteinBasis{D, P} <: AbstractBasis{D, P}
    function BernsteinBasis{D, P}() where {D, P}
        D >= 1 || throw(ArgumentError("dimension D must be ≥ 1"))
        P >= 0 || throw(ArgumentError("degree P must be ≥ 0"))
        new{D, P}()
    end
end

@inline n_coeffs(::BernsteinBasis{D, P}) where {D, P} = binomial(D + P, P)

"""
    bernstein_multiindices(D, P) :: Vector{NTuple{D+1, Int}}

The list of multi-indices (α_0, α_1, ..., α_D) with sum = P, in canonical
order. Length is binomial(D+P, P).
"""
function bernstein_multiindices(D::Int, P::Int)
    indices = NTuple{D+1, Int}[]
    _push_tuples_summing_to!(indices, D + 1, P)
    return indices
end

const _bernstein_multiindex_cache = Dict{Tuple{Int, Int}, Vector}()
function _cached_bernstein_multiindices(D::Int, P::Int)
    key = (D, P)
    if !haskey(_bernstein_multiindex_cache, key)
        _bernstein_multiindex_cache[key] = bernstein_multiindices(D, P)
    end
    return _bernstein_multiindex_cache[key]::Vector{NTuple{D+1, Int}}
end

# Multinomial coefficient P! / (α_0! α_1! ... α_D!)
function multinomial_coefficient(P::Int, α::NTuple{N, Int}) where N
    sum(α) == P || throw(ArgumentError("multi-index sum must equal P"))
    coef = factorial(big(P))
    for k in α
        coef ÷= factorial(big(k))
    end
    return Int(coef)  # for typical P (≤ 12 or so) this fits Int easily
end

# Evaluate Bernstein polynomial at a point (Cartesian reference coords)
@inline function evaluate(::BernsteinBasis{D, P}, coeffs, point) where {D, P}
    length(coeffs) == binomial(D + P, P) || throw(DimensionMismatch("coeffs length doesn't match basis"))
    inds = _cached_bernstein_multiindices(D, P)
    T = promote_type(eltype(coeffs), eltype(point))

    # Compute barycentric coordinates from Cartesian
    # λ_0 = 1 - x_1 - ... - x_D, λ_i = x_i
    λ = _to_barycentric(point, Val(D), T)

    result = zero(T)
    @inbounds for k in eachindex(inds)
        α = inds[k]
        m = T(multinomial_coefficient(P, α))
        # m * prod(λ_i ^ α_i)
        for i in 1:(D+1)
            m *= λ[i] ^ α[i]
        end
        result += coeffs[k] * m
    end
    return result
end

@inline function _to_barycentric(point, ::Val{D}, ::Type{T}) where {D, T}
    # λ_i = point[i] for i = 1:D, λ_0 = 1 - sum
    s = zero(T)
    for i in 1:D
        s += T(point[i])
    end
    return ntuple(i -> i == 1 ? one(T) - s : T(point[i-1]), Val(D + 1))
end

# Gradient in Cartesian coordinates
# Use the recursion: ∂B_α^P / ∂x_d = P * (B_{α-e_d}^{P-1} - B_{α-e_0}^{P-1})
# where e_i is the unit vector that decreases α_i by 1 (and undefined if α_i=0).
# We compute it directly via finite-difference of the barycentric form.
@inline function gradient(::BernsteinBasis{D, P}, coeffs, point) where {D, P}
    length(coeffs) == binomial(D + P, P) || throw(DimensionMismatch("coeffs length doesn't match basis"))
    inds = _cached_bernstein_multiindices(D, P)
    T = promote_type(eltype(coeffs), eltype(point))
    λ = _to_barycentric(point, Val(D), T)

    # ∂(∏ λ_i^α_i) / ∂x_d
    # x_d → λ_d = x_d, λ_0 = 1 - Σ x_d. So ∂λ_d/∂x_d = 1, ∂λ_0/∂x_d = -1, others 0.
    # Hence ∂(∏ λ_i^α_i) / ∂x_d = α_d * λ_d^(α_d-1) * (∏_{i≠d, i≠0} λ_i^α_i) * λ_0^α_0
    #                            - α_0 * λ_0^(α_0-1) * (∏_{i≠0} λ_i^α_i)

    grad_arr = zeros(T, D)
    @inbounds for k in eachindex(inds)
        α = inds[k]
        c = coeffs[k] * T(multinomial_coefficient(P, α))
        # Compute base = ∏ λ_i^α_i
        base = one(T)
        for i in 1:(D+1)
            base *= λ[i] ^ α[i]
        end
        # For each Cartesian direction d (= 1:D), update grad
        for d in 1:D
            # λ index for x_d is i = d+1 in the barycentric tuple (1-indexed)
            # Recall: λ[1] = λ_0, λ[2] = λ_1 = x_1, ..., λ[D+1] = λ_D = x_D
            ib = d + 1
            # ∂/∂x_d term from λ_d^α_d: α_d * λ_d^(α_d-1) * (rest)
            if α[ib] > 0
                # Derivative of λ_d^α_d wrt x_d, multiplied by other λ^α factors
                # = (α_d / λ_d) * base, when λ_d > 0
                if !iszero(λ[ib])
                    grad_arr[d] += c * α[ib] / λ[ib] * base
                else
                    # λ_d = 0; only contribution is when α_d = 1 (then derivative is 1)
                    # Actually if α_d = 1, λ_d^α_d = λ_d, derivative is 1; if α_d > 1, derivative is 0.
                    # General formula: derivative of λ_d^α_d at λ_d=0 is α_d * 0^(α_d-1) which is 0 if α_d>1, 1 if α_d=1.
                    if α[ib] == 1
                        # base / λ_d would be 0/0; recompute base without λ_d^α_d factor
                        b2 = one(T)
                        for i in 1:(D+1)
                            i == ib && continue
                            b2 *= λ[i] ^ α[i]
                        end
                        grad_arr[d] += c * b2
                    end
                end
            end
            # ∂/∂x_d term from λ_0^α_0: -α_0 * λ_0^(α_0-1) * (rest)
            if α[1] > 0
                if !iszero(λ[1])
                    grad_arr[d] -= c * α[1] / λ[1] * base
                else
                    if α[1] == 1
                        b2 = one(T)
                        for i in 1:(D+1)
                            i == 1 && continue
                            b2 *= λ[i] ^ α[i]
                        end
                        grad_arr[d] -= c * b2
                    end
                end
            end
        end
    end
    return ntuple(d -> grad_arr[d], Val(D))
end

"""
    all_bernstein_coeffs_positive(coeffs) :: Bool

Sufficient (not necessary) test for positivity of a Bernstein polynomial:
if all coefficients are positive, the polynomial is positive everywhere
on the reference simplex. The converse may fail: a positive polynomial
can still have some negative Bernstein coefficients (the test gets
sharper under degree elevation).

Use this for cheap shell-crossing detection on Jacobian polynomials
reconstructed in Bernstein form.
"""
@inline function all_bernstein_coeffs_positive(coeffs)
    @inbounds for c in coeffs
        c > zero(c) || return false
    end
    return true
end

"""
    bernstein_positivity_certificate(coeffs, basis::BernsteinBasis{D,P}; atol = 0)
        -> (positive::Bool, offending::Union{Nothing, NTuple{D+1, Int}})

Strict-positivity certificate. Returns `(true, nothing)` if every coefficient
satisfies `c > -atol` (i.e. is positive up to tolerance `atol`); otherwise
`(false, α)` where `α` is the multi-index of the first offending coefficient in
the basis's standard ordering.

`atol >= 0` is a tolerance for treating tiny negative values as zero. With the
default `atol = 0`, the test is strict positivity (`c > 0`).

By the convex-hull property (see `BernsteinBasis`), `true` is sufficient for
strict positivity on the entire reference simplex (in the limit `atol → 0`);
`false` does NOT imply non-positivity (the test gets sharper under degree
elevation).

The multi-index `α = (α_0, α_1, ..., α_D)` has length `D+1` (barycentric) and
satisfies `sum(α) == P`.
"""
function bernstein_positivity_certificate(coeffs, basis::BernsteinBasis{D, P};
                                          atol = zero(eltype(coeffs))) where {D, P}
    length(coeffs) == binomial(D + P, P) ||
        throw(DimensionMismatch("coeffs length doesn't match basis"))
    inds = _cached_bernstein_multiindices(D, P)
    threshold = -atol
    @inbounds for k in eachindex(inds)
        if !(coeffs[k] > threshold)
            return (false, inds[k])
        end
    end
    return (true, nothing)
end

# ============================================================================
# LagrangeBasis (1D, on equispaced or Chebyshev-Lobatto nodes)
# ============================================================================

"""
    LagrangeBasis{D, P}(nodes) <: AbstractBasis{D, P}

Nodal Lagrange basis on the given node set. Currently supports D=1 only.
Coefficients are values at the nodes; evaluation uses the barycentric
Lagrange formula (numerically stable).

# Construction

```julia
# Equispaced Lagrange basis on [0, 1]
nodes = collect(range(0.0, 1.0, length=4))  # 4 nodes for P=3
basis = LagrangeBasis{1, 3}(nodes)
```

# Reference: Berrut & Trefethen, "Barycentric Lagrange interpolation",
# SIAM Review 46:501–517 (2004).
"""
struct LagrangeBasis{D, P, T} <: AbstractBasis{D, P}
    nodes::Vector{T}
    weights::Vector{T}    # barycentric weights
    function LagrangeBasis{D, P}(nodes::AbstractVector{T}) where {D, P, T}
        D == 1 || throw(ArgumentError("LagrangeBasis currently supports D=1 only"))
        length(nodes) == P + 1 || throw(ArgumentError("LagrangeBasis{1, $P} requires $(P+1) nodes, got $(length(nodes))"))
        n = P + 1
        # Compute barycentric weights: w_j = 1 / prod_{i ≠ j} (x_j - x_i)
        weights = Vector{T}(undef, n)
        for j in 1:n
            prod = one(T)
            for i in 1:n
                i == j && continue
                prod *= nodes[j] - nodes[i]
            end
            weights[j] = one(T) / prod
        end
        new{D, P, T}(collect(nodes), weights)
    end
end

@inline n_coeffs(::LagrangeBasis{D, P, T}) where {D, P, T} = P + 1

# Barycentric formula for Lagrange interpolation
# p(x) = (sum_j w_j / (x - x_j) * f_j) / (sum_j w_j / (x - x_j))
# Special case when x = x_j: p(x_j) = f_j
function evaluate(basis::LagrangeBasis{1, P, T}, coeffs, point) where {P, T}
    length(coeffs) == P + 1 || throw(DimensionMismatch("coeffs length doesn't match basis"))
    x = first(point)  # 1D
    nodes = basis.nodes
    weights = basis.weights
    Tres = promote_type(eltype(coeffs), typeof(x))
    num = zero(Tres)
    den = zero(Tres)
    @inbounds for j in eachindex(nodes)
        diff = x - nodes[j]
        if iszero(diff)
            return Tres(coeffs[j])
        end
        wj = weights[j] / diff
        num += wj * coeffs[j]
        den += wj
    end
    return num / den
end

function gradient(basis::LagrangeBasis{1, P, T}, coeffs, point) where {P, T}
    # Use the differentiation formula for barycentric Lagrange (Berrut & Trefethen eq. 9.4):
    # p'(x_k) = sum_{j ≠ k} (w_j/w_k) / (x_k - x_j) * (f_j - f_k)   (at a node)
    # at non-node points, finite-difference of the formula.
    # For simplicity: use the standard p(x) form and differentiate analytically.
    length(coeffs) == P + 1 || throw(DimensionMismatch("coeffs length doesn't match basis"))
    x = first(point)
    nodes = basis.nodes
    weights = basis.weights
    Tres = promote_type(eltype(coeffs), typeof(x))

    # Check if x is at a node
    @inbounds for k in eachindex(nodes)
        if x == nodes[k]
            # Use the "second form" differentiation matrix entry at k
            # D_kk = -sum_{j ≠ k} D_kj
            # D_kj = (w_j / w_k) / (x_k - x_j)   for j ≠ k
            deriv = zero(Tres)
            for j in eachindex(nodes)
                j == k && continue
                Dkj = (weights[j] / weights[k]) / (nodes[k] - nodes[j])
                deriv += Dkj * (coeffs[j] - coeffs[k])
            end
            return (deriv,)
        end
    end

    # General point: differentiate the barycentric formula
    # p(x) = N(x) / D(x) where N = sum w_j/(x-x_j) f_j, D = sum w_j/(x-x_j)
    # p'(x) = (N' D - N D') / D^2
    # N'(x) = -sum w_j/(x-x_j)^2 f_j; similarly D'(x) = -sum w_j/(x-x_j)^2
    N = zero(Tres); D = zero(Tres)
    Np = zero(Tres); Dp = zero(Tres)
    @inbounds for j in eachindex(nodes)
        diff = x - nodes[j]
        wj = weights[j] / diff
        N += wj * coeffs[j]
        D += wj
        Np -= weights[j] / diff^2 * coeffs[j]
        Dp -= weights[j] / diff^2
    end
    pp = (Np * D - N * Dp) / D^2
    return (pp,)
end

# ============================================================================
# Change of basis: monomial → Bernstein (1D)
# ============================================================================

"""
    change_basis(target::BernsteinBasis{1, P}, source::MonomialBasis{1, P}, src_coeffs)

Convert a 1D polynomial from monomial to Bernstein form on [0,1].

Given a polynomial p(x) = sum c_k x^k, the equivalent Bernstein form
p(x) = sum b_i B_{i,P}(x) has coefficients b_i = sum_{k=0}^i (i choose k)/(P choose k) c_k.
"""
function change_basis(::BernsteinBasis{1, P}, ::MonomialBasis{1, P},
                      src_coeffs::AbstractVector) where P
    length(src_coeffs) == P + 1 || throw(DimensionMismatch("src coeffs length doesn't match"))
    T = float(eltype(src_coeffs))
    bern = zeros(T, P + 1)
    @inbounds for i in 0:P
        for k in 0:i
            bern[i+1] += T(binomial(i, k)) / T(binomial(P, k)) * src_coeffs[k+1]
        end
    end
    return bern
end

"""
    change_basis(target::MonomialBasis{1, P}, source::BernsteinBasis{1, P}, src_coeffs)

Convert a 1D polynomial from Bernstein to monomial form on [0,1].

Given Bernstein coefficients b_i, the equivalent monomial coefficients are
c_k = (P choose k) sum_{i=0}^k (-1)^{k-i} (k choose i) b_i.
"""
function change_basis(::MonomialBasis{1, P}, ::BernsteinBasis{1, P},
                      src_coeffs::AbstractVector) where P
    length(src_coeffs) == P + 1 || throw(DimensionMismatch("src coeffs length doesn't match"))
    T = float(eltype(src_coeffs))
    mono = zeros(T, P + 1)
    @inbounds for k in 0:P
        s = zero(T)
        for i in 0:k
            s += (-one(T))^(k-i) * T(binomial(k, i)) * src_coeffs[i+1]
        end
        mono[k+1] = T(binomial(P, k)) * s
    end
    return mono
end

# ============================================================================
# Change of basis: D-dim simplex Bernstein ↔ Monomial
# ============================================================================
#
# For the D-simplex with barycentric coordinates λ = (λ_0, ..., λ_D), the
# Bernstein basis functions are
#
#     B_α(λ) = (P! / α!) · λ_0^{α_0} λ_1^{α_1} ... λ_D^{α_D}
#
# where α is a barycentric multi-index of length D+1 with |α| = P. In
# Cartesian reference coordinates (x_1, ..., x_D), λ_0 = 1 - Σ x_d and
# λ_i = x_i for i ≥ 1, so
#
#     B_α(x) = (P! / α!) · (1 - Σ x_d)^{α_0} · ∏_{d=1..D} x_d^{α_d}.
#
# Expanding (1 - Σ x_d)^{α_0} via the multinomial theorem yields a finite
# sum of monomials x^β. Each Bernstein function B_α therefore has a closed
# expression as a Σ_β c_{α,β} x^β with β = α_{1:D} + γ, |γ| ≤ α_0:
#
#     c_{α,β} = (P! / (α_1!...α_D!)) · (-1)^{|γ|} / (γ_1!...γ_D! · (α_0 - |γ|)!)
#
# The MonomialBasis on the simplex still indexes total-degree monomials
# x^β with |β| ≤ P. Both bases have n = binomial(D + P, P) functions, so
# the change-of-basis matrix is square; its inverse exists and is the
# reverse map.
#
# We cache the matrices once per (D, P) under a thread-safe lock; build is
# O(n_coeffs * binomial(P + D, D)) and trivial for the typical D ≤ 3, P ≤ 4
# regime that AdaptiveField uses.

const _BERN_TO_MONO_CACHE = Dict{Tuple{Int, Int}, Matrix{Float64}}()
const _MONO_TO_BERN_CACHE = Dict{Tuple{Int, Int}, Matrix{Float64}}()
const _CHANGE_BASIS_CACHE_LOCK = ReentrantLock()

# Enumerate barycentric multi-indices α of length D+1 with |α| = P, in the
# canonical order produced by `_push_tuples_summing_to!`. (Same as
# `bernstein_multiindices(D, P)`, but returns Vector{Vector{Int}} for
# uniform handling alongside `_enumerate_mono_multiindices_simplex`.)
function _enumerate_bern_multiindices(D::Int, P::Int)
    return _cached_bernstein_multiindices(D, P)
end

# Enumerate Cartesian multi-indices β of length D with |β| ≤ P, in HG's
# graded-lex order matching `monomial_exponents(D, P)`.
function _enumerate_mono_multiindices_simplex(D::Int, P::Int)
    return _cached_monomial_exponents(D, P)
end

# Add to row indexed by β=α[2..D+1]+γ in `M[:, j]` the coefficient
# (P! / (α_1!...α_D!)) · (-1)^|γ| / (γ_1!...γ_D! · (α_0-|γ|)!).
#
# We pre-compute the lookup `mono_idx_lookup` from β-tuple → row index of
# the monomial in the graded-lex layout once per (D, P) build; the inner
# loop just performs index arithmetic.
function _expand_bernstein_into_monomials!(M::Matrix{Float64}, j::Int,
                                            α::NTuple{Np1, Int},
                                            mono_idx_lookup::Dict{NTuple{Dl, Int}, Int},
                                            P::Int) where {Np1, Dl}
    α0 = α[1]
    # Pre-divide P! by α_1!...α_Dl! once per Bernstein function.
    αrest_fact = one(Float64)
    for d in 1:Dl
        αrest_fact *= Float64(factorial(α[d + 1]))
    end
    Pfac = Float64(factorial(P))
    base_coef = Pfac / αrest_fact

    # Enumerate γ ∈ ℤ_{≥0}^Dl with |γ| ≤ α0 (small; α0 ≤ P).
    γ = zeros(Int, Dl)
    _expand_bern_recursive!(M, j, α, γ, 1, α0, base_coef, mono_idx_lookup)
    return
end

# Recursively walk γ. At each level d we pick γ[d] ∈ 0..(α0 - sum_so_far);
# at the leaf (d > Dl) we have a complete γ and emit the monomial term.
function _expand_bern_recursive!(M::Matrix{Float64}, j::Int,
                                  α::NTuple{Np1, Int}, γ::Vector{Int},
                                  d::Int, remaining::Int, base_coef::Float64,
                                  mono_idx_lookup::Dict{NTuple{Dl, Int}, Int}
                                  ) where {Np1, Dl}
    if d > Dl
        # Leaf: γ complete, |γ| = sum(γ), tail = α0 - |γ| = remaining.
        γabs = 0
        γfact = one(Float64)
        for k in 1:Dl
            γabs += γ[k]
            γfact *= Float64(factorial(γ[k]))
        end
        tail = remaining  # α0 - |γ|
        sign = isodd(γabs) ? -1.0 : 1.0
        coef = base_coef * sign / (γfact * Float64(factorial(tail)))
        # β_d = α[d+1] + γ[d] for d=1..Dl
        β = ntuple(k -> α[k + 1] + γ[k], Val(Dl))
        row = mono_idx_lookup[β]::Int
        @inbounds M[row, j] += coef
        return
    end
    @inbounds for v in 0:remaining
        γ[d] = v
        _expand_bern_recursive!(M, j, α, γ, d + 1, remaining - v, base_coef,
                                mono_idx_lookup)
    end
    @inbounds γ[d] = 0
    return
end

# Build the Bernstein → Monomial matrix for the D-simplex at degree P. Column
# j is the monomial coefficient vector of B_{α_j} in the canonical layout.
function _build_bern_to_mono_matrix(D::Int, P::Int)
    bern_idx = _enumerate_bern_multiindices(D, P)
    mono_idx = _enumerate_mono_multiindices_simplex(D, P)
    n = length(bern_idx)
    @assert length(mono_idx) == n "Bernstein and monomial counts must match"
    # Fast row-lookup: β-tuple → graded-lex row index.
    lookup = Dict{NTuple{D, Int}, Int}()
    sizehint!(lookup, n)
    @inbounds for (k, β) in enumerate(mono_idx)
        lookup[β] = k
    end
    M = zeros(Float64, n, n)
    @inbounds for j in 1:n
        α = bern_idx[j]    # NTuple{D+1, Int}
        _expand_bernstein_into_monomials!(M, j, α, lookup, P)
    end
    return M
end

# Thread-safe cached fetch of the Bernstein → Monomial matrix.
function _bernstein_to_monomial_matrix(::Val{D}, ::Val{P}) where {D, P}
    key = (D, P)
    M = lock(_CHANGE_BASIS_CACHE_LOCK) do
        get(_BERN_TO_MONO_CACHE, key, nothing)
    end
    M === nothing || return M
    # Build outside the lock to avoid blocking other (D, P) lookups.
    M_built = _build_bern_to_mono_matrix(D, P)
    return lock(_CHANGE_BASIS_CACHE_LOCK) do
        get!(_BERN_TO_MONO_CACHE, key, M_built)
    end
end

# Thread-safe cached fetch of the Monomial → Bernstein matrix (matrix inverse).
function _monomial_to_bernstein_matrix(::Val{D}, ::Val{P}) where {D, P}
    key = (D, P)
    M = lock(_CHANGE_BASIS_CACHE_LOCK) do
        get(_MONO_TO_BERN_CACHE, key, nothing)
    end
    M === nothing || return M
    Mfwd = _bernstein_to_monomial_matrix(Val(D), Val(P))
    Minv_built = inv(Mfwd)
    return lock(_CHANGE_BASIS_CACHE_LOCK) do
        get!(_MONO_TO_BERN_CACHE, key, Minv_built)
    end
end

"""
    change_basis(target::MonomialBasis{D, P}, source::BernsteinBasis{D, P}, bern_coeffs)

Convert a polynomial on the D-simplex from Bernstein form to total-degree
monomial form. The reference simplex is parameterized by Cartesian
coordinates (x_1, ..., x_D) with `λ_0 = 1 - Σ x_d`. The conversion matrix
is built once per `(D, P)` pair and cached (thread-safe).
"""
function change_basis(::MonomialBasis{D, P}, ::BernsteinBasis{D, P},
                      bern_coeffs::AbstractVector) where {D, P}
    n = binomial(D + P, P)
    length(bern_coeffs) == n ||
        throw(DimensionMismatch("Bernstein coeffs length $(length(bern_coeffs)) ≠ $n"))
    M = _bernstein_to_monomial_matrix(Val(D), Val(P))
    T = float(eltype(bern_coeffs))
    if T === Float64
        return M * bern_coeffs
    else
        return convert(Vector{T}, M * bern_coeffs)
    end
end

"""
    change_basis(target::BernsteinBasis{D, P}, source::MonomialBasis{D, P}, mono_coeffs)

Convert a polynomial on the D-simplex from total-degree monomial form to
Bernstein form. Inverse of the Bernstein → Monomial map; the matrix is
cached per `(D, P)` pair.
"""
function change_basis(::BernsteinBasis{D, P}, ::MonomialBasis{D, P},
                      mono_coeffs::AbstractVector) where {D, P}
    n = binomial(D + P, P)
    length(mono_coeffs) == n ||
        throw(DimensionMismatch("Monomial coeffs length $(length(mono_coeffs)) ≠ $n"))
    Minv = _monomial_to_bernstein_matrix(Val(D), Val(P))
    T = float(eltype(mono_coeffs))
    if T === Float64
        return Minv * mono_coeffs
    else
        return convert(Vector{T}, Minv * mono_coeffs)
    end
end

# ============================================================================
# Convenience: direct positivity test for monomial polys (1D) via Bernstein
# ============================================================================

"""
    is_positive_certificate(::MonomialBasis{1, P}, mono_coeffs) :: Bool

Sufficient certificate: convert to Bernstein, check all coefficients positive.
Returns false if positivity cannot be certified (the polynomial may still be
positive in fact).
"""
function is_positive_certificate(mb::MonomialBasis{1, P}, mono_coeffs) where P
    bern_coeffs = change_basis(BernsteinBasis{1, P}(), mb, mono_coeffs)
    return all_bernstein_coeffs_positive(bern_coeffs)
end

end # module Bases
