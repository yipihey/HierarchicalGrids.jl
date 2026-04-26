"""
    Quadrature

Gauss quadrature rules for numerical integration over standard reference
domains:

- 1D interval [0, 1]: Gauss–Legendre, exact for polynomials of degree 2n-1.
- 2D triangle (the standard simplex): symmetric rules from Dunavant 1985.
- 2D quadrilateral [0, 1]²: tensor product of 1D Gauss–Legendre.
- 3D tetrahedron: symmetric rules from Keast 1986 (low orders) and Witherden-Vincent 2015 (higher).
- 3D cube [0, 1]³: tensor product of 1D Gauss–Legendre.

Common interface:

```julia
q = gauss_quadrature_interval(n)        # n-point rule, exact for deg ≤ 2n-1
q = gauss_quadrature_triangle(order)    # exact for total degree ≤ order
q = gauss_quadrature_quad(n)            # tensor product, n×n points
q = gauss_quadrature_tetrahedron(order)
q = gauss_quadrature_cube(n)

integrate(f, q) :: T                    # ∫ f(x) dx over the reference domain
```

The `QuadRule{D, T}` struct carries `points::Vector{NTuple{D, T}}` and
`weights::Vector{T}` together. The weights are normalized so that the sum
equals the volume of the reference domain (1 for interval/quad/cube,
1/2 for triangle, 1/6 for tetrahedron).
"""
module Quadrature

import LinearAlgebra
using ..Geometry: Interval, is_empty, interval_length, affine_map_from_reference
using ..Bases: AbstractBasis, n_coeffs, evaluate

export QuadRule
export gauss_quadrature_interval, gauss_quadrature_triangle
export gauss_quadrature_quad, gauss_quadrature_tetrahedron, gauss_quadrature_cube
export integrate, n_quad_points
export integrate_polynomial_on_subinterval, action_error_l2

"""
    QuadRule{D, T}

A quadrature rule on a D-dimensional reference domain. `points[k]` is the
k-th node (NTuple{D, T}); `weights[k]` is the corresponding weight.

The integral of a function f over the domain is approximated by
`sum(weights[k] * f(points[k]) for k in 1:n_quad_points(q))`.
"""
struct QuadRule{D, T}
    points::Vector{NTuple{D, T}}
    weights::Vector{T}
    function QuadRule{D, T}(points::Vector{NTuple{D, T}}, weights::Vector{T}) where {D, T}
        length(points) == length(weights) || throw(ArgumentError("points and weights must have same length"))
        new{D, T}(points, weights)
    end
end

"""
    n_quad_points(q::QuadRule) :: Int

Number of nodes (and weights) in the rule.
"""
@inline n_quad_points(q::QuadRule) = length(q.points)

"""
    integrate(f, q::QuadRule) :: T

Numerically integrate the function `f` over the reference domain using
the quadrature rule `q`. `f` must accept an NTuple{D, T} argument and
return a scalar (or any type closed under +).
"""
@inline function integrate(f, q::QuadRule{D, T}) where {D, T}
    s = f(q.points[1]) * q.weights[1]
    @inbounds for k in 2:length(q.points)
        s += f(q.points[k]) * q.weights[k]
    end
    return s
end

# ============================================================================
# 1D Gauss–Legendre on [0, 1]
# ============================================================================

# Standard formulas: Golub-Welsch algorithm (Eigenvalues of Jacobi matrix)
# For [-1, 1] then linear remap to [0, 1]: x_new = (x + 1) / 2; w_new = w / 2

"""
    gauss_quadrature_interval(n::Int; T=Float64) :: QuadRule{1, T}

n-point Gauss–Legendre quadrature on [0, 1]. Exact for polynomials of
degree at most 2n - 1.
"""
function gauss_quadrature_interval(n::Int; T::Type=Float64)
    n >= 1 || throw(ArgumentError("Need n ≥ 1 quadrature points"))
    pts_m1to1, wts_m1to1 = _gauss_legendre_nodes_weights(n, T)
    # Remap [-1, 1] → [0, 1]: x_new = (x + 1)/2, w_new = w / 2
    points = [((x + one(T)) / T(2),) for x in pts_m1to1]
    weights = [w / T(2) for w in wts_m1to1]
    return QuadRule{1, T}(points, weights)
end

# Golub-Welsch: nodes are eigenvalues of the symmetric tridiagonal Jacobi matrix
# For Legendre on [-1, 1]: a_k = 0, b_k = k / sqrt(4k² - 1)
function _gauss_legendre_nodes_weights(n::Int, ::Type{T}) where T
    n == 1 && return ([zero(T)], [T(2)])
    # Build the n×n symmetric tridiagonal Jacobi matrix
    # Diagonal: a_k = 0; sub/super: b_k for k = 1:n-1
    diag = zeros(Float64, n)
    subdiag = Float64[Float64(k) / sqrt(Float64(4*k^2 - 1)) for k in 1:(n-1)]
    M = _symtridiag(diag, subdiag)
    eigvals_, eigvecs_ = _symtridiag_eigen(M)
    # Sort by eigenvalue (already sorted from LAPACK, but be safe)
    perm = sortperm(eigvals_)
    nodes_f64 = eigvals_[perm]
    # Weights: w_k = 2 * (first component of k-th eigenvector)^2  (for Legendre on [-1,1])
    weights_f64 = [2.0 * eigvecs_[1, p]^2 for p in perm]
    return (T.(nodes_f64), T.(weights_f64))
end

# Build symmetric tridiagonal matrix as a regular Matrix (small n, no need for SymTridiagonal type plumbing)
function _symtridiag(diag::Vector{Float64}, subdiag::Vector{Float64})
    n = length(diag)
    M = zeros(Float64, n, n)
    for k in 1:n
        M[k, k] = diag[k]
    end
    for k in 1:(n-1)
        M[k, k+1] = subdiag[k]
        M[k+1, k] = subdiag[k]
    end
    return M
end

# Symmetric eigen via the LinearAlgebra stdlib
function _symtridiag_eigen(M::Matrix{Float64})
    F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(M))
    return (F.values, F.vectors)
end

# ============================================================================
# 2D Quadrilateral [0, 1]²: tensor product
# ============================================================================

"""
    gauss_quadrature_quad(n::Int; T=Float64) :: QuadRule{2, T}

n×n tensor-product Gauss–Legendre quadrature on [0, 1]². Exact for
polynomials of degree at most 2n - 1 in each variable separately.
"""
function gauss_quadrature_quad(n::Int; T::Type=Float64)
    q1d = gauss_quadrature_interval(n; T=T)
    points = NTuple{2, T}[]
    weights = T[]
    for i in 1:n_quad_points(q1d), j in 1:n_quad_points(q1d)
        push!(points, (q1d.points[i][1], q1d.points[j][1]))
        push!(weights, q1d.weights[i] * q1d.weights[j])
    end
    return QuadRule{2, T}(points, weights)
end

# ============================================================================
# 3D Cube [0, 1]³: tensor product
# ============================================================================

"""
    gauss_quadrature_cube(n::Int; T=Float64) :: QuadRule{3, T}

n³ tensor-product Gauss–Legendre quadrature on [0, 1]³.
"""
function gauss_quadrature_cube(n::Int; T::Type=Float64)
    q1d = gauss_quadrature_interval(n; T=T)
    points = NTuple{3, T}[]
    weights = T[]
    for i in 1:n_quad_points(q1d), j in 1:n_quad_points(q1d), k in 1:n_quad_points(q1d)
        push!(points, (q1d.points[i][1], q1d.points[j][1], q1d.points[k][1]))
        push!(weights, q1d.weights[i] * q1d.weights[j] * q1d.weights[k])
    end
    return QuadRule{3, T}(points, weights)
end

# ============================================================================
# 2D Triangle: symmetric rules from Dunavant 1985
# ============================================================================
#
# Reference triangle: {(x, y) : x ≥ 0, y ≥ 0, x + y ≤ 1}, area = 1/2.
# Weights normalized so they sum to 1/2.
#
# The triangle rules below use barycentric coordinates (λ_0, λ_1, λ_2) with
# sum 1; we convert to Cartesian (x, y) via x = λ_1, y = λ_2 (so λ_0 = 1-x-y).
#
# Source: Dunavant, "High degree efficient symmetrical Gaussian quadrature
# rules for the triangle," IJNME 21:1129–1148 (1985). Also cross-checked
# against the Witherden-Vincent 2015 tabulations.

"""
    gauss_quadrature_triangle(order::Int; T=Float64) :: QuadRule{2, T}

Symmetric quadrature on the reference triangle {(x,y) : x≥0, y≥0, x+y≤1},
exact for polynomials of total degree ≤ `order`.

Supported orders: 1, 2, 3, 4, 5, 6, 7. Higher orders raise an error;
extend by adding more rule tabulations from Dunavant.
"""
function gauss_quadrature_triangle(order::Int; T::Type=Float64)
    bary_pts, weights = _triangle_rule(order, T)
    # Convert barycentric to Cartesian: (x, y) = (λ_1, λ_2) given (λ_0, λ_1, λ_2)
    points = [(p[2], p[3]) for p in bary_pts]
    return QuadRule{2, T}(points, weights)
end

function _triangle_rule(order::Int, ::Type{T}) where T
    if order <= 1
        # 1-point rule: centroid, weight = area = 1/2
        return ([(T(1)/T(3), T(1)/T(3), T(1)/T(3))], [T(1)/T(2)])
    elseif order == 2
        # 3-point rule: midpoints of edges, each with weight 1/6
        bary = [(T(0), T(1)/T(2), T(1)/T(2)),
                (T(1)/T(2), T(0), T(1)/T(2)),
                (T(1)/T(2), T(1)/T(2), T(0))]
        weights = fill(T(1)/T(6), 3)
        return (bary, weights)
    elseif order == 3
        # 4-point rule: centroid (negative weight) + 3 vertices-shifted points.
        # Use the standard 4-point Dunavant rule:
        bary = [
            (T(1)/T(3), T(1)/T(3), T(1)/T(3)),  # centroid
            (T(3)/T(5), T(1)/T(5), T(1)/T(5)),
            (T(1)/T(5), T(3)/T(5), T(1)/T(5)),
            (T(1)/T(5), T(1)/T(5), T(3)/T(5)),
        ]
        # Weights from Dunavant Table I, order 3 — already include area factor:
        # -27/96 + 3 * 25/96 = 48/96 = 1/2 = area of reference triangle.
        weights = T[
            -T(27) / T(96),
             T(25) / T(96),
             T(25) / T(96),
             T(25) / T(96),
        ]
        return (bary, weights)
    elseif order == 4
        # 6-point rule, Dunavant Table I order 4.
        # Two orbits: 3-fold S₂ orbit at α = 0.445948..., 3-fold S₂ orbit at α = 0.091576...
        a1 = T(0.445948490915965)  # corresponds to α with one barycentric = 1-2α
        a2 = T(0.091576213509771)
        bary = [
            (T(1) - T(2)*a1, a1, a1),
            (a1, T(1) - T(2)*a1, a1),
            (a1, a1, T(1) - T(2)*a1),
            (T(1) - T(2)*a2, a2, a2),
            (a2, T(1) - T(2)*a2, a2),
            (a2, a2, T(1) - T(2)*a2),
        ]
        w1 = T(0.223381589678011) * T(1)/T(2)  # from Dunavant; multiply by area
        w2 = T(0.109951743655322) * T(1)/T(2)
        weights = [w1, w1, w1, w2, w2, w2]
        return (bary, weights)
    elseif order == 5
        # 7-point rule, Dunavant Table I order 5.
        bary = NTuple{3, T}[]
        weights = T[]
        # Centroid
        push!(bary, (T(1)/T(3), T(1)/T(3), T(1)/T(3)))
        push!(weights, T(0.225) * T(1)/T(2))
        # Two 3-fold orbits
        a1 = T(0.470142064105115)
        push!(bary, (T(1) - T(2)*a1, a1, a1))
        push!(bary, (a1, T(1) - T(2)*a1, a1))
        push!(bary, (a1, a1, T(1) - T(2)*a1))
        w1 = T(0.132394152788506) * T(1)/T(2)
        push!(weights, w1); push!(weights, w1); push!(weights, w1)

        a2 = T(0.101286507323456)
        push!(bary, (T(1) - T(2)*a2, a2, a2))
        push!(bary, (a2, T(1) - T(2)*a2, a2))
        push!(bary, (a2, a2, T(1) - T(2)*a2))
        w2 = T(0.125939180544827) * T(1)/T(2)
        push!(weights, w2); push!(weights, w2); push!(weights, w2)

        return (bary, weights)
    elseif order == 6
        # 12-point rule, Dunavant Table I order 6
        bary = NTuple{3, T}[]
        weights = T[]
        # 3-fold orbit 1
        a1 = T(0.063089014491502)
        push!(bary, (T(1) - T(2)*a1, a1, a1))
        push!(bary, (a1, T(1) - T(2)*a1, a1))
        push!(bary, (a1, a1, T(1) - T(2)*a1))
        w1 = T(0.050844906370207) * T(1)/T(2)
        for _ in 1:3; push!(weights, w1); end
        # 3-fold orbit 2
        a2 = T(0.249286745170910)
        push!(bary, (T(1) - T(2)*a2, a2, a2))
        push!(bary, (a2, T(1) - T(2)*a2, a2))
        push!(bary, (a2, a2, T(1) - T(2)*a2))
        w2 = T(0.116786275726379) * T(1)/T(2)
        for _ in 1:3; push!(weights, w2); end
        # 6-fold orbit (S₃ orbit, no symmetry constraint)
        a3 = T(0.310352451033785)
        b3 = T(0.053145049844816)
        c3 = T(1) - a3 - b3
        # All 6 permutations
        for perm in [(a3, b3, c3), (a3, c3, b3), (b3, a3, c3),
                     (b3, c3, a3), (c3, a3, b3), (c3, b3, a3)]
            push!(bary, perm)
        end
        w3 = T(0.082851075618374) * T(1)/T(2)
        for _ in 1:6; push!(weights, w3); end

        return (bary, weights)
    elseif order == 7
        # 13-point rule, Dunavant Table I order 7
        bary = NTuple{3, T}[]
        weights = T[]
        # Centroid
        push!(bary, (T(1)/T(3), T(1)/T(3), T(1)/T(3)))
        push!(weights, T(-0.149570044467682) * T(1)/T(2))
        # 3-fold orbit 1
        a1 = T(0.260345966079040)
        push!(bary, (T(1) - T(2)*a1, a1, a1))
        push!(bary, (a1, T(1) - T(2)*a1, a1))
        push!(bary, (a1, a1, T(1) - T(2)*a1))
        w1 = T(0.175615257433208) * T(1)/T(2)
        for _ in 1:3; push!(weights, w1); end
        # 3-fold orbit 2
        a2 = T(0.065130102902216)
        push!(bary, (T(1) - T(2)*a2, a2, a2))
        push!(bary, (a2, T(1) - T(2)*a2, a2))
        push!(bary, (a2, a2, T(1) - T(2)*a2))
        w2 = T(0.053347235608839) * T(1)/T(2)
        for _ in 1:3; push!(weights, w2); end
        # 6-fold orbit
        a3 = T(0.638444188569809)
        b3 = T(0.312865496004875)
        c3 = T(1) - a3 - b3
        for perm in [(a3, b3, c3), (a3, c3, b3), (b3, a3, c3),
                     (b3, c3, a3), (c3, a3, b3), (c3, b3, a3)]
            push!(bary, perm)
        end
        w3 = T(0.077113760890257) * T(1)/T(2)
        for _ in 1:6; push!(weights, w3); end

        return (bary, weights)
    else
        throw(ArgumentError("Triangle quadrature only supported up to order 7; got $order"))
    end
end

# ============================================================================
# 3D Tetrahedron: symmetric rules
# ============================================================================
#
# Reference tetrahedron: {(x, y, z) : x ≥ 0, y ≥ 0, z ≥ 0, x + y + z ≤ 1}, vol = 1/6.
# Barycentric: (λ_0, λ_1, λ_2, λ_3) with sum 1.
# Cartesian: (x, y, z) = (λ_1, λ_2, λ_3).

"""
    gauss_quadrature_tetrahedron(order::Int; T=Float64) :: QuadRule{3, T}

Symmetric quadrature on the reference tetrahedron, exact for total degree
≤ `order`. Supported orders: 1, 2, 3.
"""
function gauss_quadrature_tetrahedron(order::Int; T::Type=Float64)
    bary_pts, weights = _tetrahedron_rule(order, T)
    points = [(p[2], p[3], p[4]) for p in bary_pts]
    return QuadRule{3, T}(points, weights)
end

function _tetrahedron_rule(order::Int, ::Type{T}) where T
    vol = T(1) / T(6)
    if order <= 1
        return ([(T(1)/T(4), T(1)/T(4), T(1)/T(4), T(1)/T(4))], [vol])
    elseif order == 2
        # 4-point rule (Keast 1986): vertices at α = (5 - sqrt(5)) / 20, others = (5 + 3 sqrt(5)) / 20
        a = (T(5) - sqrt(T(5))) / T(20)
        b = (T(5) + T(3) * sqrt(T(5))) / T(20)
        bary = [
            (b, a, a, a),
            (a, b, a, a),
            (a, a, b, a),
            (a, a, a, b),
        ]
        w = vol / T(4)
        return (bary, fill(w, 4))
    elseif order == 3
        # 5-point rule (Keast 1986): centroid (negative wt) + 4 corner-shifted points
        bary = NTuple{4, T}[]
        weights = T[]
        push!(bary, (T(1)/T(4), T(1)/T(4), T(1)/T(4), T(1)/T(4)))
        push!(weights, T(-4)/T(5) * vol)
        # 4-fold orbit at α = 1/2, β = 1/6
        a = T(1)/T(2)
        b = T(1)/T(6)
        for perm in [(a, b, b, b), (b, a, b, b), (b, b, a, b), (b, b, b, a)]
            push!(bary, perm)
        end
        w = T(9)/T(20) * vol
        for _ in 1:4
            push!(weights, w)
        end
        return (bary, weights)
    else
        throw(ArgumentError("Tetrahedron quadrature only supported up to order 3; got $order"))
    end
end

# ============================================================================
# Polynomial moment integration over a sub-interval
# ============================================================================

"""
    integrate_polynomial_on_subinterval(coeffs, basis, parent::Interval,
                                          sub::Interval, quad::QuadRule{1})

Integrate a polynomial defined in the *reference frame of `parent`* over a
sub-interval `sub ⊆ parent`. Returns the integral value.

The polynomial coefficients `coeffs` are interpreted in the basis `basis`
on the unit interval [0, 1] (the canonical reference domain). The polynomial
is evaluated by mapping each quadrature point of `quad` (which lives on
[0, 1]) to a point in `sub`, then mapping that into the reference frame of
`parent` (also [0, 1]) for evaluation. The integral is scaled by
`length(sub)` (the sub-interval's physical length) so the result is the
integral of the polynomial-as-physical-density over `sub`.

This is the 1D specialization of "integrate a polynomial moment over an
overlap polytope". Higher-dimensional analogues (intersection of a triangle
with a quad cell, etc.) live in the companion r3d.jl package.

# Arguments

- `coeffs` — polynomial coefficients on the unit interval [0,1] in `basis`
  (length must equal `n_coeffs(basis)`).
- `basis::AbstractBasis{1, P}` — polynomial basis on [0, 1].
- `parent::Interval` — the physical interval the polynomial represents.
- `sub::Interval` — the sub-interval to integrate over (must satisfy
  `sub ⊆ parent`; not enforced).
- `quad::QuadRule{1, T}` — quadrature rule on [0, 1] with sufficient order.

# Returns

`integral` — the value `∫_{sub} p(ξ(x)) dx` where `ξ(x)` is the mapping
from physical x ∈ parent to reference coordinate in [0,1].

# Notes

- If `sub` is empty (disjoint from parent or zero length), returns `zero(T)`.
- Quadrature must be exact for the polynomial degree (use a rule with
  `n_pts >= ceil((P+1)/2)`).
"""
function integrate_polynomial_on_subinterval(
        coeffs, basis::AbstractBasis{1, P},
        parent::Interval{T}, sub::Interval{T},
        quad::QuadRule{1, T}) where {P, T}
    is_empty(sub) && return zero(T)
    nc = n_coeffs(basis)
    length(coeffs) == nc || throw(DimensionMismatch(
        "coeffs has $(length(coeffs)) entries, expected $nc for basis"))
    parent_len = parent.hi - parent.lo
    parent_len > 0 || throw(ArgumentError("parent interval must have positive length"))
    sub_len = sub.hi - sub.lo

    # For each quadrature point t ∈ [0,1]:
    #   physical x = sub.lo + t * sub_len
    #   reference ξ in parent's frame = (x - parent.lo) / parent_len
    # Result is sum_k w_k * p(ξ_k) * sub_len
    s = zero(T)
    @inbounds for k in 1:length(quad.points)
        t = quad.points[k][1]
        x = sub.lo + t * sub_len
        ξ = (x - parent.lo) / parent_len
        s += quad.weights[k] * evaluate(basis, coeffs, (ξ,))
    end
    return s * sub_len
end

# ============================================================================
# Generic action-error indicator (basis-agnostic)
# ============================================================================

"""
    action_error_l2(action_p, action_pplus1, quad::QuadRule{D, T};
                    el_residual = zero(T))

Compute a local action-error indicator: the square root of the integrated
squared difference between two reconstructions, plus an EL-residual penalty.

This is the framework's generic AMR indicator for high-order schemes:
    ∆S = sqrt( ∫ (L_{p+1}(q) - L_p(q))² dq + el_residual² )

where `L_p` is the action density (or any per-cell local quantity) at the
current reconstruction order, and `L_{p+1}` is the same quantity at one
higher order. Cells with large `∆S` are refinement candidates.

`action_p` and `action_pplus1` are functions of the reference-domain point
(NTuple{D, T}) returning a scalar. They are evaluated at the quadrature
nodes of `quad`, and the result is integrated over the reference domain.

This function knows nothing about polynomial bases or fields — it is a pure
quadrature-of-squared-difference primitive. Convenience wrappers that take
`PolynomialView`s live in the Storage module.
"""
function action_error_l2(action_p, action_pplus1, quad::QuadRule{D, T};
                          el_residual::Real = zero(T)) where {D, T}
    s = zero(T)
    @inbounds for k in 1:length(quad.points)
        pt = quad.points[k]
        d = action_pplus1(pt) - action_p(pt)
        s += quad.weights[k] * d * d
    end
    res = T(el_residual)
    return sqrt(s + res * res)
end

end # module Quadrature
