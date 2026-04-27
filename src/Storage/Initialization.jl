"""
    Initialization

L² projection helpers that initialize a `PolynomialFieldSet` from an
analytical function. Per-cell projection: for each cell, the routine
solves the local mass system

    M c = b,    M[α, β] = ∫_ref φ_α(ξ) φ_β(ξ) dξ,
                b[α]    = ∫_ref f(x(ξ)) φ_α(ξ) dξ

where the φ's are the basis functions of the field, the integrals are
on the cell's reference domain ([0, 1]^D for axis-aligned cells, the
unit simplex for simplicial cells), and `x(ξ)` is the affine map from
reference to physical coordinates. The constant volume Jacobian cancels
between `M` and `b`, so it's omitted on both sides.

Two methods are provided:

- `init_field_from!(field, frame::EulerianFrame, f; quadrature_order)` —
  projects onto every cell of a `HierarchicalMesh` (with a physical frame).
- `init_field_from!(field, mesh::SimplicialMesh, f; quadrature_order)` —
  projects onto every simplex of a `SimplicialMesh`.

`f(x::NTuple{D, T})::T` is called with `x` in **physical** coordinates.
"""
module Initialization

import LinearAlgebra
using LinearAlgebra: cholesky, Symmetric, factorize

using OhMyThreads.TaskLocalValues: TaskLocalValue
using ..Bases: AbstractBasis, n_coeffs, evaluate
using ..Quadrature: QuadRule, n_quad_points,
    gauss_quadrature_interval, gauss_quadrature_quad, gauss_quadrature_cube,
    gauss_quadrature_triangle, gauss_quadrature_tetrahedron
using ..Storage: PolynomialFieldSet, basis_of, n_elements, n_coeffs_per_element
using ..Mesh: SimplicialMesh, n_simplices, simplex_vertex_positions
import ..Mesh
using ..Overlap: EulerianFrame, cell_physical_box
using ..Threading: AbstractParallelBackend, Sequential, OhMyThreadsBackend,
                   default_backend, parallel_foreach

export init_field_from!

# ============================================================================
# Reference-domain quadrature dispatch
# ============================================================================

# Axis-aligned (cube) reference domain: [0, 1]^D.
# `quadrature_order = 2P + 1` is the desired exactness; the tensor-product
# n-point Gauss-Legendre rule is exact up to degree 2n - 1, so we need
# n = ceil((quadrature_order + 1) / 2).
@inline function _aabb_quadrature(::Val{1}, order::Int, ::Type{T}) where T
    n = max(1, cld(order + 1, 2))
    return gauss_quadrature_interval(n; T = T)
end
@inline function _aabb_quadrature(::Val{2}, order::Int, ::Type{T}) where T
    n = max(1, cld(order + 1, 2))
    return gauss_quadrature_quad(n; T = T)
end
@inline function _aabb_quadrature(::Val{3}, order::Int, ::Type{T}) where T
    n = max(1, cld(order + 1, 2))
    return gauss_quadrature_cube(n; T = T)
end

# Simplex reference domain: unit D-simplex.
@inline function _simplex_quadrature(::Val{1}, order::Int, ::Type{T}) where T
    n = max(1, cld(order + 1, 2))
    return gauss_quadrature_interval(n; T = T)
end
@inline function _simplex_quadrature(::Val{2}, order::Int, ::Type{T}) where T
    return gauss_quadrature_triangle(order; T = T)
end
@inline function _simplex_quadrature(::Val{3}, order::Int, ::Type{T}) where T
    return gauss_quadrature_tetrahedron(order; T = T)
end

# ============================================================================
# Per-cell projection kernel
# ============================================================================

# Build the basis-Gram matrix M[α, β] = ∫_ref φ_α(ξ) φ_β(ξ) dξ via quadrature.
# The integral is on the reference domain implied by `quad`. We use the basis's
# own `evaluate` to get φ_α(ξ) by selecting a unit coefficient vector.
function _basis_mass_matrix(basis::AbstractBasis{D, P},
                            quad::QuadRule{D, T}) where {D, P, T}
    nc = n_coeffs(basis)
    M = zeros(T, nc, nc)
    # Per quadrature node, compute the vector of basis values φ(ξ) by
    # repeatedly evaluating with unit coefficient vectors. For each node we
    # then accumulate w * φ φ^T into M.
    phi = Vector{T}(undef, nc)
    @inbounds for k in 1:n_quad_points(quad)
        ξ = quad.points[k]
        w = quad.weights[k]
        for α in 1:nc
            phi[α] = _basis_value(basis, α, ξ, T)
        end
        for α in 1:nc
            phi_α = phi[α]
            for β in 1:nc
                M[α, β] += w * phi_α * phi[β]
            end
        end
    end
    return M
end

# Evaluate the α-th basis function at ξ by passing a unit coefficient vector.
@inline function _basis_value(basis::AbstractBasis{D, P}, α::Int, ξ, ::Type{T}) where {D, P, T}
    nc = n_coeffs(basis)
    coeffs = ntuple(k -> k == α ? one(T) : zero(T), nc)
    return T(evaluate(basis, coeffs, ξ))
end

# Build the per-cell RHS b[α] = ∫_ref f(x(ξ)) φ_α(ξ) dξ.
function _assemble_rhs!(b::Vector{T}, f, affine_map,
                        basis::AbstractBasis{D, P},
                        quad::QuadRule{D, T}) where {D, P, T}
    nc = n_coeffs(basis)
    fill!(b, zero(T))
    @inbounds for k in 1:n_quad_points(quad)
        ξ = quad.points[k]
        w = quad.weights[k]
        x = affine_map(ξ)
        fx = T(f(x))
        for α in 1:nc
            b[α] += w * fx * _basis_value(basis, α, ξ, T)
        end
    end
    return b
end

# Solve M c = b in place (M may be modified). Mass matrices are symmetric
# positive-definite for any non-degenerate basis, so Cholesky is the right
# tool; fall back to a generic factorization if Cholesky fails (e.g. for
# weakly-conditioned bases at high P).
function _solve_mass_system(M::Matrix{T}, b::Vector{T}) where T
    try
        F = cholesky(Symmetric(M))
        return F \ b
    catch err
        err isa LinearAlgebra.PosDefException || rethrow()
        return factorize(M) \ b
    end
end

# Assign a coefficient vector to one cell of `field`.
@inline function _assign_cell!(field::PolynomialFieldSet, cell_idx::Int, coeffs::AbstractVector)
    nc = n_coeffs_per_element(field)
    names = _field_names(field)
    @inbounds for name in names
        view = getproperty(field, name)
        # Materialize as a tuple of the right length for the bulk setter.
        view[cell_idx] = ntuple(k -> coeffs[k], nc)
    end
    return field
end

@inline _field_names(::PolynomialFieldSet{L, B, Names, ST, S}) where {L, B, Names, ST, S} = Names

# ============================================================================
# Public API
# ============================================================================

"""
    init_field_from!(field::PolynomialFieldSet{D, P, T, B},
                       frame::EulerianFrame{D, T}, f;
                       quadrature_order::Int = 2P + 1) -> field

Project the analytical function `f` onto the polynomial basis of `field`,
cell by cell, by L² projection on the reference cube `[0, 1]^D`. Writes
the resulting coefficients into `field` in place and returns it.

`f(x::NTuple{D, T})::T` is called with `x` in **physical** coordinates.

Default `quadrature_order = 2P + 1` exactly integrates polynomials up to
degree `2P` on the reference cube — sufficient for L² inner products of
two degree-P basis functions. Pass a larger value for higher integrand
exactness when `f` is a more complicated function (e.g. trigonometric).

Every named field of `field` is filled with the same coefficient vector;
`init_field_from!` is a single-target helper. To initialize multiple
fields with different functions, call this routine on per-field views or
project a tuple-valued function and split afterwards.
"""
function init_field_from!(field::PolynomialFieldSet,
                           frame::EulerianFrame{D, T}, f;
                           quadrature_order::Int = 2 * _basis_degree(basis_of(field)) + 1,
                           backend::AbstractParallelBackend = default_backend()) where {D, T}
    basis = basis_of(field)
    _check_dim(basis, D)
    quad = _aabb_quadrature(Val(D), quadrature_order, T)
    M = _basis_mass_matrix(basis, quad)
    nc = n_coeffs(basis)

    n = n_elements(field)
    # Pre-build the lazy per-cell caches on the calling thread before
    # fanning out — the cache machinery is not thread-safe under
    # concurrent first-access, so we touch it serially first.
    Mesh.ensure_caches!(frame.mesh)
    # Per-task scratch: each task gets its own RHS vector + mass-matrix
    # working copy so the per-cell allocations happen once per task on
    # first touch, not once per cell.
    b_tls = TaskLocalValue{Vector{T}}(() -> Vector{T}(undef, nc))
    Mwork_tls = TaskLocalValue{Matrix{T}}(() -> Matrix{T}(undef, nc, nc))
    parallel_foreach(backend, function (i)
        lo, hi = cell_physical_box(frame, i)
        # Capture lo/hi as locals so the closure is non-allocating.
        affine = let lo = lo, hi = hi
            ξ -> ntuple(d -> lo[d] + (hi[d] - lo[d]) * T(ξ[d]), Val(D))
        end
        b = b_tls[]
        Mwork = Mwork_tls[]
        _assemble_rhs!(b, f, affine, basis, quad)
        copyto!(Mwork, M)
        c = _solve_mass_system(Mwork, b)
        _assign_cell!(field, i, c)
        return
    end, 1:n)
    return field
end

"""
    init_field_from!(field::PolynomialFieldSet{D, P, T, B},
                       mesh::SimplicialMesh{D, T}, f;
                       quadrature_order::Int = 2P + 1) -> field

Project `f` onto the polynomial basis of `field`, simplex by simplex, by
L² projection on the unit D-simplex. The affine map ξ ↦ x for simplex `s`
uses the simplex's *current* vertex positions (`positions`, not
`reference_positions`):

    x = v₁ + Σ_k (v_{k+1} − v₁) · ξ_k

`f(x)::T` is called with `x` in physical coordinates.

For D = 2 / D = 3 the underlying triangle / tetrahedron quadrature rules
support up to order 7 / 3 respectively; `quadrature_order` is clamped
internally to the supported range so the default `2P + 1` is safe for all
practical degrees.
"""
function init_field_from!(field::PolynomialFieldSet,
                           mesh::SimplicialMesh{D, T}, f;
                           quadrature_order::Int = 2 * _basis_degree(basis_of(field)) + 1,
                           backend::AbstractParallelBackend = default_backend()) where {D, T}
    basis = basis_of(field)
    _check_dim(basis, D)
    order = _clamp_simplex_order(D, quadrature_order)
    quad = _simplex_quadrature(Val(D), order, T)
    M = _basis_mass_matrix(basis, quad)
    nc = n_coeffs(basis)

    ns = n_simplices(mesh)
    n_elements(field) == ns ||
        throw(DimensionMismatch("field has $(n_elements(field)) elements; mesh has $ns simplices"))

    # Per-task scratch: same pattern as the axis-aligned overload.
    b_tls = TaskLocalValue{Vector{T}}(() -> Vector{T}(undef, nc))
    Mwork_tls = TaskLocalValue{Matrix{T}}(() -> Matrix{T}(undef, nc, nc))
    parallel_foreach(backend, function (s)
        verts = simplex_vertex_positions(mesh, s)
        anchor = ntuple(d -> T(verts[1][d]), Val(D))
        edges = ntuple(k -> ntuple(d -> T(verts[k+1][d]) - anchor[d], Val(D)), Val(D))
        affine = let anchor = anchor, edges = edges
            ξ -> ntuple(d -> anchor[d] +
                              sum(edges[k][d] * T(ξ[k]) for k in 1:D),
                        Val(D))
        end
        b = b_tls[]
        Mwork = Mwork_tls[]
        _assemble_rhs!(b, f, affine, basis, quad)
        copyto!(Mwork, M)
        c = _solve_mass_system(Mwork, b)
        _assign_cell!(field, s, c)
        return
    end, 1:ns)
    return field
end

# ============================================================================
# Helpers
# ============================================================================

@inline _basis_degree(::AbstractBasis{D, P}) where {D, P} = P

@inline function _check_dim(::AbstractBasis{Dbasis, P}, Dframe::Int) where {Dbasis, P}
    Dbasis == Dframe ||
        throw(ArgumentError("basis dimension D=$Dbasis does not match frame/mesh dimension D=$Dframe"))
end

# Triangle quadrature supports orders 1..7; tetrahedron supports 1..3.
@inline function _clamp_simplex_order(D::Int, order::Int)
    order = max(1, order)
    if D == 2
        return min(order, 7)
    elseif D == 3
        return min(order, 3)
    end
    return order
end

end # module Initialization
