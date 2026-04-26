"""
    PolynomialFieldSet{L, B, Names, ScalarTypes, Storage}

A set of named polynomial-valued fields stored over `n` mesh elements,
with each polynomial expressed in basis `B` (an `AbstractBasis{D, P}`).

The number of coefficients per field per element is fixed by the basis
(via `n_coeffs(B)`). Fields can have different scalar types but share
the same basis.

`L` controls the memory layout exactly as for `FieldSet`:

- `SoA`: each field stores its `n_coeffs * n` coefficients as one
  contiguous Vector. Within a field, coefficient `k` of element `i` is
  at offset `(i-1) * n_coeffs + k` (element-major; coefficients of a
  single element are contiguous, suitable for fast per-element evaluation).

- `AoS`: each element stores all of its polynomial coefficients (across
  all fields) as inline NTuples in a single Vector of NamedTuples.

- `Blocked{B}`: groups of B elements stored together (within-block
  layout follows the inner layout, defaulting to SoA).

# Access

```julia
basis = MonomialBasis{1, 3}()
fields = allocate_polynomial_fields(SoA(), basis, 100;
                                    density=Float64, momentum=Float64)

poly = fields.density[i]              # PolynomialView for cell i
val_at_q = poly(q_point)              # evaluate at point
g = gradient_at(poly, q_point)        # gradient at point
poly[k]                               # k-th coefficient
poly[k] = new_value                   # set k-th coefficient

fields.density[i] = (c0, c1, c2, c3)  # bulk assign coefficients
```
"""
struct PolynomialFieldSet{L<:AbstractLayout, B, Names, ScalarTypes, Storage}
    n::Int
    basis::B
    storage::Storage
end

"""
    allocate_polynomial_fields(layout, basis, n_elements; field=ScalarType, ...)

Allocate a PolynomialFieldSet. Each keyword's value is the scalar type of
that field's polynomial coefficients.
"""
function allocate_polynomial_fields(layout::L, basis::B, n::Integer;
                                     field_kwargs...) where {L<:AbstractLayout, B}
    names = Tuple(keys(field_kwargs))
    types = Tuple(values(field_kwargs))
    nc = n_coeffs(basis)
    storage = _make_polynomial_storage(L, nc, Int(n), names, types)
    Names = names
    ScalarTypes = Tuple{types...}
    return PolynomialFieldSet{L, B, Names, ScalarTypes, typeof(storage)}(Int(n), basis, storage)
end

function allocate_polynomial_fields(::Type{L}, basis::B, n::Integer;
                                     field_kwargs...) where {L<:AbstractLayout, B}
    return allocate_polynomial_fields(L(), basis, n; field_kwargs...)
end

# ============================================================================
# Storage construction per layout
# ============================================================================

# SoA: NamedTuple of contiguous Vectors of length nc * n
function _make_polynomial_storage(::Type{SoA}, nc::Int, n::Int, names::Tuple, types::Tuple)
    arrays = map(T -> Vector{T}(undef, nc * n), types)
    return NamedTuple{names}(arrays)
end

# AoS: single Vector of NamedTuple where each field is an NTuple{nc, T}
function _make_polynomial_storage(::Type{AoS}, nc::Int, n::Int, names::Tuple, types::Tuple)
    inner_types = map(T -> NTuple{nc, T}, types)
    ElType = NamedTuple{names, Tuple{inner_types...}}
    return Vector{ElType}(undef, n)
end

# Blocked: Vector of inner-storage blocks
function _make_polynomial_storage(::Type{Blocked{B, IL}}, nc::Int, n::Int,
                                   names::Tuple, types::Tuple) where {B, IL}
    n_full_blocks = n ÷ B
    remainder = n % B
    n_blocks = n_full_blocks + (remainder > 0 ? 1 : 0)

    BlockType = typeof(_make_polynomial_storage(IL, nc, B, names, types))
    blocks = Vector{BlockType}(undef, n_blocks)
    for b in 1:n_full_blocks
        blocks[b] = _make_polynomial_storage(IL, nc, B, names, types)
    end
    if remainder > 0
        blocks[end] = _make_polynomial_storage(IL, nc, remainder, names, types)
    end
    return blocks
end

# ============================================================================
# Queries
# ============================================================================

@inline n_elements(pfs::PolynomialFieldSet) = pfs.n
@inline field_names(::PolynomialFieldSet{L, B, Names, ST, S}) where {L, B, Names, ST, S} = Names
@inline basis_of(pfs::PolynomialFieldSet) = pfs.basis
@inline n_coeffs_per_element(pfs::PolynomialFieldSet) = n_coeffs(pfs.basis)
@inline _layout_type_poly(::PolynomialFieldSet{L, B, N, ST, S}) where {L, B, N, ST, S} = L

# ============================================================================
# Property access
# ============================================================================

const _POLY_FIELDSET_INTERNAL = (:n, :basis, :storage)

function Base.getproperty(pfs::PolynomialFieldSet{L, B, Names, ST, S},
                          name::Symbol) where {L, B, Names, ST, S}
    if name in _POLY_FIELDSET_INTERNAL
        return getfield(pfs, name)
    end
    if name in Names
        return PolynomialFieldView{typeof(pfs), name}(pfs)
    end
    throw(KeyError(name))
end

function Base.setproperty!(pfs::PolynomialFieldSet, name::Symbol, value)
    if name in _POLY_FIELDSET_INTERNAL
        setfield!(pfs, name, value)
    else
        throw(ArgumentError("PolynomialFieldSet: cannot set field :$name as a whole; " *
                              "use indexed assignment `fields.$name[i] = coeffs` instead."))
    end
end

function Base.propertynames(::PolynomialFieldSet{L, B, Names, ST, S}) where {L, B, Names, ST, S}
    return (Names..., _POLY_FIELDSET_INTERNAL...)
end

# ============================================================================
# PolynomialFieldView and PolynomialView
# ============================================================================

"""
    PolynomialFieldView{PFS, name}

The view returned by `pfs.fieldname`. Indexing with an integer gives a
`PolynomialView` for that element.
"""
struct PolynomialFieldView{PFS, name}
    pfs::PFS
end

@inline Base.length(v::PolynomialFieldView) = v.pfs.n
@inline Base.eachindex(v::PolynomialFieldView) = 1:v.pfs.n
@inline Base.size(v::PolynomialFieldView) = (v.pfs.n,)
@inline Base.firstindex(::PolynomialFieldView) = 1
@inline Base.lastindex(v::PolynomialFieldView) = v.pfs.n

"""
    PolynomialView{PFS, name, B}

Per-element polynomial view. Supports:

- `poly(point)` — evaluate the polynomial
- `gradient_at(poly, point)` — gradient
- `poly[k]` — get/set k-th coefficient
- `length(poly)` — number of coefficients
- iteration over coefficients
"""
struct PolynomialView{PFS, name, B}
    pfs::PFS
    i::Int
end

@inline function Base.getindex(fv::PolynomialFieldView{PFS, name}, i::Integer) where {PFS, name}
    return PolynomialView{PFS, name, typeof(fv.pfs.basis)}(fv.pfs, Int(i))
end

@inline function Base.setindex!(fv::PolynomialFieldView{PFS, name}, coeffs, i::Integer) where {PFS, name}
    nc = n_coeffs(fv.pfs.basis)
    length(coeffs) == nc || throw(DimensionMismatch("expected $nc coefficients, got $(length(coeffs))"))
    @inbounds for k in 1:nc
        _set_poly_coeff!(fv.pfs, Val(name), Int(i), k, coeffs[k])
    end
    return coeffs
end

@inline Base.length(::PolynomialView{PFS, name, B}) where {PFS, name, B} = n_coeffs(B())

@inline function Base.getindex(pv::PolynomialView{PFS, name, B}, k::Integer) where {PFS, name, B}
    return _get_poly_coeff(pv.pfs, Val(name), pv.i, Int(k))
end

@inline function Base.setindex!(pv::PolynomialView{PFS, name, B}, value, k::Integer) where {PFS, name, B}
    _set_poly_coeff!(pv.pfs, Val(name), pv.i, Int(k), value)
    return value
end

@inline Base.iterate(pv::PolynomialView, k::Int=1) = k > length(pv) ? nothing : (pv[k], k+1)

# Evaluation: `poly(point)`
@inline function (pv::PolynomialView{PFS, name, B})(point) where {PFS, name, B}
    nc = n_coeffs(B())
    coeffs = ntuple(k -> pv[k], nc)
    return evaluate(B(), coeffs, point)
end

"""
    gradient_at(poly::PolynomialView, point)

Gradient of the polynomial at `point` (in reference coordinates).
"""
@inline function gradient_at(pv::PolynomialView{PFS, name, B}, point) where {PFS, name, B}
    nc = n_coeffs(B())
    coeffs = ntuple(k -> pv[k], nc)
    return gradient(B(), coeffs, point)
end

function Base.collect(pv::PolynomialView)
    nc = length(pv)
    [pv[k] for k in 1:nc]
end

# ============================================================================
# Layout-specific accessors for polynomial coefficients
# ============================================================================

# All accessors go through the PFS so we have access to the basis (and hence nc).

@inline function _get_poly_coeff(pfs::PolynomialFieldSet, ::Val{name},
                                  i::Int, k::Int) where name
    L = _layout_type_poly(pfs)
    nc = n_coeffs(pfs.basis)
    return _get_poly_coeff_impl(pfs.storage, L, Val(name), i, k, nc)
end

@inline function _set_poly_coeff!(pfs::PolynomialFieldSet, ::Val{name},
                                   i::Int, k::Int, value) where name
    L = _layout_type_poly(pfs)
    nc = n_coeffs(pfs.basis)
    _set_poly_coeff_impl!(pfs.storage, L, Val(name), i, k, nc, value)
    return value
end

# SoA
@inline function _get_poly_coeff_impl(storage::NamedTuple, ::Type{SoA},
                                       ::Val{name}, i::Int, k::Int, nc::Int) where name
    return getfield(storage, name)[(i-1)*nc + k]
end
@inline function _set_poly_coeff_impl!(storage::NamedTuple, ::Type{SoA},
                                        ::Val{name}, i::Int, k::Int, nc::Int, value) where name
    getfield(storage, name)[(i-1)*nc + k] = value
    return value
end

# AoS
@inline function _get_poly_coeff_impl(storage::Vector, ::Type{AoS},
                                       ::Val{name}, i::Int, k::Int, ::Int) where name
    return getfield(storage[i], name)[k]
end
@inline function _set_poly_coeff_impl!(storage::Vector, ::Type{AoS},
                                        ::Val{name}, i::Int, k::Int, ::Int, value) where name
    elt = storage[i]
    inner = getfield(elt, name)
    new_inner = Base.setindex(inner, value, k)
    storage[i] = merge(elt, NamedTuple{(name,)}((new_inner,)))
    return value
end

# Blocked: dispatch into the inner storage. Block size B is a type param;
# Blocked{B} enforces B being a power of 2, so use bit shifts.
@inline function _get_poly_coeff_impl(storage::Vector, ::Type{Blocked{B, IL}},
                                       ::Val{name}, i::Int, k::Int, nc::Int) where {B, IL, name}
    block_idx = ((i - 1) >> trailing_zeros(B)) + 1
    within_block = ((i - 1) & (B - 1)) + 1
    return _get_poly_coeff_impl(storage[block_idx], IL, Val(name), within_block, k, nc)
end
@inline function _set_poly_coeff_impl!(storage::Vector, ::Type{Blocked{B, IL}},
                                        ::Val{name}, i::Int, k::Int, nc::Int, value) where {B, IL, name}
    block_idx = ((i - 1) >> trailing_zeros(B)) + 1
    within_block = ((i - 1) & (B - 1)) + 1
    _set_poly_coeff_impl!(storage[block_idx], IL, Val(name), within_block, k, nc, value)
    return value
end

# ============================================================================
# Show
# ============================================================================

function Base.show(io::IO, pfs::PolynomialFieldSet{L, B, Names, ST, S}) where {L, B, Names, ST, S}
    print(io, "PolynomialFieldSet{", L, ", ", B, "}(n=$(pfs.n)) with fields: ")
    for (k, name) in enumerate(Names)
        if k > 1; print(io, ", "); end
        print(io, "$name::", fieldtype(ST, k))
    end
end

function Base.show(io::IO, pv::PolynomialView{PFS, name, B}) where {PFS, name, B}
    nc = n_coeffs(B())
    print(io, "PolynomialView{$name, basis=$B}(coeffs=[")
    for k in 1:nc
        k > 1 && print(io, ", ")
        print(io, pv[k])
    end
    print(io, "])")
end

# ============================================================================
# Polynomial-aware action-error indicator
# ============================================================================

"""
    is_strictly_positive(field::PolynomialFieldView; atol = 0)
        -> (positive::Bool, offending::Union{Nothing, Tuple{Int, NTuple{N, Int}}})

Per-cell strict-positivity certificate for a polynomial field stored in
Bernstein basis. Iterates cells in order; on the first failure returns
`(false, (cell_index, offending_multi_index))`. Otherwise returns
`(true, nothing)`.

Throws `ArgumentError` if the underlying basis is not a `BernsteinBasis`
(e.g. `MonomialBasis` does not have the convex-hull property, so the
certificate would be unsound).

Returning `false` does NOT imply non-positivity: the certificate is
sufficient but not necessary (sharper under degree elevation).
"""
function is_strictly_positive(field::PolynomialFieldView{PFS, name};
                              atol = nothing) where {PFS, name}
    pfs = field.pfs
    basis = pfs.basis
    basis isa BernsteinBasis ||
        throw(ArgumentError("is_strictly_positive requires a BernsteinBasis " *
                             "(convex-hull property); got $(typeof(basis))"))
    nc = n_coeffs(basis)
    @inbounds for i in 1:pfs.n
        # Materialize coefficients of cell i as an NTuple for the certificate.
        coeffs = ntuple(k -> _get_poly_coeff(pfs, Val(name), i, k), nc)
        a = atol === nothing ? zero(eltype(coeffs)) : atol
        positive, offending = bernstein_positivity_certificate(coeffs, basis; atol=a)
        if !positive
            return (false, (i, offending))
        end
    end
    return (true, nothing)
end

"""
    is_strictly_positive(pfs::PolynomialFieldSet, name::Symbol; atol = 0)

Convenience: forward to the field-view variant by extracting `pfs.<name>`.
"""
function is_strictly_positive(pfs::PolynomialFieldSet, name::Symbol; atol = nothing)
    return is_strictly_positive(getproperty(pfs, name); atol=atol)
end

"""
    polynomial_action_error(poly_p, poly_pplus1, quad::QuadRule;
                              transform = identity, el_residual = 0)

Compute the local action-error indicator for one element by comparing two
reconstructions of the same field at consecutive polynomial orders.

`poly_p` and `poly_pplus1` are anything callable as `poly(point) -> value`
(typically `PolynomialView`s for the same field reconstructed at orders p
and p+1, but any pair of point-callable objects works). `transform` is an
optional function `(value, point) -> scalar` mapping field values into the
action-density scalar; the default `identity` compares the polynomial values
directly. `el_residual` is the discretization-residual penalty term.

Returns the scalar indicator
```
    sqrt( ∫_ref (transform(poly_pplus1(q), q) - transform(poly_p(q), q))² dq
         + el_residual² )
```
which is the framework's generic AMR primary for high-order schemes.

This is a thin convenience over `Quadrature.action_error_l2`.
"""
function polynomial_action_error(poly_p, poly_pplus1, quad::QuadRule;
                                   transform = nothing,
                                   el_residual::Real = 0)
    if transform === nothing
        return action_error_l2(poly_p, poly_pplus1, quad; el_residual=el_residual)
    else
        # Wrap each callable to apply the transform at every quadrature point.
        f_p   = pt -> transform(poly_p(pt),   pt)
        f_pp  = pt -> transform(poly_pplus1(pt), pt)
        return action_error_l2(f_p, f_pp, quad; el_residual=el_residual)
    end
end

"""
    polynomial_action_error_per_element(field_p::PolynomialFieldView,
                                          field_pplus1::PolynomialFieldView,
                                          quad::QuadRule;
                                          transform = nothing,
                                          el_residual = nothing) -> Vector

Compute the action-error indicator for every element in a polynomial field,
producing a vector suitable for passing to `refine_by_indicator!`.

`field_p` and `field_pplus1` must be `PolynomialFieldView`s of the same
length (same number of elements). They typically represent the same physical
field stored at orders p and p+1.

# Arguments

- `field_p`, `field_pplus1` — point-callable per-element views (returned by
  `pfs.fieldname`). Each `field[i]` is a `PolynomialView` callable at points.
- `quad` — quadrature rule on the reference domain shared by both bases.
- `transform` — optional `(value, point) -> scalar` transform; defaults to
  identity (compare polynomial values directly).
- `el_residual` — either `nothing` (no per-element residual penalty) or
  a vector of length `n_elements` with per-element EL-residual proxies.

# Returns

A `Vector{Float64}` of length `length(field_p)` with the per-element
indicator. Pass it to `refine_by_indicator!(mesh, indicator; refine_threshold=...)`.
"""
function polynomial_action_error_per_element(field_p::PolynomialFieldView,
                                               field_pplus1::PolynomialFieldView,
                                               quad::QuadRule;
                                               transform = nothing,
                                               el_residual = nothing)
    n = length(field_p)
    length(field_pplus1) == n || throw(DimensionMismatch(
        "field_p and field_pplus1 have different element counts: $n vs $(length(field_pplus1))"))
    if el_residual !== nothing
        length(el_residual) == n || throw(DimensionMismatch(
            "el_residual has length $(length(el_residual)), expected $n"))
    end
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        res = el_residual === nothing ? 0.0 : Float64(el_residual[i])
        out[i] = polynomial_action_error(field_p[i], field_pplus1[i], quad;
                                          transform=transform, el_residual=res)
    end
    return out
end
