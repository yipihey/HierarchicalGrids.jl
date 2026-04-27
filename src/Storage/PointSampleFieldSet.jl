# ============================================================================
# PointSampleFieldSet (PR-11 — Path B for block-based AMR)
#
# Point-sample storage variant: each cell holds an N^D array of point values
# per field, sampled at the equispaced Lagrange nodes of the unit reference
# cube [0, 1]^D. Sibling to `PolynomialFieldSet` (Path A); both can be
# orchestrated through `for_each_block!` with the same kernel signature.
#
# The "basis" for these samples is a tensor-product equispaced Lagrange
# nodal basis: each axis carries `N` nodes at the points
#
#     ξ_k = (k - 1) / (N - 1)    for  k = 1, ..., N    (P + 1 = N nodes)
#
# A multi-index (k_1, ..., k_D) ∈ {1..N}^D maps to flat index
#
#     idx = k_1 + (k_2 - 1) * N + (k_3 - 1) * N^2 + ...    (column-major)
#
# This is Julia's natural column-major flattening — the same convention as
# `Array{T, D}` linear indexing — and is the convention the kernel-facing
# `bv[Val(:rho), (i, j)]` accessor uses.
#
# Layout flexibility: the layout machinery is identical to PolynomialFieldSet
# (SoA / AoS / Blocked{B}) — we just store `N^D` scalars per cell rather than
# `n_coeffs(basis)` polynomial coefficients.
# ============================================================================

"""
    PointSampleFieldSet{D, N, Names, ScalarTypes, Storage, Layout}

A set of named point-sampled fields stored over `n` mesh elements. Each
cell holds an `N^D` block of values per field, sampled at the equispaced
Lagrange nodes of the unit reference cube `[0, 1]^D`.

The basis is implicitly a tensor-product equispaced Lagrange nodal basis
of degree `P = N - 1`, with `N^D` nodes per cell. Layout flexes across
`SoA`, `AoS`, `Blocked{B}` per the existing layout machinery — same as
`PolynomialFieldSet`.

# Access

```julia
pfs = allocate_point_sample_fields(SoA(), Val(2), Val(3), 100; rho=Float64)

view = pfs.rho[i]              # PointSampleView for cell i
view[k]                        # k-th flat point value (1..N^D)
view[k] = v                    # set k-th flat point value
view[(i, j)]                   # value at multi-index (i, j) (D=2)
view((0.25, 0.5))              # interpolated value at ref point ξ
pfs.rho[i] = (v_1, ..., v_{N^D})   # bulk assign all points
```

# Convention

Multi-index `(k_1, ..., k_D)` (1-indexed) maps to flat index
`k_1 + (k_2 - 1) * N + (k_3 - 1) * N^2 + ...` (column-major, matching
Julia's `Array{T, D}`).
"""
struct PointSampleFieldSet{D, N, L<:AbstractLayout, Names, ScalarTypes, Storage}
    n::Int
    storage::Storage
end

"""
    allocate_point_sample_fields(layout, ::Val{D}, ::Val{N}, n_cells; field=Type, ...)

Allocate a PointSampleFieldSet with `N^D` point values per cell. Mirrors
`allocate_polynomial_fields`.
"""
function allocate_point_sample_fields(layout::L, ::Val{D}, ::Val{N}, n::Integer;
                                       field_kwargs...) where {L<:AbstractLayout, D, N}
    D >= 1 || throw(ArgumentError("PointSampleFieldSet: D must be ≥ 1, got $D"))
    N >= 1 || throw(ArgumentError("PointSampleFieldSet: N must be ≥ 1, got $N"))
    names = Tuple(keys(field_kwargs))
    types = Tuple(values(field_kwargs))
    npts = N^D
    storage = _make_polynomial_storage(L, npts, Int(n), names, types)
    Names = names
    ScalarTypes = Tuple{types...}
    return PointSampleFieldSet{D, N, L, Names, ScalarTypes, typeof(storage)}(
        Int(n), storage)
end

function allocate_point_sample_fields(::Type{L}, ::Val{D}, ::Val{N}, n::Integer;
                                       field_kwargs...) where {L<:AbstractLayout, D, N}
    return allocate_point_sample_fields(L(), Val(D), Val(N), n; field_kwargs...)
end

# ============================================================================
# Queries
# ============================================================================

@inline n_elements(pfs::PointSampleFieldSet) = pfs.n
@inline field_names(::PointSampleFieldSet{D, N, L, Names, ST, S}) where {D, N, L, Names, ST, S} = Names
@inline n_points_per_cell(::PointSampleFieldSet{D, N, L, Names, ST, S}) where {D, N, L, Names, ST, S} = N^D
@inline n_points_per_axis(::PointSampleFieldSet{D, N, L, Names, ST, S}) where {D, N, L, Names, ST, S} = N
@inline spatial_dim(::PointSampleFieldSet{D, N, L, Names, ST, S}) where {D, N, L, Names, ST, S} = D
@inline _layout_type_pts(::PointSampleFieldSet{D, N, L, Names, ST, S}) where {D, N, L, Names, ST, S} = L

# ============================================================================
# Multi-index ↔ flat conversions (column-major / Julia-natural)
# ============================================================================

# (k_1, ..., k_D) -> k_1 + (k_2-1)*N + (k_3-1)*N^2 + ...
@inline function point_multi_to_flat(::Val{N}, idx::NTuple{D, Int}) where {D, N}
    f = 0
    stride = 1
    @inbounds for d in 1:D
        f += (idx[d] - 1) * stride
        stride *= N
    end
    return f + 1
end

# Flat (1-based) -> (k_1, ..., k_D) (each in 1..N)
@inline function point_flat_to_multi(::Val{D}, ::Val{N}, flat::Int) where {D, N}
    f = flat - 1
    return ntuple(d -> begin
        v = (f ÷ (N ^ (d - 1))) % N
        v + 1
    end, Val(D))
end

# ============================================================================
# Property access
# ============================================================================

const _PTS_FIELDSET_INTERNAL = (:n, :storage)

function Base.getproperty(pfs::PointSampleFieldSet{D, N, L, Names, ST, S},
                          name::Symbol) where {D, N, L, Names, ST, S}
    if name in _PTS_FIELDSET_INTERNAL
        return getfield(pfs, name)
    end
    if name in Names
        return PointSampleFieldView{typeof(pfs), name}(pfs)
    end
    throw(KeyError(name))
end

function Base.setproperty!(pfs::PointSampleFieldSet, name::Symbol, value)
    if name in _PTS_FIELDSET_INTERNAL
        setfield!(pfs, name, value)
    else
        throw(ArgumentError("PointSampleFieldSet: cannot set field :$name as a whole; " *
                              "use indexed assignment `fields.$name[i] = values` instead."))
    end
end

function Base.propertynames(::PointSampleFieldSet{D, N, L, Names, ST, S}
                              ) where {D, N, L, Names, ST, S}
    return (Names..., _PTS_FIELDSET_INTERNAL...)
end

# ============================================================================
# PointSampleFieldView and PointSampleView
# ============================================================================

"""
    PointSampleFieldView{PFS, name}

The view returned by `pfs.fieldname`. Indexing with an integer gives a
`PointSampleView` for that element.
"""
struct PointSampleFieldView{PFS, name}
    pfs::PFS
end

@inline Base.length(v::PointSampleFieldView) = v.pfs.n
@inline Base.eachindex(v::PointSampleFieldView) = 1:v.pfs.n
@inline Base.size(v::PointSampleFieldView) = (v.pfs.n,)
@inline Base.firstindex(::PointSampleFieldView) = 1
@inline Base.lastindex(v::PointSampleFieldView) = v.pfs.n

"""
    PointSampleView{PFS, name, D, N}

Per-element point-sample view. Supports:

- `view[k]`           — get/set k-th flat point value (1..N^D)
- `view[(k_1,..k_D)]` — get/set value at multi-index
- `length(view)`      — N^D
- iteration over flat point values
"""
struct PointSampleView{PFS, name, D, N}
    pfs::PFS
    i::Int
end

@inline function Base.getindex(fv::PointSampleFieldView{PFS, name}, i::Integer
                                 ) where {PFS, name}
    pfs = fv.pfs
    D = spatial_dim(pfs)
    N = n_points_per_axis(pfs)
    return PointSampleView{PFS, name, D, N}(pfs, Int(i))
end

@inline function Base.setindex!(fv::PointSampleFieldView{PFS, name}, values, i::Integer
                                  ) where {PFS, name}
    pfs = fv.pfs
    npts = n_points_per_cell(pfs)
    length(values) == npts || throw(DimensionMismatch(
        "expected $npts point values, got $(length(values))"))
    @inbounds for k in 1:npts
        _set_poly_coeff!_pts(pfs, Val(name), Int(i), k, values[k])
    end
    return values
end

@inline Base.length(::PointSampleView{PFS, name, D, N}) where {PFS, name, D, N} = N^D
@inline Base.size(::PointSampleView{PFS, name, D, N}) where {PFS, name, D, N} = (N^D,)

@inline function Base.getindex(pv::PointSampleView{PFS, name, D, N}, k::Integer
                                 ) where {PFS, name, D, N}
    return _get_poly_coeff_pts(pv.pfs, Val(name), pv.i, Int(k))
end

@inline function Base.setindex!(pv::PointSampleView{PFS, name, D, N}, value, k::Integer
                                  ) where {PFS, name, D, N}
    _set_poly_coeff!_pts(pv.pfs, Val(name), pv.i, Int(k), value)
    return value
end

# Multi-index access: pv[(i, j)] for D=2
@inline function Base.getindex(pv::PointSampleView{PFS, name, D, N},
                                idx::NTuple{D, Integer}) where {PFS, name, D, N}
    flat = point_multi_to_flat(Val(N), ntuple(d -> Int(idx[d]), Val(D)))
    return _get_poly_coeff_pts(pv.pfs, Val(name), pv.i, flat)
end

@inline function Base.setindex!(pv::PointSampleView{PFS, name, D, N}, value,
                                  idx::NTuple{D, Integer}) where {PFS, name, D, N}
    flat = point_multi_to_flat(Val(N), ntuple(d -> Int(idx[d]), Val(D)))
    _set_poly_coeff!_pts(pv.pfs, Val(name), pv.i, flat, value)
    return value
end

@inline Base.iterate(pv::PointSampleView, k::Int=1) = k > length(pv) ? nothing : (pv[k], k+1)

# ----------------------------------------------------------------------------
# Tensor-product Lagrange evaluation at equispaced nodes
# ----------------------------------------------------------------------------

# Equispaced barycentric weights on [0, 1] with N nodes (P = N - 1).
# w_k = (-1)^k * binomial(P, k-1)  (up to a constant factor that cancels).
@inline function _equispaced_bary_weights(::Val{N}) where {N}
    return ntuple(k -> begin
        # k is 1..N; the corresponding 0-based index is (k-1)
        k0 = k - 1
        # binomial(P, k0) where P = N - 1
        b = _binom(N - 1, k0)
        s = isodd(k0) ? -1 : 1
        Float64(s * b)
    end, Val(N))
end

# Compile-time-friendly binomial computed via product (no factorial overflow
# concerns for the small N values we expect here, N ≤ ~10).
@inline function _binom(n::Int, k::Int)
    k < 0 && return 0
    k > n && return 0
    k = min(k, n - k)
    out = 1
    @inbounds for i in 1:k
        out = out * (n - i + 1) ÷ i
    end
    return out
end

# Equispaced node positions on [0, 1] with N nodes.
@inline function _equispaced_nodes(::Val{N}) where {N}
    return ntuple(k -> Float64(k - 1) / Float64(N - 1), Val(N))
end

# Specialization for N = 1 (a single node at the middle, by convention 0.0):
# the polynomial is constant, evaluate trivially.
@inline _equispaced_nodes(::Val{1}) = (0.0,)
@inline _equispaced_bary_weights(::Val{1}) = (1.0,)

# 1-D barycentric Lagrange evaluation at a point ξ ∈ [0, 1] with N samples
# at equispaced nodes, given the N node values `vals::NTuple{N, T}`.
@inline function _eval_1d_lagrange(vals::NTuple{N, T}, ξ) where {N, T}
    nodes = _equispaced_nodes(Val(N))
    weights = _equispaced_bary_weights(Val(N))
    Tres = promote_type(T, typeof(ξ))
    num = zero(Tres)
    den = zero(Tres)
    @inbounds for j in 1:N
        diff = ξ - nodes[j]
        if iszero(diff)
            return Tres(vals[j])
        end
        wj = weights[j] / diff
        num += wj * vals[j]
        den += wj
    end
    return num / den
end

# Evaluate the 1-D Lagrange basis function L_j(ξ) at the equispaced node set
# of size N. (Used to build the tensor-product evaluator.)
@inline function _lagrange_basis_value(::Val{N}, j::Int, ξ) where {N}
    nodes = _equispaced_nodes(Val(N))
    weights = _equispaced_bary_weights(Val(N))
    # If ξ is at a node, only that node's basis function is 1.
    @inbounds for k in 1:N
        if ξ == nodes[k]
            return Float64(k == j ? 1.0 : 0.0)
        end
    end
    den = 0.0
    num = 0.0
    @inbounds for k in 1:N
        diff = ξ - nodes[k]
        wj = weights[k] / diff
        if k == j
            num = wj
        end
        den += wj
    end
    return num / den
end

"""
    eval_point_samples(::Val{D}, ::Val{N}, vals, ξ::NTuple{D, T}) -> T

Tensor-product Lagrange interpolation at reference point `ξ` in the unit
cube `[0, 1]^D`. `vals` is an iterable of length `N^D` in column-major
multi-index order (matching the flat layout used by `PointSampleFieldSet`).
"""
@inline function eval_point_samples(::Val{D}, ::Val{N}, vals, ξ) where {D, N}
    # Compute basis values along each axis.
    L = ntuple(d -> ntuple(j -> _lagrange_basis_value(Val(N), j, ξ[d]), Val(N)),
                 Val(D))
    Tres = promote_type(eltype(vals), typeof(ξ[1]))
    s = zero(Tres)
    # Iterate flat 1..N^D, decoding the multi-index on the fly.
    @inbounds for f in 1:(N^D)
        midx = point_flat_to_multi(Val(D), Val(N), f)
        prod_L = one(Tres)
        for d in 1:D
            prod_L *= L[d][midx[d]]
        end
        s += vals[f] * prod_L
    end
    return s
end

# Evaluation: `view(point)` — interpolates at the reference point.
@inline function (pv::PointSampleView{PFS, name, D, N})(point) where {PFS, name, D, N}
    npts = N^D
    vals = ntuple(k -> pv[k], Val(npts))
    return eval_point_samples(Val(D), Val(N), vals, point)
end

function Base.collect(pv::PointSampleView)
    n = length(pv)
    [pv[k] for k in 1:n]
end

# ============================================================================
# Layout-specific accessors (mirrors PolynomialFieldSet's _get_poly_coeff)
# ============================================================================

@inline function _get_poly_coeff_pts(pfs::PointSampleFieldSet{D, N, L, Names, ST, S},
                                       ::Val{name}, i::Int, k::Int
                                       ) where {D, N, L, Names, ST, S, name}
    npts = N^D
    return _get_poly_coeff_impl(pfs.storage, L, Val(name), i, k, npts)
end

@inline function _set_poly_coeff!_pts(pfs::PointSampleFieldSet{D, N, L, Names, ST, S},
                                        ::Val{name}, i::Int, k::Int, value
                                        ) where {D, N, L, Names, ST, S, name}
    npts = N^D
    _set_poly_coeff_impl!(pfs.storage, L, Val(name), i, k, npts, value)
    return value
end

# ============================================================================
# Show
# ============================================================================

function Base.show(io::IO, pfs::PointSampleFieldSet{D, N, L, Names, ST, S}
                    ) where {D, N, L, Names, ST, S}
    print(io, "PointSampleFieldSet{D=", D, ", N=", N, ", ", L, "}(n=$(pfs.n)) with fields: ")
    for (k, name) in enumerate(Names)
        if k > 1; print(io, ", "); end
        print(io, "$name::", fieldtype(ST, k))
    end
end

function Base.show(io::IO, pv::PointSampleView{PFS, name, D, N}
                    ) where {PFS, name, D, N}
    npts = N^D
    print(io, "PointSampleView{$name, D=$D, N=$N}(values=[")
    for k in 1:npts
        k > 1 && print(io, ", ")
        print(io, pv[k])
    end
    print(io, "])")
end
