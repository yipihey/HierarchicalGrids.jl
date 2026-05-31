# ============================================================================
# CellAverageFieldSet — finite-volume (cell-average) field storage
#
# Third sibling to `PolynomialFieldSet` (Path A) and `PointSampleFieldSet`
# (Path B). Each cell holds a SINGLE scalar per field: the *finite-volume
# cell average* — the volume integral of the quantity over the cell divided
# by the cell volume,
#
#     ū_i = (1 / |Ω_i|) ∫_{Ω_i} u(x) dx .
#
# This is mathematically the degree-0 case, but it is a field MODEL in its
# own right, with finite-volume conservation semantics baked into its
# refinement operators:
#
# - Prolongation (coarse → fine): piecewise-constant injection. Each child's
#   average equals the parent's average. Conservative because children of a
#   split have equal volume, so the volume-weighted sum Σ ūₖ·|Ωₖ| over the
#   children equals ū_parent·|Ω_parent| exactly.
# - Restriction (fine → coarse): the parent's average is the volume-weighted
#   average of its children's averages. Children of a single (an)isotropic
#   split have equal volume by construction, so this reduces to the
#   arithmetic mean — exact and conservative (total value×volume preserved
#   to round-off).
#
# NOT a `PointSampleFieldSet{…, N=1}`. A point sample is the *value at a
# point*; its coarsening is nodal interpolation. A cell average is the
# *mean over a volume*; its coarsening is the conservative volume-weighted
# mean. The distinction is semantic, intentional, and reflected in the type.
#
# Layout flexibility: rides the EXACT layout machinery shared with
# `PolynomialFieldSet` / `PointSampleFieldSet` (SoA / AoS / Blocked{B}) — it
# stores exactly one scalar per cell, i.e. `nc = 1` coefficients. No layout
# helper is forked.
# ============================================================================

"""
    CellAverageFieldSet{D, L, Names, ScalarTypes, Storage}

A set of named finite-volume cell-average fields stored over `n` mesh cells.
Each cell carries a single scalar per field: the volume average of the
quantity over that cell (∫ over the cell, divided by the cell volume).

`D` is the spatial dimension. `L` is the memory layout (`SoA`, `AoS`,
`Blocked{B}`), flexing across exactly the same layout machinery as
`PolynomialFieldSet` / `PointSampleFieldSet`. Fields can have different
scalar types (`ScalarTypes`) but share the layout and dimension.

# Access

```julia
fields = allocate_cell_average_fields(SoA(), Val(2), 100; rho=Float64)

fields.rho[i]            # read the cell average of field :rho at cell i
fields.rho[i] = v        # write the cell average at cell i
```

The accessor is deliberately simpler than the polynomial / point-sample
views: a cell-average field has one scalar per cell, so `fields.rho[i]`
returns/sets that scalar directly (there is no per-cell coefficient view).

# Convention

A cell average is the finite-volume mean over the cell. This is **not** a
point value: do not confuse `CellAverageFieldSet` with
`PointSampleFieldSet{…, N=1}` (a single nodal point value with interpolatory
coarsening). The two differ in both semantics (volume average vs. point
value) and in their AMR operators (conservative volume-weighted mean vs.
nodal interpolation).
"""
struct CellAverageFieldSet{D, L<:AbstractLayout, Names, ScalarTypes, Storage}
    n::Int
    storage::Storage
end

"""
    allocate_cell_average_fields(layout, ::Val{D}, n_cells; field=Type, ...)

Allocate a `CellAverageFieldSet` holding one scalar per field per cell.
Mirrors `allocate_polynomial_fields` / `allocate_point_sample_fields`: each
keyword's value is the scalar type of that field's cell average, and any
number of fields may be supplied NamedTuple-style.

```julia
fields = allocate_cell_average_fields(AoS(), Val(3), 64; rho=Float64, p=Float32)
```
"""
function allocate_cell_average_fields(layout::L, ::Val{D}, n::Integer;
                                       field_kwargs...) where {L<:AbstractLayout, D}
    D >= 1 || throw(ArgumentError("CellAverageFieldSet: D must be ≥ 1, got $D"))
    names = Tuple(keys(field_kwargs))
    types = Tuple(values(field_kwargs))
    # One scalar per cell per field: ride the shared (nc) layout machinery
    # with nc = 1 — the same helper PolynomialFieldSet / PointSampleFieldSet
    # use, no fork.
    storage = _make_polynomial_storage(L, 1, Int(n), names, types)
    Names = names
    ScalarTypes = Tuple{types...}
    return CellAverageFieldSet{D, L, Names, ScalarTypes, typeof(storage)}(Int(n), storage)
end

function allocate_cell_average_fields(::Type{L}, ::Val{D}, n::Integer;
                                       field_kwargs...) where {L<:AbstractLayout, D}
    return allocate_cell_average_fields(L(), Val(D), n; field_kwargs...)
end

# ============================================================================
# Queries
# ============================================================================

@inline n_elements(cafs::CellAverageFieldSet) = cafs.n
@inline field_names(::CellAverageFieldSet{D, L, Names, ST, S}) where {D, L, Names, ST, S} = Names
@inline spatial_dim(::CellAverageFieldSet{D, L, Names, ST, S}) where {D, L, Names, ST, S} = D
@inline _layout_type_ca(::CellAverageFieldSet{D, L, Names, ST, S}) where {D, L, Names, ST, S} = L

# ============================================================================
# Property access
# ============================================================================

const _CA_FIELDSET_INTERNAL = (:n, :storage)

function Base.getproperty(cafs::CellAverageFieldSet{D, L, Names, ST, S},
                          name::Symbol) where {D, L, Names, ST, S}
    if name in _CA_FIELDSET_INTERNAL
        return getfield(cafs, name)
    end
    if name in Names
        return CellAverageFieldView{typeof(cafs), name}(cafs)
    end
    throw(KeyError(name))
end

function Base.setproperty!(cafs::CellAverageFieldSet, name::Symbol, value)
    if name in _CA_FIELDSET_INTERNAL
        setfield!(cafs, name, value)
    else
        throw(ArgumentError("CellAverageFieldSet: cannot set field :$name as a whole; " *
                              "use indexed assignment `fields.$name[i] = value` instead."))
    end
end

function Base.propertynames(::CellAverageFieldSet{D, L, Names, ST, S}
                              ) where {D, L, Names, ST, S}
    return (Names..., _CA_FIELDSET_INTERNAL...)
end

# ============================================================================
# CellAverageFieldView
# ============================================================================

"""
    CellAverageFieldView{CAFS, name}

The view returned by `cafs.fieldname`. Indexing with an integer gets/sets
the cell average of that field at the given cell:

- `view[i]`      — read the cell average at cell `i`
- `view[i] = v`  — write the cell average at cell `i`
- `length(view)` — number of cells
- iteration over per-cell averages

Unlike `PolynomialFieldView` / `PointSampleFieldView`, indexing returns the
scalar average directly (a cell-average field has a single scalar per cell,
so there is no per-cell coefficient/sample sub-view).
"""
struct CellAverageFieldView{CAFS, name}
    cafs::CAFS
end

@inline Base.length(v::CellAverageFieldView) = v.cafs.n
@inline Base.eachindex(v::CellAverageFieldView) = 1:v.cafs.n
@inline Base.size(v::CellAverageFieldView) = (v.cafs.n,)
@inline Base.firstindex(::CellAverageFieldView) = 1
@inline Base.lastindex(v::CellAverageFieldView) = v.cafs.n

@inline function Base.getindex(fv::CellAverageFieldView{CAFS, name}, i::Integer
                                 ) where {CAFS, name}
    return _get_cell_average(fv.cafs, Val(name), Int(i))
end

@inline function Base.setindex!(fv::CellAverageFieldView{CAFS, name}, value, i::Integer
                                  ) where {CAFS, name}
    _set_cell_average!(fv.cafs, Val(name), Int(i), value)
    return value
end

@inline Base.iterate(fv::CellAverageFieldView, i::Int=1) =
    i > length(fv) ? nothing : (fv[i], i + 1)

function Base.collect(fv::CellAverageFieldView)
    n = length(fv)
    return [fv[i] for i in 1:n]
end

# ============================================================================
# Layout-specific accessors (reuse the shared nc-aware machinery with nc = 1)
# ============================================================================

# A cell average is one scalar per cell: route through the same
# `_get_poly_coeff_impl` / `_set_poly_coeff_impl!` the other two models use,
# fixing nc = 1 and k = 1.
@inline function _get_cell_average(cafs::CellAverageFieldSet{D, L, Names, ST, S},
                                    ::Val{name}, i::Int
                                    ) where {D, L, Names, ST, S, name}
    return _get_poly_coeff_impl(cafs.storage, L, Val(name), i, 1, 1)
end

@inline function _set_cell_average!(cafs::CellAverageFieldSet{D, L, Names, ST, S},
                                     ::Val{name}, i::Int, value
                                     ) where {D, L, Names, ST, S, name}
    _set_poly_coeff_impl!(cafs.storage, L, Val(name), i, 1, 1, value)
    return value
end

# NOTE on Solver integration: the orchestrators' `CellView` / multi-field
# `HaloView` consume per-cell field views through the `_coeffs_for_cell` /
# `_set_coeffs_for_cell!` abstraction (defined in the Solver layer, with
# methods for `PolynomialFieldView` and `PointSampleFieldView`). A
# `CellAverageFieldView` satisfies that same abstraction — its per-cell
# "coefficient" is the single scalar average — so the Solver layer adds the
# two matching `_coeffs_for_cell` methods rather than special-casing.

# ============================================================================
# Show
# ============================================================================

function Base.show(io::IO, cafs::CellAverageFieldSet{D, L, Names, ST, S}
                    ) where {D, L, Names, ST, S}
    print(io, "CellAverageFieldSet{D=", D, ", ", L, "}(n=$(cafs.n)) with fields: ")
    for (k, name) in enumerate(Names)
        if k > 1; print(io, ", "); end
        print(io, "$name::", fieldtype(ST, k))
    end
end
