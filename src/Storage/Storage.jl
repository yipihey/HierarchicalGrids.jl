"""
    Storage

Layer 2.5: layout-flexible field storage. Decouples access patterns from
memory layouts, inspired by Taichi's `ti.root` design.

Users write kernels using natural per-element access:

```julia
fields.density[i] = fields.density[i] + fields.velocity[i] * dt
```

The actual memory layout (SoA, AoS, blocked, etc.) is determined by the
type parameter of the FieldSet. Switching layouts is a one-line change in
the constructor; the kernel code is unchanged.

# Layouts provided

- `SoA`: structure-of-arrays. Each field is a separate contiguous Vector.
  Best for kernels that stream over one or two fields at a time.

- `AoS`: array-of-structures. Each element is a struct containing all fields,
  stored in one Vector. Best for kernels that access many fields per element.

- `Blocked{B}`: cells grouped into blocks of size B; within each block, the
  layout is SoA. Best for stencil operations with spatial locality.

# Adding a new layout

To add a new layout `MyLayout`:

1. Define `struct MyLayout <: AbstractLayout end` (with type parameters as needed).
2. Implement `_make_storage(::Type{MyLayout}, n, names, types)` returning the
   internal storage object.
3. Implement `_get_field(storage, layout::Type{MyLayout}, ::Val{name}, i)`
   and `_set_field!(storage, layout::Type{MyLayout}, ::Val{name}, i, value)`.
4. Implement `_resize_storage!(storage, layout::Type{MyLayout}, n)`.

The user-facing API (`fields.name[i]`) works automatically through the
property dispatch.
"""
module Storage

# Bases is included before Storage at the top level; access via parent module.
using ..Bases: AbstractBasis, n_coeffs, evaluate, gradient
# Quadrature is included before Storage at the top level; we use it for the
# polynomial-aware action-error indicator.
using ..Quadrature: QuadRule, action_error_l2

export AbstractLayout, SoA, AoS, Blocked
export FieldSet, n_elements, field_names, has_field
export resize_fields!, allocate_fields
export PolynomialFieldSet, allocate_polynomial_fields
export PolynomialFieldView, PolynomialView
export gradient_at, n_coeffs_per_element, basis_of
export polynomial_action_error, polynomial_action_error_per_element

# ============================================================================
# Layout types
# ============================================================================

"""
    AbstractLayout

Marker abstract type for memory layouts. Concrete layouts are types like
`SoA`, `AoS`, `Blocked{B}`.
"""
abstract type AbstractLayout end

"""
    SoA <: AbstractLayout

Structure-of-arrays layout. Each field is stored in its own contiguous
Vector. Indexing `fields.name[i]` is a direct array access.

Best for kernels that stream over one or two fields at a time, since
adjacent elements of one field are adjacent in memory.
"""
struct SoA <: AbstractLayout end

"""
    AoS <: AbstractLayout

Array-of-structures layout. Each element is a NamedTuple of all fields,
stored in one Vector. Indexing `fields.name[i]` accesses the named field
of the element at index i.

Best for kernels that access many fields of the same element together,
since all fields of an element are adjacent in memory.
"""
struct AoS <: AbstractLayout end

"""
    Blocked{BlockSize, InnerLayout} <: AbstractLayout

Hierarchical layout: elements are grouped into blocks of `BlockSize`, with
each block using `InnerLayout` (typically SoA) internally.

For BlockSize=8 and InnerLayout=SoA: storage is a Vector of blocks, where
each block contains contiguous arrays of length 8 for each field. Stencil
operations within a block hit cache extremely well.

This is the closest analog to Taichi's `ti.root.dense(ti.i, M).dense(ti.i, B)`
pattern.
"""
struct Blocked{BlockSize, InnerLayout<:AbstractLayout} <: AbstractLayout
    function Blocked{BlockSize, InnerLayout}() where {BlockSize, InnerLayout}
        BlockSize > 0 ||
            throw(ArgumentError("Blocked: BlockSize must be positive (got $BlockSize)"))
        ispow2(BlockSize) ||
            throw(ArgumentError("Blocked: BlockSize must be a power of 2 for efficient " *
                                  "indexing (got $BlockSize)"))
        new{BlockSize, InnerLayout}()
    end
end

# Convenience: Blocked{B} defaults to SoA inner
Blocked{B}() where B = Blocked{B, SoA}()

# ============================================================================
# FieldSet — the user-facing type
# ============================================================================

"""
    FieldSet{L<:AbstractLayout, Names, Types, Storage}

A set of named per-element fields with a configurable memory layout.

`L` is the layout type (e.g., `SoA`, `AoS`, `Blocked{8}`).
`Names` is a tuple of Symbols naming the fields.
`Types` is a tuple of element types for each field.
`Storage` is the internal storage type, determined by the layout.

Users construct via `allocate_fields(layout, n; field=Type, ...)` and access
via property syntax: `fields.name[i]`.

# Example

```julia
# SoA layout for a particle simulation
particles = allocate_fields(SoA(), 1000; pos=NTuple{3, Float32}, vel=NTuple{3, Float32}, mass=Float32)

# Same access syntax regardless of layout
particles.pos[1] = (0.0f0, 0.0f0, 0.0f0)
particles.mass[1] = 1.0f0

# Switching to AoS is a one-line change
particles_aos = allocate_fields(AoS(), 1000; pos=NTuple{3, Float32}, vel=NTuple{3, Float32}, mass=Float32)
particles_aos.pos[1] = (0.0f0, 0.0f0, 0.0f0)  # works the same
```
"""
mutable struct FieldSet{L<:AbstractLayout, Names, Types, Storage}
    n::Int
    storage::Storage
end

"""
    allocate_fields(layout::AbstractLayout, n::Integer; fields...)

Construct a FieldSet with `n` elements using the specified layout. Field
names and types are passed as keyword arguments.

# Example

```julia
fields = allocate_fields(SoA(), 1000; density=Float32, momentum=NTuple{3, Float32})
```
"""
function allocate_fields(layout::L, n::Integer; field_kwargs...) where L<:AbstractLayout
    names = Tuple(keys(field_kwargs))
    types = Tuple(values(field_kwargs))
    Names = names
    Types = Tuple{types...}
    storage = _make_storage(L, Int(n), names, types)
    return FieldSet{L, Names, Types, typeof(storage)}(Int(n), storage)
end

# Backward-compatible signature using a Type instead of an instance
function allocate_fields(::Type{L}, n::Integer; field_kwargs...) where L<:AbstractLayout
    return allocate_fields(L(), n; field_kwargs...)
end

# Basic queries
@inline n_elements(fs::FieldSet) = fs.n
@inline field_names(::FieldSet{L, Names, Types, S}) where {L, Names, Types, S} = Names
@inline has_field(fs::FieldSet{L, Names, Types, S}, name::Symbol) where {L, Names, Types, S} = name in Names

# ============================================================================
# Storage construction per layout
# ============================================================================

# SoA: NamedTuple of Vectors
function _make_storage(::Type{SoA}, n::Int, names::Tuple, types::Tuple)
    arrays = map(T -> Vector{T}(undef, n), types)
    return NamedTuple{names}(arrays)
end

# AoS: single Vector of NamedTuple
function _make_storage(::Type{AoS}, n::Int, names::Tuple, types::Tuple)
    ElType = NamedTuple{names, Tuple{types...}}
    return Vector{ElType}(undef, n)
end

# Blocked{B, InnerLayout}: array of blocks, each with InnerLayout
function _make_storage(::Type{Blocked{B, IL}}, n::Int, names::Tuple, types::Tuple) where {B, IL}
    # Each block is itself a small storage of size B (or less for the last block)
    n_full_blocks = n ÷ B
    remainder = n % B
    n_blocks = n_full_blocks + (remainder > 0 ? 1 : 0)

    blocks = Vector{Any}(undef, n_blocks)
    for b in 1:n_full_blocks
        blocks[b] = _make_storage(IL, B, names, types)
    end
    if remainder > 0
        blocks[end] = _make_storage(IL, remainder, names, types)
    end
    # Stable element type for the block array
    BlockType = typeof(_make_storage(IL, B, names, types))
    typed_blocks = Vector{BlockType}(undef, n_blocks)
    for b in 1:n_blocks
        typed_blocks[b] = blocks[b]
    end
    return typed_blocks
end

# ============================================================================
# Property access via getproperty (the user-facing magic)
# ============================================================================

# Internal field names that should be accessed normally
const _FIELDSET_INTERNAL_FIELDS = (:n, :storage)

function Base.getproperty(fs::FieldSet{L, Names, Types, S}, name::Symbol) where {L, Names, Types, S}
    if name in _FIELDSET_INTERNAL_FIELDS
        return getfield(fs, name)
    end
    if name in Names
        return FieldView{typeof(fs), name}(fs)
    end
    throw(KeyError(name))
end

# Don't allow setproperty! on user-facing field names; use indexing
function Base.setproperty!(fs::FieldSet, name::Symbol, value)
    if name in _FIELDSET_INTERNAL_FIELDS
        setfield!(fs, name, value)
    else
        throw(ArgumentError("FieldSet: cannot set field :$name as a whole; " *
                              "use indexed assignment `fields.$name[i] = value` instead."))
    end
end

function Base.propertynames(fs::FieldSet{L, Names, Types, S}) where {L, Names, Types, S}
    return (Names..., _FIELDSET_INTERNAL_FIELDS...)
end

# ============================================================================
# FieldView — a thin wrapper for indexed field access
# ============================================================================

"""
    FieldView{FS, name}

Returned by `fields.name`; provides indexed access (`fields.name[i]`) that
dispatches to the appropriate layout-specific accessor at compile time.

Type parameters carry the FieldSet type and the field name as a Symbol,
so dispatch is fully statically resolved.
"""
struct FieldView{FS, name}
    fs::FS
end

@inline Base.length(fv::FieldView) = fv.fs.n
@inline Base.eachindex(fv::FieldView) = 1:fv.fs.n
@inline Base.size(fv::FieldView) = (fv.fs.n,)
@inline Base.firstindex(::FieldView) = 1
@inline Base.lastindex(fv::FieldView) = fv.fs.n

# Index access — dispatches to layout-specific implementation
@inline function Base.getindex(fv::FieldView{FS, name}, i::Integer) where {FS, name}
    return _get_field(fv.fs.storage, _layout_type(fv.fs), Val(name), Int(i))
end

@inline function Base.setindex!(fv::FieldView{FS, name}, value, i::Integer) where {FS, name}
    _set_field!(fv.fs.storage, _layout_type(fv.fs), Val(name), Int(i), value)
    return value
end

# Helper to extract layout type from FieldSet type
@inline _layout_type(::FieldSet{L, N, T, S}) where {L, N, T, S} = L

# ============================================================================
# Layout-specific field accessors
# ============================================================================

# --- SoA ---
@inline function _get_field(storage::NamedTuple, ::Type{SoA}, ::Val{name}, i::Int) where name
    return getfield(storage, name)[i]
end

@inline function _set_field!(storage::NamedTuple, ::Type{SoA}, ::Val{name}, i::Int, value) where name
    getfield(storage, name)[i] = value
    return value
end

# --- AoS ---
@inline function _get_field(storage::Vector, ::Type{AoS}, ::Val{name}, i::Int) where name
    return getfield(storage[i], name)
end

@inline function _set_field!(storage::Vector, ::Type{AoS}, ::Val{name}, i::Int, value) where name
    # NamedTuple is immutable, so we have to construct a new one
    current = storage[i]
    storage[i] = merge(current, NamedTuple{(name,)}((value,)))
    return value
end

# --- Blocked{B, InnerLayout} ---
@inline function _get_field(storage::Vector, ::Type{Blocked{B, IL}}, ::Val{name}, i::Int) where {B, IL, name}
    block_idx = ((i - 1) >> trailing_zeros(B)) + 1  # ((i-1) ÷ B) + 1, fast for power-of-2 B
    within_block = ((i - 1) & (B - 1)) + 1          # ((i-1) % B) + 1
    return _get_field(storage[block_idx], IL, Val(name), within_block)
end

@inline function _set_field!(storage::Vector, ::Type{Blocked{B, IL}}, ::Val{name}, i::Int, value) where {B, IL, name}
    block_idx = ((i - 1) >> trailing_zeros(B)) + 1
    within_block = ((i - 1) & (B - 1)) + 1
    _set_field!(storage[block_idx], IL, Val(name), within_block, value)
    return value
end

# ============================================================================
# Resizing
# ============================================================================

"""
    resize_fields!(fs::FieldSet, new_n::Integer)

Resize the FieldSet to have `new_n` elements. Existing data is preserved
where indices overlap; new elements are uninitialized.
"""
function resize_fields!(fs::FieldSet{L, Names, Types, S}, new_n::Integer) where {L, Names, Types, S}
    types_tuple = Tuple(fieldtype(Types, k) for k in 1:fieldcount(Types))
    _resize_storage!(fs.storage, L, Int(new_n), Names, types_tuple)
    fs.n = Int(new_n)
    return fs
end

# Layout-specific resize
function _resize_storage!(storage::NamedTuple, ::Type{SoA}, new_n::Int,
                            names::Tuple, types::Tuple)
    for name in keys(storage)
        resize!(getfield(storage, name), new_n)
    end
end

function _resize_storage!(storage::Vector, ::Type{AoS}, new_n::Int,
                            names::Tuple, types::Tuple)
    resize!(storage, new_n)
end

function _resize_storage!(storage::Vector, ::Type{Blocked{B, IL}}, new_n::Int,
                            names::Tuple, types::Tuple) where {B, IL}
    n_full_blocks = new_n ÷ B
    remainder = new_n % B
    n_blocks_needed = n_full_blocks + (remainder > 0 ? 1 : 0)
    n_blocks_current = length(storage)

    if n_blocks_needed > n_blocks_current
        # Grow: append new blocks. Each new block needs its own inner storage.
        old_len = length(storage)
        resize!(storage, n_blocks_needed)
        for b in (old_len + 1):n_blocks_needed
            block_size = (b == n_blocks_needed && remainder > 0) ? remainder : B
            storage[b] = _make_storage(IL, block_size, names, types)
        end
    elseif n_blocks_needed < n_blocks_current
        resize!(storage, n_blocks_needed)
    end
    # Note: doesn't currently grow/shrink the *last* block when remainder
    # changes within an existing block. That's a TODO if needed; for the
    # framework's current usage (allocate-once, no resize) it doesn't matter.
end

# ============================================================================
# Iteration helpers
# ============================================================================

"""
    foreach_element(f, fs::FieldSet)

Apply function `f(i)` for each element index. Equivalent to `for i in 1:n_elements(fs); f(i); end`.

Layout-aware: for blocked layouts, iterates block-by-block to maximize
cache reuse.
"""
function foreach_element(f, fs::FieldSet)
    @inbounds for i in 1:fs.n
        f(i)
    end
end

# Show methods
function Base.show(io::IO, fs::FieldSet{L, Names, Types, S}) where {L, Names, Types, S}
    print(io, "FieldSet{", L, "}(n=$(fs.n)) with fields: ")
    for (k, name) in enumerate(Names)
        if k > 1; print(io, ", "); end
        print(io, "$name::", fieldtype(Types, k))
    end
end

# ============================================================================
# Polynomial field storage (depends on Bases for n_coeffs/evaluate/gradient)
# ============================================================================
include("PolynomialFieldSet.jl")

end # module Storage
