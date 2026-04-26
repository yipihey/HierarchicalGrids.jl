# Layouts

This document explains the Storage layer in depth: what layouts are provided, when to use which, and how to add your own.

## The problem

Different operations on the same data want different memory layouts.

A streaming kernel that updates one field across all cells (`for i; density[i] = ...; end`) wants that field's data to be contiguous — so it should be a stand-alone Vector. This is **structure-of-arrays (SoA)**.

A per-cell update that reads several fields, computes, and writes them all (`density, momentum, energy ← f(density, momentum, energy)`) wants all the fields for one cell to be adjacent — so each cell should be a struct holding all its fields. This is **array-of-structures (AoS)**.

A stencil operation that touches a cell's spatial neighbors wants those neighbors to be nearby in memory, regardless of the field structure. This wants a **blocked** layout: cells are grouped into spatial blocks, and within each block the layout can be SoA or AoS as preferred.

In a traditional code, you pick one and live with it. Picking wrong can cost a factor of 2-5 in cache performance for the affected kernels. Switching means rewriting kernels.

## The solution

Decouple the layout from the access pattern. The user writes:

```julia
fields.density[i] = fields.density[i] + fields.velocity[i] * dt
```

The layout determines what `fields.density[i]` actually does in memory. The compiler generates direct memory operations either way (no runtime indirection); the source code is the same.

This is the Taichi approach, adapted to native Julia. Julia's multiple dispatch + compile-time type parameter resolution gives us the same machinery without needing a custom DSL.

## Provided layouts

### `SoA` — structure-of-arrays

Each field is a separate contiguous Vector. `fields.density` is one Vector; `fields.velocity` is another.

**Best for**: streaming kernels that touch one or two fields at a time. The CPU prefetcher loves this: predictable, contiguous access.

**Worst for**: kernels that touch many fields per cell. Each field access pulls a different cache line.

```julia
fields = allocate_fields(SoA(), n; density = Float32, velocity = NTuple{3, Float32})
```

### `AoS` — array-of-structures

All fields of one cell are bundled into a NamedTuple, stored in one Vector.

**Best for**: per-cell update kernels that read all the fields, compute, and write back. One cache line covers (most of) one cell.

**Worst for**: streaming kernels that read just one field. You're paying for cache lines that contain fields you don't want.

```julia
fields = allocate_fields(AoS(), n; density = Float32, velocity = NTuple{3, Float32})
```

A subtle point: the underlying NamedTuple is immutable, so writing one field of an AoS cell requires constructing a new NamedTuple and storing it. The framework hides this from you (`fields.density[i] = x` just works), but it does mean writes are slightly more expensive than reads.

### `Blocked{B, InnerLayout}` — hierarchical

Cells are grouped into blocks of `B` cells. Within each block, `InnerLayout` (typically SoA) is used. `B` should be a power of 2 for fast indexing.

**Best for**: stencil operations that need spatial locality. If neighboring cells are in the same block, accessing them is cache-friendly even though the global layout is non-trivial.

**Best for SIMD**: with `B` matching your SIMD width (e.g., 8 for AVX-512 with 32-bit floats), within-block kernels can vectorize cleanly.

```julia
fields = allocate_fields(Blocked{8, SoA}(), n; density = Float32, velocity = NTuple{3, Float32})
```

The block size is a type parameter, so the compiler specializes indexing per block size — no runtime divisions.

## Choosing a layout

The honest answer is: benchmark both. The framework makes this cheap.

```julia
function my_kernel!(fields, dt)
    for i in 1:n_elements(fields)
        # ... your kernel
    end
end

fields_soa = allocate_fields(SoA(), n; ...)
fields_aos = allocate_fields(AoS(), n; ...)
fields_blk = allocate_fields(Blocked{8, SoA}(), n; ...)

# Same kernel, three layouts
@time my_kernel!(fields_soa, dt)
@time my_kernel!(fields_aos, dt)
@time my_kernel!(fields_blk, dt)
```

Pick the one that's fastest for your access pattern.

Some heuristics that often hold:

- **Pure streaming, one field**: SoA wins by ~20-50%.
- **Per-cell update, all fields**: AoS wins by ~10-30%.
- **Stencil operations**: Blocked wins, sometimes by 2x or more.
- **Mixed patterns**: SoA is usually a safe default.

## How the dispatch works

The user-facing access goes through Julia's `getproperty`:

```julia
function Base.getproperty(fs::FieldSet{L, Names, ...}, name::Symbol) where {L, Names, ...}
    if name in Names
        return FieldView{typeof(fs), name}(fs)
    end
    # ...
end
```

`FieldView{FS, name}` is a thin wrapper carrying the FieldSet and the field name as type parameters. Indexing it dispatches on the layout type:

```julia
@inline function Base.getindex(fv::FieldView{FS, name}, i::Integer) where {FS, name}
    return _get_field(fv.fs.storage, _layout_type(fv.fs), Val(name), Int(i))
end

@inline _get_field(storage::NamedTuple, ::Type{SoA}, ::Val{name}, i::Int) where name =
    getfield(storage, name)[i]

@inline _get_field(storage::Vector, ::Type{AoS}, ::Val{name}, i::Int) where name =
    getfield(storage[i], name)
```

Because `name` is a type parameter (a `Val`), Julia's compiler inlines and specializes everything at compile time. The generated machine code is the same as if you'd written the layout-specific access directly. There's no runtime cost for the abstraction.

## Adding a new layout

Suppose you want to add `AoSoA{W}` — array of (struct of (arrays of size W)) — for SIMD-friendly access. Here's the procedure:

1. **Define the layout type**:

```julia
struct AoSoA{Width} <: AbstractLayout end
```

2. **Implement `_make_storage`**:

```julia
function _make_storage(::Type{AoSoA{W}}, n::Int, names::Tuple, types::Tuple) where W
    n_blocks = cld(n, W)  # ceiling division
    # Each block is a NamedTuple of NTuple{W, T} for each field
    BlockType = NamedTuple{names, Tuple{(NTuple{W, T} for T in types)...}}
    return Vector{BlockType}(undef, n_blocks)
end
```

3. **Implement `_get_field` and `_set_field!`**:

```julia
@inline function _get_field(storage::Vector, ::Type{AoSoA{W}}, ::Val{name}, i::Int) where {W, name}
    block_idx = ((i - 1) >> trailing_zeros(W)) + 1   # i ÷ W, fast for power-of-2
    within = ((i - 1) & (W - 1)) + 1                  # i % W
    return getfield(storage[block_idx], name)[within]
end

# (and analogous _set_field!)
```

4. **Implement `_resize_storage!`**:

```julia
function _resize_storage!(storage::Vector, ::Type{AoSoA{W}}, new_n::Int) where W
    n_blocks_needed = cld(new_n, W)
    resize!(storage, n_blocks_needed)
end
```

5. **Tests**. Use the same access patterns as for SoA/AoS — they should give identical results.

That's it. No solver code changes; the layout is now available to all framework users.

## Limitations and notes

**Mixed-type fields**: All currently-provided layouts assume each field has a fixed type. Heterogeneous storage (e.g., some cells have extra fields) isn't supported; you'd handle that by maintaining multiple FieldSets.

**Dynamic field addition**: The set of field names is fixed at allocation. Adding a field after the fact requires allocating a new FieldSet and migrating data.

**Sparse storage**: The mesh provides spatial sparsity (only cells that exist are stored). Per-field sparsity (some cells have a value, others don't) isn't a layout concern; you'd implement it with a separate sparse data structure.

**Compile times**: Many type parameters mean many compiler specializations. For development, this can be noticeable; in production, it's a one-time cost.

## A note on Taichi

The design here is inspired by Taichi's `ti.root.dense().place()` system. The two systems differ in important ways:

- Taichi uses a custom DSL compiled to LLVM; HierarchicalGrids uses native Julia with multiple dispatch.
- Taichi's hierarchical fields can be sparse (`bitmasked`, `pointer` storage); HierarchicalGrids relies on the mesh for sparsity.
- Taichi has a unified system for CPU, GPU, and other accelerators; HierarchicalGrids is currently CPU-only.

But the core insight is the same: decouple the layout from the access pattern, and you give your users (and your future self) the freedom to optimize without rewriting.
