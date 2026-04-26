"""
    Memory

Layer 3 (foundational): memory management infrastructure for fragmentation
resistance.

Long-running scientific simulations are vulnerable to memory fragmentation
when they repeatedly allocate and free buffers of varying sizes. Enzo and
similar codes have hit this in production runs. This module provides
explicit infrastructure to avoid the problem:

- `FieldBufferPool`: recycles buffers of similar sizes for short-lived field
  storage. Allocations after warmup come from the pool, not the system
  allocator. Memory pages stay allocated; only metadata changes.

- `ScratchBuffer`: per-thread persistent workspace. Allocations during a
  computation come from a single buffer; deallocations are LIFO (cursor
  movement), so fragmentation can't happen.

- `Arena`: per-operation bulk allocation. All allocations during the
  operation share a single buffer; the entire arena is reset at the end.
  Excellent for temporaries that don't need individual lifetime management.

# When to use which

- **Long-lived data** (mesh, persistent fields): use Julia's standard
  `Vector` with `sizehint!`. Allocate once, grow rarely.

- **Active-grid field data** (per-grid buffers reused across grids): use
  `FieldBufferPool`. Acquire/release pattern.

- **Per-step temporaries** (flux buffers, working space): use `ScratchBuffer`
  or `Arena`. Cheap to allocate, cheap to discard.

# Production deployment

For long production runs, also consider linking Julia against jemalloc
(via `LD_PRELOAD`) as a defense-in-depth measure. The system allocator's
fragmentation behavior on long-running scientific workloads is poor;
jemalloc and mimalloc are designed specifically to handle this.
"""
module Memory

export FieldBufferPool, acquire_buffer!, release_buffer!, pool_stats
export ScratchBuffer, with_scratch
export Arena, allocate_in_arena, reset_arena!

# ============================================================================
# FieldBufferPool — pooled allocation for short-lived buffers
# ============================================================================

"""
    FieldBufferPool{T}

A pool of reusable `Vector{T}` buffers, organized by size class. Acquired
buffers are taken from the pool when available, allocated fresh otherwise.
Released buffers are returned to the pool for reuse.

Buffers are bucketed into power-of-2 size classes to maximize reuse: a
request for 5000 elements gets an 8192-element buffer; a subsequent
request for 7000 elements can use the same buffer.

Thread-safe via internal lock. For high-contention scenarios, per-thread
pools are recommended.
"""
mutable struct FieldBufferPool{T}
    # available[size_class_idx] = list of free buffers of that size
    available::Dict{Int, Vector{Vector{T}}}
    # in_use: maps each currently-loaned buffer to its size class
    in_use::IdDict{Vector{T}, Int}
    # statistics
    n_acquisitions::Int
    n_pool_hits::Int
    n_fresh_allocations::Int
    total_bytes_allocated::Int
    lock::ReentrantLock
end

FieldBufferPool{T}() where T = FieldBufferPool{T}(
    Dict{Int, Vector{Vector{T}}}(),
    IdDict{Vector{T}, Int}(),
    0, 0, 0, 0,
    ReentrantLock()
)

"""
    _next_pool_size(n::Int)

Round up to the next power of 2, with a minimum of 16. Used to bucket
allocation requests into reusable size classes.
"""
@inline function _next_pool_size(n::Int)
    n = max(n, 16)
    return 1 << (8 * sizeof(Int) - leading_zeros(n - 1))
end

"""
    acquire_buffer!(pool::FieldBufferPool{T}, size::Integer)

Acquire a buffer of at least `size` elements from the pool. The returned
buffer may be larger than requested (up to the next power of 2). Use
`view(buffer, 1:size)` if you need a slice of the requested size.

Always release the buffer with `release_buffer!` when done, or it will leak
from the pool (though it will still be GC'd when references are dropped).
"""
function acquire_buffer!(pool::FieldBufferPool{T}, size::Integer) where T
    pool_size = _next_pool_size(Int(size))
    lock(pool.lock) do
        pool.n_acquisitions += 1
        if haskey(pool.available, pool_size) && !isempty(pool.available[pool_size])
            buffer = pop!(pool.available[pool_size])
            pool.in_use[buffer] = pool_size
            pool.n_pool_hits += 1
            return buffer
        else
            buffer = Vector{T}(undef, pool_size)
            pool.in_use[buffer] = pool_size
            pool.n_fresh_allocations += 1
            pool.total_bytes_allocated += pool_size * sizeof(T)
            return buffer
        end
    end
end

"""
    release_buffer!(pool::FieldBufferPool{T}, buffer::Vector{T})

Return a buffer to the pool. The buffer must have been acquired from this
pool. After release, the caller must not use the buffer again.
"""
function release_buffer!(pool::FieldBufferPool{T}, buffer::Vector{T}) where T
    lock(pool.lock) do
        pool_size = get(pool.in_use, buffer, 0)
        if pool_size == 0
            throw(ArgumentError("release_buffer!: buffer was not acquired from this pool " *
                                  "(or was already released)."))
        end
        delete!(pool.in_use, buffer)
        push!(get!(pool.available, pool_size, Vector{Vector{T}}()), buffer)
    end
    return nothing
end

"""
    pool_stats(pool::FieldBufferPool)

Return a NamedTuple of pool usage statistics:
- `n_acquisitions`: total acquire_buffer! calls
- `n_pool_hits`: served from pool (no allocation)
- `n_fresh_allocations`: had to allocate fresh
- `hit_rate`: pool_hits / acquisitions
- `total_bytes_allocated`: cumulative bytes allocated by this pool
- `n_in_use`: buffers currently checked out
- `n_available`: buffers currently in pool
"""
function pool_stats(pool::FieldBufferPool)
    lock(pool.lock) do
        n_avail = sum(length(v) for v in values(pool.available); init=0)
        return (
            n_acquisitions = pool.n_acquisitions,
            n_pool_hits = pool.n_pool_hits,
            n_fresh_allocations = pool.n_fresh_allocations,
            hit_rate = pool.n_acquisitions > 0 ? pool.n_pool_hits / pool.n_acquisitions : 0.0,
            total_bytes_allocated = pool.total_bytes_allocated,
            n_in_use = length(pool.in_use),
            n_available = n_avail,
        )
    end
end

# ============================================================================
# ScratchBuffer — stack-style allocator
# ============================================================================

"""
    ScratchBuffer{T}

A persistent buffer used as a stack allocator for temporaries. Allocations
move a cursor forward; deallocations move it back (LIFO discipline). Within
the buffer, no fragmentation is possible because allocations are stack-like.

The underlying buffer grows as needed but never shrinks during normal
operation, so after warmup it converges to a steady-state size with no
allocations in the hot path.

Use `with_scratch(scratch, n) do view; ...; end` for safe scoped allocation.
"""
mutable struct ScratchBuffer{T}
    buffer::Vector{T}
    cursor::Int  # number of elements currently in use
end

ScratchBuffer{T}(initial_size::Integer=1024) where T =
    ScratchBuffer{T}(Vector{T}(undef, Int(initial_size)), 0)

"""
    with_scratch(f, scratch::ScratchBuffer{T}, n::Integer)

Allocate `n` elements from the scratch buffer, pass them as a view to `f`,
and release them when `f` returns (even if it throws).

# Example
```julia
scratch = ScratchBuffer{Float64}()
with_scratch(scratch, 100) do view
    for i in eachindex(view)
        view[i] = compute_something(i)
    end
    # use the view
end
```

# Lifetime of the view

The view passed to `f` is a `@view` into `scratch.buffer`. If the
underlying buffer is `resize!`d **while `f` is still running** — for
example, a nested `with_scratch` call inside `f` requested more space
than was free — the outer view becomes invalid (use-after-free).

To use nested allocations safely, ensure `scratch.buffer` is sized for
the *combined* peak usage before the outer call. The simplest way is to
construct the `ScratchBuffer` with `initial_size` large enough for the
worst case, or call `sizehint!(scratch.buffer, max_total_n)` once at
program start. Steady-state operation never resizes once the high-water
mark is established.
"""
function with_scratch(f, scratch::ScratchBuffer{T}, n::Integer) where T
    n = Int(n)
    needed_size = scratch.cursor + n
    if length(scratch.buffer) < needed_size
        # Grow the underlying buffer; this is the only allocation that happens
        # after warmup. Rare event.
        new_size = max(needed_size, 2 * length(scratch.buffer))
        resize!(scratch.buffer, new_size)
    end
    start = scratch.cursor + 1
    stop = scratch.cursor + n
    view_to_use = @view scratch.buffer[start:stop]
    scratch.cursor += n
    try
        return f(view_to_use)
    finally
        scratch.cursor -= n
    end
end

# ============================================================================
# Arena — bulk allocation that's freed all at once
# ============================================================================

"""
    Arena

A bulk-allocation arena. Memory is allocated by moving a cursor forward;
nothing is ever individually freed. The entire arena is reset at the end
of an operation, releasing all allocations at once.

Excellent for per-operation temporaries where individual lifetimes don't
matter and the operation has a clear end-point.

# Example
```julia
arena = Arena(1_000_000)  # 1MB initial size

function step!(...)
    reset_arena!(arena)
    flux_buffer = allocate_in_arena(arena, Float32, n_cells)
    temp_buffer = allocate_in_arena(arena, Float32, n_cells * 4)
    # ... use the buffers
    # No frees needed; reset at next call
end
```
"""
mutable struct Arena
    buffer::Vector{UInt8}
    cursor::Int
end

Arena(initial_bytes::Integer=1_000_000) = Arena(Vector{UInt8}(undef, Int(initial_bytes)), 0)

"""
    allocate_in_arena(arena::Arena, ::Type{T}, n::Integer)

Allocate `n` elements of type `T` from the arena. Returns a `Vector{T}`
backed by `unsafe_wrap` on the arena's buffer.

# Lifetime — IMPORTANT

The returned vector becomes invalid as soon as **any** of the following
happens, after which using it is undefined behaviour (use-after-free):

1. `reset_arena!(arena)` is called.
2. A subsequent `allocate_in_arena` call triggers an internal `resize!`
   of the arena's buffer (when the requested allocation doesn't fit).
3. The arena itself is garbage-collected.

The intended pattern is:

- Reset the arena at the start of an operation.
- Compute the maximum bytes you'll need for that operation in advance and
  call `sizehint!(arena.buffer, max_bytes)` (or pass a large enough
  `initial_bytes` to the `Arena` constructor) so no resize happens during
  the operation.
- Use all returned views within the operation; reset before the next
  operation.

If you can't bound max bytes in advance, use `ScratchBuffer` instead —
its allocations don't invalidate previous views during steady-state use.
"""
function allocate_in_arena(arena::Arena, ::Type{T}, n::Integer) where T
    n = Int(n)
    bytes_needed = sizeof(T) * n
    # Align to T's alignment
    alignment = max(1, Base.datatype_alignment(T))
    misalignment = arena.cursor % alignment
    if misalignment != 0
        arena.cursor += alignment - misalignment
    end

    if arena.cursor + bytes_needed > length(arena.buffer)
        # Grow the underlying buffer
        new_size = max(arena.cursor + bytes_needed, 2 * length(arena.buffer))
        resize!(arena.buffer, new_size)
    end

    ptr = pointer(arena.buffer, arena.cursor + 1)
    arena.cursor += bytes_needed

    # Wrap the bytes as a Vector{T}; this is unsafe and the resulting vector
    # is only valid until the arena is reset or grown
    return unsafe_wrap(Array, Ptr{T}(ptr), n; own=false)
end

"""
    reset_arena!(arena::Arena)

Reset the arena's cursor to zero. All previously-allocated memory in the
arena is now invalid (do not use the previously-returned views).
"""
function reset_arena!(arena::Arena)
    arena.cursor = 0
    return arena
end

end # module Memory
