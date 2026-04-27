"""
    Backends

Backend trait for HierarchicalGrids' parallel iteration verbs.

This module provides:

- `AbstractParallelBackend` — the abstract trait type.
- `Sequential` — single-threaded fallback, zero task overhead.
- `OhMyThreadsBackend` — wraps an OhMyThreads scheduler + chunksize.
- Three iteration verbs specialized per backend:
  - `parallel_foreach(backend, f, iter)`
  - `parallel_mapreduce(backend, f, op, iter; init)`
  - `parallel_chunked(backend, f, mesh, n_chunks)`
- Default-backend machinery: `default_backend()`, `set_default_backend!`.

The trait dispatch is designed to be zero-allocation: with both
`Sequential()` and a non-allocating closure, `parallel_foreach` reduces
to a plain `for` loop after specialization.

The existing `parallel_for_cells` / `parallel_for_chunks` /
`parallel_reduce_cells` API is preserved as a thin shim over these
verbs so PR-1/2/3 can opt into the new abstractions incrementally
without breaking any current call sites.
"""

using OhMyThreads: tforeach, tmapreduce, index_chunks

# ============================================================================
# Trait definition
# ============================================================================

"""
    AbstractParallelBackend

Trait root for HG's parallel iteration verbs. Concrete subtypes select
the implementation strategy used by `parallel_foreach`,
`parallel_mapreduce`, and `parallel_chunked`.
"""
abstract type AbstractParallelBackend end

"""
    Sequential <: AbstractParallelBackend

A no-op backend: all parallel verbs reduce to plain sequential loops.
Useful as a fallback (`Threads.nthreads() == 1`), as a reference for
determinism tests, and as the explicit `:serial` scheduler synonym.
"""
struct Sequential <: AbstractParallelBackend end

"""
    OhMyThreadsBackend(scheduler::Symbol = :dynamic, chunksize::Int = 0)

Backend that forwards to `OhMyThreads.jl`. The scheduler must be one
of `:dynamic`, `:static`, `:greedy`, or `:serial` (validated at
construction). When `chunksize > 0` it is forwarded to OhMyThreads as
the per-chunk size; `chunksize == 0` lets OhMyThreads pick.

Note: OhMyThreads also has its own `:serial` scheduler, but for HG the
canonical "single-threaded fallback" backend is `Sequential()`. The
top-level shim layer maps `scheduler = :serial` to `Sequential()`.
"""
struct OhMyThreadsBackend <: AbstractParallelBackend
    scheduler::Symbol
    chunksize::Int

    function OhMyThreadsBackend(scheduler::Symbol, chunksize::Int)
        scheduler in (:dynamic, :static, :greedy, :serial) ||
            throw(ArgumentError(
                "OhMyThreadsBackend: scheduler must be one of " *
                ":dynamic, :static, :greedy, :serial (got $(scheduler))"))
        chunksize >= 0 ||
            throw(ArgumentError(
                "OhMyThreadsBackend: chunksize must be >= 0 (got $(chunksize))"))
        return new(scheduler, chunksize)
    end
end

OhMyThreadsBackend(s::Symbol = :dynamic) = OhMyThreadsBackend(s, 0)

# ============================================================================
# Default-backend machinery
# ============================================================================

const _DEFAULT_BACKEND = Ref{AbstractParallelBackend}(OhMyThreadsBackend(:dynamic, 0))

"""
    default_backend() :: AbstractParallelBackend

Return the current default parallel backend. Used by the top-level
shim API (`parallel_for_cells` and friends) when no explicit `backend`
or `scheduler` argument is supplied.
"""
default_backend() = _DEFAULT_BACKEND[]

"""
    set_default_backend!(b::AbstractParallelBackend) -> b

Replace the process-global default backend. Returns `b`.
"""
function set_default_backend!(b::AbstractParallelBackend)
    _DEFAULT_BACKEND[] = b
    return b
end

# ============================================================================
# Verb 1: parallel_foreach
# ============================================================================

"""
    parallel_foreach(backend, f, iter)

Apply `f(x)` for every `x` in `iter`, in parallel as dictated by
`backend`. Returns `nothing`.
"""
function parallel_foreach end

@inline function parallel_foreach(::Sequential, f::F, iter) where {F}
    @inbounds for x in iter
        f(x)
    end
    return nothing
end

@inline function parallel_foreach(b::OhMyThreadsBackend, f::F, iter) where {F}
    if b.chunksize > 0
        tforeach(iter; scheduler = b.scheduler, chunksize = b.chunksize) do x
            f(x)
        end
    else
        tforeach(iter; scheduler = b.scheduler) do x
            f(x)
        end
    end
    return nothing
end

# ============================================================================
# Verb 2: parallel_mapreduce
# ============================================================================

"""
    parallel_mapreduce(backend, f, op, iter; init)

Apply `f(x)` to every `x` in `iter` and combine the results with `op`,
seeded by `init`. Returns the reduced value. Mirrors the contract of
`Base.mapreduce` but runs in parallel under non-`Sequential` backends.

`op` must be associative. For floating-point reductions, exact byte-
equality across schedulers is not guaranteed; use `Sequential()` (or
the `:serial` scheduler synonym) as the canonical reference.
"""
function parallel_mapreduce end

@inline function parallel_mapreduce(::Sequential, f::F, op::OP, iter; init) where {F, OP}
    return mapreduce(f, op, iter; init = init)
end

@inline function parallel_mapreduce(b::OhMyThreadsBackend, f::F, op::OP, iter; init) where {F, OP}
    if b.chunksize > 0
        return tmapreduce(f, op, iter;
                          init = init,
                          scheduler = b.scheduler,
                          chunksize = b.chunksize)
    else
        return tmapreduce(f, op, iter;
                          init = init,
                          scheduler = b.scheduler)
    end
end

# ============================================================================
# Verb 3: parallel_chunked
# ============================================================================

"""
    parallel_chunked(backend, f, mesh, n_chunks)

Partition the cell range `1:n_cells(mesh)` into `n_chunks`
approximately-equal contiguous ranges and call `f(mesh, chunk)` once
per chunk. `chunk` is a `ThreadChunk` carrying the cell range plus
identifying metadata. Returns `nothing`.

Under `Sequential()`, chunks are processed in order on the calling
thread. Under `OhMyThreadsBackend`, each chunk becomes one task; the
scheduler dictates how those tasks are distributed.
"""
function parallel_chunked end

@inline function parallel_chunked(::Sequential, f::F, mesh, n_chunks::Integer) where {F}
    chunks = partition_for_threads(mesh, n_chunks)
    @inbounds for chunk in chunks
        f(mesh, chunk)
    end
    return nothing
end

@inline function parallel_chunked(b::OhMyThreadsBackend, f::F, mesh, n_chunks::Integer) where {F}
    chunks = partition_for_threads(mesh, n_chunks)
    # `chunking = false` → one task per chunk; we did our own chunking.
    tforeach(eachindex(chunks); scheduler = b.scheduler, chunking = false) do k
        f(mesh, chunks[k])
    end
    return nothing
end
