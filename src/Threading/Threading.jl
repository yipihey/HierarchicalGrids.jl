"""
    Threading

Layer 3 (foundational): chunk-based parallelism for shared-memory threading,
designed to mirror future MPI domain decomposition.

The framework partitions work over the cell array (or any other indexable
collection) into chunks. In shared-memory mode each chunk is one task; in
a future MPI mode each chunk would be one process. The logical structure
is the same.

# Backend: OhMyThreads.jl

This module is implemented over `OhMyThreads.jl` for composability,
correct task migration handling, and explicit chunk-based scheduling.

The default scheduler is `:dynamic`, which gives:

- **Composability**: nested parallel calls don't oversubscribe.
- **Load balancing**: chunks are pulled from a queue rather than statically
  assigned, so uneven per-cell work doesn't strand threads.
- **Correctness with task migration**: per-task local storage idioms
  (`OhMyThreads.TaskLocalValue`) work correctly under dynamic scheduling.

For workloads where every cell takes nearly the same time and you want
to avoid the dynamic-scheduling overhead, pass `scheduler = :static`.

# Public API

- `partition_for_threads(mesh, n_chunks)` — explicit `ThreadChunk`
  partition. Used directly by callers who want chunk-level control.
- `parallel_for_cells(f, mesh; scheduler, ...)` — apply `f(mesh, i)`
  to every cell in parallel.
- `parallel_for_chunks(f, mesh, n_chunks; scheduler, ...)` — apply
  `f(mesh, chunk)` per chunk.
- `parallel_reduce_cells(f, op, mesh; init, scheduler, ...)` —
  per-cell reduction with `tmapreduce`.

All three accept `scheduler::Symbol` (`:dynamic`, `:static`, `:greedy`,
`:serial`) and any other keyword arguments OhMyThreads supports
(`nchunks`, `chunksize`, `chunking`, etc.). They forward to the
underlying OhMyThreads scheduler.
"""
module Threading

using ..Mesh
using OhMyThreads: tforeach, tmapreduce

export ThreadChunk, partition_for_threads
export parallel_for_cells, parallel_reduce_cells, parallel_for_chunks

# ============================================================================
# ThreadChunk — the unit of parallel work
# ============================================================================

"""
    ThreadChunk

A range of cell indices assigned to one task (or, in future, one MPI process).

# Fields

- `cell_range::UnitRange{UInt32}` — the contiguous range of cell indices
  owned by this chunk.
- `chunk_id::UInt32` — 1-based identifier within the partition.
- `n_chunks::UInt32` — total number of chunks in this partition.
- `boundary_cells::Vector{UInt32}` — indices of cells in this chunk whose
  neighbors are in other chunks. Empty for now; populated when neighbor
  information is added (reserved for future MPI use and thread-aware
  algorithms that need inter-chunk boundary handling).
"""
struct ThreadChunk
    cell_range::UnitRange{UInt32}
    chunk_id::UInt32
    n_chunks::UInt32
    boundary_cells::Vector{UInt32}
end

ThreadChunk(range::UnitRange, id::Integer, n::Integer) =
    ThreadChunk(UInt32(first(range)):UInt32(last(range)), UInt32(id), UInt32(n), UInt32[])

# ============================================================================
# Partitioning
# ============================================================================

"""
    partition_for_threads(mesh::HierarchicalMesh, n_chunks::Integer = Threads.nthreads())

Partition the mesh's cell array into `n_chunks` chunks of approximately
equal size. Returns a `Vector{ThreadChunk}`.

Currently uses simple equal-size partitioning. Future versions could:
- Respect subtree boundaries (chunks own complete subtrees).
- Balance by per-cell work estimate.
- Optimize for NUMA topology.
"""
function partition_for_threads(mesh::HierarchicalMesh, n_chunks::Integer=Threads.nthreads())
    n_chunks = max(1, Int(n_chunks))
    n = n_cells(mesh)
    n_chunks = min(n_chunks, n)  # don't over-partition

    chunks = Vector{ThreadChunk}(undef, n_chunks)
    base = n ÷ n_chunks
    remainder = n % n_chunks

    start = UInt32(1)
    for k in 1:n_chunks
        size = base + (k <= remainder ? 1 : 0)
        chunks[k] = ThreadChunk(start:(start + UInt32(size) - UInt32(1)), k, n_chunks)
        start += UInt32(size)
    end

    return chunks
end

# ============================================================================
# Parallel iteration
# ============================================================================

"""
    parallel_for_cells(f, mesh::HierarchicalMesh; scheduler = :dynamic, kwargs...)

Apply `f(mesh, i)` for every cell index `i` in parallel.

# Keyword arguments

- `scheduler` — `:dynamic` (default), `:static`, `:greedy`, or `:serial`.
- `nchunks`, `chunksize`, `chunking` — forwarded to the OhMyThreads scheduler.

When `Threads.nthreads() == 1` or `scheduler === :serial`, falls through
to a sequential loop with no task-creation overhead.

# Cache safety

This primitive forces `Mesh.ensure_caches!(mesh)` on the calling thread
before fanning out, so per-task accesses like `find_parent`, `level_of`,
`cell_unit_box`, etc. see a populated, read-only cache. (The lazy-cache
machinery is not thread-safe under concurrent first-access; pre-building
sidesteps the race.)
"""
function parallel_for_cells(f, mesh::HierarchicalMesh;
                              scheduler::Symbol = :dynamic, kwargs...)
    n = n_cells(mesh)
    Mesh.ensure_caches!(mesh)
    if Threads.nthreads() == 1 || scheduler === :serial
        @inbounds for i in 1:n
            f(mesh, i)
        end
        return nothing
    end
    tforeach(1:n; scheduler = scheduler, kwargs...) do i
        f(mesh, i)
    end
    return nothing
end

"""
    parallel_for_chunks(f, mesh::HierarchicalMesh,
                        n_chunks::Integer = Threads.nthreads();
                        scheduler = :dynamic, kwargs...)

Apply `f(mesh, chunk)` once per `ThreadChunk` in parallel. Useful when
the function needs per-chunk state (e.g. a per-task accumulator).

The chunks are constructed via `partition_for_threads(mesh, n_chunks)`,
then dispatched as tasks. With `scheduler = :dynamic`, chunks are pulled
from a queue, so uneven per-chunk work doesn't strand tasks. With
`scheduler = :static`, chunks are statically assigned.

The `OhMyThreads`-level chunking (`nchunks`/`chunksize`) is **ignored**
here — this primitive is for explicit chunk-level control. Pass
`n_chunks` to control partitioning.
"""
function parallel_for_chunks(f, mesh::HierarchicalMesh,
                              n_chunks::Integer = Threads.nthreads();
                              scheduler::Symbol = :dynamic)
    Mesh.ensure_caches!(mesh)
    chunks = partition_for_threads(mesh, n_chunks)
    if Threads.nthreads() == 1 || length(chunks) == 1 || scheduler === :serial
        for chunk in chunks
            f(mesh, chunk)
        end
        return nothing
    end
    # `chunking = false` → exactly one task per chunk (we did our own chunking).
    tforeach(eachindex(chunks); scheduler = scheduler, chunking = false) do k
        f(mesh, chunks[k])
    end
    return nothing
end

"""
    parallel_reduce_cells(f, op, mesh::HierarchicalMesh; init,
                           scheduler = :dynamic, kwargs...)

Apply `f(mesh, i)` to every cell, combining results with `op` starting
from `init`. Returns the final reduction.

Falls through to a sequential loop when `Threads.nthreads() == 1`,
`n_cells(mesh) < 1000`, or `scheduler === :serial` — for tiny meshes the
parallel overhead exceeds the gain.

# `init` and `op` requirements

`op` must be associative. `init` should be the identity element for
`op` and **immutable** (e.g. `0`, `0.0`, `()`); under multi-task
execution OhMyThreads passes the same `init` object to every task, so a
mutable `init` combined with a mutating `op` would race. For mutable
accumulators (e.g. an `OverlapBuilder`), use `OhMyThreads.tmapreduce`
directly without `init` — it'll seed the reduction from one of the
mapped values, giving each task a fresh accumulator naturally.

# Cache safety

Forces `Mesh.ensure_caches!(mesh)` on the calling thread before any
parallel fan-out. See `parallel_for_cells` for rationale.
"""
function parallel_reduce_cells(f, op, mesh::HierarchicalMesh; init,
                                  scheduler::Symbol = :dynamic, kwargs...)
    n = n_cells(mesh)
    Mesh.ensure_caches!(mesh)
    if Threads.nthreads() == 1 || n < 1000 || scheduler === :serial
        result = init
        @inbounds for i in 1:n
            result = op(result, f(mesh, i))
        end
        return result
    end
    return tmapreduce(op, 1:n; init = init, scheduler = scheduler, kwargs...) do i
        f(mesh, i)
    end
end

end # module Threading
