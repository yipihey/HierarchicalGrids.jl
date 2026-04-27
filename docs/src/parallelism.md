# Parallelism

HierarchicalGrids.jl ships a small backend trait — `AbstractParallelBackend` —
that every parallelizable verb in the public API understands. Two concrete
backends ship in the box:

- `Sequential()` — the deterministic single-thread fallback. All parallel
  verbs reduce to plain `for` loops; no tasks are spawned, no scheduler
  overhead is paid.
- `OhMyThreadsBackend(scheduler::Symbol = :dynamic)` — forwards each parallel
  verb to [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl).

The default is `OhMyThreadsBackend(:dynamic)` whenever Julia starts with
`Threads.nthreads() > 1`. With `JULIA_NUM_THREADS=1` the default is still the
OhMyThreads backend, but every parallel verb reduces to a serial loop because
OhMyThreads has nothing to schedule.

## Backend selection

You can change the default for a whole session:

```julia
using HierarchicalGrids
set_default_backend!(Sequential())                     # canonical reference
set_default_backend!(OhMyThreadsBackend(:static))      # homogeneous workloads
set_default_backend!(OhMyThreadsBackend(:dynamic))     # default; AMR-friendly
```

Or override per call. Every public verb that does work proportional to
`n_cells(mesh)` (or `n_simplices(lag)`) accepts an explicit `backend` keyword
argument:

| Verb                          | Backend kwarg |
|-------------------------------|---------------|
| `compute_overlap`             | `parallel = true; scheduler = :dynamic` (legacy) or `backend = ...` |
| `polynomial_remap_l_to_e!`    | `backend = ...` |
| `polynomial_remap_e_to_l!`    | `backend = ...` |
| `init_field_from!`            | `backend = ...` |
| `build_neighbor_graph`        | `backend = ...` |
| `audit_overlap`               | `backend = ...` |
| `refine_by_indicator!`        | `backend = ...` |

Verbs that don't take a `backend` kwarg are not (yet) parallelized. Adding
parallelism is local, deliberate, and benchmarked — not a default.

## OhMyThreads schedulers

`OhMyThreadsBackend` carries a `scheduler::Symbol` chosen at construction.
Four values are accepted:

- **`:dynamic`** *(default)*. Tasks are stolen by idle workers as the queue
  drains. The right pick when per-iteration cost is irregular — exactly the
  case for AMR meshes with non-uniform refinement, where a leaf at level 6
  can cost ~64× more than a leaf at level 3.
- **`:static`**. Each task gets a contiguous range of the iteration space at
  creation time and runs to completion. Slightly less overhead than `:dynamic`,
  best for homogeneous workloads where every iteration costs roughly the same
  (e.g. `init_field_from!` on a fully refined uniform mesh).
- **`:greedy`**. Work-stealing variant with a different overhead profile;
  useful to compare against `:dynamic` on a particular workload. In practice
  it tracks `:dynamic` closely on the workloads in `benchmarks/`.
- **`:serial`**. Single-thread fallback. Equivalent to `Sequential()` from
  HG's perspective; prefer `Sequential()` for clarity unless you specifically
  need an `OhMyThreadsBackend` typed object (e.g. for plumbing through code
  that always expects an `OhMyThreadsBackend`).

When in doubt, use `:dynamic` — it's the default for a reason. Fall back to
`Sequential()` for determinism tests or when the parallel infrastructure
itself is suspect.

## Thread-count recipes

### Apple Silicon (macOS)

Apple's M-series chips have asymmetric cores: high-performance "P-cores" and
energy-efficient "E-cores". The OS scheduler decides which physical core
runs each Julia thread; for compute-bound parallel work you want Julia
threads to land on P-cores. The blunt-but-effective recipe:

```sh
JULIA_NUM_THREADS=<P-core count> julia --project ...
```

Approximate P-core counts (verify with `topology_summary()`):

| Chip                       | P-cores | E-cores |
|----------------------------|--------:|--------:|
| M1 / M2 / M3 (vanilla)     | 4       | 4       |
| M2 Max                     | 8       | 4       |
| M1 / M2 / M3 Pro           | 6 – 8   | 2 – 4   |
| M3 Max                     | 12      | 4       |
| M1 / M2 Ultra              | 16      | 4 – 8   |

Pinning is currently a no-op on macOS. The kernel exposes Quality-of-Service
classes (`qos_class_t`) rather than CPU affinity, and the
`p_cores_only` strategy is deferred to a future revision:

```julia
julia> using HierarchicalGrids.Hardware
julia> pin_threads!(:p_cores_only)
[ Info: pin_threads!(:p_cores_only): Apple Silicon — pinning deferred to v2.
       Recommend: JULIA_NUM_THREADS=<P-core count>
```

### AMD dual-socket NUMA (Linux)

Two-socket EPYC and Threadripper boxes have NUMA nodes whose memory bandwidth
falls off a cliff across the inter-socket fabric. The recipe:

```sh
JULIA_NUM_THREADS=<physical cores> julia --project ...
```

Then in-Julia, after `using HierarchicalGrids` and `using ThreadPinning` (a
weak dependency of HG via the package extension):

```julia
using ThreadPinning   # bring in the optional pinning extension
using HierarchicalGrids.Hardware
pin_threads!(:numa)   # pin threads to NUMA-local cores
```

`ThreadPinning` is a `[weakdep]`; it must be added explicitly to the user's
project (`Pkg.add("ThreadPinning")`) before HG's pinning extension activates.
With pinning in place, expect near-linear scaling on coarse-grained workloads
(`init_field_from!`, `build_neighbor_graph`).

### Single-thread baseline

```sh
JULIA_NUM_THREADS=1 julia --project ...
```

This skips the parallel infrastructure entirely. Useful for debugging,
reproducibility, and bisecting whether a regression is in the kernel or in
the threading layer.

## Composition rule: do not nest backends

HG's parallel verbs read `default_backend()` (or the explicit `backend`
kwarg) at the top of each public call. They do **not** propagate that
choice back to the caller's own threads.

The practical implication: if you call `compute_overlap` from inside your
own `Threads.@spawn` or `tforeach`, you must pass `backend = Sequential()`
to avoid oversubscribing the machine. Otherwise each outer task starts an
inner pool, and the OS thread scheduler thrashes.

```julia
# inside a parallel outer loop the user manages
Threads.@threads for i in eachindex(my_problems)
    # Force the inner verb to single-thread; the outer loop is the
    # parallelism source.
    overlap = compute_overlap(lag[i], frame[i]; parallel = false)
    ...
end
```

A clean alternative — and the one HG prefers internally — is to pick one
level of parallelism and stick with it. Two-level nested parallelism is a
performance trap on every shared-memory machine we've measured.

## Per-task scratch idiom (for contributors)

When a parallel kernel needs per-task scratch — a buffer, a builder, a
candidate vector — the right idiom is **"allocate fresh inside the task
body"** or **`OhMyThreads.TaskLocalValue`**. Do **not** use a
`Vector`-of-`threadid()`-indexed-buffers: under task migration this is racy.

The canonical example is the parallel overlap kernel at
`src/Overlap/compute.jl` (`_compute_overlap_parallel`):

```julia
final_builder = OhMyThreads.tmapreduce(
    merge_builder!,
    chunk_ranges;
    scheduler = scheduler,
) do chunk_range
    builder     = OverlapBuilder{D, T}(moment_order)   # per-task scratch
    scratch     = PairScratch(Val(D), T)
    moments_buf = Vector{T}(undef, n_phys)
    candidates  = Int32[]; sizehint!(candidates, 64)
    for k in chunk_range
        ...
    end
    builder
end
```

One allocation per task (not per leaf) keeps GC pressure tiny. Use this
pattern in any new parallel kernel that needs scratch.

## Cache pre-build before parallel access

Several mesh-internal caches are built lazily on first access:
`cell_unit_box`, `parent_index`, `subtree_size`, the neighbor graph. If two
tasks both touch a fresh cache for the first time concurrently, the lazy
builder fires twice and the cache state can race.

The fix is one line at the top of the kernel:

```julia
HierarchicalGrids.Mesh.ensure_caches!(mesh)
parallel_foreach(backend, ...) do ci
    # safe: caches are pre-built
    ...
end
```

PR-1 hit this race in `init_field_from!` for the `cell_unit_box` cache. If
you write a new parallel kernel that touches mesh internals, **assume nothing
is built** and call `ensure_caches!(mesh)` first.

## See also

- [Benchmarking](benchmarking.md) — running the harness, comparing runs,
  contributing baselines.
- [Architecture](architecture.md) — the layered design that separates the
  threading concern (Layer 3) from the geometric kernels (Layer 4+).
