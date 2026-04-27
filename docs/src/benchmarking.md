# Benchmarking

HierarchicalGrids ships a self-contained benchmark harness under
`benchmarks/`. It exists to make two questions cheap to answer:

1. *Did this change make X slower?*
2. *How does the parallel verb scale on this hardware?*

Calibrated baselines for the second question live in
[`benchmarks/PERF.md`](../../benchmarks/PERF.md).

## Running the harness

```sh
julia --project=benchmarks benchmarks/bench_runner.jl
```

That sweeps every (backend × thread count × workload × size) combination
registered in `benchmarks/workloads.jl`. Results are written to
`benchmarks/results/<host>-<sha>-<timestamp>.json`.

For thread counts that differ from the launching process's
`Threads.nthreads()`, the runner spawns a Julia subprocess with
`JULIA_NUM_THREADS=N`, runs that single combination, and collects the
serialized result. A single invocation thus produces a complete cross-thread
sweep — no manual relaunches, no shell loops.

`JULIA_NUM_THREADS` on the *outer* call sets the upper bound: the runner's
`SWEEP.thread_counts` is deduplicated against `Threads.nthreads()`, so
launching with `--threads=8` includes 8 in the sweep, while `--threads=4`
caps it at 4.

For the slimmer Apple Silicon baseline that ships with PR-5, see
`benchmarks/bench_apple_silicon.jl`. It reuses the same workload registry
with `thread_counts = [1, 4]` so the sweep finishes in minutes, not hours.

## Output schema

Each JSON file is a single object with these top-level keys:

```json
{
  "host":          "...",
  "git_sha":       "abcdef1",
  "julia_version": "1.12.x",
  "timestamp":     "...",
  "topology":      { "n_cpus": ..., "model": "...", "machine": "...", "nthreads": ... },
  "results": [
    {
      "workload":  "compute_overlap",
      "size":      "medium",
      "backend":   "OhMyThreadsBackend(:dynamic)",
      "threads":   4,
      "median_ns": 1.234e6,
      "min_ns":    1.111e6,
      "mean_ns":   1.345e6,
      "samples":   8,
      "memory":    1234,
      "allocs":    56
    },
    ...
  ]
}
```

`median_ns` is the canonical metric we cite in `PERF.md`. `min_ns` is more
sensitive to noise floors; `mean_ns` is reported but rarely the right number
to compare across runs.

## Comparing two runs

```sh
julia --project=benchmarks benchmarks/bench_compare.jl baseline.json new.json
```

Output is Markdown — one table per workload, with median timings, speedup
(`baseline / new`, so >1 means the new run is faster), and a `REGRESSION`
flag whenever the new run is more than 5% slower. Pipe to a file if you want
to attach it to a PR description:

```sh
julia --project=benchmarks benchmarks/bench_compare.jl old.json new.json > comparison.md
```

The comparator joins on `(workload, size, backend, threads)`. Combinations
that exist in only one of the two files are flagged but not silently dropped.

## Interpreting `PERF.md`

Each per-workload table looks like:

| Size | Threads | Sequential | OhMyThreads(:dynamic) | OhMyThreads(:static) | OhMyThreads(:greedy) |
|---|---|---|---|---|---|
| medium | 1 | 12.0 ms | 12.4 ms (1.0×) | 12.5 ms (1.0×) | 12.4 ms (1.0×) |
| medium | 4 | n/a    |  3.3 ms (3.6×) |  3.2 ms (3.7×) |  3.5 ms (3.4×) |

`Sequential` only runs at 1 thread — by definition. The other backends'
1-thread numbers measure scheduler overhead vs. a hand-written `for` loop.
At higher thread counts the parenthesized speedup is computed against the
same workload's `Sequential` 1-thread median — i.e. the speedup against the
canonical reference, not against the 1-thread number of the same backend.

When the parallel run is *worse than 1.0×*, the cell flags it explicitly:
`(0.78×, regression)`. Each such row carries a one-line note explaining why
(typically: task-spawn overhead exceeds parallelizable work, or a sequential
inner loop dominates the wall time).

## Adding a new workload

The registry pattern is documented in the comment block at the top of
`benchmarks/workloads.jl`. The contract:

```julia
function build_my_workload(size::Symbol)
    # Build per-call state once. The build cost is NOT measured.
    return (; ...)
end

function run_my_workload(state, backend::AbstractParallelBackend)
    # The hot loop. Wrapped by `BenchmarkTools.@benchmark`.
    return ...   # discarded; we only care about wall time
end

WORKLOADS[:my_workload] = (
    build = build_my_workload,
    run   = run_my_workload,
    sizes = [:small, :medium, :large],
)
```

Then add `:my_workload` to `SWEEP.workloads` in `bench_runner.jl` so it's
included in the cross-thread sweep.

## Contributing baselines

The `PERF.md` shipped with this repository covers Apple Silicon. We
explicitly want contributions for:

- AMD dual-socket NUMA (EPYC, Threadripper)
- Intel Xeon (Sapphire Rapids and friends)
- ARM server (Ampere Altra, Graviton)

If you have access to one of those, the contribution flow is:

1. `JULIA_NUM_THREADS=<physical cores> julia --project=benchmarks benchmarks/bench_runner.jl`
2. Inspect the JSON under `benchmarks/results/`.
3. Curate a new section in `benchmarks/PERF.md` mirroring the Apple Silicon
   format. Include `Sys.cpu_info()` summary and a `topology_summary()` dump.
4. PR with both the JSON and the updated `PERF.md`.

For NUMA boxes, please run with `pin_threads!(:numa)` from
`HierarchicalGrids.Hardware` (requires `using ThreadPinning`) so the
baseline reflects the recommended deployment, not the OS-default placement.

## See also

- [Parallelism](parallelism.md) — backend selection, scheduler guidance,
  hardware-specific recipes.
- [`benchmarks/PERF.md`](../../benchmarks/PERF.md) — the calibrated baseline
  for HG's reference Apple Silicon hardware.
