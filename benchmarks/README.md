# HierarchicalGrids Benchmarks

A self-contained sweep harness for the parallel verbs introduced in PR-0
(`parallel_foreach`, `parallel_mapreduce`, `parallel_chunked`) plus the
existing `compute_overlap` workload. Future PRs (PR-1/2/3) register
their workloads in `workloads.jl`.

## One-time setup

```sh
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
```

This installs `BenchmarkTools`, `JSON3`, `Statistics`, and `dev`-links
the local copy of `HierarchicalGrids` so the harness measures the
working tree.

## Running a sweep

```sh
julia --project=benchmarks benchmarks/bench_runner.jl
```

The runner sweeps:

- **Backends:** `Sequential`, `OhMyThreadsBackend(:dynamic | :static | :greedy)`.
- **Thread counts:** `[1, 2, 4, 8, max(8, Threads.nthreads())]`, deduplicated.
- **Workloads × sizes:** see `workloads.jl`. PR-0 ships only
  `:compute_overlap` at sizes `:small`, `:medium`, `:large`.

For thread counts that differ from the launching process's
`Threads.nthreads()`, the runner spawns a Julia subprocess with
`JULIA_NUM_THREADS=N`, runs that single combination, and collects the
serialized result. This means a single invocation of `bench_runner.jl`
produces a complete cross-thread-count sweep without manual relaunches.

Results land in:

```
benchmarks/results/<host>-<git-sha-short>-<timestamp>.json
```

These JSON files are **not** committed in PR-0 — calibrated baselines
ship in PR-5.

## Comparing two runs

```sh
julia --project=benchmarks benchmarks/bench_compare.jl baseline.json new.json
```

Prints a Markdown table per workload with median timings, speedup, and
a `REGRESSION` flag when the new run is more than 5% slower.

Pipe to a file if you want to commit the report:

```sh
julia --project=benchmarks benchmarks/bench_compare.jl baseline.json new.json > comparison.md
```

## Adding a new workload

See the comment block at the top of `benchmarks/workloads.jl`. The
short version:

1. Write `build_<name>(size::Symbol)` returning per-call state.
2. Write `run_<name>(state, backend::AbstractParallelBackend)`.
3. Add an entry to `WORKLOADS`.
4. Reference the workload symbol in `bench_runner.jl`'s `SWEEP.workloads`.

PR-1 will add `:polynomial_remap_l_to_e`, PR-2 `:init_field_from`, and
PR-3 `:overlap_audit` — each one drops in by following these four steps.
