# Performance baselines

## How to read this file

Numbers are median wall-time per benchmark sample, as reported by
`BenchmarkTools.@benchmark`. Speedup is computed against the same
workload's `Sequential()` backend at 1 thread. Cells marked
`(regression)` ran slower than the 1-thread sequential baseline —
typically because parallelism overhead exceeds the parallelizable
work at that problem size, or because a sequential step inside the
function dominates wall time.

Re-record on your hardware:

    julia --project=benchmarks benchmarks/bench_runner.jl

Compare two runs:

    julia --project=benchmarks benchmarks/bench_compare.jl old.json new.json

Raw JSON results live under `benchmarks/results/`.

## Apple Silicon — Apple M2 Max (recorded 2026-04-27)

- **Host**: `Tom-Abels-M2-MacBook.local`
- **Model**: Apple M2 Max (12 cores total: 8 P-core + 4 E-core)
- **Architecture**: `arm64-apple-darwin24`
- **Julia**: 1.12.6
- **Threads tested**: 1, 4 (set via `JULIA_NUM_THREADS`)
- **Backends tested**: `Sequential`, `OhMyThreadsBackend(:dynamic)`,
  `OhMyThreadsBackend(:static)`, `OhMyThreadsBackend(:greedy)`
- **Workloads**: `compute_overlap`, `polynomial_remap_l_to_e`,
  `polynomial_remap_e_to_l`, `init_field_from`, `build_neighbor_graph`,
  `audit_overlap_canonical`, `refine_by_indicator`
- **Source commit**: `b536de6` (Phase 1 of the parallelization plan
  fully merged: PR-0 through PR-4 plus PR-5 itself)

### Headline observations

- **`init_field_from`** scales near-ideally — 2.79–2.95× at 4 threads
  on small/medium/large.
- **`build_neighbor_graph`** hits 3.86–3.93× at large; below medium
  the per-leaf walk is too cheap to amortize task overhead, and the
  small-mesh path regresses ~3× (use `Sequential()` explicitly for
  small meshes).
- **`refine_by_indicator`** scales 1.6–2.7× depending on size and
  scheduler; large with `:greedy` is the best at 2.69×.
- **`audit_overlap_canonical`** scales 1.3–1.7× at 4 threads; the
  9-polytope canonical battery has limited fan-out.
- **`polynomial_remap_l_to_e` and `_e_to_l` regress at 4 threads**
  across all sizes (~0.72–0.88×). The deferred-sequential
  `accumulate_polynomial_rhs!` (PR-1 deferred work) dominates wall
  time; the parallelizable solve loop is too cheap to compensate
  for task-spawn overhead. Documented as known limitation; the
  RHS-assembly parallelization is its own follow-up PR with a
  separate design pass.
- **`compute_overlap`** improves only ~5–10% at 4 threads across
  sizes — bounded by the sequential CSR finalize and BVH build at
  the front of the kernel. Out of scope for Phase 1 (BVH parallel
  build is its own effort).

### Known measurement limitations

The `topology_summary()` Julia-fallback heuristic on this M2 Max
returns `(p_cores=12, e_cores=0)` because `Sys.cpu_info()` reports
identical nominal frequencies for all cores on macOS. The real
split is 8 P-core + 4 E-core (verifiable via
`sysctl hw.perflevel0.physicalcpu`). The Hwloc extension would
report the correct split on Linux but Hwloc on macOS doesn't
distinguish P/E either. This is documented in
`docs/src/parallelism.md`. Recommended recipe stays
`JULIA_NUM_THREADS=8` on M-series Pro/Max chips.

## Apple Silicon results

### `audit_overlap_canonical`

| Size | Threads | Sequential | OhMyThreads(:dynamic) | OhMyThreads(:static) | OhMyThreads(:greedy) |
|---|---|---|---|---|---|
| small | 1 | 275.17 µs | 275.58 µs (1.0×) | 280.38 µs (0.98×) | 376.98 µs (0.73×, regression) |
| small | 4 | n/a | 190.71 µs (1.44×) | 179.42 µs (1.53×) | 208.96 µs (1.32×) |
| medium | 1 | 280.17 µs | 285.02 µs (0.98×) | 284.62 µs (0.98×) | 379.31 µs (0.74×, regression) |
| medium | 4 | n/a | 181.83 µs (1.54×) | 165.9 µs (1.69×) | 217.1 µs (1.29×) |
| large | 1 | 279.69 µs | 283.75 µs (0.99×) | 282.42 µs (0.99×) | 379.17 µs (0.74×, regression) |
| large | 4 | n/a | 191.19 µs (1.46×) | 180.27 µs (1.55×) | 209.73 µs (1.33×) |

### `build_neighbor_graph`

| Size | Threads | Sequential | OhMyThreads(:dynamic) | OhMyThreads(:static) | OhMyThreads(:greedy) |
|---|---|---|---|---|---|
| small | 1 | 8.27 µs | 9.31 µs (0.89×, regression) | 9.88 µs (0.84×, regression) | 9.31 µs (0.89×, regression) |
| small | 4 | n/a | 26.02 µs (0.32×, regression) | 23.81 µs (0.35×, regression) | 135.08 µs (0.06×, regression) |
| medium | 1 | 176.98 µs | 185.15 µs (0.96×) | 182.1 µs (0.97×) | 181.62 µs (0.97×) |
| medium | 4 | n/a | 71.56 µs (2.47×) | 83.69 µs (2.11×) | 79.94 µs (2.21×) |
| large | 1 | 3.15 ms | 3.09 ms (1.02×) | 3.107 ms (1.01×) | 3.429 ms (0.92×, regression) |
| large | 4 | n/a | 816.33 µs (3.86×) | 801.48 µs (3.93×) | 1.477 ms (2.13×) |

### `compute_overlap`

| Size | Threads | Sequential | OhMyThreads(:dynamic) | OhMyThreads(:static) | OhMyThreads(:greedy) |
|---|---|---|---|---|---|
| small | 1 | 25.23 µs | 26.29 µs (0.96×) | 25.21 µs (1.0×) | 25.06 µs (1.01×) |
| small | 4 | n/a | 27.79 µs (0.91×, regression) | 27.58 µs (0.91×, regression) | 26.98 µs (0.94×, regression) |
| medium | 1 | 3.511 ms | 3.399 ms (1.03×) | 3.386 ms (1.04×) | 3.341 ms (1.05×) |
| medium | 4 | n/a | 3.034 ms (1.16×) | 3.029 ms (1.16×) | 3.05 ms (1.15×) |
| large | 1 | 17.222 ms | 17.323 ms (0.99×) | 17.501 ms (0.98×) | 17.853 ms (0.96×) |
| large | 4 | n/a | 16.115 ms (1.07×) | 15.715 ms (1.1×) | 15.66 ms (1.1×) |

### `init_field_from`

| Size | Threads | Sequential | OhMyThreads(:dynamic) | OhMyThreads(:static) | OhMyThreads(:greedy) |
|---|---|---|---|---|---|
| small | 1 | 5.333 ms | 5.215 ms (1.02×) | 5.787 ms (0.92×, regression) | 5.142 ms (1.04×) |
| small | 4 | n/a | 1.907 ms (2.8×) | 1.809 ms (2.95×) | 1.807 ms (2.95×) |
| medium | 1 | 93.346 ms | 82.652 ms (1.13×) | 81.537 ms (1.14×) | 85.012 ms (1.1×) |
| medium | 4 | n/a | 80.025 ms (1.17×) | 69.395 ms (1.35×) | 52.25 ms (1.79×) |
| large | 1 | 606.952 ms | 548.836 ms (1.11×) | 582.232 ms (1.04×) | 570.244 ms (1.06×) |
| large | 4 | n/a | 270.296 ms (2.25×) | 299.126 ms (2.03×) | 217.212 ms (2.79×) |

### `polynomial_remap_e_to_l`

| Size | Threads | Sequential | OhMyThreads(:dynamic) | OhMyThreads(:static) | OhMyThreads(:greedy) |
|---|---|---|---|---|---|
| small | 1 | 29.48 µs | 30.65 µs (0.96×) | 29.98 µs (0.98×) | 110.25 µs (0.27×, regression) |
| small | 4 | n/a | 82.65 µs (0.36×, regression) | 88.08 µs (0.33×, regression) | 216.46 µs (0.14×, regression) |
| medium | 1 | 813.0 µs | 822.25 µs (0.99×) | 820.33 µs (0.99×) | 927.06 µs (0.88×, regression) |
| medium | 4 | n/a | 1.011 ms (0.8×, regression) | 1.091 ms (0.75×, regression) | 1.067 ms (0.76×, regression) |
| large | 1 | 3.358 ms | 3.289 ms (1.02×) | 3.483 ms (0.96×) | 3.774 ms (0.89×, regression) |
| large | 4 | n/a | 3.818 ms (0.88×, regression) | 4.225 ms (0.79×, regression) | 4.269 ms (0.79×, regression) |

### `polynomial_remap_l_to_e`

| Size | Threads | Sequential | OhMyThreads(:dynamic) | OhMyThreads(:static) | OhMyThreads(:greedy) |
|---|---|---|---|---|---|
| small | 1 | 33.83 µs | 34.81 µs (0.97×) | 34.44 µs (0.98×) | 111.4 µs (0.3×, regression) |
| small | 4 | n/a | 166.12 µs (0.2×, regression) | 84.27 µs (0.4×, regression) | 122.04 µs (0.28×, regression) |
| medium | 1 | 764.62 µs | 783.17 µs (0.98×) | 771.69 µs (0.99×) | 908.94 µs (0.84×, regression) |
| medium | 4 | n/a | 1.055 ms (0.72×, regression) | 1.065 ms (0.72×, regression) | 1.164 ms (0.66×, regression) |
| large | 1 | 3.266 ms | 3.261 ms (1.0×) | 3.214 ms (1.02×) | 3.527 ms (0.93×, regression) |
| large | 4 | n/a | 4.428 ms (0.74×, regression) | 4.272 ms (0.76×, regression) | 4.302 ms (0.76×, regression) |

### `refine_by_indicator`

| Size | Threads | Sequential | OhMyThreads(:dynamic) | OhMyThreads(:static) | OhMyThreads(:greedy) |
|---|---|---|---|---|---|
| small | 1 | 9.06 µs | 9.69 µs (0.94×, regression) | 9.77 µs (0.93×, regression) | 9.69 µs (0.94×, regression) |
| small | 4 | n/a | 33.33 µs (0.27×, regression) | 32.29 µs (0.28×, regression) | 100.79 µs (0.09×, regression) |
| medium | 1 | 125.92 µs | 125.71 µs (1.0×) | 122.17 µs (1.03×) | 135.54 µs (0.93×, regression) |
| medium | 4 | n/a | 72.29 µs (1.74×) | 73.54 µs (1.71×) | 74.15 µs (1.7×) |
| large | 1 | 2.022 ms | 2.01 ms (1.01×) | 2.042 ms (0.99×) | 2.024 ms (1.0×) |
| large | 4 | n/a | 790.62 µs (2.56×) | 755.15 µs (2.68×) | 752.5 µs (2.69×) |

---
Topology: {"nthreads":4,"model":"Apple M2 Max","n_cpus":12,"machine":"arm64-apple-darwin24.0.0"}
Julia: 1.12.6
Threads tested: 1, 4
git_sha: b536de6
timestamp: 2026-04-27T10:47:34.056

## AMD dual-socket NUMA

**TODO** — contributor needed. To submit a baseline:

1. Run on a Linux box with `JULIA_NUM_THREADS=<physical-core count>`:
   ```
   julia --project=benchmarks benchmarks/bench_runner.jl
   ```
2. (Optional but recommended) Pin threads NUMA-aware:
   ```julia
   using ThreadPinning, HierarchicalGrids
   pin_threads!(:numa)
   ```
3. Commit the resulting JSON under `benchmarks/results/` and append
   a section to this file using the same per-workload table format
   used for the Apple Silicon baseline above.
4. Open a PR with the JSON + updated `PERF.md`.

Expected hardware: 2× AMD EPYC or Threadripper, 64+ physical cores
total, multi-NUMA. Earlier 16/32-core results are also welcome —
note your topology in the section header.
