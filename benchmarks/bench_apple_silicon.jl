#!/usr/bin/env julia
# benchmarks/bench_apple_silicon.jl
#
# Slimmer first-pass driver used to record the PR-5 Apple Silicon baseline.
# Reuses the helpers in `bench_runner.jl` but pins the sweep to thread
# counts [1, 4] across all sizes & backends so the full sweep finishes in
# a reasonable amount of wall time on a laptop.
#
# Usage:
#
#   julia --project=benchmarks --threads=4 \
#         benchmarks/bench_apple_silicon.jl
#
# Output JSON lands at benchmarks/results/<host>-<sha>-<ts>.json, same
# schema as bench_runner.jl.
# ============================================================================

using Pkg
Pkg.activate(@__DIR__)

using BenchmarkTools
using JSON3
using Serialization
using Statistics
using Dates

include(joinpath(@__DIR__, "workloads.jl"))

# --------------------------------------------------------------------------
# Sweep configuration (slimmer than bench_runner.jl's defaults)
# --------------------------------------------------------------------------

const BENCH_SECONDS = 6
const BENCH_SAMPLES = 8
const BENCH_EVALS   = 1

const BACKEND_SPECS = [
    (label = "Sequential",                   build = () -> Sequential()),
    (label = "OhMyThreadsBackend(:dynamic)", build = () -> OhMyThreadsBackend(:dynamic)),
    (label = "OhMyThreadsBackend(:static)",  build = () -> OhMyThreadsBackend(:static)),
    (label = "OhMyThreadsBackend(:greedy)",  build = () -> OhMyThreadsBackend(:greedy)),
]

const SWEEP = (
    workloads     = [:compute_overlap,
                     :polynomial_remap_l_to_e,
                     :polynomial_remap_e_to_l,
                     :init_field_from,
                     :refine_by_indicator,
                     :build_neighbor_graph,
                     :audit_overlap_canonical],
    sizes         = [:small, :medium, :large],
    thread_counts = [1, 4],
)

# --------------------------------------------------------------------------
# Single-combination runner (parallel to bench_runner.jl's run_single)
# --------------------------------------------------------------------------

function run_single(workload::Symbol, size::Symbol, backend_label::String)
    wl = WORKLOADS[workload]
    state = wl.build(size)
    backend = _backend_from_label(backend_label)
    f = wl.run

    f(state, backend)  # warm-up

    bench = @benchmark $f($state, $backend) samples=BENCH_SAMPLES evals=BENCH_EVALS seconds=BENCH_SECONDS

    return Dict(
        "workload"  => String(workload),
        "size"      => String(size),
        "backend"   => backend_label,
        "threads"   => Threads.nthreads(),
        "median_ns" => time(Statistics.median(bench)),
        "min_ns"    => time(minimum(bench)),
        "mean_ns"   => time(Statistics.mean(bench)),
        "samples"   => length(bench.times),
        "memory"    => bench.memory,
        "allocs"    => bench.allocs,
    )
end

function _backend_from_label(label::String)
    for spec in BACKEND_SPECS
        spec.label == label && return spec.build()
    end
    throw(ArgumentError("Unknown backend label: $label"))
end

function run_in_subprocess(workload::Symbol, size::Symbol, backend_label::String,
                            threads::Int)
    tmpfile = tempname() * ".bin"
    # Reuse bench_runner.jl's --single dispatch; it has the same run_single
    # signature so the serialized result is interchangeable.
    runner_script = abspath(joinpath(@__DIR__, "bench_runner.jl"))
    project = abspath(@__DIR__)

    cmd = setenv(
        `$(Base.julia_cmd()) --project=$project --threads=$threads $runner_script
            --single --workload=$(workload) --size=$(size)
            --backend=$(backend_label) --output=$(tmpfile)`,
        copy(ENV);
    )

    run(cmd)
    result = open(tmpfile, "r") do io
        deserialize(io)
    end
    rm(tmpfile; force = true)
    return result
end

function topology_summary()
    cpus = Sys.cpu_info()
    return Dict(
        "n_cpus"   => length(cpus),
        "model"    => isempty(cpus) ? "unknown" : cpus[1].model,
        "nthreads" => Threads.nthreads(),
        "machine"  => Sys.MACHINE,
    )
end

function git_sha()
    try
        return strip(read(`git -C $(@__DIR__)/.. rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function run_sweep()
    results = Dict[]
    current_threads = Threads.nthreads()
    n_total = 0
    for workload in SWEEP.workloads, size in SWEEP.sizes,
            spec in BACKEND_SPECS, threads in SWEEP.thread_counts
        spec.label == "Sequential" && threads != 1 && continue
        n_total += 1
    end

    n_done = 0
    t0 = time()
    for workload in SWEEP.workloads, size in SWEEP.sizes,
            spec in BACKEND_SPECS, threads in SWEEP.thread_counts

        if spec.label == "Sequential" && threads != 1
            continue
        end

        n_done += 1
        elapsed = round(time() - t0; digits = 1)
        @info "[$n_done/$n_total | t+$(elapsed)s] $workload/$size $(spec.label) threads=$threads"
        try
            r = if threads == current_threads
                run_single(workload, size, spec.label)
            else
                run_in_subprocess(workload, size, spec.label, threads)
            end
            push!(results, r)
        catch e
            @warn "Benchmark errored" workload size backend=spec.label threads exception=e
        end
    end

    payload = Dict(
        "host"          => gethostname(),
        "git_sha"       => git_sha(),
        "julia_version" => string(VERSION),
        "timestamp"     => string(Dates.now()),
        "topology"      => topology_summary(),
        "results"       => results,
        "sweep_label"   => "apple_silicon_pr5_baseline",
    )
    return payload
end

function write_results(payload)
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    sha_short = payload["git_sha"]
    fname = "apple-silicon-$(sha_short).json"
    path = joinpath(results_dir, fname)
    open(path, "w") do io
        JSON3.write(io, payload)
    end
    @info "Wrote results" path
    return path
end

function main()
    payload = run_sweep()
    write_results(payload)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
