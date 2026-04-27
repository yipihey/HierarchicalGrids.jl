#!/usr/bin/env julia
# benchmarks/bench_runner.jl
#
# Driver script that sweeps (backend × thread_count × workload × size)
# combinations, runs each via BenchmarkTools, and emits a JSON document
# under `benchmarks/results/`.
#
# Usage:
#
#   julia --project=benchmarks benchmarks/bench_runner.jl
#
# The script relaunches a Julia subprocess (with JULIA_NUM_THREADS=N) when
# the requested thread count differs from the current one, then collects
# its result via a temp file. The subprocess invokes the same script with
# `--single` and a configuration JSON describing exactly one combination.
#
# To add a new workload, edit `workloads.jl` and add an entry to `WORKLOADS`.
# Then list its symbol below in the SWEEP `workloads` array.
# ============================================================================

using Pkg
Pkg.activate(@__DIR__)

using BenchmarkTools
using JSON3
using Serialization
using Statistics
using Dates

include(joinpath(@__DIR__, "workloads.jl"))

# ============================================================================
# Sweep configuration
# ============================================================================

const BENCH_SECONDS = 10
const BENCH_SAMPLES = 10
const BENCH_EVALS   = 1

# Backends to sweep. We include `Sequential()` as a baseline plus the
# three OhMyThreads schedulers worth comparing (`:greedy` is included for
# load-balanced workloads; `:serial` is omitted because `Sequential()`
# already covers that contract).
const BACKEND_SPECS = [
    (label = "Sequential",                build = () -> Sequential()),
    (label = "OhMyThreadsBackend(:dynamic)", build = () -> OhMyThreadsBackend(:dynamic)),
    (label = "OhMyThreadsBackend(:static)",  build = () -> OhMyThreadsBackend(:static)),
    (label = "OhMyThreadsBackend(:greedy)",  build = () -> OhMyThreadsBackend(:greedy)),
]

const SWEEP = (
    workloads     = [:compute_overlap, :refine_by_indicator],
    sizes         = [:small, :medium, :large],
    thread_counts = unique([1, 2, 4, 8, max(8, Threads.nthreads())]),
)

# ============================================================================
# Single-combination runner (used both in-process and from subprocess)
# ============================================================================

"""
    run_single(workload::Symbol, size::Symbol, backend_label::String)

Run BenchmarkTools.@benchmark for one (workload, size, backend) combination
using the current process's `Threads.nthreads()`. Returns a `Dict` of
result fields.
"""
function run_single(workload::Symbol, size::Symbol, backend_label::String)
    wl = WORKLOADS[workload]
    state = wl.build(size)
    backend = _backend_from_label(backend_label)
    f = wl.run

    # Warm-up.
    f(state, backend)

    bench = @benchmark $f($state, $backend) samples=BENCH_SAMPLES evals=BENCH_EVALS seconds=BENCH_SECONDS

    return Dict(
        "workload"   => String(workload),
        "size"       => String(size),
        "backend"    => backend_label,
        "threads"    => Threads.nthreads(),
        "median_ns"  => time(Statistics.median(bench)),
        "min_ns"     => time(minimum(bench)),
        "mean_ns"    => time(Statistics.mean(bench)),
        "samples"    => length(bench.times),
        "memory"     => bench.memory,
        "allocs"     => bench.allocs,
    )
end

function _backend_from_label(label::String)
    for spec in BACKEND_SPECS
        spec.label == label && return spec.build()
    end
    throw(ArgumentError("Unknown backend label: $label"))
end

# ============================================================================
# Subprocess relaunch (for differing thread counts)
# ============================================================================

"""
    run_in_subprocess(workload, size, backend_label, threads) -> Dict

Spawn a fresh Julia process with `JULIA_NUM_THREADS=threads`, ask it to
run the single-combination benchmark, and read back the serialized result.
"""
function run_in_subprocess(workload::Symbol, size::Symbol, backend_label::String,
                            threads::Int)
    tmpfile = tempname() * ".bin"
    script = abspath(@__FILE__)
    project = abspath(@__DIR__)

    cmd = setenv(
        `$(Base.julia_cmd()) --project=$project --threads=$threads $script
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

# ============================================================================
# Top-level sweep
# ============================================================================

function topology_summary()
    cpus = Sys.cpu_info()
    return Dict(
        "n_cpus"     => length(cpus),
        "model"      => isempty(cpus) ? "unknown" : cpus[1].model,
        "nthreads"   => Threads.nthreads(),
        "machine"    => Sys.MACHINE,
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

    for workload in SWEEP.workloads, size in SWEEP.sizes,
            spec in BACKEND_SPECS, threads in SWEEP.thread_counts

        # Sequential backend is single-threaded by definition; only run at
        # threads = 1 to avoid duplicating identical results.
        if spec.label == "Sequential" && threads != 1
            continue
        end

        @info "Running" workload size backend=spec.label threads
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
    )

    return payload
end

function write_results(payload)
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    sha_short = payload["git_sha"]
    ts = replace(payload["timestamp"], r"[:.]" => "-")
    fname = "$(payload["host"])-$(sha_short)-$(ts).json"
    path = joinpath(results_dir, fname)
    open(path, "w") do io
        JSON3.write(io, payload)
    end
    @info "Wrote results" path
    return path
end

# ============================================================================
# CLI dispatch
# ============================================================================

function parse_kvargs(argv)
    out = Dict{String, String}()
    flags = Set{String}()
    for a in argv
        if startswith(a, "--") && occursin('=', a)
            k, v = split(a[3:end], '='; limit = 2)
            out[String(k)] = String(v)
        elseif startswith(a, "--")
            push!(flags, String(a[3:end]))
        end
    end
    return flags, out
end

function main(argv = ARGS)
    flags, kv = parse_kvargs(argv)

    if "single" in flags
        wl   = Symbol(kv["workload"])
        sz   = Symbol(kv["size"])
        be   = kv["backend"]
        out  = kv["output"]
        result = run_single(wl, sz, be)
        open(out, "w") do io
            serialize(io, result)
        end
        return
    end

    payload = run_sweep()
    write_results(payload)
    return
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
