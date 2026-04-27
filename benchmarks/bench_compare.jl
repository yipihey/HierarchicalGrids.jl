#!/usr/bin/env julia
# benchmarks/bench_compare.jl
#
# Loads two JSON files produced by `bench_runner.jl` and prints a Markdown
# comparison table per workload. Speedup is `baseline_median / new_median`
# (>1 means the new run is faster). A regression flag is printed when the
# new run is >5% slower than the baseline.
#
# Usage:
#
#   julia --project=benchmarks benchmarks/bench_compare.jl <baseline.json> <new.json>
#
# Output goes to stdout; pipe to a file as needed.
# ============================================================================

using Pkg
Pkg.activate(@__DIR__)

using JSON3

const REGRESSION_THRESHOLD = 0.05  # 5%

function load(path)
    return open(path, "r") do io
        JSON3.read(io, Dict{String, Any})
    end
end

function key(r)
    return (String(r["workload"]), String(r["size"]),
            String(r["backend"]), Int(r["threads"]))
end

function index_results(payload)
    out = Dict{Tuple{String, String, String, Int}, Dict{String, Any}}()
    for r in payload["results"]
        d = Dict{String, Any}(String(k) => v for (k, v) in r)
        out[key(d)] = d
    end
    return out
end

function fmt_ns(ns)
    if ns >= 1e9
        return string(round(ns / 1e9; digits = 3), " s")
    elseif ns >= 1e6
        return string(round(ns / 1e6; digits = 3), " ms")
    elseif ns >= 1e3
        return string(round(ns / 1e3; digits = 3), " µs")
    else
        return string(round(Int, ns), " ns")
    end
end

function compare(baseline_path, new_path)
    base_payload = load(baseline_path)
    new_payload  = load(new_path)
    base_idx = index_results(base_payload)
    new_idx  = index_results(new_payload)

    println("# Benchmark comparison")
    println()
    println("- baseline: `$baseline_path` (sha=$(get(base_payload, "git_sha", "?")))")
    println("- new:      `$new_path` (sha=$(get(new_payload, "git_sha", "?")))")
    println()

    workloads = sort(collect(unique(k[1] for k in keys(new_idx))))
    for wl in workloads
        println("## Workload: $wl")
        println()
        println("| Size | Backend | Threads | Baseline median | New median | Speedup | Flag |")
        println("|------|---------|---------|------------------|-------------|---------|------|")

        for k in sort(collect(keys(new_idx)))
            (kwl, sz, be, th) = k
            kwl == wl || continue
            new_r  = new_idx[k]
            base_r = get(base_idx, k, nothing)
            if base_r === nothing
                println("| $sz | $be | $th | (missing) | $(fmt_ns(new_r["median_ns"])) | – | new |")
                continue
            end
            base_med = Float64(base_r["median_ns"])
            new_med  = Float64(new_r["median_ns"])
            speedup  = base_med / new_med
            flag = if speedup < (1.0 - REGRESSION_THRESHOLD)
                "REGRESSION"
            elseif speedup > (1.0 + REGRESSION_THRESHOLD)
                "improvement"
            else
                ""
            end
            println("| $sz | $be | $th | $(fmt_ns(base_med)) | $(fmt_ns(new_med)) | $(round(speedup; digits = 2))× | $flag |")
        end
        println()
    end
end

function main(argv = ARGS)
    if length(argv) != 2
        println(stderr, "Usage: bench_compare.jl <baseline.json> <new.json>")
        exit(2)
    end
    compare(argv[1], argv[2])
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
