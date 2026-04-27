#!/usr/bin/env julia
# benchmarks/format_perf_tables.jl
#
# Reads a JSON file from `bench_runner.jl` (or `bench_apple_silicon.jl`)
# and emits the per-workload tables consumed by `benchmarks/PERF.md`.
#
# Usage:
#
#   julia --project=benchmarks benchmarks/format_perf_tables.jl results.json
# ============================================================================

using Pkg
Pkg.activate(@__DIR__)

using JSON3

const BACKEND_ORDER = [
    "Sequential",
    "OhMyThreadsBackend(:dynamic)",
    "OhMyThreadsBackend(:static)",
    "OhMyThreadsBackend(:greedy)",
]

const SIZE_ORDER = ["small", "medium", "large"]

function fmt_ns(ns)
    isnothing(ns) && return "—"
    if ns >= 1e9
        return string(round(ns / 1e9; digits = 3), " s")
    elseif ns >= 1e6
        return string(round(ns / 1e6; digits = 3), " ms")
    elseif ns >= 1e3
        return string(round(ns / 1e3; digits = 2), " µs")
    else
        return string(round(Int, ns), " ns")
    end
end

function format_speedup(speedup)
    isnothing(speedup) && return ""
    if speedup < 0.95
        return " ($(round(speedup; digits = 2))×, regression)"
    else
        return " ($(round(speedup; digits = 2))×)"
    end
end

function build_index(payload)
    out = Dict{Tuple{String, String, String, Int}, Dict{Symbol, Any}}()
    for r in payload["results"]
        key = (String(r["workload"]), String(r["size"]),
               String(r["backend"]), Int(r["threads"]))
        d = Dict{Symbol, Any}()
        for (k, v) in r
            d[Symbol(String(k))] = v
        end
        out[key] = d
    end
    return out
end

function emit_table(io, idx, workload, thread_counts)
    workload_keys = [k for k in keys(idx) if k[1] == workload]
    isempty(workload_keys) && return
    sizes_present = sort(unique(k[2] for k in workload_keys);
                         by = sz -> findfirst(==(sz), SIZE_ORDER))

    println(io, "| Size | Threads | Sequential | OhMyThreads(:dynamic) | OhMyThreads(:static) | OhMyThreads(:greedy) |")
    println(io, "|---|---|---|---|---|---|")

    for sz in sizes_present
        seq_key = (workload, sz, "Sequential", 1)
        seq_med = haskey(idx, seq_key) ? Float64(idx[seq_key][:median_ns]) : nothing

        for th in thread_counts
            cells = String[fmt_ns(seq_med)]
            for be in ["OhMyThreadsBackend(:dynamic)",
                       "OhMyThreadsBackend(:static)",
                       "OhMyThreadsBackend(:greedy)"]
                k = (workload, sz, be, th)
                if haskey(idx, k)
                    med = Float64(idx[k][:median_ns])
                    if seq_med === nothing
                        push!(cells, fmt_ns(med))
                    else
                        speedup = seq_med / med
                        push!(cells, fmt_ns(med) * format_speedup(speedup))
                    end
                else
                    push!(cells, "—")
                end
            end
            seq_cell = th == 1 ? cells[1] : "n/a"
            println(io, "| $sz | $th | $seq_cell | $(cells[2]) | $(cells[3]) | $(cells[4]) |")
        end
    end
    println(io)
end

function main(argv = ARGS)
    if isempty(argv)
        println(stderr, "Usage: format_perf_tables.jl <results.json>")
        exit(2)
    end
    payload = open(argv[1], "r") do io
        JSON3.read(io, Dict{String, Any})
    end
    idx = build_index(payload)

    threads_present = sort(unique(k[4] for k in keys(idx)))

    workloads = sort(unique(k[1] for k in keys(idx)))
    for wl in workloads
        println("### `$wl`")
        println()
        emit_table(stdout, idx, wl, threads_present)
    end

    # Side info
    println("---")
    println("Topology: ", JSON3.write(get(payload, "topology", Dict())))
    println("Julia: ", get(payload, "julia_version", "?"))
    println("Threads tested: ", join(threads_present, ", "))
    println("git_sha: ", get(payload, "git_sha", "?"))
    println("timestamp: ", get(payload, "timestamp", "?"))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
