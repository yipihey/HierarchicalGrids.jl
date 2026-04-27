"""
    Hardware

Layer 3 (foundational): hardware introspection and thread-pinning façade.

This submodule provides a stable, best-effort API for asking the runtime
about its execution environment (sockets, NUMA nodes, P/E core counts) and
for opting into NUMA- or core-aware thread pinning. The richer
implementations live in package extensions:

- `HierarchicalGridsHwlocExt` (loaded when `using Hwloc`) — provides
  authoritative topology data via libhwloc.
- `HierarchicalGridsThreadPinningExt` (loaded when `using ThreadPinning`)
  — actually pins threads via `ThreadPinning.pinthreads`.

Without those extensions loaded the API still succeeds: `topology_summary`
returns a degraded `:julia_fallback` report based on `Sys.cpu_info()`, and
`pin_threads!` becomes a documented no-op that hints at how to opt in.

# Apple Silicon caveat

P/E core control on macOS requires the `qos_class_t` API and is deferred
to a future revision. On macOS, `pin_threads!(:p_cores_only)` warns and
no-ops; the recommended workaround is to set
`JULIA_NUM_THREADS=<P-core count>` before launching Julia.

# Public API

- [`topology_summary`](@ref) — best-effort hardware introspection.
- [`pin_threads!`](@ref) — request a thread-pinning strategy.
"""
module Hardware

export topology_summary, pin_threads!

"""
    topology_summary() -> NamedTuple

Best-effort hardware introspection. Always succeeds. Returns a NamedTuple
with the fields:

  (sockets::Int, numa_nodes::Int, p_cores::Int, e_cores::Int,
   threads::Int, source::Symbol, degraded::Bool)

`source` is `:hwloc` if the Hwloc extension is loaded, otherwise
`:julia_fallback`. `degraded == true` indicates the report is best-effort
from `Sys.cpu_info()` rather than authoritative topology data.

On Apple Silicon (`Sys.ARCH === :aarch64 && Sys.isapple()`), `p_cores`
and `e_cores` are populated by parsing `Sys.cpu_info()` (heuristic; P-cores
have higher max frequency than E-cores). On other platforms `e_cores` is
0 (homogeneous cores).
"""
# Implementation dispatch via a Ref{Symbol} so the Hwloc extension's
# __init__ can flip the marker without overwriting any methods. Julia 1.12
# forbids method overwriting during precompilation, so the extension only
# ADDS new methods to `_topology_summary_impl` (with a fresh `Val{:hwloc}`
# signature) and flips this Ref to route there.
const _TOPOLOGY_BACKEND = Ref{Symbol}(:julia_fallback)

function _topology_summary_impl(::Val{:julia_fallback})
    info = Sys.cpu_info()
    threads = Threads.nthreads()
    if Sys.ARCH === :aarch64 && Sys.isapple()
        speeds = unique([c.speed for c in info])
        if length(speeds) >= 2
            sort!(speeds, rev = true)
            p_speed = speeds[1]
            e_count = count(c -> c.speed != p_speed, info)
            p_count = length(info) - e_count
            return (sockets = 1, numa_nodes = 1, p_cores = p_count,
                    e_cores = e_count, threads = threads,
                    source = :julia_fallback, degraded = true)
        end
    end
    return (sockets = 1, numa_nodes = 1, p_cores = length(info),
            e_cores = 0, threads = threads,
            source = :julia_fallback, degraded = true)
end

topology_summary() = _topology_summary_impl(Val(_TOPOLOGY_BACKEND[]))

"""
    pin_threads!(strategy::Symbol = :auto) -> NamedTuple

Request a thread-pinning strategy. Valid `strategy` values:

- `:numa`         — Linux: pin to NUMA-local cores. macOS: no-op (UMA).
- `:cores`        — Linux: round-robin across physical cores. macOS: no-op.
- `:p_cores_only` — Apple Silicon: warns, recommends
  `JULIA_NUM_THREADS=<P-core count>` (P/E control deferred to v2).
  Linux: synonym for `:cores`.
- `:none`         — explicitly do nothing.
- `:auto`         — `:numa` on Linux multi-socket; `:none` elsewhere.

Returns a NamedTuple describing the action taken:

  (strategy_requested::Symbol, strategy_applied::Symbol,
   threads_affected::Int, message::String)

Always succeeds; logs a one-line `@info` when the action is degraded
versus the requested strategy. Without the ThreadPinning extension loaded,
`:numa`, `:cores`, and `:p_cores_only` warn and return `:none`.
"""
# Same Ref-flip pattern for pin_threads! so the ThreadPinning extension's
# __init__ can install richer behavior by adding a `_pin_threads_impl(::Val{:thread_pinning}, ...)`
# method and flipping this Ref.
const _PIN_THREADS_BACKEND = Ref{Symbol}(:fallback)

function _pin_threads_impl(::Val{:fallback}, strategy::Symbol)
    if strategy === :none
        return (strategy_requested = :none, strategy_applied = :none,
                threads_affected = 0, message = "no-op (explicit)")
    end
    if strategy === :auto
        @info "pin_threads!(:auto): ThreadPinning not loaded; for NUMA-aware pinning add `using ThreadPinning`."
        return (strategy_requested = :auto, strategy_applied = :none,
                threads_affected = 0,
                message = "ThreadPinning extension not loaded")
    end
    @info "pin_threads!(:$strategy): ThreadPinning extension not loaded; pinning skipped."
    return (strategy_requested = strategy, strategy_applied = :none,
            threads_affected = 0,
            message = "ThreadPinning extension not loaded")
end

function pin_threads!(strategy::Symbol = :auto)
    strategy ∈ (:auto, :numa, :cores, :p_cores_only, :none) ||
        throw(ArgumentError("strategy must be one of :auto, :numa, :cores, :p_cores_only, :none; got :$strategy"))
    return _pin_threads_impl(Val(_PIN_THREADS_BACKEND[]), strategy)
end

end # module
