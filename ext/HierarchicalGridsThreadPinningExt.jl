module HierarchicalGridsThreadPinningExt

using HierarchicalGrids
using HierarchicalGrids.Hardware
using ThreadPinning

# Add a NEW method to _pin_threads_impl (no overwrite of the :fallback
# method).
#
# ThreadPinning accepts (among others) the symbols :cores, :numa, :sockets,
# :cputhreads, :firstn, :random, :affinitymask, :current. We map our
# user-facing strategies onto :numa and :cores; :p_cores_only on Linux is
# a documented synonym for :cores (Apple Silicon QoS handling deferred).
function HierarchicalGrids.Hardware._pin_threads_impl(::Val{:thread_pinning}, strategy::Symbol)
    if strategy === :none
        return (strategy_requested = :none, strategy_applied = :none,
                threads_affected = 0, message = "no-op (explicit)")
    end

    # macOS: qos_class_t deferred — :p_cores_only warns and no-ops; other
    # strategies are also no-ops because macOS has uniform memory.
    if Sys.isapple()
        if strategy === :p_cores_only
            @warn "pin_threads!(:p_cores_only) on macOS is a no-op (P/E core control deferred to v2). Recommend `JULIA_NUM_THREADS=<P-core count>` instead."
            return (strategy_requested = :p_cores_only, strategy_applied = :none,
                    threads_affected = 0,
                    message = "macOS P/E control deferred to v2; use JULIA_NUM_THREADS")
        end
        @info "pin_threads!(:$strategy) on macOS: macOS has uniform memory; pinning skipped."
        return (strategy_requested = strategy, strategy_applied = :none,
                threads_affected = 0,
                message = "macOS UMA; pinning not applicable")
    end

    # Linux/Windows path.
    applied = strategy
    if strategy === :auto
        applied = :numa  # best default on Linux multi-socket; degrades to a single domain on UMA.
    elseif strategy === :p_cores_only
        applied = :cores
    end

    try
        ThreadPinning.pinthreads(applied)
        return (strategy_requested = strategy, strategy_applied = applied,
                threads_affected = Threads.nthreads(),
                message = "pinned via ThreadPinning.pinthreads(:$applied)")
    catch err
        @warn "pin_threads!(:$applied) failed" exception = err
        return (strategy_requested = strategy, strategy_applied = :none,
                threads_affected = 0,
                message = "ThreadPinning error: $(typeof(err))")
    end
end

# Flip the Ref at extension load time so pin_threads! routes here.
function __init__()
    HierarchicalGrids.Hardware._PIN_THREADS_BACKEND[] = :thread_pinning
    return nothing
end

end # module
