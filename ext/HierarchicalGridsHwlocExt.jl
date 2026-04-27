module HierarchicalGridsHwlocExt

using HierarchicalGrids
using HierarchicalGrids.Hardware
using Hwloc

# Add a NEW method to _topology_summary_impl (no overwrite of the
# :julia_fallback method).
#
# Hwloc.jl's public no-arg API: num_packages() (sockets), num_numa_nodes(),
# num_physical_cores(). On Apple Silicon we still use the Sys.cpu_info()
# frequency heuristic to split P/E counts because libhwloc doesn't always
# expose that distinction directly.
function HierarchicalGrids.Hardware._topology_summary_impl(::Val{:hwloc})
    sockets    = try Hwloc.num_packages()        catch; 1 end
    numa_nodes = try Hwloc.num_numa_nodes()      catch; 1 end
    physical   = try Hwloc.num_physical_cores()  catch; length(Sys.cpu_info()) end
    threads    = Threads.nthreads()
    p_cores = physical
    e_cores = 0
    if Sys.ARCH === :aarch64 && Sys.isapple()
        info = Sys.cpu_info()
        speeds = unique([c.speed for c in info])
        if length(speeds) >= 2
            sort!(speeds, rev = true)
            p_speed = speeds[1]
            e_count = count(c -> c.speed != p_speed, info)
            p_cores = length(info) - e_count
            e_cores = e_count
        end
    end
    return (sockets = sockets, numa_nodes = numa_nodes,
            p_cores = p_cores, e_cores = e_cores,
            threads = threads, source = :hwloc, degraded = false)
end

# Flip the Ref at extension load time so topology_summary() routes here.
function __init__()
    HierarchicalGrids.Hardware._TOPOLOGY_BACKEND[] = :hwloc
    return nothing
end

end # module
