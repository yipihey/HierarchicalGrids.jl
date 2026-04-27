# Tests for the Hardware façade (PR-4).
#
# These tests intentionally exercise ONLY the no-extension path: in this
# project's test environment, neither Hwloc nor ThreadPinning is in
# `[extras]`, so the package extensions in `ext/` should not be loaded.
# That keeps the public API contract testable without dragging optional
# native deps (libhwloc, syscall pinning) into CI.
#
# To manually verify the extension paths:
#
#     julia --project=. -e 'using Pkg; Pkg.add("Hwloc")'
#     julia --project=. -e 'using HierarchicalGrids, Hwloc; @show topology_summary()'
#         # expect (..., source = :hwloc, degraded = false)
#
#     julia --project=. -e 'using Pkg; Pkg.add("ThreadPinning")'
#     julia --project=. -e 'using HierarchicalGrids, ThreadPinning; @show pin_threads!(:numa)'
#         # on macOS expect strategy_applied = :none with a UMA message;
#         # on Linux expect strategy_applied = :numa.

using Test
using HierarchicalGrids

@testset "topology_summary (extension absent)" begin
    s = topology_summary()
    @test s isa NamedTuple
    @test haskey(s, :sockets)
    @test haskey(s, :numa_nodes)
    @test haskey(s, :p_cores)
    @test haskey(s, :e_cores)
    @test haskey(s, :threads)
    @test haskey(s, :source)
    @test haskey(s, :degraded)

    @test s.source === :julia_fallback
    @test s.degraded === true
    @test s.sockets >= 1
    @test s.numa_nodes >= 1
    @test s.p_cores >= 1
    @test s.e_cores >= 0
    @test s.threads == Threads.nthreads()

    # Apple Silicon heuristic sanity: P + E cores account for every entry
    # in Sys.cpu_info(), regardless of whether the heterogeneous path was
    # taken.
    if Sys.ARCH === :aarch64 && Sys.isapple()
        @test s.p_cores + s.e_cores == length(Sys.cpu_info())
        @test s.p_cores >= 1
    else
        @test s.e_cores == 0
        @test s.p_cores == length(Sys.cpu_info())
    end
end

@testset "topology_summary type stability" begin
    # The fallback should be type-stable: the return type is a fully
    # concrete NamedTuple of Ints/Symbols/Bools.
    @inferred topology_summary()
end

@testset "pin_threads! (extension absent)" begin
    # :none — explicit no-op, no log.
    r = pin_threads!(:none)
    @test r isa NamedTuple
    @test r.strategy_requested === :none
    @test r.strategy_applied === :none
    @test r.threads_affected == 0
    @test occursin("no-op", r.message)

    # :numa — without ThreadPinning, must degrade to :none and explain.
    r = @test_logs (:info,) match_mode=:any pin_threads!(:numa)
    @test r.strategy_requested === :numa
    @test r.strategy_applied === :none
    @test r.threads_affected == 0
    @test occursin("ThreadPinning", r.message)

    # :cores — same degradation path.
    r = @test_logs (:info,) match_mode=:any pin_threads!(:cores)
    @test r.strategy_requested === :cores
    @test r.strategy_applied === :none
    @test occursin("ThreadPinning", r.message)

    # :p_cores_only — same degradation path without ThreadPinning.
    r = @test_logs (:info,) match_mode=:any pin_threads!(:p_cores_only)
    @test r.strategy_requested === :p_cores_only
    @test r.strategy_applied === :none

    # :auto — same degradation path with a hint.
    r = @test_logs (:info,) match_mode=:any pin_threads!(:auto)
    @test r.strategy_requested === :auto
    @test r.strategy_applied === :none

    # Default argument is :auto.
    r2 = @test_logs (:info,) match_mode=:any pin_threads!()
    @test r2.strategy_requested === :auto
end

@testset "pin_threads! argument validation" begin
    @test_throws ArgumentError pin_threads!(:bogus)
    @test_throws ArgumentError pin_threads!(:invalid)
    @test_throws ArgumentError pin_threads!(:NUMA)  # case-sensitive
end
