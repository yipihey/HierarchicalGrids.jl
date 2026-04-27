using Test
using HierarchicalGrids

# ============================================================================
# Dispatch overhead: parallel_foreach(Sequential(), ...) should reduce to a
# plain for-loop after specialization, with zero allocations from the trait
# dispatch itself.
#
# We measure with @allocated against an `identity`-style closure that itself
# does no work and no allocation. Any allocations reported are attributable
# to the dispatch layer.
# ============================================================================

@inline _noop(x) = (x; nothing)

# Warm both methods before timing.
_warm_seq(r)  = parallel_foreach(Sequential(), _noop, r)
_warm_omt(r)  = parallel_foreach(OhMyThreadsBackend(:serial), _noop, r)

@testset "Sequential dispatch: zero allocations" begin
    r = 1:10
    _warm_seq(r); _warm_seq(r)   # warm
    allocs = @allocated _warm_seq(r)
    @test allocs == 0
end

@testset "OhMyThreadsBackend dispatch: bounded allocations" begin
    # OhMyThreads inherently allocates task structures; we just check that
    # the per-call overhead is bounded and doesn't grow with workload size.
    small = 1:10
    big = 1:10_000
    _warm_omt(small); _warm_omt(small)
    _warm_omt(big);   _warm_omt(big)

    a_small = @allocated _warm_omt(small)
    a_big   = @allocated _warm_omt(big)
    # Allow some constant-factor variation, but shouldn't grow ~1000x with
    # the iter size. We assert big <= 4 * small + a small constant.
    @test a_big <= 4 * a_small + 4096
end

@testset "set_default_backend! round-trips correctly" begin
    original = default_backend()
    try
        b = OhMyThreadsBackend(:greedy, 4)
        set_default_backend!(b)
        @test default_backend() === b
        # Can be set to Sequential too.
        set_default_backend!(Sequential())
        @test default_backend() isa Sequential
    finally
        set_default_backend!(original)
    end
end

# ============================================================================
# Bridge byte-equality: existing parallel_for_cells with default_backend()
# tweaks should match the legacy `:dynamic` scheduler.
# ============================================================================

@testset "Backend-driven parallel_for_cells matches scheduler-driven" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3, 4, 5])
    n = n_cells(mesh)

    # Drive via scheduler symbol.
    buf_sch = zeros(Float64, n)
    parallel_for_cells(mesh; scheduler = :dynamic) do _, i
        buf_sch[i] = sin(Float64(i))
    end

    # Drive via explicit backend.
    buf_be = zeros(Float64, n)
    parallel_for_cells(mesh; scheduler = nothing,
                       backend = OhMyThreadsBackend(:dynamic)) do _, i
        buf_be[i] = sin(Float64(i))
    end
    @test buf_be == buf_sch

    # Drive via Sequential() backend.
    buf_seq = zeros(Float64, n)
    parallel_for_cells(mesh; scheduler = nothing, backend = Sequential()) do _, i
        buf_seq[i] = sin(Float64(i))
    end
    @test buf_seq == buf_sch
end
