using Test
using HierarchicalGrids

# ============================================================================
# Backend trait construction & validation
# ============================================================================

@testset "Backend construction & validation" begin
    @test Sequential() isa AbstractParallelBackend

    b = OhMyThreadsBackend()
    @test b isa AbstractParallelBackend
    @test b.scheduler === :dynamic
    @test b.chunksize == 0

    @test OhMyThreadsBackend(:static).scheduler === :static
    @test OhMyThreadsBackend(:greedy, 16).chunksize == 16

    # Invalid scheduler symbol
    @test_throws ArgumentError OhMyThreadsBackend(:foo)
    @test_throws ArgumentError OhMyThreadsBackend(:parallel, 0)
    @test_throws ArgumentError OhMyThreadsBackend(:dynamic, -1)
end

@testset "default_backend / set_default_backend! round-trip" begin
    original = default_backend()
    try
        b1 = OhMyThreadsBackend(:static, 8)
        ret = set_default_backend!(b1)
        @test ret === b1
        @test default_backend() === b1

        b2 = Sequential()
        set_default_backend!(b2)
        @test default_backend() === b2
    finally
        set_default_backend!(original)
    end
    @test default_backend() === original
end

# ============================================================================
# parallel_foreach: byte-equality across backends
# ============================================================================

@testset "parallel_foreach: Sequential matches plain loop" begin
    out = Int[]
    parallel_foreach(Sequential(), x -> push!(out, x), 1:100)
    @test out == collect(1:100)

    # Per-slot mutation (race-free, since each x maps to its own slot)
    buf = zeros(Int, 50)
    parallel_foreach(Sequential(), i -> (buf[i] = i^2), 1:50)
    @test buf == [i^2 for i in 1:50]
end

@testset "parallel_foreach: OhMyThreadsBackend per-slot writes" begin
    # Each x writes to its own slot; safe across schedulers.
    for sch in (:dynamic, :static, :greedy, :serial)
        buf = zeros(Float64, 200)
        parallel_foreach(OhMyThreadsBackend(sch), i -> (buf[i] = sin(Float64(i))),
                         1:200)
        @test buf == [sin(Float64(i)) for i in 1:200]
    end
end

@testset "parallel_foreach: per-task partial-vector accumulation pattern" begin
    # Demonstrates how to safely accumulate a Vector{Int} under any backend
    # via partial-vector + reduce. (Used as a recipe for downstream PRs.)
    iter = 1:1000
    expected = collect(iter)

    # Sequential: trivial — direct push! to a single vector.
    seq_out = Int[]
    parallel_foreach(Sequential(), x -> push!(seq_out, x), iter)
    @test seq_out == expected

    # Parallel-safe accumulation: per-task partials reduced via tmapreduce.
    # We use parallel_mapreduce to materialize per-task vectors and concat.
    parts = parallel_mapreduce(OhMyThreadsBackend(:dynamic),
                               x -> [x], vcat, iter; init = Int[])
    @test sort(parts) == expected
end

# ============================================================================
# parallel_mapreduce: bit-equality for associative-stable reducers
# ============================================================================

@testset "parallel_mapreduce: integer reductions are bit-equal across schedulers" begin
    iter = 1:10_000
    ref = mapreduce(identity, +, iter; init = 0)
    @test parallel_mapreduce(Sequential(), identity, +, iter; init = 0) == ref
    for sch in (:dynamic, :static, :greedy, :serial)
        @test parallel_mapreduce(OhMyThreadsBackend(sch), identity, +, iter;
                                 init = 0) == ref
    end
end

@testset "parallel_mapreduce: float reduction matches sequential reference" begin
    # Float64 + is not strictly associative, so we test for numerical
    # closeness (not bit-equality) across non-:serial schedulers.
    iter = 1:1000
    ref = mapreduce(sin, +, iter; init = 0.0)
    seq = parallel_mapreduce(Sequential(), sin, +, iter; init = 0.0)
    @test seq == ref   # Sequential is mapreduce verbatim, bit-equal.
    for sch in (:dynamic, :static, :greedy, :serial)
        got = parallel_mapreduce(OhMyThreadsBackend(sch), sin, +, iter;
                                 init = 0.0)
        @test isapprox(got, ref; rtol = 1e-12)
    end
end

# ============================================================================
# parallel_chunked: matches sequential equivalent
# ============================================================================

@testset "parallel_chunked: covers every cell exactly once" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3, 4])  # 33 cells

    # Sequential reference: covers every cell exactly once.
    seq_seen = Set{UInt32}()
    parallel_chunked(Sequential(), (m, chunk) -> begin
        for i in chunk.cell_range
            push!(seq_seen, i)
        end
    end, mesh, 4)
    @test length(seq_seen) == n_cells(mesh)
    @test minimum(seq_seen) == 1
    @test maximum(seq_seen) == n_cells(mesh)

    # Parallel: same coverage (use a lock to keep the per-chunk side-
    # effect race-free).
    par_seen = Set{UInt32}()
    par_lock = ReentrantLock()
    parallel_chunked(OhMyThreadsBackend(:dynamic), (m, chunk) -> begin
        local local_seen = collect(chunk.cell_range)
        lock(par_lock) do
            for i in local_seen
                push!(par_seen, i)
            end
        end
    end, mesh, 4)
    @test par_seen == seq_seen
end

@testset "parallel_chunked: per-chunk accumulator (safe pattern)" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3])

    # Compute sum of cell indices via per-chunk partials and a lock.
    n = n_cells(mesh)
    expected = sum(1:n)
    for backend in (Sequential(), OhMyThreadsBackend(:dynamic),
                    OhMyThreadsBackend(:static))
        total = Ref(0)
        total_lock = ReentrantLock()
        parallel_chunked(backend, (m, chunk) -> begin
            local s = 0
            for i in chunk.cell_range
                s += Int(i)
            end
            lock(total_lock) do
                total[] += s
            end
        end, mesh, 4)
        @test total[] == expected
    end
end
