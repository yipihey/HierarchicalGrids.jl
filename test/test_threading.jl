using Test
using HierarchicalGrids

@testset "ThreadChunk construction" begin
    chunk = HierarchicalGrids.Threading.ThreadChunk(1:100, 1, 4)
    @test chunk.cell_range == 1:100
    @test chunk.chunk_id == 1
    @test chunk.n_chunks == 4
    @test isempty(chunk.boundary_cells)
end

@testset "Partition for threads" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])  # 9 cells
    refine_cells!(mesh, [2, 3, 4])  # +24 cells (3 cells refined, each gets 8 children)
    # Total: 9 + 24 = 33 cells

    chunks = partition_for_threads(mesh, 4)
    @test length(chunks) == 4
    # All cells should be covered by exactly one chunk
    covered = Set{UInt32}()
    for chunk in chunks
        for i in chunk.cell_range
            @test !(i in covered)  # no overlap
            push!(covered, i)
        end
    end
    @test length(covered) == n_cells(mesh)
    @test minimum(covered) == 1
    @test maximum(covered) == n_cells(mesh)
end

@testset "Partition with fewer cells than chunks" begin
    mesh = HierarchicalMesh{3}()  # 1 cell
    chunks = partition_for_threads(mesh, 8)
    # Should not over-partition
    @test length(chunks) <= n_cells(mesh)
    @test length(chunks) == 1
end

@testset "Parallel for cells" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])

    counts = zeros(Int, n_cells(mesh))
    parallel_for_cells(mesh) do m, i
        counts[i] += 1
    end
    @test all(c == 1 for c in counts)
end

@testset "Parallel reduce cells" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3])

    # Sum of cell indices
    total = parallel_reduce_cells(+, mesh; init=0) do m, i
        i
    end
    @test total == sum(1:n_cells(mesh))

    # Count leaves
    n_leaves = parallel_reduce_cells(+, mesh; init=0) do m, i
        is_leaf(m.cells[i]) ? 1 : 0
    end
    expected_leaves = count(is_leaf, mesh.cells)
    @test n_leaves == expected_leaves
end

@testset "Parallel for chunks" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])

    chunk_sizes = Int[]
    chunk_lock = ReentrantLock()
    parallel_for_chunks(mesh, 3) do m, chunk
        lock(chunk_lock) do
            push!(chunk_sizes, length(chunk.cell_range))
        end
    end
    # Sizes should sum to n_cells
    @test sum(chunk_sizes) == n_cells(mesh)
end

# ============================================================================
# Scheduler kwarg coverage (OhMyThreads backend)
#
# The new threading API accepts a `scheduler::Symbol` argument that is
# forwarded to OhMyThreads. All schedulers must produce identical results
# for any pure (race-free) per-cell function.
# ============================================================================

@testset "Scheduler kwarg: all schedulers produce identical results" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3, 4])
    n = n_cells(mesh)

    # parallel_for_cells: each scheduler should produce the same buffer.
    # (Each cell writes to its own slot; no race.)
    schedulers = (:dynamic, :static, :greedy, :serial)
    results = Dict{Symbol, Vector{Float64}}()
    for sch in schedulers
        buf = zeros(Float64, n)
        parallel_for_cells(mesh; scheduler = sch) do _, i
            buf[i] = sin(Float64(i)) + cos(Float64(i)^2)
        end
        results[sch] = buf
    end
    # All schedulers should produce identical buffers
    for sch in schedulers
        @test results[sch] == results[:dynamic]
    end
end

@testset "Scheduler kwarg: parallel_reduce_cells matches across schedulers" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3, 4])  # ~33 cells; below the 1000-cell parallel threshold

    # Below threshold: even with scheduler=:dynamic the implementation
    # falls through to sequential. Confirm the result matches the explicit
    # serial scheduler.
    f(_, i) = Float64(i) * 0.5

    serial_total = parallel_reduce_cells(f, +, mesh; init = 0.0, scheduler = :serial)
    dyn_total    = parallel_reduce_cells(f, +, mesh; init = 0.0, scheduler = :dynamic)
    expected     = sum(f(mesh, i) for i in 1:n_cells(mesh))

    @test serial_total ≈ expected
    @test dyn_total ≈ expected
end

@testset "Scheduler kwarg: :serial short-circuits (no scheduler init cost)" begin
    # Sanity: :serial path returns the right answer for a tiny mesh too,
    # where any task-creation overhead would be visible.
    mesh = HierarchicalMesh{2}()  # one cell, root only
    counter = Ref(0)
    parallel_for_cells(mesh; scheduler = :serial) do _, _
        counter[] += 1
    end
    @test counter[] == 1
end

@testset "parallel_for_cells: callbacks that touch lazy caches don't race" begin
    # Regression: `parallel_for_cells` and friends now pre-build the
    # mesh's lazy caches before fanning out, so a callback that touches
    # `find_parent`/`level_of`/`cell_unit_box`/etc. on a fresh mesh
    # doesn't race the lazy `Mesh.ensure_caches!` resize. This test
    # exercises that path with multiple schedulers.
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])           # 5 cells
    refine_cells!(mesh, [2, 3, 4, 5])  # 21 cells

    for sch in (:dynamic, :static)
        buf = zeros(UInt32, n_cells(mesh))
        parallel_for_cells(mesh; scheduler = sch) do m, i
            # Touch the lazy parent cache from inside a task. Before the
            # fix, the first task to do this triggered ensure_caches! →
            # rebuild_caches! → resize!(_parents, n) and other tasks
            # racing on the same resize crashed with
            # ConcurrencyViolationError. After the fix, the cache was
            # already populated before the parallel section.
            buf[i] = find_parent(m, i)
        end
        # Root (cell 1) has parent = ROOT_PARENT (typemax(UInt32) sentinel).
        # All non-root cells have valid parent indices in 1:n_cells(mesh).
        @test buf[1] == HierarchicalGrids.Mesh.ROOT_PARENT
        for i in 2:n_cells(mesh)
            @test 1 <= buf[i] <= n_cells(mesh)
        end
    end
end

@testset "parallel_reduce_cells: callbacks that touch lazy caches don't race" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3, 4, 5])  # 21 cells (below 1000-cell threshold)

    # With nthreads()>1 and a small mesh, parallel_reduce_cells falls
    # through to serial. Force the parallel path by using a larger mesh.
    big = HierarchicalMesh{2}()
    refine_cells!(big, [1])
    for _ in 1:5
        leaves = enumerate_leaves(big)
        refine_cells!(big, leaves)
    end
    @test n_cells(big) > 1000

    # Reduction whose callback touches the parent cache.
    f(m, i) = level_of(m, i)
    expected_serial = parallel_reduce_cells(f, +, big; init=0, scheduler=:serial)

    for sch in (:dynamic, :static)
        got = parallel_reduce_cells(f, +, big; init=0, scheduler=sch)
        @test got == expected_serial
    end
end
