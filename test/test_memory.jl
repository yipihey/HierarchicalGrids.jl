using Test
using HierarchicalGrids

@testset "FieldBufferPool — basic acquire/release" begin
    pool = FieldBufferPool{Float64}()

    # First acquisition: allocates fresh
    buf1 = acquire_buffer!(pool, 100)
    @test length(buf1) >= 100
    stats = pool_stats(pool)
    @test stats.n_acquisitions == 1
    @test stats.n_pool_hits == 0
    @test stats.n_fresh_allocations == 1

    # Release
    release_buffer!(pool, buf1)
    stats = pool_stats(pool)
    @test stats.n_in_use == 0
    @test stats.n_available == 1

    # Second acquisition of same size: hits the pool
    buf2 = acquire_buffer!(pool, 100)
    stats = pool_stats(pool)
    @test stats.n_acquisitions == 2
    @test stats.n_pool_hits == 1
    @test stats.n_fresh_allocations == 1
    @test stats.hit_rate == 0.5

    release_buffer!(pool, buf2)
end

@testset "FieldBufferPool — size bucketing" begin
    pool = FieldBufferPool{Int32}()

    # Requests for 100 and 110 should bucket to the same size class (128)
    buf1 = acquire_buffer!(pool, 100)
    release_buffer!(pool, buf1)

    buf2 = acquire_buffer!(pool, 110)
    stats = pool_stats(pool)
    @test stats.n_pool_hits == 1  # reused the 128-element buffer
    release_buffer!(pool, buf2)

    # Request for 200 needs a new size class (256)
    buf3 = acquire_buffer!(pool, 200)
    stats = pool_stats(pool)
    @test stats.n_pool_hits == 1  # no hit
    @test stats.n_fresh_allocations == 2
    release_buffer!(pool, buf3)
end

@testset "FieldBufferPool — many cycles maintain pool" begin
    pool = FieldBufferPool{Float32}()

    # Simulate many timesteps acquiring and releasing similar-sized buffers
    for cycle in 1:100
        buffers = [acquire_buffer!(pool, 1000) for _ in 1:5]
        for b in buffers
            release_buffer!(pool, b)
        end
    end

    stats = pool_stats(pool)
    # After warmup, hit rate should be very high
    @test stats.hit_rate > 0.9
    # Number of fresh allocations should be small (just the initial warmup)
    @test stats.n_fresh_allocations <= 5
end

@testset "ScratchBuffer — basic usage" begin
    scratch = ScratchBuffer{Float64}(100)

    # Use a chunk of the buffer
    result = with_scratch(scratch, 50) do view
        @test length(view) == 50
        for i in eachindex(view)
            view[i] = Float64(i)
        end
        sum(view)
    end
    @test result == sum(1:50)

    # After the with_scratch call, cursor should be back to 0
    @test scratch.cursor == 0
end

@testset "ScratchBuffer — nested usage" begin
    scratch = ScratchBuffer{Int}(1000)

    result = with_scratch(scratch, 100) do outer_view
        for i in eachindex(outer_view)
            outer_view[i] = i
        end
        # Cursor should now be at 100
        @test scratch.cursor == 100

        with_scratch(scratch, 50) do inner_view
            @test length(inner_view) == 50
            @test scratch.cursor == 150
            for i in eachindex(inner_view)
                inner_view[i] = i * 10
            end
            # Outer view should still be intact
            @test outer_view[1] == 1
            @test outer_view[100] == 100
        end

        # After inner releases, cursor back to 100
        @test scratch.cursor == 100
        sum(outer_view)
    end

    @test scratch.cursor == 0
    @test result == sum(1:100)
end

@testset "ScratchBuffer — grows when needed" begin
    scratch = ScratchBuffer{Float64}(10)
    initial_capacity = length(scratch.buffer)

    # Request more than initial capacity
    with_scratch(scratch, 100) do view
        @test length(view) == 100
    end

    # Buffer should have grown
    @test length(scratch.buffer) >= 100
end

@testset "Arena — basic allocation" begin
    arena = Arena(10_000)

    # Allocate a few buffers
    a = allocate_in_arena(arena, Float64, 100)
    b = allocate_in_arena(arena, Int32, 200)
    c = allocate_in_arena(arena, Float32, 50)

    @test length(a) == 100
    @test length(b) == 200
    @test length(c) == 50

    # Write and read
    for i in eachindex(a)
        a[i] = Float64(i)
    end
    for i in eachindex(b)
        b[i] = Int32(i * 2)
    end

    @test a[1] == 1.0
    @test a[100] == 100.0
    @test b[1] == 2
    @test b[200] == 400

    # Reset arena
    reset_arena!(arena)
    @test arena.cursor == 0
end

@testset "Arena — grows when needed" begin
    arena = Arena(100)
    initial_capacity = length(arena.buffer)

    # Allocate more than initial capacity
    a = allocate_in_arena(arena, Float64, 100)  # 800 bytes, way more than 100
    @test length(a) == 100
    @test length(arena.buffer) >= 800
end

@testset "Arena — alignment is respected" begin
    arena = Arena(10_000)

    # Allocate a UInt8 buffer first to misalign the cursor
    a = allocate_in_arena(arena, UInt8, 7)  # cursor at 7 after this
    @test length(a) == 7

    # Now allocate Int64 — cursor should be aligned to 8
    b = allocate_in_arena(arena, Int64, 5)
    @test length(b) == 5
    # The pointer to b should be 8-byte aligned
    @test UInt(pointer(b)) % 8 == 0
end
