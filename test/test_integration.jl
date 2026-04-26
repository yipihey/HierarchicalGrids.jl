using Test
using HierarchicalGrids

@testset "Mesh + Storage integration" begin
    # Create a mesh with refinement
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2])

    # Allocate fields parallel to cells (one per cell)
    n = n_cells(mesh)
    fields = allocate_fields(SoA(), n; density=Float32, momentum=NTuple{3, Float32})

    # Initialize fields
    for i in 1:n
        fields.density[i] = Float32(i)
        fields.momentum[i] = (Float32(i), 0.0f0, 0.0f0)
    end

    # Verify
    for i in 1:n
        @test fields.density[i] == Float32(i)
    end

    # Refine more — fields would need to be resized
    initial_n = n_cells(mesh)
    refine_cells!(mesh, [3])
    new_n = n_cells(mesh)

    resize_fields!(fields, new_n)
    @test n_elements(fields) == new_n
    # Old data preserved
    for i in 1:initial_n
        @test fields.density[i] == Float32(i)
    end
end

@testset "Conservation under refinement" begin
    # The classic test: total mass should be conserved under refinement
    # if we distribute the parent's mass equally among children.
    mesh = HierarchicalMesh{3}()
    fields = allocate_fields(SoA(), 1; density=Float64)
    fields.density[1] = 8.0  # mass per unit volume = 8 in unit cube

    initial_total_mass = let
        num, den = HierarchicalGrids.Geometry.cell_volume(mesh, 1)
        fields.density[1] * (num // den)
    end
    @test initial_total_mass == 8

    # Refine the root and distribute mass
    refine_cells!(mesh, [1])
    new_n = n_cells(mesh)
    resize_fields!(fields, new_n)

    # Each child gets the same density as parent (mass-conservative refinement)
    for i in 2:new_n
        fields.density[i] = 8.0
    end

    # Total mass = sum over leaves of density * volume
    total_mass = sum(2:new_n) do i
        if is_leaf(mesh.cells[i])
            num, den = HierarchicalGrids.Geometry.cell_volume(mesh, i)
            fields.density[i] * (num // den)
        else
            0 // 1
        end
    end
    @test total_mass == 8 // 1  # exact!
end

@testset "Threading + Storage integration" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3])
    n = n_cells(mesh)

    fields = allocate_fields(SoA(), n; value=Float64)

    # Initialize in parallel
    parallel_for_cells(mesh) do m, i
        fields.value[i] = Float64(i * i)
    end
    for i in 1:n
        @test fields.value[i] == Float64(i * i)
    end

    # Reduce in parallel
    total = parallel_reduce_cells(+, mesh; init=0.0) do m, i
        fields.value[i]
    end
    @test total ≈ sum(Float64(i * i) for i in 1:n)
end

@testset "Memory pool + Storage" begin
    pool = FieldBufferPool{Float64}()

    # Simulate a workflow: allocate field buffers from the pool repeatedly
    for cycle in 1:10
        buf = acquire_buffer!(pool, 1000)
        for i in 1:1000
            buf[i] = Float64(i + cycle)
        end
        release_buffer!(pool, buf)
    end

    stats = pool_stats(pool)
    # After the first cycle, all subsequent acquisitions should hit the pool
    @test stats.n_pool_hits == 9
    @test stats.n_fresh_allocations == 1
end
