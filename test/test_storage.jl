using Test
using HierarchicalGrids
using HierarchicalGrids.Storage

@testset "SoA layout — basics" begin
    fields = allocate_fields(SoA(), 100; density=Float32, momentum=NTuple{3, Float32}, energy=Float32)
    @test n_elements(fields) == 100
    @test field_names(fields) == (:density, :momentum, :energy)
    @test has_field(fields, :density)
    @test has_field(fields, :momentum)
    @test !has_field(fields, :temperature)

    # Set and get
    fields.density[1] = 1.5f0
    @test fields.density[1] == 1.5f0

    fields.momentum[1] = (1.0f0, 2.0f0, 3.0f0)
    @test fields.momentum[1] == (1.0f0, 2.0f0, 3.0f0)

    # Independent fields
    fields.energy[1] = 100.0f0
    @test fields.density[1] == 1.5f0  # unchanged
end

@testset "AoS layout — basics" begin
    fields = allocate_fields(AoS(), 100; density=Float32, momentum=NTuple{3, Float32}, energy=Float32)
    @test n_elements(fields) == 100
    @test field_names(fields) == (:density, :momentum, :energy)

    # Set and get
    fields.density[1] = 1.5f0
    @test fields.density[1] == 1.5f0

    fields.momentum[1] = (1.0f0, 2.0f0, 3.0f0)
    @test fields.momentum[1] == (1.0f0, 2.0f0, 3.0f0)

    # All fields of element 1 should be coherent (we wrote them all)
    @test fields.density[1] == 1.5f0  # still 1.5 even after momentum write
end

@testset "SoA vs AoS — same access semantics" begin
    n = 50
    fields_soa = allocate_fields(SoA(), n; x=Float64, y=Float64)
    fields_aos = allocate_fields(AoS(), n; x=Float64, y=Float64)

    # Write the same data to both
    for i in 1:n
        fields_soa.x[i] = Float64(i)
        fields_soa.y[i] = Float64(i * 2)
        fields_aos.x[i] = Float64(i)
        fields_aos.y[i] = Float64(i * 2)
    end

    # Read back: should give identical results regardless of layout
    for i in 1:n
        @test fields_soa.x[i] == fields_aos.x[i]
        @test fields_soa.y[i] == fields_aos.y[i]
        @test fields_soa.x[i] == Float64(i)
    end
end

@testset "Blocked layout" begin
    # Block size 8, inner SoA
    fields = allocate_fields(Blocked{8, SoA}(), 100; density=Float32, momentum=NTuple{3, Float32})
    @test n_elements(fields) == 100

    # Set and get across block boundaries
    for i in 1:100
        fields.density[i] = Float32(i)
    end
    for i in 1:100
        @test fields.density[i] == Float32(i)
    end

    # Write across block boundary specifically
    fields.density[8] = 88.0f0  # last element of first block
    fields.density[9] = 99.0f0  # first element of second block
    @test fields.density[8] == 88.0f0
    @test fields.density[9] == 99.0f0
end

@testset "Blocked layout with non-multiple-of-blocksize" begin
    # 100 elements with block size 8: 12 full blocks + 1 partial block of 4
    fields = allocate_fields(Blocked{8, SoA}(), 100; x=Int32)
    for i in 1:100
        fields.x[i] = Int32(i)
    end
    for i in 1:100
        @test fields.x[i] == Int32(i)
    end
end

@testset "Different block sizes" begin
    for B in [2, 4, 8, 16, 32]
        local fields = allocate_fields(Blocked{B, SoA}(), 50; x=Float32)
        for i in 1:50
            fields.x[i] = Float32(i)
        end
        for i in 1:50
            @test fields.x[i] == Float32(i)
        end
    end
end

@testset "Resize fields" begin
    fields = allocate_fields(SoA(), 10; x=Float32, y=Float32)
    for i in 1:10
        fields.x[i] = Float32(i)
        fields.y[i] = Float32(-i)
    end

    # Grow
    resize_fields!(fields, 20)
    @test n_elements(fields) == 20
    # Old data should still be there for indices 1..10
    for i in 1:10
        @test fields.x[i] == Float32(i)
        @test fields.y[i] == Float32(-i)
    end
    # New entries are uninitialized; just check we can write to them
    for i in 11:20
        fields.x[i] = Float32(i)
    end
    for i in 11:20
        @test fields.x[i] == Float32(i)
    end

    # Shrink
    resize_fields!(fields, 5)
    @test n_elements(fields) == 5
    for i in 1:5
        @test fields.x[i] == Float32(i)
    end
end

@testset "Resize fields: AoS layout" begin
    fields = allocate_fields(AoS(), 10; x=Float32, y=Float32)
    for i in 1:10
        fields.x[i] = Float32(i)
        fields.y[i] = Float32(-i)
    end

    resize_fields!(fields, 15)
    @test n_elements(fields) == 15
    # Existing elements preserved
    for i in 1:10
        @test fields.x[i] == Float32(i)
        @test fields.y[i] == Float32(-i)
    end

    resize_fields!(fields, 4)
    @test n_elements(fields) == 4
    for i in 1:4
        @test fields.x[i] == Float32(i)
    end
end

@testset "Resize fields: Blocked layout grows by appending blocks" begin
    # Regression: previously the Blocked resize path called
    # `_make_storage(IL, block_size, fieldnames(eltype(storage[1])),
    # Tuple(typeof(storage[1])))` which malformed the types argument and
    # threw `MethodError(length, ...)` when the new size required adding
    # a block. After the fix, names/types are threaded through from
    # `resize_fields!` correctly.
    fields = allocate_fields(Blocked{8, SoA}(), 16; density=Float64, vel=Float64)
    for i in 1:16
        fields.density[i] = Float64(i)
        fields.vel[i]     = Float64(2*i)
    end

    # Grow by exactly one block.
    resize_fields!(fields, 24)
    @test n_elements(fields) == 24
    for i in 1:16
        @test fields.density[i] == Float64(i)
        @test fields.vel[i]     == Float64(2*i)
    end
    # New slots writable.
    for i in 17:24
        fields.density[i] = Float64(100 + i)
    end
    for i in 17:24
        @test fields.density[i] == Float64(100 + i)
    end

    # Grow into a partial block (remainder).
    resize_fields!(fields, 28)
    @test n_elements(fields) == 28
    fields.density[28] = -1.0
    @test fields.density[28] == -1.0

    # Shrink (drops trailing blocks).
    resize_fields!(fields, 8)
    @test n_elements(fields) == 8
    @test fields.density[1] == 1.0
end

@testset "FieldView properties" begin
    fields = allocate_fields(SoA(), 10; x=Float32)
    fv = fields.x
    @test length(fv) == 10
    @test eachindex(fv) == 1:10
    @test size(fv) == (10,)
    @test firstindex(fv) == 1
    @test lastindex(fv) == 10
end

@testset "Layout abstraction — kernel reuse" begin
    # The same kernel should work for any layout
    function update_kernel!(fields, dt)
        for i in 1:n_elements(fields)
            old_pos = fields.pos[i]
            vel = fields.vel[i]
            fields.pos[i] = (old_pos[1] + vel[1] * dt,
                            old_pos[2] + vel[2] * dt,
                            old_pos[3] + vel[3] * dt)
        end
    end

    n = 50

    # Try with SoA
    soa = allocate_fields(SoA(), n; pos=NTuple{3, Float32}, vel=NTuple{3, Float32})
    for i in 1:n
        soa.pos[i] = (Float32(i), 0.0f0, 0.0f0)
        soa.vel[i] = (1.0f0, 0.0f0, 0.0f0)
    end

    # Try with AoS
    aos = allocate_fields(AoS(), n; pos=NTuple{3, Float32}, vel=NTuple{3, Float32})
    for i in 1:n
        aos.pos[i] = (Float32(i), 0.0f0, 0.0f0)
        aos.vel[i] = (1.0f0, 0.0f0, 0.0f0)
    end

    # Try with Blocked{8, SoA}
    blocked = allocate_fields(Blocked{8, SoA}(), n; pos=NTuple{3, Float32}, vel=NTuple{3, Float32})
    for i in 1:n
        blocked.pos[i] = (Float32(i), 0.0f0, 0.0f0)
        blocked.vel[i] = (1.0f0, 0.0f0, 0.0f0)
    end

    # Apply the same kernel to all three
    dt = 0.1f0
    update_kernel!(soa, dt)
    update_kernel!(aos, dt)
    update_kernel!(blocked, dt)

    # All three should produce identical results
    for i in 1:n
        @test soa.pos[i] == aos.pos[i]
        @test soa.pos[i] == blocked.pos[i]
        @test soa.pos[i][1] ≈ Float32(i) + 0.1f0
    end
end

@testset "Higher-dimensional element types" begin
    # Particle in 6D phase space
    fields = allocate_fields(SoA(), 10; pos=NTuple{6, Float64}, mass=Float64)
    fields.pos[1] = (1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
    @test fields.pos[1] == (1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
end

@testset "Show method" begin
    fields = allocate_fields(SoA(), 100; density=Float32, momentum=NTuple{3, Float32})
    str = sprint(show, fields)
    @test occursin("FieldSet", str)
    @test occursin("100", str)
    @test occursin("density", str)
    @test occursin("momentum", str)
end
