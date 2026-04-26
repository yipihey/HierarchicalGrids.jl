using Test
using HierarchicalGrids
using HierarchicalGrids.BitPrimitives

@testset "Native bit operations" begin
    @test count_ones_native(UInt8(0b00000000)) == 0
    @test count_ones_native(UInt8(0b11111111)) == 8
    @test count_ones_native(UInt8(0b10101010)) == 4
    @test count_ones_native(UInt32(0xDEADBEEF)) == count_ones(UInt32(0xDEADBEEF))

    @test leading_zeros_native(UInt32(0)) == 32
    @test leading_zeros_native(UInt32(1)) == 31
    @test leading_zeros_native(UInt32(0x80000000)) == 0

    @test trailing_zeros_native(UInt32(0)) == 32
    @test trailing_zeros_native(UInt32(1)) == 0
    @test trailing_zeros_native(UInt32(0b1100)) == 2
end

@testset "PDEP (parallel bit deposit)" begin
    # Basic case from documentation
    @test pdep(UInt32(0b11), UInt32(0b101)) == UInt32(0b101)

    # Edge cases
    @test pdep(UInt32(0), UInt32(0xFFFFFFFF)) == UInt32(0)
    @test pdep(UInt32(0xFFFFFFFF), UInt32(0)) == UInt32(0)

    # For mask = all ones, pdep is identity
    @test pdep(UInt32(0x12345678), UInt32(0xFFFFFFFF)) == UInt32(0x12345678)

    # Specific patterns
    @test pdep(UInt32(0b001), UInt32(0b111)) == UInt32(0b001)
    @test pdep(UInt32(0b010), UInt32(0b111)) == UInt32(0b010)
    @test pdep(UInt32(0b011), UInt32(0b111)) == UInt32(0b011)
    @test pdep(UInt32(0b001), UInt32(0b101)) == UInt32(0b001)  # bit 0 of src -> bit 0 of mask
    @test pdep(UInt32(0b010), UInt32(0b101)) == UInt32(0b100)  # bit 1 of src -> bit 2 of mask

    # 64-bit version
    @test pdep(UInt64(0xFF), UInt64(0xFFFF)) == UInt64(0x00FF)

    # 8-bit and 16-bit promote correctly
    @test pdep(UInt8(0b01), UInt8(0b101)) == UInt8(0b001)
    @test pdep(UInt16(0xF), UInt16(0xFF)) == UInt16(0x0F)
end

@testset "PEXT (parallel bit extract)" begin
    # Basic case
    @test pext(UInt32(0b101), UInt32(0b101)) == UInt32(0b11)

    # Inverse property: pext(pdep(x, m), m) == x (for x within mask range)
    for src in [UInt32(0b00), UInt32(0b01), UInt32(0b10), UInt32(0b11)]
        for mask in [UInt32(0b011), UInt32(0b101), UInt32(0b110), UInt32(0b111)]
            expected_src = src & ((UInt32(1) << count_ones(mask)) - 1)
            @test pext(pdep(src, mask), mask) == expected_src
        end
    end

    # Edge cases
    @test pext(UInt32(0), UInt32(0xFFFFFFFF)) == UInt32(0)
    @test pext(UInt32(0xFFFFFFFF), UInt32(0)) == UInt32(0)
    @test pext(UInt32(0xFFFFFFFF), UInt32(0xFFFFFFFF)) == UInt32(0xFFFFFFFF)
end

@testset "Composite bit operations" begin
    # bit_for_axis
    @test bit_for_axis(UInt8(0b111), 0) == 0  # axis 0 contributes bit 0
    @test bit_for_axis(UInt8(0b111), 1) == 1  # axis 1 contributes bit 1
    @test bit_for_axis(UInt8(0b111), 2) == 2  # axis 2 contributes bit 2

    # Anisotropic: only axes 0 and 2 split
    @test bit_for_axis(UInt8(0b101), 0) == 0  # axis 0 -> bit 0
    @test bit_for_axis(UInt8(0b101), 2) == 1  # axis 2 -> bit 1

    # Anisotropic: only axes 1 and 3 split
    @test bit_for_axis(UInt8(0b1010), 1) == 0  # axis 1 -> bit 0
    @test bit_for_axis(UInt8(0b1010), 3) == 1  # axis 3 -> bit 1

    # nth_set_bit_position
    @test nth_set_bit_position(UInt8(0b10101), 0) == 0
    @test nth_set_bit_position(UInt8(0b10101), 1) == 2
    @test nth_set_bit_position(UInt8(0b10101), 2) == 4

    # positions_of_set_bits
    @test positions_of_set_bits(UInt8(0b10101)) == (0, 2, 4)
    @test positions_of_set_bits(UInt8(0b111)) == (0, 1, 2)
    @test positions_of_set_bits(UInt8(0)) == ()
end

@testset "Type selection by dimension" begin
    # sibling_index_type
    @test sibling_index_type(Val(1)) == UInt8
    @test sibling_index_type(Val(3)) == UInt8
    @test sibling_index_type(Val(7)) == UInt8
    @test sibling_index_type(Val(8)) == UInt16
    @test sibling_index_type(Val(15)) == UInt16
    @test sibling_index_type(Val(16)) == UInt32
    @test sibling_index_type(Val(31)) == UInt32
    @test sibling_index_type(Val(32)) == UInt64

    # split_mask_type matches sibling_index_type
    for D in [1, 3, 7, 8, 15, 16, 31, 32]
        @test split_mask_type(Val(D)) == sibling_index_type(Val(D))
    end

    # vertex_int_type defaults to Int16
    @test vertex_int_type(Val(3)) == Int16

    # volume_int_type sizing
    @test volume_int_type(Val(3), Val(15)) == Int64  # ~50 bits, fits Int64
    @test volume_int_type(Val(4), Val(15)) == Int128 # ~70 bits, needs Int128
    @test volume_int_type(Val(8), Val(15)) == Int128 # ~134 bits, would need >Int128 but we cap at Int128
end

@testset "PDEP/PEXT fallback consistency" begin
    # If hardware has BMI2, the fallback should give the same result.
    # If no BMI2, we're already using the fallback. Either way, this tests
    # that the fallback is correct.
    for src in UInt32[0, 1, 0xFF, 0xFFFF, 0xFFFFFFFF, 0x12345678, 0xDEADBEEF]
        for mask in UInt32[0, 0xF, 0xFF, 0xFFFF, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555]
            @test BitPrimitives.pdep_fallback(src, mask) == pdep(src, mask)
            @test BitPrimitives.pext_fallback(src, mask) == pext(src, mask)
        end
    end
end
