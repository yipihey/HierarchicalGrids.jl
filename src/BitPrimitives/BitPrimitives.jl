"""
    BitPrimitives

Hardware bit operations and integer type selection for the HierarchicalGrids
framework. All operations are pure functions, no allocations.

This module is the only place that touches hardware-specific bit manipulation
intrinsics (POPCNT, PDEP, PEXT). Substituting fallback implementations for
non-x86 hardware is a one-place change.

# Type selection by dimension

The `sibling_index_type(D)` and `split_mask_type(D)` functions choose
appropriate unsigned integer types based on dimension. For D ≤ 7, UInt8
suffices (covers up to 2^7 = 128 children); for D ≤ 15, UInt16; etc.

The `volume_int_type(D, vertex_bits)` function chooses an integer wide
enough to hold an exact volume computed from vertices of the given bit
width, with ~6 bits of headroom for small sums. For D=3 with Int16
vertices, returns `Int64` (54 bits used out of 63); for D=4 it bumps to
`Int128`.
"""
module BitPrimitives

export count_ones_native, leading_zeros_native, trailing_zeros_native
export pdep, pext, bit_for_axis
export sibling_index_type, split_mask_type, volume_int_type, vertex_int_type
export nth_set_bit_position, positions_of_set_bits

# ============================================================================
# Hardware capability detection
# ============================================================================

"""
    HAS_BMI2

Whether the host CPU supports BMI2 instructions (PDEP, PEXT). Set at
module load time. On non-x86 hardware, always false; software fallback is used.
"""
const HAS_BMI2 = let
    if Sys.ARCH == :x86_64 || Sys.ARCH == :i686
        # Try to detect BMI2; conservative default is false
        try
            # PDEP is BMI2; if it exists as an LLVM intrinsic and we can call it
            test_result = ccall("llvm.x86.bmi.pdep.32", llvmcall, UInt32,
                                (UInt32, UInt32), UInt32(1), UInt32(1))
            test_result == UInt32(1)
        catch
            false
        end
    else
        false
    end
end

# ============================================================================
# Native bit operations (re-exports for clarity, with naming convention)
# ============================================================================

"""
    count_ones_native(x::Unsigned)

Population count: number of set bits in x. Maps to POPCNT on x86,
equivalent on ARM. Single-cycle to ~3-cycle operation depending on hardware.
"""
@inline count_ones_native(x::Unsigned) = count_ones(x)

"""
    leading_zeros_native(x::Unsigned)

Number of leading zero bits. Maps to LZCNT on x86, CLZ on ARM.
"""
@inline leading_zeros_native(x::Unsigned) = leading_zeros(x)

"""
    trailing_zeros_native(x::Unsigned)

Number of trailing zero bits. Maps to TZCNT on x86, RBIT+CLZ on ARM.
"""
@inline trailing_zeros_native(x::Unsigned) = trailing_zeros(x)

# ============================================================================
# PDEP / PEXT — parallel bit deposit and extract
# ============================================================================

"""
    pdep(src::Unsigned, mask::Unsigned)

Parallel bit deposit. For each bit set in `mask`, take the next bit from
`src` (starting from the low bit) and place it at that position. Other bits
of the result are zero.

Used in our framework to scatter sibling_index bits across the axes
indicated by split_mask. For example, with split_mask = 0b101 (axes 0 and 2
split, axis 1 not), and sibling_index = 0b11 (lower bit for axis 0 = 1,
next bit for axis 2 = 1), pdep(0b11, 0b101) = 0b101.

# Example
```jldoctest
julia> pdep(UInt32(0b11), UInt32(0b101))
0x00000005
```
"""
@inline function pdep(src::UInt32, mask::UInt32)
    if HAS_BMI2
        return ccall("llvm.x86.bmi.pdep.32", llvmcall, UInt32, (UInt32, UInt32), src, mask)
    else
        return pdep_fallback(src, mask)
    end
end

@inline function pdep(src::UInt64, mask::UInt64)
    if HAS_BMI2
        return ccall("llvm.x86.bmi.pdep.64", llvmcall, UInt64, (UInt64, UInt64), src, mask)
    else
        return pdep_fallback(src, mask)
    end
end

# Smaller types promote to UInt32
@inline pdep(src::UInt8, mask::UInt8) = UInt8(pdep(UInt32(src), UInt32(mask)))
@inline pdep(src::UInt16, mask::UInt16) = UInt16(pdep(UInt32(src), UInt32(mask)))

"""
    pext(src::Unsigned, mask::Unsigned)

Parallel bit extract. Inverse of pdep. For each bit set in `mask`, take the
bit at that position from `src` and place it consecutively in the low bits
of the result.

Used in our framework to compute sibling_index from a per-axis position
vector. With split_mask = 0b101 and position = 0b101 (axes 0 and 2 set,
axis 1 unset), pext(0b101, 0b101) = 0b11.

# Example
```jldoctest
julia> pext(UInt32(0b101), UInt32(0b101))
0x00000003
```
"""
@inline function pext(src::UInt32, mask::UInt32)
    if HAS_BMI2
        return ccall("llvm.x86.bmi.pext.32", llvmcall, UInt32, (UInt32, UInt32), src, mask)
    else
        return pext_fallback(src, mask)
    end
end

@inline function pext(src::UInt64, mask::UInt64)
    if HAS_BMI2
        return ccall("llvm.x86.bmi.pext.64", llvmcall, UInt64, (UInt64, UInt64), src, mask)
    else
        return pext_fallback(src, mask)
    end
end

@inline pext(src::UInt8, mask::UInt8) = UInt8(pext(UInt32(src), UInt32(mask)))
@inline pext(src::UInt16, mask::UInt16) = UInt16(pext(UInt32(src), UInt32(mask)))

# Software fallback for non-BMI2 hardware
function pdep_fallback(src::T, mask::T) where T<:Unsigned
    result = zero(T)
    src_bit = 0
    for i in 0:(8*sizeof(T) - 1)
        if (mask >> i) & one(T) == one(T)
            result |= ((src >> src_bit) & one(T)) << i
            src_bit += 1
        end
    end
    return result
end

function pext_fallback(src::T, mask::T) where T<:Unsigned
    result = zero(T)
    dst_bit = 0
    for i in 0:(8*sizeof(T) - 1)
        if (mask >> i) & one(T) == one(T)
            result |= ((src >> i) & one(T)) << dst_bit
            dst_bit += 1
        end
    end
    return result
end

# ============================================================================
# Composite bit operations for our use cases
# ============================================================================

"""
    bit_for_axis(mask::Unsigned, axis::Integer)

Given a split_mask and an axis index, return the bit position in
sibling_index that encodes this axis's position. Returns the count of split
axes with index strictly less than `axis`.

For mask = 0b1101 (axes 0, 2, 3 split), bit_for_axis(mask, 2) returns 1
(axis 0 is below axis 2 and is split, contributing bit 0; axis 2 then
contributes bit 1).

If `axis` is not a split axis itself, the returned value is meaningless
(but well-defined). Callers should check `(mask >> axis) & 1 == 1` first.
"""
@inline function bit_for_axis(mask::T, axis::Integer) where T<:Unsigned
    return count_ones(mask & ((one(T) << axis) - one(T)))
end

"""
    nth_set_bit_position(mask::Unsigned, n::Integer)

Find the position of the nth set bit in mask (0-indexed). Returns
sizeof(mask)*8 if n is out of range.

For mask = 0b10101 and n = 1, returns 2 (the second set bit is at position 2).
"""
function nth_set_bit_position(mask::T, n::Integer) where T<:Unsigned
    count = 0
    for i in 0:(8*sizeof(T) - 1)
        if (mask >> i) & one(T) == one(T)
            if count == n
                return i
            end
            count += 1
        end
    end
    return 8 * sizeof(T)
end

"""
    positions_of_set_bits(mask::Unsigned)

Return a tuple of bit positions that are set in mask. Useful for iterating
over the split axes of a cell.

For mask = 0b1101, returns (0, 2, 3).
"""
function positions_of_set_bits(mask::T) where T<:Unsigned
    n = count_ones(mask)
    positions = ntuple(i -> nth_set_bit_position(mask, i-1), n)
    return positions
end

# ============================================================================
# Type selection by dimension
# ============================================================================

"""
    sibling_index_type(::Val{D}) where D

Choose the unsigned integer type wide enough to hold a sibling_index for
dimension D. Sibling_index can range from 0 to 2^popcount(split_mask)-1,
which is at most 2^D - 1.

| D     | Max sibling_index | Type   |
|-------|-------------------|--------|
| 1-7   | 1-127             | UInt8  |
| 8-15  | 255-32767         | UInt16 |
| 16-31 | 65535+            | UInt32 |
| 32-63 |                   | UInt64 |
"""
@inline sibling_index_type(::Val{D}) where D = D <= 7 ? UInt8 :
                                                D <= 15 ? UInt16 :
                                                D <= 31 ? UInt32 : UInt64

"""
    split_mask_type(::Val{D}) where D

Choose the unsigned integer type for a D-dimensional split_mask. Same widths
as sibling_index_type since both have one bit per axis.
"""
@inline split_mask_type(::Val{D}) where D = sibling_index_type(Val(D))

"""
    vertex_int_type(::Val{D}) where D

Default vertex coordinate type. Int16 is the canonical choice (32K resolution
per axis is plenty for cell-relative positions; with hierarchical relative
coordinates, this scales without limit).

Users can override by parameterizing types explicitly.
"""
@inline vertex_int_type(::Val{D}) where D = Int16

"""
    volume_int_type(::Val{D}, ::Val{vertex_bits}) where {D, vertex_bits}

Choose an integer type wide enough to hold an exact d-dimensional volume
computed from vertices of the given bit width. Volume of a d-cube is the
product of d edge lengths; product of d Int16 differences is up to
~D * (vertex_bits + 1) bits, plus 6 bits of headroom for accumulating
sums across a few hundred cells.

Returns `Int64` when the rough bound `D * (vertex_bits + 1) + 6 ≤ 63`,
`Int128` otherwise (capped — beyond Int128 would need BitIntegers.jl).

| D     | vertex_bits | bits_needed | type    |
|-------|-------------|-------------|---------|
| 3     | 15 (Int16)  | 54          | `Int64` |
| 4     | 15 (Int16)  | 70          | `Int128`|
| 8+    | 15 (Int16)  | 134+        | `Int128` (capped; user must verify no overflow) |

Note: the +6 headroom is conservative for individual cell volumes plus
sums of a few hundred neighbor cells. If you sum volumes across a deeply
refined tree (millions of cells), you may need a wider integer than this
function returns. The function is exported but currently unused inside
the framework — `Float64` is the de-facto coordinate type elsewhere.
"""
@inline function volume_int_type(::Val{D}, ::Val{vertex_bits}) where {D, vertex_bits}
    bits_needed = D * (vertex_bits + 1) + 6  # rough upper bound
    if bits_needed <= 63
        return Int64
    elseif bits_needed <= 127
        return Int128
    else
        # Beyond Int128, would need BitIntegers.jl; we fall back to Int128
        # and rely on the user to be aware of this limitation
        return Int128
    end
end

end # module BitPrimitives
