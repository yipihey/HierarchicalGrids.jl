using HierarchicalGrids
using Test

# ============================================================================
# Tests for the IntExact-based exact-rational overlap adapter
# (`src/Overlap/r3d_int_adapter.jl`).
#
# These exercise `overlap_simplex_box_exact!` for D=2 and D=3 against
# canonical analytic moments, conservation under the Kuhn cube
# decomposition, the empty / fully-contained edge cases, and the
# accumulator-overflow guard.
#
# All inputs are exact integers on a discretized lattice; the adapter
# returns exact `Rational{R}` values throughout. PR-3 will dequantize
# at the float boundary; this PR keeps the boundary exact.
# ============================================================================

# ----------------------------------------------------------------------------
# Helpers — canonical 6-tet Kuhn decomposition of an integer cube, with the
# same orientation handling as `_kuhn_unit_cube_offset` in
# `test_overlap_3d.jl`. r3djl's IntExact `init_tet!` does not enforce
# orientation, but `volume_exact` and `moments_exact!` can return negative
# values for inverted tets; the adapter guards against this with a swap.
# We replicate the swap helper here for the conservation test (the adapter
# also swaps internally — this just keeps the test legible).
# ----------------------------------------------------------------------------
function _kuhn_unit_cube_int(s::Int)
    # Eight corners of [0, s]^3.
    corners = NTuple{3, Int64}[
        (0, 0, 0),  # 1
        (s, 0, 0),  # 2
        (0, s, 0),  # 3
        (s, s, 0),  # 4
        (0, 0, s),  # 5
        (s, 0, s),  # 6
        (0, s, s),  # 7
        (s, s, s),  # 8
    ]
    sv = [
        1 1 1 1 1 1;
        2 2 3 3 5 5;
        4 6 4 7 6 7;
        8 8 8 8 8 8
    ]
    # Build the 6 tetrahedra as NTuple{4, NTuple{3, Int64}}.
    tets = Vector{NTuple{4, NTuple{3, Int64}}}(undef, 6)
    for i in 1:6
        v1 = corners[sv[1, i]]
        v2 = corners[sv[2, i]]
        v3 = corners[sv[3, i]]
        v4 = corners[sv[4, i]]
        # Compute 6 × signed volume; if negative, swap last two (same
        # convention as `_kuhn_unit_cube_offset`).
        a1 = v2[1] - v1[1]; a2 = v2[2] - v1[2]; a3 = v2[3] - v1[3]
        b1 = v3[1] - v1[1]; b2 = v3[2] - v1[2]; b3 = v3[3] - v1[3]
        c1 = v4[1] - v1[1]; c2 = v4[2] - v1[2]; c3 = v4[3] - v1[3]
        sixv = a1 * (b2 * c3 - b3 * c2) -
               a2 * (b1 * c3 - b3 * c1) +
               a3 * (b1 * c2 - b2 * c1)
        if sixv < 0
            tets[i] = (v1, v2, v4, v3)
        else
            tets[i] = (v1, v2, v3, v4)
        end
    end
    return tets
end

# ============================================================================
# 1. Unit triangle (scaled to integer lattice) inside a unit box (D=2)
# ============================================================================

@testset "IntExact D=2: unit triangle inside [0, s]^2 ⇒ exact area s²/2" begin
    s = Int64(1024)   # 10-bit lattice
    scratch = IntPairScratch(Val(2), Int64; capacity = 32)

    R = Int128
    m = Vector{Rational{R}}(undef, moments_length(2, 2))

    verts = ((Int64(0), Int64(0)), (s, Int64(0)), (Int64(0), s))
    box_lo = (Int64(0), Int64(0))
    box_hi = (s, s)

    vol, centroid, _ = overlap_simplex_box_exact!(m, scratch,
                                                    verts, box_lo, box_hi, 2;
                                                    accumulator = R)

    @test vol == Rational{R}(s)^2 // 2
    @test centroid[1] == Rational{R}(s) // 3
    @test centroid[2] == Rational{R}(s) // 3

    # Analytic moments (graded-lex order):
    # (0,0): s²/2     (1,0): s³/6     (0,1): s³/6
    # (2,0): s⁴/12    (1,1): s⁴/24    (0,2): s⁴/12
    s_R = Rational{R}(s)
    @test m[1] == s_R^2 // 2
    @test m[2] == s_R^3 // 6
    @test m[3] == s_R^3 // 6
    @test m[4] == s_R^4 // 12
    @test m[5] == s_R^4 // 24
    @test m[6] == s_R^4 // 12
end

# ============================================================================
# 2. Unit tetrahedron inside a unit box (D=3) — volume s³/6
# ============================================================================

@testset "IntExact D=3: unit tetrahedron inside [0, s]^3 ⇒ exact volume s³/6" begin
    s = Int64(64)
    scratch = IntPairScratch(Val(3), Int64; capacity = 32)

    R = Int128
    m = Vector{Rational{R}}(undef, moments_length(3, 1))

    verts = ((Int64(0), Int64(0), Int64(0)),
             (s, Int64(0), Int64(0)),
             (Int64(0), s, Int64(0)),
             (Int64(0), Int64(0), s))
    box_lo = (Int64(0), Int64(0), Int64(0))
    box_hi = (s, s, s)

    vol, centroid, _ = overlap_simplex_box_exact!(m, scratch,
                                                    verts, box_lo, box_hi, 1;
                                                    accumulator = R)

    @test vol == Rational{R}(s)^3 // 6
    # Centroid of unit corner tet at (s/4, s/4, s/4).
    @test centroid[1] == Rational{R}(s) // 4
    @test centroid[2] == Rational{R}(s) // 4
    @test centroid[3] == Rational{R}(s) // 4
end

# ============================================================================
# 3. Off-axis partial overlap — D=2 triangle clipped by a smaller box.
#
# Triangle (0,0), (s, 0), (0, s) area s²/2. Clip by box [0, s/2] × [0, s].
# The kept region is a trapezoid (0,0)-(s/2,0)-(s/2, s/2)-(0, s).
# Vertex enumeration on triangle: y ≤ s - x. At x = s/2, y ≤ s/2 so the
# clip line meets the hypotenuse at (s/2, s/2).
#
# Trapezoid analytic answers (let h = s/2):
#   area = ∫_0^h ∫_0^{s - x} dy dx = ∫_0^h (s - x) dx
#        = s·h - h²/2 = s · (s/2) - (s/2)² / 2 = s²/2 - s²/8 = 3 s²/8
#   M(1, 0) = ∫_0^h ∫_0^{s - x} x dy dx = ∫_0^h x (s - x) dx
#           = s · h²/2 - h³/3 = s³/8 - s³/24 = 2 s³/24 = s³/12
#   M(0, 1) = ∫_0^h ∫_0^{s - x} y dy dx = ∫_0^h (s - x)² / 2 dx
#           = -[(s - x)³ / 6]_0^h = (s³ - (s - h)³) / 6
#           = (s³ - (s/2)³) / 6 = (8 s³/8 - s³/8) / 6 = 7 s³/48
# ============================================================================

@testset "IntExact D=2: off-axis partial overlap (trapezoid)" begin
    s = Int64(8)         # small so analytic numbers stay readable
    h = s ÷ 2            # = 4
    scratch = IntPairScratch(Val(2), Int64; capacity = 32)

    R = Int128
    m = Vector{Rational{R}}(undef, moments_length(2, 1))

    verts = ((Int64(0), Int64(0)), (s, Int64(0)), (Int64(0), s))
    box_lo = (Int64(0), Int64(0))
    box_hi = (h, s)

    vol, centroid, _ = overlap_simplex_box_exact!(m, scratch,
                                                    verts, box_lo, box_hi, 1;
                                                    accumulator = R)

    s_R = Rational{R}(s)

    # area = 3 s²/8
    @test vol == 3 * s_R^2 // 8
    # M(1,0) = s³/12, M(0,1) = 7 s³/48
    @test m[2] == s_R^3 // 12
    @test m[3] == 7 * s_R^3 // 48

    # Centroid:
    #   x_c = (s³/12) / (3 s²/8) = 8 s / 36 = 2 s / 9
    #   y_c = (7 s³/48) / (3 s²/8) = 7 s · 8 / (48 · 3) = 56 s / 144 = 7 s / 18
    @test centroid[1] == 2 * s_R // 9
    @test centroid[2] == 7 * s_R // 18
end

# ============================================================================
# 4. Conservation: Kuhn 6-tet decomposition of an integer cube ⇒ sum of
# exact-volumes equals s³.
# ============================================================================

@testset "IntExact D=3: Kuhn 6-tet decomposition conserves cube volume exactly" begin
    s = Int64(16)
    tets = _kuhn_unit_cube_int(Int(s))
    scratch = IntPairScratch(Val(3), Int64; capacity = 32)
    R = Int128
    m = Vector{Rational{R}}(undef, moments_length(3, 0))
    box_lo = (Int64(0), Int64(0), Int64(0))
    box_hi = (s, s, s)

    total = zero(Rational{R})
    for tet in tets
        vol, _, _ = overlap_simplex_box_exact!(m, scratch,
                                                 tet, box_lo, box_hi, 0;
                                                 accumulator = R)
        # Each tet contributes s³/6 since it sits entirely inside the cube.
        @test vol == Rational{R}(s)^3 // 6
        total += vol
    end
    @test total == Rational{R}(s)^3
end

# ============================================================================
# 5. Empty case: triangle / tet entirely outside the box.
# ============================================================================

@testset "IntExact D=2: triangle outside the box ⇒ empty result" begin
    scratch = IntPairScratch(Val(2), Int64)
    R = Int128
    m = Vector{Rational{R}}(undef, moments_length(2, 1))

    # Triangle around (100, 100) — far from the unit box [0, 10]².
    verts = ((Int64(100), Int64(100)), (Int64(110), Int64(100)),
             (Int64(100), Int64(110)))
    box_lo = (Int64(0), Int64(0))
    box_hi = (Int64(10), Int64(10))

    vol, centroid, _ = overlap_simplex_box_exact!(m, scratch,
                                                    verts, box_lo, box_hi, 1;
                                                    accumulator = R)

    @test vol == zero(Rational{R})
    @test centroid == (zero(Rational{R}), zero(Rational{R}))
    @test all(m[k] == zero(Rational{R}) for k in eachindex(m))
end

@testset "IntExact D=3: tet outside the box ⇒ empty result" begin
    scratch = IntPairScratch(Val(3), Int64)
    R = Int128
    m = Vector{Rational{R}}(undef, moments_length(3, 1))

    verts = ((Int64(100), Int64(100), Int64(100)),
             (Int64(110), Int64(100), Int64(100)),
             (Int64(100), Int64(110), Int64(100)),
             (Int64(100), Int64(100), Int64(110)))
    box_lo = (Int64(0), Int64(0), Int64(0))
    box_hi = (Int64(10), Int64(10), Int64(10))

    vol, centroid, _ = overlap_simplex_box_exact!(m, scratch,
                                                    verts, box_lo, box_hi, 1;
                                                    accumulator = R)

    @test vol == zero(Rational{R})
    @test centroid == (zero(Rational{R}), zero(Rational{R}), zero(Rational{R}))
    @test all(m[k] == zero(Rational{R}) for k in eachindex(m))
end

# ============================================================================
# 6. Fully-contained: small triangle / tet entirely inside a much larger
# box ⇒ overlap volume equals the simplex's own integer area / volume.
# ============================================================================

@testset "IntExact D=2: triangle fully inside a larger box ⇒ exact triangle area" begin
    scratch = IntPairScratch(Val(2), Int64)
    R = Int128
    m = Vector{Rational{R}}(undef, moments_length(2, 1))

    # Triangle (3,3)-(7,3)-(3,9). Base = 4, height = 6 → area = 12.
    verts = ((Int64(3), Int64(3)), (Int64(7), Int64(3)), (Int64(3), Int64(9)))
    box_lo = (Int64(0), Int64(0))
    box_hi = (Int64(20), Int64(20))

    vol, centroid, _ = overlap_simplex_box_exact!(m, scratch,
                                                    verts, box_lo, box_hi, 1;
                                                    accumulator = R)

    @test vol == 12 // 1
    # Centroid of a triangle = mean of vertices: ((3 + 7 + 3)/3, (3 + 3 + 9)/3)
    #                                          = (13/3, 5).
    @test centroid[1] == Rational{R}(13) // 3
    @test centroid[2] == Rational{R}(5) // 1
end

@testset "IntExact D=3: tet fully inside a larger box ⇒ exact tet volume" begin
    scratch = IntPairScratch(Val(3), Int64)
    R = Int128
    m = Vector{Rational{R}}(undef, moments_length(3, 1))

    # Corner-tet at offset (2,2,2) with edge 6 → volume = 6³/6 = 36.
    e = Int64(6)
    o = Int64(2)
    verts = ((o, o, o), (o + e, o, o), (o, o + e, o), (o, o, o + e))
    box_lo = (Int64(0), Int64(0), Int64(0))
    box_hi = (Int64(20), Int64(20), Int64(20))

    vol, centroid, _ = overlap_simplex_box_exact!(m, scratch,
                                                    verts, box_lo, box_hi, 1;
                                                    accumulator = R)

    @test vol == 36 // 1
    # Centroid = mean of vertices = (o + e/4, o + e/4, o + e/4) = (7/2, 7/2, 7/2).
    @test centroid[1] == Rational{R}(7) // 2
    @test centroid[2] == Rational{R}(7) // 2
    @test centroid[3] == Rational{R}(7) // 2
end

# ============================================================================
# 7. Accumulator-overflow guard.
#
# `T = Int16` vertices on a small lattice. With box-only (axis-aligned)
# clips and a small triangle, IntExact's GCD reduction in `clip!` should
# keep numerators / denominators within Int64. We pass `accumulator = Int64`
# and assert clean execution; document the empirical result.
#
# To exercise a few oblique-ish shapes we use a triangle whose vertices
# don't all sit on the box corners and clip by a smaller box. The case
# is easy enough that GCD reduction handles it; this test mainly
# documents the observed behavior so future r3djl changes that
# regress on bit-width growth surface here.
# ============================================================================

@testset "IntExact accumulator: Int16 + Int64 stays in range under GCD reduction" begin
    scratch = IntPairScratch(Val(2), Int16; capacity = 64)
    R = Int64
    m = Vector{Rational{R}}(undef, moments_length(2, 1))

    # Modest 8-bit lattice; box clip keeps a trapezoid.
    verts = ((Int16(0), Int16(0)), (Int16(120), Int16(0)),
             (Int16(0), Int16(120)))
    box_lo = (Int16(10), Int16(10))
    box_hi = (Int16(80), Int16(80))

    # Clip is bounded by 4 axis-aligned planes ⇒ shouldn't blow up.
    vol, centroid, _ = overlap_simplex_box_exact!(m, scratch,
                                                    verts, box_lo, box_hi, 1;
                                                    accumulator = R)

    # Should report a positive area; bit-exact value depends on GCD steps,
    # but it must equal the analytic exact rational. The clipped region is
    # a quadrilateral with corners (10,10), (80,10), (80,30), (10,80) —
    # bounded by y ≤ 120 - x. We compute area analytically:
    # area = ∫_{x=10}^{80} (min(80, 120 - x) - 10) dx
    #      = ∫_{10}^{40} (80 - 10) dx + ∫_{40}^{80} (120 - x - 10) dx
    #      = 70 · 30 + ∫_{40}^{80} (110 - x) dx
    #      = 2100 + [110x - x²/2]_{40}^{80}
    #      = 2100 + (8800 - 3200) - (4400 - 800)
    #      = 2100 + 5600 - 3600
    #      = 4100
    @test vol == Rational{R}(4100) // 1

    # Now the oblique-ish stress case: ask for moment_order = 2. Bit
    # growth is mild because the box clip is axis-aligned, but several
    # rational arithmetic steps run.
    m2 = Vector{Rational{R}}(undef, moments_length(2, 2))
    overlap_simplex_box_exact!(m2, scratch, verts, box_lo, box_hi, 2;
                                accumulator = R)
    # Volume must round-trip identically.
    @test m2[1] == Rational{R}(4100) // 1
    # Other moments are positive rationals; we just verify they're sane
    # (no overflow into negative — Julia's Int64 silently wraps under
    # `*` so the assertion below would catch a 64-bit overflow).
    for k in 2:length(m2)
        @test m2[k] > zero(Rational{R})
    end
end

# Empirical-finding documentation: the accumulator-overflow guard above
# confirms that IntExact's per-`clip!` GCD reduction keeps shared-
# denominator products well within Int64 for Int16-coordinate axis-aligned
# clips through moment_order = 2. Callers facing many oblique clips or
# higher moment orders should follow `_default_accumulator`'s table
# (Int128 for Int16/Int32; BigInt for Int64+ or D=4).

# ============================================================================
# 8. Default accumulator table sanity check.
# ============================================================================

@testset "IntExact default-accumulator table" begin
    using HierarchicalGrids.Overlap: _default_accumulator
    @test _default_accumulator(Int16, Val(2)) === Int128
    @test _default_accumulator(Int16, Val(3)) === Int128
    @test _default_accumulator(Int32, Val(2)) === Int128
    @test _default_accumulator(Int32, Val(3)) === Int128
    @test _default_accumulator(Int64, Val(2)) === BigInt
    @test _default_accumulator(Int64, Val(3)) === BigInt
    @test _default_accumulator(Int16, Val(4)) === BigInt
    @test _default_accumulator(Int64, Val(4)) === BigInt
end

# ============================================================================
# 9. Module-load convention check ran at module load (loud failure on
# divergence). We re-run it here as a regression test so the harness
# surfaces a clear assertion if it ever starts failing in CI.
# ============================================================================

@testset "IntExact: module-load convention check is callable and passes" begin
    @test HierarchicalGrids.Overlap._verify_intexact_plane_convention() === true
end
