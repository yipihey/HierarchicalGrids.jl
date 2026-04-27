using HierarchicalGrids
using Test

# ============================================================================
# Tests for the IntExact-based exact-rational overlap adapter at D = 4
# (`src/Overlap/r3d_int_adapter.jl`).
#
# Coverage scope reflects upstream R3D.IntExact's D = 4 capabilities:
#   - `volume_exact(D=4)` ships and is exercised end-to-end here
#   - `moments_exact!(D=4)` ERRORS upstream (research-grade); the adapter
#     therefore exposes only `moment_order = 0` at D = 4 and we test
#     that `moment_order >= 1` throws a scoped error rather than letting
#     the upstream error bubble up at depth.
#
# All inputs are exact integers on a discretized lattice; the adapter
# returns exact `Rational{R}` values throughout. PR-3 will dequantize
# at the float boundary; this PR keeps the boundary exact.
# ============================================================================

# ----------------------------------------------------------------------------
# 1. Unit pentachoron (corner simplex) inside a unit box
# ----------------------------------------------------------------------------

@testset "IntExact D=4: unit pentachoron inside [0, s]^4 ⇒ exact volume s⁴/24" begin
    s = Int64(12)   # small lattice; volume = 864
    scratch = IntPairScratch(Val(4), Int64; capacity = 64)

    R = BigInt
    m = Vector{Rational{R}}(undef, moments_length(4, 0))

    verts = ((Int64(0), Int64(0), Int64(0), Int64(0)),
             (s, Int64(0), Int64(0), Int64(0)),
             (Int64(0), s, Int64(0), Int64(0)),
             (Int64(0), Int64(0), s, Int64(0)),
             (Int64(0), Int64(0), Int64(0), s))
    box_lo = (Int64(0), Int64(0), Int64(0), Int64(0))
    box_hi = (s, s, s, s)

    vol, centroid, mout = overlap_simplex_box_exact!(m, scratch,
                                                       verts, box_lo, box_hi, 0;
                                                       accumulator = R)

    # Exact volume of the corner pentachoron = s⁴ / 24.
    @test vol == Rational{R}(s)^4 // 24
    @test mout[1] == vol

    # Centroid is a zero placeholder at D=4 P=0 (we don't have first
    # moments without moments_exact!, which is unimplemented upstream).
    @test centroid == ntuple(_ -> zero(Rational{R}), Val(4))
end

# ----------------------------------------------------------------------------
# 2. Empty case: pentachoron entirely outside the box ⇒ zero volume.
# ----------------------------------------------------------------------------

@testset "IntExact D=4: pentachoron outside box ⇒ empty result" begin
    s = Int64(12)
    scratch = IntPairScratch(Val(4), Int64; capacity = 64)

    R = BigInt
    m = Vector{Rational{R}}(undef, moments_length(4, 0))

    # Place the pentachoron far from the box.
    off = Int64(100)
    verts = ((off, off, off, off),
             (off + s, off, off, off),
             (off, off + s, off, off),
             (off, off, off + s, off),
             (off, off, off, off + s))
    box_lo = (Int64(0), Int64(0), Int64(0), Int64(0))
    box_hi = (s, s, s, s)

    vol, centroid, mout = overlap_simplex_box_exact!(m, scratch,
                                                       verts, box_lo, box_hi, 0;
                                                       accumulator = R)

    @test vol == zero(Rational{R})
    @test mout[1] == zero(Rational{R})
    @test all(centroid .== zero(Rational{R}))
end

# ----------------------------------------------------------------------------
# 3. Partial-overlap case with closed-form expected volume.
#
# Take the pentachoron P_s = {x_i ≥ 0, x_1 + x_2 + x_3 + x_4 ≤ s}
# (vertices at origin and the four axis tips at distance s) clipped by
# the half-space x_1 ≤ s/2. The cap removed is itself a pentachoron
# scaled by 1/2 along x_1: kept = s⁴/24 − (s/2)⁴/24 = s⁴/24 · 15/16.
# ----------------------------------------------------------------------------

@testset "IntExact D=4: partial clip x₁ ≤ s/2 ⇒ analytic 15/16 volume" begin
    s = Int64(12)   # s/2 = 6 integer
    scratch = IntPairScratch(Val(4), Int64; capacity = 64)

    R = BigInt
    m = Vector{Rational{R}}(undef, moments_length(4, 0))

    # The corner pentachoron lives in [0, s]^4 already; we use a clipping
    # box that slices it on x_1 only: [0, s/2] × [0, s] × [0, s] × [0, s].
    verts = ((Int64(0), Int64(0), Int64(0), Int64(0)),
             (s, Int64(0), Int64(0), Int64(0)),
             (Int64(0), s, Int64(0), Int64(0)),
             (Int64(0), Int64(0), s, Int64(0)),
             (Int64(0), Int64(0), Int64(0), s))
    box_lo = (Int64(0), Int64(0), Int64(0), Int64(0))
    box_hi = (Int64(s ÷ 2), s, s, s)

    vol, centroid, mout = overlap_simplex_box_exact!(m, scratch,
                                                       verts, box_lo, box_hi, 0;
                                                       accumulator = R)

    # kept_volume = s⁴/24 − (s/2)⁴/24 = (s⁴ − s⁴/16) / 24 = 15 s⁴ / (24 · 16)
    expected = (Rational{R}(15) * Rational{R}(s)^4) // (Rational{R}(24) * Rational{R}(16))
    @test vol == expected
    @test mout[1] == expected
    @test all(centroid .== zero(Rational{R}))
end

# ----------------------------------------------------------------------------
# 4. Pentachoron strictly contained in box ⇒ unclipped volume.
#
# Sanity check: when the pentachoron is fully inside the box, the volume
# must equal the unclipped exact volume regardless of box geometry.
# ----------------------------------------------------------------------------

@testset "IntExact D=4: pentachoron fully inside box ⇒ unchanged volume" begin
    s = Int64(12)
    scratch = IntPairScratch(Val(4), Int64; capacity = 64)

    R = BigInt
    m = Vector{Rational{R}}(undef, moments_length(4, 0))

    verts = ((Int64(0), Int64(0), Int64(0), Int64(0)),
             (s, Int64(0), Int64(0), Int64(0)),
             (Int64(0), s, Int64(0), Int64(0)),
             (Int64(0), Int64(0), s, Int64(0)),
             (Int64(0), Int64(0), Int64(0), s))
    # Strictly larger box, integer corners.
    big = Int64(100)
    box_lo = (-big, -big, -big, -big)
    box_hi = (big, big, big, big)

    vol, _, _ = overlap_simplex_box_exact!(m, scratch,
                                            verts, box_lo, box_hi, 0;
                                            accumulator = R)
    @test vol == Rational{R}(s)^4 // 24
end

# ----------------------------------------------------------------------------
# 5. Negative-orientation pentachoron ⇒ adapter swaps to make positive.
#
# Swap two vertices to flip parity. The adapter's signed-volume guard
# should detect the flipped sign and swap two trailing vertices, so the
# returned volume is still positive and equal to the canonical s⁴/24.
# ----------------------------------------------------------------------------

@testset "IntExact D=4: orientation swap on inverted pentachoron" begin
    s = Int64(12)
    scratch = IntPairScratch(Val(4), Int64; capacity = 64)

    R = BigInt
    m = Vector{Rational{R}}(undef, moments_length(4, 0))

    # Positive-orientation tuple:
    #   (0, e1, e2, e3, e4) gives positive 4-volume because det of cols
    #   e1, e2, e3, e4 (columns of identity) is +1.
    # Swap v4 and v5 to flip parity:
    verts_neg = ((Int64(0), Int64(0), Int64(0), Int64(0)),
                 (s, Int64(0), Int64(0), Int64(0)),
                 (Int64(0), s, Int64(0), Int64(0)),
                 (Int64(0), Int64(0), Int64(0), s),
                 (Int64(0), Int64(0), s, Int64(0)))
    box_lo = (Int64(0), Int64(0), Int64(0), Int64(0))
    box_hi = (s, s, s, s)

    vol, _, _ = overlap_simplex_box_exact!(m, scratch,
                                            verts_neg, box_lo, box_hi, 0;
                                            accumulator = R)
    @test vol == Rational{R}(s)^4 // 24
end

# ----------------------------------------------------------------------------
# 6. Degenerate pentachoron (5 coplanar vertices) ⇒ empty.
# ----------------------------------------------------------------------------

@testset "IntExact D=4: degenerate (coplanar) pentachoron ⇒ empty" begin
    s = Int64(12)
    scratch = IntPairScratch(Val(4), Int64; capacity = 64)

    R = BigInt
    m = Vector{Rational{R}}(undef, moments_length(4, 0))

    # All 5 vertices lie in the hyperplane x_4 = 0 ⇒ signed 4-volume = 0.
    verts = ((Int64(0), Int64(0), Int64(0), Int64(0)),
             (s, Int64(0), Int64(0), Int64(0)),
             (Int64(0), s, Int64(0), Int64(0)),
             (Int64(0), Int64(0), s, Int64(0)),
             (s, s, s, Int64(0)))
    box_lo = (Int64(0), Int64(0), Int64(0), Int64(0))
    box_hi = (s, s, s, s)

    vol, centroid, mout = overlap_simplex_box_exact!(m, scratch,
                                                      verts, box_lo, box_hi, 0;
                                                      accumulator = R)

    @test vol == zero(Rational{R})
    @test mout[1] == zero(Rational{R})
    @test all(centroid .== zero(Rational{R}))
end

# ----------------------------------------------------------------------------
# 7. Tile-decomposition conservation: sum of pentachoron volumes for the
#    Freudenthal/Kuhn-style triangulation of [0, s]^4 should equal s⁴.
#
# A 4-cube admits a Freudenthal triangulation into D! = 24 pentachora,
# one per permutation π ∈ S_4: the simplex
#   conv{ 0, e_{π(1)}, e_{π(1)} + e_{π(2)}, …, sum_k e_{π(k)} = (s,s,s,s) }
# Each has volume s⁴ / 24, so the 24-tile sum is s⁴ exactly.
# ----------------------------------------------------------------------------

@testset "IntExact D=4: Kuhn 24-pentachoron tile decomposition of [0, s]^4" begin
    s = Int64(6)
    scratch = IntPairScratch(Val(4), Int64; capacity = 64)
    R = BigInt
    m = Vector{Rational{R}}(undef, moments_length(4, 0))

    box_lo = (Int64(0), Int64(0), Int64(0), Int64(0))
    box_hi = (s, s, s, s)

    # Generate all 24 permutations of (1, 2, 3, 4).
    perms = NTuple{4, Int}[]
    for a in 1:4, b in 1:4, c in 1:4, d in 1:4
        if a != b && a != c && a != d && b != c && b != d && c != d
            push!(perms, (a, b, c, d))
        end
    end
    @test length(perms) == 24

    total = zero(Rational{R})
    for π in perms
        # Construct the Kuhn pentachoron for permutation π.
        # vertex_k = sum_{i=1}^{k-1} s * e_{π(i)}, k = 1..5.
        vk = NTuple{5, NTuple{4, Int64}}(ntuple(k -> begin
            v = [Int64(0), Int64(0), Int64(0), Int64(0)]
            for i in 1:(k - 1)
                v[π[i]] = s
            end
            (v[1], v[2], v[3], v[4])
        end, 5))

        vol, _, _ = overlap_simplex_box_exact!(m, scratch,
                                                vk, box_lo, box_hi, 0;
                                                accumulator = R)
        @test vol == Rational{R}(s)^4 // 24
        total += vol
    end
    @test total == Rational{R}(s)^4
end

# ----------------------------------------------------------------------------
# 8. moment_order >= 1 at D=4 throws a scoped adapter error.
#
# Upstream R3D.IntExact.moments_exact! errors at D ≥ 4 because polynomial
# moments at D = 4 are research-grade. The adapter surfaces this at the
# entry point, not deep in r3djl, so callers see a clean error message.
# ----------------------------------------------------------------------------

@testset "IntExact D=4: moment_order ≥ 1 throws scoped error" begin
    s = Int64(12)
    scratch = IntPairScratch(Val(4), Int64; capacity = 64)
    R = BigInt

    verts = ((Int64(0), Int64(0), Int64(0), Int64(0)),
             (s, Int64(0), Int64(0), Int64(0)),
             (Int64(0), s, Int64(0), Int64(0)),
             (Int64(0), Int64(0), s, Int64(0)),
             (Int64(0), Int64(0), Int64(0), s))
    box_lo = (Int64(0), Int64(0), Int64(0), Int64(0))
    box_hi = (s, s, s, s)

    for P in (1, 2)
        m = Vector{Rational{R}}(undef, moments_length(4, P))
        @test_throws ErrorException overlap_simplex_box_exact!(
            m, scratch, verts, box_lo, box_hi, P; accumulator = R)
    end
end

# ----------------------------------------------------------------------------
# 9. Accumulator-default check + empirical Int128-vs-BigInt agreement for
#    a single-axis-clipped pentachoron with `T = Int32` storage.
#
# Empirical finding (probed during PR-5 build): at D=4 the dominant
# overflow risk is the storage type `T` (positions overflow during clip
# before the accumulator ever sees them). When `T = Int32` is intact,
# `Int128` accumulator agrees with `BigInt` accumulator exactly on the
# 8-plane corner-pentachoron clip. This test pins that behavior.
#
# We don't expose the looser `Int128` accumulator as the default at D=4
# because: (1) the upstream docstring promotes BigInt at D=4 to defend
# against per-4-simplex 4-fold numerator products, and (2) the storage
# bottleneck means tightening the accumulator does not unlock larger s.
# ----------------------------------------------------------------------------

@testset "IntExact D=4: default accumulator is BigInt; Int128 agrees in-range" begin
    # Default accumulator is BigInt for any T at D=4.
    @test HierarchicalGrids.Overlap._default_accumulator(Int16,  Val(4)) === BigInt
    @test HierarchicalGrids.Overlap._default_accumulator(Int32,  Val(4)) === BigInt
    @test HierarchicalGrids.Overlap._default_accumulator(Int64,  Val(4)) === BigInt
    @test HierarchicalGrids.Overlap._default_accumulator(Int128, Val(4)) === BigInt
    @test HierarchicalGrids.Overlap._default_accumulator(BigInt, Val(4)) === BigInt

    # Empirical agreement: Int32 storage, modest s, 8-plane clip, Int128
    # vs BigInt accumulator both produce the same exact volume.
    s = Int32(1000)
    box_lo32 = (Int32(0), Int32(0), Int32(0), Int32(0))
    box_hi32 = (Int32(s ÷ 2), Int32(s ÷ 2), Int32(s ÷ 2), Int32(s ÷ 2))
    verts32 = ((Int32(0), Int32(0), Int32(0), Int32(0)),
               (s, Int32(0), Int32(0), Int32(0)),
               (Int32(0), s, Int32(0), Int32(0)),
               (Int32(0), Int32(0), s, Int32(0)),
               (Int32(0), Int32(0), Int32(0), s))

    scratch32 = IntPairScratch(Val(4), Int32; capacity = 64)

    m_big = Vector{Rational{BigInt}}(undef, moments_length(4, 0))
    vol_big, _, _ = overlap_simplex_box_exact!(m_big, scratch32,
                                                 verts32, box_lo32, box_hi32, 0;
                                                 accumulator = BigInt)

    m_128 = Vector{Rational{Int128}}(undef, moments_length(4, 0))
    vol_128, _, _ = overlap_simplex_box_exact!(m_128, scratch32,
                                                 verts32, box_lo32, box_hi32, 0;
                                                 accumulator = Int128)

    @test vol_big > zero(Rational{BigInt})
    @test vol_128 > zero(Rational{Int128})
    @test Rational{BigInt}(numerator(vol_128), denominator(vol_128)) == vol_big
end
