using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh
using Test
using Random

# ============================================================================
# Tests for the IntegerLattice quantization helpers
# (`src/Overlap/quantize.jl`).
#
# PR-2 of the IntExact integration plan. These cover constructor
# validation, round-trip identity for grid-aligned vertices, the
# bounded round-trip error for off-grid vertices, the `quantize_strict`
# rejection path, and `unscale_volume` correctness for D=1,2,3.
# `unscale_moment` (degree-aware, lattice-frame) is covered as well.
# ============================================================================

@testset "IntegerLattice constructor validation" begin
    # Basic happy path.
    lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 16, int_type = Int32)
    @test lat.bits == 16
    @test lat.int_type === Int32
    @test lat.scale ≈ (2^16 - 1) / 1.0
    @test lat.lo === (0.0, 0.0)
    @test lat.hi === (1.0, 1.0)

    # 1D, 3D variants.
    lat1 = IntegerLattice((0.0,), (2.0,); bits = 8, int_type = Int16)
    @test lat1.scale ≈ (2^8 - 1) / 2.0
    lat3 = IntegerLattice((-1.0, 0.0, 1.0), (1.0, 5.0, 2.0); bits = 12, int_type = Int32)
    # Longest axis has extent 5.0 (axis 2): -1→1 = 2, 0→5 = 5, 1→2 = 1.
    @test lat3.scale ≈ (2^12 - 1) / 5.0

    # hi <= lo throws.
    @test_throws ArgumentError IntegerLattice((0.0, 0.0), (1.0, 0.0); bits = 16)
    @test_throws ArgumentError IntegerLattice((0.0, 0.0), (0.0, 1.0); bits = 16)
    @test_throws ArgumentError IntegerLattice((1.0, 1.0), (0.5, 2.0); bits = 16)

    # bits < 1 throws.
    @test_throws ArgumentError IntegerLattice((0.0,), (1.0,); bits = 0)
end

@testset "IntegerLattice overflow guard" begin
    # bits=64 with int_type=Int32 must throw at construction (2^64 - 1
    # cannot fit in Int32).
    @test_throws ArgumentError IntegerLattice(
        (0.0, 0.0), (1.0, 1.0); bits = 64, int_type = Int32)
    # bits=32 with int_type=Int32: 2^32 - 1 > typemax(Int32) (= 2^31 - 1).
    @test_throws ArgumentError IntegerLattice(
        (0.0, 0.0), (1.0, 1.0); bits = 32, int_type = Int32)
    # bits=31 with int_type=Int32: 2^31 - 1 == typemax(Int32). Should succeed.
    lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 31, int_type = Int32)
    @test lat.bits == 31
    # bits=63 with Int64 succeeds; bits=64 with Int64 fails.
    @test (IntegerLattice((0.0,), (1.0,); bits = 63, int_type = Int64)).bits == 63
    @test_throws ArgumentError IntegerLattice((0.0,), (1.0,); bits = 64, int_type = Int64)
    # BigInt is unbounded — any bits should work.
    @test (IntegerLattice((0.0,), (1.0,); bits = 256, int_type = BigInt)).bits == 256
end

@testset "IntegerLattice from EulerianFrame" begin
    mesh = HierarchicalMesh{2}()
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 2.0))
    lat = IntegerLattice(frame; bits = 16, int_type = Int32)
    @test lat.lo === (0.0, 0.0)
    @test lat.hi === (1.0, 2.0)
    # Longest axis has extent 2.0.
    @test lat.scale ≈ (2^16 - 1) / 2.0
    # Defaults: bits=16, int_type=Int32.
    lat_def = IntegerLattice(frame)
    @test lat_def.bits == 16
    @test lat_def.int_type === Int32
end

@testset "lat_resolution" begin
    lat = IntegerLattice((0.0,), (10.0,); bits = 8, int_type = Int16)
    @test lat_resolution(lat) ≈ 10.0 / (2^8 - 1)
    # Equal-scale invariant: resolution is the same on every axis.
    lat2 = IntegerLattice((0.0, 0.0), (1.0, 4.0); bits = 12)
    @test lat_resolution(lat2) ≈ 4.0 / (2^12 - 1)
end

@testset "quantize round-trip — grid-aligned" begin
    # On lattice points, dequantize ∘ quantize must be exact (or within
    # one float ulp due to double rounding through Float64 scale).
    lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 16, int_type = Int32)
    res = lat_resolution(lat)
    # Lower corner.
    @test quantize((0.0, 0.0), lat) == (Int32(0), Int32(0))
    @test dequantize((Int32(0), Int32(0)), lat) === (0.0, 0.0)
    # Some integer steps (these land exactly on the grid up to float).
    p = quantize((0.5, 0.0), lat)
    back = dequantize(p, lat)
    @test abs(back[1] - 0.5) <= res / 2 + eps(1.0)
    @test abs(back[2] - 0.0) <= res / 2 + eps(1.0)
    # Upper corner: 2^bits - 1 steps.
    p_hi = quantize((1.0, 1.0), lat)
    @test p_hi[1] == Int32(2^16 - 1)
    @test p_hi[2] == Int32(2^16 - 1)
    @test dequantize(p_hi, lat) == (1.0, 1.0)
end

@testset "quantize round-trip — off-grid bounded error" begin
    # Off-lattice vertex: round-trip error must be ≤ lat_resolution / 2.
    lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 10, int_type = Int32)
    res = lat_resolution(lat)
    # Choose a vertex strictly between lattice points.
    v = (0.5 + res / 4, 0.25 + res / 3)
    p = quantize(v, lat)
    back = dequantize(p, lat)
    @test abs(back[1] - v[1]) <= res / 2 + 1e-12
    @test abs(back[2] - v[2]) <= res / 2 + 1e-12
end

@testset "quantize property: dequantize ∘ quantize within resolution/2" begin
    # Property-based loop: random in-range vertices, error must always
    # be bounded by lat_resolution/2 per axis.
    rng = MersenneTwister(0xCAFEBABE)
    for D in (1, 2, 3)
        lat_lo = ntuple(_ -> -1.0, Val(D))
        lat_hi = ntuple(_ -> 3.0, Val(D))
        lat = IntegerLattice(lat_lo, lat_hi; bits = 14, int_type = Int32)
        res = lat_resolution(lat)
        for _ in 1:200
            v = ntuple(d -> lat_lo[d] + (lat_hi[d] - lat_lo[d]) * rand(rng), Val(D))
            p = quantize(v, lat)
            back = dequantize(p, lat)
            for d in 1:D
                @test abs(back[d] - v[d]) <= res / 2 + 1e-12
            end
        end
    end
end

@testset "quantize clamps out-of-range to representable integer range" begin
    # Way-out-of-range inputs must NOT raise InexactError; they clamp to
    # the int_type's representable range.
    lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 16, int_type = Int32)
    p_neg = quantize((-1e20, -1e20), lat)
    @test p_neg[1] == typemin(Int32)
    @test p_neg[2] == typemin(Int32)
    p_pos = quantize((1e20, 1e20), lat)
    @test p_pos[1] == typemax(Int32)
    @test p_pos[2] == typemax(Int32)
end

@testset "quantize_strict rejects off-grid" begin
    lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 10, int_type = Int32)
    res = lat_resolution(lat)
    # Vertex at 0.5 + res/4 — exceeds atol=res/8.
    v = (0.5 + res / 4, 0.5)
    @test_throws ArgumentError quantize_strict(v, lat; atol = res / 8)
    # Same vertex with the default atol (= res/2) succeeds.
    p = quantize_strict(v, lat)
    @test p isa NTuple{2, Int32}
end

@testset "quantize_strict accepts grid-aligned" begin
    # Construct an integer point, dequantize it to physical, then
    # quantize_strict at zero tolerance; must succeed (round-trip
    # through float rounding leaves the point exactly on the grid).
    lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 16, int_type = Int32)
    p_int = (Int32(123), Int32(45678))
    v_phys = dequantize(p_int, lat)
    # Tight tolerance: a few float ulps.
    p_back = quantize_strict(v_phys, lat; atol = 1e-9)
    @test p_back == p_int
end

@testset "unscale_volume — 1D length" begin
    lat = IntegerLattice((0.0,), (4.0,); bits = 16, int_type = Int32)
    # Integer length on the lattice: 100 lattice steps.
    A_int = Rational{Int128}(100, 1)
    expected_phys = 100 * lat_resolution(lat)
    @test unscale_volume(A_int, lat) ≈ expected_phys
end

@testset "unscale_volume — 2D area" begin
    lat = IntegerLattice((0.0, 0.0), (2.0, 3.0); bits = 16, int_type = Int32)
    # Rectangle: integer width=10, height=20, integer area=200 lattice cells².
    A_int = Rational{Int128}(200, 1)
    expected_phys = 200 * lat_resolution(lat)^2
    @test unscale_volume(A_int, lat) ≈ expected_phys
    # Cross-check with quantization of a known physical rectangle.
    res = lat_resolution(lat)
    w_int = Int(round(0.5 / res))
    h_int = Int(round(0.7 / res))
    A2 = Rational{Int128}(w_int * h_int, 1)
    A2_phys = unscale_volume(A2, lat)
    expected_phys2 = (w_int * res) * (h_int * res)
    @test A2_phys ≈ expected_phys2
end

@testset "unscale_volume — 3D volume" begin
    lat = IntegerLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0); bits = 12, int_type = Int32)
    res = lat_resolution(lat)
    # 5 × 7 × 11 lattice-unit box.
    V_int = Rational{Int128}(5 * 7 * 11, 1)
    @test unscale_volume(V_int, lat) ≈ (5 * 7 * 11) * res^3
end

@testset "unscale_moment — degree zero == unscale_volume" begin
    lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 12, int_type = Int32)
    m = Rational{Int128}(7, 1)
    @test unscale_moment(m, lat, 0) ≈ unscale_volume(m, lat)
end

@testset "unscale_moment — degree-1 lattice-frame factor" begin
    # For a degree-1 moment, the conversion factor is (1/scale)^(D+1).
    # The result is in the LATTICE FRAME — physical-frame moments
    # additionally need shift_moments! with Δ = lat.lo. This test
    # asserts the scale-only behavior.
    lat = IntegerLattice((0.0, 0.0), (2.0, 2.0); bits = 10, int_type = Int32)
    res = lat_resolution(lat)
    # Take a triangle (0,0)-(s,0)-(0,s) in lattice integer coords; its
    # M(1,0) = s^3 / 6. The lattice-frame physical moment is then
    # (s * res)^3 / 6 — what unscale_moment returns at total_degree=1
    # (factor (1/scale)^(D+1) = res^(D+1) = res^3).
    s = 200
    m_int = Rational{Int128}(s^3, 6)
    expected = (s * res)^3 / 6
    @test unscale_moment(m_int, lat, 1) ≈ expected
end

@testset "unscale_moment — degree-2 factor" begin
    lat = IntegerLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0); bits = 10, int_type = Int32)
    res = lat_resolution(lat)
    m_int = Rational{Int128}(42, 5)
    # D + degree = 3 + 2 = 5.
    expected = Float64(42 // 5) * res^5
    @test unscale_moment(m_int, lat, 2) ≈ expected
end
