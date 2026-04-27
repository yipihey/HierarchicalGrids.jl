using Test
using HierarchicalGrids

# ============================================================================
# Tests for `describe(::GeometricOverlap)` (Item C of the debuggability
# batch). Distributional summary: total volume, n_entries, n_empty,
# per-leaf entry-count histogram, volume-distribution percentiles.
# ============================================================================

# ---------------------------------------------------------------------------
# Test 1 — smoke test on a small overlap
# ---------------------------------------------------------------------------

@testset "describe: smoke test on small overlap" begin
    positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    sv = Int32[1 1; 2 3; 3 4]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    o = compute_overlap(lag, frame; backend = :float, moment_order = 1)
    @test n_entries(o) > 0

    buf = IOBuffer()
    describe(o; io = buf)
    s = String(take!(buf))
    @test occursin("GeometricOverlap{2, Float64} summary", s)
    @test occursin("spatial dimension:", s)
    @test occursin("moment order:", s)
    @test occursin("n_entries:", s)
    @test occursin("total volume:", s)
    @test occursin("volume percentiles:", s)
    @test occursin("min:", s)
    @test occursin("p10:", s)
    @test occursin("p50:", s)
    @test occursin("p90:", s)
    @test occursin("max:", s)
    @test occursin("entries-per-Eulerian-leaf:", s)
    @test occursin("entries-per-Lagrangian-simplex:", s)
end

# ---------------------------------------------------------------------------
# Test 2 — empty overlap prints sensibly
# ---------------------------------------------------------------------------

@testset "describe: zero-entry overlap" begin
    # An overlap that has no entries (no Lagrangian simplex inside the
    # frame). Build one by hand via the OverlapBuilder.
    b = OverlapBuilder{2, Float64}(1)
    o = finalize_overlap(b, 3, 4)  # 3 simplices, 4 cells, no entries
    @test n_entries(o) == 0

    buf = IOBuffer()
    describe(o; io = buf)
    s = String(take!(buf))
    @test occursin("GeometricOverlap{2, Float64} summary", s)
    @test occursin("n_entries:         0", s)
    @test occursin("(overlap is empty", s)
end

# ---------------------------------------------------------------------------
# Test 3 — single-entry overlap
# ---------------------------------------------------------------------------

@testset "describe: single-entry overlap" begin
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    o = compute_overlap(lag, frame; backend = :float, moment_order = 1)
    @test n_entries(o) == 1

    buf = IOBuffer()
    describe(o; io = buf)
    s = String(take!(buf))
    @test occursin("n_entries:         1", s)
    # With a single entry the percentiles should all match.
    @test occursin("min:", s)
    @test occursin("max:", s)
    @test occursin("entries-per-Eulerian-leaf:", s)
    @test occursin("min: 1", s)
    @test occursin("max: 1", s)
end

# ---------------------------------------------------------------------------
# Test 4 — return value is `nothing`
# ---------------------------------------------------------------------------

@testset "describe: returns nothing" begin
    b = OverlapBuilder{2, Float64}(1)
    push_overlap!(b, 1, 1, 0.5, (0.5, 0.5), [0.5, 0.25, 0.25])
    o = finalize_overlap(b, 1, 1)
    buf = IOBuffer()
    @test describe(o; io = buf) === nothing
end

# ---------------------------------------------------------------------------
# Test 5 — percentiles are computed correctly on a known distribution
# ---------------------------------------------------------------------------

@testset "describe: percentiles render numerically" begin
    # Build a manual overlap with known volume distribution.
    b = OverlapBuilder{2, Float64}(0)
    for k in 1:10
        v = Float64(k) / 10  # 0.1, 0.2, ..., 1.0
        push_overlap!(b, 1, k, v, (0.0, 0.0), [v])
    end
    o = finalize_overlap(b, 1, 10)
    buf = IOBuffer()
    describe(o; io = buf)
    s = String(take!(buf))
    # Median should be around 0.55 (linear interpolation between 0.5 and 0.6).
    @test occursin("p50:   0.55", s) || occursin("p50:   0.5", s) ||
          occursin("p50:   0.6", s)
    @test occursin("min:   0.1", s)
    @test occursin("max:   1.0", s)
end
