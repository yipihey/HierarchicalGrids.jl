using Test
using HierarchicalGrids
using HierarchicalGrids.Diagnostics: OverlapDropReport

# ============================================================================
# Tests for the `:exact`-backend drop audit (Item A of the debuggability
# batch). When `compute_overlap(...; audit_drops = true, backend = :exact)`
# is called, the per-pair loop tracks pairs the upstream IntExact backend
# silently drops and returns an `OverlapDropReport` alongside the overlap.
#
# Coverage:
#   1. Clean case: simple single triangle inside an unrefined Eulerian
#      cell ⇒ all-zero drop counts.
#   2. Negative-volume / moments-throw case: two-triangle tile of
#      [0, 1]^2 against a 1-level-refined Eulerian — confirmed to
#      trigger upstream drops at D = 2.
#   3. Show formatting (clean and dirty).
#   4. Backward compatibility: `audit_drops = false` (default) returns
#      a plain `GeometricOverlap{D, T}`, not a tuple.
#   5. Validation: `audit_drops = true` with `backend = :float` errors.
# ============================================================================

# ---------------------------------------------------------------------------
# Test 1 — clean case
# ---------------------------------------------------------------------------

@testset "drop audit: clean single triangle, all-zero counts" begin
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    result = compute_overlap(lag, frame;
                             backend = :exact,
                             moment_order = 1,
                             audit_drops = true)
    @test result isa Tuple
    @test length(result) == 2
    o, report = result
    @test o isa GeometricOverlap{2, Float64}
    @test n_entries(o) == 1
    @test report isa OverlapDropReport
    @test report.n_negative_volume == 0
    @test report.n_moments_throw == 0
    @test report.n_empty == 0
    @test isempty(report.drops)
end

# ---------------------------------------------------------------------------
# Test 2 — known-bad geometry triggers drops
# ---------------------------------------------------------------------------

@testset "drop audit: two-triangle tile / 1-refined Eulerian (fully fixed upstream)" begin
    # The two-triangle tiling of [0, 1]^2 (sharing the diagonal) plus a
    # 1-level-refined Eulerian (4 child leaves) used to trigger
    # `R3D.IntExact`'s D = 2 upstream bugs aggressively (5 negative-
    # volume drops + 2 `_moments_exact_d2!` `0//0` throws = 7 data-loss
    # drops out of 6 (lag, eul) pairs).
    #
    # Upstream r3djl commit 154b346 (2026-04-27) fully fixed both
    # bugs. The two diagonal-corner pairs that produce a clipped
    # polygon with `vol == 0` are now correctly classified by the
    # adapter as `:empty` (legitimate zero-area boundary intersection
    # — the Lagrangian-triangle hypotenuse meets the Eulerian
    # quadrant boundary at a single point), NOT `:negative_volume`.
    positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    sv = Int32[1 1; 2 3; 3 4]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    o, report = compute_overlap(lag, frame;
                                backend = :exact,
                                moment_order = 1,
                                audit_drops = true)
    @test report isa OverlapDropReport
    # No data-loss drops: upstream is fixed and the adapter
    # correctly distinguishes zero-area boundary intersections
    # (`:empty`) from orientation-flipped clips (`:negative_volume`).
    @test report.n_moments_throw == 0
    @test report.n_negative_volume == 0

    # Float and exact backends agree on entry count.
    o_f = compute_overlap(lag, frame; backend = :float, moment_order = 1)
    @test n_entries(o_f) == n_entries(o)
    @test isapprox(total_overlap_volume(o), total_overlap_volume(o_f); atol = 1e-4)

    # Recorded drops (if any) only carry the `:empty` kind.
    for d in report.drops
        @test d.kind === :empty
    end
end

# ---------------------------------------------------------------------------
# Test 3 — show formatting
# ---------------------------------------------------------------------------

@testset "OverlapDropReport: show formatting (clean)" begin
    report = OverlapDropReport(0, 0, 0,
        NamedTuple{(:lag_idx, :eul_idx, :kind),
                   Tuple{Int32, Int32, Symbol}}[])
    s = sprint(show, report)
    @test occursin("OverlapDropReport", s)
    @test occursin("0 data-loss drops", s)
end

@testset "OverlapDropReport: show formatting (with drops)" begin
    drops = [
        (lag_idx = Int32(12), eul_idx = Int32(34), kind = :negative_volume),
        (lag_idx = Int32(12), eul_idx = Int32(37), kind = :negative_volume),
        (lag_idx = Int32(15), eul_idx = Int32(2), kind = :moments_throw),
        (lag_idx = Int32(99), eul_idx = Int32(7), kind = :empty),
    ]
    drops_typed = NamedTuple{(:lag_idx, :eul_idx, :kind),
                              Tuple{Int32, Int32, Symbol}}[d for d in drops]
    report = OverlapDropReport(2, 1, 1, drops_typed)
    s = sprint(show, report)
    @test occursin("OverlapDropReport", s)
    @test occursin("3 drops", s)
    @test occursin("2 negative-volume", s)
    @test occursin("1 moments-throw", s)
    @test occursin("1 empty", s)
    @test occursin("lag=12", s)
    @test occursin(":negative_volume", s)
end

# ---------------------------------------------------------------------------
# Test 4 — backward compatibility
# ---------------------------------------------------------------------------

@testset "drop audit: default (audit_drops=false) is bit-for-bit unchanged" begin
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    o = compute_overlap(lag, frame; backend = :exact, moment_order = 1)
    @test o isa GeometricOverlap{2, Float64}
    @test !(o isa Tuple)

    # Explicit audit_drops=false also returns a plain overlap.
    o2 = compute_overlap(lag, frame; backend = :exact, moment_order = 1,
                         audit_drops = false)
    @test o2 isa GeometricOverlap{2, Float64}
    @test n_entries(o) == n_entries(o2)
    @test total_overlap_volume(o) === total_overlap_volume(o2)
end

# ---------------------------------------------------------------------------
# Test 5 — argument validation
# ---------------------------------------------------------------------------

@testset "drop audit: audit_drops=true rejected on :float backend" begin
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    @test_throws ArgumentError compute_overlap(lag, frame;
                                                  backend = :float,
                                                  audit_drops = true)
end
