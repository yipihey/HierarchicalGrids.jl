using Test
using HierarchicalGrids

# ============================================================================
# Helpers
# ============================================================================

const _AUDIT_BACKENDS = (
    Sequential(),
    OhMyThreadsBackend(:dynamic),
    OhMyThreadsBackend(:static),
    OhMyThreadsBackend(:greedy),
    OhMyThreadsBackend(:serial),
)

function assert_audit_reports_equal(r1, r2; ctx::String = "")
    @test r1.n_polytopes_checked == r2.n_polytopes_checked
    @test r1.n_passed == r2.n_passed
    @test r1.n_failed == r2.n_failed
    @test r1.max_volume_relative_diff == r2.max_volume_relative_diff
    @test r1.max_moment_relative_diff == r2.max_moment_relative_diff
    @test length(r1.failures) == length(r2.failures)
end

# ============================================================================
# Canonical-polytope battery
# ============================================================================

@testset "audit_overlap(): canonical battery, all backends agree" begin
    r_seq = audit_overlap(; backend = Sequential())
    @test r_seq.n_polytopes_checked == 9
    @test r_seq.n_failed == 0
    for b in _AUDIT_BACKENDS
        r = audit_overlap(; backend = b)
        assert_audit_reports_equal(r_seq, r; ctx = "canonical / $(b)")
    end
end

@testset "audit_overlap(): default backend forwards through Threading" begin
    original = default_backend()
    try
        # Setting default to Sequential MUST yield the same canonical
        # report as the default :dynamic.
        r_def = audit_overlap()
        set_default_backend!(Sequential())
        r_seq = audit_overlap()
        assert_audit_reports_equal(r_def, r_seq;
                                    ctx = "default vs explicit Sequential")
    finally
        set_default_backend!(original)
    end
end

# ============================================================================
# User-mesh per-pair audit
# ============================================================================

# Build a small (lag, frame) pair for the per-pair audit. We use the
# "two-triangle tile + 1-level refinement" geometry — the canonical case
# in `test_exact_audit.jl` that surfaces upstream IntExact drops, so the
# audit sees BOTH passes and failures (drops) and the per-task partial
# merge is exercised on both axes of the report.
function _build_lag_frame_2d()
    positions = NTuple{2, Float64}[
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
    ]
    sv = Int32[1 1; 2 3; 3 4]  # 2 triangles
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)

    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    return lag, frame
end

# Clean (no-drop) variant for tests that want a non-failing user-mesh
# audit.
function _build_clean_lag_frame_2d()
    positions = NTuple{2, Float64}[(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    return lag, frame
end

@testset "audit_overlap(lag, frame): clean user-mesh, all backends agree" begin
    lag, frame = _build_clean_lag_frame_2d()
    r_seq = audit_overlap(lag, frame; backend = Sequential(),
                          bits = 16, accumulator = Int128, atol = 1e-3)
    @test r_seq.n_polytopes_checked >= 1
    @test r_seq.n_failed == 0
    for b in _AUDIT_BACKENDS
        r = audit_overlap(lag, frame; backend = b,
                          bits = 16, accumulator = Int128, atol = 1e-3)
        assert_audit_reports_equal(r_seq, r; ctx = "clean user-mesh / $(b)")
    end
end

@testset "audit_overlap(lag, frame): drops user-mesh, all backends agree" begin
    lag, frame = _build_lag_frame_2d()
    r_seq = audit_overlap(lag, frame; backend = Sequential(),
                          bits = 16, accumulator = Int128, atol = 1e-10,
                          max_pair_diffs = 32)
    @test r_seq.n_polytopes_checked >= 1
    # Counts/maxes are byte-identical across backends. The failures list
    # is canonicalized via `sort` inside the function (sort by descending
    # volume_diff) so it's also deterministic.
    for b in _AUDIT_BACKENDS
        r = audit_overlap(lag, frame; backend = b,
                          bits = 16, accumulator = Int128, atol = 1e-10,
                          max_pair_diffs = 32)
        assert_audit_reports_equal(r_seq, r; ctx = "drops user-mesh / $(b)")
        # The (post-cap) failure list is identical across backends after the
        # deterministic sort, modulo ties on volume_diff. Check key fields
        # in order.
        @test [(f.lag_idx, f.eul_idx, f.kind, f.volume_diff)
                for f in r.failures] ==
              [(f.lag_idx, f.eul_idx, f.kind, f.volume_diff)
                for f in r_seq.failures]
    end
end

@testset "audit_overlap(lag, frame): per_pair=false respected under all backends" begin
    lag, frame = _build_lag_frame_2d()
    r_seq = audit_overlap(lag, frame; backend = Sequential(), per_pair = false,
                          bits = 16, accumulator = Int128)
    @test isempty(r_seq.failures)
    for b in _AUDIT_BACKENDS
        r = audit_overlap(lag, frame; backend = b, per_pair = false,
                          bits = 16, accumulator = Int128)
        assert_audit_reports_equal(r_seq, r; ctx = "per_pair=false / $(b)")
        @test isempty(r.failures)
    end
end
