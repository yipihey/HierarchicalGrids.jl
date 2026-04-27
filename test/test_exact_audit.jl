using Test
using HierarchicalGrids

# Tests for the IntExact audit harness
# (`src/Diagnostics/exact_audit.jl`).
#
# Covers the user-callable surface (`audit_overlap`,
# `OverlapAuditReport`), the `Base.show` formatting, and a synthetic-
# failure injection to verify the report renders failures legibly.
# The module-load consistency check (`_verify_intexact_consistency`) is
# implicitly verified: if it had thrown at load, `using HierarchicalGrids`
# above would already have errored out.

# ============================================================================
# 1. Module load passes (sanity check)
# ============================================================================

@testset "Module load: HierarchicalGrids loads without IntExact error" begin
    # If the module-load consistency check had failed,
    # `using HierarchicalGrids` would have thrown during
    # `Diagnostics.__init__`. Reaching here means the check passed.
    @test isdefined(HierarchicalGrids, :audit_overlap)
    @test isdefined(HierarchicalGrids, :OverlapAuditReport)
end

# ============================================================================
# 2. audit_overlap() returns a clean report
# ============================================================================

@testset "audit_overlap: clean report on canonical battery" begin
    report = audit_overlap()
    @test report isa OverlapAuditReport
    @test report.n_polytopes_checked > 0
    @test report.n_failed == 0
    @test report.n_passed == report.n_polytopes_checked
    @test report.max_volume_relative_diff < 1e-10
    @test report.max_moment_relative_diff < 1e-10
    @test isempty(report.failures)
end

@testset "audit_overlap: tighter atol still passes" begin
    # The observed max relative diff on canonical polytopes is at the
    # eps(Float64) floor; pass an atol just above that.
    report = audit_overlap(atol = 1e-14)
    @test report.n_failed == 0
end

# ============================================================================
# 3. Report `show` formatting
# ============================================================================

@testset "OverlapAuditReport: Base.show formatting (clean)" begin
    report = audit_overlap()
    s = sprint(show, report)
    @test occursin("OverlapAuditReport", s)
    @test occursin("checked=", s)
    @test occursin("passed=", s)
    @test occursin("failed=0", s)
    @test occursin("max_vol_rel_diff=", s)
    @test occursin("max_moment_rel_diff=", s)
end

# ============================================================================
# 4. Verbose mode doesn't error
# ============================================================================

@testset "audit_overlap: verbose=true runs without error" begin
    # Capture stdout via a pipe so verbose print doesn't pollute test
    # output. (`redirect_stdout(::IOBuffer)` is not supported; we need
    # a `Pipe`.)
    pipe = Pipe()
    Base.link_pipe!(pipe; reader_supports_async = true,
                          writer_supports_async = true)
    old_stdout = stdout
    redirect_stdout(pipe)
    try
        report = audit_overlap(verbose = true)
        @test report.n_failed == 0
    finally
        redirect_stdout(old_stdout)
        close(pipe.in)
    end
    s = read(pipe, String)
    @test occursin("[pass]", s)
    # Every polytope name shows up.
    @test occursin("unit_triangle", s)
    @test occursin("unit_tetrahedron", s)
end

# ============================================================================
# 5. Synthetic failure injection — Base.show renders failures
# ============================================================================

@testset "OverlapAuditReport: Base.show formatting (with failures)" begin
    failures = NamedTuple[(
        polytope_name = "fake_triangle",
        dim = 2,
        expected = 0.5,
        got_float = (volume = 0.5, centroid = (0.333, 0.333)),
        got_exact = (volume = 0.4, centroid = (0.3, 0.3)),
    ),
    (
        polytope_name = "fake_tet",
        dim = 3,
        expected = 1 / 6,
        got_float = (volume = 0.166, centroid = (0.25, 0.25, 0.25)),
        got_exact = (volume = 0.17, centroid = (0.26, 0.25, 0.25)),
    )]

    fake_report = OverlapAuditReport(2, 0, 2, 0.1, 0.05, failures)
    s = sprint(show, fake_report)
    @test occursin("OverlapAuditReport", s)
    @test occursin("failed=2", s)
    @test occursin("fake_triangle", s)
    @test occursin("fake_tet", s)
    @test occursin("D=2", s)
    @test occursin("D=3", s)
    @test occursin("expected=", s)
    @test occursin("float=", s)
    @test occursin("exact=", s)
end

# ============================================================================
# 6. Report fields — type checks
# ============================================================================

@testset "OverlapAuditReport: field types" begin
    report = audit_overlap()
    @test report.n_polytopes_checked isa Int
    @test report.n_passed isa Int
    @test report.n_failed isa Int
    @test report.max_volume_relative_diff isa Float64
    @test report.max_moment_relative_diff isa Float64
    @test report.failures isa Vector{NamedTuple}
end

# ============================================================================
# 7. Per-mesh audit overload (Item B of the debuggability batch)
# ============================================================================

@testset "audit_overlap(lag, frame): clean single-triangle case" begin
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    report = audit_overlap(lag, frame; bits = 16, accumulator = Int128,
                           atol = 1e-3)
    @test report isa OverlapAuditReport
    @test report.n_polytopes_checked == 1   # the single overlap pair
    @test report.n_failed == 0
    @test report.n_passed == 1
    @test report.max_volume_relative_diff < 1e-3
    @test isempty(report.failures)
end

@testset "audit_overlap(lag, frame): two-triangle tile (formerly upstream-buggy)" begin
    # The two-triangle tiling of [0, 1]^2 against a 1-level-refined
    # Eulerian used to expose `R3D.IntExact`'s D = 2 upstream bugs as
    # `:exact_dropped` failures in the audit. Both bugs are fixed
    # upstream as of r3djl commit 154b346 (2026-04-27). This geometry
    # now serves as a regression detector: no `:exact_dropped`
    # failures should appear, and any per-pair diffs must stay within
    # the lattice-resolution tolerance (~1.5e-5 at bits=16).
    positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    sv = Int32[1 1; 2 3; 3 4]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    # Use a tolerance commensurate with bits=16 lattice resolution.
    report = audit_overlap(lag, frame; bits = 16, accumulator = Int128,
                           atol = 1e-4, max_pair_diffs = 32)

    # No drops — upstream now produces valid clips for this geometry.
    kinds = Set(f.kind for f in report.failures)
    @test :exact_dropped ∉ kinds

    # If any per-pair failures remain, they must be within
    # lattice-resolution tolerance. Document the residual numerical
    # difference rather than asserting bit-equality across backends.
    if !isempty(report.failures)
        # Failures are sorted by descending volume_diff.
        diffs = [f.volume_diff for f in report.failures]
        @test diffs == sort(diffs; rev = true)
        @test all(f -> f.volume_diff < 1e-3, report.failures)
    end
end

@testset "audit_overlap(lag, frame): show formatting renders per-pair entries" begin
    positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    sv = Int32[1 1; 2 3; 3 4]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    report = audit_overlap(lag, frame)
    s = sprint(show, report)
    @test occursin("OverlapAuditReport", s)
    @test occursin("failed=", s)
    @test occursin("lag=", s)
    @test occursin("eul=", s)
    @test occursin("kind=", s)
    @test occursin("vol_diff=", s)
end

@testset "audit_overlap(lag, frame): max_pair_diffs cap respected" begin
    positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    sv = Int32[1 1; 2 3; 3 4]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    report = audit_overlap(lag, frame; max_pair_diffs = 2)
    @test length(report.failures) <= 2
end

@testset "audit_overlap(lag, frame): per_pair=false leaves failures empty" begin
    positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    sv = Int32[1 1; 2 3; 3 4]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    report = audit_overlap(lag, frame; per_pair = false)
    @test isempty(report.failures)
    # n_failed still tracks count even though we don't list them.
    @test report.n_failed > 0
end
