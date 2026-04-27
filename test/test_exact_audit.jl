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
