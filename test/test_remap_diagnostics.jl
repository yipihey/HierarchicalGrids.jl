using Test
using HierarchicalGrids
using HierarchicalGrids.Diagnostics

# ============================================================================
# RemapDiagnostics: construction, reset!, merge!
# ============================================================================

@testset "RemapDiagnostics: fresh construction" begin
    d = RemapDiagnostics{Float64}()
    @test d.liouville_min == typemax(Float64)
    @test d.liouville_max == typemin(Float64)
    @test d.total_volume_in == 0.0
    @test d.total_volume_out == 0.0
    @test d.n_negative_jacobian_cells == 0

    d32 = RemapDiagnostics{Float32}()
    @test d32.liouville_min == typemax(Float32)
    @test d32.liouville_max == typemin(Float32)

    d_default = RemapDiagnostics()
    @test d_default isa RemapDiagnostics{Float64}

    d_typearg = RemapDiagnostics(Float64)
    @test d_typearg isa RemapDiagnostics{Float64}
end

@testset "RemapDiagnostics: reset! restores fresh state" begin
    d = RemapDiagnostics{Float64}()
    d.liouville_min = 0.5
    d.liouville_max = 1.5
    d.total_volume_in = 2.0
    d.total_volume_out = 2.0
    d.n_negative_jacobian_cells = 3
    HierarchicalGrids.Diagnostics.reset!(d)
    @test d.liouville_min == typemax(Float64)
    @test d.liouville_max == typemin(Float64)
    @test d.total_volume_in == 0.0
    @test d.total_volume_out == 0.0
    @test d.n_negative_jacobian_cells == 0
end

@testset "RemapDiagnostics: merge! combines as min/max/sum" begin
    a = RemapDiagnostics{Float64}()
    a.liouville_min = 0.5
    a.liouville_max = 0.9
    a.total_volume_in = 1.0
    a.total_volume_out = 1.0
    a.n_negative_jacobian_cells = 1

    b = RemapDiagnostics{Float64}()
    b.liouville_min = 0.3
    b.liouville_max = 1.2
    b.total_volume_in = 0.5
    b.total_volume_out = 0.5
    b.n_negative_jacobian_cells = 2

    merge!(a, b)
    @test a.liouville_min == 0.3
    @test a.liouville_max == 1.2
    @test a.total_volume_in == 1.5
    @test a.total_volume_out == 1.5
    @test a.n_negative_jacobian_cells == 3
end

# ============================================================================
# Identity remap: SimplicialMesh exactly tiling [0,1]^2 onto a single
# Eulerian cell on [0,1]^2. Two triangles, full coverage.
# ============================================================================

function _identity_setup()
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    sv = Int32[1 4; 2 3; 3 2]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()        # one cell, no refinement
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    src_frames = CellReferenceFrame{2, Float64}[lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = CellReferenceFrame{2, Float64}[eulerian_frame(frame, j) for j in 1:n_cells(eul)]
    return (lag, eul, frame, src_frames, dst_frames)
end

@testset "Identity remap: per-pair Jacobian is exactly 1, full volume coverage" begin
    lag, eul, frame, src_frames, dst_frames = _identity_setup()
    overlap = compute_overlap(lag, frame; moment_order = 0)
    src = ones(1, n_simplices(lag)) .* 7.0
    dst = zeros(1, n_cells(eul))
    diag = RemapDiagnostics{Float64}()
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 0, 0;
                             diagnostics = diag)

    # Each Lagrangian triangle sits entirely inside the single Eulerian cell;
    # entry.volume = 1/2 (triangle area) and source physical volume = 1/2,
    # so the proxy = 1.
    @test isapprox(diag.liouville_min, 1.0; atol = 1e-10)
    @test isapprox(diag.liouville_max, 1.0; atol = 1e-10)
    @test isapprox(diag.total_volume_in, 1.0; atol = 1e-12)
    @test isapprox(diag.total_volume_out, 1.0; atol = 1e-12)
    @test diag.n_negative_jacobian_cells == 0
end

# ============================================================================
# Stretch (×2 in x): Lagrangian on [0,2]×[0,1], Eulerian on [0,1]^2.
# The Eulerian frame does not cover the right half of the Lagrangian domain.
# Each entry covers a fraction of the source triangle.
# ============================================================================

function _stretch_setup()
    # Two Lagrangian triangles tiling [0,2] × [0,1]
    positions = [(0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0)]
    sv = Int32[1 4; 2 3; 3 2]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)

    # Eulerian frame is [0,1]^2, single cell
    eul = HierarchicalMesh{2}()
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    src_frames = CellReferenceFrame{2, Float64}[lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = CellReferenceFrame{2, Float64}[eulerian_frame(frame, j) for j in 1:n_cells(eul)]
    return (lag, eul, frame, src_frames, dst_frames)
end

@testset "Stretch remap: Jacobian proxy < 1, no negative cells" begin
    lag, eul, frame, src_frames, dst_frames = _stretch_setup()
    overlap = compute_overlap(lag, frame; moment_order = 0)
    src = ones(1, n_simplices(lag))
    dst = zeros(1, n_cells(eul))
    diag = RemapDiagnostics{Float64}()
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 0, 0;
                             diagnostics = diag)

    # Each source triangle has physical volume = 1.0 (half of the 2x1 box).
    # The Eulerian cell [0,1]^2 covers part of each. For all overlap entries,
    # entry.volume / source_phys_vol must be in (0, 1].
    @test diag.liouville_min > 0.0
    @test diag.liouville_max <= 1.0 + 1e-12
    @test diag.liouville_min < 1.0  # qualitatively, partial overlap
    # Total overlap area equals the area of [0,1]^2 ∩ Lagrangian-domain = 1.
    @test isapprox(diag.total_volume_in, 1.0; atol = 1e-12)
    @test isapprox(diag.total_volume_out, 1.0; atol = 1e-12)
    @test diag.n_negative_jacobian_cells == 0
end

# ============================================================================
# E → L direction also writes diagnostics
# ============================================================================

@testset "E → L: diagnostics accumulator is populated" begin
    lag, eul, frame, src_frames, dst_frames = _identity_setup()
    overlap = compute_overlap(lag, frame; moment_order = 0)
    src_eul = ones(1, n_cells(eul)) .* 11.0
    dst_lag = zeros(1, n_simplices(lag))
    diag = RemapDiagnostics{Float64}()
    # E → L: source frames are Eulerian, target frames Lagrangian.
    polynomial_remap_e_to_l!(dst_lag, src_eul, overlap,
                              dst_frames, src_frames, 0, 0;
                              diagnostics = diag)
    # The Eulerian source cell has physical volume 1; total overlap volume 1.
    # entry.volume / source_phys_vol per pair is in (0, 1].
    @test diag.liouville_min > 0.0
    @test diag.liouville_max <= 1.0 + 1e-12
    @test isapprox(diag.total_volume_in, 1.0; atol = 1e-12)
    @test isapprox(diag.total_volume_out, 1.0; atol = 1e-12)
    @test diag.n_negative_jacobian_cells == 0
end

# ============================================================================
# Backward compatibility: omitting the kwarg keeps existing behavior.
# ============================================================================

@testset "No-diagnostics path: numerical result unchanged" begin
    lag, eul, frame, src_frames, dst_frames = _identity_setup()
    overlap = compute_overlap(lag, frame; moment_order = 0)
    src = ones(1, n_simplices(lag)) .* 7.0
    dst_no = zeros(1, n_cells(eul))
    dst_yes = zeros(1, n_cells(eul))
    polynomial_remap_l_to_e!(dst_no, src, overlap, src_frames, dst_frames, 0, 0)
    diag = RemapDiagnostics{Float64}()
    polynomial_remap_l_to_e!(dst_yes, src, overlap, src_frames, dst_frames, 0, 0;
                             diagnostics = diag)
    @test dst_no == dst_yes
end

# ============================================================================
# Round-trip: reset! + merge! after a real accumulation gives sane values.
# ============================================================================

@testset "Round-trip: reset! + merge! after real accumulation" begin
    lag, eul, frame, src_frames, dst_frames = _identity_setup()
    overlap = compute_overlap(lag, frame; moment_order = 0)
    src = ones(1, n_simplices(lag))
    dst = zeros(1, n_cells(eul))

    # First pass into d1.
    d1 = RemapDiagnostics{Float64}()
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 0, 0;
                             diagnostics = d1)
    @test d1.total_volume_in == d1.total_volume_out
    @test isapprox(d1.total_volume_in, 1.0; atol = 1e-12)

    # Second pass into a fresh d2; merge into d1.
    d2 = RemapDiagnostics{Float64}()
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 0, 0;
                             diagnostics = d2)
    merged = RemapDiagnostics{Float64}()
    HierarchicalGrids.Diagnostics.reset!(merged)
    merge!(merged, d1)
    merge!(merged, d2)
    @test isapprox(merged.total_volume_in, 2.0; atol = 1e-12)
    @test isapprox(merged.total_volume_out, 2.0; atol = 1e-12)
    @test merged.n_negative_jacobian_cells == 0
    @test isapprox(merged.liouville_min, 1.0; atol = 1e-10)
    @test isapprox(merged.liouville_max, 1.0; atol = 1e-10)

    # reset! and re-accumulate into d1: should match a fresh single pass.
    HierarchicalGrids.Diagnostics.reset!(d1)
    @test d1.total_volume_in == 0.0
    @test d1.total_volume_out == 0.0
    @test d1.n_negative_jacobian_cells == 0
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 0, 0;
                             diagnostics = d1)
    @test isapprox(d1.total_volume_in, 1.0; atol = 1e-12)
    @test isapprox(d1.total_volume_out, 1.0; atol = 1e-12)
end
