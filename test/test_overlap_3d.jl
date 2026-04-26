using HierarchicalGrids
using HierarchicalGrids.Diagnostics
using Test

# ============================================================================
# 3D coverage tests for compute_overlap and the polynomial-remap pipeline.
#
# These exercise the D=3 dispatch path in `r3d_adapter.jl` (which routes
# tetrahedra through r3djl's tet/box clipper) using a SimplicialMesh{3, Float64}
# tiling [0,1]^3 via Kuhn's 6-tetrahedra decomposition.
#
# Mirrors the D=2 pattern in test_overlap.jl / test_polynomial_remap.jl /
# test_remap_diagnostics.jl, but only the load-bearing cases that actually
# go through the D=3 r3djl path.
# ============================================================================

# Kuhn triangulation of [0,1]^3 into 6 tetrahedra, one per permutation of
# (e_1, e_2, e_3). All tets share the diagonal from (0,0,0) to (1,1,1) and
# walk through one face-adjacent corner at each step. We post-flip the
# vertex ordering of any tet that comes out with negative signed volume so
# r3djl receives positively oriented tetrahedra (the dispatch falls back to
# `vol <= 0 ⇒ empty` otherwise — see `_overlap_via_r3d!`).
function _kuhn_unit_cube_offset(dx::Float64=0.0, dy::Float64=0.0, dz::Float64=0.0)
    positions = [
        (dx + 0.0, dy + 0.0, dz + 0.0),  # 1
        (dx + 1.0, dy + 0.0, dz + 0.0),  # 2
        (dx + 0.0, dy + 1.0, dz + 0.0),  # 3
        (dx + 1.0, dy + 1.0, dz + 0.0),  # 4
        (dx + 0.0, dy + 0.0, dz + 1.0),  # 5
        (dx + 1.0, dy + 0.0, dz + 1.0),  # 6
        (dx + 0.0, dy + 1.0, dz + 1.0),  # 7
        (dx + 1.0, dy + 1.0, dz + 1.0),  # 8
    ]
    # Six permutation paths from corner 1 to corner 8.
    sv = Int32[
        1 1 1 1 1 1;
        2 2 3 3 5 5;
        4 6 4 7 6 7;
        8 8 8 8 8 8
    ]
    sn = zeros(Int32, 4, 6)
    mesh = SimplicialMesh{3, Float64}(positions, sv, sn)
    # Fix orientation per-tet: tetrahedra with negative signed volume have
    # their two trailing vertices swapped so r3djl sees a positively oriented
    # simplex.
    for i in 1:6
        if simplex_volume(mesh, i) < 0
            tmp = mesh.simplex_vertices[3, i]
            mesh.simplex_vertices[3, i] = mesh.simplex_vertices[4, i]
            mesh.simplex_vertices[4, i] = tmp
        end
    end
    return mesh
end

_kuhn_unit_cube() = _kuhn_unit_cube_offset(0.0, 0.0, 0.0)

# ============================================================================
# 1. Identity self-overlap
# ============================================================================

@testset "3D overlap: Kuhn 6-tet identity self-overlap on [0,1]^3" begin
    lag = _kuhn_unit_cube()
    @test n_simplices(lag) == 6
    # All tets oriented positively post-flip; each has volume 1/6.
    for i in 1:6
        @test simplex_volume(lag, i) ≈ 1/6 atol=1e-12
    end

    eul = HierarchicalMesh{3}()
    frame = EulerianFrame(eul, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    overlap = compute_overlap(lag, frame; moment_order = 1)

    # Single-cell Eulerian: each Lagrangian tet maps to exactly one entry.
    @test n_entries(overlap) == 6
    @test total_overlap_volume(overlap) ≈ 1.0 atol = 1e-12

    # All entries land in cell 1 (the root, which is also the only leaf).
    leaves = enumerate_leaves(eul)
    @test length(leaves) == 1
    es = entries_for_eul(overlap, leaves[1])
    @test length(es) == 6
    @test sum(e.volume for e in es) ≈ 1.0 atol = 1e-12
end

# ============================================================================
# 2. Refined Eulerian (8 octants)
# ============================================================================

@testset "3D overlap: refined Eulerian splits volume into 8 equal octants" begin
    lag = _kuhn_unit_cube()
    eul = HierarchicalMesh{3}()
    refine_cells!(eul, [1])    # 8 octants
    frame = EulerianFrame(eul, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    overlap = compute_overlap(lag, frame; moment_order = 1)

    leaves = enumerate_leaves(eul)
    @test length(leaves) == 8
    @test total_overlap_volume(overlap) ≈ 1.0 atol = 1e-12

    # Each octant should have total overlap volume = 0.125.
    for ci in leaves
        es = entries_for_eul(overlap, ci)
        cell_vol = sum(e.volume for e in es; init = 0.0)
        @test cell_vol ≈ 0.125 atol = 1e-12
    end
end

# ============================================================================
# 3. Translated Lagrangian
# ============================================================================

@testset "3D overlap: translated Lagrangian still tiles its support" begin
    # Lagrangian cube anchored at (0.3, 0, 0); frame is [0,2]×[0,1]×[0,1] so
    # the entire (translated) cube fits.
    lag = _kuhn_unit_cube_offset(0.3, 0.0, 0.0)
    eul = HierarchicalMesh{3}()
    refine_cells!(eul, [1])    # 8 octants in [0,2]×[0,1]×[0,1]
    frame = EulerianFrame(eul, (0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    overlap = compute_overlap(lag, frame; moment_order = 1)

    @test total_overlap_volume(overlap) ≈ 1.0 atol = 1e-12

    # The cube straddles the x = 1.0 split (since it spans [0.3, 1.3]). Both
    # octant columns x ∈ [0,1] and x ∈ [1,2] should hold positive overlap.
    leaves = enumerate_leaves(eul)
    left_total = 0.0; right_total = 0.0
    for ci in leaves
        lo, hi = cell_physical_box(frame, ci)
        es = entries_for_eul(overlap, ci)
        cell_vol = sum(e.volume for e in es; init = 0.0)
        if hi[1] <= 1.0 + 1e-12
            left_total += cell_vol
        else
            right_total += cell_vol
        end
    end
    # Cube spans x ∈ [0.3, 1.3]: left half [0.3, 1.0] = 0.7, right half [1.0, 1.3] = 0.3.
    @test left_total ≈ 0.7 atol = 1e-12
    @test right_total ≈ 0.3 atol = 1e-12
end

# ============================================================================
# 4. Empty case
# ============================================================================

@testset "3D overlap: tetrahedron entirely outside the frame" begin
    positions = [(10.0, 10.0, 10.0), (11.0, 10.0, 10.0),
                 (10.0, 11.0, 10.0), (10.0, 10.0, 11.0)]
    sv = reshape(Int32[1, 2, 3, 4], 4, 1)
    sn = zeros(Int32, 4, 1)
    lag = SimplicialMesh{3, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{3}()
    frame = EulerianFrame(eul, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    overlap = compute_overlap(lag, frame; moment_order = 1)
    @test n_entries(overlap) == 0
    @test total_overlap_volume(overlap) == 0.0
end

# ============================================================================
# 5. Partial overlap (analytical fraction)
# ============================================================================

@testset "3D overlap: tetrahedron straddling x = 0.5 keeps 7/48 of its volume" begin
    # Reference tet (0,0,0), (1,0,0), (0,1,0), (0,0,1) — volume 1/6.
    # Clipping by x ≤ 0.5 keeps the tet minus the small similar-tet piece
    # with x ≥ 0.5 (which has linear scale 1/2, volume 1/6 · (1/2)^3 = 1/48).
    # So the kept fraction has volume 1/6 - 1/48 = 7/48.
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                 (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    sv = reshape(Int32[1, 2, 3, 4], 4, 1)
    sn = zeros(Int32, 4, 1)
    lag = SimplicialMesh{3, Float64}(positions, sv, sn)

    eul = HierarchicalMesh{3}()
    frame = EulerianFrame(eul, (0.0, 0.0, 0.0), (0.5, 1.0, 1.0))
    overlap = compute_overlap(lag, frame; moment_order = 1)

    @test total_overlap_volume(overlap) ≈ 7/48 atol = 1e-12
    @test n_entries(overlap) == 1
end

# ============================================================================
# 6. Polynomial remap round-trip (degree 0)
# ============================================================================

@testset "3D polynomial remap: P=0 round-trip recovers the constant" begin
    lag = _kuhn_unit_cube()
    eul = HierarchicalMesh{3}()
    frame = EulerianFrame(eul, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    overlap = compute_overlap(lag, frame; moment_order = 0)

    src_frames = CellReferenceFrame{3, Float64}[
        lagrangian_frame(lag, i) for i in 1:n_simplices(lag)
    ]
    dst_frames = CellReferenceFrame{3, Float64}[
        eulerian_frame(frame, j) for j in 1:n_cells(eul)
    ]

    src = ones(1, n_simplices(lag))
    dst = zeros(1, n_cells(eul))
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 0, 0)

    # Constant 1 mass-weighted-averages to 1 on the single Eulerian cell.
    @test dst[1, 1] ≈ 1.0 atol = 1e-12

    # E → L round-trip.
    src_back = zeros(1, n_simplices(lag))
    polynomial_remap_e_to_l!(src_back, dst, overlap, dst_frames, src_frames, 0, 0)
    @test maximum(abs.(src_back .- src)) < 1e-12
end

# ============================================================================
# 7. Polynomial remap round-trip (degree 1, linear)
# ============================================================================

@testset "3D polynomial remap: P=1 linear round-trip recovers x + y + z" begin
    lag = _kuhn_unit_cube()
    eul = HierarchicalMesh{3}()
    frame = EulerianFrame(eul, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    P = 1
    overlap = compute_overlap(lag, frame; moment_order = 2 * P)

    src_frames = CellReferenceFrame{3, Float64}[
        lagrangian_frame(lag, i) for i in 1:n_simplices(lag)
    ]
    dst_frames = CellReferenceFrame{3, Float64}[
        eulerian_frame(frame, j) for j in 1:n_cells(eul)
    ]

    n_coeffs = moments_length(3, P)         # = 4
    @test n_coeffs == 4

    # Physical p(x, y, z) = x + y + z, expressed in graded-lex multi-indices
    # [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]: coefficients [0, 1, 1, 1].
    phys = [0.0, 1.0, 1.0, 1.0]

    # Project the physical polynomial into each Lagrangian reference frame.
    src = zeros(n_coeffs, n_simplices(lag))
    for i in 1:n_simplices(lag)
        T_mat = reference_to_physical_pullback(src_frames[i], P)
        src[:, i] = T_mat' \ phys
    end

    dst = zeros(n_coeffs, n_cells(eul))
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, P, P)

    # On a single-cell Eulerian frame matched to [0,1]^3 the destination
    # frame's pullback maps ξ_d = x_d, so the destination coefficients are
    # exactly `phys`.
    @test dst[:, 1] ≈ phys atol = 1e-10

    # Round-trip back to Lagrangian.
    src_back = zeros(n_coeffs, n_simplices(lag))
    polynomial_remap_e_to_l!(src_back, dst, overlap, dst_frames, src_frames, P, P)
    @test maximum(abs.(src_back .- src)) < 1e-10
end

# ============================================================================
# 8. RemapDiagnostics: identity remap in 3D
# ============================================================================

@testset "3D RemapDiagnostics: identity remap is exactly Liouville-1" begin
    lag = _kuhn_unit_cube()
    eul = HierarchicalMesh{3}()
    frame = EulerianFrame(eul, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    overlap = compute_overlap(lag, frame; moment_order = 0)

    src_frames = CellReferenceFrame{3, Float64}[
        lagrangian_frame(lag, i) for i in 1:n_simplices(lag)
    ]
    dst_frames = CellReferenceFrame{3, Float64}[
        eulerian_frame(frame, j) for j in 1:n_cells(eul)
    ]

    src = ones(1, n_simplices(lag))
    dst = zeros(1, n_cells(eul))
    diag = RemapDiagnostics{Float64}()
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 0, 0;
                              diagnostics = diag)

    # Each Lagrangian tet sits entirely inside the single Eulerian cell:
    # entry.volume = 1/6, source-tet physical volume = 1/6, proxy = 1.
    @test isapprox(diag.liouville_min, 1.0; atol = 1e-10)
    @test isapprox(diag.liouville_max, 1.0; atol = 1e-10)
    @test isapprox(diag.total_volume_in,  1.0; atol = 1e-12)
    @test isapprox(diag.total_volume_out, 1.0; atol = 1e-12)
    @test diag.n_negative_jacobian_cells == 0
end

@testset "3D RemapDiagnostics: refined Eulerian preserves total volume" begin
    lag = _kuhn_unit_cube()
    eul = HierarchicalMesh{3}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    overlap = compute_overlap(lag, frame; moment_order = 0)

    src_frames = CellReferenceFrame{3, Float64}[
        lagrangian_frame(lag, i) for i in 1:n_simplices(lag)
    ]
    dst_frames = CellReferenceFrame{3, Float64}[
        eulerian_frame(frame, j) for j in 1:n_cells(eul)
    ]

    src = ones(1, n_simplices(lag))
    dst = zeros(1, n_cells(eul))
    diag = RemapDiagnostics{Float64}()
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 0, 0;
                              diagnostics = diag)

    # Lagrangian source tets sit inside the frame; Eulerian splits them into
    # 8 octants ⇒ proxy < 1 per pair, no negative Jacobians, total volumes 1.
    @test diag.liouville_min > 0.0
    @test diag.liouville_max <= 1.0 + 1e-12
    @test diag.liouville_min < 1.0
    @test isapprox(diag.total_volume_in,  1.0; atol = 1e-12)
    @test isapprox(diag.total_volume_out, 1.0; atol = 1e-12)
    @test diag.n_negative_jacobian_cells == 0
end
