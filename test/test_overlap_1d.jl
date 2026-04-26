using HierarchicalGrids
using Test

# ============================================================================
# 1D parity tests for compute_overlap
#
# These exercise the closed-form 1D dispatch added to `r3d_adapter.jl`:
# interval intersection + elementary moment integrals (no r3djl call).
# All tests use SimplicialMesh{1, Float64} × HierarchicalMesh{1} via an
# EulerianFrame wrapping the Eulerian mesh in physical coordinates.
# ============================================================================

@testset "1D overlap: identity self-overlap on [0,1]" begin
    # Two segments tile [0,1]: [0, 0.5] and [0.5, 1].
    positions = [(0.0,), (0.5,), (1.0,)]
    sv = Int32[1 2; 2 3]            # columns are simplices
    sn = Int32[0 1; 2 0]            # left/right neighbours
    lag = SimplicialMesh{1, Float64}(positions, sv, sn)

    # Single-cell HierarchicalMesh on [0,1].
    eul = HierarchicalMesh{1}()
    frame = EulerianFrame(eul, (0.0,), (1.0,))

    overlap = compute_overlap(lag, frame; moment_order = 3)

    @test n_entries(overlap) == 2
    @test total_overlap_volume(overlap) ≈ 1.0 atol = 1e-12

    # Both entries land in eul cell 1; volumes are 0.5 each.
    vols = sort([e.volume for e in overlap.entries])
    @test vols ≈ [0.5, 0.5] atol = 1e-12

    # Centroids: 0.25 (segment 1) and 0.75 (segment 2).
    centroids = sort([e.centroid[1] for e in overlap.entries])
    @test centroids ≈ [0.25, 0.75] atol = 1e-12

    # Higher moments check: for [0, 0.5] the k-th raw moment is 0.5^(k+1)/(k+1).
    # Find the entry corresponding to lag simplex 1.
    e1 = first(filter(e -> e.lag_idx == 1, overlap.entries))
    @test e1.moments[1] ≈ 0.5 atol = 1e-12              # volume
    @test e1.moments[2] ≈ 0.5^2 / 2 atol = 1e-12        # ∫ x dx
    @test e1.moments[3] ≈ 0.5^3 / 3 atol = 1e-12        # ∫ x^2 dx
    @test e1.moments[4] ≈ 0.5^4 / 4 atol = 1e-12        # ∫ x^3 dx
end

@testset "1D overlap: refinement parity" begin
    # Same Lagrangian setup; refine the Eulerian root once → two leaves
    # at [0, 0.5] and [0.5, 1] which should perfectly match the segments.
    positions = [(0.0,), (0.5,), (1.0,)]
    sv = Int32[1 2; 2 3]
    sn = Int32[0 1; 2 0]
    lag = SimplicialMesh{1, Float64}(positions, sv, sn)

    eul = HierarchicalMesh{1}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0,), (1.0,))

    overlap = compute_overlap(lag, frame; moment_order = 3)

    @test total_overlap_volume(overlap) ≈ 1.0 atol = 1e-12
    # Two leaves × one matching segment each (no cross-cell contribution
    # because the segment endpoints align exactly with the leaf boundary).
    @test n_entries(overlap) == 2
    for e in overlap.entries
        @test e.volume ≈ 0.5 atol = 1e-12
    end
end

@testset "1D overlap: round-trip polynomial remap (degree 2)" begin
    # Lagrangian: two segments tiling [0,1]. Eulerian: two leaves on [0,1].
    positions = [(0.0,), (0.5,), (1.0,)]
    sv = Int32[1 2; 2 3]
    sn = Int32[0 1; 2 0]
    lag = SimplicialMesh{1, Float64}(positions, sv, sn)

    eul = HierarchicalMesh{1}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0,), (1.0,))

    P = 2
    overlap = compute_overlap(lag, frame; moment_order = 2 * P)

    src_frames = CellReferenceFrame{1, Float64}[
        lagrangian_frame(lag, i) for i in 1:n_simplices(lag)
    ]
    dst_frames = CellReferenceFrame{1, Float64}[
        eulerian_frame(frame, j) for j in 1:n_cells(eul)
    ]

    n_coeffs = moments_length(1, P)  # = P + 1 = 3 in 1D

    # Hand-build a quadratic source field as monomial coefficients in the
    # Lagrangian reference frames. Pick a target physical polynomial
    # p(x) = 1 + 2x + 3x^2 and pull it back into each simplex's reference
    # frame ξ_i ∈ [0,1] via the pullback matrix.
    phys = [1.0, 2.0, 3.0]
    src = zeros(n_coeffs, n_simplices(lag))
    for i in 1:n_simplices(lag)
        T_mat = reference_to_physical_pullback(src_frames[i], P)
        # Pullback expresses ξ^α as Σ_β T[α,β] x^β; we need source coeffs c
        # in ξ such that p(x) = Σ_α c_α ξ^α. Equivalently c_α = (Tᵀ \ phys)_α
        # restricted to the right basis ordering — same identity used in
        # test_polynomial_remap.jl.
        src[:, i] = T_mat' \ phys
    end

    # L → E
    dst = zeros(n_coeffs, n_cells(eul))
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, P, P)

    # E → L (round-trip back into the same Lagrangian basis).
    src_back = zeros(n_coeffs, n_simplices(lag))
    polynomial_remap_e_to_l!(src_back, dst, overlap, dst_frames, src_frames, P, P)

    # On a tile-and-aligned-refinement setup the L²-projection of an
    # exactly representable polynomial reproduces it: round-trip
    # coefficients must match to round-off.
    @test maximum(abs.(src_back .- src)) < 1e-10
end
