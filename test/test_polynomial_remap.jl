using Test
using HierarchicalGrids
using LinearAlgebra: det, I, issymmetric

# ============================================================================
# Cell reference frames and their pullback matrices
# ============================================================================

@testset "AxisAlignedRef construction and validation" begin
    f = AxisAlignedRef((0.0, 0.0), (1.0, 1.0))
    @test f.lo == (0.0, 0.0)
    @test f.hi == (1.0, 1.0)
    @test_throws ArgumentError AxisAlignedRef((1.0, 0.0), (0.0, 1.0))
end

@testset "SimplicialRef construction" begin
    f = SimplicialRef((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
    @test f.anchor == (0.0, 0.0)
    @test f.edges == ((1.0, 0.0), (0.0, 1.0))
end

@testset "lagrangian_frame builds correct frame from mesh" begin
    positions = [(0.5, 0.7), (1.5, 0.7), (0.5, 1.7)]
    sv = reshape(Int32[1, 2, 3], 3, 1); sn = zeros(Int32, 3, 1)
    mesh = SimplicialMesh{2, Float64}(positions, sv, sn)
    f = lagrangian_frame(mesh, 1)
    @test f.anchor == (0.5, 0.7)
    @test f.edges[1] == (1.0, 0.0)
    @test f.edges[2] == (0.0, 1.0)
end

@testset "eulerian_frame builds correct frame from frame+leaf" begin
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (4.0, 8.0))
    leaves = enumerate_leaves(eul)
    f = eulerian_frame(frame, leaves[1])
    # The first leaf is at (0, 0)..(2, 4)
    @test f.lo == (0.0, 0.0)
    @test f.hi == (2.0, 4.0)
end

@testset "Pullback P=0 returns 1×1 identity" begin
    fa = AxisAlignedRef((0.5, 1.0), (1.5, 3.0))
    @test reference_to_physical_pullback(fa, 0) == [1.0;;]
    fs = SimplicialRef((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
    @test reference_to_physical_pullback(fs, 0) == [1.0;;]
end

@testset "AxisAlignedRef pullback: unit-cube identity" begin
    # Reference frame ξ ∈ [0,1] × [0,1] → physical x = ξ. Pullback should be identity.
    f = AxisAlignedRef((0.0, 0.0), (1.0, 1.0))
    T_mat = reference_to_physical_pullback(f, 2)
    n = moments_length(2, 2)  # 6
    @test size(T_mat) == (n, n)
    @test T_mat ≈ Matrix{Float64}(I, n, n) atol=1e-14
end

@testset "AxisAlignedRef pullback: shifted box" begin
    # Box [1, 3] × [2, 6]: ξ_1 = (x - 1)/2, ξ_2 = (y - 2)/4
    # ξ_1^1 = -1/2 + (1/2) x → T[2, 1] = -1/2, T[2, 2] = 1/2
    # ξ_2^1 = -1/2 + (1/4) y → T[3, 1] = -1/2, T[3, 3] = 1/4
    f = AxisAlignedRef((1.0, 2.0), (3.0, 6.0))
    T_mat = reference_to_physical_pullback(f, 1)
    @test T_mat[1, :] ≈ [1.0, 0.0, 0.0]
    @test T_mat[2, :] ≈ [-0.5, 0.5, 0.0]
    @test T_mat[3, :] ≈ [-0.5, 0.0, 0.25]
end

@testset "SimplicialRef pullback: unit-simplex identity-anchor" begin
    # Vertices (0,0), (1,0), (0,1): ξ_1 = x, ξ_2 = y. Pullback = identity.
    f = SimplicialRef((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
    T_mat = reference_to_physical_pullback(f, 2)
    n = moments_length(2, 2)
    @test T_mat ≈ Matrix{Float64}(I, n, n) atol=1e-14
end

@testset "SimplicialRef pullback: shifted/scaled simplex" begin
    # Vertices (1,1), (3,1), (1,4): anchor=(1,1), edges=(2,0),(0,3)
    # ξ_1 = (x-1)/2, ξ_2 = (y-1)/3 (because A is diagonal)
    f = SimplicialRef((1.0, 1.0), ((2.0, 0.0), (0.0, 3.0)))
    T_mat = reference_to_physical_pullback(f, 1)
    @test T_mat[2, :] ≈ [-0.5, 0.5, 0.0]
    @test T_mat[3, :] ≈ [-1/3, 0.0, 1/3]
end

@testset "Pullback applied to known monomial" begin
    # For frame f with pullback T, evaluating ξ^α as poly in x and integrating
    # over a known overlap polytope should match the analytical value.
    # Use frame for unit triangle, evaluate ξ_1^2 over the triangle:
    # ∫_T ξ_1^2 dξ = ∫_T x^2 dξ_1 dξ_2 = 1/12 (over unit simplex)
    f = SimplicialRef((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
    T_mat = reference_to_physical_pullback(f, 2)
    # ξ_1^2 has multi-index (2, 0); flat index = 4 in graded-lex
    # T[4, :] gives x-monomial coeffs. Should be [0, 0, 0, 1, 0, 0] (just x^2).
    @test T_mat[4, :] ≈ [0.0, 0.0, 0.0, 1.0, 0.0, 0.0] atol=1e-14
end

# ============================================================================
# Reference mass matrices
# ============================================================================

@testset "AxisAlignedRef mass matrix: order 0" begin
    f = AxisAlignedRef((0.0, 0.0), (1.0, 1.0))
    M = reference_mass_matrix(f, 0)
    @test M == [1.0;;]
end

@testset "AxisAlignedRef mass matrix: order 1, ∫_{[0,1]²} ξ^α ξ^β dξ" begin
    # multi = [(0,0), (1,0), (0,1)]
    # ∫ ξ^(0,0) · ξ^(0,0) = 1
    # ∫ ξ^(0,0) · ξ^(1,0) = 1/2
    # ∫ ξ^(0,0) · ξ^(0,1) = 1/2
    # ∫ ξ^(1,0) · ξ^(1,0) = 1/3
    # ∫ ξ^(1,0) · ξ^(0,1) = 1/4
    # ∫ ξ^(0,1) · ξ^(0,1) = 1/3
    f = AxisAlignedRef((0.0, 0.0), (1.0, 1.0))
    M = reference_mass_matrix(f, 1)
    @test M[1, 1] ≈ 1.0
    @test M[1, 2] ≈ 0.5
    @test M[2, 1] ≈ 0.5
    @test M[2, 2] ≈ 1/3
    @test M[2, 3] ≈ 1/4
    @test M[3, 3] ≈ 1/3
    @test issymmetric(M)
end

@testset "SimplicialRef mass matrix: ∫_Δ ξ^α ξ^β dξ" begin
    # ∫_{Δ²} 1 dξ = 1/2
    # ∫_{Δ²} ξ_1 dξ = 1/6; ∫_{Δ²} ξ_1^2 dξ = 1/12
    # ∫_{Δ²} ξ_1 ξ_2 dξ = 1/24
    f = SimplicialRef((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
    M = reference_mass_matrix(f, 1)
    @test M[1, 1] ≈ 1/2
    @test M[1, 2] ≈ 1/6
    @test M[2, 2] ≈ 1/12
    @test M[2, 3] ≈ 1/24
    @test issymmetric(M)
end

# ============================================================================
# Per-pair scalar integration
# ============================================================================

@testset "integrate_polynomial_over_overlap: constant 1 ⇒ overlap volume" begin
    # Construct one entry: overlap volume = 0.3, moments to order 0 = [0.3]
    b = OverlapBuilder{2, Float64}(0)
    push_overlap!(b, 1, 1, 0.3, (0.0, 0.0), [0.3])
    o = finalize_overlap(b, 1, 1)
    # Source frame: any (constant pullback is 1×1 identity for P=0)
    f = SimplicialRef((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
    T_mat = reference_to_physical_pullback(f, 0)
    # Constant source = 5.0
    src_co = [5.0]
    val = integrate_polynomial_over_overlap(src_co, T_mat, o.entries[1], 0)
    @test val ≈ 5.0 * 0.3
end

@testset "integrate_polynomial_over_overlap: linear source over triangle" begin
    # Triangle with vertices (0,0), (1,0), (0,1); source p(ξ) = 2 + 3ξ_1 - 4ξ_2 in
    # the triangle's reference frame. In physical (which equals reference here),
    # ∫_T p dx = 2 * 1/2 + 3 * 1/6 - 4 * 1/6 = 1 + 1/2 - 2/3 = 5/6
    f = SimplicialRef((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))
    T_mat = reference_to_physical_pullback(f, 1)
    # Use the actual triangle overlap with the unit box (which contains it):
    # entry.moments order 1 = [1/2, 1/6, 1/6] (volume, ∫x, ∫y)
    b = OverlapBuilder{2, Float64}(1)
    push_overlap!(b, 1, 1, 0.5, (1/3, 1/3), [0.5, 1/6, 1/6])
    o = finalize_overlap(b, 1, 1)
    src_co = [2.0, 3.0, -4.0]
    val = integrate_polynomial_over_overlap(src_co, T_mat, o.entries[1], 1)
    @test val ≈ 5/6 atol=1e-12
end

# ============================================================================
# End-to-end polynomial remap: reproduction tests
# ============================================================================

# Setup: two Lagrangian triangles tile the unit square; Eulerian root refined
# once into 4 leaves. Full overlap.
function _build_tile_setup()
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    sv = Int32[1 4; 2 3; 3 2]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    src_frames = CellReferenceFrame{2, Float64}[lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = CellReferenceFrame{2, Float64}[eulerian_frame(frame, j) for j in 1:n_cells(eul)]
    return (lag, eul, frame, src_frames, dst_frames)
end

@testset "Polynomial remap: constant reproduction (P=0)" begin
    lag, eul, frame, src_frames, dst_frames = _build_tile_setup()
    overlap = compute_overlap(lag, frame; moment_order = 0)
    src = ones(1, 2) .* 7.0
    dst = zeros(1, n_cells(eul))
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 0, 0)
    for j in 1:n_cells(eul)
        if dst[1, j] != 0
            @test dst[1, j] ≈ 7.0 atol=1e-12
        end
    end
end

@testset "Polynomial remap: linear reproduction (P_src=P_dst=1)" begin
    lag, eul, frame, src_frames, dst_frames = _build_tile_setup()
    overlap = compute_overlap(lag, frame; moment_order = 2)
    # Want physical p(x, y) = 5 + 2x + 3y everywhere.
    # Per-triangle ref → physical mapping makes this:
    src = zeros(3, 2)
    src[:, 1] = [5.0, 2.0, 3.0]   # tri 1 anchor (0,0), edges (1,0),(0,1)
    src[:, 2] = [10.0, -2.0, -3.0] # tri 2 anchor (1,1), edges (-1,0),(0,-1)
    dst = zeros(3, n_cells(eul))
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 1, 1)
    for j in 1:n_cells(eul)
        if any(dst[:, j] .!= 0)
            lo, hi = cell_physical_box(frame, j)
            h1 = hi[1] - lo[1]; h2 = hi[2] - lo[2]
            expected = [5 + 2*lo[1] + 3*lo[2], 2*h1, 3*h2]
            @test dst[:, j] ≈ expected atol=1e-10
        end
    end
end

@testset "Polynomial remap: cubic reproduction (P_src=P_dst=3)" begin
    lag, eul, frame, src_frames, dst_frames = _build_tile_setup()
    overlap = compute_overlap(lag, frame; moment_order = 6)
    # Physical cubic polynomial in graded-lex order (10 coeffs)
    phys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # Project into each source frame
    src = zeros(10, 2)
    for i in 1:2
        T_mat = reference_to_physical_pullback(src_frames[i], 3)
        src[:, i] = T_mat' \ phys
    end
    dst = zeros(10, n_cells(eul))
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 3, 3)
    # Convert each destination back to physical and compare
    for j in 1:n_cells(eul)
        if any(dst[:, j] .!= 0)
            T_mat = reference_to_physical_pullback(dst_frames[j], 3)
            phys_recovered = T_mat' * dst[:, j]
            # Allow ~1e-8 due to ill-conditioning of the cubic mass system on
            # small destination cells (their physical volume is 0.25, so the
            # mass matrix entries scale as 0.25)
            @test maximum(abs.(phys_recovered .- phys)) < 1e-7
        end
    end
end

# ============================================================================
# Conservation tests
# ============================================================================

@testset "Polynomial remap: integral preserved on full overlap" begin
    lag, eul, frame, src_frames, dst_frames = _build_tile_setup()
    overlap = compute_overlap(lag, frame; moment_order = 4)  # P_src + P_dst = 2 + 2

    # Quadratic source. Use phys p(x, y) = 1 + x + y + x^2 + xy + y^2.
    phys = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # ∫_{[0,1]²} p dx = 1 + 1/2 + 1/2 + 1/3 + 1/4 + 1/3 = 35/12
    expected_total = 35/12
    src = zeros(6, 2)
    for i in 1:2
        T_mat = reference_to_physical_pullback(src_frames[i], 2)
        src[:, i] = T_mat' \ phys
    end
    dst = zeros(6, n_cells(eul))
    polynomial_remap_l_to_e!(dst, src, overlap, src_frames, dst_frames, 2, 2)

    # Sum ∫_{T_j} dst_j(ξ_j) dx = sum c_j[α] · |det J_j| · ∫_{ref} ξ^α dξ
    multi = moment_multiindices(2, 2)
    total = 0.0
    for j in 1:n_cells(eul)
        if any(dst[:, j] .!= 0)
            lo, hi = cell_physical_box(frame, j)
            jac = (hi[1] - lo[1]) * (hi[2] - lo[2])
            for (k, α) in enumerate(multi)
                # ∫_{[0,1]²} ξ^α dξ = 1/((α[1]+1)(α[2]+1))
                ref_int = 1.0 / ((α[1] + 1) * (α[2] + 1))
                total += dst[k, j] * jac * ref_int
            end
        end
    end
    @test total ≈ expected_total atol=1e-12
end

@testset "Polynomial remap: argument validation" begin
    lag, eul, frame, src_frames, dst_frames = _build_tile_setup()
    overlap = compute_overlap(lag, frame; moment_order = 2)

    # moment_order too low
    @test_throws ArgumentError accumulate_polynomial_rhs!(
        zeros(3, n_cells(eul)),
        zeros(3, n_simplices(lag)),
        [reference_to_physical_pullback(src_frames[i], 1) for i in 1:n_simplices(lag)],
        [reference_to_physical_pullback(dst_frames[j], 2) for j in 1:n_cells(eul)],
        overlap, 1, 2)  # needs order 3 but overlap has 2

    # Wrong frame array length
    @test_throws ArgumentError polynomial_remap_l_to_e!(
        zeros(3, n_cells(eul)),
        zeros(3, n_simplices(lag)),
        overlap,
        src_frames[1:1],   # wrong length
        dst_frames, 1, 1)

    # Wrong target_coeffs first dim
    @test_throws DimensionMismatch polynomial_remap_l_to_e!(
        zeros(99, n_cells(eul)),    # should be 3 for P_dst=1
        zeros(3, n_simplices(lag)),
        overlap, src_frames, dst_frames, 1, 1)

    # Wrong source_coeffs first dim
    @test_throws DimensionMismatch polynomial_remap_l_to_e!(
        zeros(3, n_cells(eul)),
        zeros(99, n_simplices(lag)),    # should be 3 for P_src=1
        overlap, src_frames, dst_frames, 1, 1)
end

# ============================================================================
# E → L direction
# ============================================================================

@testset "Polynomial remap E → L: constant reproduction" begin
    lag, eul, frame, src_frames, dst_frames = _build_tile_setup()
    overlap = compute_overlap(lag, frame; moment_order = 0)
    # Note: for E → L the "src" is Eulerian (dst_frames in our setup labels)
    # and the "dst" is Lagrangian.
    src_eul = ones(1, n_cells(eul)) .* 11.0
    # Only set covered cells (ones with overlap entries)
    for j in 1:n_cells(eul)
        if length(entries_for_eul(overlap, j)) == 0
            src_eul[1, j] = 0.0
        end
    end
    dst_lag = zeros(1, n_simplices(lag))
    polynomial_remap_e_to_l!(dst_lag, src_eul, overlap,
                              dst_frames, src_frames,  # src=eulerian frames, dst=lagrangian frames
                              0, 0)
    for i in 1:n_simplices(lag)
        if dst_lag[1, i] != 0
            @test dst_lag[1, i] ≈ 11.0 atol=1e-12
        end
    end
end

# ============================================================================
# Sanity check: ordering agrees with Bases module
# ============================================================================

@testset "Moment ordering matches Bases.monomial_exponents" begin
    using HierarchicalGrids.Bases: monomial_exponents
    for D in 1:3, P in 0:3
        from_bases = monomial_exponents(D, P)
        from_overlap = moment_multiindices(D, P)
        @test length(from_bases) == length(from_overlap)
        for k in eachindex(from_bases)
            @test collect(from_bases[k]) == from_overlap[k]
        end
    end
end
