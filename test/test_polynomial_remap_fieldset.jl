using Test
using HierarchicalGrids

# ============================================================================
# polynomial_coeffs_view / polynomial_coeffs_matrix / set_polynomial_coeffs_matrix!
# ============================================================================

@testset "polynomial_coeffs_view: SoA returns a writable view" begin
    basis = MonomialBasis{2, 1}()
    pfs = allocate_polynomial_fields(SoA, basis, 5; density = Float64)
    # Set some values via the field-view API
    for i in 1:5
        pfs.density[i] = [Float64(i), Float64(2i), Float64(3i)]
    end
    M = polynomial_coeffs_view(pfs, :density)
    @test size(M) == (3, 5)
    @test M[1, 3] == 3.0
    @test M[2, 4] == 8.0
    @test M[3, 1] == 3.0
    # Writes through the view should be visible via the field API
    M[1, 2] = 999.0
    @test pfs.density[2][1] == 999.0
end

@testset "polynomial_coeffs_view: AoS throws" begin
    basis = MonomialBasis{2, 1}()
    pfs = allocate_polynomial_fields(AoS, basis, 5; density = Float64)
    @test_throws ArgumentError polynomial_coeffs_view(pfs, :density)
end

@testset "polynomial_coeffs_view: nonexistent field throws" begin
    basis = MonomialBasis{2, 1}()
    pfs = allocate_polynomial_fields(SoA, basis, 5; density = Float64)
    @test_throws ArgumentError polynomial_coeffs_view(pfs, :velocity)
end

@testset "polynomial_coeffs_matrix and set: round-trip" begin
    basis = MonomialBasis{2, 2}()  # 6 coeffs
    n = 4
    for layout in (SoA, AoS)
        pfs = allocate_polynomial_fields(layout, basis, n; density = Float64)
        for i in 1:n
            pfs.density[i] = collect(Float64, 1:6) .+ i*10
        end
        M = polynomial_coeffs_matrix(pfs, :density)
        @test size(M) == (6, n)
        @test M[1, 1] == 11.0
        @test M[6, 4] == 46.0

        # Modify the matrix and write back
        M2 = M .+ 100
        set_polynomial_coeffs_matrix!(pfs, :density, M2)
        @test pfs.density[2][3] == 23.0 + 100.0
        @test pfs.density[4][6] == 46.0 + 100.0
    end
end

@testset "set_polynomial_coeffs_matrix!: dimension mismatch throws" begin
    basis = MonomialBasis{2, 1}()
    pfs = allocate_polynomial_fields(SoA, basis, 3; density = Float64)
    bad = zeros(2, 3)
    @test_throws DimensionMismatch set_polynomial_coeffs_matrix!(pfs, :density, bad)
end

# ============================================================================
# polynomial_remap_field!
# ============================================================================

# Helper: build the standard test mesh pair: two-triangle Lagrangian tiling
# of [0,1]^2, refined-once Eulerian quadtree, EulerianFrame on [0,1]^2.
function _build_test_pair()
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    sv = Int32[1 2; 2 4; 3 3]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    return lag, eul, frame
end

# Helper: linear field f(x, y) = a + b*x + c*y in PHYSICAL coordinates.
# Express in each simplex reference frame. For the standard tiling above:
#   Simplex 1 (0,0)-(1,0)-(0,1): ξ_1 = x, ξ_2 = y ⇒ coeffs [a, b, c]
#   Simplex 2 (1,0)-(1,1)-(0,1): x = 1 - ξ_2, y = ξ_1 + ξ_2
#     ⇒ f = a + b(1-ξ_2) + c(ξ_1 + ξ_2) = (a+b) + c*ξ_1 + (c-b)*ξ_2
function _set_linear_lagrangian!(pfs, a, b, c)
    pfs.density[1] = [a, b, c]
    pfs.density[2] = [a + b, c, c - b]
    return pfs
end

@testset "polynomial_remap_field! L→E: linear reproduction (P=1)" begin
    lag, eul, frame = _build_test_pair()
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    _set_linear_lagrangian!(src_pfs, 1.0, 2.0, 3.0)

    overlap = compute_overlap(lag, frame; moment_order = 2)  # P_src + P_dst = 2
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    polynomial_remap_field!(dst_pfs, src_pfs, :density, overlap, src_frames, dst_frames)

    # Verify exact reproduction at every leaf cell's centroid.
    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        cx = (lo[1] + hi[1])/2; cy = (lo[2] + hi[2])/2
        coeffs = collect(dst_pfs.density[ci])
        # Eulerian ref ξ ∈ [0,1]², centroid at (0.5, 0.5)
        val = coeffs[1] + 0.5*coeffs[2] + 0.5*coeffs[3]
        expected = 1.0 + 2.0*cx + 3.0*cy
        @test val ≈ expected atol=1e-10
    end
end

@testset "polynomial_remap_field! L→E: constant reproduction (P=0)" begin
    lag, eul, frame = _build_test_pair()
    basis = MonomialBasis{2, 0}()  # constant, 1 coeff
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    src_pfs.density[1] = [7.5]
    src_pfs.density[2] = [7.5]

    overlap = compute_overlap(lag, frame; moment_order = 0)
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    polynomial_remap_field!(dst_pfs, src_pfs, :density, overlap, src_frames, dst_frames)

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        @test dst_pfs.density[ci][1] ≈ 7.5 atol=1e-12
    end
end

@testset "polynomial_remap_field! L→E: cubic reproduction (P=3)" begin
    lag, eul, frame = _build_test_pair()
    basis = MonomialBasis{2, 3}()  # 10 coeffs
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)

    # Choose a known cubic polynomial in physical coords:
    #   f(x, y) = 0.5 + 1.2*x - 0.7*y + 0.3*x^2 - 0.4*x*y + 0.1*y^2
    #            + 0.05*x^3 - 0.06*x^2 y + 0.07*x*y^2 - 0.08*y^3
    # Coefficients in physical-monomial graded-lex:
    #   (0,0)=0.5, (1,0)=1.2, (0,1)=-0.7,
    #   (2,0)=0.3, (1,1)=-0.4, (0,2)=0.1,
    #   (3,0)=0.05, (2,1)=-0.06, (1,2)=0.07, (0,3)=-0.08
    f_phys = (x, y) -> 0.5 + 1.2*x - 0.7*y + 0.3*x^2 - 0.4*x*y + 0.1*y^2 +
                       0.05*x^3 - 0.06*x^2*y + 0.07*x*y^2 - 0.08*y^3

    # For each simplex, fit a cubic polynomial in its reference frame that equals f_phys
    # at every reference monomial test point. Easier: convert physical-monomial coeffs
    # to reference-monomial coeffs via the inverse pullback.
    # The pullback T satisfies ξ^α = Σ_β T[α, β] x^β. We have c_phys[β] and want c_ref[α]
    # such that Σ_α c_ref[α] ξ^α = Σ_α c_ref[α] Σ_β T[α, β] x^β = Σ_β (Σ_α c_ref[α] T[α, β]) x^β
    # ⇒ Σ_α c_ref[α] T[α, β] = c_phys[β] for every β
    # ⇒ c_phys = T^T c_ref ⇒ c_ref = (T^T)^{-1} c_phys
    c_phys = [0.5, 1.2, -0.7, 0.3, -0.4, 0.1, 0.05, -0.06, 0.07, -0.08]
    for i in 1:n_simplices(lag)
        T_pull = reference_to_physical_pullback(lagrangian_frame(lag, i), 3)
        c_ref = transpose(T_pull) \ c_phys
        src_pfs.density[i] = c_ref
    end

    overlap = compute_overlap(lag, frame; moment_order = 6)  # P_src + P_dst
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    polynomial_remap_field!(dst_pfs, src_pfs, :density, overlap, src_frames, dst_frames)

    # Verify at multiple points within each leaf
    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        for ξ_x in (0.25, 0.5, 0.75), ξ_y in (0.25, 0.5, 0.75)
            x = lo[1] + ξ_x * (hi[1] - lo[1])
            y = lo[2] + ξ_y * (hi[2] - lo[2])
            coeffs = collect(dst_pfs.density[ci])
            # Evaluate target polynomial in Eulerian reference frame
            # Coeffs are in monomial-of-ξ graded-lex order: (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),
            # (3,0),(2,1),(1,2),(0,3)
            val = coeffs[1] +
                  coeffs[2]*ξ_x + coeffs[3]*ξ_y +
                  coeffs[4]*ξ_x^2 + coeffs[5]*ξ_x*ξ_y + coeffs[6]*ξ_y^2 +
                  coeffs[7]*ξ_x^3 + coeffs[8]*ξ_x^2*ξ_y +
                  coeffs[9]*ξ_x*ξ_y^2 + coeffs[10]*ξ_y^3
            expected = f_phys(x, y)
            @test val ≈ expected atol=1e-10
        end
    end
end

@testset "polynomial_remap_field! E→L: linear reproduction" begin
    lag, eul, frame = _build_test_pair()
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)

    # Set source = f(x, y) = 5.0 + 2.0*x - 1.5*y in PHYSICAL coords for each leaf cell.
    # Need to express in Eulerian reference frame: ξ_d = (x_d - lo_d) / h_d
    # ⇒ x = lo + h*ξ; for unit-square refined-once leaves, h = 0.5
    # f = 5 + 2*(lo_x + 0.5*ξ_x) - 1.5*(lo_y + 0.5*ξ_y)
    #   = (5 + 2*lo_x - 1.5*lo_y) + 1.0*ξ_x + (-0.75)*ξ_y
    f_phys = (x, y) -> 5.0 + 2.0*x - 1.5*y
    for ci in 1:n_cells(eul)
        if is_leaf(eul.cells[ci])
            lo, hi = cell_physical_box(frame, ci)
            src_pfs.density[ci] = [f_phys(lo[1], lo[2]) +
                                    1.0*0 + 0,   # constant after subbing ξ=(0,0)
                                    1.0,
                                    -0.75]
        end
    end

    overlap = compute_overlap(lag, frame; moment_order = 2)
    src_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    dst_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    polynomial_remap_field!(dst_pfs, src_pfs, :density, overlap,
                              src_frames, dst_frames; direction = :e_to_l)

    # Verify each Lagrangian simplex reproduces f at its centroid
    for i in 1:n_simplices(lag)
        verts = simplex_vertex_positions(lag, i)
        cx = sum(v[1] for v in verts) / 3
        cy = sum(v[2] for v in verts) / 3
        # In Lagrangian reference (unit simplex), centroid is ξ = (1/3, 1/3)
        coeffs = collect(dst_pfs.density[i])
        val = coeffs[1] + (1/3)*coeffs[2] + (1/3)*coeffs[3]
        expected = f_phys(cx, cy)
        @test val ≈ expected atol=1e-10
    end
end

@testset "polynomial_remap_field!: same-name overload" begin
    lag, eul, frame = _build_test_pair()
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    _set_linear_lagrangian!(src_pfs, 1.0, 2.0, 3.0)

    overlap = compute_overlap(lag, frame; moment_order = 2)
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]

    # Use the same-name convenience overload
    polynomial_remap_field!(dst_pfs, src_pfs, :density, overlap, src_frames, dst_frames)

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        cx = (lo[1] + hi[1])/2; cy = (lo[2] + hi[2])/2
        coeffs = collect(dst_pfs.density[ci])
        val = coeffs[1] + 0.5*coeffs[2] + 0.5*coeffs[3]
        expected = 1.0 + 2.0*cx + 3.0*cy
        @test val ≈ expected atol=1e-10
    end
end

@testset "polynomial_remap_field!: AoS layout works (with copy fallback)" begin
    lag, eul, frame = _build_test_pair()
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(AoS, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(AoS, basis, n_cells(eul); density = Float64)
    _set_linear_lagrangian!(src_pfs, 1.0, 2.0, 3.0)

    overlap = compute_overlap(lag, frame; moment_order = 2)
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    polynomial_remap_field!(dst_pfs, src_pfs, :density, overlap, src_frames, dst_frames)

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        cx = (lo[1] + hi[1])/2; cy = (lo[2] + hi[2])/2
        coeffs = collect(dst_pfs.density[ci])
        val = coeffs[1] + 0.5*coeffs[2] + 0.5*coeffs[3]
        expected = 1.0 + 2.0*cx + 3.0*cy
        @test val ≈ expected atol=1e-10
    end
end

@testset "polynomial_remap_field!: validation errors" begin
    lag, eul, frame = _build_test_pair()
    basis_mono = MonomialBasis{2, 1}()
    basis_bern = BernsteinBasis{2, 1}()

    overlap = compute_overlap(lag, frame; moment_order = 2)
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]

    # Direction validation
    src_pfs = allocate_polynomial_fields(SoA, basis_mono, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis_mono, n_cells(eul); density = Float64)
    @test_throws ArgumentError polynomial_remap_field!(dst_pfs, src_pfs, :density, overlap,
                                                         src_frames, dst_frames;
                                                         direction = :sideways)

    # Non-monomial basis rejected
    src_bern = allocate_polynomial_fields(SoA, basis_bern, n_simplices(lag); density = Float64)
    @test_throws ArgumentError polynomial_remap_field!(dst_pfs, src_bern, :density, overlap,
                                                         src_frames, dst_frames)

    dst_bern = allocate_polynomial_fields(SoA, basis_bern, n_cells(eul); density = Float64)
    @test_throws ArgumentError polynomial_remap_field!(dst_bern, src_pfs, :density, overlap,
                                                         src_frames, dst_frames)

    # Wrong element count
    src_wrong = allocate_polynomial_fields(SoA, basis_mono, 99; density = Float64)
    @test_throws ArgumentError polynomial_remap_field!(dst_pfs, src_wrong, :density, overlap,
                                                         src_frames, dst_frames)
end

@testset "polynomial_remap_field!: multiple fields share one overlap" begin
    # Pattern dfmm code will use: build overlap once, remap several fields.
    lag, eul, frame = _build_test_pair()
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag);
                                          density = Float64, momentum = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul);
                                          density = Float64, momentum = Float64)

    # density: f(x,y) = 1 + 2x + 3y
    src_pfs.density[1] = [1.0, 2.0, 3.0]
    src_pfs.density[2] = [3.0, 3.0, 1.0]
    # momentum: g(x,y) = 5 - x + 2y
    #   simplex 1: ξ = (x,y), so coeffs (5, -1, 2)
    #   simplex 2: x = 1 - ξ_2, y = ξ_1 + ξ_2, so g = 5 - (1-ξ_2) + 2(ξ_1+ξ_2) = 4 + 2*ξ_1 + 3*ξ_2
    src_pfs.momentum[1] = [5.0, -1.0, 2.0]
    src_pfs.momentum[2] = [4.0, 2.0, 3.0]

    overlap = compute_overlap(lag, frame; moment_order = 2)
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]

    # Remap both fields through the same overlap
    polynomial_remap_field!(dst_pfs, src_pfs, :density,  overlap, src_frames, dst_frames)
    polynomial_remap_field!(dst_pfs, src_pfs, :momentum, overlap, src_frames, dst_frames)

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        cx = (lo[1] + hi[1])/2; cy = (lo[2] + hi[2])/2
        cd = collect(dst_pfs.density[ci])
        cm = collect(dst_pfs.momentum[ci])
        @test cd[1] + 0.5*cd[2] + 0.5*cd[3] ≈ 1.0 + 2*cx + 3*cy atol=1e-10
        @test cm[1] + 0.5*cm[2] + 0.5*cm[3] ≈ 5.0 - cx + 2*cy   atol=1e-10
    end
end

@testset "polynomial_remap_field!: source/target with different field names" begin
    lag, eul, frame = _build_test_pair()
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); rho = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul);     density = Float64)
    src_pfs.rho[1] = [1.0, 2.0, 3.0]
    src_pfs.rho[2] = [3.0, 3.0, 1.0]

    overlap = compute_overlap(lag, frame; moment_order = 2)
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    polynomial_remap_field!(dst_pfs, :density, src_pfs, :rho, overlap, src_frames, dst_frames)

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        cx = (lo[1] + hi[1])/2; cy = (lo[2] + hi[2])/2
        coeffs = collect(dst_pfs.density[ci])
        @test coeffs[1] + 0.5*coeffs[2] + 0.5*coeffs[3] ≈ 1.0 + 2*cx + 3*cy atol=1e-10
    end
end
