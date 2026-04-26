using Test
using HierarchicalGrids
using R3D

# ============================================================================
# uniform_grid_dimensions
# ============================================================================

@testset "uniform_grid_dimensions: root-only mesh is depth-0 uniform" begin
    eul = HierarchicalMesh{2}()
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    info = uniform_grid_dimensions(f)
    @test info !== nothing
    @test info.depth == 0
    @test info.ibox_lo == (0, 0)
    @test info.ibox_hi == (1, 1)
    @test info.d == (1.0, 1.0)
    @test info.leaf_index_map == reshape([1], 1, 1)
end

@testset "uniform_grid_dimensions: refined-once is depth-1, 2x2 grid" begin
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    info = uniform_grid_dimensions(f)
    @test info !== nothing
    @test info.depth == 1
    @test info.ibox_hi == (2, 2)
    @test info.d == (0.5, 0.5)
    @test size(info.leaf_index_map) == (2, 2)
    # Each entry should be a valid leaf cell index
    for k in info.leaf_index_map
        @test is_leaf(eul.cells[k])
    end
end

@testset "uniform_grid_dimensions: refined-twice (batch) is depth-2, 4x4 grid" begin
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    refine_cells!(eul, enumerate_leaves(eul))   # batch refine
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    info = uniform_grid_dimensions(f)
    @test info !== nothing
    @test info.depth == 2
    @test info.ibox_hi == (4, 4)
    @test info.d == (0.25, 0.25)
    @test size(info.leaf_index_map) == (4, 4)
end

@testset "uniform_grid_dimensions: non-uniform refinement returns nothing" begin
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    # Refine ONLY one leaf — non-uniform
    leaves = enumerate_leaves(eul)
    refine_cells!(eul, [leaves[1]])
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    info = uniform_grid_dimensions(f)
    @test info === nothing
end

@testset "uniform_grid_dimensions: non-cubic physical extent" begin
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (0.0, 0.0), (4.0, 8.0))   # 4 wide × 8 tall
    info = uniform_grid_dimensions(f)
    @test info !== nothing
    @test info.d == (2.0, 4.0)   # cell sizes match aspect ratio
end

# ============================================================================
# Streaming remap correctness vs two-phase
# ============================================================================

# Helper: standard two-triangle Lagrangian tiling, batch-uniformly-refined Eulerian
function _build_uniform_pair(eul_depth)
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    sv = Int32[1 2; 2 4; 3 3]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    for _ in 1:eul_depth
        refine_cells!(eul, enumerate_leaves(eul))
    end
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    return lag, eul, frame
end

@testset "Streaming L→E: linear reproduction (P=1)" begin
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    # f(x, y) = 1 + 2x + 3y
    src_pfs.density[1] = [1.0, 2.0, 3.0]
    src_pfs.density[2] = [3.0, 3.0, 1.0]   # adjusted for simplex 2's reference frame

    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    info = uniform_grid_dimensions(frame)
    polynomial_remap_l_to_uniform_e!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    # Verify exact reproduction at every leaf centroid
    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        cx = (lo[1] + hi[1])/2; cy = (lo[2] + hi[2])/2
        coeffs = collect(dst_pfs.density[ci])
        val = coeffs[1] + 0.5*coeffs[2] + 0.5*coeffs[3]
        @test val ≈ 1.0 + 2*cx + 3*cy atol=1e-10
    end
end

@testset "Streaming L→E: cubic reproduction (P=3)" begin
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 3}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)

    # Cubic in physical: f(x, y) = 0.5 + 1.2x - 0.7y + 0.3x² - 0.4xy + 0.1y²
    #                              + 0.05x³ - 0.06x²y + 0.07xy² - 0.08y³
    c_phys = [0.5, 1.2, -0.7, 0.3, -0.4, 0.1, 0.05, -0.06, 0.07, -0.08]
    f_phys = (x, y) -> 0.5 + 1.2*x - 0.7*y + 0.3*x^2 - 0.4*x*y + 0.1*y^2 +
                       0.05*x^3 - 0.06*x^2*y + 0.07*x*y^2 - 0.08*y^3

    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    for i in 1:n_simplices(lag)
        T_pull = reference_to_physical_pullback(src_frames[i], 3)
        src_pfs.density[i] = transpose(T_pull) \ c_phys
    end

    info = uniform_grid_dimensions(frame)
    polynomial_remap_l_to_uniform_e!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        for ξ_x in (0.25, 0.5, 0.75), ξ_y in (0.25, 0.5, 0.75)
            x = lo[1] + ξ_x * (hi[1] - lo[1])
            y = lo[2] + ξ_y * (hi[2] - lo[2])
            coeffs = collect(dst_pfs.density[ci])
            val = coeffs[1] + coeffs[2]*ξ_x + coeffs[3]*ξ_y +
                  coeffs[4]*ξ_x^2 + coeffs[5]*ξ_x*ξ_y + coeffs[6]*ξ_y^2 +
                  coeffs[7]*ξ_x^3 + coeffs[8]*ξ_x^2*ξ_y +
                  coeffs[9]*ξ_x*ξ_y^2 + coeffs[10]*ξ_y^3
            @test val ≈ f_phys(x, y) atol=1e-10
        end
    end
end

@testset "Streaming matches two-phase to round-off (L→E)" begin
    # The two paths implement identical math. They sum per-leaf
    # contributions in different orders (LIFO recursion vs sorted-by-pair),
    # so FP rounding differs at the last few bits — typically ~1e-13
    # relative for P=3 with O(1) coefficients. We assert agreement to
    # 1e-10 × scale to be robust on any platform / OhMyThreads version.
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 3}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_str = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_two = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)

    src_pfs.density[1] = [0.7, -1.3, 0.4, 2.1, -0.9, 1.5, 0.2, -0.6, 1.1, -0.8]
    src_pfs.density[2] = [-0.3, 1.8, 0.6, -0.7, 0.4, -1.0, 1.3, 0.5, -0.4, 0.9]

    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]

    info = uniform_grid_dimensions(frame)
    polynomial_remap_l_to_uniform_e!(dst_str, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    overlap = compute_overlap(lag, frame; moment_order = 6)
    polynomial_remap_field!(dst_two, src_pfs, :density, overlap, src_frames, dst_frames)

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        cs = collect(dst_str.density[ci])
        ct = collect(dst_two.density[ci])
        for k in eachindex(cs)
            scale = max(abs(cs[k]), abs(ct[k]), 1.0)
            @test abs(cs[k] - ct[k]) ≤ 1e-10 * scale
        end
    end
end

@testset "Streaming: non-cubic physical extent works" begin
    # Same shape but on a [0, 4] × [0, 8] domain
    positions = [(0.0, 0.0), (4.0, 0.0), (0.0, 8.0), (4.0, 8.0)]
    sv = Int32[1 2; 2 4; 3 3]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    refine_cells!(eul, enumerate_leaves(eul))
    frame = EulerianFrame(eul, (0.0, 0.0), (4.0, 8.0))

    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)

    # f(x, y) = 1 + x/4 + y/8  (so it's 1 at (0,0), 2 at (4, 0), 2 at (0, 8), 3 at (4, 8))
    # Express in each simplex frame:
    # Simplex 1 (0,0)-(4,0)-(0,8): ξ_1 = x/4, ξ_2 = y/8 ⇒ f = 1 + ξ_1 + ξ_2 ⇒ [1, 1, 1]
    # Simplex 2 (4,0)-(4,8)-(0,8): anchor=(4,0), edges=((0,8),(-4,8))
    #   x = 4 - 4*ξ_2,  y = 8*ξ_1 + 8*ξ_2
    #   f = 1 + (4 - 4*ξ_2)/4 + (8*ξ_1 + 8*ξ_2)/8 = 1 + 1 - ξ_2 + ξ_1 + ξ_2 = 2 + ξ_1
    #     ⇒ [2, 1, 0]
    src_pfs.density[1] = [1.0, 1.0, 1.0]
    src_pfs.density[2] = [2.0, 1.0, 0.0]

    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    info = uniform_grid_dimensions(frame)
    polynomial_remap_l_to_uniform_e!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        cx = (lo[1] + hi[1])/2; cy = (lo[2] + hi[2])/2
        coeffs = collect(dst_pfs.density[ci])
        val = coeffs[1] + 0.5*coeffs[2] + 0.5*coeffs[3]
        @test val ≈ 1.0 + cx/4 + cy/8 atol=1e-10
    end
end

# ============================================================================
# Validation errors
# ============================================================================

@testset "Streaming: throws when frame is not uniformly refined" begin
    lag, eul, frame = _build_uniform_pair(1)
    # Refine only one leaf — non-uniform
    leaves = enumerate_leaves(eul)
    refine_cells!(eul, [leaves[1]])

    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]

    @test_throws ArgumentError polynomial_remap_l_to_uniform_e!(
        dst_pfs, src_pfs, :density, lag, frame, src_frames, dst_frames)
end

@testset "Streaming: throws on non-monomial basis" begin
    lag, eul, frame = _build_uniform_pair(1)
    bern = BernsteinBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, bern, n_simplices(lag); density = Float64)
    mono = MonomialBasis{2, 1}()
    dst_pfs = allocate_polynomial_fields(SoA, mono, n_cells(eul); density = Float64)
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]

    @test_throws ArgumentError polynomial_remap_l_to_uniform_e!(
        dst_pfs, src_pfs, :density, lag, frame, src_frames, dst_frames)
end

@testset "Streaming: convenience same-name overload" begin
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    src_pfs.density[1] = [1.0, 2.0, 3.0]
    src_pfs.density[2] = [3.0, 3.0, 1.0]
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    info = uniform_grid_dimensions(frame)
    polynomial_remap_l_to_uniform_e!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        cx = (lo[1] + hi[1])/2; cy = (lo[2] + hi[2])/2
        coeffs = collect(dst_pfs.density[ci])
        val = coeffs[1] + 0.5*coeffs[2] + 0.5*coeffs[3]
        @test val ≈ 1.0 + 2*cx + 3*cy atol=1e-10
    end
end

@testset "Streaming: workspace reuse across multiple calls" begin
    # Should be able to reuse a single workspace across many remap calls
    # without errors or stale state.
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    info = uniform_grid_dimensions(frame)

    ws = R3D.Flat.VoxelizeWorkspace{2, Float64}(64)

    # First call
    src_pfs.density[1] = [1.0, 2.0, 3.0]
    src_pfs.density[2] = [3.0, 3.0, 1.0]
    polynomial_remap_l_to_uniform_e!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info;
                                       workspace = ws)
    saved_first = [collect(dst_pfs.density[ci]) for ci in 1:n_cells(eul) if is_leaf(eul.cells[ci])]

    # Modify source, second call with same workspace
    src_pfs.density[1] = [5.0, 0.0, 0.0]
    src_pfs.density[2] = [5.0, 0.0, 0.0]   # constant 5
    polynomial_remap_l_to_uniform_e!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info;
                                       workspace = ws)
    leaves = [ci for ci in 1:n_cells(eul) if is_leaf(eul.cells[ci])]
    for (k, ci) in enumerate(leaves)
        coeffs = collect(dst_pfs.density[ci])
        val = coeffs[1] + 0.5*coeffs[2] + 0.5*coeffs[3]
        @test val ≈ 5.0 atol=1e-10
        @test coeffs != saved_first[k]   # should be different from first call
    end
end

@testset "Streaming: precomputed pullbacks give identical results" begin
    # Passing precomputed src_pullbacks/dst_pullbacks via kwargs should give
    # bit-for-bit the same result as letting the function compute them.
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 3}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_no = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_yes = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)

    src_pfs.density[1] = [0.7, -1.3, 0.4, 2.1, -0.9, 1.5, 0.2, -0.6, 1.1, -0.8]
    src_pfs.density[2] = [-0.3, 1.8, 0.6, -0.7, 0.4, -1.0, 1.3, 0.5, -0.4, 0.9]

    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    info = uniform_grid_dimensions(frame)

    # No precomp
    polynomial_remap_l_to_uniform_e!(dst_no, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    # With precomp
    src_pulls = [reference_to_physical_pullback(f, 3) for f in src_frames]
    dst_pulls = Vector{Matrix{Float64}}(undef, n_cells(eul))
    for ci in 1:n_cells(eul)
        if is_leaf(eul.cells[ci])
            dst_pulls[ci] = reference_to_physical_pullback(dst_frames[ci], 3)
        else
            dst_pulls[ci] = zeros(Float64, 10, 10)
        end
    end
    polynomial_remap_l_to_uniform_e!(dst_yes, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info;
                                       src_pullbacks = src_pulls,
                                       dst_pullbacks = dst_pulls)

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        cy = collect(dst_yes.density[ci])
        cn = collect(dst_no.density[ci])
        @test cy ≈ cn atol=1e-14
    end
end

@testset "Streaming: precomputed pullbacks length validation" begin
    lag, eul, frame = _build_uniform_pair(1)
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]

    # Wrong-length src_pullbacks
    bad_src_pulls = Matrix{Float64}[zeros(3, 3) for _ in 1:99]
    @test_throws ArgumentError polynomial_remap_l_to_uniform_e!(
        dst_pfs, src_pfs, :density, lag, frame, src_frames, dst_frames;
        src_pullbacks = bad_src_pulls)

    # Wrong-length dst_pullbacks
    bad_dst_pulls = Matrix{Float64}[zeros(3, 3) for _ in 1:99]
    @test_throws ArgumentError polynomial_remap_l_to_uniform_e!(
        dst_pfs, src_pfs, :density, lag, frame, src_frames, dst_frames;
        dst_pullbacks = bad_dst_pulls)
end

# ============================================================================
# E→L streaming
# ============================================================================

@testset "Streaming E→L: linear reproduction (P=1)" begin
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)

    # Linear field f(x, y) = 5 + 2x - 1.5y on each leaf
    f_phys = (x, y) -> 5.0 + 2.0*x - 1.5*y
    for ci in 1:n_cells(eul)
        if is_leaf(eul.cells[ci])
            lo, hi = cell_physical_box(frame, ci)
            h_x = hi[1] - lo[1]; h_y = hi[2] - lo[2]
            # ξ_d = (x - lo)/h ⇒ x = lo + h*ξ
            # f = 5 + 2(lo_x + h_x*ξ_x) - 1.5(lo_y + h_y*ξ_y)
            #   = (5 + 2*lo_x - 1.5*lo_y) + 2h_x * ξ_x - 1.5h_y * ξ_y
            src_pfs.density[ci] = [f_phys(lo[1], lo[2]), 2*h_x, -1.5*h_y]
        end
    end

    src_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    dst_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    info = uniform_grid_dimensions(frame)
    polynomial_remap_uniform_e_to_l!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    # Each Lagrangian simplex should reproduce f at its centroid (ξ = (1/3, 1/3))
    for i in 1:n_simplices(lag)
        verts = simplex_vertex_positions(lag, i)
        cx = sum(v[1] for v in verts)/3; cy = sum(v[2] for v in verts)/3
        coeffs = collect(dst_pfs.density[i])
        val = coeffs[1] + (1/3)*coeffs[2] + (1/3)*coeffs[3]
        @test val ≈ f_phys(cx, cy) atol=1e-10
    end
end

@testset "Streaming E→L: cubic reproduction (P=3)" begin
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 3}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)

    c_phys = [0.5, 1.2, -0.7, 0.3, -0.4, 0.1, 0.05, -0.06, 0.07, -0.08]
    f_phys = (x, y) -> 0.5 + 1.2*x - 0.7*y + 0.3*x^2 - 0.4*x*y + 0.1*y^2 +
                       0.05*x^3 - 0.06*x^2*y + 0.07*x*y^2 - 0.08*y^3

    src_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    dst_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]

    # Set source by transforming physical coeffs to each cell's reference frame
    for ci in 1:n_cells(eul)
        if is_leaf(eul.cells[ci])
            T_pull = reference_to_physical_pullback(src_frames[ci], 3)
            src_pfs.density[ci] = transpose(T_pull) \ c_phys
        end
    end

    info = uniform_grid_dimensions(frame)
    polynomial_remap_uniform_e_to_l!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    # Each Lagrangian simplex should reproduce f at multiple points
    for i in 1:n_simplices(lag)
        coeffs = collect(dst_pfs.density[i])
        verts = simplex_vertex_positions(lag, i)
        # Sample at various reference points (interior of unit simplex)
        for (s, t) in ((0.2, 0.2), (0.5, 0.2), (0.2, 0.5), (1/3, 1/3))
            # Physical position via the reference map of simplex i
            x = verts[1][1] + s*(verts[2][1] - verts[1][1]) + t*(verts[3][1] - verts[1][1])
            y = verts[1][2] + s*(verts[2][2] - verts[1][2]) + t*(verts[3][2] - verts[1][2])
            # Evaluate target poly in simplex reference: ξ_1 = s, ξ_2 = t
            val = coeffs[1] +
                  coeffs[2]*s + coeffs[3]*t +
                  coeffs[4]*s^2 + coeffs[5]*s*t + coeffs[6]*t^2 +
                  coeffs[7]*s^3 + coeffs[8]*s^2*t + coeffs[9]*s*t^2 + coeffs[10]*t^3
            @test val ≈ f_phys(x, y) atol=1e-10
        end
    end
end

@testset "Streaming matches two-phase to round-off (E→L)" begin
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 3}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_str = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_two = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)

    # Random source coefficients
    for ci in 1:n_cells(eul)
        if is_leaf(eul.cells[ci])
            src_pfs.density[ci] = randn(10)
        end
    end

    src_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    dst_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]

    info = uniform_grid_dimensions(frame)
    polynomial_remap_uniform_e_to_l!(dst_str, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    overlap = compute_overlap(lag, frame; moment_order = 6)
    polynomial_remap_field!(dst_two, src_pfs, :density, overlap, src_frames, dst_frames; direction=:e_to_l)

    for i in 1:n_simplices(lag)
        cs = collect(dst_str.density[i])
        ct = collect(dst_two.density[i])
        # Streaming and two-phase accumulate in different orders (LIFO vs
        # sorted-by-pair-index), so FP rounding differs slightly. Compare
        # in relative terms.
        for k in eachindex(cs)
            scale = max(abs(cs[k]), abs(ct[k]), 1.0)
            @test abs(cs[k] - ct[k]) ≤ 1e-10 * scale
        end
    end
end

@testset "Streaming E→L: convenience same-name overload" begin
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)

    f_phys = (x, y) -> 5.0 + 2.0*x - 1.5*y
    for ci in 1:n_cells(eul)
        if is_leaf(eul.cells[ci])
            lo, hi = cell_physical_box(frame, ci)
            h_x = hi[1]-lo[1]; h_y = hi[2]-lo[2]
            src_pfs.density[ci] = [f_phys(lo[1], lo[2]), 2*h_x, -1.5*h_y]
        end
    end

    src_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    dst_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    info = uniform_grid_dimensions(frame)
    polynomial_remap_uniform_e_to_l!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    for i in 1:n_simplices(lag)
        verts = simplex_vertex_positions(lag, i)
        cx = sum(v[1] for v in verts)/3; cy = sum(v[2] for v in verts)/3
        coeffs = collect(dst_pfs.density[i])
        val = coeffs[1] + (1/3)*coeffs[2] + (1/3)*coeffs[3]
        @test val ≈ f_phys(cx, cy) atol=1e-10
    end
end

@testset "Streaming E→L: precomputed pullbacks give identical results" begin
    lag, eul, frame = _build_uniform_pair(2)
    basis = MonomialBasis{2, 3}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_no = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_yes = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)

    for ci in 1:n_cells(eul)
        if is_leaf(eul.cells[ci])
            src_pfs.density[ci] = randn(10)
        end
    end

    src_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    dst_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    info = uniform_grid_dimensions(frame)

    polynomial_remap_uniform_e_to_l!(dst_no, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    src_pulls = Vector{Matrix{Float64}}(undef, n_cells(eul))
    for ci in 1:n_cells(eul)
        if is_leaf(eul.cells[ci])
            src_pulls[ci] = reference_to_physical_pullback(src_frames[ci], 3)
        else
            src_pulls[ci] = zeros(Float64, 10, 10)
        end
    end
    dst_pulls = [reference_to_physical_pullback(f, 3) for f in dst_frames]
    polynomial_remap_uniform_e_to_l!(dst_yes, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info;
                                       src_pullbacks = src_pulls,
                                       dst_pullbacks = dst_pulls)

    for i in 1:n_simplices(lag)
        cy = collect(dst_yes.density[i])
        cn = collect(dst_no.density[i])
        @test cy ≈ cn atol=1e-14
    end
end

@testset "Streaming E→L: throws on non-uniform Eulerian" begin
    lag, eul, frame = _build_uniform_pair(1)
    leaves = enumerate_leaves(eul)
    refine_cells!(eul, [leaves[1]])

    basis = MonomialBasis{2, 1}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    src_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    dst_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]

    @test_throws ArgumentError polynomial_remap_uniform_e_to_l!(
        dst_pfs, src_pfs, :density, lag, frame, src_frames, dst_frames)
end

# ============================================================================
# Conservation
#
# Total integral of the source polynomial over the union of source cells
# must equal the total integral of the destination polynomial over the
# union of destination cells, when the meshes cover the same domain.
# This is *the* property dfmm depends on; it must hold to round-off
# regardless of polynomial order, mesh resolution, or partial overlaps.
# ============================================================================

# Helper: integrate a per-leaf-piecewise polynomial over an Eulerian frame.
function _integrate_eulerian(pfs, fieldname, eul, frame, P, multi)
    total = 0.0
    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        coeffs = collect(getproperty(pfs, fieldname)[ci])
        any(coeffs .!= 0) || continue
        lo, hi = cell_physical_box(frame, ci)
        jac = (hi[1] - lo[1]) * (hi[2] - lo[2])
        for (k, a) in enumerate(multi)
            ref_int = 1.0 / ((a[1] + 1) * (a[2] + 1))
            total += coeffs[k] * jac * ref_int
        end
    end
    return total
end

# Helper: integrate a per-simplex polynomial over a Lagrangian mesh.
# For the unit reference 2-simplex (vertices (0,0)/(1,0)/(0,1)),
#   ∫ ξ_1^a ξ_2^b dξ = a! b! / (a + b + 2)!
# and the physical-volume Jacobian for a 2-simplex with vertices v1, v2, v3 is
# |det J| = 2 · area(simplex).
function _integrate_lagrangian(pfs, fieldname, lag, multi)
    fact(n) = n <= 1 ? 1 : prod(1:n)
    total = 0.0
    for i in 1:n_simplices(lag)
        coeffs = collect(getproperty(pfs, fieldname)[i])
        any(coeffs .!= 0) || continue
        jac = 2 * simplex_volume(lag, i)
        for (k, a) in enumerate(multi)
            ref_int = fact(a[1]) * fact(a[2]) / fact(a[1] + a[2] + 2)
            total += coeffs[k] * jac * ref_int
        end
    end
    return total
end

@testset "Streaming L→E: conservation of integral (P=2 source, P=2 target)" begin
    lag, eul, frame = _build_uniform_pair(2)
    multi = moment_multiindices(2, 2)
    basis = MonomialBasis{2, 2}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)

    # Quadratic source f(x, y) = 1 + x + y + x² + xy + y² in physical coords.
    # Integral over [0, 1]²: 1 + 1/2 + 1/2 + 1/3 + 1/4 + 1/3 = 35/12.
    phys = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    src_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    dst_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    for i in 1:n_simplices(lag)
        T_mat = reference_to_physical_pullback(src_frames[i], 2)
        src_pfs.density[i] = transpose(T_mat) \ phys
    end

    info = uniform_grid_dimensions(frame)
    polynomial_remap_l_to_uniform_e!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    expected = 35 / 12
    actual = _integrate_eulerian(dst_pfs, :density, eul, frame, 2, multi)
    @test actual ≈ expected atol=1e-12
end

@testset "Streaming E→L: conservation of integral (P=2 source, P=2 target)" begin
    lag, eul, frame = _build_uniform_pair(2)
    multi = moment_multiindices(2, 2)
    basis = MonomialBasis{2, 2}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)

    # Same quadratic source, expressed per Eulerian leaf in its reference frame
    phys = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    src_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    dst_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    for ci in 1:n_cells(eul)
        if is_leaf(eul.cells[ci])
            T_mat = reference_to_physical_pullback(src_frames[ci], 2)
            src_pfs.density[ci] = transpose(T_mat) \ phys
        end
    end

    info = uniform_grid_dimensions(frame)
    polynomial_remap_uniform_e_to_l!(dst_pfs, src_pfs, :density,
                                       lag, frame, src_frames, dst_frames, info)

    expected = 35 / 12
    actual = _integrate_lagrangian(dst_pfs, :density, lag, multi)
    @test actual ≈ expected atol=1e-12
end

@testset "Streaming L→E→L round-trip: conservation through both directions" begin
    # Apply L→E then E→L; total integral must still equal the original.
    # Per-element values may differ (each remap is an L² projection, not
    # the identity), but the integral is invariant.
    lag, eul, frame = _build_uniform_pair(2)
    multi = moment_multiindices(2, 2)
    basis = MonomialBasis{2, 2}()
    src_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)
    mid_pfs = allocate_polynomial_fields(SoA, basis, n_cells(eul); density = Float64)
    dst_pfs = allocate_polynomial_fields(SoA, basis, n_simplices(lag); density = Float64)

    phys = [0.5, 0.7, -0.3, 0.2, 0.4, -0.1]   # arbitrary quadratic
    lag_frames = [lagrangian_frame(lag, i) for i in 1:n_simplices(lag)]
    eul_frames = [eulerian_frame(frame, ci) for ci in 1:n_cells(eul)]
    for i in 1:n_simplices(lag)
        T_mat = reference_to_physical_pullback(lag_frames[i], 2)
        src_pfs.density[i] = transpose(T_mat) \ phys
    end

    info = uniform_grid_dimensions(frame)

    # L → E
    polynomial_remap_l_to_uniform_e!(mid_pfs, src_pfs, :density,
                                       lag, frame, lag_frames, eul_frames, info)
    # E → L
    polynomial_remap_uniform_e_to_l!(dst_pfs, mid_pfs, :density,
                                       lag, frame, eul_frames, lag_frames, info)

    expected_int = _integrate_lagrangian(src_pfs, :density, lag, multi)
    middle_int   = _integrate_eulerian(mid_pfs, :density, eul, frame, 2, multi)
    final_int    = _integrate_lagrangian(dst_pfs, :density, lag, multi)
    @test middle_int ≈ expected_int atol=1e-12
    @test final_int ≈ expected_int atol=1e-12
end
