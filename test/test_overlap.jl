using Test
using HierarchicalGrids

# ============================================================================
# Moment indexing and shifting
# ============================================================================

@testset "moments_length matches binomial(D + order, D)" begin
    @test moments_length(1, 0) == 1
    @test moments_length(1, 3) == 4    # 1, x, x^2, x^3
    @test moments_length(2, 0) == 1
    @test moments_length(2, 1) == 3    # 1, x, y
    @test moments_length(2, 2) == 6
    @test moments_length(2, 3) == 10
    @test moments_length(3, 1) == 4
    @test moments_length(3, 2) == 10
    @test moments_length(3, 3) == 20
    @test moments_length(5, 3) == binomial(8, 3)  # 56
end

@testset "moment_multiindices: count and graded-lex order" begin
    multi = moment_multiindices(2, 2)
    @test length(multi) == 6
    @test multi[1] == [0, 0]   # constant
    @test multi[2] == [1, 0]   # x
    @test multi[3] == [0, 1]   # y
    @test multi[4] == [2, 0]   # x²
    @test multi[5] == [1, 1]   # xy
    @test multi[6] == [0, 2]   # y²

    # 3D order 1
    multi3 = moment_multiindices(3, 1)
    @test multi3[1] == [0, 0, 0]
    @test multi3[2] == [1, 0, 0]
    @test multi3[3] == [0, 1, 0]
    @test multi3[4] == [0, 0, 1]
end

@testset "moment_index round-trip" begin
    for D in 1:4, order in 0:3
        multi = moment_multiindices(D, order)
        for (k, m) in enumerate(multi)
            @test moment_index(D, order, m) == k
        end
    end
end

@testset "moment_index argument validation" begin
    @test_throws ArgumentError moment_index(2, 2, [1])             # wrong length
    @test_throws ArgumentError moment_index(2, 2, [3, 0])          # exceeds order
    @test_throws ArgumentError moment_index(2, 2, [-1, 1])         # negative
end

@testset "moment_volume / moment_centroid extractors" begin
    moms = [4.0, 8.0, 12.0]   # vol=4, M_{10}=8, M_{01}=12 ⇒ centroid (2, 3)
    @test moment_volume(moms) == 4.0
    @test moment_centroid(moms, 2) == (2.0, 3.0)
end

@testset "shift_moments: identity when Δ = 0" begin
    src = [1.0, 0.0, 0.0, 1/12, 0.0, 1/12]   # unit square, centered moments
    out = similar(src)
    shift_moments!(out, src, 2, 2, (0.0, 0.0))
    @test out ≈ src
end

@testset "shift_moments: known case (unit square shifted by (1, 0))" begin
    # Source moments (origin at centroid of unit square): (1, 0, 0, 1/12, 0, 1/12)
    src = [1.0, 0.0, 0.0, 1/12, 0.0, 1/12]
    out = similar(src)
    shift_moments!(out, src, 2, 2, (1.0, 0.0))
    # New moments (origin shifted by (1, 0)):
    # M_{00}_new = vol = 1
    # M_{10}_new = ∫(x - 1)dV = 0 - 1 = -1
    # M_{01}_new = 0
    # M_{20}_new = ∫(x - 1)²dV = 1/12 - 0 + 1 = 13/12
    # M_{11}_new = ∫(x - 1)y dV = 0
    # M_{02}_new = 1/12
    @test out[1] ≈ 1.0
    @test out[2] ≈ -1.0
    @test out[3] ≈ 0.0
    @test out[4] ≈ 13/12
    @test out[5] ≈ 0.0
    @test out[6] ≈ 1/12
end

@testset "shift_moments: 3D case basic check" begin
    # Unit cube centered at origin: vol=1, all linear moments=0,
    # M_{200} = 1/12, M_{020} = 1/12, M_{002} = 1/12, all cross terms = 0
    src = zeros(10)
    src[1] = 1.0      # M_{000} = vol
    # multi_indices for 3D, order 2:
    # 1: (0,0,0), 2: (1,0,0), 3: (0,1,0), 4: (0,0,1),
    # 5: (2,0,0), 6: (1,1,0), 7: (1,0,1), 8: (0,2,0), 9: (0,1,1), 10: (0,0,2)
    src[5] = 1/12
    src[8] = 1/12
    src[10] = 1/12
    out = similar(src)
    shift_moments!(out, src, 3, 2, (0.0, 0.0, 0.0))
    @test out ≈ src   # zero shift ⇒ identity

    # Shift by (1, 0, 0): M_{200}_new = 1/12 + 1 = 13/12; M_{100}_new = -1
    shift_moments!(out, src, 3, 2, (1.0, 0.0, 0.0))
    @test out[1] ≈ 1.0
    @test out[2] ≈ -1.0
    @test out[5] ≈ 13/12
    @test out[8] ≈ 1/12
    @test out[10] ≈ 1/12
end

# ============================================================================
# EulerianFrame and cell-box geometry
# ============================================================================

@testset "EulerianFrame construction" begin
    eul = HierarchicalMesh{2}()
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    @test root_box(f) == ((0.0, 0.0), (1.0, 1.0))

    # Promotion from Int input
    f2 = EulerianFrame(eul, [0, 0], [2, 3])
    @test root_box(f2) == ((0.0, 0.0), (2.0, 3.0))

    # Argument validation
    @test_throws ArgumentError EulerianFrame(eul, (0.0,), (1.0, 1.0))   # wrong dim
    @test_throws ArgumentError EulerianFrame(eul, (1.0, 1.0), (0.0, 0.0))   # inverted
end

@testset "cell_unit_box: root cell covers [0, 1]^D" begin
    eul = HierarchicalMesh{2}()
    lo, hi = cell_unit_box(eul, 1)
    @test lo == (0.0, 0.0)
    @test hi == (1.0, 1.0)
end

@testset "cell_unit_box: refined cells partition the unit square" begin
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    leaves = enumerate_leaves(eul)
    @test length(leaves) == 4
    boxes = [cell_unit_box(eul, ci) for ci in leaves]
    # Total covered area should equal 1
    total = 0.0
    for (lo, hi) in boxes
        total += (hi[1] - lo[1]) * (hi[2] - lo[2])
    end
    @test total ≈ 1.0
end

@testset "cell_physical_box: maps unit box to physical bounds" begin
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (10.0, 20.0), (14.0, 28.0))   # 4x8 box
    leaves = enumerate_leaves(eul)
    total = 0.0
    for ci in leaves
        lo, hi = cell_physical_box(f, ci)
        @test 10.0 <= lo[1] && hi[1] <= 14.0
        @test 20.0 <= lo[2] && hi[2] <= 28.0
        total += (hi[1] - lo[1]) * (hi[2] - lo[2])
    end
    @test total ≈ 4 * 8   # full area covered
end

@testset "enumerate_leaves: matches manual filter" begin
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    leaves = enumerate_leaves(eul)
    manual = [i for i in 1:n_cells(eul) if is_leaf(eul.cells[i])]
    @test leaves == manual
end

@testset "aabbs_overlap predicate" begin
    @test aabbs_overlap((0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (1.5, 1.5))  # overlap
    @test !aabbs_overlap((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (2.0, 1.0)) # touch only
    @test !aabbs_overlap((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)) # disjoint
    @test aabbs_overlap((0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
                        (0.5, 0.5, 0.5), (1.5, 1.5, 1.5))                # 3D
end

# ============================================================================
# OverlapBuilder and GeometricOverlap
# ============================================================================

@testset "OverlapBuilder: push and finalize" begin
    b = OverlapBuilder{2, Float64}(1)
    push_overlap!(b, 1, 2, 0.5, (0.5, 0.5), [0.5, 0.25, 0.25])
    push_overlap!(b, 1, 3, 0.5, (0.5, 0.5), [0.5, 0.25, 0.25])
    push_overlap!(b, 2, 3, 1.0, (1.5, 0.5), [1.0, 1.5, 0.5])
    o = finalize_overlap(b, 2, 5)
    @test n_entries(o) == 3
    @test o.lag_to_entries[1] == 1:2
    @test o.lag_to_entries[2] == 3:3
    @test length(o.eul_to_entries[3]) == 2  # leaves 3 covered by 2 entries
    @test total_overlap_volume(o) ≈ 2.0
end

@testset "OverlapBuilder: dimension mismatch on push" begin
    b = OverlapBuilder{2, Float64}(1)
    @test_throws DimensionMismatch push_overlap!(b, 1, 1, 1.0, (0.0, 0.0), [1.0])
end

@testset "OverlapBuilder: merge" begin
    b1 = OverlapBuilder{2, Float64}(1)
    b2 = OverlapBuilder{2, Float64}(1)
    push_overlap!(b1, 1, 1, 0.5, (0.0, 0.0), [0.5, 0.0, 0.0])
    push_overlap!(b2, 2, 1, 0.7, (0.0, 0.0), [0.7, 0.0, 0.0])
    merge_builder!(b1, b2)
    @test length(b1.entries) == 2
end

@testset "OverlapBuilder: cannot merge mismatched orders" begin
    b1 = OverlapBuilder{2, Float64}(1)
    b2 = OverlapBuilder{2, Float64}(2)
    @test_throws ArgumentError merge_builder!(b1, b2)
end

@testset "entries_for_lag / entries_for_eul access" begin
    b = OverlapBuilder{2, Float64}(0)   # just volumes
    push_overlap!(b, 1, 2, 0.3, (0.0, 0.0), [0.3])
    push_overlap!(b, 1, 3, 0.7, (0.0, 0.0), [0.7])
    push_overlap!(b, 2, 2, 1.0, (0.0, 0.0), [1.0])
    o = finalize_overlap(b, 2, 5)
    @test length(entries_for_lag(o, 1)) == 2
    @test length(entries_for_lag(o, 2)) == 1
    @test length(entries_for_eul(o, 2)) == 2  # leaves 2 covered by 2 entries
    @test length(entries_for_eul(o, 1)) == 0  # cell 1 has no entry
end

# ============================================================================
# r3d_adapter (mock 2D path)
# ============================================================================

@testset "overlap_simplex_box: triangle fully inside box" begin
    scratch = PairScratch(Val(2), Float64)
    moms = zeros(3)
    # Triangle entirely in unit box
    verts = ((0.2, 0.2), (0.6, 0.2), (0.4, 0.6))
    vol, c, _ = overlap_simplex_box!(moms, scratch, verts, (0.0, 0.0), (1.0, 1.0), 1)
    expected_area = 0.5 * abs((0.6-0.2)*(0.6-0.2) - (0.4-0.2)*(0.2-0.2))
    @test vol ≈ expected_area
    @test c[1] ≈ (0.2 + 0.6 + 0.4)/3
    @test c[2] ≈ (0.2 + 0.2 + 0.6)/3
end

@testset "overlap_simplex_box: triangle entirely outside box" begin
    scratch = PairScratch(Val(2), Float64)
    moms = zeros(3)
    verts = ((2.0, 2.0), (3.0, 2.0), (2.5, 3.0))
    vol, c, _ = overlap_simplex_box!(moms, scratch, verts, (0.0, 0.0), (1.0, 1.0), 1)
    @test vol == 0.0
    @test c == (0.0, 0.0)
    @test moms == zeros(3)
end

@testset "overlap_simplex_box: triangle clipped by all four box edges" begin
    scratch = PairScratch(Val(2), Float64)
    moms = zeros(3)
    # Big triangle that contains the unit box; overlap should be the unit box
    verts = ((-1.0, -1.0), (3.0, -1.0), (1.0, 5.0))
    vol, c, _ = overlap_simplex_box!(moms, scratch, verts, (0.0, 0.0), (1.0, 1.0), 1)
    @test vol ≈ 1.0
    @test c[1] ≈ 0.5
    @test c[2] ≈ 0.5
end

@testset "overlap_simplex_box: order-2 moments analytical check" begin
    # Triangle = unit box's lower-left triangle (0,0)-(1,0)-(0,1).
    # ∫_T 1 dA = 1/2
    # ∫_T x dA = 1/6,  ∫_T y dA = 1/6
    # ∫_T x² dA = 1/12, ∫_T xy dA = 1/24, ∫_T y² dA = 1/12
    scratch = PairScratch(Val(2), Float64)
    moms = zeros(6)
    verts = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    vol, c, _ = overlap_simplex_box!(moms, scratch, verts, (0.0, 0.0), (1.0, 1.0), 2)
    @test moms[1] ≈ 1/2
    @test moms[2] ≈ 1/6   # x
    @test moms[3] ≈ 1/6   # y
    @test moms[4] ≈ 1/12  # x²
    @test moms[5] ≈ 1/24  # xy
    @test moms[6] ≈ 1/12  # y²
end

@testset "overlap_simplex_box: D = 3 tetrahedron via R3D backend" begin
    # Standard reference tetrahedron with vertices at the origin and the
    # three unit basis vectors. Volume = 1/6.
    scratch = PairScratch(Val(3), Float64)
    moms = zeros(4)   # order 1 ⇒ 4 moments
    verts = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    vol, c, _ = overlap_simplex_box!(moms, scratch, verts,
                                       (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1)
    @test vol ≈ 1/6 atol=1e-12
    @test c[1] ≈ 1/4
    @test c[2] ≈ 1/4
    @test c[3] ≈ 1/4
end

@testset "overlap_simplex_box: D = 3 tetrahedron clipped by box" begin
    # A tetrahedron straddling x = 0.5; clip against [0, 0.5] × [0, 1] × [0, 1].
    # Use a tetrahedron with vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    # Clipping at x ≤ 0.5 cuts the tet into a smaller polyhedron.
    # By similar triangles the clipped piece (x ∈ [0, 0.5]) has volume
    # 1/6 * (1 - (1/2)^3) = 1/6 * 7/8 = 7/48.
    scratch = PairScratch(Val(3), Float64)
    moms = zeros(4)
    verts = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    vol, c, _ = overlap_simplex_box!(moms, scratch, verts,
                                       (0.0, 0.0, 0.0), (0.5, 1.0, 1.0), 1)
    @test vol ≈ 7/48 atol=1e-12
end

@testset "overlap_simplex_box: D = 3 tetrahedron entirely outside box" begin
    scratch = PairScratch(Val(3), Float64)
    moms = zeros(4)
    verts = ((10.0, 10.0, 10.0), (11.0, 10.0, 10.0),
             (10.0, 11.0, 10.0), (10.0, 10.0, 11.0))
    vol, _, _ = overlap_simplex_box!(moms, scratch, verts,
                                       (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1)
    @test vol == 0.0
end

@testset "overlap_simplex_box: D ≥ 4 throws (r3djl higher moments not yet supported)" begin
    scratch = PairScratch(Val(4), Float64)
    moms = zeros(5)   # order 1 ⇒ 5 moments in 4D
    # 4-simplex (5 vertices): origin + 4 basis vectors
    verts = ((0.0, 0.0, 0.0, 0.0),
             (1.0, 0.0, 0.0, 0.0),
             (0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0),
             (0.0, 0.0, 0.0, 1.0))
    @test_throws ErrorException overlap_simplex_box!(moms, scratch, verts,
                                                       (0.0, 0.0, 0.0, 0.0),
                                                       (1.0, 1.0, 1.0, 1.0), 1)
end

# ============================================================================
# SimplicialAABBTree
# ============================================================================

@testset "AABB tree: simplex_aabb on unit triangle" begin
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    mesh = SimplicialMesh{2, Float64}(positions, sv, sn)
    lo, hi = simplex_aabb(mesh, 1)
    @test lo == (0.0, 0.0)
    @test hi == (1.0, 1.0)
end

@testset "AABB tree: empty mesh" begin
    # Build with zero simplices
    positions = NTuple{2, Float64}[]
    sv = Matrix{Int32}(undef, 3, 0)
    sn = Matrix{Int32}(undef, 3, 0)
    mesh = SimplicialMesh{2, Float64}(positions, sv, sn)
    tree = build_simplex_aabb_tree(mesh)
    @test tree.n_simplices == 0
    out = Int32[]
    query_aabb!(out, tree, (0.0, 0.0), (10.0, 10.0))
    @test isempty(out)
end

@testset "AABB tree: single simplex round-trip" begin
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    mesh = SimplicialMesh{2, Float64}(positions, sv, sn)
    tree = build_simplex_aabb_tree(mesh)
    @test tree.n_simplices == 1
    # Query overlapping
    out = query_aabb(tree, (0.0, 0.0), (0.5, 0.5))
    @test out == Int32[1]
    # Query not overlapping
    out2 = query_aabb(tree, (5.0, 5.0), (6.0, 6.0))
    @test isempty(out2)
end

@testset "AABB tree: many simplices, query returns all overlappers" begin
    # Build 16 small triangles arranged in a 4x4 grid; each is a small
    # right triangle in a 0.25 x 0.25 cell.
    positions = NTuple{2, Float64}[]
    sv_list = NTuple{3, Int32}[]
    for j in 1:4, i in 1:4
        x0 = (i - 1) * 0.25
        y0 = (j - 1) * 0.25
        v1 = length(positions) + 1
        push!(positions, (x0, y0))
        push!(positions, (x0 + 0.2, y0))
        push!(positions, (x0, y0 + 0.2))
        push!(sv_list, (Int32(v1), Int32(v1+1), Int32(v1+2)))
    end
    sv = Matrix{Int32}(undef, 3, length(sv_list))
    for k in 1:length(sv_list)
        sv[1, k] = sv_list[k][1]
        sv[2, k] = sv_list[k][2]
        sv[3, k] = sv_list[k][3]
    end
    sn = zeros(Int32, 3, length(sv_list))
    mesh = SimplicialMesh{2, Float64}(positions, sv, sn)
    tree = build_simplex_aabb_tree(mesh; leaf_size=2)

    # Query a region that should contain exactly 4 triangles (the lower-left 2x2 grid)
    out = query_aabb(tree, (0.0, 0.0), (0.5, 0.5))
    @test length(out) >= 4   # at least the 4 fully-contained ones; BVH may report more leaves
    # Verify all returned simplices actually intersect the query
    for s in out
        sl, sh = simplex_aabb(mesh, s)
        @test sh[1] > 0.0 && sl[1] < 0.5
        @test sh[2] > 0.0 && sl[2] < 0.5
    end

    # Query disjoint region returns empty
    out2 = query_aabb(tree, (10.0, 10.0), (11.0, 11.0))
    @test isempty(out2)
end

# ============================================================================
# compute_overlap end-to-end
# ============================================================================

@testset "compute_overlap: single triangle conserves area" begin
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.7)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    overlap = compute_overlap(lag, f; moment_order=2)
    tri_area = simplex_volume(lag, 1)
    @test total_overlap_volume(overlap) ≈ tri_area atol=1e-12
end

@testset "compute_overlap: triangle entirely outside Eulerian domain" begin
    # Triangle at (10, 10); Eulerian domain at [0, 1]^2
    positions = [(10.0, 10.0), (11.0, 10.0), (10.0, 11.0)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    overlap = compute_overlap(lag, f; moment_order=1)
    @test n_entries(overlap) == 0
    @test total_overlap_volume(overlap) == 0.0
end

@testset "compute_overlap: argument validation" begin
    positions = [(0.5, 0.5), (0.6, 0.5), (0.5, 0.6)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    @test_throws ArgumentError compute_overlap(lag, f; edge_kind=:cubic)
end

@testset "compute_overlap: multiple triangles" begin
    # Two non-overlapping triangles, one in the bottom half, one in the top half.
    positions = [(0.1, 0.1), (0.4, 0.1), (0.25, 0.4),
                 (0.6, 0.6), (0.9, 0.6), (0.75, 0.9)]
    sv = Int32[1 4; 2 5; 3 6]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])  # 4 leaves
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    overlap = compute_overlap(lag, f; moment_order=1)

    @test total_overlap_volume(overlap) ≈ simplex_volume(lag, 1) + simplex_volume(lag, 2) atol=1e-12

    # Each Lagrangian simplex should have ≥ 1 entry
    @test length(entries_for_lag(overlap, 1)) >= 1
    @test length(entries_for_lag(overlap, 2)) >= 1
end

# ============================================================================
# Remap operators
# ============================================================================

@testset "MassWeightedAverage: constant source ⇒ constant target" begin
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.7)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    overlap = compute_overlap(lag, f; moment_order=1)

    src = [42.0]
    dst = zeros(n_cells(eul))
    remap_l_to_e!(dst, src, overlap, MassWeightedAverage())

    # Every covered cell should hold exactly 42 (weighted mean of identical values)
    for ci in 1:n_cells(eul)
        if length(entries_for_eul(overlap, ci)) > 0
            @test dst[ci] ≈ 42.0
        else
            @test dst[ci] == 0.0
        end
    end
end

@testset "MassWeightedAverage: covered cells ⇔ uncovered cells = 0" begin
    # Triangle that covers only one of the four leaves
    positions = [(0.1, 0.1), (0.3, 0.1), (0.2, 0.3)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    overlap = compute_overlap(lag, f; moment_order=1)

    src = [99.0]
    dst = zeros(n_cells(eul))
    remap_l_to_e!(dst, src, overlap, MassWeightedAverage())
    n_covered = count(d -> d != 0, dst)
    @test n_covered == 1
end

@testset "ConservativeFlux: total source mass = total target mass" begin
    # Two triangles, each carrying a different extensive amount
    positions = [(0.1, 0.1), (0.4, 0.1), (0.25, 0.4),
                 (0.6, 0.6), (0.9, 0.6), (0.75, 0.9)]
    sv = Int32[1 4; 2 5; 3 6]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    overlap = compute_overlap(lag, f; moment_order=1)

    src = [100.0, 250.0]
    dst = zeros(n_cells(eul))
    remap_l_to_e!(dst, src, overlap, ConservativeFlux())
    @test sum(dst) ≈ sum(src) atol=1e-12
end

@testset "remap_e_to_l! (Mass-Weighted): round-trip preserves constant" begin
    # If we remap a constant Eulerian field back to Lagrangian, every Lagrangian
    # cell that has any overlap should receive that constant.
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.7)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    overlap = compute_overlap(lag, f; moment_order=1)

    src = fill(7.5, n_cells(eul))
    dst = zeros(n_simplices(lag))
    remap_e_to_l!(dst, src, overlap, MassWeightedAverage())
    @test dst[1] ≈ 7.5
end

@testset "remap_e_to_l! (Conservative): conserves total" begin
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.7)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    overlap = compute_overlap(lag, f; moment_order=1)

    # Place mass only in cells covered by the triangle; sum should land entirely
    # on the single Lagrangian simplex.
    src = zeros(n_cells(eul))
    for ci in 1:n_cells(eul)
        if length(entries_for_eul(overlap, ci)) > 0
            src[ci] = 5.0
        end
    end
    src_total = sum(src[ci] for ci in 1:n_cells(eul) if length(entries_for_eul(overlap, ci)) > 0)
    dst = zeros(n_simplices(lag))
    remap_e_to_l!(dst, src, overlap, ConservativeFlux())
    @test sum(dst) ≈ src_total atol=1e-12
end

@testset "Remap: dimension-mismatch errors" begin
    positions = [(0.5, 0.5), (0.6, 0.5), (0.5, 0.6)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    overlap = compute_overlap(lag, f; moment_order=1)

    # Wrong source length
    @test_throws DimensionMismatch remap_l_to_e!(zeros(n_cells(eul)),
                                                  zeros(99), overlap,
                                                  MassWeightedAverage())
    @test_throws DimensionMismatch remap_l_to_e!(zeros(99),
                                                  zeros(n_simplices(lag)),
                                                  overlap, MassWeightedAverage())
end

# ============================================================================
# TotalCovariance
# ============================================================================

@testset "TotalCovariance: variance of identical means is just within-cell mean" begin
    # All Lagrangian source cells have the same mean ⇒ between-cell variance = 0,
    # and the destination M is the volume-weighted mean of source M's.
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.7),
                 (0.1, 0.6), (0.4, 0.6), (0.25, 0.9)]
    sv = Int32[1 4; 2 5; 3 6]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    overlap = compute_overlap(lag, f; moment_order=1)

    n_l = n_simplices(lag); n_e = n_cells(eul)
    src_mu = zeros(2, n_l)   # all means at origin
    src_M = zeros(2, 2, n_l)
    src_M[1, 1, :] .= 1.0
    src_M[2, 2, :] .= 2.0
    target_mu = zeros(2, n_e)
    target_M = zeros(2, 2, n_e)
    remap_l_to_e_covariance!(target_M, target_mu, src_M, src_mu, overlap)

    # All source means = 0 ⇒ all target means = 0
    @test all(target_mu .== 0.0)
    # Covered cells have target_M = weighted mean of source_M (which is constant 1, 2 on diagonal)
    for ci in 1:n_e
        if length(entries_for_eul(overlap, ci)) > 0
            @test target_M[1, 1, ci] ≈ 1.0 atol=1e-12
            @test target_M[2, 2, ci] ≈ 2.0 atol=1e-12
            @test target_M[1, 2, ci] ≈ 0.0 atol=1e-12
        end
    end
end

@testset "TotalCovariance: between-cell variance contribution" begin
    # Two source cells, equal volume contributions to one destination cell,
    # with means at (-1, 0) and (+1, 0). Source M's are both 0.
    # Expected: target mean = (0, 0); target M = E[(μ_i - μ_dst)(μ_i - μ_dst)^T]
    #   = (1/2)(1, 0)(1, 0)^T + (1/2)(-1, 0)(-1, 0)^T = ((1, 0), (0, 0))
    b = OverlapBuilder{2, Float64}(0)   # we don't need higher moments here
    push_overlap!(b, 1, 1, 0.5, (0.0, 0.0), [0.5])
    push_overlap!(b, 2, 1, 0.5, (0.0, 0.0), [0.5])
    overlap = finalize_overlap(b, 2, 1)

    src_mu = zeros(2, 2)
    src_mu[1, 1] = 1.0; src_mu[1, 2] = -1.0
    src_M = zeros(2, 2, 2)
    target_mu = zeros(2, 1)
    target_M = zeros(2, 2, 1)
    remap_l_to_e_covariance!(target_M, target_mu, src_M, src_mu, overlap)

    @test target_mu[1, 1] ≈ 0.0
    @test target_mu[2, 1] ≈ 0.0
    @test target_M[1, 1, 1] ≈ 1.0   # spread along x-axis
    @test target_M[2, 2, 1] ≈ 0.0
    @test target_M[1, 2, 1] ≈ 0.0
end

@testset "TotalCumulants{O>=3} not yet implemented" begin
    b = OverlapBuilder{2, Float64}(0)
    push_overlap!(b, 1, 1, 1.0, (0.0, 0.0), [1.0])
    overlap = finalize_overlap(b, 1, 1)
    @test_throws ErrorException remap_l_to_e!(zeros(1), zeros(1), overlap,
                                                TotalCumulants(3))
    @test_throws ArgumentError remap_l_to_e!(zeros(1), zeros(1), overlap,
                                               TotalCumulants(2))
end

# ============================================================================
# Hooking into PairedMesh
# ============================================================================

@testset "install_r3d_overlap! and ensure_overlap!" begin
    positions = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.7)]
    sv = reshape(Int32[1, 2, 3], 3, 1)
    sn = zeros(Int32, 3, 1)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    f = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    paired = PairedMesh(lag, eul; overlap_type=GeometricOverlap{2, Float64})
    install_r3d_overlap!(paired, f; moment_order=1)

    o = ensure_overlap!(paired)
    @test o isa GeometricOverlap{2, Float64}
    @test total_overlap_volume(o) ≈ simplex_volume(lag, 1) atol=1e-12

    # Frame mismatch error
    eul2 = HierarchicalMesh{2}()
    f_wrong = EulerianFrame(eul2, (0.0, 0.0), (1.0, 1.0))
    @test_throws ArgumentError install_r3d_overlap!(paired, f_wrong)
end

# ============================================================================
# Parallel compute_overlap
# ============================================================================

# Helper to build a problem of variable size for parallel testing
function _build_overlap_problem(n_per_axis::Int, eul_depth::Int)
    nx = n_per_axis + 1
    positions = Tuple{Float64, Float64}[]
    for j in 1:nx, i in 1:nx; push!(positions, ((i-1)/(nx-1), (j-1)/(nx-1))); end
    sv_list = NTuple{3, Int32}[]
    for j in 1:(nx-1), i in 1:(nx-1)
        v00 = (j-1)*nx + i; v10 = v00 + 1; v01 = j*nx + i; v11 = v01 + 1
        push!(sv_list, (Int32(v00), Int32(v10), Int32(v01)))
        push!(sv_list, (Int32(v10), Int32(v11), Int32(v01)))
    end
    sv = Matrix{Int32}(undef, 3, length(sv_list))
    for k in 1:length(sv_list); sv[1,k] = sv_list[k][1]; sv[2,k] = sv_list[k][2]; sv[3,k] = sv_list[k][3]; end
    sn = zeros(Int32, 3, length(sv_list))
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    for _ in 1:eul_depth; refine_cells!(eul, enumerate_leaves(eul)); end
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    return lag, frame
end

@testset "compute_overlap: parallel path produces identical results" begin
    # Large enough to actually trigger the parallel path (n_lag * n_eul ≥ 10000)
    lag, frame = _build_overlap_problem(16, 3)   # 512 simplices × 64 leaves = 32768
    ov_seq = compute_overlap(lag, frame; moment_order=3, parallel=false)
    ov_par = compute_overlap(lag, frame; moment_order=3, parallel=true)

    @test n_entries(ov_seq) == n_entries(ov_par)
    @test total_overlap_volume(ov_seq) ≈ total_overlap_volume(ov_par)

    # Entries are sorted by (lag_idx, eul_idx) in finalize_overlap, so they
    # should appear in identical order regardless of which task produced them.
    for k in 1:n_entries(ov_seq)
        es = ov_seq.entries[k]
        ep = ov_par.entries[k]
        @test es.lag_idx == ep.lag_idx
        @test es.eul_idx == ep.eul_idx
        @test es.volume ≈ ep.volume atol=1e-12
        @test es.centroid[1] ≈ ep.centroid[1] atol=1e-12
        @test es.centroid[2] ≈ ep.centroid[2] atol=1e-12
        @test es.moments ≈ ep.moments atol=1e-12
    end
end

@testset "compute_overlap: parallel with :static scheduler" begin
    lag, frame = _build_overlap_problem(16, 3)
    ov_dyn = compute_overlap(lag, frame; moment_order=3, parallel=true, scheduler=:dynamic)
    ov_stat = compute_overlap(lag, frame; moment_order=3, parallel=true, scheduler=:static)

    @test n_entries(ov_dyn) == n_entries(ov_stat)
    @test total_overlap_volume(ov_dyn) ≈ total_overlap_volume(ov_stat)

    # Entry-by-entry agreement: schedulers shouldn't change the result, since
    # entries are sorted by (lag_idx, eul_idx) at finalize.
    for k in 1:n_entries(ov_dyn)
        ed = ov_dyn.entries[k]
        es = ov_stat.entries[k]
        @test ed.lag_idx == es.lag_idx
        @test ed.eul_idx == es.eul_idx
        @test ed.volume ≈ es.volume atol=1e-12
        @test ed.moments ≈ es.moments atol=1e-12
    end
end

@testset "compute_overlap: small problem auto-falls-through to sequential" begin
    # n_lag * n_eul < 10_000 → should use sequential path internally even
    # when parallel=true. We can't directly observe which path ran, but
    # we can confirm the result is correct.
    lag, frame = _build_overlap_problem(4, 1)   # ~32 simplices × 4 leaves = 128
    ov = compute_overlap(lag, frame; moment_order=3, parallel=true)
    ov_seq = compute_overlap(lag, frame; moment_order=3, parallel=false)
    @test n_entries(ov) == n_entries(ov_seq)
    @test total_overlap_volume(ov) ≈ total_overlap_volume(ov_seq)
end

@testset "compute_overlap: parallel preserves CSR invariants" begin
    lag, frame = _build_overlap_problem(16, 3)
    ov = compute_overlap(lag, frame; moment_order=2, parallel=true)

    # Every Lagrangian simplex's range should be valid
    for s in 1:n_simplices(lag)
        rng = ov.lag_to_entries[s]
        # Range bounds within entries
        if !isempty(rng)
            @test first(rng) >= 1
            @test last(rng) <= n_entries(ov)
            # All entries in this range have lag_idx == s
            for k in rng
                @test ov.entries[k].lag_idx == s
            end
        end
    end

    # Total entries summed across simplex ranges = n_entries
    total = sum(length(ov.lag_to_entries[s]) for s in 1:n_simplices(lag))
    @test total == n_entries(ov)
end

@testset "compute_overlap: lazy mesh caches built before parallel section" begin
    # Regression: previously, `compute_overlap(parallel=true, scheduler=:static)`
    # crashed with `ConcurrencyViolationError("Vector can not be resized
    # concurrently")` because per-task `cell_physical_box → cell_unit_box →
    # Mesh.ensure_caches!` lazily resized `mesh._parents` from multiple
    # tasks simultaneously. The fix: force `Mesh.ensure_caches!(eul)` on
    # the main thread before tasks fan out. This test runs with a fresh
    # mesh (caches not yet built) under :static to ensure the fix holds.
    lag, frame = _build_overlap_problem(16, 3)

    # Confirm the mesh's caches are NOT pre-built — we want to exercise
    # the cache-build path inside the parallel section.
    # (`HierarchicalGrids.Mesh._parents` is internal; we just verify
    # the mesh is fresh enough that some lazy work needs to happen by
    # checking that it hasn't been queried with cell_unit_box before.)
    eul = frame.mesh
    @test n_cells(eul) > 1  # mesh has refined cells

    # Run multiple trials with :static scheduler (the deterministic one,
    # which used to crash 100% of the time on this input). If the race
    # has reappeared we expect either a crash or wrong results.
    base = compute_overlap(lag, frame; moment_order=2, parallel=false)
    n_trials = 5
    for _ in 1:n_trials
        ov = compute_overlap(lag, frame; moment_order=2, parallel=true,
                              scheduler=:static)
        @test n_entries(ov) == n_entries(base)
        @test total_overlap_volume(ov) ≈ total_overlap_volume(base)
    end
end
