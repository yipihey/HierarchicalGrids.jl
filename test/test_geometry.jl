using Test
using HierarchicalGrids
using HierarchicalGrids.Mesh
using HierarchicalGrids.Geometry

@testset "Cell extent" begin
    mesh = HierarchicalMesh{3}()
    # Root cell: extent 1 (denominator) along each axis
    extent = cell_extent(mesh, 1)
    @test extent == (1, 1, 1)

    # After refinement, children have extent 2 (denominator) along each axis
    refine_cells!(mesh, [1])
    children = find_children(mesh, 1)
    for ci in children
        e = cell_extent(mesh, ci)
        @test e == (2, 2, 2)
    end
end

@testset "Anisotropic cell extent" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1], [0b001])  # split only along x
    children = find_children(mesh, 1)
    for ci in children
        e = cell_extent(mesh, ci)
        @test e == (2, 1, 1)  # x has denominator 2, y and z have 1
    end
end

@testset "Cell volume — exact integers" begin
    mesh = HierarchicalMesh{3}()
    # Root volume = 1/1 = 1
    num, den = cell_volume(mesh, 1)
    @test num // den == 1 // 1

    # After isotropic refinement, each child has volume 1/8
    refine_cells!(mesh, [1])
    for ci in find_children(mesh, 1)
        num, den = cell_volume(mesh, ci)
        @test num // den == 1 // 8
    end

    # Sum of children volumes equals parent volume — exact!
    total_num = 0
    common_den = 0
    for ci in find_children(mesh, 1)
        num, den = cell_volume(mesh, ci)
        if common_den == 0
            common_den = den
        else
            @test den == common_den  # all children at same level
        end
        total_num += num
    end
    @test total_num // common_den == 1 // 1
end

@testset "Cell volume conservation under refinement" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])  # 8 children
    refine_cells!(mesh, [2])  # refine first child
    refine_cells!(mesh, [3])  # refine first grandchild

    # Sum of all leaf volumes should be exactly 1
    total = 0 // 1
    for i in 1:n_cells(mesh)
        if is_leaf(mesh.cells[i])
            num, den = cell_volume(mesh, i)
            total += num // den
        end
    end
    @test total == 1 // 1
end

@testset "Anisotropic volume" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1], [0b001])  # split along x: 2 children, each with volume 1/2
    for ci in find_children(mesh, 1)
        num, den = cell_volume(mesh, ci)
        @test num // den == 1 // 2
    end

    # Sum is still 1
    total = 0 // 1
    for ci in find_children(mesh, 1)
        num, den = cell_volume(mesh, ci)
        total += num // den
    end
    @test total == 1 // 1
end

@testset "Cell center in parent" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])

    # Each child should have center at one of (±1, ±1, ±1) in our scheme
    centers = Set{Tuple{Int, Int, Int}}()
    for ci in find_children(mesh, 1)
        c = cell_center_in_parent(mesh.cells[ci], Int32)
        push!(centers, (Int(c[1]), Int(c[2]), Int(c[3])))
    end
    expected = Set([(x, y, z) for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    @test centers == expected
end

@testset "Position relative to ancestor" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2])  # refine first child to get grandchildren

    # Grandchildren should have positions in [0, 4)^3 in root's frame
    # (since they're 2 levels below root, with 4 = 2^2 cells per axis)

    grandchildren = find_children(mesh, 2)
    positions = Set{NTuple{3, Int}}()
    for gi in grandchildren
        pos = HierarchicalGrids.Geometry.position_relative_to_ancestor(mesh, gi, 1)
        push!(positions, pos)
    end

    # All 8 grandchildren of cell 2 should be in the same quarter of root
    # (the quarter corresponding to cell 2's position)
    @test length(positions) == 8
    # cell 2 is at sibling_index 0 -> position (0,0,0) in root, so its grandchildren
    # should be in [0,2)^3 of the root's [0,4)^3 frame.
    for pos in positions
        @test 0 <= pos[1] < 2
        @test 0 <= pos[2] < 2
        @test 0 <= pos[3] < 2
    end
end

@testset "Distance squared between cells" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])

    children = find_children(mesh, 1)
    # Distance between a cell and itself is 0
    @test HierarchicalGrids.Geometry.distance_squared(mesh, children[1], children[1]) == 0

    # Distance between adjacent cells (differ in one coordinate) should be 1
    # Distance between diagonal cells (differ in all coordinates) should be 3
    distances = Set{Int}()
    for i in eachindex(children), j in eachindex(children)
        if i != j
            d = HierarchicalGrids.Geometry.distance_squared(mesh, children[i], children[j])
            push!(distances, Int(d))
        end
    end
    @test 1 in distances  # adjacent
    @test 3 in distances  # diagonal
    # All distances should be 1, 2, or 3 (sums of 0s and 1s squared)
    @test all(d in (1, 2, 3) for d in distances)
end

@testset "Predicates" begin
    @test sign_of_axis(Int32(5)) == 1
    @test sign_of_axis(Int32(0)) == 0
    @test sign_of_axis(Int32(-3)) == -1

    @test in_box((1, 2, 3), (0, 0, 0), (5, 5, 5)) == true
    @test in_box((6, 2, 3), (0, 0, 0), (5, 5, 5)) == false  # x out of range
    @test in_box((-1, 2, 3), (0, 0, 0), (5, 5, 5)) == false  # x below
    @test in_box((0, 0, 0), (0, 0, 0), (5, 5, 5)) == true   # boundary inclusive
    @test in_box((5, 5, 5), (0, 0, 0), (5, 5, 5)) == true   # boundary inclusive
end

@testset "Axis-aligned box volume" begin
    @test axis_aligned_box_volume((10, 20, 30)) == 6000
    @test axis_aligned_box_volume((1,)) == 1
    @test axis_aligned_box_volume((100, 100)) == 10000
    @test axis_aligned_box_volume((Int64(1000), Int64(1000), Int64(1000))) == Int64(10^9)
end

@testset "Interval construction" begin
    # Type-parametric construction
    I = Interval{Float64}(0.0, 1.0)
    @test I.lo == 0.0
    @test I.hi == 1.0
    @test typeof(I) === Interval{Float64}

    # Promotion-based outer constructor
    J = Interval(0, 1.0)
    @test typeof(J) === Interval{Float64}
    @test J.lo === 0.0
    @test J.hi === 1.0

    # Float32
    K = Interval{Float32}(0.0f0, 2.5f0)
    @test typeof(K) === Interval{Float32}
end

@testset "Interval emptiness and length" begin
    I = Interval(0.0, 1.0)
    @test !is_empty(I)
    @test interval_length(I) == 1.0

    # Degenerate (single point) is treated as empty
    P = Interval(0.5, 0.5)
    @test is_empty(P)
    @test interval_length(P) == 0.0

    # Inverted is empty
    R = Interval(1.0, 0.0)
    @test is_empty(R)
    @test interval_length(R) == 0.0

    # NaN is empty
    N = Interval(NaN, NaN)
    @test is_empty(N)
    @test interval_length(N) == 0.0
end

@testset "Interval intersection" begin
    A = Interval(0.0, 2.0)
    B = Interval(1.0, 3.0)
    C = interval_intersection(A, B)
    @test !is_empty(C)
    @test C.lo == 1.0
    @test C.hi == 2.0
    @test interval_length(C) == 1.0

    # Disjoint
    D = Interval(5.0, 6.0)
    E = interval_intersection(A, D)
    @test is_empty(E)
    @test interval_length(E) == 0.0

    # Touching at a point — empty (zero measure)
    F = Interval(2.0, 4.0)
    G = interval_intersection(A, F)
    @test is_empty(G)

    # Self-intersection
    @test interval_intersection(A, A).lo == A.lo
    @test interval_intersection(A, A).hi == A.hi

    # Containment: A ⊂ B
    H = interval_intersection(Interval(0.5, 1.0), A)
    @test H.lo == 0.5 && H.hi == 1.0
end

@testset "Interval intersection promotion" begin
    # Mixed-type intersection should still work via outer constructor promotion
    a = Interval(0, 2.0)        # Float64 after promotion
    b = Interval(1.0f0, 3.0f0)  # Float32
    c = interval_intersection(a, b)
    @test c.lo == 1.0
    @test c.hi == 2.0
end

@testset "Interval contains" begin
    I = Interval(0.0, 1.0)
    @test interval_contains(I, 0.5)
    @test interval_contains(I, 0.0)   # boundary
    @test interval_contains(I, 1.0)   # boundary
    @test !interval_contains(I, -0.1)
    @test !interval_contains(I, 1.1)
    @test !interval_contains(Interval(NaN, NaN), 0.5)
end

@testset "Affine maps to and from reference" begin
    I = Interval(2.0, 6.0)
    # Forward: x ∈ I → t ∈ [0, 1]
    @test affine_map_to_reference(I, 2.0) == 0.0
    @test affine_map_to_reference(I, 6.0) == 1.0
    @test affine_map_to_reference(I, 4.0) == 0.5

    # Inverse: t ∈ [0, 1] → x ∈ I
    @test affine_map_from_reference(I, 0.0) == 2.0
    @test affine_map_from_reference(I, 1.0) == 6.0
    @test affine_map_from_reference(I, 0.5) == 4.0

    # Round-trip
    for x in (2.0, 3.7, 5.999)
        @test affine_map_from_reference(I, affine_map_to_reference(I, x)) ≈ x
    end

    # Empty intervals throw
    E = Interval(1.0, 0.0)  # inverted, hence empty
    @test_throws ArgumentError affine_map_to_reference(E, 0.5)
    @test_throws ArgumentError affine_map_from_reference(E, 0.5)
end
