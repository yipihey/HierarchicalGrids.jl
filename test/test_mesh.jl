using Test
using HierarchicalGrids
using HierarchicalGrids.Mesh
using HierarchicalGrids.BitPrimitives

@testset "CellMeta basics" begin
    # Construction
    c = CellMeta{3}(0, 0, FLAG_LEAF)
    @test sibling_index(c) == 0
    @test split_mask(c) == 0
    @test flags(c) == FLAG_LEAF
    @test is_leaf(c)
    @test !is_boundary(c)
    @test !is_dirty(c)

    # Type inference
    M = sibling_index_type(Val(3))
    @test typeof(c) == CellMeta{3, M}

    # Different dimension
    c2d = CellMeta{2}(0, 3, FLAG_LEAF)
    @test typeof(c2d) == CellMeta{2, UInt8}

    c10d = CellMeta{10}(0, 0, FLAG_LEAF)
    @test typeof(c10d) == CellMeta{10, UInt16}
end

@testset "Mask helpers" begin
    @test FULLY_ISOTROPIC_MASK(Val(1)) == UInt8(0b1)
    @test FULLY_ISOTROPIC_MASK(Val(2)) == UInt8(0b11)
    @test FULLY_ISOTROPIC_MASK(Val(3)) == UInt8(0b111)
    @test FULLY_ISOTROPIC_MASK(Val(8)) == UInt8(0xFF)

    @test isotropic_mask(3) == UInt32(0b111)
    @test isotropic_mask(4) == UInt32(0b1111)

    # children_count for various split patterns
    c = CellMeta{3}(0, 0b111, FLAG_LEAF)  # all axes split
    @test children_count(c) == 8

    c = CellMeta{3}(0, 0b001, FLAG_LEAF)  # only x split
    @test children_count(c) == 2

    c = CellMeta{3}(0, 0b101, FLAG_LEAF)  # x and z split
    @test children_count(c) == 4

    c = CellMeta{3}(0, 0b000, FLAG_LEAF)  # no axes split (root with no siblings)
    @test children_count(c) == 1
end

@testset "Empty/single-cell mesh" begin
    mesh = HierarchicalMesh{3}()
    @test n_cells(mesh) == 1
    @test root_cell_index(mesh) == 1
    @test is_leaf(mesh.cells[1])
    @test mesh.canonical_reference_level == 0

    # Caches should be invalid initially but rebuild on demand
    @test !caches_valid(mesh)
    @test level_of(mesh, 1) == 0  # triggers rebuild
    @test caches_valid(mesh)
    @test find_parent(mesh, 1) == HierarchicalGrids.Mesh.ROOT_PARENT
end

@testset "Refinement basics" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    # After refining root: 1 (root, now non-leaf) + 8 children = 9
    @test n_cells(mesh) == 9
    @test !is_leaf(mesh.cells[1])
    for i in 2:9
        @test is_leaf(mesh.cells[i])
    end

    # Check children have correct sibling indices
    sibs = [Int(mesh.cells[i].sibling_index) for i in 2:9]
    @test sort(sibs) == collect(0:7)

    # Children all have the same split_mask (the parent's chosen pattern)
    for i in 2:9
        @test mesh.cells[i].split_mask == FULLY_ISOTROPIC_MASK(Val(3))
    end
end

@testset "Refinement with custom split_mask (anisotropic)" begin
    mesh = HierarchicalMesh{3}()
    # Refine only along x axis
    refine_cells!(mesh, [1], [0b001])
    # 1 root + 2 children
    @test n_cells(mesh) == 3
    @test !is_leaf(mesh.cells[1])
    @test mesh.cells[1].split_mask == 0b001
    @test is_leaf(mesh.cells[2])
    @test is_leaf(mesh.cells[3])
    @test mesh.cells[2].sibling_index == 0
    @test mesh.cells[3].sibling_index == 1
    @test mesh.cells[2].split_mask == 0b001
end

@testset "Multi-level refinement and parent walks" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])  # 9 cells now
    # Refine cell 2 (first child of root)
    refine_cells!(mesh, [2])
    # 1 root + 8 children, but the first child is now non-leaf with 8 children
    # So total: 1 + 1 (refined child) + 8 (its children) + 7 (other children) = 17
    @test n_cells(mesh) == 17

    # Walk to find parent of a deep cell
    @test find_parent(mesh, 1) == HierarchicalGrids.Mesh.ROOT_PARENT
    @test find_parent(mesh, 2) == 1  # cell 2 is child of root
    @test find_parent(mesh, 3) == 2  # cell 3 is first child of cell 2
    @test find_parent(mesh, 10) == 2  # cell 10 is last child of cell 2

    # Levels
    @test level_of(mesh, 1) == 0
    @test level_of(mesh, 2) == 1
    @test level_of(mesh, 3) == 2
    @test level_of(mesh, 10) == 2  # all children of cell 2 at level 2
    @test level_of(mesh, 11) == 1  # this is the second child of root
end

@testset "Find children" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])

    children_of_root = find_children(mesh, 1)
    @test length(children_of_root) == 8
    # In our DFS layout with root at index 1 and all children leaves, children are at 2..9
    @test sort(collect(children_of_root)) == collect(UInt32, 2:9)

    # Leaf cells have no children
    for c in 2:9
        @test isempty(find_children(mesh, c))
    end
end

@testset "LCA — Lowest Common Ancestor" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    # Refine first child, then refine the next-original-child
    # (after refinement of cell 2, indices shift; what was originally cell 3
    # becomes cell 11 because 8 children were inserted after cell 2)
    refine_cells!(mesh, [2])
    # Now find what was the second child of root (originally cell 3, now shifted)
    second_root_child = find_children(mesh, 1)[2]
    refine_cells!(mesh, [Int(second_root_child)])

    # Find updated children references
    second_root_child = find_children(mesh, 1)[2]  # re-find after refinement
    children_of_first = find_children(mesh, 2)
    children_of_second = find_children(mesh, Int(second_root_child))

    # LCA of two siblings
    @test find_lca(mesh, children_of_first[1], children_of_first[2]) == 2

    # LCA of two cells with different parents
    @test find_lca(mesh, children_of_first[1], children_of_second[1]) == 1

    # LCA of cell with itself
    @test find_lca(mesh, 5, 5) == 5

    # LCA of a cell and its parent
    @test find_lca(mesh, children_of_first[1], 2) == 2

    # LCA of any cell and root
    @test find_lca(mesh, children_of_first[1], 1) == 1
end

@testset "Position in parent" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])

    # Each child has a different position
    positions = Set{NTuple{3, Int}}()
    for child_idx in find_children(mesh, 1)
        pos = position_in_parent(mesh.cells[child_idx])
        push!(positions, pos)
    end
    # All 8 corners of a 3D cube should be represented
    @test length(positions) == 8
    expected = Set([(x, y, z) for x in 0:1 for y in 0:1 for z in 0:1])
    @test positions == expected
end

@testset "Anisotropic position in parent" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1], [0b101])  # split along axes 0 and 2

    children = find_children(mesh, 1)
    @test length(children) == 4

    # The non-split axis should have position 0 for all children
    for child_idx in children
        pos = position_in_parent(mesh.cells[child_idx])
        @test pos[2] == 0  # axis 1 (the y axis) is not split
        # Axes 0 and 2 should give all 4 combinations
    end

    # Collect all positions and check they cover the 4 corners of the xz plane
    positions = Set{Tuple{Int, Int, Int}}()
    for child_idx in children
        push!(positions, position_in_parent(mesh.cells[child_idx]))
    end
    expected = Set([(0,0,0), (1,0,0), (0,0,1), (1,0,1)])
    @test positions == expected
end

@testset "Cache invalidation" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    # Trigger cache rebuild
    level_of(mesh, 1)
    @test caches_valid(mesh)
    # Refine again — should invalidate caches
    refine_cells!(mesh, [2])
    @test !caches_valid(mesh)
    # Access triggers rebuild
    level_of(mesh, 1)
    @test caches_valid(mesh)
end

@testset "Coarsening" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    @test n_cells(mesh) == 9

    coarsen_cells!(mesh, [1])
    @test n_cells(mesh) == 1
    @test is_leaf(mesh.cells[1])
end

@testset "Cell paths" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2])

    # Path of root: empty
    path = cell_path(mesh, 1)
    @test isempty(path.sibling_indices)
    @test find_at_path(mesh, path) == 1

    # Path of a child of root
    children = find_children(mesh, 1)
    for ci in children
        path = cell_path(mesh, ci)
        @test length(path.sibling_indices) == 1
        @test find_at_path(mesh, path) == ci
    end

    # Path of a grandchild
    grandchildren = find_children(mesh, 2)
    for gi in grandchildren
        path = cell_path(mesh, gi)
        @test length(path.sibling_indices) == 2
        @test find_at_path(mesh, path) == gi
    end
end

@testset "Higher-dimensional mesh" begin
    # 4D
    mesh4 = HierarchicalMesh{4}()
    refine_cells!(mesh4, [1])
    @test n_cells(mesh4) == 17  # 1 + 16 children

    # 1D
    mesh1 = HierarchicalMesh{1}()
    refine_cells!(mesh1, [1])
    @test n_cells(mesh1) == 3  # 1 + 2 children

    # 2D
    mesh2 = HierarchicalMesh{2}()
    refine_cells!(mesh2, [1])
    @test n_cells(mesh2) == 5  # 1 + 4 children
end

@testset "DFS order invariant" begin
    # After refinement, the cells should still be in DFS order:
    # for any non-leaf cell at index i with subtree size s,
    # cells i+1..i+s-1 should all be descendants.
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 5])

    # Trigger cache rebuild to populate subtree_sizes
    level_of(mesh, 1)

    for i in 1:n_cells(mesh)
        sub_size = mesh._subtree_sizes[i]
        # All cells in [i, i+sub_size-1] should have level >= level_of(i)
        for j in i:(i + Int(sub_size) - 1)
            @test level_of(mesh, j) >= level_of(mesh, i)
        end
    end
end

@testset "refine_cells! / coarsen_cells! argument validation" begin
    # Refining a non-leaf cell should throw ArgumentError (was @assert,
    # which can be stripped at high optimization).
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])             # cell 1 is now a non-leaf
    @test_throws ArgumentError refine_cells!(mesh, [1])

    # Empty split mask is rejected.
    mesh2 = HierarchicalMesh{3}()
    @test_throws ArgumentError refine_cells!(mesh2, [1], UInt8[0])

    # Coarsening a leaf cell should throw.
    mesh3 = HierarchicalMesh{3}()
    @test_throws ArgumentError coarsen_cells!(mesh3, [1])    # root is a leaf

    # Coarsening when grandchildren exist (non-leaf children) should throw.
    mesh4 = HierarchicalMesh{3}()
    refine_cells!(mesh4, [1])            # cell 1 has 8 children
    refine_cells!(mesh4, [2])            # cell 2 (child of 1) has its own children
    # Now coarsening cell 1 should fail because child 2 isn't a leaf.
    @test_throws ArgumentError coarsen_cells!(mesh4, [1])
end
