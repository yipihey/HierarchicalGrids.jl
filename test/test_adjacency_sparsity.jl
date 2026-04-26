using Test
using HierarchicalGrids
using HierarchicalGrids: cell_adjacency_sparsity, face_neighbors, face_fine_neighbors
using SparseArrays: nnz

# Brute-force enumeration of the leaf adjacency (consistent with
# face_neighbors) for cross-checking.
function brute_force_leaf_adjacency(mesh::HierarchicalMesh{D}) where {D}
    leaves = UInt32[i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])]
    pos = Dict(c => k for (k, c) in enumerate(leaves))
    N = length(leaves)
    A = falses(N, N)
    for i in leaves
        nb_tup = face_neighbors(mesh, i)
        for f in 1:(2*D)
            for nb in face_fine_neighbors(mesh, i, f)
                ki = pos[i]; kj = pos[nb]
                A[ki, kj] = true; A[kj, ki] = true
            end
        end
        A[pos[i], pos[i]] = true
    end
    return A
end

@testset "2x2 leaf mesh, depth=1 matches brute force" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    A_sp = cell_adjacency_sparsity(mesh; depth=1, leaves_only=true)

    A_ref = brute_force_leaf_adjacency(mesh)
    @test size(A_sp) == size(A_ref)
    @test Matrix(A_sp) == A_ref

    # Symmetric
    @test Matrix(A_sp) == Matrix(A_sp)'
end

@testset "Depth=2 includes face-of-face neighbors on a 4x4 grid" begin
    # Build a 4x4 leaf grid by refining the root, then refining each of its
    # children. Result: 16 leaves arranged in a 4x4 grid.
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    HierarchicalGrids.rebuild_caches!(mesh)
    lv1_leaves = [i for i in 1:n_cells(mesh)
                  if HierarchicalGrids.is_leaf(mesh[i]) && mesh._levels[i] == 1]
    refine_cells!(mesh, lv1_leaves)

    leaves = [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])]
    @test length(leaves) == 16

    A1 = Matrix(cell_adjacency_sparsity(mesh; depth=1, leaves_only=true))
    A2 = Matrix(cell_adjacency_sparsity(mesh; depth=2, leaves_only=true))

    # Symmetric, diagonal true.
    @test A1 == A1'
    @test A2 == A2'
    for k in 1:size(A1,1)
        @test A1[k, k] && A2[k, k]
    end

    # Depth 2 must be a superset of depth 1.
    @test all(A2[A1])

    # Strictly larger: at least one cell pair becomes adjacent at depth 2
    # but not at depth 1.
    @test sum(A2) > sum(A1)

    # In a uniform 4x4 grid, the depth-1 graph has 16 self + 24 boundary
    # edges (4 horizontal in 4 rows + 4 vertical in 4 cols)*2 = 48 off-
    # diagonal entries. So nnz(A1) should be 16 + 48 = 64.
    @test sum(A1) == 64
end

@testset "leaves_only=false includes interior nodes as isolated diagonal" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    A_all = cell_adjacency_sparsity(mesh; depth=1, leaves_only=false)
    @test size(A_all, 1) == n_cells(mesh)
    # The root (cell 1) is non-leaf and hence has no representative
    # neighbors; it should appear only on the diagonal.
    @test A_all[1, 1] == true
    for j in 2:size(A_all, 2)
        @test A_all[1, j] == false
    end
end
