using Test
using Random
using HierarchicalGrids
using HierarchicalGrids: face_neighbors, face_fine_neighbors,
    build_neighbor_graph, ensure_neighbor_graph!

# ============================================================================
# Helpers
# ============================================================================

# Compare two NeighborGraphs for byte-equality. `representatives` is a
# Vector{NTuple{...}} so `==` works directly. `fine` is a Dict — values
# are sorted Vector{UInt32}, so an SET comparison is safe (Dict
# insertion order does not matter for the contract).
function assert_graphs_equal(g1, g2; ctx::String = "")
    @test g1.representatives == g2.representatives
    # Same key set
    keys1 = sort!(collect(keys(g1.fine)))
    keys2 = sort!(collect(keys(g2.fine)))
    @test keys1 == keys2
    for k in keys1
        @test g1.fine[k] == g2.fine[k]
    end
    return nothing
end

const _ALL_BACKENDS = (
    Sequential(),
    OhMyThreadsBackend(:dynamic),
    OhMyThreadsBackend(:static),
    OhMyThreadsBackend(:greedy),
    OhMyThreadsBackend(:serial),
)

# ============================================================================
# Determinism: representatives + fine match across all backends
# ============================================================================

@testset "build_neighbor_graph: backend determinism (uniform 2D)" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])           # 4 leaves
    refine_cells!(mesh, [2, 3, 4, 5])  # 16 leaves total
    g_seq = build_neighbor_graph(mesh; backend = Sequential())
    for b in _ALL_BACKENDS
        g = build_neighbor_graph(mesh; backend = b)
        assert_graphs_equal(g_seq, g; ctx = "uniform 2D / $(b)")
    end
end

@testset "build_neighbor_graph: backend determinism (uniform 3D)" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3, 4, 5])
    g_seq = build_neighbor_graph(mesh; backend = Sequential())
    for b in _ALL_BACKENDS
        g = build_neighbor_graph(mesh; backend = b)
        assert_graphs_equal(g_seq, g; ctx = "uniform 3D / $(b)")
    end
end

# ============================================================================
# Hanging-node coverage (unbalanced; the `fine` Dict gets populated)
# ============================================================================

@testset "build_neighbor_graph: hanging nodes (unbalanced 2D)" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])           # cells 2..5 are leaves
    refine_cells!(mesh, [2])           # refine ONE leaf so its siblings are coarse
    refine_cells!(mesh, [6])           # deepen further on one side
    g_seq = build_neighbor_graph(mesh; backend = Sequential())
    @test !isempty(g_seq.fine)         # this case must have hanging-node entries
    for b in _ALL_BACKENDS
        g = build_neighbor_graph(mesh; backend = b)
        assert_graphs_equal(g_seq, g; ctx = "unbalanced 2D / $(b)")
    end
end

@testset "build_neighbor_graph: hanging nodes (unbalanced 3D)" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2])
    refine_cells!(mesh, [10])
    g_seq = build_neighbor_graph(mesh; backend = Sequential())
    for b in _ALL_BACKENDS
        g = build_neighbor_graph(mesh; backend = b)
        assert_graphs_equal(g_seq, g; ctx = "unbalanced 3D / $(b)")
    end
end

# ============================================================================
# face_neighbors / face_fine_neighbors round-trip
# (the cached graph is built lazily through ensure_neighbor_graph! — verify
# that the parallel build is invoked transparently)
# ============================================================================

@testset "face_neighbors: parallel build via ensure_neighbor_graph!" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2])
    refine_cells!(mesh, [6])
    # Reference: explicit Sequential build
    g_seq = build_neighbor_graph(mesh; backend = Sequential())

    # Force a parallel build through the cached path by setting the
    # default backend, then triggering face_neighbors.
    original = default_backend()
    try
        for b in _ALL_BACKENDS
            set_default_backend!(b)
            # Invalidate cached graph (refine triggers the listener; cheap
            # path: directly clear the cached field via the ensure_caches
            # rebuild).
            mesh._cached_neighbor_graph = nothing
            g = ensure_neighbor_graph!(mesh)
            assert_graphs_equal(g_seq, g; ctx = "default-backend $(b)")
        end
    finally
        set_default_backend!(original)
    end
end

# ============================================================================
# Randomized fuzz: multiple random refinement sequences
# ============================================================================

# Build a random unbalanced mesh by repeatedly refining a randomly-picked
# leaf. The sequence is reproducible via the supplied seed.
function _build_random_mesh(D::Int, seed::Int, n_refines::Int)
    rng = MersenneTwister(seed)
    mesh = HierarchicalMesh{D}()
    for _ in 1:n_refines
        leaves = [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh.cells[i])]
        isempty(leaves) && break
        refine_cells!(mesh, [leaves[rand(rng, 1:length(leaves))]])
    end
    return mesh
end

@testset "build_neighbor_graph: random-mesh fuzz (2D)" begin
    for seed in 1:5
        mesh = _build_random_mesh(2, seed, 6)
        g_seq = build_neighbor_graph(mesh; backend = Sequential())
        for b in (OhMyThreadsBackend(:dynamic), OhMyThreadsBackend(:static))
            g = build_neighbor_graph(mesh; backend = b)
            assert_graphs_equal(g_seq, g; ctx = "fuzz 2D seed=$(seed) / $(b)")
        end
    end
end

@testset "build_neighbor_graph: random-mesh fuzz (3D)" begin
    for seed in 1:5
        mesh = _build_random_mesh(3, seed, 4)
        g_seq = build_neighbor_graph(mesh; backend = Sequential())
        for b in (OhMyThreadsBackend(:dynamic), OhMyThreadsBackend(:static))
            g = build_neighbor_graph(mesh; backend = b)
            assert_graphs_equal(g_seq, g; ctx = "fuzz 3D seed=$(seed) / $(b)")
        end
    end
end

# ============================================================================
# Empty-mesh / single-cell edge cases
# ============================================================================

@testset "build_neighbor_graph: single-cell mesh under all backends" begin
    mesh = HierarchicalMesh{2}()
    g_seq = build_neighbor_graph(mesh; backend = Sequential())
    @test g_seq.representatives == [(UInt32(0), UInt32(0), UInt32(0), UInt32(0))]
    @test isempty(g_seq.fine)
    for b in _ALL_BACKENDS
        g = build_neighbor_graph(mesh; backend = b)
        assert_graphs_equal(g_seq, g; ctx = "single-cell / $(b)")
    end
end
