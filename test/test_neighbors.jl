using Test
using HierarchicalGrids
using HierarchicalGrids: face_neighbors, face_fine_neighbors,
    build_neighbor_graph, ensure_neighbor_graph!,
    face_neighbors_with_bcs, FrameBoundaries, PERIODIC, OUTFLOW

# ---------------------------------------------------------------------------
# Brute-force reference: for each pair of leaves, decide adjacency by AABB
# ---------------------------------------------------------------------------

# We compute unit-cube boxes via the public Overlap API (cell_unit_box) to
# avoid relying on the private Mesh helper that the implementation uses.
function brute_force_face_neighbors(mesh::HierarchicalMesh{D}) where {D}
    leaves = [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])]
    boxes = Dict{Int, Tuple{NTuple{D, Float64}, NTuple{D, Float64}}}()
    for i in leaves
        boxes[i] = HierarchicalGrids.cell_unit_box(mesh, i)
    end

    eps = 1e-12
    out = Dict{Int, Dict{Int, Vector{Int}}}()  # cell -> face_idx -> neighbors
    for i in leaves
        face_map = Dict{Int, Vector{Int}}()
        (a_lo, a_hi) = boxes[i]
        for d in 1:D
            for j in leaves
                j == i && continue
                (b_lo, b_hi) = boxes[j]
                # axis d: lo face touches if a_lo[d] == b_hi[d]
                if abs(a_lo[d] - b_hi[d]) <= eps
                    # check overlap on other axes
                    ok = true
                    for d2 in 1:D
                        d2 == d && continue
                        if !(a_hi[d2] > b_lo[d2] + eps && b_hi[d2] > a_lo[d2] + eps)
                            ok = false; break
                        end
                    end
                    if ok
                        push!(get!(face_map, 2*d - 1, Int[]), j)
                    end
                end
                if abs(a_hi[d] - b_lo[d]) <= eps
                    ok = true
                    for d2 in 1:D
                        d2 == d && continue
                        if !(a_hi[d2] > b_lo[d2] + eps && b_hi[d2] > a_lo[d2] + eps)
                            ok = false; break
                        end
                    end
                    if ok
                        push!(get!(face_map, 2*d, Int[]), j)
                    end
                end
            end
        end
        out[i] = face_map
    end
    return out
end

@testset "Uniform 2D mesh: 4 children, interior face contract" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    @test n_cells(mesh) == 5

    # Cells 2..5 are leaves with sibling indices 0..3
    leaves = [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])]
    @test length(leaves) == 4

    bf = brute_force_face_neighbors(mesh)
    for i in leaves
        nb = face_neighbors(mesh, i)
        # 4 faces in 2D
        @test length(nb) == 4
        for f in 1:4
            ref = sort(get(bf[i], f, Int[]))
            if isempty(ref)
                @test nb[f] == 0
            else
                @test nb[f] in UInt32.(ref)
                # face_fine_neighbors should equal the sorted reference set
                ffn = sort(Int.(face_fine_neighbors(mesh, i, f)))
                @test ffn == ref
            end
        end
    end

    # Each leaf has exactly 2 boundary faces and 2 interior faces.
    for i in leaves
        nb = face_neighbors(mesh, i)
        @test count(==(UInt32(0)), nb) == 2
    end
end

@testset "Multi-level 2D refinement (unbalanced) coarse-fine contract" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])           # cells 2..5 are leaves
    # Refine cell 2's subtree by refining cell 2 (turn it into 4 children).
    refine_cells!(mesh, [2])           # 9 cells total: cell 2 is now non-leaf,
                                       # its children (cells 3..6) are new leaves;
                                       # cells 7..9 are old siblings 1,2,3 of cell 2
                                       # (DFS reordering).

    # Find a leaf that is a sibling of the original cell 2 — by looking at the
    # current mesh, find leaves at level 1.
    HierarchicalGrids.rebuild_caches!(mesh)
    levels = mesh._levels
    leaves_lv1 = [i for i in 1:n_cells(mesh)
                  if HierarchicalGrids.is_leaf(mesh[i]) && levels[i] == 1]
    leaves_lv2 = [i for i in 1:n_cells(mesh)
                  if HierarchicalGrids.is_leaf(mesh[i]) && levels[i] == 2]
    # 3 lv-1 leaves (the un-refined siblings) + 4 lv-2 leaves (refined children)
    @test length(leaves_lv1) == 3
    @test length(leaves_lv2) == 4

    bf = brute_force_face_neighbors(mesh)

    # Pick a coarse leaf that has at least one fine neighbor and verify the
    # coarse-fine contract.
    found_hanging = false
    for i in leaves_lv1
        nb = face_neighbors(mesh, i)
        for f in 1:4
            ref_set = sort(get(bf[i], f, Int[]))
            isempty(ref_set) && continue
            if length(ref_set) >= 2
                found_hanging = true
                # representative is one of them
                @test Int(nb[f]) in ref_set
                # face_fine_neighbors enumerates all
                ffn = sort(Int.(face_fine_neighbors(mesh, i, f)))
                @test ffn == ref_set
            end
        end
    end
    @test found_hanging

    # Cross-check every leaf cell against brute force
    for i in [leaves_lv1; leaves_lv2]
        nb = face_neighbors(mesh, i)
        for f in 1:4
            ref = sort(get(bf[i], f, Int[]))
            ffn = sort(Int.(face_fine_neighbors(mesh, i, f)))
            @test ffn == ref
            if isempty(ref)
                @test nb[f] == 0
            else
                @test Int(nb[f]) in ref
            end
        end
    end
end

@testset "3D uniform mesh: 8 children, 6 faces" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])
    leaves = [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])]
    @test length(leaves) == 8

    bf = brute_force_face_neighbors(mesh)
    for i in leaves
        nb = face_neighbors(mesh, i)
        @test length(nb) == 6
        # Each corner has 3 boundary faces and 3 interior faces
        @test count(==(UInt32(0)), nb) == 3
        for f in 1:6
            ref = sort(get(bf[i], f, Int[]))
            ffn = sort(Int.(face_fine_neighbors(mesh, i, f)))
            @test ffn == ref
            if isempty(ref)
                @test nb[f] == 0
            else
                @test Int(nb[f]) in ref
            end
        end
    end
end

@testset "Cache invalidation on refinement" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    g1 = ensure_neighbor_graph!(mesh)
    @test mesh._cached_neighbor_graph === g1

    # Refine — listener should clear the cache.
    refine_cells!(mesh, [2])
    @test mesh._cached_neighbor_graph === nothing

    # Force a rebuild and assert it differs from the stale graph.
    g2 = ensure_neighbor_graph!(mesh)
    @test g2 !== g1
    @test length(g2.representatives) == n_cells(mesh)

    # And matches a fresh build.
    g3 = build_neighbor_graph(mesh)
    @test g2.representatives == g3.representatives
    # Fine dicts may differ in ordering but should be equal as dicts.
    @test g2.fine == g3.fine
end

@testset "Balanced refinement enforces 2:1 level diff" begin
    mesh = HierarchicalMesh{2}(; balanced=true)
    refine_cells!(mesh, [1])  # 5 cells: root + 4 leaves at level 1
    # Refine cell 2 (one of the level-1 leaves).
    refine_cells!(mesh, [2])
    # Now refine one of the level-2 leaves. After rebuild, cell 2 is the
    # parent; its children are 3..6 (level 2). Pick the first.
    HierarchicalGrids.rebuild_caches!(mesh)
    lv2_leaves = [i for i in 1:n_cells(mesh)
                  if HierarchicalGrids.is_leaf(mesh[i]) && mesh._levels[i] == 2]
    @test !isempty(lv2_leaves)
    refine_cells!(mesh, [lv2_leaves[1]])

    # All face-neighbors must satisfy |level(i) - level(j)| <= 1 for every
    # leaf pair. Use brute-force adjacency.
    HierarchicalGrids.rebuild_caches!(mesh)
    levels = mesh._levels
    bf = brute_force_face_neighbors(mesh)
    leaves = [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])]
    for i in leaves
        for f in 1:4
            for j in get(bf[i], f, Int[])
                @test abs(Int(levels[i]) - Int(levels[j])) <= 1
            end
        end
    end
end

@testset "face_neighbors_with_bcs cache: hit, invalidate, alloc-free" begin
    # 2-level uniformly refined 2D mesh: 4 root children, then split each →
    # 16 leaves arranged in a 4×4 grid on the unit square.
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh,
        [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])])

    bcs_xy = FrameBoundaries(((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)))
    bcs_x  = FrameBoundaries(((PERIODIC, PERIODIC), (OUTFLOW,  OUTFLOW)))

    leaves = [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])]

    # First call builds + caches.
    @test mesh._cached_bc_neighbor_table === nothing
    nb1 = face_neighbors_with_bcs(mesh, leaves[1], bcs_xy)
    @test mesh._cached_bc_neighbor_table !== nothing
    cache_dict = mesh._cached_bc_neighbor_table
    @test isa(cache_dict, Dict)
    @test haskey(cache_dict, (true, true))
    table_xy = cache_dict[(true, true)]
    @test length(table_xy) == n_cells(mesh)

    # Second call hits the cache: returns identical bytes and the table
    # vector is the SAME object (===), confirming no rebuild.
    nb2 = face_neighbors_with_bcs(mesh, leaves[1], bcs_xy)
    @test nb1 === nb2
    @test cache_dict[(true, true)] === table_xy

    # Different mask → distinct cache entry, doesn't displace the (T,T) one.
    face_neighbors_with_bcs(mesh, leaves[1], bcs_x)
    @test haskey(cache_dict, (true, false))
    @test cache_dict[(true, true)] === table_xy

    # Doubly-periodic on a 4x4 mesh: the lo-x leaf at (0, 0) should wrap
    # in -x to a hi-x leaf, and in -y to a hi-y leaf. Sanity-check the
    # cache contents pointwise against the per-call function.
    for i in leaves
        @test face_neighbors_with_bcs(mesh, i, bcs_xy) === table_xy[Int(i)]
    end

    # Refinement invalidates: listener clears the cache slot.
    refine_cells!(mesh, [leaves[1]])
    @test mesh._cached_bc_neighbor_table === nothing

    # Subsequent call rebuilds.
    leaves2 = [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])]
    face_neighbors_with_bcs(mesh, leaves2[1], bcs_xy)
    @test mesh._cached_bc_neighbor_table !== nothing
    @test length(mesh._cached_bc_neighbor_table[(true, true)]) == n_cells(mesh)

    # Allocation profile: warm-path call should not allocate (single
    # table read + tuple return). Julia 1.10's narrower inference leaves
    # a small residual; 1.11+ folds it away.
    face_neighbors_with_bcs(mesh, leaves2[1], bcs_xy)  # warm
    bytes = @allocated face_neighbors_with_bcs(mesh, leaves2[1], bcs_xy)
    if VERSION >= v"1.11"
        @test bytes == 0
    else
        @test bytes < 96
    end
end

@testset "face_neighbors_with_bcs: result equality vs scratch build" begin
    # Confirm the cache produces the same per-cell tuples as a fresh
    # build over a few mesh shapes.
    function fresh_build(mesh, mask)
        # Direct call to the internal builder, bypassing the cache.
        return HierarchicalGrids.Mesh._build_periodic_bc_table(mesh, mask)
    end

    # 2D 2-level uniform mesh, doubly periodic.
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh,
        [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])])
    bcs = FrameBoundaries(((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)))
    table_cached = HierarchicalGrids.Mesh._ensure_periodic_bc_table!(
        mesh, (true, true))
    table_fresh = fresh_build(mesh, (true, true))
    @test table_cached == table_fresh

    # Mixed BC: x-periodic only.
    mesh2 = HierarchicalMesh{2}()
    refine_cells!(mesh2, [1])
    refine_cells!(mesh2,
        [i for i in 1:n_cells(mesh2) if HierarchicalGrids.is_leaf(mesh2[i])])
    table_x = HierarchicalGrids.Mesh._ensure_periodic_bc_table!(
        mesh2, (true, false))
    table_x_fresh = HierarchicalGrids.Mesh._build_periodic_bc_table(
        mesh2, (true, false))
    @test table_x == table_x_fresh
end
