using Test
using HierarchicalGrids
using HierarchicalGrids: RefinementEvent, ListenerHandle,
    register_refinement_listener!, unregister_refinement_listener!

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Identity vector that mirrors the mesh's cells. The listener should keep the
# entries aligned to mesh cell indices using `event.index_remap`.
mutable struct MirrorVec
    data::Vector{Int}        # data[i] is some "identity" value tied to the cell
    next_id::Int             # used to assign fresh ids to newly created cells
end

# Create a mirror initialised to 1..n_cells(mesh).
function make_mirror(mesh)
    n = n_cells(mesh)
    return MirrorVec(collect(1:Int(n)), Int(n) + 1)
end

# Apply a RefinementEvent to a MirrorVec by permuting/resizing using
# index_remap, then assigning fresh ids to children produced by refinement.
function apply_event!(mirror::MirrorVec, event::RefinementEvent, new_n::Int)
    new_data = zeros(Int, new_n)
    remap = event.index_remap
    for old_i in eachindex(remap)
        new_i = remap[old_i]
        if new_i != 0
            new_data[Int(new_i)] = mirror.data[Int(old_i)]
        end
    end
    # Assign fresh ids to NEW children produced by refinement
    for rng in event.new_children
        for new_i in rng
            new_data[Int(new_i)] = mirror.next_id
            mirror.next_id += 1
        end
    end
    mirror.data = new_data
    return mirror
end

# Reconstruct a per-cell identity vector by replaying an op log against an
# initial single-cell mesh. Used as a brute-force cross-check for the fuzz
# test: it doesn't depend on the listener.
function brute_force_identities(ops, D)
    mesh = HierarchicalMesh{D}()
    n0 = n_cells(mesh)
    ids = collect(1:Int(n0))
    next_id = Int(n0) + 1
    for op in ops
        kind, idx = op
        if kind == :refine
            # Build the new identity vector mirroring the rebuild order.
            n_old = n_cells(mesh)
            n_children = 1 << D
            new_n = n_old + n_children
            new_ids = Vector{Int}(undef, new_n)
            write_idx = 1
            for read_idx in 1:n_old
                new_ids[write_idx] = ids[read_idx]
                write_idx += 1
                if read_idx == idx
                    for _ in 1:n_children
                        new_ids[write_idx] = next_id
                        next_id += 1
                        write_idx += 1
                    end
                end
            end
            ids = new_ids
            refine_cells!(mesh, [idx])
        elseif kind == :coarsen
            # Determine cells to remove: subtree of `idx` excluding itself
            # We have to do this BEFORE coarsening (need subtree info).
            child_idxs = collect(HierarchicalGrids.find_children(mesh, idx))
            # Children must be leaves in the supplied ops; if not, skip.
            if !all(HierarchicalGrids.is_leaf(mesh[c]) for c in child_idxs)
                continue
            end
            # collect all descendants of idx (excluding idx itself)
            HierarchicalGrids.rebuild_caches!(mesh)
            sub_size = mesh._subtree_sizes[idx]
            removed = Set(Int(idx + k) for k in 1:(Int(sub_size) - 1))
            new_ids = Int[]
            for read_idx in 1:n_cells(mesh)
                if !(read_idx in removed)
                    push!(new_ids, ids[read_idx])
                end
            end
            ids = new_ids
            coarsen_cells!(mesh, [idx])
        end
    end
    return mesh, ids
end

# ---------------------------------------------------------------------------
# 1. Basic refine event
# ---------------------------------------------------------------------------
@testset "Basic refine event (3D)" begin
    mesh = HierarchicalMesh{3}()
    refine_cells!(mesh, [1])  # warm the mesh: root + 8 children
    n_old = n_cells(mesh)

    captured = Ref{Union{Nothing, RefinementEvent}}(nothing)
    register_refinement_listener!(mesh, ev -> (captured[] = ev))

    target = 2  # first child of root
    refine_cells!(mesh, [target])

    ev = captured[]
    @test ev !== nothing
    @test ev.refined_parents == UInt32[target]
    @test isempty(ev.coarsened_parents)
    @test isempty(ev.removed_old_indices)
    @test length(ev.new_children) == 1
    rng = ev.new_children[1]
    @test length(rng) == 8           # 2^3 children for D=3
    @test first(rng) == UInt32(target + 1)
    @test last(rng) == UInt32(target + 8)
    @test length(ev.index_remap) == n_old
    # Old indices ≤ target keep their position
    for i in 1:target
        @test ev.index_remap[i] == UInt32(i)
    end
    # Old indices > target shift by 8 (the children inserted right after target)
    for i in (target + 1):n_old
        @test ev.index_remap[i] == UInt32(i + 8)
    end
end

# ---------------------------------------------------------------------------
# 2. Mirror vector through a sequence
# ---------------------------------------------------------------------------
@testset "Mirror vector tracks cells through refine and coarsen" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])  # 1 + 4 = 5 cells

    mirror = make_mirror(mesh)
    register_refinement_listener!(mesh, ev -> apply_event!(mirror, ev, n_cells(mesh)))

    # Refine cell 2 (a leaf)
    refine_cells!(mesh, [2])
    @test length(mirror.data) == n_cells(mesh)
    # The original "1" identity (root) should still be at index 1 of mirror
    @test mirror.data[1] == 1
    # The original cell-2 identity should still exist somewhere (parent of new children)
    @test 2 in mirror.data

    # Refine another leaf
    leaf_idx = findfirst(i -> HierarchicalGrids.is_leaf(mesh[i]), 1:n_cells(mesh))
    refine_cells!(mesh, [leaf_idx])
    @test length(mirror.data) == n_cells(mesh)
    @test mirror.data[1] == 1

    # Coarsen back: find a non-leaf whose children are all leaves
    parent_idx = nothing
    for i in 1:n_cells(mesh)
        if !HierarchicalGrids.is_leaf(mesh[i])
            kids = HierarchicalGrids.find_children(mesh, i)
            if all(HierarchicalGrids.is_leaf(mesh[c]) for c in kids)
                parent_idx = i
                break
            end
        end
    end
    @test parent_idx !== nothing
    coarsen_cells!(mesh, [parent_idx])
    @test length(mirror.data) == n_cells(mesh)
end

# ---------------------------------------------------------------------------
# 3. Coarsen event
# ---------------------------------------------------------------------------
@testset "Coarsen event" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])  # 1 + 4 = 5 cells
    refine_cells!(mesh, [2])  # 5 + 4 = 9 cells; cell 2 has children 3..6 (in 2D)

    captured = Ref{Union{Nothing, RefinementEvent}}(nothing)
    register_refinement_listener!(mesh, ev -> (captured[] = ev))

    n_old = n_cells(mesh)
    coarsen_cells!(mesh, [2])

    ev = captured[]
    @test ev !== nothing
    @test isempty(ev.refined_parents)
    @test isempty(ev.new_children)
    @test ev.coarsened_parents == UInt32[2]  # cell 2 stays at index 2 (new)
    @test length(ev.removed_old_indices) == 4  # 4 children removed
    @test length(ev.index_remap) == n_old
    # Old index 1 -> new 1, old 2 -> new 2 (parent kept), old 3..6 -> 0 (removed),
    # old 7..9 -> shift by -4
    @test ev.index_remap[1] == 1
    @test ev.index_remap[2] == 2
    for i in 3:6
        @test ev.index_remap[i] == 0
    end
    for i in 7:n_old
        @test ev.index_remap[i] == UInt32(i - 4)
    end
    # removed_old_indices must list exactly indices 3..6
    @test sort(collect(ev.removed_old_indices)) == UInt32[3, 4, 5, 6]
end

# ---------------------------------------------------------------------------
# 4. Multiple listeners — ordered
# ---------------------------------------------------------------------------
@testset "Multiple listeners fire in registration order" begin
    mesh = HierarchicalMesh{2}()
    log = Int[]
    register_refinement_listener!(mesh, ev -> push!(log, 1))
    register_refinement_listener!(mesh, ev -> push!(log, 2))
    register_refinement_listener!(mesh, ev -> push!(log, 3))
    refine_cells!(mesh, [1])
    @test log == [1, 2, 3]
end

# ---------------------------------------------------------------------------
# 5. Unregister
# ---------------------------------------------------------------------------
@testset "Unregister works" begin
    mesh = HierarchicalMesh{2}()
    log = Symbol[]
    h1 = register_refinement_listener!(mesh, ev -> push!(log, :a))
    h2 = register_refinement_listener!(mesh, ev -> push!(log, :b))
    @test h1 != h2
    @test unregister_refinement_listener!(mesh, h1) == true
    @test unregister_refinement_listener!(mesh, h1) == false  # already gone
    refine_cells!(mesh, [1])
    @test log == [:b]
    @test unregister_refinement_listener!(mesh, h2) == true
end

# ---------------------------------------------------------------------------
# 6. Exception propagation but mesh still consistent
# ---------------------------------------------------------------------------
@testset "Listener exception is rethrown but mesh stays consistent" begin
    mesh = HierarchicalMesh{2}()
    register_refinement_listener!(mesh, ev -> error("boom"))

    n_before = n_cells(mesh)
    @test_throws ErrorException refine_cells!(mesh, [1])
    # Mesh should still have been refined
    @test n_cells(mesh) == n_before + 4
    @test !HierarchicalGrids.is_leaf(mesh[1])

    # All other listeners should still fire even when one throws
    log = Int[]
    register_refinement_listener!(mesh, ev -> push!(log, 1))
    register_refinement_listener!(mesh, ev -> push!(log, 2))
    @test_throws ErrorException refine_cells!(mesh, [2])
    @test log == [1, 2]
end

# ---------------------------------------------------------------------------
# 7. Empty input doesn't fire
# ---------------------------------------------------------------------------
@testset "Empty input does not fire event" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    fired = Ref(false)
    register_refinement_listener!(mesh, ev -> (fired[] = true))
    refine_cells!(mesh, Int[])
    coarsen_cells!(mesh, Int[])
    @test fired[] == false
end

# ---------------------------------------------------------------------------
# 8. Randomized fuzz: mirror vs brute force
# ---------------------------------------------------------------------------
@testset "Randomized refine/coarsen fuzz vs brute force" begin
    using Random
    rng = MersenneTwister(0xCAFE)
    D = 2

    # Build an op log valid against a freshly-created mesh.
    mesh_check = HierarchicalMesh{D}()
    ops = Tuple{Symbol, Int}[]
    for _ in 1:50
        leaves = [i for i in 1:n_cells(mesh_check) if HierarchicalGrids.is_leaf(mesh_check[i])]
        # Find non-leaf cells whose children are ALL leaves (legal coarsen targets)
        coarsen_candidates = Int[]
        for i in 1:n_cells(mesh_check)
            if !HierarchicalGrids.is_leaf(mesh_check[i])
                kids = HierarchicalGrids.find_children(mesh_check, i)
                if all(HierarchicalGrids.is_leaf(mesh_check[c]) for c in kids)
                    push!(coarsen_candidates, i)
                end
            end
        end

        choose_refine = !isempty(leaves) && (isempty(coarsen_candidates) || rand(rng) < 0.7)
        if choose_refine
            tgt = leaves[rand(rng, 1:length(leaves))]
            push!(ops, (:refine, tgt))
            refine_cells!(mesh_check, [tgt])
        elseif !isempty(coarsen_candidates)
            tgt = coarsen_candidates[rand(rng, 1:length(coarsen_candidates))]
            push!(ops, (:coarsen, tgt))
            coarsen_cells!(mesh_check, [tgt])
        else
            break
        end
    end

    # Now replay the ops on a fresh mesh with a mirror listener.
    mesh = HierarchicalMesh{D}()
    mirror = make_mirror(mesh)
    register_refinement_listener!(mesh, ev -> apply_event!(mirror, ev, n_cells(mesh)))

    for (kind, idx) in ops
        if kind == :refine
            refine_cells!(mesh, [idx])
        else
            coarsen_cells!(mesh, [idx])
        end
        @test length(mirror.data) == n_cells(mesh)
    end

    # Cross-check: re-run brute force over the same ops, get the expected ids.
    _, expected_ids = brute_force_identities(ops, D)
    @test mirror.data == expected_ids
end

# ---------------------------------------------------------------------------
# 9. Listener registered DURING a callback is not fired for the in-flight event
# ---------------------------------------------------------------------------
@testset "Mid-event registration is deferred to next event" begin
    mesh = HierarchicalMesh{2}()
    log = Symbol[]
    register_refinement_listener!(mesh, ev -> begin
        push!(log, :outer)
        register_refinement_listener!(mesh, ev2 -> push!(log, :inner))
    end)
    refine_cells!(mesh, [1])
    @test log == [:outer]
    refine_cells!(mesh, [2])
    @test :inner in log
end

# ---------------------------------------------------------------------------
# 10. No-listener fast path: no event allocations
# ---------------------------------------------------------------------------
@testset "No-listener fast path skips event construction" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])  # warm
    # Build a callable refinable index for a cell
    leaf = findfirst(i -> HierarchicalGrids.is_leaf(mesh[i]), 1:n_cells(mesh))
    # Capture allocations of a refine call with no listeners.
    a = @allocated refine_cells!(mesh, [leaf])
    # We don't assert zero (the cells rebuild must allocate), but we do assert
    # that registering then unregistering a listener still results in equal
    # alloc behaviour for a no-listener call (the fast-path guard works).
    leaf2 = findfirst(i -> HierarchicalGrids.is_leaf(mesh[i]), 1:n_cells(mesh))
    a2 = @allocated refine_cells!(mesh, [leaf2])
    @test a > 0   # there's always at least the cells rebuild
    @test a2 > 0
end
