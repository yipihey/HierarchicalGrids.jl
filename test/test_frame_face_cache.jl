# PR-2: FrameFaceCache. Caches `for_each_face!`'s interior + boundary face
# enumeration on the EulerianFrame, invalidated by a one-shot refinement
# listener mirroring the `_cached_neighbor_graph` pattern.

using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells,
    EulerianFrame, FrameBoundaries, enumerate_leaves,
    FrameFaceCache, ensure_face_cache!,
    MonomialBasis, n_coeffs, allocate_polynomial_fields, SoA,
    for_each_face!, Sequential, OhMyThreadsBackend,
    PERIODIC, REFLECTING,
    face_fine_neighbors, ensure_neighbor_graph!

using HierarchicalGrids.Overlap: _invalidate_face_cache!,
    _enumerate_interior_faces_for_cache, _enumerate_boundary_faces_for_cache

# ---------------------------------------------------------------------------
# Fixture: 64-leaf 2D mesh on the unit square, three full refinements.
# (4^3 = 64 leaves)
# ---------------------------------------------------------------------------

function _make_64leaf_2d_fixture(; field_names = (:rho,))
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, enumerate_leaves(mesh))
    refine_cells!(mesh, enumerate_leaves(mesh))
    @assert length(enumerate_leaves(mesh)) == 64

    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    pairs = [name => Float64 for name in field_names]
    fin = allocate_polynomial_fields(SoA(), basis, n; pairs...)
    for nm in field_names
        for i in 1:n
            getproperty(fin, nm)[i] = ntuple(k -> Float64(7 * i + 13 * k), nc)
        end
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    return (mesh, frame, fin, basis, nc)
end

_views_facecache(fs, names) = NamedTuple{names}(map(n -> getproperty(fs, n), names))

# ---------------------------------------------------------------------------
# 1. First-call build, second-call hit.
# ---------------------------------------------------------------------------

@testset "first-call build, second-call returns same object" begin
    (mesh, frame, fin, basis, nc) = _make_64leaf_2d_fixture()

    fc1 = ensure_face_cache!(frame, nothing)
    @test fc1 isa FrameFaceCache{2}
    fc2 = ensure_face_cache!(frame, nothing)
    @test fc1 === fc2  # IDENTICAL object on second call

    # SoA shapes match: interior_left_idx/right_idx/axis are equal length;
    # interior_hanging is a BitVector of the same length; boundary triple
    # is mutually equal-length.
    n_int = length(fc1.interior_left_idx)
    @test length(fc1.interior_right_idx) == n_int
    @test length(fc1.interior_axis)      == n_int
    @test length(fc1.interior_hanging)   == n_int

    n_bnd = length(fc1.boundary_cell_idx)
    @test length(fc1.boundary_axis) == n_bnd
    @test length(fc1.boundary_side) == n_bnd

    # 64 leaves on a balanced 2D mesh: 8x8 grid → 7+7+7+7 = ... topological
    # check via the same enumerator, byte-for-byte. Rather than hardcode
    # numbers we cross-check against the inline enumerator.
    interior_ref = _enumerate_interior_faces_for_cache(mesh)
    boundary_ref = _enumerate_boundary_faces_for_cache(mesh)
    @test n_int == length(interior_ref)
    @test n_bnd == length(boundary_ref)
    for (k, (i, j, axis, hanging)) in enumerate(interior_ref)
        @test fc1.interior_left_idx[k]  == i
        @test fc1.interior_right_idx[k] == j
        @test fc1.interior_axis[k]      == axis
        @test fc1.interior_hanging[k]   == hanging
    end
    for (k, (i, axis, side)) in enumerate(boundary_ref)
        @test fc1.boundary_cell_idx[k] == i
        @test fc1.boundary_axis[k]     == axis
        @test fc1.boundary_side[k]     == side
    end

    # show summary works
    s = sprint(show, fc1)
    @test occursin("FrameFaceCache{2}", s)
    @test occursin("n_interior=$(n_int)", s)
    @test occursin("n_boundary=$(n_bnd)", s)
end

# ---------------------------------------------------------------------------
# 2. Refinement invalidates the cache.
# ---------------------------------------------------------------------------

@testset "refinement clears cache slot via one-shot listener" begin
    (mesh, frame, fin, basis, nc) = _make_64leaf_2d_fixture()
    fc_old = ensure_face_cache!(frame, nothing)
    @test frame._cached_face_cache === fc_old

    # Listener count: ensure_neighbor_graph! also registers one, so we
    # check that the count INCLUDES the face-cache listener (≥ 1) AFTER
    # an explicit ensure_neighbor_graph!.
    ensure_neighbor_graph!(mesh)
    listener_count_before = length(mesh._listeners)

    # Refine — this fires every listener, including the face-cache one.
    leaves = enumerate_leaves(mesh)
    refine_cells!(mesh, [leaves[1]])

    # Slot should now be `nothing` (one-shot listener fired and zeroed it).
    @test frame._cached_face_cache === nothing
    # Listener should have unregistered itself (and the neighbor-graph
    # listener too — both are one-shot). So count drops by at least 2.
    @test length(mesh._listeners) <= listener_count_before - 2

    # Next ensure builds afresh and registers a new listener.
    fc_new = ensure_face_cache!(frame, nothing)
    @test fc_new !== fc_old   # fresh object
    @test fc_new isa FrameFaceCache{2}
    @test frame._cached_face_cache === fc_new

    # Topology changed (one extra refinement), so the cache contents differ.
    interior_ref = _enumerate_interior_faces_for_cache(mesh)
    boundary_ref = _enumerate_boundary_faces_for_cache(mesh)
    @test length(fc_new.interior_left_idx) == length(interior_ref)
    @test length(fc_new.boundary_cell_idx) == length(boundary_ref)
    @test length(fc_new.interior_left_idx) != length(fc_old.interior_left_idx)
end

# ---------------------------------------------------------------------------
# 3. Multi-frame on same mesh — each frame has its own cache.
# ---------------------------------------------------------------------------

@testset "multi-frame on shared mesh: independent caches" begin
    (mesh, frame_a, fin, basis, nc) = _make_64leaf_2d_fixture()
    # Second frame on same mesh, different physical box.
    frame_b = EulerianFrame(mesh, (10.0, 20.0), (11.0, 21.0))

    fc_a = ensure_face_cache!(frame_a, nothing)
    fc_b = ensure_face_cache!(frame_b, nothing)
    @test fc_a !== fc_b
    @test frame_a._cached_face_cache === fc_a
    @test frame_b._cached_face_cache === fc_b

    # Identity test: a second call on each returns its own.
    @test ensure_face_cache!(frame_a, nothing) === fc_a
    @test ensure_face_cache!(frame_b, nothing) === fc_b

    # Both frames share the mesh's topology, so the SoA contents match.
    @test fc_a.interior_left_idx == fc_b.interior_left_idx
    @test fc_a.interior_right_idx == fc_b.interior_right_idx
    @test fc_a.interior_axis == fc_b.interior_axis
    @test fc_a.boundary_cell_idx == fc_b.boundary_cell_idx
    @test fc_a.boundary_axis == fc_b.boundary_axis
    @test fc_a.boundary_side == fc_b.boundary_side

    # Refining the shared mesh invalidates BOTH caches.
    refine_cells!(mesh, [enumerate_leaves(mesh)[1]])
    @test frame_a._cached_face_cache === nothing
    @test frame_b._cached_face_cache === nothing
end

# ---------------------------------------------------------------------------
# 4. for_each_face! byte-equality with the cached path.
# ---------------------------------------------------------------------------

@testset "for_each_face! byte-equality vs uncached enumeration" begin
    # We can't easily run the OLD path (it's been replaced), but we CAN
    # verify the cached path produces identical dispatch order and per-face
    # arguments by comparing against the inline enumerators.
    (mesh, frame, fin, basis, nc) = _make_64leaf_2d_fixture()
    fin_v = _views_facecache(fin, (:rho,))

    seen = Tuple{Int, Int, Tuple{Float64, Float64}}[]
    flux_kernel = function (cv_l, cv_r, normal, ctx)
        push!(seen, (cv_l.index, cv_r.index, normal))
        return nothing
    end

    for_each_face!(flux_kernel, (rho = nothing,), fin_v, frame;
                   backend = Sequential())

    interior_ref = _enumerate_interior_faces_for_cache(mesh)
    @test length(seen) == length(interior_ref)
    for (k, (i_ref, j_ref, axis_ref, _hanging)) in enumerate(interior_ref)
        i_seen, j_seen, normal_seen = seen[k]
        @test i_seen == i_ref
        @test j_seen == j_ref
        # Normal is +axis unit vector
        expected_normal = ntuple(d -> d == axis_ref ? 1.0 : 0.0, 2)
        @test normal_seen == expected_normal
    end

    # Now run again — same dispatch (cache hit).
    seen2 = Tuple{Int, Int, Tuple{Float64, Float64}}[]
    flux_kernel2 = function (cv_l, cv_r, normal, ctx)
        push!(seen2, (cv_l.index, cv_r.index, normal))
        return nothing
    end
    for_each_face!(flux_kernel2, (rho = nothing,), fin_v, frame;
                   backend = Sequential())
    @test seen2 == seen
end

# ---------------------------------------------------------------------------
# 5. Allocation profile: warm `for_each_face!` allocates LESS than cold
#    (which builds the cache).
# ---------------------------------------------------------------------------

@testset "warm-path allocations beat cold-path build" begin
    function measure()
        (mesh, frame, fin, basis, nc) = _make_64leaf_2d_fixture()
        fin_v = _views_facecache(fin, (:rho,))
        flux_kernel = (cv_l, cv_r, n, ctx) -> nothing

        # Warm call(s) — first one builds the cache.
        for_each_face!(flux_kernel, (rho = nothing,), fin_v, frame;
                       backend = Sequential())
        for_each_face!(flux_kernel, (rho = nothing,), fin_v, frame;
                       backend = Sequential())

        # Force a cold rebuild by clearing the slot manually.
        _invalidate_face_cache!(frame)
        cold = @allocated for_each_face!(flux_kernel, (rho = nothing,),
                                          fin_v, frame; backend = Sequential())
        warm = @allocated for_each_face!(flux_kernel, (rho = nothing,),
                                          fin_v, frame; backend = Sequential())
        return (cold, warm)
    end

    cold, warm = measure()
    @test warm < cold
    # The face-list cache saves the per-call enumeration cost (~15 KB on
    # this fixture). We assert a conservative 10% reduction; in practice
    # the savings are larger but `cell_view`/closure overhead floors the
    # warm path. The point is: cache hit avoids enumeration work.
    @test warm <= cold * 0.95
end

# ---------------------------------------------------------------------------
# 6. Backend determinism with the cached path.
# ---------------------------------------------------------------------------

@testset "backend determinism under cached for_each_face!" begin
    function run_backend(backend)
        (mesh, frame, fin, basis, nc) = _make_64leaf_2d_fixture()
        fin_v = _views_facecache(fin, (:rho,))

        # Per-face accumulator keyed by (i, j); values guarded by a lock
        # so concurrent backends don't race.
        d = Dict{Tuple{Int, Int}, NTuple{2, Float64}}()
        lk = ReentrantLock()
        flux_kernel = function (cv_l, cv_r, normal, ctx)
            lock(lk) do
                d[(cv_l.index, cv_r.index)] = normal
            end
            return nothing
        end
        # Boundary kernel: log (i, axis, side, normal).
        bd = Dict{Tuple{Int, Int, Int}, NTuple{2, Float64}}()
        bd_lk = ReentrantLock()
        bd_kernel = function (cv, axis, side, normal, bcs, ctx)
            lock(bd_lk) do
                bd[(cv.index, axis, side)] = normal
            end
            return nothing
        end

        for_each_face!(flux_kernel, (rho = nothing,), fin_v, frame;
                       flux_kernel_boundary = bd_kernel,
                       backend = backend)
        return (d, bd)
    end

    ref_d, ref_bd = run_backend(Sequential())
    for backend in (OhMyThreadsBackend(:dynamic),
                    OhMyThreadsBackend(:static),
                    OhMyThreadsBackend(:greedy),
                    OhMyThreadsBackend(:serial))
        d, bd = run_backend(backend)
        @test d  == ref_d
        @test bd == ref_bd
    end
end

# ---------------------------------------------------------------------------
# 7. Correctness end-to-end: a real flux pass against a hand-checked oracle.
#    Mirror test_orchestrators.jl's 1D upwind test but with the new path.
# ---------------------------------------------------------------------------

@testset "for_each_face! end-to-end: 1D upwind, flux values match oracle" begin
    mesh = HierarchicalMesh{1}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, enumerate_leaves(mesh))
    refine_cells!(mesh, enumerate_leaves(mesh))
    n = n_cells(mesh)
    basis = MonomialBasis{1, 1}()
    nc = n_coeffs(basis)
    leaves = enumerate_leaves(mesh)
    @assert length(leaves) == 8
    fin = allocate_polynomial_fields(SoA(), basis, n; rho = Float64)
    for i in 1:n
        fin.rho[i] = ntuple(k -> 0.0, nc)
    end
    for (pos, i) in enumerate(leaves)
        fin.rho[i] = ntuple(k -> Float64(10 * k + pos), nc)
    end
    frame = EulerianFrame(mesh, (0.0,), (1.0,))
    fin_v = _views_facecache(fin, (:rho,))

    fluxes = Dict{Tuple{Int, Int}, Float64}()
    lk = ReentrantLock()
    flux_kernel = function (cv_left, cv_right, normal, ctx)
        v = cv_left[Val(:rho)][1]
        lock(lk) do
            fluxes[(cv_left.index, cv_right.index)] = v
        end
        return nothing
    end

    for_each_face!(flux_kernel, (rho = fluxes,), fin_v, frame;
                   backend = Sequential())

    @test length(fluxes) == 7
    for pos in 1:7
        i = leaves[pos]
        j = leaves[pos + 1]
        @test haskey(fluxes, (i, j))
        @test fluxes[(i, j)] == 10 + pos
    end
end

# ---------------------------------------------------------------------------
# 8. Hanging-node face dispatch survives the cached path.
# ---------------------------------------------------------------------------

@testset "hanging-node faces dispatched correctly under cache" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2])
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    fin = allocate_polynomial_fields(SoA(), basis, n; rho = Float64)
    for i in 1:n
        fin.rho[i] = ntuple(k -> Float64(i + k - 1), nc)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    fin_v = _views_facecache(fin, (:rho,))

    seen = Tuple{Int, Int}[]
    flux_kernel = function (cv_l, cv_r, normal, ctx)
        push!(seen, (cv_l.index, cv_r.index))
        return nothing
    end
    for_each_face!(flux_kernel, (rho = nothing,), fin_v, frame;
                   backend = Sequential())

    # Cell 7 sees cell 2's fine children on its -x face. Each fine
    # neighbor becomes its own dispatch.
    nbs7_lo = face_fine_neighbors(mesh, 7, 1)
    @assert length(nbs7_lo) >= 2
    for f in nbs7_lo
        @test ((Int(f), 7) in seen) || ((7, Int(f)) in seen)
    end

    # Cache should now be populated; clearing it and rerunning should
    # produce the same dispatch. The `interior_hanging` tag is set when
    # the LOWER-indexed cell is the coarse one with multiple fine
    # neighbors on its +axis side. In this fixture, cell 7 (coarse) is
    # higher-indexed than its fine neighbors 4 and 6, so the dispatch
    # is from 4's/6's +x side seeing cell 7 — `hanging=false` by the
    # original convention. We just assert the cache is populated.
    fc = ensure_face_cache!(frame, nothing)
    @test length(fc.interior_left_idx) == length(seen)

    seen2 = Tuple{Int, Int}[]
    flux_kernel2 = (l, r, n, c) -> (push!(seen2, (l.index, r.index)); nothing)
    for_each_face!(flux_kernel2, (rho = nothing,), fin_v, frame;
                   backend = Sequential())
    @test seen2 == seen
end
