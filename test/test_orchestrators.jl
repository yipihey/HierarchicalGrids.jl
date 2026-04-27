using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells, level_of,
    cell_physical_box, EulerianFrame, FrameBoundaries, enumerate_leaves,
    BCKind, PERIODIC, REFLECTING, OUTFLOW, DIRICHLET, INFLOW,
    MonomialBasis, n_coeffs, allocate_polynomial_fields, SoA,
    face_neighbors, face_fine_neighbors, ensure_neighbor_graph!,
    parallel_foreach, Sequential, OhMyThreadsBackend,
    CellView, cell_view, ghost_depth, halo_view_multi,
    for_each_cell!, for_each_face!

const SolverHaloView = HierarchicalGrids.Solver.HaloView

# ============================================================================
# Setup helpers
# ============================================================================

# 1D mesh with 8 equal-size leaves on [0, 1]; basis has nc coefficients and
# the field :rho is initialized so cell `i` has coefficients (10*k + idx_pos),
# where idx_pos is its position 1..8 along the x axis.
function _setup1d_8(; field_names = (:rho,))
    mesh = HierarchicalMesh{1}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, enumerate_leaves(mesh))
    refine_cells!(mesh, enumerate_leaves(mesh))
    n = n_cells(mesh)
    basis = MonomialBasis{1, 1}()
    nc = n_coeffs(basis)
    leaves = enumerate_leaves(mesh)
    @assert length(leaves) == 8
    # ordered_leaves are leaves sorted by physical x — but DFS order on a
    # binary 1D refinement IS already left-to-right, so leaves[k] has
    # physical position k.
    pairs = [n => Float64 for n in field_names]
    fin  = allocate_polynomial_fields(SoA(), basis, n; pairs...)
    fout = allocate_polynomial_fields(SoA(), basis, n; pairs...)
    for nm in field_names
        for i in 1:n
            getproperty(fin, nm)[i]  = ntuple(k -> 0.0, nc)
            getproperty(fout, nm)[i] = ntuple(k -> 0.0, nc)
        end
        # Set leaves' coefficients by physical-order index `pos`.
        for (pos, i) in enumerate(leaves)
            getproperty(fin, nm)[i] = ntuple(k -> Float64(10 * k + pos), nc)
        end
    end
    frame = EulerianFrame(mesh, (0.0,), (1.0,))
    return (mesh, frame, fin, fout, basis, nc, leaves)
end

# Materialize a NamedTuple-of-PolynomialFieldViews from a PolynomialFieldSet
# so the orchestrator tests can build the views the same way the orchestrator
# accepts them (NamedTuple of PolynomialFieldView).
function _views(fs, names)
    return NamedTuple{names}(map(n -> getproperty(fs, n), names))
end

# ============================================================================
# 1. Single-cell update: doubling the :rho coefficients
# ============================================================================

@testset "for_each_cell!: single-cell doubling (ghost_depth=0)" begin
    (mesh, frame, fin, fout, basis, nc, leaves) = _setup1d_8()
    fin_v  = _views(fin,  (:rho,))
    fout_v = _views(fout, (:rho,))

    kernel = function (cv, hv, ctx)
        old = cv[Val(:rho)]
        cv[Val(:rho)] = ntuple(k -> 2.0 * old[k], nc)
        return nothing
    end

    for_each_cell!(kernel, fout_v, fin_v, frame; ghost_depth = 0,
                   backend = Sequential())

    # Every leaf should now have 2x coefficients in fout; all non-leaf
    # cells (root + intermediate) remain at the initial 0.0.
    for (pos, i) in enumerate(leaves)
        for k in 1:nc
            @test fout.rho[i][k] == 2.0 * (10 * k + pos)
        end
    end
end

# ============================================================================
# 2. 3-point stencil with periodic BC: centered difference
# ============================================================================

@testset "for_each_cell!: 3-point stencil, periodic BC, 1D" begin
    (mesh, frame, fin, fout, basis, nc, leaves) = _setup1d_8()
    fin_v  = _views(fin,  (:rho,))
    fout_v = _views(fout, (:rho,))
    bcs = FrameBoundaries(((PERIODIC, PERIODIC),))

    # Kernel: write the FIRST coefficient of fout to (rho_right - rho_left),
    # using ONLY the first coefficient of each. (Using `.rho[k=1]` keeps the
    # arithmetic simple and lets us check exact values.)
    kernel = function (cv, hv, ctx)
        right = hv[Val(:rho), (1,)]
        left  = hv[Val(:rho), (-1,)]
        @assert right !== nothing
        @assert left !== nothing
        diff = right[1] - left[1]
        # Write only into coefficient 1; keep the rest zero.
        cv[Val(:rho)] = ntuple(k -> k == 1 ? diff : 0.0, nc)
        return nothing
    end

    for_each_cell!(kernel, fout_v, fin_v, frame; ghost_depth = 1, bcs = bcs,
                   backend = Sequential())

    # Cell at physical position `pos` has rho[1] = 10 + pos initially
    # (coefficients are 10*k + pos so k=1 => 10+pos). The right neighbor
    # under periodic BC has pos+1 mod 8 (in 1..8); left has pos-1 mod 8.
    for (pos, i) in enumerate(leaves)
        right_pos = mod1(pos + 1, 8)
        left_pos  = mod1(pos - 1, 8)
        expected = (10 + right_pos) - (10 + left_pos)
        @test fout.rho[i][1] == expected
    end
end

# ============================================================================
# 3. Reflecting BC: boundary cells get central-cell value
# ============================================================================

@testset "for_each_cell!: 3-point stencil, reflecting BC, 1D" begin
    (mesh, frame, fin, fout, basis, nc, leaves) = _setup1d_8()
    fin_v  = _views(fin,  (:rho,))
    fout_v = _views(fout, (:rho,))
    bcs = FrameBoundaries(((REFLECTING, REFLECTING),))

    kernel = function (cv, hv, ctx)
        right = hv[Val(:rho), (1,)]
        left  = hv[Val(:rho), (-1,)]
        @assert right !== nothing
        @assert left !== nothing
        diff = right[1] - left[1]
        cv[Val(:rho)] = ntuple(k -> k == 1 ? diff : 0.0, nc)
        return nothing
    end

    for_each_cell!(kernel, fout_v, fin_v, frame; ghost_depth = 1, bcs = bcs,
                   backend = Sequential())

    # Boundary cells: pos=1 (left edge) reflects → left = central. pos=8
    # (right edge) reflects → right = central. Central rho[1] is 10+pos.
    @test fout.rho[leaves[1]][1] == ((10 + 2) - (10 + 1))   # right=pos2, left=pos1
    @test fout.rho[leaves[8]][1] == ((10 + 8) - (10 + 7))   # right=pos8, left=pos7

    # Interior cells: standard centered differences.
    for pos in 2:7
        i = leaves[pos]
        @test fout.rho[i][1] == ((10 + pos + 1) - (10 + pos - 1))
    end
end

# ============================================================================
# 4. Multi-field: kernel reads :rho, :u; writes :rho_new, :u_new
# ============================================================================

@testset "for_each_cell!: multi-field" begin
    # CellView requires fields_in and fields_out to share names (the
    # double-buffering convention: read from old binding, write to new
    # binding, swap between time steps). So multi-field kernels read AND
    # write under the same set of names.
    (mesh, frame, fin, fout, basis, nc, leaves) = _setup1d_8(field_names = (:rho, :u))
    # Initialize :u with a different pattern.
    for (pos, i) in enumerate(leaves)
        fin.u[i] = ntuple(k -> Float64(100 * k + pos), nc)
    end

    fin_v  = _views(fin,  (:rho, :u))
    fout_v = _views(fout, (:rho, :u))

    kernel = function (cv, hv, ctx)
        ρ = cv[Val(:rho)]
        u = cv[Val(:u)]
        cv[Val(:rho)] = ntuple(k -> ρ[k] + 1.0, nc)
        cv[Val(:u)]   = ntuple(k -> 0.5 * u[k], nc)
        return nothing
    end

    for_each_cell!(kernel, fout_v, fin_v, frame; ghost_depth = 0,
                   backend = Sequential())

    for (pos, i) in enumerate(leaves)
        for k in 1:nc
            @test fout.rho[i][k] == 10 * k + pos + 1.0
            @test fout.u[i][k]   == 0.5 * (100 * k + pos)
        end
    end
end

# ============================================================================
# 5. No-cache-race regression: fresh frame, multi-thread run
# ============================================================================

@testset "for_each_cell!: pre-built caches under multi-thread fan-out" begin
    # Build a fresh mesh whose caches are NOT yet populated; the orchestrator
    # must pre-build before going parallel, otherwise the lazy cache machinery
    # would race.
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, enumerate_leaves(mesh))
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    fin  = allocate_polynomial_fields(SoA(), basis, n; rho = Float64)
    fout = allocate_polynomial_fields(SoA(), basis, n; rho = Float64)
    for i in 1:n
        fin.rho[i]  = ntuple(k -> Float64(i * 10 + k), nc)
        fout.rho[i] = ntuple(k -> 0.0, nc)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    fin_v  = _views(fin,  (:rho,))
    fout_v = _views(fout, (:rho,))

    kernel = function (cv, hv, ctx)
        old = cv[Val(:rho)]
        cv[Val(:rho)] = ntuple(k -> 3.0 * old[k], nc)
        return nothing
    end

    # Use a dynamic backend; the orchestrator pre-builds caches/graph.
    for_each_cell!(kernel, fout_v, fin_v, frame; ghost_depth = 0,
                   backend = OhMyThreadsBackend(:dynamic))

    # All leaves get triple, non-leaves stay zero.
    for i in enumerate_leaves(mesh)
        for k in 1:nc
            @test fout.rho[i][k] == 3.0 * (10 * i + k)
        end
    end
end

# ============================================================================
# 6. Backend determinism across all five backends
# ============================================================================

@testset "for_each_cell!: backend determinism across all five backends" begin
    (mesh, frame, fin, fout, basis, nc, leaves) = _setup1d_8()
    fin_v  = _views(fin,  (:rho,))

    function run(backend)
        fout_local = allocate_polynomial_fields(SoA(), basis, n_cells(mesh); rho = Float64)
        for i in 1:n_cells(mesh)
            fout_local.rho[i] = ntuple(k -> 0.0, nc)
        end
        fout_v = _views(fout_local, (:rho,))
        kernel = function (cv, hv, ctx)
            old = cv[Val(:rho)]
            cv[Val(:rho)] = ntuple(k -> 7.0 * old[k] - 3.0, nc)
            return nothing
        end
        for_each_cell!(kernel, fout_v, fin_v, frame; ghost_depth = 0,
                       backend = backend)
        return [collect(fout_local.rho[i]) for i in 1:n_cells(mesh)]
    end

    ref = run(Sequential())
    for backend in (OhMyThreadsBackend(:dynamic),
                    OhMyThreadsBackend(:static),
                    OhMyThreadsBackend(:greedy),
                    OhMyThreadsBackend(:serial))
        snap = run(backend)
        @test snap == ref
    end
end

# ============================================================================
# 7. for_each_face! — single flux pass on a 1D mesh
# ============================================================================

@testset "for_each_face!: single flux pass, 1D, no BC" begin
    (mesh, frame, fin, fout, basis, nc, leaves) = _setup1d_8()
    # Pre-allocate an upwind flux buffer keyed by face id. Use a Dict so we
    # can check exactly the (i, j) pairs the orchestrator dispatched.
    fluxes = Dict{Tuple{Int, Int}, Float64}()
    fluxes_lock = ReentrantLock()

    fin_v = _views(fin, (:rho,))
    fluxes_out = (rho = fluxes,)

    flux_kernel = function (cv_left, cv_right, normal, ctx)
        # Upwind w.r.t. positive velocity: flux = rho_left[1] (the constant
        # coefficient), tagged by (left.index, right.index).
        v = cv_left[Val(:rho)][1]
        lock(fluxes_lock) do
            fluxes[(cv_left.index, cv_right.index)] = v
        end
        return nothing
    end

    for_each_face!(flux_kernel, fluxes_out, fin_v, frame;
                   backend = Sequential())

    # The 8 leaves form 7 interior faces between consecutive cells in
    # physical order. Each face is dispatched exactly once. The dict's
    # value is rho[left, k=1] = 11 + pos(left).
    @test length(fluxes) == 7
    for pos in 1:7
        i = leaves[pos]
        j = leaves[pos + 1]
        # The orchestrator's convention is "lower-indexed cell is left",
        # which matches physical order in the DFS-built mesh.
        @test haskey(fluxes, (i, j))
        # Cell at physical position `pos` has rho[1] = 10*1 + pos = 10+pos.
        @test fluxes[(i, j)] == 10 + pos
    end
end

# ============================================================================
# 8. for_each_face! — boundary face dispatch
# ============================================================================

@testset "for_each_face!: boundary dispatch — periodic vs reflecting" begin
    # Periodic: every face is interior under wrap. The orchestrator's
    # boundary list is built from face_neighbors == 0; periodic wiring
    # doesn't affect that, so boundary cells STILL have 0 entries on the
    # outer face. We expect `flux_kernel_boundary` to be invoked twice
    # (left + right edge of the 1D mesh) regardless of bcs — boundary
    # detection is a topology question, not a BC question.
    (mesh, frame, fin, fout, basis, nc, leaves) = _setup1d_8()
    fin_v = _views(fin, (:rho,))

    n_boundary = Ref(0)
    boundary_log = Tuple{Int, Int, Int}[]
    flux_kernel = (cv_l, cv_r, n, ctx) -> nothing
    flux_kernel_boundary = function (cv, axis, side, normal, bcs, ctx)
        n_boundary[] += 1
        push!(boundary_log, (cv.index, axis, side))
        return nothing
    end

    bcs_p = FrameBoundaries(((PERIODIC, PERIODIC),))
    for_each_face!(flux_kernel, (rho = nothing,), fin_v, frame;
                   bcs = bcs_p,
                   flux_kernel_boundary = flux_kernel_boundary,
                   backend = Sequential())
    @test n_boundary[] == 2   # left edge of cell 1 and right edge of cell 8
    # axis=1, sides 1 (lo) at leaves[1] and 2 (hi) at leaves[8]
    @test (leaves[1], 1, 1) in boundary_log
    @test (leaves[8], 1, 2) in boundary_log

    # Reset and run with reflecting — same boundary topology, same dispatch.
    n_boundary[] = 0
    empty!(boundary_log)
    bcs_r = FrameBoundaries(((REFLECTING, REFLECTING),))
    for_each_face!(flux_kernel, (rho = nothing,), fin_v, frame;
                   bcs = bcs_r,
                   flux_kernel_boundary = flux_kernel_boundary,
                   backend = Sequential())
    @test n_boundary[] == 2
    @test (leaves[1], 1, 1) in boundary_log
    @test (leaves[8], 1, 2) in boundary_log

    # Without `flux_kernel_boundary`, no boundary dispatch happens.
    n_boundary[] = 0
    for_each_face!(flux_kernel, (rho = nothing,), fin_v, frame;
                   bcs = bcs_r,
                   flux_kernel_boundary = nothing,
                   backend = Sequential())
    @test n_boundary[] == 0
end

# ============================================================================
# 9. for_each_face! — hanging-node face is dispatched per fine neighbor
# ============================================================================

@testset "for_each_face!: hanging-node face dispatch (2D, unbalanced)" begin
    # Build the same configuration as the views test:
    #   refine root → 4 leaves
    #   refine cell 2 → 4 fine leaves; cells 3, 4, 5 stay coarse
    # Cell 7 (lower-right coarse, 0.5..1 × 0..0.5) sees cell 2's fine
    # children on its -x face (face index 1).
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
    fin_v = _views(fin, (:rho,))

    # Count balanced-mesh face calls: refine ALL 4 root children for the
    # comparison count.
    function balanced_face_count()
        m2 = HierarchicalMesh{2}()
        refine_cells!(m2, [1])
        refine_cells!(m2, [2, 3, 4, 5])
        f2 = EulerianFrame(m2, (0.0, 0.0), (1.0, 1.0))
        fin2 = allocate_polynomial_fields(SoA(), basis, n_cells(m2); rho = Float64)
        for i in 1:n_cells(m2)
            fin2.rho[i] = ntuple(k -> 0.0, nc)
        end
        counter = Ref(0)
        kern = (l, r, nrm, c) -> (counter[] += 1; nothing)
        for_each_face!(kern, (rho = nothing,), _views(fin2, (:rho,)), f2;
                       backend = Sequential())
        return counter[]
    end

    n_calls = Ref(0)
    seen = Tuple{Int, Int}[]
    flux_kernel = function (cv_l, cv_r, normal, ctx)
        n_calls[] += 1
        push!(seen, (cv_l.index, cv_r.index))
        return nothing
    end

    for_each_face!(flux_kernel, (rho = nothing,), fin_v, frame;
                   backend = Sequential())

    # Locate cell 7's hanging face neighbors.
    nbs7_lo = face_fine_neighbors(mesh, 7, 1)   # -x face
    @assert length(nbs7_lo) >= 2

    # The unbalanced mesh has fewer interior faces overall than a fully
    # balanced one (only one quadrant got refined), but each fine neighbor
    # of cell 7's -x face becomes its own dispatch.
    n_unbalanced = n_calls[]
    n_balanced   = balanced_face_count()
    @test n_unbalanced != n_balanced

    # Concretely, the hanging-node face contributes one call per fine
    # neighbor. We sanity-check that each (coarse=7, fine) pair is in the
    # dispatch list at least once.
    for f in nbs7_lo
        # The orchestrator visits the +axis side of the lower-indexed
        # cell — cell 7 (coarse) is HIGHER indexed than its fine
        # neighbors (which are children of cell 2). So in the orchestrator's
        # `(i, j)` convention, the fine cell is `i` and cell 7 is `j`.
        # That's the +x face of the fine cell.
        @test ((Int(f), 7) in seen) || ((7, Int(f)) in seen)
    end
end
