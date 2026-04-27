using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells, level_of,
    cell_physical_box, EulerianFrame, FrameBoundaries,
    BCKind, PERIODIC, REFLECTING, OUTFLOW, DIRICHLET, INFLOW,
    MonomialBasis, n_coeffs, allocate_polynomial_fields, SoA,
    face_neighbors, face_fine_neighbors, ensure_neighbor_graph!,
    parallel_foreach, Sequential, OhMyThreadsBackend,
    CellView, cell_view, ghost_depth, halo_view_multi

# `HaloView` exists in two submodules: the legacy single-field
# `Storage.HaloView` (PR-C) and the new multi-field `Solver.HaloView`
# (PR-6). To avoid the import collision we reach for the multi-field
# version through a fully-qualified alias inside the test module.
const SolverHaloView = HierarchicalGrids.Solver.HaloView

# Setup: 2x2 leaf mesh on [0,1]^2 with two named Float64 fields :rho and :u
# initialized to a known per-cell pattern.
function _setup2d()
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    fin  = allocate_polynomial_fields(SoA(), basis, n; rho=Float64, u=Float64)
    fout = allocate_polynomial_fields(SoA(), basis, n; rho=Float64, u=Float64)
    for i in 1:n
        fin.rho[i] = ntuple(k -> Float64(i + k - 1),    nc)
        fin.u[i]   = ntuple(k -> Float64(10*i + k - 1), nc)
        fout.rho[i] = ntuple(_ -> 0.0, nc)
        fout.u[i]   = ntuple(_ -> 0.0, nc)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    return (mesh, frame, fin, fout, basis, nc)
end

# 4x4 mesh (2x2 root then refine each child).
function _setup2d_4x4()
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3, 4, 5])
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    fin  = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    fout = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    for i in 1:n
        fin.rho[i] = ntuple(k -> Float64(100*i + k - 1), nc)
        fout.rho[i] = ntuple(_ -> 0.0, nc)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    return (mesh, frame, fin, fout, basis, nc)
end

@testset "CellView basic read/write" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d()
    # cell 5 is the upper-right quadrant
    cv = cell_view((rho = fin.rho, u = fin.u),
                   (rho = fout.rho, u = fout.u), frame, 5)
    @test cv isa CellView
    @test cv.index == 5

    # Read inputs
    pv_rho = cv[:rho]
    @test pv_rho[1] == 5.0          # i + k - 1 = 5 + 0
    @test pv_rho[2] == 6.0
    pv_u = cv[:u]
    @test pv_u[1] == 50.0           # 10*i + k - 1 = 50

    # Write to output buffer
    new_rho = ntuple(k -> Float64(1000 + k), nc)
    cv[:rho] = new_rho
    # fout.rho[5] should now hold new_rho
    for k in 1:nc
        @test fout.rho[5][k] == new_rho[k]
    end
    # fin.rho[5] must NOT have changed
    @test fin.rho[5][1] == 5.0
    @test fin.rho[5][2] == 6.0
end

@testset "CellView metadata is precomputed and correct" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d()
    for i in [2, 3, 4, 5]
        cv = cell_view((rho = fin.rho, u = fin.u),
                       (rho = fout.rho, u = fout.u), frame, i)
        p_lo, p_hi = cell_physical_box(frame, i)
        expected_coords = ntuple(d -> (p_lo[d] + p_hi[d]) / 2, 2)
        expected_vol = (p_hi[1] - p_lo[1]) * (p_hi[2] - p_lo[2])
        @test cv.coords == expected_coords
        @test cv.volume ≈ expected_vol
        @test cv.level == Int(level_of(mesh, i))
        @test cv.index == i
    end

    # Property access is allocation-free (after warmup)
    cv = cell_view((rho = fin.rho, u = fin.u),
                   (rho = fout.rho, u = fout.u), frame, 5)
    f = cv -> (cv.coords, cv.volume, cv.level, cv.index)
    f(cv)
    a = @allocated f(cv)
    @test a == 0
end

@testset "CellView: getindex returns a view (no copy of coefficients)" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d()
    cv = cell_view((rho = fin.rho, u = fin.u),
                   (rho = fout.rho, u = fout.u), frame, 3)
    # Warm
    f = cv -> cv[:rho]
    f(cv)
    # PolynomialView is a tiny immutable struct; the Symbol path may
    # incur a small dispatch cost (literal-folding-dependent), but the
    # Val-keyed path is fully type-stable and zero-allocation.
    g = cv -> cv[Val(:rho)]
    g(cv)
    a = @allocated g(cv)
    @test a == 0

    # Type stability: @inferred succeeds for the Val-keyed access.
    @inferred g(cv)

    # Reading via the same coefficient column twice must give bitwise-
    # identical values (we did not copy).
    pv1 = cv[:rho]
    pv2 = cv[:rho]
    for k in 1:nc
        @test pv1[k] === pv2[k]
    end
end

@testset "HaloView: non-boundary neighbors (depth=1)" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_4x4()
    # cell 4 is at (0.25..0.5, 0..0.25). Its (1, 0) neighbor is cell 8
    # (at 0.5..0.75, 0..0.25). Its (-1, 0) is cell 3.
    fb = nothing
    hv = halo_view_multi((rho = fin.rho,), mesh, 4, 1; bcs = fb)
    @test hv isa SolverHaloView
    @test ghost_depth(hv) == 1

    # Cross-check with face_neighbors directly
    nbs = face_neighbors(mesh, 4)
    @test nbs[2] == 8        # axis 1 hi
    @test nbs[1] == 3        # axis 1 lo

    pv_right = hv[:rho, (1, 0)]
    @test pv_right !== nothing
    @test pv_right[1] == 100.0 * 8

    pv_left = hv[:rho, (-1, 0)]
    @test pv_left !== nothing
    @test pv_left[1] == 100.0 * 3
end

@testset "HaloView: depth=2 walks correctly" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_4x4()
    # Pre-build the neighbor graph
    ensure_neighbor_graph!(mesh)
    # cell 3 -> +x -> cell 4 -> +x -> cell 8
    hv = halo_view_multi((rho = fin.rho,), mesh, 3, 2; bcs = nothing)
    pv = hv[:rho, (2, 0)]
    @test pv !== nothing
    @test pv[1] == 100.0 * 8
    # Three-arg form too
    pv2 = hv[:rho, (2, 0), 2]
    @test pv2[1] == 100.0 * 8
    # Out-of-range depth
    @test_throws ArgumentError hv[:rho, (3, 0)]
    @test_throws ArgumentError hv[:rho, (1, 1), 1]   # |off|=2 > 1
end

@testset "HaloView: periodic boundary wraps" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d()
    # All-periodic on x; reflecting on y. Cell 3 is the bottom-right cell;
    # its (1, 0) neighbor should wrap to cell 2 (the bottom-left, on the
    # opposite x-wall).
    bcs = FrameBoundaries(((PERIODIC, PERIODIC), (REFLECTING, REFLECTING)))
    hv = halo_view_multi((rho = fin.rho,), mesh, 3, 1; bcs = bcs)
    pv = hv[:rho, (1, 0)]
    @test pv !== nothing
    # Cell 2 has rho coefficients (2, 3) — the wrap should give us cell 2.
    @test pv[1] == 2.0
    @test pv[2] == 3.0
end

@testset "HaloView: reflecting boundary returns central cell" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d()
    bcs = FrameBoundaries(((REFLECTING, REFLECTING), (REFLECTING, REFLECTING)))
    # Cell 3 hits the +x wall. Reflecting → returns cell 3's own coefficients.
    hv = halo_view_multi((rho = fin.rho,), mesh, 3, 1; bcs = bcs)
    pv = hv[:rho, (1, 0)]
    @test pv !== nothing
    @test pv[1] == 3.0          # cell 3, coeff 0
    @test pv[2] == 4.0
end

@testset "HaloView: outflow boundary returns central cell" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d()
    bcs = FrameBoundaries(((OUTFLOW, OUTFLOW), (OUTFLOW, OUTFLOW)))
    hv = halo_view_multi((rho = fin.rho,), mesh, 3, 1; bcs = bcs)
    pv = hv[:rho, (1, 0)]
    @test pv !== nothing
    @test pv[1] == 3.0
end

@testset "HaloView: dirichlet/inflow returns nothing (PR-13 placeholder)" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d()
    for kind in (DIRICHLET, INFLOW)
        bcs = FrameBoundaries(((kind, kind), (kind, kind)))
        hv = halo_view_multi((rho = fin.rho,), mesh, 3, 1; bcs = bcs)
        @test hv[:rho, (1, 0)] === nothing
    end
end

@testset "HaloView: no-BC boundary returns nothing" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d()
    hv = halo_view_multi((rho = fin.rho,), mesh, 3, 1; bcs = nothing)
    # Cell 3 is on the +x wall (face_neighbors(3)[2] == 0).
    @test hv[:rho, (1, 0)] === nothing
    # Interior step OK (cell 3 -> -x -> cell 2)
    pv = hv[:rho, (-1, 0)]
    @test pv !== nothing
    @test pv[1] == 2.0
end

@testset "HaloView: hanging-node face returns the representative" begin
    # Build a 2D mesh with one coarse-fine face:
    #   refine root → 4 leaves (2,3,4,5)
    #   refine cell 2 → 4 fine leaves; cells 3,4,5 stay coarse
    # Cell 7 (the lower-right coarse cell at (0.5..1, 0..0.5)) now sees
    # cell 2's fine children (cells 4 and 6) on its -x face.
    # `face_neighbors(mesh, 7)[1]` is the lowest-indexed fine neighbour;
    # `face_fine_neighbors(mesh, 7, 1)` gives the full list.
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2])
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    fin = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    for i in 1:n
        fin.rho[i] = ntuple(k -> Float64(i + k - 1), nc)
    end

    nbs = face_neighbors(mesh, 7)
    rep = Int(nbs[1])
    @test rep != 0
    fines = face_fine_neighbors(mesh, 7, 1)
    @test length(fines) >= 2     # hanging-node face has >1 fine neighbours

    hv = halo_view_multi((rho = fin.rho,), mesh, 7, 1; bcs = nothing)
    pv = hv[:rho, (-1, 0)]
    @test pv !== nothing
    # The single-tuple-offset API returns the representative.
    @test pv[1] == fin.rho[rep][1]

    # The 4-arg :face_fine_neighbors form returns the full list of views.
    list = hv[:rho, :face_fine_neighbors, 1, 1]
    @test length(list) == length(fines)
    for k in eachindex(fines)
        @test list[k][1] == fin.rho[Int(fines[k])][1]
    end
end

@testset "HaloView: read-only contract" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d()
    hv = halo_view_multi((rho = fin.rho,), mesh, 3, 1; bcs = nothing)
    # No setindex! method exists for HaloView; any attempt to write must
    # fail with MethodError.
    new_coeffs = ntuple(k -> 0.0, nc)
    @test_throws MethodError (hv[:rho, (-1, 0)] = new_coeffs)
end

@testset "Backend determinism (smoke): CellView under both backends" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d()
    n = n_cells(mesh)

    function run(backend)
        # Reset fout
        for i in 1:n
            fout.rho[i] = ntuple(_ -> 0.0, nc)
            fout.u[i]   = ntuple(_ -> 0.0, nc)
        end
        kernel = function(i)
            cv = cell_view((rho = fin.rho, u = fin.u),
                           (rho = fout.rho, u = fout.u), frame, i)
            new_rho = ntuple(k -> 2.0 * cv[:rho][k], nc)
            new_u   = ntuple(k -> -1.0 * cv[:u][k], nc)
            cv[:rho] = new_rho
            cv[:u]   = new_u
        end
        parallel_foreach(backend, kernel, 1:n)
        # Snapshot
        snap_rho = [collect(fout.rho[i]) for i in 1:n]
        snap_u   = [collect(fout.u[i])   for i in 1:n]
        return (snap_rho, snap_u)
    end

    ref = run(Sequential())
    par = run(OhMyThreadsBackend(:dynamic))
    @test ref == par
end
