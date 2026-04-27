using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells,
    EulerianFrame, FrameBoundaries, enumerate_leaves,
    PERIODIC, REFLECTING,
    SoA, allocate_point_sample_fields,
    point_multi_to_flat, eval_point_samples,
    Sequential, OhMyThreadsBackend,
    BlockView, BlockHaloView, block_view, block_halo_view, for_each_block!

# ----------------------------------------------------------------------------
# Setup: 4x4 leaf mesh on [0, 1]^2, point-sample fields (D=2, N=3).
# ----------------------------------------------------------------------------

function _setup_4x4_pts(; N = 3, field_names = (:rho,))
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, enumerate_leaves(mesh))
    n = n_cells(mesh)
    npts = N^2
    pairs = [name => Float64 for name in field_names]
    fin  = allocate_point_sample_fields(SoA(), Val(2), Val(N), n; pairs...)
    fout = allocate_point_sample_fields(SoA(), Val(2), Val(N), n; pairs...)
    for nm in field_names
        for i in 1:n
            getproperty(fin,  nm)[i] = ntuple(k -> Float64(i * 100 + k), npts)
            getproperty(fout, nm)[i] = ntuple(_ -> 0.0, npts)
        end
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    return (mesh, frame, fin, fout, N, npts)
end

_views(fs, names) = NamedTuple{names}(map(n -> getproperty(fs, n), names))

# ============================================================================
# 1. Multi-index write kernel: bv[Val(:rho), (i, j)] = i + j
# ============================================================================

@testset "for_each_block! Path B: multi-index write kernel" begin
    (mesh, frame, fin, fout, N, npts) = _setup_4x4_pts()
    fin_v  = _views(fin,  (:rho,))
    fout_v = _views(fout, (:rho,))

    kernel = function (bv, bhv, ctx)
        for j in 1:N, i in 1:N
            bv[Val(:rho), (i, j)] = Float64(i + j)
        end
        return nothing
    end

    for_each_block!(kernel, fout_v, fin_v, frame; backend = Sequential())

    for c in enumerate_leaves(mesh)
        for j in 1:N, i in 1:N
            flat = point_multi_to_flat(Val(N), (i, j))
            @test fout.rho[c][flat] == Float64(i + j)
        end
    end
end

# ============================================================================
# 2. Backend determinism — five backends produce byte-equal output
# ============================================================================

@testset "for_each_block! Path B: backend determinism (5 backends)" begin
    (mesh, frame, fin, fout, N, npts) = _setup_4x4_pts()
    fin_v = _views(fin, (:rho,))

    function run(backend)
        fout_local = allocate_point_sample_fields(SoA(), Val(2), Val(N),
                                                    n_cells(mesh); rho = Float64)
        for i in 1:n_cells(mesh)
            fout_local.rho[i] = ntuple(_ -> 0.0, npts)
        end
        fout_v = _views(fout_local, (:rho,))
        kernel = function (bv, bhv, ctx)
            v = bv(Val(:rho), (0.25, 0.75))      # interpolated value
            for j in 1:N, i in 1:N
                bv[Val(:rho), (i, j)] = 7.0 * v - 3.0 * (i + j)
            end
            return nothing
        end
        for_each_block!(kernel, fout_v, fin_v, frame; backend = backend)
        return [collect(fout_local.rho[i]) for i in 1:n_cells(mesh)]
    end

    ref = run(Sequential())
    for backend in (OhMyThreadsBackend(:dynamic),
                    OhMyThreadsBackend(:static),
                    OhMyThreadsBackend(:greedy),
                    OhMyThreadsBackend(:serial))
        snap = run(backend)
        @test snap == ref       # byte-equal
    end
end

# ============================================================================
# 3. bv(Val(:rho), ξ) interpolates against eval_point_samples reference
# ============================================================================

@testset "for_each_block! Path B: bv(:rho, ξ) interpolation" begin
    # Fill cell i, point (a, b) with the value f((ξ_a, ξ_b)) = ξ_a + 2*ξ_b
    # (a degree-1 polynomial; tensor-Lagrange at any ξ recovers exactly).
    N = 3
    nodes = (0.0, 0.5, 1.0)
    f((x, y)) = x + 2 * y

    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    n = n_cells(mesh)
    fin  = allocate_point_sample_fields(SoA(), Val(2), Val(N), n;
                                          rho = Float64, interp = Float64)
    fout = allocate_point_sample_fields(SoA(), Val(2), Val(N), n;
                                          rho = Float64, interp = Float64)
    for i in 1:n
        for b in 1:N, a in 1:N
            ξ = (nodes[a], nodes[b])
            flat = point_multi_to_flat(Val(N), (a, b))
            fin.rho[i][flat] = f(ξ)
            fin.interp[i][flat] = 0.0
        end
        fout.rho[i] = ntuple(_ -> 0.0, N^2)
        fout.interp[i] = ntuple(_ -> 0.0, N^2)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))

    fin_v  = _views(fin,  (:rho, :interp))
    fout_v = _views(fout, (:rho, :interp))

    test_ξ = (0.25, 0.5)
    kernel = function (bv, bhv, ctx)
        v = bv(Val(:rho), test_ξ)
        # Store the interpolated value at the (1, 1) node of :interp.
        bv[Val(:interp), (1, 1)] = v
        return nothing
    end
    for_each_block!(kernel, fout_v, fin_v, frame; backend = Sequential())

    expected = f(test_ξ)
    for c in enumerate_leaves(mesh)
        flat = point_multi_to_flat(Val(N), (1, 1))
        @test isapprox(fout.interp[c][flat], expected; atol = 1e-12)
        # And cross-check against the explicit eval_point_samples reference.
        @test fout.interp[c][flat] ==
              eval_point_samples(Val(2), Val(N),
                                  ntuple(k -> fin.rho[c][k], N^2), test_ξ)
    end
end

# ============================================================================
# 4. bhv(Val(:rho), (1, 0), ξ) — neighbor evaluation under periodic wrap
# ============================================================================

@testset "for_each_block! Path B: bhv periodic neighbor eval" begin
    # 4x4 leaf mesh, fully-periodic. Each cell i has constant point-sample
    # value `i` (cell-id field). The kernel reads its +x neighbor's value at
    # (0.0, 0.5) and stores it at the (1, 1) slot of :rho_nb in fout.
    # Periodic wrap at the +x boundary should resolve to the cell on the
    # opposite +x wall.
    (mesh, frame, fin, fout, N, npts) = _setup_4x4_pts(field_names = (:rho, :rho_nb))
    # Re-init :rho with cell-constant values (not the i*100+k pattern).
    n = n_cells(mesh)
    for i in 1:n
        fin.rho[i] = ntuple(_ -> Float64(i), npts)
        fin.rho_nb[i] = ntuple(_ -> 0.0, npts)
        fout.rho[i] = ntuple(_ -> 0.0, npts)
        fout.rho_nb[i] = ntuple(_ -> 0.0, npts)
    end
    fin_v  = _views(fin,  (:rho, :rho_nb))
    fout_v = _views(fout, (:rho, :rho_nb))
    bcs = FrameBoundaries(((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)))

    kernel = function (bv, bhv, ctx)
        v_plus_x = bhv(Val(:rho), (1, 0), (0.0, 0.5))
        @assert v_plus_x !== nothing            # full-periodic ⇒ resolves
        bv[Val(:rho_nb), (1, 1)] = v_plus_x
        return nothing
    end

    for_each_block!(kernel, fout_v, fin_v, frame;
                     ghost_depth = 1, bcs = bcs, backend = Sequential())

    # Cross-check: every leaf's expected value is the +x neighbor's
    # cell-constant rho — exactly what `bhv` resolved to.
    for c in enumerate_leaves(mesh)
        flat = point_multi_to_flat(Val(3), (1, 1))
        bhv = block_halo_view(fin_v, mesh, c, 1; bcs = bcs)
        expected = bhv(Val(:rho), (1, 0), (0.0, 0.5))
        @test fout.rho_nb[c][flat] == expected
    end

    # And: bhv[Val(:rho), (1, 0), (i, j)] gives the same neighbor's point
    # value (since the neighbor's :rho is constant, this equals the
    # neighbor's cell index).
    for c in enumerate_leaves(mesh)
        bhv = block_halo_view(fin_v, mesh, c, 1; bcs = bcs)
        v_idx = bhv[Val(:rho), (1, 0), (2, 2)]
        v_eval = bhv(Val(:rho), (1, 0), (0.5, 0.5))
        @test v_idx !== nothing
        @test v_idx == v_eval
    end
end

# ============================================================================
# 5. Path A still rejects multi-index access with a clear message
# ============================================================================

@testset "BlockView Path A: multi-index access errors clearly" begin
    using HierarchicalGrids: MonomialBasis, allocate_polynomial_fields, n_coeffs
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    fin  = allocate_polynomial_fields(SoA(), basis, n; rho = Float64)
    fout = allocate_polynomial_fields(SoA(), basis, n; rho = Float64)
    for i in 1:n
        fin.rho[i]  = ntuple(k -> Float64(i + k - 1), nc)
        fout.rho[i] = ntuple(_ -> 0.0, nc)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    bv = block_view((rho = fin.rho,), (rho = fout.rho,), frame, 1)

    err_get = nothing
    try
        bv[Val(:rho), (1, 1)]
    catch e
        err_get = e
    end
    @test err_get isa ArgumentError
    @test occursin("polynomial blocks", err_get.msg)
    @test occursin("Path A", err_get.msg)

    err_set = nothing
    try
        bv[Val(:rho), (1, 1)] = 0.0
    catch e
        err_set = e
    end
    @test err_set isa ArgumentError
end
