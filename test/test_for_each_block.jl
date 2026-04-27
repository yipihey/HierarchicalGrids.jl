using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells,
    EulerianFrame, FrameBoundaries, enumerate_leaves,
    PERIODIC, REFLECTING,
    MonomialBasis, n_coeffs, evaluate,
    allocate_polynomial_fields, SoA,
    Sequential, OhMyThreadsBackend,
    BlockView, BlockHaloView, block_view, block_halo_view, for_each_block!

# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------

# 4x4 leaf mesh on [0, 1]^2. Each cell carries a degree-1 polynomial with
# coefficients (i, i+1, i+2) (a known-pattern reference).
function _setup_4x4_blocks(; field_names = (:rho,))
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, enumerate_leaves(mesh))
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    pairs = [n => Float64 for n in field_names]
    fin  = allocate_polynomial_fields(SoA(), basis, n; pairs...)
    fout = allocate_polynomial_fields(SoA(), basis, n; pairs...)
    for nm in field_names
        for i in 1:n
            getproperty(fin, nm)[i]  = ntuple(k -> Float64(i + k - 1), nc)
            getproperty(fout, nm)[i] = ntuple(k -> 0.0, nc)
        end
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    return (mesh, frame, fin, fout, basis, nc)
end

function _views(fs, names)
    return NamedTuple{names}(map(n -> getproperty(fs, n), names))
end

# ============================================================================
# 1. Constant kernel: every block's :rho set to a known constant tuple
# ============================================================================

@testset "for_each_block!: constant kernel sets all leaves" begin
    (mesh, frame, fin, fout, basis, nc) = _setup_4x4_blocks()
    fin_v  = _views(fin, (:rho,))
    fout_v = _views(fout, (:rho,))

    kernel = function (bv, bhv, ctx)
        bv[Val(:rho)] = ntuple(_ -> 42.0, nc)
        return nothing
    end

    for_each_block!(kernel, fout_v, fin_v, frame; backend = Sequential())

    for i in enumerate_leaves(mesh)
        for k in 1:nc
            @test fout.rho[i][k] == 42.0
        end
    end
end

# ============================================================================
# 2. Point-eval kernel: cell-centered evaluation written as a degree-0 entry
# ============================================================================

@testset "for_each_block!: point-eval kernel writes value at center" begin
    (mesh, frame, fin, fout, basis, nc) = _setup_4x4_blocks()
    fin_v  = _views(fin, (:rho,))
    fout_v = _views(fout, (:rho,))

    # Cell `i` has polynomial f_i(x, y) = i + (i+1)*x + (i+2)*y.
    # f_i(0.5, 0.5) = i + 0.5*(i+1) + 0.5*(i+2) = 2*i + 1.5.
    kernel = function (bv, bhv, ctx)
        v = bv(Val(:rho), (0.5, 0.5))
        # Write to coefficient 1 only; zero the rest.
        bv[Val(:rho)] = ntuple(k -> k == 1 ? v : 0.0, nc)
        return nothing
    end

    for_each_block!(kernel, fout_v, fin_v, frame; backend = Sequential())

    for i in enumerate_leaves(mesh)
        @test fout.rho[i][1] == 2.0 * i + 1.5
        @test fout.rho[i][2] == 0.0
        @test fout.rho[i][3] == 0.0
    end
end

# ============================================================================
# 3. Backend determinism — five backends produce byte-identical output
# ============================================================================

@testset "for_each_block!: byte-equal determinism across all five backends" begin
    (mesh, frame, fin, fout, basis, nc) = _setup_4x4_blocks()
    fin_v = _views(fin, (:rho,))

    function run(backend)
        fout_local = allocate_polynomial_fields(SoA(), basis, n_cells(mesh); rho = Float64)
        for i in 1:n_cells(mesh)
            fout_local.rho[i] = ntuple(_ -> 0.0, nc)
        end
        fout_v = _views(fout_local, (:rho,))
        kernel = function (bv, bhv, ctx)
            v = bv(Val(:rho), (0.25, 0.75))
            bv[Val(:rho)] = ntuple(k -> 7.0 * v - 3.0 * k, nc)
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
# 4. Wide stencil with BlockHaloView point-eval — periodic wrap correctness
# ============================================================================

@testset "for_each_block!: 9-point stencil under fully-periodic BC" begin
    # 4x4 leaf mesh with fully-periodic BCs. The kernel reads 8 neighbors
    # via `bhv(:rho, off, ξ)` plus the center via `bv(:rho, ξ)` and writes
    # the sum into the output's coefficient-0 slot. Periodic wrap means
    # every neighbor read resolves; check the sum against direct evaluation
    # of every leaf's polynomial at ξ = (0.5, 0.5).
    (mesh, frame, fin, fout, basis, nc) = _setup_4x4_blocks()
    fin_v  = _views(fin, (:rho,))
    fout_v = _views(fout, (:rho,))
    bcs = FrameBoundaries(((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)))

    offs = ((-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1))

    kernel = function (bv, bhv, ctx)
        s = bv(Val(:rho), (0.5, 0.5))
        for off in offs
            v = bhv(Val(:rho), off, (0.5, 0.5))
            @assert v !== nothing      # full-periodic ⇒ all hops resolve
            s += v
        end
        bv[Val(:rho)] = ntuple(k -> k == 1 ? s : 0.0, nc)
        return nothing
    end

    for_each_block!(kernel, fout_v, fin_v, frame;
                     ghost_depth = 2, bcs = bcs, backend = Sequential())

    # Cross-check: the analytic answer at every leaf.
    # Each leaf's polynomial is f_i(ξ) = i + (i+1)*ξ_1 + (i+2)*ξ_2.
    # At ξ = (0.5, 0.5): f_i = i + 0.5*(i+1) + 0.5*(i+2) = 2i + 1.5.
    # The kernel's output is the SUM of the central leaf's value and 8
    # neighbor leaves' values. We mirror the periodic-walk via the same
    # halo-view machinery to compute the expected sum.
    leaves = enumerate_leaves(mesh)
    for c in leaves
        bhv = block_halo_view(fin_v, mesh, c, 2; bcs = bcs)
        bv  = block_view(fin_v, fout_v, frame, c)
        expected = bv(Val(:rho), (0.5, 0.5))
        for off in offs
            v = bhv(Val(:rho), off, (0.5, 0.5))
            @assert v !== nothing
            expected += v
        end
        @test fout.rho[c][1] ≈ expected
    end
end

# ============================================================================
# 5. Backend determinism with non-trivial BlockHaloView usage
# ============================================================================

@testset "for_each_block!: stencil + halo backend determinism" begin
    (mesh, frame, fin, fout, basis, nc) = _setup_4x4_blocks(field_names = (:rho, :u))
    # Initialize :u differently from :rho.
    for (i, _) in enumerate(1:n_cells(mesh))
        fin.u[i] = ntuple(k -> Float64(2 * i - k), nc)
    end
    fin_v = _views(fin, (:rho, :u))
    bcs = FrameBoundaries(((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)))

    function run(backend)
        fout_local = allocate_polynomial_fields(SoA(), basis, n_cells(mesh);
                                                  rho = Float64, u = Float64)
        for i in 1:n_cells(mesh)
            fout_local.rho[i] = ntuple(_ -> 0.0, nc)
            fout_local.u[i]   = ntuple(_ -> 0.0, nc)
        end
        fout_v = _views(fout_local, (:rho, :u))
        kernel = function (bv, bhv, ctx)
            ρ_c = bv(Val(:rho), (0.5, 0.5))
            ρ_e = bhv(Val(:rho), (1, 0),  (0.0, 0.5))   # left edge of +x neighbor
            ρ_w = bhv(Val(:rho), (-1, 0), (1.0, 0.5))   # right edge of -x neighbor
            u_c = bv(Val(:u), (0.5, 0.5))
            new_rho_0 = ρ_c - 0.1 * (u_c * (ρ_e - ρ_w))
            bv[Val(:rho)] = ntuple(k -> k == 1 ? new_rho_0 : 0.0, nc)
            bv[Val(:u)]   = ntuple(_ -> 0.0, nc)
            return nothing
        end
        for_each_block!(kernel, fout_v, fin_v, frame;
                         ghost_depth = 1, bcs = bcs, backend = backend)
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
# 6. ghost_depth = 0 default works (kernel ignores the halo)
# ============================================================================

@testset "for_each_block!: ghost_depth=0 default works" begin
    (mesh, frame, fin, fout, basis, nc) = _setup_4x4_blocks()
    fin_v  = _views(fin, (:rho,))
    fout_v = _views(fout, (:rho,))
    kernel = function (bv, bhv, ctx)
        bv[Val(:rho)] = ntuple(k -> -1.0 * (bv[Val(:rho)])[k], nc)
        return nothing
    end
    for_each_block!(kernel, fout_v, fin_v, frame; backend = Sequential())
    for i in enumerate_leaves(mesh)
        for k in 1:nc
            @test fout.rho[i][k] == -1.0 * (i + k - 1)
        end
    end
end
