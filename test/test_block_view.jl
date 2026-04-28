using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells, level_of,
    cell_physical_box, EulerianFrame, FrameBoundaries, enumerate_leaves,
    BCKind, PERIODIC, REFLECTING, OUTFLOW, DIRICHLET, INFLOW,
    MonomialBasis, BernsteinBasis, n_coeffs, evaluate,
    allocate_polynomial_fields, SoA,
    face_neighbors, ensure_neighbor_graph!,
    BlockView, BlockHaloView, block_view, block_halo_view, ghost_depth

# 4x4 leaf mesh on [0,1]^2 with rho field initialized so that cell `i`
# has polynomial f_i(x, y) = i + (i+1)*x + (i+2)*y under MonomialBasis{2,1}.
function _setup2d_4x4_blocks(; basis = MonomialBasis{2, 1}())
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, [2, 3, 4, 5])
    n = n_cells(mesh)
    nc = n_coeffs(basis)
    fin  = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    fout = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    for i in 1:n
        fin.rho[i] = ntuple(k -> Float64(i + k - 1), nc)
        fout.rho[i] = ntuple(_ -> 0.0, nc)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    return (mesh, frame, fin, fout, basis, nc)
end

# 2x2 leaf mesh — small smoke for BC tests.
function _setup2d_blocks()
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    fin  = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    fout = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    for i in 1:n
        fin.rho[i] = ntuple(k -> Float64(i + k - 1), nc)
        fout.rho[i] = ntuple(_ -> 0.0, nc)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    return (mesh, frame, fin, fout, basis, nc)
end

# ============================================================================
# 1. BlockView coefficient read/write
# ============================================================================

@testset "BlockView: coefficient read/write" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_4x4_blocks()
    bv = block_view((rho = fin.rho,), (rho = fout.rho,), frame, 8)
    @test bv isa BlockView
    @test bv.index == 8

    pv = bv[Val(:rho)]
    @test pv[1] == 8.0
    @test pv[2] == 9.0
    @test pv[3] == 10.0

    new_rho = ntuple(k -> Float64(1000 + k), nc)
    bv[Val(:rho)] = new_rho
    for k in 1:nc
        @test fout.rho[8][k] == new_rho[k]
    end
    @test fin.rho[8][1] == 8.0   # input untouched

    # Symbol-key form too.
    bv2 = block_view((rho = fin.rho,), (rho = fout.rho,), frame, 9)
    pv2 = bv2[:rho]
    @test pv2[1] == 9.0
    bv2[:rho] = ntuple(_ -> 7.0, nc)
    @test fout.rho[9][1] == 7.0
end

# ============================================================================
# 2. BlockView point evaluation against a known polynomial
# ============================================================================

@testset "BlockView: point evaluation matches the polynomial" begin
    # Set rho on cell `i` to f(ξ) = ξ[1] + 2*ξ[2] (constant across cells).
    # MonomialBasis{2, 1} ordering: (1, x, y), so coefficients are (0, 1, 2).
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    fin  = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    fout = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    for i in 1:n
        fin.rho[i]  = (0.0, 1.0, 2.0)
        fout.rho[i] = (0.0, 0.0, 0.0)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))

    bv = block_view((rho = fin.rho,), (rho = fout.rho,), frame, 4)
    # f((0.5, 0.5)) = 0 + 1*0.5 + 2*0.5 = 1.5
    @test bv(Val(:rho), (0.5, 0.5)) == 1.5
    # f((0.0, 0.0)) = 0
    @test bv(Val(:rho), (0.0, 0.0)) == 0.0
    # f((1.0, 0.0)) = 1
    @test bv(Val(:rho), (1.0, 0.0)) == 1.0
    # f((0.0, 1.0)) = 2
    @test bv(Val(:rho), (0.0, 1.0)) == 2.0

    # Symbol-key form
    @test bv(:rho, (0.25, 0.25)) == 0.75
end

# ============================================================================
# 3. BlockView metadata
# ============================================================================

@testset "BlockView: metadata accessors" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_4x4_blocks()
    for i in [4, 8, 12]
        bv = block_view((rho = fin.rho,), (rho = fout.rho,), frame, i)
        p_lo, p_hi = cell_physical_box(frame, i)
        expected_coords = ntuple(d -> (p_lo[d] + p_hi[d]) / 2, 2)
        expected_vol = (p_hi[1] - p_lo[1]) * (p_hi[2] - p_lo[2])
        @test bv.coords == expected_coords
        @test bv.volume ≈ expected_vol
        @test bv.level == Int(level_of(mesh, i))
        @test bv.index == i
        @test bv.basis === basis
        @test bv.degree == 1                # MonomialBasis{2, 1}
        @test bv.n_coeffs == nc
    end

    # propertynames lists the new properties
    bv = block_view((rho = fin.rho,), (rho = fout.rho,), frame, 4)
    pnames = propertynames(bv)
    for nm in (:fields_in, :fields_out, :index, :coords, :volume, :level,
                :basis, :degree, :n_coeffs)
        @test nm in pnames
    end
end

# ============================================================================
# 4. Type stability of point evaluation
# ============================================================================

@testset "BlockView: type stability of point eval" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_4x4_blocks()
    bv = block_view((rho = fin.rho,), (rho = fout.rho,), frame, 8)

    f = bv -> bv(Val(:rho), (0.5, 0.5))
    f(bv)            # warm
    @inferred f(bv)
    @test typeof(f(bv)) === Float64
end

# ============================================================================
# 5. Allocation: warm point eval is zero bytes
# ============================================================================

@testset "BlockView: zero-allocation point evaluation" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_4x4_blocks()
    bv = block_view((rho = fin.rho,), (rho = fout.rho,), frame, 8)

    # Wrap the call in a function so the global-variable capture path
    # doesn't show up as an allocation.
    function repeat_eval(bv, n)
        s = 0.0
        for _ in 1:n
            s += bv(Val(:rho), (0.5, 0.5))
        end
        return s
    end
    repeat_eval(bv, 1)        # warm
    a = @allocated repeat_eval(bv, 100)
    # Julia 1.10 leaves a small residual on this point-eval path.
    if VERSION >= v"1.11"
        @test a == 0
    else
        @test a < 96
    end
end

# ============================================================================
# 6. BlockHaloView point evaluation cross-check against neighbor's basis
# ============================================================================

@testset "BlockHaloView: point evaluation matches neighbor's polynomial" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_4x4_blocks()
    # cell 4's +x neighbor is cell 8 (fin.rho[8] = (8, 9, 10)).
    nbs = face_neighbors(mesh, 4)
    @test nbs[2] == 8

    bhv = block_halo_view((rho = fin.rho,), mesh, 4, 1; bcs = nothing)
    @test bhv isa BlockHaloView
    @test ghost_depth(bhv) == 1

    # Evaluate the +x neighbor at ξ = (0.0, 0.5) in the neighbor's frame.
    v = bhv(Val(:rho), (1, 0), (0.0, 0.5))
    @test v !== nothing
    # cell 8's polynomial under MonomialBasis{2, 1} is
    #   f(x, y) = 8 + 9*x + 10*y, so f(0.0, 0.5) = 8 + 0 + 5 = 13.
    @test v == 13.0
    # Cross-check against `evaluate(basis, neighbor_coeffs, ξ)` directly.
    nb_coeffs = ntuple(k -> fin.rho[8][k], nc)
    @test v == evaluate(basis, nb_coeffs, (0.0, 0.5))

    # Symbol-key form.
    @test bhv(:rho, (1, 0), (0.0, 0.5)) == 13.0

    # And the coefficient-read form still works.
    pv = bhv[Val(:rho), (1, 0)]
    @test pv !== nothing
    @test pv[1] == 8.0
end

# ============================================================================
# 7. BlockHaloView BC handling matrix (mirrors HaloView in PR-6)
# ============================================================================

@testset "BlockHaloView: PERIODIC wraps" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_blocks()
    bcs = FrameBoundaries(((PERIODIC, PERIODIC), (REFLECTING, REFLECTING)))
    bhv = block_halo_view((rho = fin.rho,), mesh, 3, 1; bcs = bcs)
    pv = bhv[Val(:rho), (1, 0)]
    @test pv !== nothing
    @test pv[1] == 2.0      # wrap to cell 2
    # Point eval through the wrap.
    v = bhv(Val(:rho), (1, 0), (0.5, 0.5))
    nb_coeffs = ntuple(k -> fin.rho[2][k], nc)
    @test v == evaluate(basis, nb_coeffs, (0.5, 0.5))
end

@testset "BlockHaloView: REFLECTING/OUTFLOW returns central block" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_blocks()
    for kind in (REFLECTING, OUTFLOW)
        bcs = FrameBoundaries(((kind, kind), (kind, kind)))
        bhv = block_halo_view((rho = fin.rho,), mesh, 3, 1; bcs = bcs)
        pv = bhv[Val(:rho), (1, 0)]
        @test pv !== nothing
        @test pv[1] == 3.0      # cell 3's own coeff_0
        # Point eval falls back to central block.
        v = bhv(Val(:rho), (1, 0), (0.5, 0.5))
        own = ntuple(k -> fin.rho[3][k], nc)
        @test v == evaluate(basis, own, (0.5, 0.5))
    end
end

@testset "BlockHaloView: DIRICHLET/INFLOW returns nothing (PR-13 placeholder)" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_blocks()
    for kind in (DIRICHLET, INFLOW)
        bcs = FrameBoundaries(((kind, kind), (kind, kind)))
        bhv = block_halo_view((rho = fin.rho,), mesh, 3, 1; bcs = bcs)
        @test bhv[Val(:rho), (1, 0)] === nothing
        @test bhv(Val(:rho), (1, 0), (0.5, 0.5)) === nothing
    end
end

@testset "BlockHaloView: no-BC boundary returns nothing" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_blocks()
    bhv = block_halo_view((rho = fin.rho,), mesh, 3, 1; bcs = nothing)
    @test bhv[Val(:rho), (1, 0)] === nothing
    @test bhv(Val(:rho), (1, 0), (0.0, 0.5)) === nothing
    # Interior step still works.
    pv = bhv[Val(:rho), (-1, 0)]
    @test pv !== nothing
    @test pv[1] == 2.0
end

@testset "BlockHaloView: out-of-range offset throws" begin
    (mesh, frame, fin, fout, basis, nc) = _setup2d_4x4_blocks()
    bhv = block_halo_view((rho = fin.rho,), mesh, 3, 1; bcs = nothing)
    @test_throws ArgumentError bhv[Val(:rho), (2, 0)]
    @test_throws ArgumentError bhv(Val(:rho), (2, 0), (0.0, 0.5))
end

# ============================================================================
# 8. BlockHaloView: BernsteinBasis path (covers the second @generated method)
# ============================================================================

@testset "BlockView: point eval with BernsteinBasis" begin
    # Build a simple 2x2 mesh with a Bernstein P=1 polynomial. Bernstein P=1
    # on the 2-simplex has 3 coefficients (one per vertex of the triangle);
    # the basis sums to one for any barycentric coordinate. With all-ones
    # coefficients, the polynomial is identically 1.
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    n = n_cells(mesh)
    basis = BernsteinBasis{2, 1}()
    nc = n_coeffs(basis)
    @test nc == 3
    fin  = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    fout = allocate_polynomial_fields(SoA(), basis, n; rho=Float64)
    for i in 1:n
        fin.rho[i] = ntuple(_ -> 1.0, nc)
        fout.rho[i] = ntuple(_ -> 0.0, nc)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))

    bv = block_view((rho = fin.rho,), (rho = fout.rho,), frame, 3)
    @test bv.basis isa BernsteinBasis
    # All-ones coeffs ⇒ Bernstein partition of unity ⇒ value 1 anywhere.
    @test bv(Val(:rho), (0.25, 0.25)) ≈ 1.0
    @test bv(Val(:rho), (0.0, 0.0)) ≈ 1.0

    # Cross-check against `evaluate` directly.
    coeffs = ntuple(k -> fin.rho[3][k], nc)
    @test bv(Val(:rho), (0.25, 0.25)) == evaluate(basis, coeffs, (0.25, 0.25))
end
