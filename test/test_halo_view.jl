using Test
using HierarchicalGrids
using HierarchicalGrids: halo_view, HaloView, allocate_polynomial_fields,
    SoA, MonomialBasis, n_coeffs

@testset "HaloView 2D 2x2 grid: neighbor access and boundary" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()  # bilinear-ish on 2D
    pfs = allocate_polynomial_fields(SoA(), basis, n; density=Float64)

    # Fill with a deterministic per-cell value: coefficients of cell i are
    # (i, i+1, i+2, ...).
    nc = n_coeffs(basis)
    for i in 1:n
        pfs.density[i] = ntuple(k -> Float64(i + k - 1), nc)
    end

    hv = halo_view(pfs.density, mesh, 1)
    @test hv isa HaloView

    # Cell 2 is the lower-x lower-y child (sibling 0); right neighbor (axis 1
    # hi) is cell 3 (sibling 1).
    pv = hv[2, (1, 0)]
    @test pv !== nothing
    @test pv[1] == 3.0  # density coeff 1 of cell 3

    # Boundary: cell 2 has no left neighbor.
    @test hv[2, (-1, 0)] === nothing
    @test hv[2, (0, -1)] === nothing

    # Any leaf has at least one boundary face — sanity:
    for i in 2:5
        any_bnd = any((hv[i, off] === nothing) for off in ((-1,0),(1,0),(0,-1),(0,1)))
        @test any_bnd
    end
end

@testset "HaloView allocation cost (depth=1 fast path)" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    pfs = allocate_polynomial_fields(SoA(), basis, n; density=Float64)
    for i in 1:n
        pfs.density[i] = ntuple(k -> Float64(i), n_coeffs(basis))
    end
    hv = halo_view(pfs.density, mesh, 1)

    # Warm up: build the neighbor graph.
    _ = hv[2, (1, 0)]

    # Indexing should be allocation-free in the cached path.
    f = (hv, i) -> @inbounds hv[i, (1, 0)]
    f(hv, 2)  # warm
    a = @allocated f(hv, 2)
    # Allow a small number of bytes for the inevitable indirection through
    # the `nothing`-vs-PolynomialView union, but the bulk of the work
    # (the graph lookup) must not allocate.
    @test a <= 64
end

@testset "HaloView depth bound enforcement" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    basis = MonomialBasis{2, 1}()
    pfs = allocate_polynomial_fields(SoA(), basis, n_cells(mesh); density=Float64)
    for i in 1:n_cells(mesh)
        pfs.density[i] = ntuple(k -> 0.0, n_coeffs(basis))
    end
    hv = halo_view(pfs.density, mesh, 1)
    @test_throws ArgumentError hv[2, (2, 0)]
    @test_throws ArgumentError hv[2, (1, 1)]
    # Depth=1 allows offset (0,1), (1,0), etc.
    _ = hv[2, (1, 0)]
    _ = hv[2, (0, 1)]
end
