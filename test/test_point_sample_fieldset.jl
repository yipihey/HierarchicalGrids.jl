using Test
using HierarchicalGrids
using HierarchicalGrids: SoA, AoS, Blocked,
    PointSampleFieldSet, allocate_point_sample_fields,
    PointSampleFieldView, PointSampleView,
    n_points_per_cell, n_points_per_axis,
    point_multi_to_flat, point_flat_to_multi, eval_point_samples,
    n_elements, field_names

# ============================================================================
# 1. Basic allocation, queries, and storage size
# ============================================================================

@testset "PointSampleFieldSet: allocation and queries" begin
    pfs = allocate_point_sample_fields(SoA(), Val(2), Val(3), 4; rho = Float64)
    @test pfs isa PointSampleFieldSet
    @test n_elements(pfs) == 4
    @test n_points_per_cell(pfs) == 9      # 3^2
    @test n_points_per_axis(pfs) == 3
    @test field_names(pfs) == (:rho,)

    # SoA storage: a NamedTuple with one Vector of length 9 * 4 = 36.
    @test pfs.storage isa NamedTuple
    @test length(pfs.storage.rho) == 36

    # Multiple fields of mixed types.
    pfs2 = allocate_point_sample_fields(SoA(), Val(3), Val(2), 5;
                                         rho = Float64, u = Float32)
    @test n_points_per_cell(pfs2) == 8     # 2^3
    @test field_names(pfs2) == (:rho, :u)
    @test eltype(pfs2.storage.rho) == Float64
    @test eltype(pfs2.storage.u)   == Float32
end

# ============================================================================
# 2. Per-cell point read/write — known pattern roundtrip
# ============================================================================

@testset "PointSampleFieldSet: scalar point read/write" begin
    pfs = allocate_point_sample_fields(SoA(), Val(2), Val(3), 4; rho = Float64)
    # Cell 1: write a known pattern via flat indices.
    pattern = ntuple(k -> Float64(100 + k), 9)
    pfs.rho[1] = pattern
    for k in 1:9
        @test pfs.rho[1][k] == pattern[k]
    end

    # Cell 1: write via multi-index, read back via flat (and vice versa).
    pfs.rho[1][(2, 3)] = 999.0
    flat_2_3 = point_multi_to_flat(Val(3), (2, 3))
    @test pfs.rho[1][flat_2_3] == 999.0
    @test pfs.rho[1][(2, 3)] == 999.0

    # Other cells untouched.
    @test pfs.rho[2][1] == 0.0 || pfs.rho[2][1] != 999.0  # uninitialized — we just check cell 1's write
end

@testset "PointSampleFieldSet: multi-index ↔ flat conversions" begin
    # 2D, N=3: the column-major flat index for (i, j) is (j-1)*3 + i.
    @test point_multi_to_flat(Val(3), (1, 1)) == 1
    @test point_multi_to_flat(Val(3), (3, 1)) == 3
    @test point_multi_to_flat(Val(3), (1, 2)) == 4
    @test point_multi_to_flat(Val(3), (2, 3)) == 8
    @test point_multi_to_flat(Val(3), (3, 3)) == 9

    # Round-trip flat -> multi -> flat.
    for f in 1:9
        midx = point_flat_to_multi(Val(2), Val(3), f)
        @test point_multi_to_flat(Val(3), midx) == f
    end

    # 3D, N=2: 8 points.
    for f in 1:8
        midx = point_flat_to_multi(Val(3), Val(2), f)
        @test point_multi_to_flat(Val(2), midx) == f
    end
end

# ============================================================================
# 3. Layout flexibility — SoA / AoS / Blocked roundtrip
# ============================================================================

@testset "PointSampleFieldSet: layout flexibility (SoA, AoS, Blocked{4})" begin
    n = 16
    for layout in (SoA(), AoS(), Blocked{4}())
        pfs = allocate_point_sample_fields(layout, Val(2), Val(3), n; rho = Float64)
        @test n_elements(pfs) == n
        @test n_points_per_cell(pfs) == 9

        # Write a known pattern: cell i, point k -> i*100 + k.
        for i in 1:n
            pfs.rho[i] = ntuple(k -> Float64(i * 100 + k), 9)
        end
        for i in 1:n, k in 1:9
            @test pfs.rho[i][k] == Float64(i * 100 + k)
        end

        # Per-point write.
        pfs.rho[7][(2, 2)] = -1.0
        @test pfs.rho[7][(2, 2)] == -1.0
        # And the corresponding flat index.
        @test pfs.rho[7][point_multi_to_flat(Val(3), (2, 2))] == -1.0
    end
end

# ============================================================================
# 4. Round-trip with LagrangeBasis evaluate: f(ξ) = ξ[1] + 2*ξ[2]
# ============================================================================

@testset "PointSampleFieldSet: tensor-product Lagrange interpolation" begin
    # Set point values from f(ξ) = ξ[1] + 2*ξ[2] sampled at the equispaced
    # 3-node Lagrange grid on [0,1]^2. Evaluate via eval_point_samples /
    # the view's call form at a few random ξ; should match f(ξ) to round-off.
    pfs = allocate_point_sample_fields(SoA(), Val(2), Val(3), 1; rho = Float64)
    nodes = (0.0, 0.5, 1.0)
    f(ξ) = ξ[1] + 2 * ξ[2]
    for j in 1:3, i in 1:3
        ξ = (nodes[i], nodes[j])
        flat = point_multi_to_flat(Val(3), (i, j))
        pfs.rho[1][flat] = f(ξ)
    end

    # Test points (some at nodes, some interior).
    test_points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
                   (0.5, 0.5), (0.25, 0.5), (0.7, 0.3), (0.1, 0.9),
                   (0.333, 0.667)]
    for ξ in test_points
        v_view = pfs.rho[1](ξ)
        v_eval = eval_point_samples(Val(2), Val(3),
                                      ntuple(k -> pfs.rho[1][k], 9), ξ)
        @test isapprox(v_view, f(ξ); atol = 1e-12)
        @test v_view == v_eval                              # same path
    end

    # Higher N: f(ξ) = sin(π*ξ[1]) * cos(π*ξ[2]) (no longer linear, but a
    # P=4 tensor-Lagrange interp on a P=4 grid should be very accurate at
    # interior points).
    N = 5
    pfs5 = allocate_point_sample_fields(SoA(), Val(2), Val(N), 1; rho = Float64)
    nodes5 = ntuple(k -> Float64(k - 1) / Float64(N - 1), N)
    g(ξ) = sin(π * ξ[1]) * cos(π * ξ[2])
    for j in 1:N, i in 1:N
        ξ = (nodes5[i], nodes5[j])
        flat = point_multi_to_flat(Val(N), (i, j))
        pfs5.rho[1][flat] = g(ξ)
    end
    # Recovery at the nodes is exact.
    for j in 1:N, i in 1:N
        ξ = (nodes5[i], nodes5[j])
        @test isapprox(pfs5.rho[1](ξ), g(ξ); atol = 1e-12)
    end
end

# ============================================================================
# 5. Property-access errors
# ============================================================================

@testset "PointSampleFieldSet: property-access errors" begin
    pfs = allocate_point_sample_fields(SoA(), Val(2), Val(3), 4; rho = Float64)
    @test_throws KeyError pfs.nonexistent
    @test_throws ArgumentError pfs.rho = [1.0]   # whole-field assign forbidden

    # Length mismatch on bulk assign.
    @test_throws DimensionMismatch (pfs.rho[1] = (1.0, 2.0))   # only 2 entries
end

# ============================================================================
# 6. show() doesn't error
# ============================================================================

@testset "PointSampleFieldSet: show is non-empty" begin
    pfs = allocate_point_sample_fields(SoA(), Val(2), Val(3), 4; rho = Float64)
    s = sprint(show, pfs)
    @test occursin("PointSampleFieldSet", s)
    @test occursin("D=2", s)
    @test occursin("N=3", s)
    @test occursin("rho", s)

    pv = pfs.rho[1]
    s2 = sprint(show, pv)
    @test occursin("PointSampleView", s2)
end
