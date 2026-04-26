using Test
using HierarchicalGrids
using HierarchicalGrids.Initialization: init_field_from!

@testset "init_field_from!" begin

    # ------------------------------------------------------------------
    # Test 1: project a constant onto degree-2 Bernstein on [0, 1]
    # ------------------------------------------------------------------
    @testset "constant on [0,1] (degree-2 Bernstein, 1D)" begin
        eul = HierarchicalMesh{1}()  # single root cell
        frame = EulerianFrame(eul, (0.0,), (1.0,))
        basis = BernsteinBasis{1, 2}()
        fields = allocate_polynomial_fields(SoA(), basis, n_cells(eul); rho = Float64)

        init_field_from!(fields, frame, x -> 1.0)

        # All Bernstein coefficients of the constant 1 should be 1.
        @test fields.rho[1][1] ≈ 1.0 atol=1e-12
        @test fields.rho[1][2] ≈ 1.0 atol=1e-12
        @test fields.rho[1][3] ≈ 1.0 atol=1e-12
    end

    # ------------------------------------------------------------------
    # Test 2: project f(x) = x[1] onto degree-1 Bernstein with D=2
    # Linear Bernstein coefficients are the basis-function values at the
    # simplex corners; corners are (0,0), (1,0), (0,1), so f-values at
    # corners are 0, 1, 0 (and these should be the coefficients exactly).
    # ------------------------------------------------------------------
    @testset "linear corners on [0,1]^2 (degree-1 Bernstein, 2D)" begin
        eul = HierarchicalMesh{2}()
        frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
        basis = BernsteinBasis{2, 1}()
        fields = allocate_polynomial_fields(SoA(), basis, n_cells(eul); u = Float64)

        init_field_from!(fields, frame, x -> x[1])

        c = collect(fields.u[1])
        # Bernstein degree-1 multi-indices in (α_0, α_1, α_2) order: (1,0,0), (0,1,0), (0,0,1)
        # corresponding to corners λ_0=1 (origin), λ_1=1 (e_1), λ_2=1 (e_2).
        # f at those corners: 0, 1, 0.
        @test c[1] ≈ 0.0 atol=1e-12
        @test c[2] ≈ 1.0 atol=1e-12
        @test c[3] ≈ 0.0 atol=1e-12
    end

    # ------------------------------------------------------------------
    # Test 3: project f(x) = x^2 onto degree-2 Bernstein on [0,1]; should
    # recover the polynomial exactly.
    # ------------------------------------------------------------------
    @testset "polynomial recovery: x^2 on [0,1] (degree-2 Bernstein)" begin
        eul = HierarchicalMesh{1}()
        frame = EulerianFrame(eul, (0.0,), (1.0,))
        basis = BernsteinBasis{1, 2}()
        fields = allocate_polynomial_fields(SoA(), basis, n_cells(eul); u = Float64)

        f = x -> x[1]^2
        init_field_from!(fields, frame, f)

        # Evaluate the projected polynomial at multiple points (in cell-reference
        # coords ξ ∈ [0,1] which equals physical x for this single-cell test)
        # and compare to f.
        for ξ in (0.0, 0.1, 0.27, 0.5, 0.73, 0.95, 1.0)
            p = fields.u[1]((ξ,))
            @test isapprox(p, f((ξ,)), atol=1e-12)
        end
    end

    # ------------------------------------------------------------------
    # Test 4: refine HierarchicalMesh{1} once, project sin(2π x); per-cell
    # L² error should be small for degree-3 reconstruction.
    # ------------------------------------------------------------------
    @testset "multi-cell sinusoid (degree-3 Bernstein, 1D)" begin
        eul = HierarchicalMesh{1}()
        # Refine to 4 leaves: split root, then split both children, so each
        # leaf covers a quarter-period of sin(2π x). Cubic L² projection on
        # a quarter-period is excellent.
        refine_cells!(eul, [1])
        children = enumerate_leaves(eul)
        refine_cells!(eul, children)
        frame = EulerianFrame(eul, (0.0,), (1.0,))
        basis = BernsteinBasis{1, 3}()
        fields = allocate_polynomial_fields(SoA(), basis, n_cells(eul); u = Float64)

        f = x -> sin(2π * x[1])
        init_field_from!(fields, frame, f; quadrature_order = 12)

        # For each leaf cell, evaluate the projected polynomial vs f at a
        # set of quadrature-like sample points in physical coords and
        # check that the L²-style RMS error per cell is very small.
        leaves = enumerate_leaves(eul)
        @test length(leaves) == 4
        for j in leaves
            lo, hi = cell_physical_box(frame, j)
            h = hi[1] - lo[1]
            err2 = 0.0
            n_samp = 50
            for k in 0:(n_samp - 1)
                ξ = (k + 0.5) / n_samp
                x_phys = (lo[1] + h * ξ,)
                p_val = fields.u[j]((ξ,))
                f_val = f(x_phys)
                err2 += (p_val - f_val)^2
            end
            rms = sqrt(err2 / n_samp)
            # Cubic L² projection of sin on a quarter-period interval is very small.
            @test rms < 1e-3
        end

        # A non-leaf cell (the root) is also written but with garbage
        # geometry (it covers the whole domain); the helper writes to every
        # element of `field` (n_cells), which is fine — leaves are what
        # matter for downstream physics. Sanity: at least the leaves above
        # came out finite.
        for j in leaves
            for k in 1:length(fields.u[j])
                @test isfinite(fields.u[j][k])
            end
        end
    end

    # ------------------------------------------------------------------
    # Test 5: SimplicialMesh overload — exact recovery of a linear function.
    # ------------------------------------------------------------------
    @testset "SimplicialMesh: f(x) = x[1] + x[2] exact at degree ≥ 1" begin
        positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        mesh = SimplicialMesh{2, Float64}(positions, sv, sn)

        basis = BernsteinBasis{2, 1}()
        fields = allocate_polynomial_fields(SoA(), basis, n_simplices(mesh); u = Float64)

        f = x -> x[1] + x[2]
        init_field_from!(fields, mesh, f)

        # Bernstein degree-1 coefficients at the three vertices should match
        # f at those vertices exactly: 0, 1, 1.
        c = collect(fields.u[1])
        @test c[1] ≈ 0.0 atol=1e-12   # (α_0=1) → vertex 1 = (0,0)
        @test c[2] ≈ 1.0 atol=1e-12   # (α_1=1) → vertex 2 = (1,0)
        @test c[3] ≈ 1.0 atol=1e-12   # (α_2=1) → vertex 3 = (0,1)

        # And the projected polynomial agrees with f at interior points too.
        # Reference simplex point ξ = (0.25, 0.5) → physical x = anchor +
        # ξ_1 * edge_1 + ξ_2 * edge_2 = (0,0) + 0.25*(1,0) + 0.5*(0,1) = (0.25, 0.5)
        ξ = (0.25, 0.5)
        @test isapprox(fields.u[1](ξ), f((0.25, 0.5)); atol=1e-12)
    end

end
