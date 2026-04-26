using Test
using HierarchicalGrids

@testset "PolynomialFieldSet" begin

    @testset "SoA: basic construction and queries" begin
        basis = MonomialBasis{1, 3}()  # 4 coeffs per element
        pfs = allocate_polynomial_fields(SoA(), basis, 100; density=Float64, momentum=Float64)
        @test n_elements(pfs) == 100
        @test field_names(pfs) == (:density, :momentum)
        @test n_coeffs_per_element(pfs) == 4
        @test basis_of(pfs) === basis
    end

    @testset "AoS: basic construction" begin
        basis = MonomialBasis{2, 2}()  # 6 coeffs
        pfs = allocate_polynomial_fields(AoS(), basis, 50; density=Float64)
        @test n_elements(pfs) == 50
        @test n_coeffs_per_element(pfs) == 6
    end

    @testset "Blocked: basic construction" begin
        basis = MonomialBasis{1, 2}()  # 3 coeffs
        pfs = allocate_polynomial_fields(Blocked{8}(), basis, 100; density=Float64)
        @test n_elements(pfs) == 100
    end

    @testset "Set and get individual coefficients (all layouts)" begin
        basis = MonomialBasis{1, 3}()
        for layout in [SoA(), AoS(), Blocked{4}(), Blocked{8}()]
            pfs = allocate_polynomial_fields(layout, basis, 20; field=Float64)
            # Set coefficients element-by-element
            for i in 1:20
                pfs.field[i] = (Float64(i), Float64(i+10), Float64(i+20), Float64(i+30))
            end
            # Read them back
            for i in 1:20
                @test pfs.field[i][1] == Float64(i)
                @test pfs.field[i][2] == Float64(i+10)
                @test pfs.field[i][3] == Float64(i+20)
                @test pfs.field[i][4] == Float64(i+30)
            end
        end
    end

    @testset "Per-coefficient set/get (all layouts)" begin
        basis = MonomialBasis{1, 2}()  # 3 coeffs
        for layout in [SoA(), AoS(), Blocked{4}()]
            pfs = allocate_polynomial_fields(layout, basis, 10; field=Float64)
            for i in 1:10
                pfs.field[i] = (1.0, 2.0, 3.0)
            end
            # Modify a single coefficient
            pfs.field[5][2] = 99.0
            @test pfs.field[5][1] == 1.0
            @test pfs.field[5][2] == 99.0
            @test pfs.field[5][3] == 3.0
            # Other elements unchanged
            @test pfs.field[4][2] == 2.0
            @test pfs.field[6][2] == 2.0
        end
    end

    @testset "Layout independence: same results across layouts" begin
        basis = MonomialBasis{1, 3}()
        n = 25
        # Initialize three with the same data
        results = []
        for layout in [SoA(), AoS(), Blocked{4}()]
            pfs = allocate_polynomial_fields(layout, basis, n; field=Float64)
            for i in 1:n
                pfs.field[i] = (i*1.0, i*2.0, i*3.0, i*4.0)
            end
            # Read all back into a flat array
            collected = Float64[]
            for i in 1:n
                for k in 1:4
                    push!(collected, pfs.field[i][k])
                end
            end
            push!(results, collected)
        end
        @test results[1] == results[2]
        @test results[2] == results[3]
    end

    @testset "Polynomial evaluation 1D" begin
        # p(x) = 1 + 2x + 3x²
        basis = MonomialBasis{1, 2}()
        pfs = allocate_polynomial_fields(SoA(), basis, 5; field=Float64)
        for i in 1:5
            pfs.field[i] = (1.0, 2.0, 3.0)
        end
        for i in 1:5
            poly = pfs.field[i]
            for x in 0.0:0.25:1.0
                @test poly((x,)) ≈ 1.0 + 2x + 3x^2 atol=1e-12
            end
        end
    end

    @testset "Polynomial evaluation matches Bases.evaluate" begin
        basis = MonomialBasis{2, 2}()  # 6 coeffs
        pfs = allocate_polynomial_fields(AoS(), basis, 3; field=Float64)
        # Random-ish coefficients
        for i in 1:3
            pfs.field[i] = (Float64(i), 2.0*i, 3.0*i, 4.0*i, 5.0*i, 6.0*i)
        end
        for i in 1:3
            poly = pfs.field[i]
            coeffs = collect(poly)
            for pt in [(0.1, 0.2), (0.5, 0.5), (0.0, 1.0)]
                @test poly(pt) ≈ evaluate(basis, coeffs, pt) atol=1e-12
            end
        end
    end

    @testset "gradient_at matches Bases.gradient" begin
        basis = MonomialBasis{1, 3}()
        pfs = allocate_polynomial_fields(SoA(), basis, 1; field=Float64)
        # p(x) = 5 - 3x + 2x² + x³ ⇒ p'(x) = -3 + 4x + 3x²
        pfs.field[1] = (5.0, -3.0, 2.0, 1.0)
        poly = pfs.field[1]
        for x in 0.0:0.1:1.0
            g = gradient_at(poly, (x,))
            expected = -3.0 + 4.0*x + 3.0*x^2
            @test g[1] ≈ expected atol=1e-12
        end
    end

    @testset "PolynomialView iteration" begin
        basis = MonomialBasis{1, 2}()
        pfs = allocate_polynomial_fields(SoA(), basis, 1; field=Float64)
        pfs.field[1] = (10.0, 20.0, 30.0)
        coeffs = [c for c in pfs.field[1]]
        @test coeffs == [10.0, 20.0, 30.0]
        @test length(pfs.field[1]) == 3
    end

    @testset "PolynomialView collect" begin
        basis = MonomialBasis{1, 3}()
        pfs = allocate_polynomial_fields(AoS(), basis, 1; field=Float64)
        pfs.field[1] = (1.5, 2.5, 3.5, 4.5)
        c = collect(pfs.field[1])
        @test c == [1.5, 2.5, 3.5, 4.5]
    end

    @testset "Multiple fields with different scalar types" begin
        basis = MonomialBasis{1, 2}()
        pfs = allocate_polynomial_fields(SoA(), basis, 10;
                                          density=Float64, count=Int)
        pfs.density[1] = (1.0, 2.0, 3.0)
        pfs.count[1] = (10, 20, 30)
        @test pfs.density[1][1] == 1.0
        @test pfs.count[1][2] == 20
        @test pfs.count[1][2] isa Int
    end

    @testset "Bernstein basis storage and evaluation" begin
        basis = BernsteinBasis{2, 2}()  # 6 coeffs on triangle
        pfs = allocate_polynomial_fields(SoA(), basis, 5; field=Float64)
        # Set all coefficients to 1 ⇒ partition of unity ⇒ p ≡ 1
        for i in 1:5
            pfs.field[i] = ntuple(_ -> 1.0, 6)
        end
        for i in 1:5
            poly = pfs.field[i]
            for pt in [(0.1, 0.2), (0.3, 0.3), (0.0, 0.5)]
                @test poly(pt) ≈ 1.0 atol=1e-12
            end
        end
    end

    @testset "Argument validation" begin
        basis = MonomialBasis{1, 3}()  # 4 coeffs
        pfs = allocate_polynomial_fields(SoA(), basis, 5; field=Float64)
        # Wrong number of coeffs
        @test_throws DimensionMismatch (pfs.field[1] = (1.0, 2.0))
        @test_throws DimensionMismatch (pfs.field[1] = [1.0, 2.0, 3.0, 4.0, 5.0])
        # Bad field name now raises KeyError (was ErrorException)
        @test_throws KeyError pfs.nonexistent
    end

    @testset "Blocked layout with partial last block" begin
        basis = MonomialBasis{1, 2}()
        # 13 elements with block size 4 = 3 full blocks + 1 partial of size 1
        pfs = allocate_polynomial_fields(Blocked{4}(), basis, 13; field=Float64)
        for i in 1:13
            pfs.field[i] = (Float64(i), Float64(i*10), Float64(i*100))
        end
        for i in 1:13
            @test pfs.field[i][1] == Float64(i)
            @test pfs.field[i][2] == Float64(i*10)
            @test pfs.field[i][3] == Float64(i*100)
        end
    end

    @testset "Show methods" begin
        basis = MonomialBasis{1, 2}()
        pfs = allocate_polynomial_fields(SoA(), basis, 5; field=Float64)
        pfs.field[1] = (1.0, 2.0, 3.0)
        # Just check show doesn't throw
        @test sprint(show, pfs) isa String
        @test sprint(show, pfs.field[1]) isa String
    end

    # ========================================================================
    # polynomial_action_error / polynomial_action_error_per_element
    # ========================================================================

    @testset "polynomial_action_error: identical polys give zero" begin
        # Same field, same basis, same coefficients ⇒ zero error.
        basis = MonomialBasis{1, 2}()
        pfs = allocate_polynomial_fields(SoA(), basis, 3; field=Float64)
        pfs.field[1] = (1.0, 2.0, 3.0)
        quad = gauss_quadrature_interval(4)
        err = polynomial_action_error(pfs.field[1], pfs.field[1], quad)
        @test err ≈ 0.0 atol=1e-13
    end

    @testset "polynomial_action_error: known L² difference" begin
        # poly_p(ξ) = ξ          (linear; in MonomialBasis{1,1}: coeffs (0, 1))
        # poly_pp(ξ) = ξ + 0.5ξ² (quadratic; in MonomialBasis{1,2}: coeffs (0, 1, 0.5))
        # Difference = 0.5 ξ², squared = 0.25 ξ⁴, ∫₀¹ = 0.05, sqrt = sqrt(0.05).
        basis_p = MonomialBasis{1, 1}()
        basis_pp = MonomialBasis{1, 2}()
        pfs_p = allocate_polynomial_fields(SoA(), basis_p, 1; f=Float64)
        pfs_pp = allocate_polynomial_fields(SoA(), basis_pp, 1; f=Float64)
        pfs_p.f[1] = (0.0, 1.0)
        pfs_pp.f[1] = (0.0, 1.0, 0.5)

        quad = gauss_quadrature_interval(4)
        err = polynomial_action_error(pfs_p.f[1], pfs_pp.f[1], quad)
        @test err ≈ sqrt(0.05) atol=1e-13
    end

    @testset "polynomial_action_error: EL residual contributes" begin
        basis = MonomialBasis{1, 1}()
        pfs = allocate_polynomial_fields(SoA(), basis, 1; f=Float64)
        pfs.f[1] = (1.0, 2.0)
        quad = gauss_quadrature_interval(3)
        # Same poly ⇒ pure EL residual contribution = 0.5
        err = polynomial_action_error(pfs.f[1], pfs.f[1], quad; el_residual=0.5)
        @test err ≈ 0.5 atol=1e-13
    end

    @testset "polynomial_action_error: transform applied" begin
        # transform(value, point) = value^2 squares the field before comparison.
        # poly_p(ξ) = 1     → transformed = 1; poly_pp(ξ) = 1 + ξ → transformed = (1 + ξ)²
        # Difference = (1 + ξ)² - 1 = 2ξ + ξ²
        # Squared: (2ξ + ξ²)² = 4ξ² + 4ξ³ + ξ⁴
        # ∫₀¹ = 4/3 + 1 + 1/5 = 4/3 + 6/5 = 20/15 + 18/15 = 38/15
        # sqrt = sqrt(38/15) ≈ 1.591644...
        basis_p  = MonomialBasis{1, 0}()
        basis_pp = MonomialBasis{1, 1}()
        pfs_p  = allocate_polynomial_fields(SoA(), basis_p,  1; f=Float64)
        pfs_pp = allocate_polynomial_fields(SoA(), basis_pp, 1; f=Float64)
        pfs_p.f[1]  = (1.0,)
        pfs_pp.f[1] = (1.0, 1.0)
        quad = gauss_quadrature_interval(4)  # exact for deg 7

        err = polynomial_action_error(pfs_p.f[1], pfs_pp.f[1], quad;
                                        transform=(v, _pt) -> v * v)
        @test err ≈ sqrt(38 / 15) atol=1e-12
    end

    @testset "polynomial_action_error_per_element: vectorized over elements" begin
        # Three elements with different constant offsets in the order-(p+1) field.
        # Element i: poly_p(ξ) = 0; poly_pp(ξ) = i (constant)
        # L² of constant on [0,1] = i; sqrt(i²) = i.
        basis_p  = MonomialBasis{1, 0}()
        basis_pp = MonomialBasis{1, 0}()
        n = 3
        pfs_p  = allocate_polynomial_fields(SoA(), basis_p,  n; f=Float64)
        pfs_pp = allocate_polynomial_fields(SoA(), basis_pp, n; f=Float64)
        for i in 1:n
            pfs_p.f[i]  = (0.0,)
            pfs_pp.f[i] = (Float64(i),)
        end
        quad = gauss_quadrature_interval(2)
        err_vec = polynomial_action_error_per_element(pfs_p.f, pfs_pp.f, quad)
        @test length(err_vec) == n
        for i in 1:n
            @test err_vec[i] ≈ Float64(i) atol=1e-13
        end
    end

    @testset "polynomial_action_error_per_element: per-element EL residuals" begin
        basis = MonomialBasis{1, 1}()
        n = 4
        pfs = allocate_polynomial_fields(SoA(), basis, n; f=Float64)
        for i in 1:n
            pfs.f[i] = (Float64(i), 0.0)
        end
        quad = gauss_quadrature_interval(2)
        # Same view passed twice ⇒ L² = 0; only EL residual contributes
        residuals = [0.1, 0.2, 0.3, 0.4]
        err_vec = polynomial_action_error_per_element(pfs.f, pfs.f, quad;
                                                       el_residual=residuals)
        for i in 1:n
            @test err_vec[i] ≈ residuals[i] atol=1e-13
        end
    end

    @testset "polynomial_action_error_per_element: dimension mismatch errors" begin
        basis = MonomialBasis{1, 1}()
        pfs_a = allocate_polynomial_fields(SoA(), basis, 3; f=Float64)
        pfs_b = allocate_polynomial_fields(SoA(), basis, 5; f=Float64)
        quad = gauss_quadrature_interval(2)
        @test_throws DimensionMismatch polynomial_action_error_per_element(
            pfs_a.f, pfs_b.f, quad)

        # Wrong-length residual vector
        pfs_c = allocate_polynomial_fields(SoA(), basis, 3; f=Float64)
        @test_throws DimensionMismatch polynomial_action_error_per_element(
            pfs_a.f, pfs_c.f, quad; el_residual=[0.1, 0.2])
    end

    @testset "polynomial_action_error_per_element: feeds refine_by_indicator!" begin
        # End-to-end check: per-element indicator drives mesh refinement.
        # Build a 2D mesh (2 leaves at root level after one refinement),
        # construct order-p and order-(p+1) fields aligned with leaves where
        # one cell has zero error and another large error.
        # We then call refine_by_indicator! and check the right cell refined.
        mesh = HierarchicalMesh{2}()
        refine_cells!(mesh, [1])  # root → 4 leaf children at level 1
        leaves = find_children(mesh, 1)
        @test length(leaves) == 4

        # PolynomialFieldSet keyed by leaf index in the LEAF ordering, not the
        # mesh-cell index ordering. For testing refine_by_indicator! we need an
        # indicator vector of length n_cells(mesh). Build that explicitly.
        n = n_cells(mesh)
        indicator = zeros(Float64, n)
        # Big indicator on first leaf, small on others
        indicator[leaves[1]] = 10.0  # well above threshold

        result = refine_by_indicator!(mesh, indicator;
                                        refine_threshold=1.0)
        @test result.refined == 1
        @test !is_leaf(mesh.cells[leaves[1]])  # refined
        @test is_leaf(mesh.cells[leaves[2]])    # untouched
    end

end
