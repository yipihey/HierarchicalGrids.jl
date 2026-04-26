using Test
using HierarchicalGrids
using HierarchicalGrids.Quadrature
using HierarchicalGrids.Bases

@testset "Quadrature" begin

    @testset "1D Gauss-Legendre on [0, 1]" begin
        # Sum of weights = 1 (length of [0, 1])
        for n in 1:8
            q = gauss_quadrature_interval(n)
            @test sum(q.weights) ≈ 1.0 atol=1e-13
            @test n_quad_points(q) == n
            # All points in [0, 1]
            @test all(0.0 <= p[1] <= 1.0 for p in q.points)
        end
    end

    @testset "1D: integrates polynomials exactly up to degree 2n-1" begin
        # n-point GL is exact for polys of degree ≤ 2n-1
        # Test: ∫₀¹ x^k dx = 1/(k+1)
        for n in 1:6
            q = gauss_quadrature_interval(n)
            for k in 0:(2n-1)
                expected = 1.0 / (k + 1)
                computed = integrate(p -> p[1]^k, q)
                @test computed ≈ expected atol=1e-12
            end
        end
    end

    @testset "1D: 4-point rule integrates cubic exactly" begin
        # Hand-checked: ∫₀¹ (1 + 2x - 3x² + 4x³) dx = 1 + 1 - 1 + 1 = 2
        q = gauss_quadrature_interval(2)  # exact for ≤ degree 3
        result = integrate(p -> 1 + 2p[1] - 3p[1]^2 + 4p[1]^3, q)
        @test result ≈ 2.0 atol=1e-13
    end

    @testset "1D: integrates with Bases evaluations" begin
        # Use a monomial basis and integrate
        # p(x) = 1 + 2x + 3x² ⇒ ∫₀¹ p = 1 + 1 + 1 = 3
        b = MonomialBasis{1, 2}()
        coeffs = [1.0, 2.0, 3.0]
        q = gauss_quadrature_interval(3)
        result = integrate(pt -> evaluate(b, coeffs, pt), q)
        @test result ≈ 3.0 atol=1e-13
    end

    @testset "2D quad: tensor product is exact" begin
        # ∫∫ x^i y^j dx dy = 1/((i+1)(j+1))
        for n in 1:4
            q = gauss_quadrature_quad(n)
            @test sum(q.weights) ≈ 1.0 atol=1e-13
            @test n_quad_points(q) == n^2
            for i in 0:(2n-1), j in 0:(2n-1)
                expected = 1.0 / ((i+1) * (j+1))
                computed = integrate(p -> p[1]^i * p[2]^j, q)
                @test computed ≈ expected atol=1e-12
            end
        end
    end

    @testset "3D cube: tensor product" begin
        for n in 1:3
            q = gauss_quadrature_cube(n)
            @test sum(q.weights) ≈ 1.0 atol=1e-13
            @test n_quad_points(q) == n^3
            for i in 0:(2n-1), j in 0:(2n-1), k in 0:(2n-1)
                expected = 1.0 / ((i+1) * (j+1) * (k+1))
                computed = integrate(p -> p[1]^i * p[2]^j * p[3]^k, q)
                @test computed ≈ expected atol=1e-12
            end
        end
    end

    @testset "Triangle: weights sum to area = 1/2" begin
        for order in 1:7
            q = gauss_quadrature_triangle(order)
            @test sum(q.weights) ≈ 0.5 atol=1e-12
        end
    end

    @testset "Triangle: integrates monomials exactly" begin
        # For the reference triangle T = {x,y ≥ 0, x+y ≤ 1}:
        # ∫_T x^i y^j dx dy = i! j! / (i+j+2)!
        function triangle_monomial_exact(i, j)
            num = factorial(big(i)) * factorial(big(j))
            den = factorial(big(i + j + 2))
            return Float64(num // den)
        end

        for order in 1:7
            q = gauss_quadrature_triangle(order)
            for i in 0:order, j in 0:(order - i)
                expected = triangle_monomial_exact(i, j)
                computed = integrate(p -> p[1]^i * p[2]^j, q)
                @test computed ≈ expected atol=1e-10
            end
        end
    end

    @testset "Triangle: barycentric symmetry" begin
        # The constant function ≡ 1 integrates to 1/2
        for order in 1:7
            q = gauss_quadrature_triangle(order)
            @test integrate(_ -> 1.0, q) ≈ 0.5 atol=1e-13
        end
    end

    @testset "Triangle: integrates Bernstein basis to expected averages" begin
        # On the simplex: ∫_T B_α^P(λ) dλ = area(T) / binomial(d+P, P)
        # where area(T) = 1/d! for the standard d-simplex; here d=2 so area = 1/2.
        # Each Bernstein basis function integrates to (1/2) / binomial(2+P, P).
        # Cap at P=3 so 2P=6 is within our supported orders.
        for P in 1:3
            b = BernsteinBasis{2, P}()
            n = n_coeffs(b)
            q = gauss_quadrature_triangle(2P)  # exact for degree 2P
            expected_per_basis = 0.5 / binomial(2 + P, P)
            # Integrate each basis function (set one coeff to 1, others 0)
            for k in 1:n
                coeffs = zeros(Float64, n)
                coeffs[k] = 1.0
                computed = integrate(p -> evaluate(b, coeffs, p), q)
                @test computed ≈ expected_per_basis atol=1e-10
            end
        end
    end

    @testset "Tetrahedron: weights sum to volume = 1/6" begin
        for order in 1:3
            q = gauss_quadrature_tetrahedron(order)
            @test sum(q.weights) ≈ 1/6 atol=1e-13
        end
    end

    @testset "Tetrahedron: integrates monomials exactly" begin
        # For the reference tetrahedron T = {x,y,z ≥ 0, x+y+z ≤ 1}:
        # ∫_T x^i y^j z^k dx dy dz = i! j! k! / (i+j+k+3)!
        function tet_monomial_exact(i, j, k)
            num = factorial(big(i)) * factorial(big(j)) * factorial(big(k))
            den = factorial(big(i + j + k + 3))
            return Float64(num // den)
        end

        for order in 1:3
            q = gauss_quadrature_tetrahedron(order)
            for i in 0:order, j in 0:(order - i), k in 0:(order - i - j)
                expected = tet_monomial_exact(i, j, k)
                computed = integrate(p -> p[1]^i * p[2]^j * p[3]^k, q)
                @test computed ≈ expected atol=1e-10
            end
        end
    end

    @testset "Argument validation" begin
        @test_throws ArgumentError gauss_quadrature_interval(0)
        @test_throws ArgumentError gauss_quadrature_triangle(8)
        @test_throws ArgumentError gauss_quadrature_tetrahedron(4)
    end

    @testset "integrate_polynomial_on_subinterval (monomial basis)" begin
        # p(ξ) = 1 + 2ξ + 3ξ² in monomial basis on [0,1]
        # parent = [0, 2], so x ↦ ξ = x/2
        # The polynomial evaluated as a function of x: 1 + x + 0.75x²
        # ∫₀² (1 + x + 0.75x²) dx = 2 + 2 + 0.75·8/3 = 6.0
        basis = MonomialBasis{1, 2}()
        coeffs = (1.0, 2.0, 3.0)
        parent = Interval(0.0, 2.0)
        quad = gauss_quadrature_interval(3)  # exact for deg ≤ 5

        full = integrate_polynomial_on_subinterval(coeffs, basis, parent, parent, quad)
        @test full ≈ 6.0 atol=1e-12

        # Sub-interval [0.5, 1.5]
        sub = Interval(0.5, 1.5)
        partial = integrate_polynomial_on_subinterval(coeffs, basis, parent, sub, quad)
        # Antiderivative: x + x²/2 + 0.25 x³
        # F(1.5) - F(0.5) = (1.5 + 1.125 + 0.84375) - (0.5 + 0.125 + 0.03125)
        #                 = 3.46875 - 0.65625 = 2.8125
        @test partial ≈ 2.8125 atol=1e-12
    end

    @testset "integrate_polynomial_on_subinterval — empty sub returns zero" begin
        basis = MonomialBasis{1, 2}()
        coeffs = (1.0, 2.0, 3.0)
        parent = Interval(0.0, 2.0)
        quad = gauss_quadrature_interval(3)

        # Disjoint intersection produces an empty interval, which integrates to 0
        empty_sub = interval_intersection(parent, Interval(3.0, 4.0))
        result = integrate_polynomial_on_subinterval(coeffs, basis, parent, empty_sub, quad)
        @test result == 0.0
    end

    @testset "integrate_polynomial_on_subinterval — partition matches whole" begin
        # Splitting an interval into two pieces and summing the polynomial
        # integrals must equal the integral over the whole.
        basis = MonomialBasis{1, 3}()
        coeffs = (1.5, -0.7, 2.3, 0.4)  # arbitrary cubic
        parent = Interval(-1.0, 4.0)
        quad = gauss_quadrature_interval(4)  # exact for deg ≤ 7

        whole = integrate_polynomial_on_subinterval(coeffs, basis, parent, parent, quad)

        # Split at x = 1.5
        a = interval_intersection(parent, Interval(-1.0, 1.5))
        b = interval_intersection(parent, Interval(1.5, 4.0))
        sum_partial = integrate_polynomial_on_subinterval(coeffs, basis, parent, a, quad) +
                      integrate_polynomial_on_subinterval(coeffs, basis, parent, b, quad)
        @test sum_partial ≈ whole atol=1e-12
    end

    @testset "integrate_polynomial_on_subinterval (Bernstein basis) — partition of unity" begin
        # The sum of Bernstein basis functions of degree P on [0,1] is 1 identically.
        # So integrating sum_i b_i = 1 over a sub-interval gives its length.
        basis = BernsteinBasis{1, 3}()
        coeffs = ntuple(_ -> 1.0, n_coeffs(basis))  # all-ones coefficients ⇒ p(ξ) = 1
        parent = Interval(0.0, 5.0)
        quad = gauss_quadrature_interval(3)

        # ∫_sub 1 dx = length(sub)
        sub = Interval(1.0, 4.0)
        result = integrate_polynomial_on_subinterval(coeffs, basis, parent, sub, quad)
        @test result ≈ 3.0 atol=1e-12

        # Whole parent
        result_whole = integrate_polynomial_on_subinterval(coeffs, basis, parent, parent, quad)
        @test result_whole ≈ 5.0 atol=1e-12
    end

    @testset "integrate_polynomial_on_subinterval — argument validation" begin
        basis = MonomialBasis{1, 2}()
        quad = gauss_quadrature_interval(3)
        parent = Interval(0.0, 1.0)
        sub = Interval(0.0, 1.0)
        # Wrong number of coefficients
        @test_throws DimensionMismatch integrate_polynomial_on_subinterval(
            (1.0, 2.0), basis, parent, sub, quad)
        # Zero-length parent
        @test_throws ArgumentError integrate_polynomial_on_subinterval(
            (1.0, 2.0, 3.0), basis, Interval(1.0, 1.0), sub, quad)
    end

    @testset "action_error_l2 — basic difference of two functions" begin
        quad = gauss_quadrature_interval(4)
        # Two functions on [0,1]: f_p(x) = x; f_{p+1}(x) = x + 0.5 x²
        # Difference = 0.5 x²; (diff)² = 0.25 x⁴
        # ∫₀¹ 0.25 x⁴ dx = 0.05
        # sqrt(0.05) ≈ 0.2236067977...
        fp = pt -> pt[1]
        fpp = pt -> pt[1] + 0.5 * pt[1]^2
        err = action_error_l2(fp, fpp, quad)
        @test err ≈ sqrt(0.05) atol=1e-13
    end

    @testset "action_error_l2 — zero when reconstructions agree" begin
        quad = gauss_quadrature_interval(3)
        f = pt -> 1.0 + 2.0 * pt[1]
        @test action_error_l2(f, f, quad) ≈ 0.0 atol=1e-13
    end

    @testset "action_error_l2 — EL residual penalty" begin
        quad = gauss_quadrature_interval(4)
        f = pt -> pt[1]
        # Same function ⇒ no L² difference; only the EL residual contributes
        err = action_error_l2(f, f, quad; el_residual=0.7)
        @test err ≈ 0.7 atol=1e-13

        # Combined: sqrt(L²² + el_residual²)
        fp = pt -> pt[1]
        fpp = pt -> pt[1] + 0.5 * pt[1]^2
        # L² contribution = 0.05, EL² = 0.04, sum = 0.09
        err2 = action_error_l2(fp, fpp, quad; el_residual=0.2)
        @test err2 ≈ sqrt(0.09) atol=1e-13
    end

    @testset "action_error_l2 — 2D quadrature on triangle" begin
        # Use a triangle quadrature; verify the basic L² semantics still hold there.
        qtri = gauss_quadrature_triangle(3)
        # Two scalar fields: f_p(q) = q[1] + q[2]; f_{p+1}(q) = q[1] + q[2] + q[1]*q[2]
        # Difference = q[1]*q[2]; (diff)² = q[1]²·q[2]²
        # ∫_T x²y² dA on the unit triangle = 2! · 2! / (2+2+2)! = 4/720 = 1/180
        # sqrt(1/180) = 0.0745355...
        fp  = pt -> pt[1] + pt[2]
        fpp = pt -> pt[1] + pt[2] + pt[1]*pt[2]
        # Need higher-order rule for x²y² (degree 4); use order 4
        qtri_hi = gauss_quadrature_triangle(4)
        err = action_error_l2(fp, fpp, qtri_hi)
        @test err ≈ sqrt(1/180) atol=1e-12
    end

end
