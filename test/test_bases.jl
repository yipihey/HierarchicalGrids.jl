using Test
using Random
using HierarchicalGrids
using HierarchicalGrids.Bases
using HierarchicalGrids.Bases: monomial_exponents, bernstein_multiindices,
                               multinomial_coefficient, is_positive_certificate,
                               _bernstein_to_monomial_matrix,
                               _monomial_to_bernstein_matrix

@testset "Bases" begin

    @testset "MonomialBasis: n_coeffs" begin
        @test n_coeffs(MonomialBasis{1, 0}()) == 1
        @test n_coeffs(MonomialBasis{1, 1}()) == 2
        @test n_coeffs(MonomialBasis{1, 3}()) == 4
        @test n_coeffs(MonomialBasis{2, 0}()) == 1
        @test n_coeffs(MonomialBasis{2, 1}()) == 3
        @test n_coeffs(MonomialBasis{2, 2}()) == 6
        @test n_coeffs(MonomialBasis{2, 3}()) == 10
        @test n_coeffs(MonomialBasis{3, 2}()) == 10
    end

    @testset "MonomialBasis: exponent ordering" begin
        # 1D, P=3: [1, x, x², x³]
        e = monomial_exponents(1, 3)
        @test e == [(0,), (1,), (2,), (3,)]

        # 2D, P=2: [1, x, y, x², xy, y²]
        e2 = monomial_exponents(2, 2)
        @test length(e2) == 6
        @test e2[1] == (0, 0)         # 1
        @test (1, 0) in e2[2:3]       # x or y at degree 1
        @test (0, 1) in e2[2:3]
        # Degree 2: x², xy, y²
        @test (2, 0) in e2[4:6]
        @test (1, 1) in e2[4:6]
        @test (0, 2) in e2[4:6]

        # 3D, P=1
        e3 = monomial_exponents(3, 1)
        @test length(e3) == 4
        @test e3[1] == (0, 0, 0)
    end

    @testset "MonomialBasis: evaluate constant polynomial" begin
        b = MonomialBasis{1, 3}()
        # p(x) = 5 (constant); coeffs = [5, 0, 0, 0]
        for x in 0.0:0.2:1.0
            @test evaluate(b, [5.0, 0.0, 0.0, 0.0], (x,)) == 5.0
        end

        b2 = MonomialBasis{2, 2}()
        coeffs = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # constant 3
        @test evaluate(b2, coeffs, (0.5, 0.3)) == 3.0
    end

    @testset "MonomialBasis: evaluate linear polynomial" begin
        # p(x) = 2 + 3x ⇒ coeffs = [2, 3, 0, 0]
        b = MonomialBasis{1, 3}()
        for x in 0.0:0.2:1.0
            @test evaluate(b, [2.0, 3.0, 0.0, 0.0], (x,)) ≈ 2.0 + 3.0 * x
        end
    end

    @testset "MonomialBasis: evaluate cubic polynomial" begin
        # p(x) = 1 - 2x + 3x² - 4x³
        b = MonomialBasis{1, 3}()
        coeffs = [1.0, -2.0, 3.0, -4.0]
        for x in 0.0:0.1:1.0
            expected = 1.0 - 2x + 3x^2 - 4x^3
            @test evaluate(b, coeffs, (x,)) ≈ expected
        end
    end

    @testset "MonomialBasis: evaluate 2D polynomial" begin
        # p(x, y) = 1 + 2x + 3y + 4x² + 5xy + 6y²
        b = MonomialBasis{2, 2}()
        e = monomial_exponents(2, 2)
        coeff_for_exp = Dict((0,0)=>1.0, (1,0)=>2.0, (0,1)=>3.0,
                              (2,0)=>4.0, (1,1)=>5.0, (0,2)=>6.0)
        coeffs = [coeff_for_exp[ex] for ex in e]

        for x in 0.0:0.25:1.0, y in 0.0:0.25:1.0
            expected = 1 + 2x + 3y + 4x^2 + 5x*y + 6y^2
            @test evaluate(b, coeffs, (x, y)) ≈ expected
        end
    end

    @testset "MonomialBasis: gradient 1D" begin
        # p(x) = 1 + 2x + 3x² ⇒ p'(x) = 2 + 6x
        b = MonomialBasis{1, 2}()
        coeffs = [1.0, 2.0, 3.0]
        for x in 0.0:0.1:1.0
            g = gradient(b, coeffs, (x,))
            @test g[1] ≈ 2.0 + 6.0 * x
        end
    end

    @testset "MonomialBasis: gradient 2D" begin
        # p(x, y) = x² + xy + y²
        # ∂p/∂x = 2x + y; ∂p/∂y = x + 2y
        b = MonomialBasis{2, 2}()
        e = monomial_exponents(2, 2)
        coeff_for_exp = Dict((0,0)=>0.0, (1,0)=>0.0, (0,1)=>0.0,
                              (2,0)=>1.0, (1,1)=>1.0, (0,2)=>1.0)
        coeffs = [coeff_for_exp[ex] for ex in e]

        for x in 0.0:0.25:1.0, y in 0.0:0.25:1.0
            g = gradient(b, coeffs, (x, y))
            @test g[1] ≈ 2x + y atol=1e-12
            @test g[2] ≈ x + 2y atol=1e-12
        end
    end

    @testset "BernsteinBasis: n_coeffs and multi-indices" begin
        @test n_coeffs(BernsteinBasis{1, 2}()) == 3
        @test n_coeffs(BernsteinBasis{2, 3}()) == 10  # binomial(5, 3) = 10
        @test n_coeffs(BernsteinBasis{3, 2}()) == 10  # binomial(5, 2) = 10

        # Multi-indices for D=1, P=2: (α_0, α_1) with sum = 2
        mi = bernstein_multiindices(1, 2)
        @test length(mi) == 3
        @test (2, 0) in mi
        @test (1, 1) in mi
        @test (0, 2) in mi

        # Multi-indices for D=2, P=2: (α_0, α_1, α_2) with sum = 2
        mi2 = bernstein_multiindices(2, 2)
        @test length(mi2) == 6
        @test (2, 0, 0) in mi2
        @test (0, 2, 0) in mi2
        @test (0, 0, 2) in mi2
        @test (1, 1, 0) in mi2
        @test (1, 0, 1) in mi2
        @test (0, 1, 1) in mi2
    end

    @testset "BernsteinBasis: multinomial coefficients" begin
        @test multinomial_coefficient(2, (2, 0)) == 1   # 2!/(2! 0!)
        @test multinomial_coefficient(2, (1, 1)) == 2   # 2!/(1! 1!)
        @test multinomial_coefficient(3, (1, 1, 1)) == 6  # 3!/(1! 1! 1!)
        @test multinomial_coefficient(4, (2, 1, 1)) == 12  # 4!/(2! 1! 1!)
        @test multinomial_coefficient(0, (0, 0)) == 1
    end

    @testset "BernsteinBasis: partition of unity" begin
        # Sum of all Bernstein basis functions equals 1 everywhere
        # If all coefficients = 1, the polynomial = 1 everywhere
        for D in 1:2, P in 0:4
            b = BernsteinBasis{D, P}()
            n = n_coeffs(b)
            coeffs = ones(Float64, n)
            # Sample some interior points of the simplex
            test_points = if D == 1
                [(0.0,), (0.25,), (0.5,), (0.75,), (1.0,)]
            else  # D == 2
                [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                 (0.3, 0.3), (0.5, 0.25), (0.25, 0.5)]
            end
            for pt in test_points
                # Only test points inside or on the simplex
                if D == 1 || (pt[1] >= 0 && pt[2] >= 0 && pt[1] + pt[2] <= 1)
                    @test evaluate(b, coeffs, pt) ≈ 1.0 atol=1e-12
                end
            end
        end
    end

    @testset "BernsteinBasis: convex-hull / positivity property" begin
        # If all coefficients in [a, b], then polynomial values in [a, b]
        b = BernsteinBasis{1, 3}()
        coeffs = [2.0, 5.0, 3.0, 4.0]
        a, c = minimum(coeffs), maximum(coeffs)
        for x in 0.0:0.05:1.0
            v = evaluate(b, coeffs, (x,))
            @test a - 1e-12 <= v <= c + 1e-12
        end

        @test all_bernstein_coeffs_positive(coeffs)
        @test !all_bernstein_coeffs_positive([-1.0, 1.0, 1.0])
        @test !all_bernstein_coeffs_positive([0.0, 1.0, 1.0])  # zero is not positive
    end

    @testset "BernsteinBasis: 1D agrees with monomial conversion" begin
        # p(x) = 2 - x + 3x² in monomial form, P=2
        mono_coeffs = [2.0, -1.0, 3.0]
        bm = MonomialBasis{1, 2}()
        bb = BernsteinBasis{1, 2}()

        bern_coeffs = change_basis(bb, bm, mono_coeffs)

        # Both should evaluate to the same values
        for x in 0.0:0.05:1.0
            v_mono = evaluate(bm, mono_coeffs, (x,))
            v_bern = evaluate(bb, bern_coeffs, (x,))
            @test v_mono ≈ v_bern atol=1e-12
        end
    end

    @testset "Round-trip basis conversion 1D" begin
        # Mono → Bern → Mono should be identity
        for P in 0:6
            mono_coeffs = collect(1.0:Float64(P+1))
            bb = BernsteinBasis{1, P}()
            bm = MonomialBasis{1, P}()
            bern_coeffs = change_basis(bb, bm, mono_coeffs)
            mono_back = change_basis(bm, bb, bern_coeffs)
            @test mono_back ≈ mono_coeffs atol=1e-10
        end
    end

    @testset "BernsteinBasis: gradient 1D" begin
        # p(x) = 2 - x + 3x²; p'(x) = -1 + 6x
        # In Bernstein form on [0,1]:
        bm = MonomialBasis{1, 2}()
        bb = BernsteinBasis{1, 2}()
        mono_coeffs = [2.0, -1.0, 3.0]
        bern_coeffs = change_basis(bb, bm, mono_coeffs)

        for x in 0.05:0.1:0.95
            g = gradient(bb, bern_coeffs, (x,))
            @test g[1] ≈ -1.0 + 6.0 * x atol=1e-10
        end
    end

    @testset "BernsteinBasis: gradient 2D" begin
        # Constant polynomial has zero gradient
        b = BernsteinBasis{2, 2}()
        coeffs = ones(6) .* 7.0   # p ≡ 7
        for (x, y) in [(0.1, 0.2), (0.3, 0.3), (0.0, 0.5)]
            g = gradient(b, coeffs, (x, y))
            @test g[1] ≈ 0.0 atol=1e-10
            @test g[2] ≈ 0.0 atol=1e-10
        end
    end

    @testset "is_positive_certificate" begin
        # p(x) = 1 + x + x² is positive on [0, 1] AND has all-positive Bernstein coefficients
        bm = MonomialBasis{1, 2}()
        @test is_positive_certificate(bm, [1.0, 1.0, 1.0])

        # p(x) = (x - 0.5)² + 0.01 = 0.26 - x + x² is positive on [0,1] but the
        # Bernstein test is only SUFFICIENT, not necessary; here it returns false.
        # This is a known limitation of low-degree Bernstein certificates: degree
        # elevation can sharpen, but base degree fails. Document the behavior.
        @test !is_positive_certificate(MonomialBasis{1, 2}(), [0.26, -1.0, 1.0])

        # p(x) = -1 + x is not positive (e.g. at x=0)
        @test !is_positive_certificate(MonomialBasis{1, 1}(), [-1.0, 1.0])

        # p(x) = 2 + x is positive on [0,1] AND Bernstein coefficients all positive
        @test is_positive_certificate(MonomialBasis{1, 1}(), [2.0, 1.0])
    end

    @testset "LagrangeBasis: 1D construction" begin
        # P = 3 needs 4 nodes
        nodes = [0.0, 1/3, 2/3, 1.0]
        b = LagrangeBasis{1, 3}(nodes)
        @test n_coeffs(b) == 4

        # Wrong number of nodes throws
        @test_throws ArgumentError LagrangeBasis{1, 3}([0.0, 1.0])
    end

    @testset "LagrangeBasis: interpolation property" begin
        # At each node, Lagrange polynomial equals the coefficient there
        nodes = [0.0, 0.25, 0.75, 1.0]
        b = LagrangeBasis{1, 3}(nodes)
        coeffs = [1.0, 2.0, 3.0, 5.0]
        for k in 1:4
            v = evaluate(b, coeffs, (nodes[k],))
            @test v ≈ coeffs[k] atol=1e-12
        end
    end

    @testset "LagrangeBasis: matches polynomial through nodes" begin
        # p(x) = 1 + 2x + 3x² + 4x³ at 4 nodes
        nodes = [0.0, 1/3, 2/3, 1.0]
        coeffs = [1.0 + 2.0*x + 3.0*x^2 + 4.0*x^3 for x in nodes]
        b = LagrangeBasis{1, 3}(nodes)

        # Evaluation should match the polynomial at non-node points
        for x in 0.05:0.05:0.95
            v = evaluate(b, coeffs, (x,))
            @test v ≈ 1.0 + 2.0*x + 3.0*x^2 + 4.0*x^3 atol=1e-12
        end
    end

    @testset "LagrangeBasis: gradient 1D" begin
        # p(x) = x² ⇒ p'(x) = 2x
        nodes = [0.0, 0.5, 1.0]
        coeffs = [n^2 for n in nodes]
        b = LagrangeBasis{1, 2}(nodes)

        # Test at non-node points
        for x in 0.1:0.1:0.9
            g = gradient(b, coeffs, (x,))
            @test g[1] ≈ 2.0 * x atol=1e-12
        end

        # And at nodes
        for n in nodes
            g = gradient(b, coeffs, (n,))
            @test g[1] ≈ 2.0 * n atol=1e-12
        end
    end

    @testset "Edge cases: P=0 (constant)" begin
        bm = MonomialBasis{1, 0}()
        @test n_coeffs(bm) == 1
        @test evaluate(bm, [3.5], (0.7,)) == 3.5
        g = gradient(bm, [3.5], (0.7,))
        @test g == (0.0,)

        bb = BernsteinBasis{1, 0}()
        @test n_coeffs(bb) == 1
        @test evaluate(bb, [3.5], (0.7,)) ≈ 3.5
    end

    @testset "Argument validation" begin
        @test_throws ArgumentError MonomialBasis{0, 2}()
        @test_throws ArgumentError MonomialBasis{1, -1}()
        @test_throws ArgumentError BernsteinBasis{0, 2}()

        b = MonomialBasis{1, 2}()
        @test_throws DimensionMismatch evaluate(b, [1.0, 2.0], (0.5,))  # wrong length
    end

    # --------------------------------------------------------------------
    # D-dim simplex Bernstein ↔ Monomial change_basis (PR-4)
    # --------------------------------------------------------------------

    @testset "change_basis D-dim: round-trip identity" begin
        rng = MersenneTwister(0xb157)
        for D in 1:3, P in 1:3
            n = binomial(D + P, P)
            bern_coeffs = randn(rng, n)
            bb = BernsteinBasis{D, P}()
            bm = MonomialBasis{D, P}()
            mono = change_basis(bm, bb, bern_coeffs)
            bern_back = change_basis(bb, bm, mono)
            @test bern_back ≈ bern_coeffs atol=1e-10 rtol=1e-10
            # And the other direction
            mono_coeffs = randn(rng, n)
            bern2 = change_basis(bb, bm, mono_coeffs)
            mono_back = change_basis(bm, bb, bern2)
            @test mono_back ≈ mono_coeffs atol=1e-10 rtol=1e-10
        end
    end

    @testset "change_basis D-dim: polynomial evaluation parity" begin
        rng = MersenneTwister(0xc0ffee)
        for D in 1:3, P in 1:3
            n = binomial(D + P, P)
            bern_coeffs = randn(rng, n)
            bb = BernsteinBasis{D, P}()
            bm = MonomialBasis{D, P}()
            mono_coeffs = change_basis(bm, bb, bern_coeffs)
            # Sample interior points of the reference D-simplex.
            for _ in 1:8
                # Generate a random point inside {x : x_d ≥ 0, Σ x_d ≤ 1}.
                # Use a simple rejection / scaling approach: pick uniform
                # x in [0, 1]^D, scale by α/Σx_d so that α ≤ 1.
                pt_vec = rand(rng, D)
                s = sum(pt_vec)
                if s > 0.95
                    pt_vec .*= 0.9 / s   # contract into the interior
                end
                pt = NTuple{D, Float64}(pt_vec)
                v_bern = evaluate(bb, bern_coeffs, pt)
                v_mono = evaluate(bm, mono_coeffs, pt)
                @test isapprox(v_bern, v_mono; atol=1e-10, rtol=1e-10)
            end
        end
    end

    @testset "change_basis D-dim: matches existing 1D-specialized methods" begin
        # The D-generic transform must reproduce the 1D-specialized result
        # when D=1. We compare matrices directly to bypass any `dispatch`
        # ambiguity (the 1D-specialized methods take priority via dispatch).
        for P in 1:6
            M_dgen = _bernstein_to_monomial_matrix(Val(1), Val(P))
            # 1D-specialized formula: c_k = (P choose k) Σ_{i=0..k}
            #     (-1)^{k-i} (k choose i) b_i, i.e.
            # row k+1 has entries (P choose k)(-1)^{k-i}(k choose i) at i+1.
            M_ref = zeros(P + 1, P + 1)
            for k in 0:P, i in 0:k
                M_ref[k + 1, i + 1] = binomial(P, k) * (-1)^(k - i) * binomial(k, i)
            end
            @test maximum(abs.(M_dgen .- M_ref)) < 1e-12

            # Inverse direction
            Minv_dgen = _monomial_to_bernstein_matrix(Val(1), Val(P))
            Minv_ref = zeros(P + 1, P + 1)
            for i in 0:P, k in 0:i
                Minv_ref[i + 1, k + 1] = binomial(i, k) / binomial(P, k)
            end
            @test maximum(abs.(Minv_dgen .- Minv_ref)) < 1e-12
        end
    end

    @testset "change_basis D-dim: cache hit returns same matrix" begin
        # First call builds and caches; second call returns the cached
        # matrix object (same `===` identity). This guards against an
        # accidental rebuild on every call.
        bb = BernsteinBasis{2, 3}()
        bm = MonomialBasis{2, 3}()
        c = randn(MersenneTwister(7), n_coeffs(bb))
        # Prime the cache.
        _ = change_basis(bm, bb, c)
        M1 = _bernstein_to_monomial_matrix(Val(2), Val(3))
        M2 = _bernstein_to_monomial_matrix(Val(2), Val(3))
        @test M1 === M2  # same object identity ⇒ cached, not rebuilt

        Minv1 = _monomial_to_bernstein_matrix(Val(2), Val(3))
        Minv2 = _monomial_to_bernstein_matrix(Val(2), Val(3))
        @test Minv1 === Minv2
    end

end
