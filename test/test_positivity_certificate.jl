using Test
using HierarchicalGrids
using HierarchicalGrids.Bases: bernstein_positivity_certificate
using HierarchicalGrids.Storage: is_strictly_positive

@testset "Bernstein positivity certificate" begin

    @testset "All-positive coefficients return (true, nothing)" begin
        for D in 1:3, P in 1:3
            basis = BernsteinBasis{D, P}()
            n = n_coeffs(basis)
            coeffs = collect(1.0:Float64(n))
            positive, offending = bernstein_positivity_certificate(coeffs, basis)
            @test positive === true
            @test offending === nothing
        end
    end

    @testset "Single negative coefficient returns matching multi-index" begin
        for D in 1:3, P in 1:3
            basis = BernsteinBasis{D, P}()
            n = n_coeffs(basis)
            inds = HierarchicalGrids.Bases.bernstein_multiindices(D, P)
            # Try flipping each position to negative one at a time.
            for k in 1:n
                coeffs = collect(1.0:Float64(n))
                coeffs[k] = -1.0
                positive, offending = bernstein_positivity_certificate(coeffs, basis)
                @test positive === false
                # The certificate scans in-order and should report the first
                # failing index — which is k since prior coeffs are positive.
                @test offending == inds[k]
                @test length(offending) == D + 1
                @test sum(offending) == P
            end
        end
    end

    @testset "Tolerance treats tiny negatives as zero" begin
        basis = BernsteinBasis{2, 2}()
        n = n_coeffs(basis)
        coeffs = collect(1.0:Float64(n))
        coeffs[3] = -1e-12
        # With atol=1e-10, -1e-12 is within tolerance => (true, nothing)
        positive, offending = bernstein_positivity_certificate(coeffs, basis; atol = 1e-10)
        @test positive === true
        @test offending === nothing
        # Without tolerance, the same coeffs should fail.
        positive0, offending0 = bernstein_positivity_certificate(coeffs, basis)
        @test positive0 === false
        @test offending0 !== nothing
    end

    @testset "Zero coefficient fails strict positivity (atol=0)" begin
        basis = BernsteinBasis{1, 2}()
        coeffs = [1.0, 0.0, 1.0]
        positive, offending = bernstein_positivity_certificate(coeffs, basis)
        @test positive === false
        # Multi-index for the second basis function with D=1, P=2 is (1, 1).
        @test offending == (1, 1)
    end

    @testset "Field-level: 4-cell PolynomialFieldSet, cell 3 has α=(1,0,0) negative" begin
        # D=2, P=1 -> n_coeffs = 3, multi-indices: (1,0,0), (0,1,0), (0,0,1)
        basis = BernsteinBasis{2, 1}()
        pfs = allocate_polynomial_fields(SoA(), basis, 4; rho = Float64)
        # Initialize all cells with positive coefficients.
        for i in 1:4
            pfs.rho[i] = (1.0, 1.0, 1.0)
        end
        # Make cell 3 coeff at multi-index (1,0,0) negative (the 1st coeff).
        pfs.rho[3] = (-1.0, 1.0, 1.0)

        positive, offending = is_strictly_positive(pfs.rho)
        @test positive === false
        @test offending == (3, (1, 0, 0))
    end

    @testset "Field-level: all-positive returns (true, nothing)" begin
        basis = BernsteinBasis{2, 1}()
        pfs = allocate_polynomial_fields(SoA(), basis, 4; rho = Float64)
        for i in 1:4
            pfs.rho[i] = (1.0, 2.0, 3.0)
        end
        positive, offending = is_strictly_positive(pfs.rho)
        @test positive === true
        @test offending === nothing
    end

    @testset "Wrong-basis (Monomial) throws ArgumentError" begin
        basis = MonomialBasis{2, 1}()  # not Bernstein
        pfs = allocate_polynomial_fields(SoA(), basis, 4; rho = Float64)
        for i in 1:4
            pfs.rho[i] = (1.0, 1.0, 1.0)
        end
        @test_throws ArgumentError is_strictly_positive(pfs.rho)
    end
end
