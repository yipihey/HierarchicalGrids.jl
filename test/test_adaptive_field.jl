using Test
using HierarchicalGrids
using HierarchicalGrids: MonomialBasis, allocate_polynomial_fields, SoA,
                          HierarchicalMesh, refine_cells!, coarsen_cells!,
                          AdaptiveField, dispose!, n_cells, n_coeffs,
                          register_refinement_listener!,
                          unregister_refinement_listener!

@testset "AdaptiveField" begin

    @testset "Construction registers a listener" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = (1.0,)

        n_listeners_before = length(mesh._listeners)
        af = AdaptiveField(field, mesh)
        @test length(mesh._listeners) == n_listeners_before + 1
        @test af.disposed == false
        @test af.listener_handle != 0
        @test parent(af) === af.field

        # Cleanup
        dispose!(af)
    end

    @testset "Refine — constant prolongation (degree 0, D=2)" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = (3.5,)
        af = AdaptiveField(field, mesh)

        refine_cells!(mesh, [1])
        @test n_cells(mesh) == 5  # 1 parent + 4 children for D=2 isotropic
        @test af.field.n == 5

        # Parent (now non-leaf) retains its coefficient
        @test af.field.rho[1][1] == 3.5
        # Each child inherits the parent's value
        for child in 2:5
            @test af.field.rho[child][1] == 3.5
        end

        dispose!(af)
    end

    @testset "Coarsen — degree-0 average" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = (0.0,)
        af = AdaptiveField(field, mesh)

        refine_cells!(mesh, [1])
        # Set the four children to 2, 4, 6, 8.
        af.field.rho[2] = (2.0,)
        af.field.rho[3] = (4.0,)
        af.field.rho[4] = (6.0,)
        af.field.rho[5] = (8.0,)

        coarsen_cells!(mesh, [1])
        @test n_cells(mesh) == 1
        @test af.field.n == 1
        # (2+4+6+8)/4 = 5
        @test af.field.rho[1][1] == 5.0

        dispose!(af)
    end

    @testset "Refine then coarsen round-trip (degree 0, bit-exact)" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = (7.0,)
        af = AdaptiveField(field, mesh)

        refine_cells!(mesh, [1])
        coarsen_cells!(mesh, [1])
        # Each child got 7.0 by constant prolongation; mean of four 7s is 7.
        @test af.field.rho[1][1] === 7.0

        dispose!(af)
    end

    @testset "dispose! unregisters and stops field updates" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = (1.0,)
        af = AdaptiveField(field, mesh)

        n_before = length(mesh._listeners)
        dispose!(af)
        @test length(mesh._listeners) == n_before - 1
        @test af.disposed == true

        # Capture the (frozen) field n before refinement.
        frozen_n = af.field.n
        @test frozen_n == 1

        # Refining now should NOT touch the wrapped field, because the
        # listener has been unregistered.
        refine_cells!(mesh, [1])
        @test n_cells(mesh) == 5
        # af.field is still the original (n=1) PolynomialFieldSet.
        @test af.field.n == frozen_n

        # Calling dispose! again is a no-op.
        dispose!(af)
        @test af.disposed == true
    end

    @testset "Higher-degree coarsening — warning + constant-moment fallback" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 2}()  # degree 2 in 2D
        nc = n_coeffs(basis)
        @test nc > 1

        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = ntuple(_ -> 0.0, nc)
        af = AdaptiveField(field, mesh)

        refine_cells!(mesh, [1])

        # Give each child a distinct nontrivial coefficient vector.
        # Constant moments are 1, 1, 1, 1 -> average 1; higher moments
        # vary but should be zeroed.
        for i in 2:5
            af.field.rho[i] = ntuple(k -> k == 1 ? 1.0 : Float64(i + k), nc)
        end

        # The warning should fire on the first higher-degree coarsen.
        @test_logs (:warn, r"AdaptiveField") coarsen_cells!(mesh, [1])
        @test af.field.n == 1

        coeffs_after = collect(af.field.rho[1])
        @test coeffs_after[1] == 1.0          # constant moment averaged
        @test all(c == 0.0 for c in coeffs_after[2:end])  # rest zeroed

        dispose!(af)
    end

    @testset "Base.parent(af) returns the underlying field" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        af = AdaptiveField(field, mesh)
        @test parent(af) === af.field
        # After a refinement the wrapper is rebuilt; parent should track it.
        refine_cells!(mesh, [1])
        @test parent(af) === af.field
        dispose!(af)
    end

    @testset "Multi-field, multiple refinements (D=2)" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1;
                                            rho=Float64, p=Float64)
        field.rho[1] = (1.0,)
        field.p[1]   = (10.0,)
        af = AdaptiveField(field, mesh)

        refine_cells!(mesh, [1])
        @test af.field.n == 5
        @test all(af.field.rho[i][1] == 1.0 for i in 1:5)
        @test all(af.field.p[i][1]   == 10.0 for i in 1:5)

        # Refine one of the children (cell 3).
        refine_cells!(mesh, [3])
        @test af.field.n == n_cells(mesh)
        # All cells should still have the original values (constant
        # prolongation).
        for i in 1:n_cells(mesh)
            @test af.field.rho[i][1] == 1.0
            @test af.field.p[i][1]   == 10.0
        end

        dispose!(af)
    end

    @testset "AoS layout supported" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(HierarchicalGrids.AoS(), basis, 1;
                                            rho=Float64)
        field.rho[1] = (2.5,)
        af = AdaptiveField(field, mesh)
        refine_cells!(mesh, [1])
        @test af.field.n == 5
        for i in 1:5
            @test af.field.rho[i][1] == 2.5
        end
        coarsen_cells!(mesh, [1])
        @test af.field.n == 1
        @test af.field.rho[1][1] == 2.5
        dispose!(af)
    end

end
