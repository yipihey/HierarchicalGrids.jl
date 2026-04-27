using Test
using HierarchicalGrids

@testset "HierarchicalGrids.jl" begin
    @testset "Layer 0: BitPrimitives" begin
        include("test_bit_primitives.jl")
    end

    @testset "Layer 1: Mesh" begin
        include("test_mesh.jl")
    end

    @testset "Layer 1: SimplicialMesh" begin
        include("test_simplicial_mesh.jl")
    end

    @testset "Layer 1: CompositeMesh and refine_by_indicator!" begin
        include("test_composite_and_refinement.jl")
    end

    @testset "Layer 1: Refinement-event observer" begin
        include("test_refinement_listener.jl")
    end

    @testset "Layer 1: step_with_amr! driver" begin
        include("test_step_with_amr.jl")
    end

    @testset "Layer 1: Face-neighbor graph" begin
        include("test_neighbors.jl")
    end

    @testset "Layer 1: Cell-adjacency sparsity" begin
        include("test_adjacency_sparsity.jl")
    end

    @testset "Layer 1: Boundary conditions (PR-D)" begin
        include("test_boundary_conditions.jl")
    end

    @testset "Layer 1: Periodic SimplicialMesh (PR-D)" begin
        include("test_periodic_simplicial.jl")
    end

    @testset "Layer 2.5: Halo view" begin
        include("test_halo_view.jl")
    end

    @testset "Layer 2: Geometry" begin
        include("test_geometry.jl")
    end

    @testset "Layer 2: Bases" begin
        include("test_bases.jl")
    end

    @testset "Layer 2: Bernstein positivity certificate" begin
        include("test_positivity_certificate.jl")
    end

    @testset "Layer 2: Quadrature" begin
        include("test_quadrature.jl")
    end

    @testset "Layer 2.5: Storage" begin
        include("test_storage.jl")
    end

    @testset "Layer 2.5: PolynomialFieldSet" begin
        include("test_polynomial_fieldset.jl")
    end

    @testset "Layer 2.5: init_field_from!" begin
        include("test_init_field_from.jl")
    end

    @testset "Layer 3 (foundational): Threading" begin
        include("test_threading.jl")
    end

    @testset "Layer 3 (foundational): Memory" begin
        include("test_memory.jl")
    end

    @testset "Layer 3 (foundational): Diagnostics" begin
        include("test_diagnostics.jl")
    end

    @testset "Layer 3 (foundational): IntExact audit harness" begin
        include("test_exact_audit.jl")
    end

    @testset "Layer 4: Overlap" begin
        include("test_overlap.jl")
    end

    @testset "Layer 4: Overlap (1D)" begin
        include("test_overlap_1d.jl")
    end

    @testset "Layer 4: Overlap (3D)" begin
        include("test_overlap_3d.jl")
    end

    @testset "Layer 4: Overlap IntExact adapter (D=2,3)" begin
        include("test_overlap_int_adapter.jl")
    end

    @testset "Layer 4: IntegerLattice quantization helpers" begin
        include("test_quantize.jl")
    end

    @testset "Layer 4: Overlap IntExact adapter (D=4)" begin
        include("test_overlap_4d.jl")
    end

    @testset "Layer 4: Overlap :exact backend (compute_overlap)" begin
        include("test_overlap_exact_backend.jl")
    end

    @testset "Layer 4: Polynomial remap" begin
        include("test_polynomial_remap.jl")
    end

    @testset "Layer 4: Polynomial remap diagnostics" begin
        include("test_remap_diagnostics.jl")
    end

    @testset "Layer 4: Polynomial remap (FieldSet wrapper)" begin
        include("test_polynomial_remap_fieldset.jl")
    end

    @testset "Layer 4: Polynomial remap (streaming)" begin
        include("test_polynomial_remap_streaming.jl")
    end

    @testset "Integration tests" begin
        include("test_integration.jl")
    end
end
