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

    @testset "Layer 1: Face-neighbor graph (parallel build, PR-2)" begin
        include("test_neighbor_graph_threading.jl")
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

    @testset "Layer 2.5: init_field_from! (backend determinism)" begin
        include("test_init_field_from_threading.jl")
    end

    @testset "Layer 3 (foundational): Threading" begin
        include("test_threading.jl")
    end

    @testset "Layer 3 (foundational): Backend trait" begin
        include("test_backends.jl")
    end

    @testset "Layer 3 (foundational): Backend dispatch overhead" begin
        include("test_backend_dispatch.jl")
    end

    @testset "Layer 3 (foundational): refine_by_indicator! threading" begin
        include("test_refine_by_indicator_threading.jl")
    end

    @testset "Layer 3 (foundational): Memory" begin
        include("test_memory.jl")
    end

    @testset "Layer 3 (foundational): Hardware façade" begin
        include("test_hardware.jl")
    end

    @testset "Layer 3 (foundational): Diagnostics" begin
        include("test_diagnostics.jl")
    end

    @testset "Layer 3 (foundational): IntExact audit harness" begin
        include("test_exact_audit.jl")
    end

    @testset "Layer 3 (foundational): IntExact audit threading (PR-2)" begin
        include("test_audit_overlap_threading.jl")
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

    @testset "Layer 4: Overlap :exact backend drop audit" begin
        include("test_overlap_drop_audit.jl")
    end

    @testset "Layer 4: GeometricOverlap describe" begin
        include("test_overlap_describe.jl")
    end

    @testset "Layer 4: Overlap (EulerianFrame × EulerianFrame, PR-12)" begin
        include("test_inter_frame_overlap.jl")
    end

    @testset "Layer 4: FrameFaceCache (PR-2 face-list cache)" begin
        include("test_frame_face_cache.jl")
    end

    @testset "Layer 4: Physical-AABB cache" begin
        include("test_physical_box_cache.jl")
    end

    @testset "Layer 4: Polynomial remap" begin
        include("test_polynomial_remap.jl")
    end

    @testset "Layer 4: Polynomial remap (backend determinism)" begin
        include("test_polynomial_remap_threading.jl")
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

    @testset "Phase 2 (Solver): CellView and HaloView" begin
        include("test_views.jl")
    end

    @testset "Phase 2 (Solver): Orchestrators (PR-7)" begin
        include("test_orchestrators.jl")
    end

    @testset "Phase 2 (Solver): Orchestrators threading determinism (PR-7)" begin
        include("test_orchestrators_threading.jl")
    end

    @testset "Phase 2 (Solver): KernelContext (PR-9)" begin
        include("test_kernel_context.jl")
    end

    @testset "Phase 2 (Solver): BlockView (PR-10)" begin
        include("test_block_view.jl")
    end

    @testset "Phase 2 (Solver): for_each_block! (PR-10)" begin
        include("test_for_each_block.jl")
    end

    @testset "Layer 2.5: PointSampleFieldSet (PR-11)" begin
        include("test_point_sample_fieldset.jl")
    end

    @testset "Phase 2 (Solver): for_each_block! Path B (PR-11)" begin
        include("test_for_each_block_pointsample.jl")
    end

    @testset "Phase 2 (Solver): AdaptiveField (PR-8)" begin
        include("test_adaptive_field.jl")
    end

    @testset "Phase 2 (Solver): PatchHierarchy (PR-13)" begin
        include("test_patch_hierarchy.jl")
    end

    @testset "Phase 2 (Solver): GeometricMultigrid Poisson solver" begin
        include("test_geometric_multigrid.jl")
    end

    @testset "Phase 2 (Solver): ABec / variable-coefficient operator" begin
        include("test_abec_laplacian.jl")
    end

    @testset "Phase 2 (Solver): MAC projection" begin
        include("test_mac_projection.jl")
    end

    @testset "Phase 2 (Solver): Krylov.jl bridge" begin
        include("test_krylov_bridge.jl")
    end

    @testset "Phase 2 (Solver): AMG bottom" begin
        include("test_amg_bottom.jl")
    end

    @testset "Phase 2 (Solver): Radiation diffusion" begin
        include("test_radiation_diffusion.jl")
    end

    @testset "Phase 2 (Solver): Stiff chemistry" begin
        include("test_stiff_chemistry.jl")
    end

    @testset "Phase 2 (Solver): Node Laplacian" begin
        include("test_node_laplacian.jl")
    end

    @testset "Phase 2 (Solver): VectorABec" begin
        include("test_vector_abec.jl")
    end

    @testset "Phase 2 (Solver): Edge fields" begin
        include("test_edge_fields.jl")
    end

    @testset "Phase 2 (Solver): HYPRE bottom" begin
        include("test_hypre_bottom.jl")
    end

    @testset "Phase 2 (Solver): Tensor viscosity" begin
        include("test_tensor_op.jl")
    end

    @testset "Phase 2 (Solver): Curl-curl operator" begin
        include("test_curl_curl.jl")
    end

    @testset "Phase 2 (Solver): Multi-level node Laplacian" begin
        include("test_node_laplacian_ml.jl")
    end

    @testset "Integration tests" begin
        include("test_integration.jl")
    end

    @testset "Examples (smoke): cfd_block_diffusion" begin
        include("test_example_cfd_block_diffusion.jl")
    end

    @testset "Worked example: CFDPatchAMR (PR-16)" begin
        include("test_examples_cfd_patch_amr.jl")
    end

    @testset "Phase 2 worked example: cell-based advection (PR-14)" begin
        include("test_cfd_cell_advection.jl")
    end

    @testset "Phase 2 worked example: Sod-tube compressible (PR-3)" begin
        include("test_cfd_compressible_sod.jl")
    end

    @testset "Phase 2 worked example: Implicit Navier-Stokes (JFNK)" begin
        include("test_cfd_implicit_ns.jl")
    end

    @testset "Phase 2 worked example: Incompressible NS (CN + SDIRK2 JFNK)" begin
        include("test_cfd_incompressible_ns.jl")
    end

    @testset "Phase 2 worked example: 2nd-order AMR scalar advection" begin
        include("test_cfd_amr_advection_2o.jl")
    end
end
