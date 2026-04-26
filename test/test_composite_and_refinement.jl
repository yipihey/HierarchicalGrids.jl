using Test
using HierarchicalGrids

@testset "CompositeMesh" begin

    @testset "Basic construction and access" begin
        mesh1 = HierarchicalMesh{2}()
        mesh2 = HierarchicalMesh{2}()

        composite = CompositeMesh((main = mesh1, refined = mesh2))
        @test composite.main === mesh1
        @test composite.refined === mesh2
        @test :main in propertynames(composite)
        @test :refined in propertynames(composite)
        # show should not throw
        @test sprint(show, composite) isa String
    end

    @testset "Mixed mesh types" begin
        eul = HierarchicalMesh{2}()
        positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        lag = SimplicialMesh{2, Float64}(positions, sv, sn)

        composite = CompositeMesh((lagrangian = lag, eulerian = eul))
        @test composite.lagrangian === lag
        @test composite.eulerian === eul
    end

end

@testset "PairedMesh" begin

    @testset "Basic construction" begin
        positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        lag = SimplicialMesh{2, Float64}(positions, sv, sn)
        eul = HierarchicalMesh{2}()

        paired = PairedMesh(lag, eul)
        @test paired.lagrangian === lag
        @test paired.eulerian === eul
        @test overlap_cache(paired) === nothing
    end

    @testset "Overlap compute and cache" begin
        positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        lag = SimplicialMesh{2, Float64}(positions, sv, sn)
        eul = HierarchicalMesh{2}()

        paired = PairedMesh(lag, eul)

        # ensure_overlap! should error before a function is registered
        @test_throws ArgumentError ensure_overlap!(paired)

        # Register a dummy overlap function
        compute_calls = Ref(0)
        set_overlap_compute_function!(paired) do pm
            compute_calls[] += 1
            return "overlap_data_$(compute_calls[])"
        end

        # First call computes
        ov = ensure_overlap!(paired)
        @test ov == "overlap_data_1"
        @test compute_calls[] == 1
        @test overlap_cache(paired) == "overlap_data_1"

        # Second call uses cache
        ov2 = ensure_overlap!(paired)
        @test ov2 == "overlap_data_1"
        @test compute_calls[] == 1  # not recomputed

        # Invalidate and recompute
        invalidate_overlap!(paired)
        @test overlap_cache(paired) === nothing
        ov3 = ensure_overlap!(paired)
        @test ov3 == "overlap_data_2"
        @test compute_calls[] == 2
    end

    @testset "update_lagrangian_positions! invalidates cache" begin
        positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        lag = SimplicialMesh{2, Float64}(positions, sv, sn)
        eul = HierarchicalMesh{2}()
        paired = PairedMesh(lag, eul)

        compute_calls = Ref(0)
        set_overlap_compute_function!(paired) do pm
            compute_calls[] += 1
            return "v$(compute_calls[])"
        end

        ensure_overlap!(paired)
        @test compute_calls[] == 1

        # Update positions
        new_positions = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]
        update_lagrangian_positions!(paired, new_positions)

        @test vertex_position(paired.lagrangian, 2) == (2.0, 0.0)
        @test overlap_cache(paired) === nothing  # invalidated

        ensure_overlap!(paired)
        @test compute_calls[] == 2
    end

    @testset "update_lagrangian_positions! length validation" begin
        positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        lag = SimplicialMesh{2, Float64}(positions, sv, sn)
        eul = HierarchicalMesh{2}()
        paired = PairedMesh(lag, eul)

        # Wrong length
        @test_throws ArgumentError update_lagrangian_positions!(paired,
            [(0.0, 0.0), (1.0, 0.0)])
    end

end

@testset "refine_by_indicator!" begin

    @testset "Basic refinement on flat indicator" begin
        # Start with a fresh mesh — single root cell
        mesh = HierarchicalMesh{2}()
        n_initial = n_cells(mesh)

        # Indicator: very high for cell 1
        indicator = fill(0.0, n_initial)
        indicator[1] = 100.0

        result = refine_by_indicator!(mesh, indicator;
                                      refine_threshold=10.0)
        @test result.refined == 1
        @test n_cells(mesh) > n_initial  # children added
    end

    @testset "Refinement only on cells above threshold" begin
        mesh = HierarchicalMesh{2}()
        # First, refine root manually so we have multiple cells
        refine_cells!(mesh, [1])
        n_after_first = n_cells(mesh)

        # Now flag only a couple of cells
        leaves = [ci for ci in 1:n_cells(mesh) if is_leaf(mesh.cells[ci])]
        n_leaves = length(leaves)
        indicator = zeros(n_cells(mesh))
        # High indicator for the first leaf only
        indicator[leaves[1]] = 50.0

        result = refine_by_indicator!(mesh, indicator; refine_threshold=10.0)
        @test result.refined == 1
    end

    @testset "max_level cap" begin
        mesh = HierarchicalMesh{2}()
        # Refine root explicitly
        refine_cells!(mesh, [1])
        # Now all leaves are at level 1.
        # Try to refine all of them with max_level = 1 — should refine none
        n_before = n_cells(mesh)
        indicator = fill(100.0, n_cells(mesh))
        result = refine_by_indicator!(mesh, indicator;
                                      refine_threshold=10.0, max_level=1)
        @test result.refined == 0
        @test n_cells(mesh) == n_before
    end

    @testset "Indicator as function" begin
        mesh = HierarchicalMesh{2}()
        # Function indicator: high for index 1
        n_before = n_cells(mesh)
        result = refine_by_indicator!(mesh,
                                      i -> i == 1 ? 100.0 : 0.0;
                                      refine_threshold=10.0)
        @test result.refined == 1
    end

    @testset "Indicator length mismatch" begin
        mesh = HierarchicalMesh{2}()
        @test_throws ArgumentError refine_by_indicator!(mesh,
            [1.0, 2.0, 3.0]; refine_threshold=0.5)  # wrong length
    end

end
