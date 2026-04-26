using Test
using HierarchicalGrids

@testset "SimplicialMesh" begin

    @testset "1D: simple line of two segments" begin
        # Vertices at x=0, 1, 2; two segments: [v1,v2] and [v2,v3]
        positions = [(0.0,), (1.0,), (2.0,)]
        sv = Int32[1 2; 2 3]  # segments are columns
        sn = Int32[0 1; 2 0]  # segment 1's right neighbor is segment 2; segment 2's left is segment 1
        mesh = SimplicialMesh{1, Float64}(positions, sv, sn)

        @test n_vertices(mesh) == 3
        @test n_simplices(mesh) == 2
        @test spatial_dimension(mesh) == 1

        @test vertex_position(mesh, 1) == (0.0,)
        @test vertex_position(mesh, 3) == (2.0,)

        @test simplex_vertex_indices(mesh, 1) == (1, 2)
        @test simplex_vertex_indices(mesh, 2) == (2, 3)

        @test simplex_volume(mesh, 1) == 1.0
        @test simplex_volume(mesh, 2) == 1.0
        @test simplex_reference_volume(mesh, 1) == 1.0

        @test volume_jacobian(mesh, 1) == 1.0
    end

    @testset "1D: motion deforms simplex" begin
        positions = [(0.0,), (1.0,)]
        sv = reshape(Int32[1, 2], 2, 1)
        sn = reshape(Int32[0, 0], 2, 1)
        mesh = SimplicialMesh{1, Float64}(positions, sv, sn)

        @test simplex_volume(mesh, 1) == 1.0

        # Move vertex 2 to x = 3 → length becomes 3
        set_vertex_position!(mesh, 2, (3.0,))
        @test simplex_volume(mesh, 1) == 3.0
        @test simplex_reference_volume(mesh, 1) == 1.0  # ref unchanged
        @test volume_jacobian(mesh, 1) ≈ 3.0

        # Reset reference to current
        set_reference_to_current!(mesh)
        @test simplex_reference_volume(mesh, 1) == 3.0
        @test volume_jacobian(mesh, 1) ≈ 1.0
    end

    @testset "2D: single triangle" begin
        positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        mesh = SimplicialMesh{2, Float64}(positions, sv, sn)

        @test n_simplices(mesh) == 1
        # Standard reference triangle has area 1/2
        @test simplex_volume(mesh, 1) ≈ 0.5
        @test volume_jacobian(mesh, 1) ≈ 1.0

        # All faces are boundary
        for k in 1:3
            @test is_boundary_face(mesh, 1, k)
            @test simplex_neighbor(mesh, 1, k) == 0
        end
    end

    @testset "2D: deformation gradient identity" begin
        positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        mesh = SimplicialMesh{2, Float64}(positions, sv, sn)

        F = deformation_gradient(mesh, 1)
        @test F[1][1] ≈ 1.0
        @test F[1][2] ≈ 0.0
        @test F[2][1] ≈ 0.0
        @test F[2][2] ≈ 1.0
    end

    @testset "2D: deformation gradient pure shear" begin
        # Reference: standard triangle
        # Current: shear x → x + 0.5*y
        positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        ref = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        mesh = SimplicialMesh{2, Float64}(positions, sv, sn; reference_positions=ref)

        F = deformation_gradient(mesh, 1)
        # Expected F = [1 0.5; 0 1]
        @test F[1][1] ≈ 1.0
        @test F[1][2] ≈ 0.5
        @test F[2][1] ≈ 0.0
        @test F[2][2] ≈ 1.0
        # det F = 1 ⇒ volume preserved
        @test volume_jacobian(mesh, 1) ≈ 1.0
    end

    @testset "2D: deformation gradient pure dilation" begin
        # Triangle scaled by 2 in x, 3 in y
        ref = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        cur = [(0.0, 0.0), (2.0, 0.0), (0.0, 3.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        mesh = SimplicialMesh{2, Float64}(cur, sv, sn; reference_positions=ref)

        F = deformation_gradient(mesh, 1)
        @test F[1][1] ≈ 2.0
        @test F[1][2] ≈ 0.0
        @test F[2][1] ≈ 0.0
        @test F[2][2] ≈ 3.0
        @test volume_jacobian(mesh, 1) ≈ 6.0
    end

    @testset "2D: 2-triangle mesh with shared edge" begin
        # Square [0,1]² split into two triangles
        positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        # Triangle 1: vertices 1, 2, 3 (lower-right)
        # Triangle 2: vertices 1, 3, 4 (upper-left)
        sv = Int32[1 1; 2 3; 3 4]
        # Neighbors: triangle 1's "opposite vertex 1" face (between v2-v3) is boundary;
        # opposite vertex 2 face (between v1-v3) is shared with triangle 2;
        # opposite vertex 3 face (between v1-v2) is boundary.
        sn = Int32[0 0; 2 1; 0 0]
        mesh = SimplicialMesh{2, Float64}(positions, sv, sn)

        @test n_simplices(mesh) == 2
        @test simplex_volume(mesh, 1) ≈ 0.5
        @test simplex_volume(mesh, 2) ≈ 0.5

        @test simplex_neighbor(mesh, 1, 2) == 2
        @test simplex_neighbor(mesh, 2, 2) == 1
    end

    @testset "Edge enumeration" begin
        positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        sv = Int32[1 1; 2 3; 3 4]
        sn = Int32[0 0; 2 1; 0 0]
        mesh = SimplicialMesh{2, Float64}(positions, sv, sn)

        edges = enumerate_edges(mesh)
        @test length(edges) == 5  # square: 4 boundary + 1 diagonal
        # All v1 < v2
        for (v1, v2) in edges
            @test v1 < v2
        end
    end

    @testset "Inversion detection" begin
        ref = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        cur = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        mesh = SimplicialMesh{2, Float64}(cur, sv, sn; reference_positions=ref)
        @test !has_inverted_simplex(mesh)

        # Move vertex 3 to make triangle invert
        set_vertex_position!(mesh, 3, (0.0, -1.0))
        @test has_inverted_simplex(mesh)
    end

    @testset "Distortion metric" begin
        ref = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        # Identity deformation: distortion should be 1
        sv = reshape(Int32[1, 2, 3], 3, 1)
        sn = zeros(Int32, 3, 1)
        mesh = SimplicialMesh{2, Float64}(ref, sv, sn)
        @test distortion_metric(mesh, 1) ≈ 1.0 atol=1e-12

        # Scaling by 10 should give distortion ≈ 1 (the metric is normalized)
        set_vertex_position!(mesh, 2, (10.0, 0.0))
        set_vertex_position!(mesh, 3, (0.0, 10.0))
        # Pure isotropic dilation: F = 10*I, F⁻¹ = 0.1*I
        # ‖F‖_F = sqrt(100+100) = sqrt(200) ≈ 14.14
        # ‖F⁻¹‖_F = sqrt(0.02) ≈ 0.1414
        # product / 2 = 14.14 * 0.1414 / 2 ≈ 1.0
        @test distortion_metric(mesh, 1) ≈ 1.0 atol=1e-10

        # Highly anisotropic: F = diag(100, 0.01)
        set_vertex_position!(mesh, 2, (100.0, 0.0))
        set_vertex_position!(mesh, 3, (0.0, 0.01))
        # ‖F‖_F · ‖F⁻¹‖_F / 2: distortion >> 1
        @test distortion_metric(mesh, 1) > 100.0
    end

    @testset "max_distortion across mesh" begin
        # Two-triangle square mesh, deform second triangle
        positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        sv = Int32[1 1; 2 3; 3 4]
        sn = Int32[0 0; 2 1; 0 0]
        mesh = SimplicialMesh{2, Float64}(positions, sv, sn)
        d0 = max_distortion(mesh)
        @test d0 ≈ 1.0 atol=1e-10

        # Deform vertex 4 sharply
        set_vertex_position!(mesh, 4, (-2.0, 5.0))
        d1 = max_distortion(mesh)
        @test d1 > d0
    end

    @testset "3D: single tetrahedron" begin
        positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3, 4], 4, 1)
        sn = zeros(Int32, 4, 1)
        mesh = SimplicialMesh{3, Float64}(positions, sv, sn)

        @test n_simplices(mesh) == 1
        # Standard reference tetrahedron has volume 1/6
        @test simplex_volume(mesh, 1) ≈ 1/6
        @test volume_jacobian(mesh, 1) ≈ 1.0
        @test distortion_metric(mesh, 1) ≈ 1.0 atol=1e-10
    end

    @testset "3D: deformation gradient" begin
        positions = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)]
        ref = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        sv = reshape(Int32[1, 2, 3, 4], 4, 1)
        sn = zeros(Int32, 4, 1)
        mesh = SimplicialMesh{3, Float64}(positions, sv, sn; reference_positions=ref)

        F = deformation_gradient(mesh, 1)
        @test F[1][1] ≈ 2.0
        @test F[2][2] ≈ 3.0
        @test F[3][3] ≈ 4.0
        @test volume_jacobian(mesh, 1) ≈ 24.0
    end

    @testset "Argument validation" begin
        positions = [(0.0,), (1.0,)]
        @test_throws ArgumentError SimplicialMesh{1, Float64}(positions,
            Int32[1 2; 2 3], Int32[0 0; 0 0])  # vertex 3 doesn't exist (only 2 vertices)
    end

end
