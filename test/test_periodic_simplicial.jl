using Test
using HierarchicalGrids

@testset "1D 4-segment SimplicialMesh, periodic" begin
    # Vertices at x = 0, 0.25, 0.5, 0.75, 1.0
    positions = [(0.0,), (0.25,), (0.5,), (0.75,), (1.0,)]
    # 4 segments: each is [v_lo, v_hi] (low-x first).
    sv = Int32[1 2 3 4
               2 3 4 5]
    # simplex_neighbors[k, s] is the simplex sharing the face opposite
    # vertex k of simplex s (per docstring). For segment [v_lo, v_hi]:
    #   k=1 → face opposite v_lo (at high-x end)  → "right" neighbor
    #   k=2 → face opposite v_hi (at low-x end)   → "left"  neighbor
    sn = Int32[2 3 4 0      # k=1 (right): seg1→2, seg2→3, seg3→4, seg4→boundary
               0 1 2 3]     # k=2 (left):  seg1→boundary, seg2→1, seg3→2, seg4→3
    mesh = SimplicialMesh{1, Float64}(positions, sv, sn)

    @test n_simplices(mesh) == 4
    @test simplex_neighbor(mesh, 1, 2) == 0     # left of seg 1 (face opp. v_hi)
    @test simplex_neighbor(mesh, 4, 1) == 0     # right of seg 4 (face opp. v_lo)

    # Make periodic over [0, 1]
    periodic!(mesh, (true,), ((0.0, 1.0),))

    # Segment 1's left wraps to segment 4; segment 4's right wraps to seg 1.
    @test simplex_neighbor(mesh, 1, 2) == 4
    @test simplex_neighbor(mesh, 4, 1) == 1

    # Interior wiring is unchanged
    @test simplex_neighbor(mesh, 1, 1) == 2
    @test simplex_neighbor(mesh, 2, 2) == 1
    @test simplex_neighbor(mesh, 2, 1) == 3
    @test simplex_neighbor(mesh, 3, 2) == 2
    @test simplex_neighbor(mesh, 3, 1) == 4
    @test simplex_neighbor(mesh, 4, 2) == 3
end

@testset "2D triangulation of [0,1]², x-periodic only" begin
    # 2x2 grid of squares, each split into 2 triangles → 8 triangles, 9 vertices
    # Vertices on a 3x3 grid:
    # (0,0)=1 (0.5,0)=2 (1,0)=3
    # (0,0.5)=4 (0.5,0.5)=5 (1,0.5)=6
    # (0,1)=7 (0.5,1)=8 (1,1)=9
    positions = [
        (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
        (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
        (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
    ]

    # Each square split into "lower" (vertices a, b, c forming lower-right
    # triangle) and "upper" (a, c, d forming upper-left triangle) where the
    # vertex ordering is counter-clockwise:
    #   lower: (lo-x lo-y), (hi-x lo-y), (hi-x hi-y)
    #   upper: (lo-x lo-y), (hi-x hi-y), (lo-x hi-y)
    # Square corners (i,j) for i,j ∈ {0,1}:
    function tri_pair(a, b, c, d)
        # returns (lower [a,b,c], upper [a,c,d])
        return ([a, b, c], [a, c, d])
    end

    # Squares:
    # SW = (1,2,5,4)
    sw_lower, sw_upper = tri_pair(1, 2, 5, 4)
    # SE = (2,3,6,5)
    se_lower, se_upper = tri_pair(2, 3, 6, 5)
    # NW = (4,5,8,7)
    nw_lower, nw_upper = tri_pair(4, 5, 8, 7)
    # NE = (5,6,9,8)
    ne_lower, ne_upper = tri_pair(5, 6, 9, 8)

    tris = [sw_lower, sw_upper, se_lower, se_upper, nw_lower, nw_upper, ne_lower, ne_upper]
    sv = Matrix{Int32}(undef, 3, 8)
    for (s, t) in enumerate(tris)
        for k in 1:3
            sv[k, s] = Int32(t[k])
        end
    end

    # Build neighbors by brute force: simplex_neighbors[k, s] is the simplex
    # that shares the face opposite vertex k of simplex s. We search.
    sn = zeros(Int32, 3, 8)
    function face_set(s_idx)
        return (Set([sv[2, s_idx], sv[3, s_idx]]),
                Set([sv[1, s_idx], sv[3, s_idx]]),
                Set([sv[1, s_idx], sv[2, s_idx]]))
    end
    faces = [face_set(s) for s in 1:8]
    for s in 1:8, k in 1:3
        my_face = faces[s][k]
        for j in 1:8
            j == s && continue
            if my_face in faces[j]
                sn[k, s] = Int32(j)
                break
            end
        end
    end

    mesh = SimplicialMesh{2, Float64}(positions, sv, sn)

    # Count boundary faces before periodic!
    n_bnd_before = count(==(Int32(0)), sn)
    @test n_bnd_before > 0

    # Make x-periodic, leave y reflecting (left untouched)
    periodic!(mesh, (true, false), ((0.0, 1.0), (0.0, 1.0)))

    # Boundary faces on x-walls should now be wired up; faces on y-walls
    # remain 0.
    n_bnd_after = 0
    for s in 1:8, k in 1:3
        if mesh.simplex_neighbors[k, s] == 0
            n_bnd_after += 1
        end
    end
    @test n_bnd_after < n_bnd_before
    # On a 2x2 grid the x-walls have 4 boundary faces (2 per side), so we
    # eliminate exactly 4 boundary entries.
    @test n_bnd_before - n_bnd_after == 4

    # Verify pairing reciprocity: every newly-wired face has its partner.
    for s in 1:8, k in 1:3
        n = mesh.simplex_neighbors[k, s]
        n == 0 && continue
        # The partner simplex must point back to s on some face.
        found_back = false
        for k2 in 1:3
            if mesh.simplex_neighbors[k2, n] == Int32(s)
                found_back = true
                break
            end
        end
        @test found_back
    end
end

@testset "Round-trip overlap (deferred ghost note)" begin
    # The test contract item 3 specifies: "SimplicialMesh tiling [0,1] made
    # periodic, against itself shifted by exactly one period — assert overlap
    # volume == 1 (or document if recursive ghosting was deferred)."
    #
    # PR-D defers compute_overlap-side periodic ghost generation to a
    # follow-up. The neighbor-graph wiring (face_neighbors_with_bcs) and
    # the SimplicialMesh `periodic!` helper cover all dfmm halo needs at
    # this layer. We verify here that the deferred path does NOT silently
    # produce wrap entries — without ghosts, a Lagrangian mesh whose
    # vertices sit on [1, 2] has zero overlap with an Eulerian frame on
    # [0, 1] regardless of the FrameBoundaries flag.

    # 1D: SimplicialMesh on [1, 2] (one segment shifted by exactly +1
    # period from the [0, 1] frame).
    positions = [(1.0,), (2.0,)]
    sv = reshape(Int32[1, 2], 2, 1)
    sn = reshape(Int32[0, 0], 2, 1)
    lag = SimplicialMesh{1, Float64}(positions, sv, sn)

    eul = HierarchicalMesh{1}()
    frame = EulerianFrame(eul, (0.0,), (1.0,))

    # Without BCs: zero overlap (mesh is outside the frame).
    ov_plain = compute_overlap(lag, frame; moment_order=0)
    @test total_overlap_volume(ov_plain) == 0.0

    # With periodic BCs: still zero in PR-D (ghost generation deferred).
    fb = FrameBoundaries(((PERIODIC, PERIODIC),))
    ov_periodic = compute_overlap(lag, frame; moment_order=0, frame_bcs=fb)
    @test total_overlap_volume(ov_periodic) == 0.0
    # Document the deferral with a comment-test:
    @test true  # PR-D: periodic ghost generation in compute_overlap deferred
end

@testset "pin_boundary_simplices! and is_pinned" begin
    # 1D mesh with 4 segments
    positions = [(0.0,), (1.0,), (2.0,), (3.0,), (4.0,)]
    sv = Int32[1 2 3 4
               2 3 4 5]
    sn = Int32[2 3 4 0
               0 1 2 3]
    mesh = SimplicialMesh{1, Float64}(positions, sv, sn)

    # All simplices unpinned by default
    for s in 1:n_simplices(mesh)
        @test is_pinned(mesh, s) == false
    end

    pin_boundary_simplices!(mesh, [1, 4])
    @test is_pinned(mesh, 1) == true
    @test is_pinned(mesh, 2) == false
    @test is_pinned(mesh, 3) == false
    @test is_pinned(mesh, 4) == true

    # Out-of-range indices throw
    @test_throws BoundsError pin_boundary_simplices!(mesh, [0])
    @test_throws BoundsError pin_boundary_simplices!(mesh, [5])

    # Re-pinning is idempotent
    pin_boundary_simplices!(mesh, [1])
    @test is_pinned(mesh, 1) == true
end
