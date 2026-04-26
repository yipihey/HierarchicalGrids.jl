using Test
using HierarchicalGrids
using HierarchicalGrids.BoundaryConditions: validate, is_periodic_axis
using HierarchicalGrids: face_neighbors_with_bcs

@testset "BCKind enum + defaults" begin
    # All five kinds exist as instances of BCKind
    for k in (PERIODIC, INFLOW, OUTFLOW, REFLECTING, DIRICHLET)
        @test k isa BCKind
    end

    # default_bc(Val(D)) returns all-REFLECTING
    @test default_bc(Val(1)) === ((REFLECTING, REFLECTING),)
    @test default_bc(Val(2)) === ((REFLECTING, REFLECTING),
                                  (REFLECTING, REFLECTING))
    @test default_bc(Val(3)) === ((REFLECTING, REFLECTING),
                                  (REFLECTING, REFLECTING),
                                  (REFLECTING, REFLECTING))
end

@testset "validate rejects half-periodic axis" begin
    # PERIODIC on one side only is invalid
    bad1 = ((PERIODIC, REFLECTING),)
    bad2 = ((REFLECTING, REFLECTING), (PERIODIC, OUTFLOW))
    @test_throws ArgumentError validate(bad1)
    @test_throws ArgumentError validate(bad2)

    # Symmetric periodic OR all-non-periodic both pass
    @test validate(((PERIODIC, PERIODIC),)) === ((PERIODIC, PERIODIC),)
    ok2 = ((PERIODIC, PERIODIC), (REFLECTING, REFLECTING))
    @test validate(ok2) === ok2
    ok3 = ((INFLOW, OUTFLOW), (DIRICHLET, REFLECTING))
    @test validate(ok3) === ok3

    # FrameBoundaries constructor also enforces
    @test_throws ArgumentError FrameBoundaries(bad1)
    @test_throws ArgumentError FrameBoundaries(bad2)
end

@testset "FrameBoundaries constructor + accessors" begin
    spec = ((PERIODIC, PERIODIC), (INFLOW, OUTFLOW), (DIRICHLET, REFLECTING))
    fb = FrameBoundaries(spec)
    @test fb isa FrameBoundaries{3}
    @test fb.spec === spec

    # bc accessor
    @test bc(fb, 1, 1) === PERIODIC
    @test bc(fb, 1, 2) === PERIODIC
    @test bc(fb, 2, 1) === INFLOW
    @test bc(fb, 2, 2) === OUTFLOW
    @test bc(fb, 3, 1) === DIRICHLET
    @test bc(fb, 3, 2) === REFLECTING

    # is_periodic_axis on FrameBoundaries
    @test is_periodic_axis(fb, 1) == true
    @test is_periodic_axis(fb, 2) == false
    @test is_periodic_axis(fb, 3) == false

    # is_periodic_axis on BoundarySpec directly
    @test is_periodic_axis(spec, 1) == true
    @test is_periodic_axis(spec, 2) == false

    # Default constructor (all REFLECTING)
    fb2 = FrameBoundaries(Val(2))
    @test fb2.spec === default_bc(Val(2))
    @test is_periodic_axis(fb2, 1) == false
    @test is_periodic_axis(fb2, 2) == false

    fb3 = FrameBoundaries(2)
    @test fb3.spec === default_bc(Val(2))
end

@testset "Periodic neighbor wrap on 2x2 leaf mesh, x-periodic" begin
    # 2D mesh, refine root once → 4 leaves arranged 2x2 on the unit square
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    @test n_cells(mesh) == 5

    # Identify the 4 leaves and their unit boxes
    leaves = [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])]
    @test length(leaves) == 4

    # Build a (leaf_idx -> unit_box) map for clarity
    boxes = Dict(i => HierarchicalGrids.cell_unit_box(mesh, i) for i in leaves)

    # Helper: look up the leaf at (x_half, y_half) where each is :lo or :hi.
    function find_leaf(xh::Symbol, yh::Symbol)
        for i in leaves
            (lo, hi) = boxes[i]
            x_match = xh === :lo ? lo[1] < 0.25 : lo[1] > 0.25
            y_match = yh === :lo ? lo[2] < 0.25 : lo[2] > 0.25
            x_match && y_match && return i
        end
        return 0
    end

    LL = find_leaf(:lo, :lo)
    LH = find_leaf(:lo, :hi)
    HL = find_leaf(:hi, :lo)
    HH = find_leaf(:hi, :hi)
    @test LL > 0 && LH > 0 && HL > 0 && HH > 0

    # x-periodic only
    fb = FrameBoundaries(((PERIODIC, PERIODIC), (REFLECTING, REFLECTING)))

    # Without BCs: lo-x face of LL is boundary (0)
    nb_LL_plain = face_neighbors(mesh, LL)
    @test nb_LL_plain[1] == 0  # x-lo of LL is on x=0 wall

    # With BCs: lo-x face of LL wraps to HL (the hi-x leaf at the same y)
    nb_LL = face_neighbors_with_bcs(mesh, LL, fb)
    @test nb_LL[1] == HL          # x-lo wraps to HL
    @test nb_LL[2] == HL          # x-hi (within the domain) is HL
    # y faces unchanged: y-lo is boundary, y-hi is LH
    @test nb_LL[3] == 0
    @test nb_LL[4] == LH

    # Symmetrically for HL: x-hi wraps to LL
    nb_HL = face_neighbors_with_bcs(mesh, HL, fb)
    @test nb_HL[1] == LL
    @test nb_HL[2] == LL
    @test nb_HL[3] == 0
    @test nb_HL[4] == HH

    # And for the upper row
    nb_LH = face_neighbors_with_bcs(mesh, LH, fb)
    @test nb_LH[1] == HH
    @test nb_LH[2] == HH
    @test nb_LH[3] == LL
    @test nb_LH[4] == 0
end

@testset "Mixed BC: x-periodic, y-reflecting on 2x2 mesh" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    leaves = [i for i in 1:n_cells(mesh) if HierarchicalGrids.is_leaf(mesh[i])]

    fb = FrameBoundaries(((PERIODIC, PERIODIC), (REFLECTING, REFLECTING)))

    # For every leaf, at least one of the two y faces (face 3 = y-lo, 4 = y-hi)
    # must remain 0 (boundary).
    for i in leaves
        nb = face_neighbors_with_bcs(mesh, i, fb)
        # x faces (1, 2) should never be 0 on a 2x2 mesh (either interior or
        # wrap-around).
        @test nb[1] != 0
        @test nb[2] != 0
        # y boundary entries should remain 0 (no wrap on y)
        nb_plain = face_neighbors(mesh, i)
        if nb_plain[3] == 0
            @test nb[3] == 0
        end
        if nb_plain[4] == 0
            @test nb[4] == 0
        end
    end

    # Reflecting/Inflow/Outflow/Dirichlet are advisory at this layer — verify
    # the FrameBoundaries records them and the neighbor wiring is unaffected.
    fb_mixed = FrameBoundaries(((INFLOW, OUTFLOW), (DIRICHLET, REFLECTING)))
    for i in leaves
        @test face_neighbors_with_bcs(mesh, i, fb_mixed) === face_neighbors(mesh, i)
    end
end

@testset "frame_bcs is forward-compat for compute_overlap" begin
    # Build a tiny pair: SimplicialMesh covering [0,1]² and a 1-cell EulerianFrame.
    positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    sv = Int32[1 1; 2 3; 3 4]
    sn = Int32[2 1; 0 0; 0 0]
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    overlap_no_bcs = compute_overlap(lag, frame; moment_order=0)
    fb = FrameBoundaries(((REFLECTING, REFLECTING), (REFLECTING, REFLECTING)))
    overlap_with_bcs = compute_overlap(lag, frame; moment_order=0, frame_bcs=fb)
    # No periodic axes ⇒ identical entry counts and total volume.
    @test n_entries(overlap_no_bcs) == n_entries(overlap_with_bcs)
    @test total_overlap_volume(overlap_no_bcs) ≈ total_overlap_volume(overlap_with_bcs)
end
