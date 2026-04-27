using Test
using HierarchicalGrids
using HierarchicalGrids: Sequential, OhMyThreadsBackend

# ============================================================================
# EulerianFrame × EulerianFrame compute_overlap (PR-12)
#
# Box-vs-box geometric overlap between two Eulerian frames. Exercises the
# closed-form clip + per-axis moment factoring, the AABB filter, and the
# parallel/sequential backend dispatch.
# ============================================================================

# Helper: build a HierarchicalMesh{D} and refine it `levels` times uniformly
# (each level refines every current leaf), then wrap it in an EulerianFrame
# over the box [lo, hi].
function _make_uniform_frame(::Val{D}, levels::Int, lo::NTuple{D, Float64},
                              hi::NTuple{D, Float64}) where {D}
    mesh = HierarchicalMesh{D}()
    for _ in 1:levels
        leaves = enumerate_leaves(mesh)
        refine_cells!(mesh, leaves)
    end
    return EulerianFrame(mesh, lo, hi)
end

# Helper: physical centroid and volume of a single-cell frame's root box.
function _root_centroid_volume(frame::EulerianFrame{D, Float64}) where {D}
    lo, hi = root_box(frame)
    vol = prod(hi[d] - lo[d] for d in 1:D)
    centroid = ntuple(d -> 0.5 * (lo[d] + hi[d]), Val(D))
    return centroid, vol
end

@testset "Same-region same-level: single-cell × single-cell, 2D" begin
    fa = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (1.0, 1.0))
    fb = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (1.0, 1.0))
    o = compute_overlap(fa, fb; moment_order = 3)
    @test n_entries(o) == 1
    @test total_overlap_volume(o) ≈ 1.0
    e = o.entries[1]
    @test e.lag_idx == 1
    @test e.eul_idx == 1
    @test e.volume ≈ 1.0
    @test all(isapprox.(e.centroid, (0.5, 0.5); atol = 1e-12))
    # 0th moment in `moments[1]` is the volume.
    @test e.moments[1] ≈ 1.0
end

@testset "Coarse × fine 2D: 1 leaf vs 4 leaves" begin
    fa = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (1.0, 1.0))   # 1 leaf
    fb = _make_uniform_frame(Val(2), 1, (0.0, 0.0), (1.0, 1.0))   # 4 leaves
    o = compute_overlap(fa, fb; moment_order = 2)
    @test n_entries(o) == 4
    for e in o.entries
        @test e.volume ≈ 0.25
    end
    @test total_overlap_volume(o) ≈ 1.0
end

@testset "Translated frames 2D: partial overlap area = 0.49" begin
    fa = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (1.0, 1.0))
    fb = _make_uniform_frame(Val(2), 0, (0.3, 0.3), (1.3, 1.3))
    o = compute_overlap(fa, fb; moment_order = 1)
    @test n_entries(o) == 1
    @test total_overlap_volume(o) ≈ 0.49 atol = 1e-12
    # Centroid of the overlap [0.3, 1.0]^2 is (0.65, 0.65).
    e = o.entries[1]
    @test all(isapprox.(e.centroid, (0.65, 0.65); atol = 1e-12))
end

@testset "Disjoint frames 2D: no overlap entries" begin
    fa = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (0.4, 0.4))
    fb = _make_uniform_frame(Val(2), 0, (0.6, 0.6), (1.0, 1.0))
    o = compute_overlap(fa, fb)
    @test n_entries(o) == 0
    @test total_overlap_volume(o) == 0.0
end

@testset "Touching frames 2D: face-touch is not overlap" begin
    # Sharing a face only (zero-area intersection) must NOT produce an
    # entry — `aabbs_overlap` and the per-axis hi[d] > lo[d] guard agree.
    fa = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (1.0, 1.0))
    fb = _make_uniform_frame(Val(2), 0, (1.0, 0.0), (2.0, 1.0))
    o = compute_overlap(fa, fb)
    @test n_entries(o) == 0
end

@testset "Same-region same-level 3D: 1 × 1 = full volume" begin
    fa = _make_uniform_frame(Val(3), 0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    fb = _make_uniform_frame(Val(3), 0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    o = compute_overlap(fa, fb; moment_order = 1)
    @test n_entries(o) == 1
    @test total_overlap_volume(o) ≈ 1.0
end

@testset "Coarse × refined 3D: 1 leaf vs 8^2 = 64 leaves (2 levels)" begin
    fa = _make_uniform_frame(Val(3), 0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    fb = _make_uniform_frame(Val(3), 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    leaves_b = enumerate_leaves(fb.mesh)
    @test length(leaves_b) == 64   # 8 × 8 isotropic refinement
    o = compute_overlap(fa, fb; moment_order = 1)
    @test n_entries(o) == 64
    @test total_overlap_volume(o) ≈ 1.0 atol = 1e-12
    for e in o.entries
        @test e.volume ≈ 1.0 / 64
    end
end

@testset "Translated frames 3D: overlap volume = 0.7^3 = 0.343" begin
    fa = _make_uniform_frame(Val(3), 0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    fb = _make_uniform_frame(Val(3), 0, (0.3, 0.3, 0.3), (1.3, 1.3, 1.3))
    o = compute_overlap(fa, fb; moment_order = 0)
    @test n_entries(o) == 1
    @test total_overlap_volume(o) ≈ 0.7^3 atol = 1e-12
end

@testset "First moments: refined fb against single-cell fa, 2D" begin
    # Sum_e (centroid_e * volume_e) over all entries should equal the
    # single-cell fa's (centroid * volume), since fb tessellates fa.
    fa = _make_uniform_frame(Val(2), 0, (0.5, 1.0), (1.5, 3.0))
    fb = _make_uniform_frame(Val(2), 1, (0.5, 1.0), (1.5, 3.0))   # 4 leaves
    o = compute_overlap(fa, fb; moment_order = 1)
    fa_centroid, fa_volume = _root_centroid_volume(fa)
    sx = 0.0; sy = 0.0
    for e in o.entries
        sx += e.centroid[1] * e.volume
        sy += e.centroid[2] * e.volume
    end
    @test sx ≈ fa_centroid[1] * fa_volume atol = 1e-12
    @test sy ≈ fa_centroid[2] * fa_volume atol = 1e-12

    # Cross-check via the moment vector itself: moment index 2 / 3 are
    # the first moments ∫ x dV / ∫ y dV (origin at physical origin).
    # Total over fb's tessellation must equal fa's analytic ∫ x dV.
    mx = 0.0; my = 0.0
    for e in o.entries
        mx += e.moments[2]
        my += e.moments[3]
    end
    # ∫_{fa} x dV over [0.5, 1.5] × [1.0, 3.0] = (1.5^2 - 0.5^2)/2 * (3-1) = 1.0 * 2 = 2.0
    # ∫_{fa} y dV over same box       = (1.5 - 0.5) * (3.0^2 - 1.0^2)/2 = 1.0 * 4.0 = 4.0
    @test mx ≈ 2.0 atol = 1e-12
    @test my ≈ 4.0 atol = 1e-12
end

@testset "Second moments: per-axis factorization" begin
    # For two single-cell frames over [0, 2]^2, the (2, 0) moment is
    # ∫_0^2 x^2 dx · ∫_0^2 1 dy = 8/3 · 2 = 16/3.
    fa = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (2.0, 2.0))
    fb = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (2.0, 2.0))
    o = compute_overlap(fa, fb; moment_order = 2)
    e = o.entries[1]
    # Graded-lex ordering: index 1 = (0,0), 2 = (1,0), 3 = (0,1),
    # 4 = (2,0), 5 = (1,1), 6 = (0,2).
    @test e.moments[1] ≈ 4.0           # volume
    @test e.moments[2] ≈ 4.0           # ∫ x dV = 2*2 = 4
    @test e.moments[3] ≈ 4.0           # ∫ y dV
    @test e.moments[4] ≈ 16.0 / 3.0    # ∫ x^2 dV
    @test e.moments[5] ≈ 4.0           # ∫ xy dV = 2 · 2 = 4
    @test e.moments[6] ≈ 16.0 / 3.0    # ∫ y^2 dV
end

@testset "Backend determinism: Sequential vs OhMyThreadsBackend" begin
    # Build a moderately refined pair so the parallel chunking actually
    # has work to distribute. Two-level refinement on each side gives
    # 16 × 16 leaves in 2D.
    fa = _make_uniform_frame(Val(2), 2, (0.0, 0.0), (1.0, 1.0))
    fb = _make_uniform_frame(Val(2), 2, (0.1, 0.1), (1.1, 1.1))
    o_seq = compute_overlap(fa, fb; moment_order = 2,
                                       parallel = false,
                                       backend = Sequential())
    for sched in (:dynamic, :static, :greedy)
        o_par = compute_overlap(fa, fb; moment_order = 2,
                                          parallel = true,
                                          backend = OhMyThreadsBackend(sched))
        @test n_entries(o_par) == n_entries(o_seq)
        @test total_overlap_volume(o_par) ≈ total_overlap_volume(o_seq) atol = 1e-12
        # Entries are sorted by (lag_idx, eul_idx) inside finalize_overlap,
        # so direct elementwise comparison is valid.
        for k in 1:n_entries(o_seq)
            ea = o_seq.entries[k]; eb = o_par.entries[k]
            @test ea.lag_idx == eb.lag_idx
            @test ea.eul_idx == eb.eul_idx
            @test ea.volume ≈ eb.volume atol = 1e-14
            @test all(isapprox.(ea.centroid, eb.centroid; atol = 1e-14))
            @test all(isapprox.(ea.moments, eb.moments; atol = 1e-12))
        end
    end
end

@testset "moment_order = 0 (volume-only): no centroid moments" begin
    # At order 0 the moments vector has length 1 (just the volume).
    # Centroid is still computed geometrically from the box.
    fa = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (1.0, 1.0))
    fb = _make_uniform_frame(Val(2), 0, (0.25, 0.25), (1.25, 1.25))
    o = compute_overlap(fa, fb; moment_order = 0)
    @test n_entries(o) == 1
    e = o.entries[1]
    @test length(e.moments) == 1
    @test e.moments[1] ≈ 0.75 * 0.75 atol = 1e-14
    @test all(isapprox.(e.centroid, (0.625, 0.625); atol = 1e-14))
end

@testset "Empty frames: well-formed empty GeometricOverlap" begin
    # Both meshes have a single root cell (1 leaf each), shifted apart so
    # there's no overlap. Confirms the empty-leaf-list and disjoint-AABB
    # branches return a valid `GeometricOverlap` with zero entries.
    fa = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (1.0, 1.0))
    fb = _make_uniform_frame(Val(2), 0, (10.0, 10.0), (11.0, 11.0))
    o = compute_overlap(fa, fb)
    @test n_entries(o) == 0
    @test isempty(o.entries)
    @test length(o.lag_to_entries) == n_cells(fa.mesh)
    @test length(o.eul_to_entries) == n_cells(fb.mesh)
end

@testset "CSR-style indexing: entries_for_lag / entries_for_eul" begin
    fa = _make_uniform_frame(Val(2), 0, (0.0, 0.0), (1.0, 1.0))   # 1 leaf
    fb = _make_uniform_frame(Val(2), 1, (0.0, 0.0), (1.0, 1.0))   # 4 leaves
    o = compute_overlap(fa, fb; moment_order = 1)

    # All four entries should be attached to fa's single leaf (cell 1).
    @test length(entries_for_lag(o, 1)) == 4
    leaves_b = enumerate_leaves(fb.mesh)
    for ib in leaves_b
        # Each fb leaf is covered by exactly one entry (back to fa's cell 1).
        @test length(entries_for_eul(o, ib)) == 1
    end
end
