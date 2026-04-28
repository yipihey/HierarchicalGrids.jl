using Test
using HierarchicalGrids
using HierarchicalGrids: Sequential, OhMyThreadsBackend
using LinearAlgebra: det

# ============================================================================
# Determinism tests: polynomial_remap_l_to_e! / _e_to_l! must produce
# byte-identical destination coefficients across all parallel backends.
#
# The per-destination-cell solve loop is embarrassingly parallel: each `j`
# reads its own `rhs[:, j]` column, its own `dst_frames[j]`, and writes
# its own `target_coeffs[:, j]` column. The upstream RHS-assembly phase
# (`accumulate_polynomial_rhs!`) uses a two-phase pattern — per-task
# partial RHS buffers + a deterministic (fixed chunk-index order)
# sequential reduction — so the result is byte-exact regardless of
# scheduler choice as long as the partial sums are summed in the same
# order on every call.
# ============================================================================

const _PRT_BACKENDS = [
    ("Sequential",                    Sequential()),
    ("OhMyThreadsBackend(:dynamic)",  OhMyThreadsBackend(:dynamic)),
    ("OhMyThreadsBackend(:static)",   OhMyThreadsBackend(:static)),
    ("OhMyThreadsBackend(:greedy)",   OhMyThreadsBackend(:greedy)),
    ("OhMyThreadsBackend(:serial)",   OhMyThreadsBackend(:serial)),
]

# ----------------------------------------------------------------------------
# Fixture builders
# ----------------------------------------------------------------------------

# Small tile fixture: 2 triangles × 4 Eulerian leaves (single refinement).
# Mirrors `_build_tile_setup` from test_polynomial_remap.jl.
function _build_small_setup()
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    sv = Int32[1 4; 2 3; 3 2]
    sn = zeros(Int32, 3, 2)
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    src_frames = CellReferenceFrame{2, Float64}[lagrangian_frame(lag, i)
                                                 for i in 1:n_simplices(lag)]
    dst_frames = CellReferenceFrame{2, Float64}[eulerian_frame(frame, j)
                                                 for j in 1:n_cells(eul)]
    return (; lag, eul, frame, src_frames, dst_frames)
end

# Larger fixture: a regular nside × nside triangulated grid (2 tris per square)
# over [0, 1]^2 paired with an Eulerian mesh refined `r` times uniformly
# (4^r leaves at the deepest level). Picks parameters where parallelism
# actually engages.
function _build_large_setup(; nside::Int = 8, refine::Int = 3)
    h = 1.0 / nside
    positions = NTuple{2, Float64}[]
    for j in 0:nside, i in 0:nside
        push!(positions, (i * h, j * h))
    end
    n_tri = 2 * nside * nside
    sv = Matrix{Int32}(undef, 3, n_tri)
    sn = zeros(Int32, 3, n_tri)
    t = 1
    for j in 0:(nside - 1), i in 0:(nside - 1)
        v00 = j * (nside + 1) + i + 1
        v10 = v00 + 1
        v01 = v00 + (nside + 1)
        v11 = v01 + 1
        sv[1, t] = v00; sv[2, t] = v10; sv[3, t] = v11; t += 1
        sv[1, t] = v00; sv[2, t] = v11; sv[3, t] = v01; t += 1
    end
    lag = SimplicialMesh{2, Float64}(positions, sv, sn)
    eul = HierarchicalMesh{2}()
    for _ in 1:refine
        leaves = enumerate_leaves(eul)
        refine_cells!(eul, leaves)
    end
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    src_frames = CellReferenceFrame{2, Float64}[lagrangian_frame(lag, i)
                                                 for i in 1:n_simplices(lag)]
    dst_frames = CellReferenceFrame{2, Float64}[eulerian_frame(frame, j)
                                                 for j in 1:n_cells(eul)]
    return (; lag, eul, frame, src_frames, dst_frames)
end

# Build a quadratic source coefficient matrix on the Lagrangian side that
# represents a non-trivial physical polynomial in each triangle.
function _build_l_to_e_source(setup, P::Int)
    lag = setup.lag
    src_frames = setup.src_frames
    n_phys = HierarchicalGrids.moments_length(2, P)
    # A reproducible non-trivial physical polynomial.
    phys = [Float64(k) + 0.5 for k in 1:n_phys]
    n_lag = n_simplices(lag)
    src = zeros(n_phys, n_lag)
    for i in 1:n_lag
        T_mat = reference_to_physical_pullback(src_frames[i], P)
        src[:, i] = T_mat' \ phys
    end
    return src
end

# Build an Eulerian-side source coefficient matrix.
function _build_e_to_l_source(setup, P::Int)
    eul = setup.eul
    dst_frames = setup.dst_frames
    n_phys = HierarchicalGrids.moments_length(2, P)
    phys = [Float64(k) * 0.7 - 0.3 for k in 1:n_phys]
    n_eul = n_cells(eul)
    src = zeros(n_phys, n_eul)
    for j in 1:n_eul
        T_mat = reference_to_physical_pullback(dst_frames[j], P)
        src[:, j] = T_mat' \ phys
    end
    return src
end

# ----------------------------------------------------------------------------
# Determinism: l_to_e
# ----------------------------------------------------------------------------

@testset "polynomial_remap_l_to_e!: byte-equal across backends (small fixture)" begin
    setup = _build_small_setup()
    P = 2
    overlap = compute_overlap(setup.lag, setup.frame; moment_order = 2 * P)
    src = _build_l_to_e_source(setup, P)
    n_phys = HierarchicalGrids.moments_length(2, P)
    n_eul = n_cells(setup.eul)

    # Reference: Sequential.
    dst_ref = zeros(n_phys, n_eul)
    polynomial_remap_l_to_e!(dst_ref, src, overlap,
                              setup.src_frames, setup.dst_frames, P, P;
                              backend = Sequential())

    for (label, backend) in _PRT_BACKENDS
        dst = zeros(n_phys, n_eul)
        polynomial_remap_l_to_e!(dst, src, overlap,
                                  setup.src_frames, setup.dst_frames, P, P;
                                  backend = backend)
        @test dst == dst_ref
    end
end

@testset "polynomial_remap_l_to_e!: byte-equal across backends (large fixture)" begin
    setup = _build_large_setup(; nside = 8, refine = 3)
    P = 1
    overlap = compute_overlap(setup.lag, setup.frame; moment_order = 2 * P)
    src = _build_l_to_e_source(setup, P)
    n_phys = HierarchicalGrids.moments_length(2, P)
    n_eul = n_cells(setup.eul)

    # Reference: Sequential.
    dst_ref = zeros(n_phys, n_eul)
    polynomial_remap_l_to_e!(dst_ref, src, overlap,
                              setup.src_frames, setup.dst_frames, P, P;
                              backend = Sequential())

    for (label, backend) in _PRT_BACKENDS
        dst = zeros(n_phys, n_eul)
        polynomial_remap_l_to_e!(dst, src, overlap,
                                  setup.src_frames, setup.dst_frames, P, P;
                                  backend = backend)
        @test dst == dst_ref
    end
end

# ----------------------------------------------------------------------------
# Determinism: e_to_l
# ----------------------------------------------------------------------------

@testset "polynomial_remap_e_to_l!: byte-equal across backends (small fixture)" begin
    setup = _build_small_setup()
    P = 2
    overlap = compute_overlap(setup.lag, setup.frame; moment_order = 2 * P)
    src = _build_e_to_l_source(setup, P)
    n_phys = HierarchicalGrids.moments_length(2, P)
    n_lag = n_simplices(setup.lag)

    # In e_to_l: src_frames are Eulerian, dst_frames are Lagrangian (swapped).
    dst_ref = zeros(n_phys, n_lag)
    polynomial_remap_e_to_l!(dst_ref, src, overlap,
                              setup.dst_frames, setup.src_frames, P, P;
                              backend = Sequential())

    for (label, backend) in _PRT_BACKENDS
        dst = zeros(n_phys, n_lag)
        polynomial_remap_e_to_l!(dst, src, overlap,
                                  setup.dst_frames, setup.src_frames, P, P;
                                  backend = backend)
        @test dst == dst_ref
    end
end

@testset "polynomial_remap_e_to_l!: byte-equal across backends (large fixture)" begin
    setup = _build_large_setup(; nside = 8, refine = 3)
    P = 1
    overlap = compute_overlap(setup.lag, setup.frame; moment_order = 2 * P)
    src = _build_e_to_l_source(setup, P)
    n_phys = HierarchicalGrids.moments_length(2, P)
    n_lag = n_simplices(setup.lag)

    dst_ref = zeros(n_phys, n_lag)
    polynomial_remap_e_to_l!(dst_ref, src, overlap,
                              setup.dst_frames, setup.src_frames, P, P;
                              backend = Sequential())

    for (label, backend) in _PRT_BACKENDS
        dst = zeros(n_phys, n_lag)
        polynomial_remap_e_to_l!(dst, src, overlap,
                                  setup.dst_frames, setup.src_frames, P, P;
                                  backend = backend)
        @test dst == dst_ref
    end
end

# ----------------------------------------------------------------------------
# Default-backend behavior: omitting the kwarg should match the dynamic
# OhMyThreadsBackend default that PR-0 set as the process-global default.
# ----------------------------------------------------------------------------

@testset "polynomial_remap_l_to_e!: default backend matches OhMyThreadsBackend(:dynamic)" begin
    setup = _build_small_setup()
    P = 2
    overlap = compute_overlap(setup.lag, setup.frame; moment_order = 2 * P)
    src = _build_l_to_e_source(setup, P)
    n_phys = HierarchicalGrids.moments_length(2, P)
    n_eul = n_cells(setup.eul)

    dst_default = zeros(n_phys, n_eul)
    polynomial_remap_l_to_e!(dst_default, src, overlap,
                              setup.src_frames, setup.dst_frames, P, P)

    dst_dynamic = zeros(n_phys, n_eul)
    polynomial_remap_l_to_e!(dst_dynamic, src, overlap,
                              setup.src_frames, setup.dst_frames, P, P;
                              backend = OhMyThreadsBackend(:dynamic))
    @test dst_default == dst_dynamic
end

# ----------------------------------------------------------------------------
# Determinism on the parallel RHS path: pick a fixture large enough that
# `length(overlap.entries) >= _POLYNOMIAL_RHS_PARALLEL_THRESHOLD` (currently
# 2048), so `accumulate_polynomial_rhs!` actually takes the per-task
# partial-buffer code path under a non-`Sequential` backend.
# ----------------------------------------------------------------------------

@testset "accumulate_polynomial_rhs!: byte-equal across backends (entries ≥ threshold)" begin
    # nside=20, refine=5 gives 800 triangles × ~1365 leaves and ~3713
    # overlap entries — comfortably above the 2048 threshold so the
    # parallel partial-buffer kernel is exercised.
    setup = _build_large_setup(; nside = 20, refine = 5)
    P_src, P_dst = 1, 1
    overlap = compute_overlap(setup.lag, setup.frame; moment_order = P_src + P_dst)
    @test length(overlap.entries) >= 2048  # confirm we're past the threshold
    n_src = HierarchicalGrids.moments_length(2, P_src)
    n_dst = HierarchicalGrids.moments_length(2, P_dst)
    n_lag = n_simplices(setup.lag)
    n_eul = n_cells(setup.eul)

    src_pullbacks = [reference_to_physical_pullback(setup.src_frames[i], P_src)
                       for i in 1:n_lag]
    dst_pullbacks = [reference_to_physical_pullback(setup.dst_frames[j], P_dst)
                       for j in 1:n_eul]
    src_coeffs = [Float64((α + 3i) % 7) - 3.0 for α in 1:n_src, i in 1:n_lag]

    rhs_ref = zeros(n_dst, n_eul)
    accumulate_polynomial_rhs!(rhs_ref, src_coeffs, src_pullbacks, dst_pullbacks,
                                overlap, P_src, P_dst;
                                invert = false, backend = Sequential())

    for (label, backend) in _PRT_BACKENDS
        rhs = zeros(n_dst, n_eul)
        accumulate_polynomial_rhs!(rhs, src_coeffs, src_pullbacks, dst_pullbacks,
                                    overlap, P_src, P_dst;
                                    invert = false, backend = backend)
        @test rhs == rhs_ref
    end

    # Also verify the invert=true path (e_to_l direction).
    src_coeffs_e = [Float64((α + 5j) % 11) - 4.0 for α in 1:n_src, j in 1:n_eul]
    rhs_ref_e = zeros(n_dst, n_lag)
    accumulate_polynomial_rhs!(rhs_ref_e, src_coeffs_e, dst_pullbacks, src_pullbacks,
                                overlap, P_src, P_dst;
                                invert = true, backend = Sequential())
    for (label, backend) in _PRT_BACKENDS
        rhs = zeros(n_dst, n_lag)
        accumulate_polynomial_rhs!(rhs, src_coeffs_e, dst_pullbacks, src_pullbacks,
                                    overlap, P_src, P_dst;
                                    invert = true, backend = backend)
        @test rhs == rhs_ref_e
    end
end

@testset "polynomial_remap_l_to_e!: byte-equal across backends (entries ≥ threshold)" begin
    setup = _build_large_setup(; nside = 20, refine = 5)
    P = 1
    overlap = compute_overlap(setup.lag, setup.frame; moment_order = 2 * P)
    @test length(overlap.entries) >= 2048
    src = _build_l_to_e_source(setup, P)
    n_phys = HierarchicalGrids.moments_length(2, P)
    n_eul = n_cells(setup.eul)

    dst_ref = zeros(n_phys, n_eul)
    polynomial_remap_l_to_e!(dst_ref, src, overlap,
                              setup.src_frames, setup.dst_frames, P, P;
                              backend = Sequential())

    for (label, backend) in _PRT_BACKENDS
        dst = zeros(n_phys, n_eul)
        polynomial_remap_l_to_e!(dst, src, overlap,
                                  setup.src_frames, setup.dst_frames, P, P;
                                  backend = backend)
        @test dst == dst_ref
    end
end
