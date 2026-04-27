# benchmarks/workloads.jl
# ============================================================================
# WORKLOAD REGISTRATION INTERFACE
# ----------------------------------------------------------------------------
# A workload is identified by a `Symbol` and registered in `WORKLOADS`. Each
# entry maps to a NamedTuple of:
#
#   build :: (size::Symbol) -> Any
#       Build the per-call state once per benchmark sample. Should be cheap
#       relative to `run`. Called BEFORE `@benchmark` setup so the returned
#       state can be passed to `run` without the build cost contaminating
#       timings.
#
#   run :: (state, backend::AbstractParallelBackend) -> Any
#       The hot loop. Wrapped by `BenchmarkTools.@benchmark` in
#       `bench_runner.jl`. The returned value is discarded; what matters
#       is wall-clock time and allocations.
#
#   sizes :: Vector{Symbol}
#       Available sizes (e.g. `:small`, `:medium`, `:large`).
#
# Adding a new workload:
#
#   1. Write a `build_<workload>(size::Symbol)` and a
#      `run_<workload>(state, backend)`.
#   2. Push an entry to `WORKLOADS` with `:my_workload =>
#      (build = ..., run = ..., sizes = [...])`.
#   3. Reference it in `bench_runner.jl`'s SWEEP config block.
#
# PR-1/2/3 register their workloads here as they land. PR-0 ships only
# `:compute_overlap` (the existing parallelized verb).
# ============================================================================

using HierarchicalGrids
using HierarchicalGrids: Threading

# ----------------------------------------------------------------------------
# Workload :compute_overlap
# ----------------------------------------------------------------------------

"""
    build_compute_overlap(size) -> NamedTuple

Construct a SimplicialMesh + EulerianFrame pair for the given size symbol.

  :small  → ~16 simplices,  ~16 leaves
  :medium → ~256 simplices, ~256 leaves
  :large  → ~1024 simplices, ~1024 leaves
"""
function build_compute_overlap(size::Symbol)
    if size === :small
        n_lag = 16
        n_eul_refine = 2   # → 16 leaves (4^2)
    elseif size === :medium
        n_lag = 256
        n_eul_refine = 4   # → 256 leaves (4^4)
    elseif size === :large
        n_lag = 1024
        n_eul_refine = 5   # → 1024 leaves (4^5)
    else
        throw(ArgumentError("Unknown size: $size"))
    end

    # Build a regular triangulated grid covering [0,1]^2 with ~n_lag triangles.
    # We tile a sqrt(n_lag/2) × sqrt(n_lag/2) grid of squares, each split into
    # two triangles.
    nside = max(1, isqrt(n_lag ÷ 2))
    h = 1.0 / nside

    positions = Vector{NTuple{2, Float64}}()
    for j in 0:nside, i in 0:nside
        push!(positions, (i * h, j * h))
    end

    n_tri = 2 * nside * nside
    sv = Matrix{Int32}(undef, 3, n_tri)
    sn = zeros(Int32, 3, n_tri)
    t = 1
    for j in 0:(nside - 1), i in 0:(nside - 1)
        # vertex indices in the (nside+1)x(nside+1) grid (1-based)
        v00 = j * (nside + 1) + i + 1
        v10 = v00 + 1
        v01 = v00 + (nside + 1)
        v11 = v01 + 1
        sv[1, t] = v00; sv[2, t] = v10; sv[3, t] = v11
        t += 1
        sv[1, t] = v00; sv[2, t] = v11; sv[3, t] = v01
        t += 1
    end

    lag = SimplicialMesh{2, Float64}(positions, sv, sn)

    eul = HierarchicalMesh{2}()
    # Uniform refinement: refine every leaf at each level.
    for _ in 1:n_eul_refine
        leaves = enumerate_leaves(eul)
        refine_cells!(eul, leaves)
    end

    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    return (lag = lag, frame = frame)
end

"""
    run_compute_overlap(state, backend)

Compute the geometric overlap. For PR-0 we exercise the existing
`parallel = true` path; the `backend` argument selects the scheduler
when the backend is an `OhMyThreadsBackend`, and forces a sequential
run when it's `Sequential()`.
"""
function run_compute_overlap(state, backend::AbstractParallelBackend)
    lag = state.lag
    frame = state.frame
    if backend isa Sequential
        return compute_overlap(lag, frame; moment_order = 2,
                               parallel = false)
    else
        # OhMyThreadsBackend
        return compute_overlap(lag, frame; moment_order = 2,
                               parallel = true,
                               scheduler = backend.scheduler)
    end
end

# ----------------------------------------------------------------------------
# Workload :refine_by_indicator
# ----------------------------------------------------------------------------

"""
    build_refine_by_indicator(size) -> NamedTuple

Construct a 2D `HierarchicalMesh` plus a deterministic per-cell indicator
function for benchmarking the parallel candidate-evaluation passes.

  :small  → ~16 leaves   (n_levels = 2)
  :medium → ~256 leaves  (n_levels = 4)
  :large  → ~4096 leaves (n_levels = 6)

The indicator function performs a small CPU-bound computation per cell so
the per-cell cost is representative of a real refinement criterion (an
order-of-magnitude above the bare ~100 ns leaf-flag check). It returns a
deterministic value computed from the cell index, so candidate ordering
is stable across runs and parallel/sequential parity tests reproduce.
"""
function build_refine_by_indicator(size::Symbol)
    if size === :small
        n_levels = 2
    elseif size === :medium
        n_levels = 4
    elseif size === :large
        n_levels = 6
    else
        throw(ArgumentError("Unknown size: $size"))
    end

    mesh = HierarchicalMesh{2}()
    for _ in 1:n_levels
        leaves = [ci for ci in 1:n_cells(mesh) if is_leaf(mesh.cells[ci])]
        refine_cells!(mesh, leaves)
    end

    # CPU-bound deterministic indicator: a 32-iteration recurrence per
    # cell, seeded by the cell index. ~hundreds of ns per cell — small
    # enough to keep the workload candidate-evaluation-bound, large
    # enough that thread parallelism is observable on medium/large.
    indicator = function (ci::Integer)
        x = Float64(ci) * 0.123456789
        @inbounds for k in 1:32
            x = sin(x) + k * 1e-3
        end
        return x
    end

    return (mesh = mesh, indicator = indicator)
end

"""
    run_refine_by_indicator(state, backend)

Run a single `refine_by_indicator!` pass on a fresh copy of the mesh
under the supplied backend. The mesh is `deepcopy`-ed inside the run so
repeated benchmark samples don't accumulate refinement (the
mesh-mutation step itself is sequential, but each sample exercises the
parallel candidate-evaluation pass on the same starting topology).
"""
function run_refine_by_indicator(state, backend::AbstractParallelBackend)
    m = deepcopy(state.mesh)
    return refine_by_indicator!(m, state.indicator;
                                  refine_threshold = 0.0,   # always-true predicate
                                  backend = backend)
end

# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

const WORKLOADS = Dict{Symbol, NamedTuple}(
    :compute_overlap => (
        build = build_compute_overlap,
        run   = run_compute_overlap,
        sizes = [:small, :medium, :large],
    ),
    :refine_by_indicator => (
        build = build_refine_by_indicator,
        run   = run_refine_by_indicator,
        sizes = [:small, :medium, :large],
    ),
)
