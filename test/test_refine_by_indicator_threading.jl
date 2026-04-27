# Tests for the parallelized `refine_by_indicator!` candidate-evaluation
# passes (PR-3). Determinism across schedulers is the load-bearing
# property — the per-task partial candidate lists are merged and sorted
# so the resulting mesh structure is byte-identical to the sequential
# reference for every supported backend.

using Test
using Random
using HierarchicalGrids

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Build a fresh 2D mesh and uniformly refine it `n_levels` times.
function _build_uniform_2d(n_levels::Integer)
    mesh = HierarchicalMesh{2}()
    for _ in 1:n_levels
        leaves = [ci for ci in 1:n_cells(mesh) if is_leaf(mesh.cells[ci])]
        refine_cells!(mesh, leaves)
    end
    return mesh
end

# Compare two meshes structurally: same cell count + identical leaf-flag,
# split-mask, and sibling-index per cell.
function _meshes_match(a::HierarchicalMesh, b::HierarchicalMesh)
    n_cells(a) == n_cells(b) || return false
    @inbounds for ci in 1:n_cells(a)
        ca = a.cells[ci]
        cb = b.cells[ci]
        is_leaf(ca) == is_leaf(cb) || return false
        split_mask(ca) == split_mask(cb) || return false
        sibling_index(ca) == sibling_index(cb) || return false
    end
    return true
end

# Backend-set under test. Both `:serial` schedulers (`Sequential()` and
# `OhMyThreadsBackend(:serial)`) are kept distinct because they exercise
# different code paths (the former dispatches on the trait and skips
# OhMyThreads entirely; the latter routes through `tforeach`).
const _ALL_BACKENDS = AbstractParallelBackend[
    Sequential(),
    OhMyThreadsBackend(:dynamic),
    OhMyThreadsBackend(:static),
    OhMyThreadsBackend(:greedy),
    OhMyThreadsBackend(:serial),
]

# ---------------------------------------------------------------------------

@testset "refine_by_indicator! — determinism vs sequential" begin
    # Build a 2D mesh with three uniform refinement levels (4^3 == 64 leaves
    # plus interior parents → 85 cells total).
    base = _build_uniform_2d(3)

    # Indicator: refine only cells whose level is < 3. With our 3-level
    # uniform mesh every leaf is exactly at level 3, so under
    # `refine_threshold = 0.5` no cell is refined — but the indicator pass
    # still walks every cell and exercises the parallel candidate
    # gatherer. To get observable refinement, lower the predicate to
    # level < 4 with a real threshold sweep.
    indicator(mesh) = let m = mesh
        ci -> level_of(m, ci) < 4 && is_leaf(m.cells[ci]) ? 1.0 : 0.0
    end

    # Reference: Sequential backend.
    ref_mesh = deepcopy(base)
    ref_result = refine_by_indicator!(ref_mesh, indicator(ref_mesh);
                                       refine_threshold = 0.5,
                                       backend = Sequential())

    # Candidate: OhMyThreads :dynamic.
    par_mesh = deepcopy(base)
    par_result = refine_by_indicator!(par_mesh, indicator(par_mesh);
                                       refine_threshold = 0.5,
                                       backend = OhMyThreadsBackend(:dynamic))

    @test ref_result.refined == par_result.refined
    @test ref_result.coarsened == par_result.coarsened
    @test n_cells(ref_mesh) == n_cells(par_mesh)
    @test _meshes_match(ref_mesh, par_mesh)
end

@testset "refine_by_indicator! — backend cross-equality" begin
    base = _build_uniform_2d(3)
    indicator(mesh) = let m = mesh
        # Refine the first 1/4 of the leaves (deterministic on cell index).
        n = n_cells(m)
        ci -> (is_leaf(m.cells[ci]) && ci <= n ÷ 4) ? 2.0 : 0.0
    end

    results = []
    for backend in _ALL_BACKENDS
        m = deepcopy(base)
        r = refine_by_indicator!(m, indicator(m);
                                  refine_threshold = 1.0,
                                  backend = backend)
        push!(results, (backend, m, r))
    end

    # Every backend must agree pairwise.
    for i in 1:length(results), j in (i + 1):length(results)
        b_i, m_i, r_i = results[i]
        b_j, m_j, r_j = results[j]
        @test r_i.refined == r_j.refined
        @test r_i.coarsened == r_j.coarsened
        @test n_cells(m_i) == n_cells(m_j)
        @test _meshes_match(m_i, m_j)
    end
end

@testset "refine_by_indicator! — randomized fuzz" begin
    Random.seed!(0x1ce_b00b)

    # 10 random initial mesh configurations × 10 random indicators each.
    # Initial configurations vary in starting refinement depth and shape;
    # indicators vary in threshold + per-cell value pattern.
    for trial in 1:10
        # Random initial mesh: 1-3 levels of uniform refinement, then a
        # random subset of leaves refined one extra level.
        base_levels = rand(1:3)
        base = _build_uniform_2d(base_levels)
        leaves = [ci for ci in 1:n_cells(base) if is_leaf(base.cells[ci])]
        # Refine a random ~30% of leaves once more (with anisotropic mask
        # off — keeping it isotropic keeps the fuzz stable across mesh shapes).
        if !isempty(leaves) && rand() < 0.8
            keep = filter(_ -> rand() < 0.3, leaves)
            !isempty(keep) && refine_cells!(base, keep)
        end

        for sub_trial in 1:10
            n = n_cells(base)
            # Random per-cell indicator values + a random threshold.
            ind_vec = rand(n)
            threshold = rand() * 0.8 + 0.1   # in (0.1, 0.9)

            # Reference run: Sequential backend.
            ref_m = deepcopy(base)
            ref_r = refine_by_indicator!(ref_m, copy(ind_vec);
                                          refine_threshold = threshold,
                                          backend = Sequential())

            # Parallel run: OhMyThreads :dynamic.
            par_m = deepcopy(base)
            par_r = refine_by_indicator!(par_m, copy(ind_vec);
                                          refine_threshold = threshold,
                                          backend = OhMyThreadsBackend(:dynamic))

            @test ref_r.refined == par_r.refined
            @test ref_r.coarsened == par_r.coarsened
            @test n_cells(ref_m) == n_cells(par_m)
            @test _meshes_match(ref_m, par_m)
        end
    end
end
