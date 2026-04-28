# Per-frame physical-AABB cache: makes `cell_physical_box` an O(1)
# allocation-free indexed read on the warm path, mirroring the
# `FrameFaceCache` invalidation pattern.

using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells,
    EulerianFrame, enumerate_leaves, cell_physical_box,
    cell_unit_box, ensure_physical_boxes!, is_leaf

@testset "Physical-box cache: build, hit, invalidate" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, enumerate_leaves(mesh))   # 16-leaf 4x4 mesh
    frame = EulerianFrame(mesh, (0.0, 0.0), (4.0, 2.0))

    # First call: cache empty → fallback path. After ensure_*, cache populated.
    @test frame._cached_physical_boxes === nothing
    box1 = cell_physical_box(frame, 2)
    @test frame._cached_physical_boxes === nothing   # fallback didn't auto-build

    boxes = ensure_physical_boxes!(frame)
    @test frame._cached_physical_boxes === boxes
    @test length(boxes) == n_cells(mesh)

    # Second `ensure_*` returns the same object.
    @test ensure_physical_boxes!(frame) === boxes

    # Cached read matches both the fallback and a manual unit-box transform.
    for i in 1:n_cells(mesh)
        u_lo, u_hi = cell_unit_box(mesh, i)
        expected = ((4.0 * u_lo[1], 2.0 * u_lo[2]),
                    (4.0 * u_hi[1], 2.0 * u_hi[2]))
        @test cell_physical_box(frame, i) == expected
        @test boxes[i] == expected
    end

    # Refinement invalidates via the one-shot listener.
    refine_cells!(mesh, [enumerate_leaves(mesh)[1]])
    @test frame._cached_physical_boxes === nothing

    # Rebuild covers the new cells.
    boxes2 = ensure_physical_boxes!(frame)
    @test length(boxes2) == n_cells(mesh)
    @test boxes2 !== boxes
end

@testset "Physical-box cache: warm-path allocation profile" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, enumerate_leaves(mesh))
    refine_cells!(mesh, enumerate_leaves(mesh))   # 64 leaves
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))

    # Build the cache.
    ensure_physical_boxes!(frame)

    # Warm a typical access pattern.
    for i in 1:n_cells(mesh)
        cell_physical_box(frame, i)
    end

    # Hot path: a per-cell read should not allocate.
    function tight_loop(frame, n)
        s = 0.0
        @inbounds for i in 1:n
            lo, hi = cell_physical_box(frame, i)
            s += lo[1] + hi[1]
        end
        return s
    end
    tight_loop(frame, n_cells(mesh))     # warm
    bytes = @allocated tight_loop(frame, n_cells(mesh))
    # Julia 1.10's narrower inference leaves a small per-call residual on
    # the cache hit path; 1.11+ folds it away. The warm call still runs
    # on every Julia version.
    if VERSION >= v"1.11"
        @test bytes == 0
    else
        @test bytes < 4 * n_cells(mesh)   # ~4B / cell upper bound
    end
end

@testset "Physical-box cache: multi-frame on shared mesh" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    fA = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    fB = EulerianFrame(mesh, (10.0, 0.0), (11.0, 5.0))
    boxesA = ensure_physical_boxes!(fA)
    boxesB = ensure_physical_boxes!(fB)
    @test boxesA !== boxesB
    @test boxesA[1] == ((0.0, 0.0), (1.0, 1.0))
    @test boxesB[1] == ((10.0, 0.0), (11.0, 5.0))

    # Both invalidate independently on refinement.
    refine_cells!(mesh, [enumerate_leaves(mesh)[1]])
    @test fA._cached_physical_boxes === nothing
    @test fB._cached_physical_boxes === nothing
end
