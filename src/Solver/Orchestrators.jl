# ============================================================================
# Orchestrators: for_each_cell! and for_each_face!
#
# PR-7 — the kernel-driving layer that consumes the per-cell views from PR-6.
#
# Both orchestrators dispatch through PR-0's parallel-backend trait
# (`AbstractParallelBackend` + `parallel_foreach`). Mesh caches and the
# neighbor graph are pre-built before the parallel section so the lazy
# initializers (which are not thread-safe) never fire from within a task.
# This is the lesson PR-1/PR-2 surfaced: pre-warm before fan-out.
#
# The `ctx` argument is forward-looking — `KernelContext` lands in PR-9.
# For PR-7 we accept any object (including `nothing`) and pass it through
# to the kernel unchanged.
# ============================================================================

# ----------------------------------------------------------------------------
# Helpers: axis unit vectors
# ----------------------------------------------------------------------------

# Build a unit vector along axis `a` of length-`D`. The orchestrator hands
# the kernel a normal pointing from the lower-indexed cell to the higher,
# so the +axis form is always `+1` on `a` and `0` elsewhere.
@inline function _axis_unit_vector(::Val{D}, a::Int) where {D}
    return ntuple(d -> d == a ? 1.0 : 0.0, Val(D))
end

# Outward normal for a boundary face at `(axis, side)`. `side == 1` is the
# lo wall (outward normal is -axis); `side == 2` is the hi wall (+axis).
@inline function _signed_axis_vector(::Val{D}, a::Int, side::Int) where {D}
    s = side == 2 ? 1.0 : -1.0
    return ntuple(d -> d == a ? s : 0.0, Val(D))
end

# ----------------------------------------------------------------------------
# Face enumeration
# ----------------------------------------------------------------------------

# Each interior face is represented as `(i, j, axis, hanging::Bool)` where
# `i` is the lower-indexed (or coarser) cell, `j` is the neighbor, and
# `axis ∈ 1..D`. We enumerate from `i`'s +axis side only, so each
# conforming face is dispatched exactly once.
#
# For hanging-node faces (a coarse cell `i` with multiple fine neighbors
# on the +axis side), we emit one entry per fine neighbor with `hanging=true`.
# The conservation guarantee still holds because the fine neighbors
# `j > i` are reached only from this coarse cell's +axis enumeration.
function _enumerate_interior_faces(mesh::HierarchicalMesh{D}) where {D}
    faces = Tuple{Int, Int, Int, Bool}[]
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        nbs = face_neighbors(mesh, i)
        for axis in 1:D
            face_idx = 2 * axis            # +axis side
            rep = nbs[face_idx]
            rep == 0 && continue
            # Hanging-node detection: ask for the full fine-neighbor list.
            # `face_fine_neighbors` returns a one-element vector for a
            # conforming face and a multi-element vector for hanging-node.
            fines = face_fine_neighbors(mesh, i, face_idx)
            if length(fines) >= 2
                # Coarse-fine contact: emit one face per fine neighbor.
                # The coarse cell `i` is necessarily lower-indexed than
                # its descendants (DFS order in the mesh), but we still
                # filter `j > i` defensively.
                for f in fines
                    j = Int(f)
                    j > i || continue
                    push!(faces, (i, j, axis, true))
                end
            else
                j = Int(rep)
                # Visit each conforming face from its lower-indexed side.
                j > i || continue
                push!(faces, (i, j, axis, false))
            end
        end
    end
    return faces
end

# Boundary faces: cells with a 0 entry in `face_neighbors` on some side.
# Each entry is `(i, axis, side)` where `side ∈ {1=lo, 2=hi}`.
function _enumerate_boundary_faces(mesh::HierarchicalMesh{D}) where {D}
    out = Tuple{Int, Int, Int}[]
    @inbounds for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        nbs = face_neighbors(mesh, i)
        for axis in 1:D
            # lo side
            if nbs[2 * axis - 1] == 0
                push!(out, (i, axis, 1))
            end
            # hi side
            if nbs[2 * axis] == 0
                push!(out, (i, axis, 2))
            end
        end
    end
    return out
end

# ----------------------------------------------------------------------------
# for_each_cell!
# ----------------------------------------------------------------------------

"""
    for_each_cell!(kernel, fields_out, fields_in, frame::EulerianFrame{D, Float64};
                   ghost_depth::Int = 0,
                   bcs::Union{Nothing, FrameBoundaries{D}} = nothing,
                   ctx = nothing,
                   backend::AbstractParallelBackend = default_backend())

Apply `kernel(cv::CellView, hv::HaloView, ctx)` to every leaf cell of
`frame.mesh` in parallel. The kernel reads from `fields_in` (via `cv` and
`hv`) and writes to `fields_out` (via `cv`).

Synchronization is implicit: the call returns only after every kernel
invocation completes; subsequent calls observe the writes.

Per-cell `CellView` and `HaloView` are constructed inside the parallel
section. The mesh's caches (`Mesh.ensure_caches!`) and neighbor graph
(`ensure_neighbor_graph!`) are pre-built before fan-out — kernels MUST
NOT trigger lazy cache initialization or graph builds, as those are not
thread-safe.

For `ghost_depth == 0`, a depth-1 `HaloView` is still constructed (the
`HaloView` constructor requires `ghost_depth ≥ 1`); kernels that don't
reach into the halo simply ignore it. For `ghost_depth ≥ 1`, kernels
can call `hv[Val(:rho), off]` with `sum(abs, off) ≤ ghost_depth`.

The `ctx` argument is a forward-looking placeholder for
`KernelContext` (PR-9). For PR-7 it's an opaque pass-through.
"""
function for_each_cell!(kernel, fields_out, fields_in,
                          frame::EulerianFrame{D, Float64};
                          ghost_depth::Int = 0,
                          bcs::Union{Nothing, FrameBoundaries{D}} = nothing,
                          ctx = nothing,
                          backend::AbstractParallelBackend = default_backend()
                          ) where {D}
    ghost_depth >= 0 ||
        throw(ArgumentError("for_each_cell!: ghost_depth must be ≥ 0, got $ghost_depth"))

    mesh = frame.mesh

    # Pre-warm caches before any parallel fan-out (PR-1 lesson).
    Mesh.ensure_caches!(mesh)
    # The HaloView constructor below requires ghost_depth ≥ 1. We always
    # ensure the neighbor graph too, since even a depth-1 halo will walk
    # face_neighbors lazily otherwise.
    ensure_neighbor_graph!(mesh)

    leaves = enumerate_leaves(mesh)
    # `HaloView` requires ghost_depth ≥ 1; clamp the constructor depth so
    # a kernel with ghost_depth=0 still receives a usable (but unused) halo.
    halo_depth = max(1, ghost_depth)

    parallel_foreach(backend,
                     i -> begin
                         cv = cell_view(fields_in, fields_out, frame, Int(i))
                         hv = halo_view_multi(fields_in, mesh, Int(i), halo_depth; bcs = bcs)
                         kernel(cv, hv, ctx)
                         return nothing
                     end,
                     leaves)

    return nothing
end

# ----------------------------------------------------------------------------
# for_each_face!
# ----------------------------------------------------------------------------

"""
    for_each_face!(flux_kernel, fluxes_out, fields_in,
                   frame::EulerianFrame{D, Float64};
                   bcs::Union{Nothing, FrameBoundaries{D}} = nothing,
                   flux_kernel_boundary = nothing,
                   ctx = nothing,
                   backend::AbstractParallelBackend = default_backend())

Apply `flux_kernel(left_cv, right_cv, normal::NTuple{D, Float64}, ctx)`
to every interior face of `frame.mesh`. `left_cv` is the lower-indexed
cell's `CellView` and `right_cv` is its neighbor. `normal` is a unit
vector pointing from `left_cv` to `right_cv`.

The flux kernel is responsible for writing into `fluxes_out` (typically
a NamedTuple of face-indexed buffers). The cell views are read-only by
convention — they are constructed with `fields_in` bound to both the
`fields_in` and `fields_out` slots of `CellView`, so any accidental
write would clobber the input.

For boundary faces, if `flux_kernel_boundary` is provided it is called as

    flux_kernel_boundary(cv, axis::Int, side::Int,
                         normal::NTuple{D, Float64}, bcs, ctx)

where `side == 1` is the lo wall, `side == 2` is the hi wall, and
`normal` is the OUTWARD unit normal (pointing out of the domain).

Otherwise boundary faces are skipped — kernels relying on `for_each_cell!`
with a `HaloView` for BC handling can use that pathway instead.

Conservation: each interior face is dispatched exactly once. The
orchestrator visits each face from its lower-indexed cell's +axis side.

Hanging-node faces (unbalanced refinement): each fine neighbor on a
coarse-fine contact is dispatched as a separate face, with the COARSE
cell as `left_cv`. A face with `n` fine neighbors triggers `n` calls.

Caches and the neighbor graph are pre-built before the parallel section.
The `ctx` placeholder is the same forward-looking hook as `for_each_cell!`.
"""
function for_each_face!(flux_kernel, fluxes_out, fields_in,
                          frame::EulerianFrame{D, Float64};
                          bcs::Union{Nothing, FrameBoundaries{D}} = nothing,
                          flux_kernel_boundary = nothing,
                          ctx = nothing,
                          backend::AbstractParallelBackend = default_backend()
                          ) where {D}
    mesh = frame.mesh

    # Pre-warm caches and the neighbor graph before any parallel fan-out.
    Mesh.ensure_caches!(mesh)
    ensure_neighbor_graph!(mesh)

    # Build face lists deterministically (sequential pre-work).
    interior = _enumerate_interior_faces(mesh)
    boundary = _enumerate_boundary_faces(mesh)

    # Dispatch interior faces in parallel. The CellView is constructed
    # with `fields_in` bound to both slots: the flux pass treats cells
    # as read-only and writes to `fluxes_out` instead.
    parallel_foreach(backend,
                     face -> begin
                         i, j, axis, _hanging = face
                         cv_left  = cell_view(fields_in, fields_in, frame, i)
                         cv_right = cell_view(fields_in, fields_in, frame, j)
                         normal = _axis_unit_vector(Val(D), axis)
                         flux_kernel(cv_left, cv_right, normal, ctx)
                         return nothing
                     end,
                     interior)

    # Boundary faces are typically a small minority; dispatch sequentially
    # so the dispatch order is deterministic without coordination.
    if flux_kernel_boundary !== nothing
        for face in boundary
            i, axis, side = face
            cv = cell_view(fields_in, fields_in, frame, i)
            normal = _signed_axis_vector(Val(D), axis, side)
            flux_kernel_boundary(cv, axis, side, normal, bcs, ctx)
        end
    end

    # Silence unused-arg warnings: `fluxes_out` is the kernel's write
    # target and the orchestrator never touches it directly.
    fluxes_out === fluxes_out  # no-op; keep the binding live
    return nothing
end
