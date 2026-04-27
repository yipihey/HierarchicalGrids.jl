"""
    HierarchicalGrids

A flexible framework for hierarchical adaptive grids in arbitrary dimensions
with integer-exact geometry and decoupled memory layouts.

# Architecture

The library is organized in layers, each depending only on layers below:

- **Layer 0 — BitPrimitives**: hardware bit operations (POPCNT, PDEP, PEXT)
  and integer width selection. Pure functions, no allocations.
- **Layer 1 — Mesh**: cell metadata, hierarchical tree structure, anisotropic
  refinement, lazy caches. Geometry only — no fields, no physics.
- **Layer 2 — Geometry**: integer-exact predicates, volumes, axis-aligned
  clipping. All operations exact in chosen integer types.
- **Layer 2.5 — Storage**: layout-flexible field storage. Same access syntax,
  different memory layouts (SoA, AoS, Blocked).
- **Layer 3 — Threading & Memory**: chunk-based parallelism mirroring future
  MPI structure; pool allocation for fragmentation resistance.

# Design principles

The framework follows a few non-negotiable principles:

1. **Mesh purity**: the mesh contains only geometric/structural data.
   Physics, particles, fields are separate.
2. **Hierarchical relative coordinates**: cells refer to positions relative
   to their parent. No global coordinate system. Deep zoom doesn't widen
   integer types.
3. **Integer-exact geometry**: vertices are signed integers, cell-center-
   relative. Operations stay in hardware-native arithmetic.
4. **Cell-center-relative positions**: per-cell storage uses signed offsets
   from cell center, naturally symmetric for arithmetic.
5. **Composition over inheritance**: new features add new types, not new
   members of existing types.
6. **Lazy caches**: derived data (level, parent, subtree size) is computed
   on demand and invalidated on AMR.
7. **Single shared address space**: threading is the parallelism model;
   chunk-based design preserves future MPI option.
8. **Pool allocation**: short-lived buffers go through pools to prevent
   fragmentation; long-lived data is allocated once.

See `docs/architecture.md` for detailed rationale.
"""
module HierarchicalGrids

# Layer 0
include("BitPrimitives/BitPrimitives.jl")
using .BitPrimitives
export count_ones_native, leading_zeros_native, trailing_zeros_native,
       pdep, pext, bit_for_axis, sibling_index_type, volume_int_type,
       split_mask_type, vertex_int_type

# Layer 1 (boundary-condition vocabulary, used by Mesh and Overlap below)
include("Mesh/BoundaryConditions.jl")
using .BoundaryConditions
export BCKind, PERIODIC, INFLOW, OUTFLOW, REFLECTING, DIRICHLET
export BoundarySpec, default_bc
# `is_periodic_axis` and `validate` are deliberately not re-exported at the
# top level: they live in BoundaryConditions and are also reachable via the
# `FrameBoundaries` accessors. Top-level re-export would risk colliding with
# downstream code; users that want them can `using .BoundaryConditions`.

# Layer 1
include("Mesh/Mesh.jl")
using .Mesh
export CellMeta, HierarchicalMesh
export FLAG_LEAF, FLAG_BOUNDARY, FLAG_DIRTY
export is_leaf, is_boundary, is_dirty
export sibling_index, split_mask, flags
export FULLY_ISOTROPIC_MASK, isotropic_mask
export n_cells, root_cell_index
export find_parent, find_children, find_lca
export level_of, position_in_parent
export rebuild_caches!, invalidate_caches!
export refine_cells!, coarsen_cells!
export cell_path, CellPath, find_at_path
export RefinementEvent, ListenerHandle
export register_refinement_listener!, unregister_refinement_listener!

# Neighbor graph (PR-C)
export NeighborGraph, face_neighbors, face_fine_neighbors
export build_neighbor_graph, ensure_neighbor_graph!
export cell_adjacency_sparsity

# Boundary-aware neighbor wiring (PR-D)
export face_neighbors_with_bcs

# SimplicialMesh
export SimplicialMesh
export n_vertices, spatial_dimension, n_simplices
export vertex_position, reference_position, set_vertex_position!, set_reference_to_current!
export simplex_vertex_indices, simplex_vertex_positions, simplex_reference_positions
export simplex_neighbor, is_boundary_face
export simplex_volume, simplex_reference_volume
export deformation_gradient, volume_jacobian, distortion_metric
export has_inverted_simplex, max_distortion, enumerate_edges

# Periodic-wrap helper + Dirichlet pin (PR-D)
export periodic!, pin_boundary_simplices!, is_pinned

# Composite/Paired meshes
export CompositeMesh, PairedMesh
export ensure_overlap!, invalidate_overlap!, overlap_cache
export set_overlap_compute_function!
export update_lagrangian_positions!

# Indicator-driven refinement: re-exported below, alongside the
# `RefineByIndicator` submodule include (loaded after Threading).

# Layer 2
include("Geometry/Geometry.jl")
using .Geometry
export cell_extent, cell_center_in_parent
export cell_volume_at_level, cell_volume
export relative_position, distance_squared
export sign_of_axis, in_box
# 1D continuous geometry (interval primitives for remap)
export Interval, is_empty, interval_length, interval_intersection
export interval_contains, affine_map_to_reference, affine_map_from_reference

# Layer 2 (numerical-analysis primitives): polynomial bases
include("Bases/Bases.jl")
using .Bases
export AbstractBasis, MonomialBasis, BernsteinBasis, LagrangeBasis
export n_coeffs, evaluate, gradient
export all_bernstein_coeffs_positive, is_positive_certificate, change_basis
export bernstein_positivity_certificate

# Layer 2 (numerical-analysis primitives): Gauss quadrature
include("Quadrature/Quadrature.jl")
using .Quadrature
export QuadRule, gauss_quadrature_interval, gauss_quadrature_triangle
export gauss_quadrature_quad, gauss_quadrature_tetrahedron, gauss_quadrature_cube
export integrate, n_quad_points
export integrate_polynomial_on_subinterval, action_error_l2

# Layer 2.5
include("Storage/Storage.jl")
using .Storage
export AbstractLayout, SoA, AoS, Blocked
export FieldSet, n_elements, field_names
export resize_fields!, allocate_fields
export PolynomialFieldSet, allocate_polynomial_fields
export PolynomialFieldView, PolynomialView
export gradient_at, n_coeffs_per_element, basis_of
export polynomial_action_error, polynomial_action_error_per_element
export is_strictly_positive
export HaloView, halo_view

# Layer 3 (foundational pieces, full implementation later)
include("Threading/Threading.jl")
using .Threading
export ThreadChunk, partition_for_threads
export parallel_for_cells, parallel_reduce_cells, parallel_for_chunks
# Backend trait + verbs (PR-0).
export AbstractParallelBackend, Sequential, OhMyThreadsBackend
export default_backend, set_default_backend!
export parallel_foreach, parallel_mapreduce, parallel_chunked

# Layer 4 (depends on Mesh + Threading): generic indicator-driven refinement.
# Loaded after Threading so the parallel candidate-evaluation passes can
# reach the backend trait + verbs.
include("Mesh/RefineByIndicator.jl")
using .RefineByIndicator
export refine_by_indicator!

include("Memory/Memory.jl")
using .Memory
export FieldBufferPool, acquire_buffer!, release_buffer!, pool_stats
export ScratchBuffer, with_scratch
export Arena, allocate_in_arena, reset_arena!

# Layer 3 (foundational): hardware introspection and thread-pinning façade.
# Richer behavior comes from the Hwloc and ThreadPinning extensions.
include("Hardware/Hardware.jl")
using .Hardware
export topology_summary, pin_threads!

# Layer 3 (foundational): runtime diagnostics for self-consistency monitoring
include("Diagnostics/Diagnostics.jl")
using .Diagnostics
export WelfordStats, push_value!, count_samples, merge_stats!
export mean, variance, std_dev, skewness, kurtosis, excess_kurtosis
export ExponentialMovingAverage, update!, value, reset!
export PerCellStats
export RemapDiagnostics
export OverlapAuditReport, audit_overlap
export OverlapDropReport

# Layer 4: geometric-overlap and remap operators
include("Overlap/Overlap.jl")
using .Overlap
export EulerianFrame, root_box, cell_unit_box, cell_physical_box, enumerate_leaves,
       aabbs_overlap
export FrameBoundaries, bc
export OverlapEntry, GeometricOverlap, OverlapBuilder
export n_entries, entries_for_lag, entries_for_eul, total_overlap_volume
export push_overlap!, merge_builder!, finalize_overlap
export describe
export moments_length, moment_multiindices, moment_index,
       moment_volume, moment_centroid, shift_moments!, shift_moments
export SimplicialAABBTree, simplex_aabb, build_simplex_aabb_tree,
       query_aabb!, query_aabb, n_nodes
export PairScratch, overlap_simplex_box!
export IntPairScratch, overlap_simplex_box_exact!
# IntExact quantization helpers (PR-2)
export IntegerLattice, lat_resolution
export quantize, quantize_strict, dequantize
export unscale_volume, unscale_moment
export compute_overlap, install_r3d_overlap!
export RemapOperator, MassWeightedAverage, ConservativeFlux,
       TotalCovariance, TotalCumulants
export remap_l_to_e!, remap_e_to_l!, remap_l_to_e_covariance!
# `lift_cubic_triangle` and `project_lifted_moments` are stubs awaiting
# r3djl higher-order moments in D ≥ 4. They live in the Overlap module
# but aren't re-exported at the top level until they have implementations.

# Polynomial remap
export CellReferenceFrame, AxisAlignedRef, SimplicialRef
export eulerian_frame, lagrangian_frame
export reference_to_physical_pullback, reference_mass_matrix
export integrate_polynomial_over_overlap, accumulate_polynomial_rhs!
export polynomial_remap_l_to_e!, polynomial_remap_e_to_l!

# PolynomialFieldSet wrapper for the polynomial remap
export polynomial_coeffs_view, polynomial_coeffs_matrix,
       set_polynomial_coeffs_matrix!
export polynomial_remap_field!

# Streaming polynomial remap (single-pass via voxelize_fold!)
export uniform_grid_dimensions
export polynomial_remap_l_to_uniform_e!, polynomial_remap_uniform_e_to_l!

# Layer 4 (initialization): L²-projection of analytical functions onto
# PolynomialFieldSet. Depends on Bases, Quadrature, Storage, Mesh, and
# the Overlap submodule (for EulerianFrame and cell_physical_box).
include("Storage/Initialization.jl"); using .Initialization
export init_field_from!

# Layer 4 (AMR driver): topology-only `step_with_amr!` interleaver. Depends
# on Mesh (for refine_by_indicator!) and Overlap (for EulerianFrame). Lives
# under src/Mesh/ for organizational reasons but is a top-level submodule
# so it can reach both layers.
include("Mesh/AMR.jl")
using .AMR
export step_with_amr!

# Phase 2 (CFD orchestration): solver-facing per-cell views. Depends on
# Mesh (neighbor graph + level), BoundaryConditions, Storage (PolynomialFieldSet),
# and Overlap (EulerianFrame, FrameBoundaries, cell_physical_box).
#
# Note: The Solver submodule defines a multi-field `HaloView` type that
# coexists with the single-field `Storage.HaloView` introduced in PR-C.
# Both are accessible via their submodules; we re-export `CellView` and
# the new constructors at the top level but deliberately do NOT re-export
# `Solver.HaloView` to avoid colliding with `Storage.HaloView`. Reach the
# multi-field halo via `HierarchicalGrids.Solver.HaloView` (or
# `using .Solver: HaloView` in user code).
include("Solver/Solver.jl")
using .Solver: CellView, ghost_depth, cell_view, halo_view_multi,
                for_each_cell!, for_each_face!, KernelContext,
                BlockView, BlockHaloView, block_view, block_halo_view,
                for_each_block!,
                AdaptiveField, dispose!,
                PatchHierarchy, PatchBoundaryBC, PatchView, PatchHaloView,
                add_patches!, for_each_patch!,
                restrict_to_parents!, prolong_from_parents!
export CellView, BlockView, BlockHaloView
export ghost_depth, cell_view, halo_view_multi, block_view, block_halo_view
export for_each_cell!, for_each_face!, for_each_block!
export KernelContext
export AdaptiveField, dispose!
# PR-13: PatchHierarchy + for_each_patch! orchestrator. The patch-hierarchy
# accessors `n_levels`, `n_patches`, `patches_at`, and `validate` are
# deliberately NOT re-exported at the top level — they live in `Solver`
# alongside `PatchHierarchy` and are available via `Solver.<name>` or
# `using .Solver: <name>`. Top-level re-export risks colliding with
# user-defined `validate` / `n_levels`.
export PatchHierarchy, PatchBoundaryBC, PatchView, PatchHaloView
export add_patches!, for_each_patch!
export restrict_to_parents!, prolong_from_parents!

end # module HierarchicalGrids
