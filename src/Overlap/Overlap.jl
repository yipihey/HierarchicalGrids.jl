"""
    Overlap

Layer 4: geometric-overlap and remap operators.

Computes the overlap polytopes between a Lagrangian simplicial mesh
and an Eulerian hierarchical mesh, with full polynomial moment integration
per overlap pair, then provides moment-based remap operators (mass-weighted
average, conservative flux, law of total covariance) that move fields
between the two meshes through the precomputed overlap.

The geometric machinery is delegated to a polytope-clipping backend via
the small `r3d_adapter` interface. The adapter currently calls r3djl
(https://github.com/yipihey/r3djl), which provides validated `init_simplex!`,
`clip!`, and `moments!` for D = 2 and D = 3. r3djl's Phase 3 (D ≥ 4)
has mostly landed — `clip!` (single + sequential) and order-0 `moments!`
are correct, but higher-order `moments!` are still pending — so this
module currently throws on D ≥ 4. See `lifting.jl` for the planned
cubic-edge dimension lifting and the remaining r3djl blocker.

# Module structure

- `frame.jl` — `EulerianFrame` (physical-coordinate wrapper for a
  `HierarchicalMesh`), unit→physical box mapping, AABB-AABB test.
- `data.jl` — `OverlapEntry`, `GeometricOverlap`, `OverlapBuilder`.
- `moments.jl` — graded-lex moment indexing and `shift_moments!`.
- `aabb.jl` — `SimplicialAABBTree` for broad-phase pruning.
- `r3d_adapter.jl` — the only file that knows about r3djl. Contains
  `PairScratch` and `overlap_simplex_box!`.
- `compute.jl` — `compute_overlap`, `install_r3d_overlap!`.
- `remap.jl` — `RemapOperator` types and `remap_l_to_e!`/`remap_e_to_l!`.
- `lifting.jl` — placeholder for cubic-edge dimension lifting; blocked
  on r3djl D ≥ 4 simplex clip + higher-order moments.

# Quick start

```julia
using HierarchicalGrids

# Build the two meshes
lag = SimplicialMesh{2, Float64}(positions, simplex_vertices, simplex_neighbors)
eul = HierarchicalMesh{2}()
refine_cells!(eul, [1])  # any AMR you want

# Wrap the Eulerian mesh in a physical frame
frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

# Compute the overlap (sequential; one pass over Eulerian leaves with
# BVH broad-phase over Lagrangian simplices)
overlap = compute_overlap(lag, frame; moment_order = 3)

# Move a Lagrangian-side scalar field to the Eulerian side
src = rand(n_simplices(lag))
dst = zeros(n_cells(eul))
remap_l_to_e!(dst, src, overlap, MassWeightedAverage())
```
"""
module Overlap

using StaticArrays
using LinearAlgebra: det
import OhMyThreads
using ..Mesh: HierarchicalMesh, SimplicialMesh, CellMeta, PairedMesh,
              n_cells, n_simplices, is_leaf, simplex_vertex_positions,
              set_overlap_compute_function!,
              level_of, find_children, isotropic_mask
using ..Mesh  # for ROOT_PARENT, ensure_caches!, _parents
using ..BoundaryConditions: BCKind, BoundarySpec, default_bc, validate
import ..BoundaryConditions: is_periodic_axis
using ..Geometry: cell_extent
using ..Bases: AbstractBasis, MonomialBasis, n_coeffs
using ..Storage: PolynomialFieldSet, PolynomialFieldView, PolynomialView,
                  AbstractLayout, SoA, AoS, Blocked, basis_of, n_elements,
                  _layout_type_poly
using ..Diagnostics: RemapDiagnostics

# Types/functions
export EulerianFrame, root_box, cell_unit_box, cell_physical_box, enumerate_leaves,
       aabbs_overlap

# Boundary-condition companion (PR-D)
export FrameBoundaries, bc

export OverlapEntry, GeometricOverlap, OverlapBuilder
export n_entries, moment_order, entries_for_lag, entries_for_eul, total_overlap_volume
export push_overlap!, merge_builder!, finalize_overlap

export moments_length, moment_multiindices, moment_index,
       moment_volume, moment_centroid,
       shift_moments!, shift_moments

export SimplicialAABBTree, simplex_aabb, build_simplex_aabb_tree,
       query_aabb!, query_aabb, n_nodes

export PairScratch, overlap_simplex_box!

export compute_overlap, install_r3d_overlap!

export RemapOperator, MassWeightedAverage, ConservativeFlux,
       TotalCovariance, TotalCumulants
export remap_l_to_e!, remap_e_to_l!, remap_l_to_e_covariance!

# Cubic-edge dimension lifting: stubs only — exported neither at the
# Overlap module level nor at the top level until r3djl supplies higher
# moments in D ≥ 4. The functions exist (in `lifting.jl`) so that callers
# attempting `compute_overlap(...; edge_kind = :cubic)` get a clear error
# pointing at the right place; they aren't part of the public surface.

# Polynomial-aware remap (laws of total expectation through polynomial
# reconstructions, not piecewise-constant-per-cell)
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

include("frame.jl")
include("frame_boundaries.jl")
include("moments.jl")
include("data.jl")
include("aabb.jl")
include("r3d_adapter.jl")
include("compute.jl")
include("remap.jl")
include("polynomial_remap.jl")
include("polynomial_remap_fieldset.jl")
include("polynomial_remap_streaming.jl")
include("lifting.jl")

end # module Overlap
