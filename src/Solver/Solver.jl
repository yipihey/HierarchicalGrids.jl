"""
    Solver

Phase 2 application-facing solver layer. Provides `CellView` and `HaloView`,
the per-cell read/write accessors that solver kernels use to consume per-cell
polynomial coefficients and BC-resolved neighbor values.

Phase 2's orchestrators (PR-7) build these per-cell, hand them to user
kernels, and write back.

# Types

- `CellView{Names, Tin, Tout, D, T}` — per-cell view over a tuple of
  named polynomial-field-sets. Reads via `cv[:rho]`, writes via
  `cv[:rho] = ...`. Reads come from the input field-set binding;
  writes land in the output field-set binding. Property-style metadata:
  `cv.coords`, `cv.volume`, `cv.level`, `cv.index`.
- `HaloView{Names, Tin, D, T, GhostDepth, BC}` — per-cell halo view.
  `hv[:rho, (1, 0)]` returns the +x neighbor's coefficients, with
  BC-resolution at domain boundaries (PERIODIC wraps; REFLECTING/OUTFLOW
  reflect to the central cell; DIRICHLET/INFLOW return `nothing`
  pending the BC-value source from PR-13).
"""
module Solver

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

using ..Mesh
using ..Mesh: HierarchicalMesh, n_cells, is_leaf, level_of, children_count
using ..Mesh: face_neighbors, face_fine_neighbors, face_neighbors_with_bcs,
              ensure_neighbor_graph!
using ..Mesh: RefinementEvent, ListenerHandle,
              register_refinement_listener!, unregister_refinement_listener!
using ..BoundaryConditions: BCKind, BoundarySpec, is_periodic_axis,
                            PERIODIC, INFLOW, OUTFLOW, REFLECTING, DIRICHLET
using ..Bases: AbstractBasis, MonomialBasis, BernsteinBasis, n_coeffs, evaluate
using ..Storage
using ..Storage: PolynomialFieldSet, PolynomialFieldView, PolynomialView,
                 AbstractLayout, SoA, AoS, Blocked,
                 _layout_type_poly, _get_poly_coeff, _set_poly_coeff!,
                 n_elements, field_names, basis_of, n_coeffs_per_element
using ..Overlap: EulerianFrame, FrameBoundaries, cell_unit_box, cell_physical_box,
                  enumerate_leaves
using ..Threading: AbstractParallelBackend, Sequential, OhMyThreadsBackend,
                    default_backend, parallel_foreach

using OhMyThreads.TaskLocalValues: TaskLocalValue

# ----------------------------------------------------------------------------
# Includes
# ----------------------------------------------------------------------------

include("Views.jl")
include("BlockView.jl")
include("Orchestrators.jl")
include("KernelContext.jl")
include("AdaptiveField.jl")

# ----------------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------------

export CellView, HaloView, BlockView, BlockHaloView
export ghost_depth, cell_view, halo_view_multi, block_view, block_halo_view
export for_each_cell!, for_each_face!, for_each_block!
export KernelContext
export AdaptiveField, dispose!

end # module Solver
