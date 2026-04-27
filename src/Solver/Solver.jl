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
using ..Mesh: HierarchicalMesh, n_cells, is_leaf, level_of
using ..Mesh: face_neighbors, face_fine_neighbors, face_neighbors_with_bcs,
              ensure_neighbor_graph!
using ..BoundaryConditions: BCKind, BoundarySpec, is_periodic_axis,
                            PERIODIC, INFLOW, OUTFLOW, REFLECTING, DIRICHLET
using ..Storage: PolynomialFieldSet, PolynomialFieldView, PolynomialView
using ..Bases: AbstractBasis, MonomialBasis, BernsteinBasis, n_coeffs, evaluate
using ..Overlap: EulerianFrame, FrameBoundaries, cell_unit_box, cell_physical_box,
                  enumerate_leaves
using ..Threading: AbstractParallelBackend, Sequential, OhMyThreadsBackend,
                    default_backend, parallel_foreach

# ----------------------------------------------------------------------------
# Includes
# ----------------------------------------------------------------------------

include("Views.jl")
include("BlockView.jl")
include("Orchestrators.jl")

# ----------------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------------

export CellView, HaloView, BlockView, BlockHaloView
export ghost_depth, cell_view, halo_view_multi, block_view, block_halo_view
export for_each_cell!, for_each_face!, for_each_block!

end # module Solver
