"""
Cubic-edge dimension lifting (placeholder).

The dfmm design (§6.2) handles cubic-edge curved triangles in the
Bayesian remap by lifting each curved-edge constraint to a coordinate
hyperplane in extended D' = D + 3 space (one extra coordinate per curved
edge of a 2D triangle). In the lifted space the overlap region is a
straight-faced polytope, so r3djl's standard `clip!` and `moments!`
machinery applies; the moments are then projected back to D-dimensional
moments via a small linear map.

This module will provide:

    lift_cubic_triangle(vertices, edge_curvatures) -> LiftedPolytope
    project_lifted_moments(moments_5d, projection_map) -> moments_2d

# r3djl status (verified 2026-04-26 against r3djl HEAD f704cff)

Phase 3 of r3djl has *almost* fully landed:

- ✅ D ≥ 4 `init_box!`, `init_simplex!`, `clip!` (single + sequential),
      `moments!` order 0, `voxelize_fold!` order 0, facet tracking.
- ✅ **D ≥ 4 sequential `clip!` on simplex polytopes is correct.**
      Fixed in r3djl `c4496ab` (2026-04-26) via an ε-nudge on the
      cut-position formula when a kept vertex lies exactly on the cut
      plane. Four-quadrant decomposition of a unit D = 4 simplex now
      preserves total volume and respects coordinate symmetry to
      floating-point round-off.
- ✅ **D ≥ 4 `voxelize_fold!` hot loop is heap-free.** Three closure-
      boxing / escape-analysis pitfalls fixed in r3djl `f704cff`
      (2026-04-26): closures capturing mutable locals in the bisection
      loop, MVector escape into the `_reduce_helper_nd` recursion, and
      MMatrix LTD scratch. Steady-state `@allocated` is now 0.
- ❌ **Higher-order moments (P ≥ 1) at D ≥ 4 are still pending.**
      Lasserre's recursive formula on top of the new facet-tracking
      infrastructure is the next planned r3djl session.

Cubic-edge dimension lifting still needs the higher moments before it
can be wired up: the polynomial remap math we use elsewhere requires
moments to order P_src + P_dst ≥ 6. The volume-only path (e.g.
voxelize-fold a lifted simplex against a uniform Eulerian grid for
geometric overlap measurement) is now unblocked.

Until then, calling `compute_overlap` with `edge_kind = :cubic` throws a
clear error pointing here, and these stubs are deliberately not exported
at the framework's top level — they're discoverable as
`HierarchicalGrids.Overlap.lift_cubic_triangle` for inspection.

# References

- dfmm design document, §6.2 "Geometric overlap via r3d"
- Powell & Abel (2015), J. Comput. Phys. 297: 340–356
"""

"""
    lift_cubic_triangle(vertices, edge_curvatures) -> NamedTuple

NOT YET IMPLEMENTED. Lifts a cubic-edge triangle in 2D to a flat-faced
polytope in 5D.

# Arguments

- `vertices` — three triangle vertices in 2D physical space.
- `edge_curvatures` — per-edge cubic Bezier control coefficients
  parameterizing the curve away from the straight chord; shape `(2, 3)`.

# Returns

A NamedTuple `(vertices_5d, planes_5d, projection_2d)` describing the
lifted polytope and the linear projection of 5D moments back to 2D.
"""
function lift_cubic_triangle(vertices, edge_curvatures)
    error("lift_cubic_triangle: NOT YET IMPLEMENTED. " *
          "The volume-only path is unblocked (r3djl simplex sequential-clip " *
          "fixed in r3djl f704cff), but full polynomial remap still needs " *
          "r3djl higher-order moments (P ≥ 1) at D ≥ 4. " *
          "See src/Overlap/lifting.jl for the planned interface and status.")
end

"""
    project_lifted_moments(moments_5d::AbstractVector, projection_map) -> moments_2d

NOT YET IMPLEMENTED. Linear projection of 5D polynomial moments back to
2D moments via the precomputed projection map from `lift_cubic_triangle`.
"""
function project_lifted_moments(moments_5d, projection_map)
    error("project_lifted_moments: NOT YET IMPLEMENTED. " *
          "See src/Overlap/lifting.jl for the planned interface and status.")
end
