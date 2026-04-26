"""
    LagrangianPixelization

A mini-app demonstrating HierarchicalGrids.jl on a visually striking
problem: a 2D Lagrangian triangle mesh deforms under a time-dependent
flow map, and an adaptive Eulerian quadtree pixelizes it so that each
deformed Lagrangian triangle is overlapped by approximately the same
number of Eulerian leaves (an isotropy-preserving AMR criterion).

# What it exercises

- `SimplicialMesh{2, Float64}` — Lagrangian triangulation, vertices
  driven by a flow map.
- `HierarchicalMesh{2}` + `EulerianFrame{2, Float64}` — Eulerian
  quadtree pixelization target.
- `compute_overlap` — exact 2D triangle ∩ box clipping with moments.
- `refine_by_indicator!` — AMR loop driven by a per-Eulerian-cell
  indicator derived from the overlap counts.
- Standalone SVG output — no plotting dependency, vector quality,
  multi-frame sequence renders directly in the visualizer.

# Architecture

- `flow_maps.jl` — pure-function flow maps `(x, y, t) -> (x', y')`.
  A small library: rigid rotation, swirl, Taylor–Green pulse, custom.
- `lag_mesh.jl` — build an initial uniform Lagrangian triangulation of
  the unit square; advance vertex positions via a flow map.
- `pixelize.jl` — single-step AMR loop: compute overlap, derive the
  per-Eulerian-cell indicator from per-Lagrangian-simplex overlap
  counts, call `refine_by_indicator!`, iterate to a stable mesh.
- `plotting.jl` — SVG writer: deformed Lagrangian triangulation +
  Eulerian quadtree leaves, colored by overlap count or by sampled
  density. Multi-frame sequences just call the single-frame writer
  in a loop with different filenames.
- `simulation.jl` — top-level driver: build initial state, advance
  Lagrangian under a flow map, re-pixelize, write a frame, repeat.

# Quick start

```julia
using LagrangianPixelization

# Build the simulation
sim = Simulation(;
    n_per_axis = 16,                    # Lagrangian: 16×16 vertices → 512 triangles
    eul_initial_depth = 4,              # Eulerian: 16×16 starting grid
    target_eul_per_lag = 4,             # AMR target
    flow_map = swirl_map(strength=2.0), # one of the prebuilt maps
)

# Run a 32-frame sequence
run_sequence!(sim;
    n_frames = 32,
    t_final = 1.0,
    output_dir = "frames",
)

# Check the output:
#   frames/frame_0001.svg ... frames/frame_0032.svg
```

See `examples/swirl.jl` for a worked headline demo.
"""
module LagrangianPixelization

using HierarchicalGrids
using Printf: @sprintf

include("flow_maps.jl")
include("lag_mesh.jl")
include("pixelize.jl")
include("plotting.jl")
include("simulation.jl")

# --- Re-exports ---

# Selected HierarchicalGrids names that example scripts will commonly want.
# Keeps the demo scripts free of `using HierarchicalGrids` boilerplate.
export n_simplices, n_vertices, n_cells, is_leaf, n_entries
export total_overlap_volume, vertex_position, reference_position
export simplex_vertex_positions, simplex_volume, simplex_reference_volume
export HierarchicalMesh, SimplicialMesh, EulerianFrame
export refine_cells!, enumerate_leaves, level_of
export compute_overlap, refine_by_indicator!
export cell_physical_box, cell_unit_box, root_box

# Flow maps
export FlowMap, IdentityMap, RigidRotation, SwirlMap, TaylorGreenPulse
export GaussianCompression, ComposedMap, CustomMap
export apply_map, swirl_map, rigid_rotation, taylor_green_pulse

# Lagrangian mesh
export build_unit_square_lag_mesh, advance_lagrangian!

# Pixelization (AMR) and density
export PixelizationParams, pixelize!
export n_eulerian_per_lagrangian, n_lagrangian_per_eulerian
export lagrangian_density_factors, eulerian_density

# SVG output
export write_svg, SvgStyle

# Top-level driver
export Simulation, run_sequence!, step_to_time!

end # module
