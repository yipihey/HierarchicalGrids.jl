"""
    DSMCExample

A minimal Direct Simulation Monte Carlo solver demonstrating how to build
a real physics application on top of HierarchicalGrids.jl.

# Physics

Single-species hard-sphere gas in a periodic box. Standard time-splitting:
move (free-stream), sort (bin into cells), collide (NTC), sample (accumulate
per-cell macroscopics).

# Architecture

- `Simulation` is the top-level type; it owns the mesh, particles (as a
  layout-flexible FieldSet), per-cell sampling accumulators (also a FieldSet),
  and the simulation parameters.
- Particle layout (`SoA`/`AoS`/`Blocked`) is a constructor option. Same
  physics kernels work with any layout.
- Mesh is a `HierarchicalMesh{3}`. Currently uniform (no AMR) but the
  capability is there.
- Collision cells = mesh leaf cells. Cell index is the bin.

See `examples/relaxation.jl` for a worked validation problem.
"""
module DSMCExample

using HierarchicalGrids
using StaticArrays
using Random
using Statistics

# Submodules / functionality
include("particles.jl")
include("move.jl")
include("collisions.jl")
include("sampling.jl")
include("simulation.jl")

# Re-exports
export Simulation, SimParams
export step!, run_simulation!
export init_two_temperature_relaxation!
export total_kinetic_energy, total_momentum, total_particles
export temperature_per_axis, mean_velocity
export reset_sampling!, sample!, sampled_density, sampled_velocity, sampled_temperature

end # module
