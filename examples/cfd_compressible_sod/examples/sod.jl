#!/usr/bin/env julia
# Worked validation problem: 2D Sod shock-tube via HLL + AMR.
#
# Standard Sod IC, OUTFLOW in x, PERIODIC in y. Reports mass / energy
# conservation drift, shock position, and L¹ error against the analytic
# Riemann solution at t = 0.2.
#
# Run with:
#   julia --project=. examples/sod.jl

using CFDCompressibleSod
using HierarchicalGrids

println("=" ^ 70)
println("Sod shock-tube — Phase 2 worked example (PR-3)")
println("=" ^ 70)

# Run 1: uniform 32x32 base, no AMR.
println("\n--- Run 1: Uniform base mesh, no AMR ---")
config1 = SodConfig(n_initial_refines = 5, t_final = 0.2, amr_every = 0)
state1, info1 = CFDCompressibleSod.run!(config1)
println("  n_leaves (final): ", info1.n_leaves_final)
println("  n_steps:          ", info1.n_steps)
println("  t_final:          ", info1.t_final)
println("  mass drift:       ", info1.mass_drift)
println("  energy drift:     ", info1.energy_drift)
println("  momentum drift:   ", info1.momentum_drift)
println("  shock position:   ", info1.shock_position, " (analytic ~0.85)")
println("  L¹ error vs exact:", info1.l1_error)

# Run 2: AMR on, refining around the shock.
println("\n--- Run 2: AMR — coarse base, refines around shock ---")
config2 = SodConfig(n_initial_refines = 4, t_final = 0.2,
                     amr_every = 5, max_level = 3,
                     refine_threshold = 0.1, coarsen_threshold = 0.01)
state2, info2 = CFDCompressibleSod.run!(config2)
println("  n_leaves (initial): ", info2.n_leaves_initial)
println("  n_leaves (final):   ", info2.n_leaves_final)
println("  n_steps:            ", info2.n_steps)
println("  mass drift:         ", info2.mass_drift)
println("  energy drift:       ", info2.energy_drift)
println("  shock position:     ", info2.shock_position)
println("  L¹ error vs exact:  ", info2.l1_error)

println("\nAll runs done.")
