#!/usr/bin/env julia
# Worked validation problem: cell-based scalar advection.
#
# Advect a Gaussian blob across a 2D periodic domain, with optional
# AMR refinement of the leading/trailing edges. Reports mass-
# conservation drift and L¹ error against the periodically shifted
# analytic IC.
#
# Run with:
#   julia --project=. examples/advection.jl

using CFDCellAdvection
using HierarchicalGrids

println("=" ^ 70)
println("Cell-based scalar advection — Phase 2 worked example (PR-14)")
println("=" ^ 70)

# Run a uniformly-refined baseline first (no AMR).
println("\n--- Run 1: Uniform 16x16, one full period, no AMR ---")
config1 = AdvectionConfig(
    n_initial_refines = 4,    # 16x16
    velocity = (0.5, 0.5),
    dt = 0.02,
    n_steps = 100,            # T = 2.0 = one full period along both axes
    amr_every = 0,
)

state1, info1 = run!(config1)
println("  initial mass:     ", info1.initial_mass)
println("  final mass:       ", info1.final_mass)
println("  mass drift:       ", info1.mass_drift)
println("  L¹ error:         ", info1.l1_error)
println("  n_leaves (final): ", info1.n_leaves_final)
println("  cell-mean range:  [",
        state1.diagnostics.liouville_min, ", ",
        state1.diagnostics.liouville_max, "]")

# Run a coarser baseline with AMR enabled.
println("\n--- Run 2: AMR — start at 8x8, refine on gradient ---")
config2 = AdvectionConfig(
    n_initial_refines = 3,    # 8x8 base
    velocity = (0.5, 0.3),
    dt = 0.005,
    n_steps = 60,
    amr_every = 5,
    refine_threshold = 0.05,
    coarsen_threshold = 0.005,
    max_level = 5,
)

state2, info2 = run!(config2)
println("  initial mass:     ", info2.initial_mass)
println("  final mass:       ", info2.final_mass)
println("  mass drift:       ", info2.mass_drift)
println("  n_leaves (final): ", info2.n_leaves_final,
        " (started at ", 8*8, ")")
println("  cell-mean range:  [",
        state2.diagnostics.liouville_min, ", ",
        state2.diagnostics.liouville_max, "]")

println("\nAll runs done.")
