# ============================================================================
# rotation.jl — control case: rigid rotation produces no AMR refinement
# ============================================================================
#
# A rigid rotation deforms every Lagrangian triangle by the same orthogonal
# transformation. Triangle areas don't change, no triangle stretches more
# than another, so the count-based AMR criterion sees a uniform signal —
# either every cell needs more (under-resolved at the start) or none do
# (already fine enough). Compare with `swirl.jl`, where local stretching
# drives spatially-varying AMR.
#
# This is the sanity-check companion to the headline swirl demo: if the
# AMR ever produces non-uniform refinement under a rigid rotation, that's
# a bug.

using LagrangianPixelization

sim = Simulation(;
    n_per_axis        = 24,
    eul_initial_depth = 5,
    flow_map          = RigidRotation(2π; cx=0.5, cy=0.5),  # one full turn over [0,1]
    pixel_params      = PixelizationParams(target_eul_per_lag=4, max_depth=7),
    svg_style         = SvgStyle(
                          size_px            = 600,
                          color_eul_by_count = true,
                          count_color_palette = :grayscale_warm,
                        ),
)

# A rigid rotation pulls every vertex slightly outside [0,1]² near the
# corners. The driver clamps; some boundary triangles will distort
# trivially. The interior AMR pattern should still be roughly isotropic.

paths = run_sequence!(sim;
                       n_frames   = 12,
                       t_start    = 0.0,
                       t_final    = 0.25,    # quarter turn — corners move into the box
                       output_dir = joinpath(@__DIR__, "..", "frames", "rotation"),
                       verbose    = true)

println()
println("Wrote ", length(paths), " frames to frames/rotation/")
