# ============================================================================
# swirl.jl — headline demo: Lagrangian mesh under swirl + Gaussian
# compression, with the Eulerian quadtree adapting to keep equal cells
# per Lagrangian patch, colored by mass density
# ============================================================================
#
# Run from the example's directory:
#
#   $ cd examples/lagrangian_pixelization
#   $ julia --project=. examples/swirl.jl
#
# This demo composes two effects:
#
#   1. SwirlMap — a radial twist that's area-preserving (or near-so);
#      generates strong shear but doesn't change triangle areas much.
#   2. GaussianCompression — a non-area-preserving compression toward
#      an off-center point. This is what *actually* changes Lagrangian
#      triangle sizes and creates the locally-varying density that
#      drives interesting AMR.
#
# Each Lagrangian simplex carries a fixed mass (proportional to its
# reference area). When the flow compresses a simplex, density goes up;
# when it stretches, density goes down. We deposit this mass onto the
# Eulerian leaves through the geometric overlap to get the physical
# density at the Eulerian cell scale, and color the cells by that
# density. The AMR criterion is independent: refine where Lagrangian
# simplices have too few overlapping Eulerian leaves. Together, they
# show that compressed regions both render bright AND get finely
# resolved by the AMR — which is exactly the desired behavior.

using LagrangianPixelization

# -----------------------------------------------------------------------------
# Build the simulation
# -----------------------------------------------------------------------------

flow = ComposedMap(
    SwirlMap(strength=2.0),                                # area-preserving twist
    GaussianCompression(amplitude=0.6, cx=0.35, cy=0.55,   # off-center compression
                         sigma=0.18),
)

sim = Simulation(;
    n_per_axis        = 32,
    eul_initial_depth = 5,
    flow_map          = flow,
    pixel_params      = PixelizationParams(
                          target_eul_per_lag = 4,
                          max_depth          = 8,
                          max_iters          = 12,
                        ),
    svg_style         = SvgStyle(
                          size_px               = 720,
                          padding               = 24,
                          color_eul_by_count    = false,        # using cell_values
                          count_color_palette   = :rd_pu,
                          lag_stroke            = "#3a3a3a",    # dark grey
                          lag_stroke_width      = 0.55,
                          eul_stroke            = "#c8a878",    # light brown
                          eul_stroke_width      = 0.20,
                        ),
)

# -----------------------------------------------------------------------------
# Run the sequence with density coloring
# -----------------------------------------------------------------------------
#
# `color_by = :density` (the default) computes per-Eulerian-cell mass
# density assuming uniform mass-per-reference-area on the Lagrangian.
# `density_range` fixes the colormap so the same density value gets the
# same color across all frames — much nicer for animation than the
# auto-per-frame range that would otherwise wobble. Empirically, a range
# of `(0.5, 6.0)` covers most of the dynamic range for this swirl +
# compression combo: rest density is 1.0; peak compressed density at
# t=0.5 is about 5.8; stretched regions can go down to ~0.5.

paths = run_sequence!(sim;
                       n_frames      = 24,
                       t_start       = 0.0,
                       t_final       = 1.0,
                       output_dir    = joinpath(@__DIR__, "..", "frames", "swirl"),
                       verbose       = true,
                       color_by      = :density,
                       density_range = (0.5, 6.0),
                     )

println()
println("Wrote ", length(paths), " frames.")
println("First: ", paths[1])
println("Last:  ", paths[end])
println()
println("Open one in a browser, or assemble into a GIF:")
println("  magick -delay 8 frames/swirl/frame_*.svg out.gif")
println("Or with ffmpeg (requires SVG → PNG conversion first):")
println("  for f in frames/swirl/frame_*.svg; do rsvg-convert \"\$f\" > \"\${f%.svg}.png\"; done")
println("  ffmpeg -framerate 12 -pattern_type glob -i 'frames/swirl/frame_*.png' out.mp4")
