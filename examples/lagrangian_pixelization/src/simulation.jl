# ============================================================================
# Top-level driver: tie Lagrangian + Eulerian + AMR + SVG output together
# ============================================================================
#
# A `Simulation` owns:
#  - the Lagrangian mesh (built once at construction; reset to reference
#    positions before each frame, then advanced via the flow map at time t)
#  - the Eulerian mesh (rebuilt fresh before each frame so the AMR is
#    driven from a known-uniform start; this is much simpler than tracking
#    a long-lived mesh and worrying about coarsening invariants)
#  - the EulerianFrame (cheap to construct; carries the physical box)
#  - the flow map and pixelization parameters
#
# The "fresh Eulerian mesh per frame" choice is deliberate. For an actual
# time-evolving solver you'd retain the mesh and use coarsening + refinement
# together; for a visualizer where each frame is independent it's cleaner.

"""
    Simulation(; n_per_axis=16, eul_initial_depth=4,
                  flow_map=SwirlMap(strength=2.0),
                  pixel_params=PixelizationParams(),
                  domain_lo=(0.0, 0.0), domain_hi=(1.0, 1.0),
                  svg_style=SvgStyle())

Top-level container for the mini-app. Defaults give a runnable demo.

- `n_per_axis` — Lagrangian vertex grid resolution. `16` → 16×16 vertices,
  450 triangles. The visual is good from `8` (chunky, fast) up to `64`
  (detailed, slower).
- `eul_initial_depth` — initial uniform Eulerian quadtree depth before AMR
  refinement kicks in. `4` gives a 16×16 starting grid. The AMR then
  refines selectively from there.
- `flow_map` — a `FlowMap` instance. Try `SwirlMap`, `RigidRotation`,
  `TaylorGreenPulse`, or wrap a custom function with `CustomMap`.
- `pixel_params` — `PixelizationParams` for the AMR loop.
- `domain_lo`, `domain_hi` — physical bounds. Default is the unit square;
  flow maps and Lagrangian mesh assume this for now.
- `svg_style` — visual style for SVG output.
"""
mutable struct Simulation{F <: FlowMap}
    n_per_axis::Int
    eul_initial_depth::Int
    flow_map::F
    pixel_params::PixelizationParams
    domain_lo::NTuple{2, Float64}
    domain_hi::NTuple{2, Float64}
    svg_style::SvgStyle

    # Working state
    lag::SimplicialMesh{2, Float64}
end

function Simulation(; n_per_axis::Integer = 16,
                      eul_initial_depth::Integer = 4,
                      flow_map::FlowMap = SwirlMap(strength=2.0),
                      pixel_params::PixelizationParams = PixelizationParams(),
                      domain_lo = (0.0, 0.0),
                      domain_hi = (1.0, 1.0),
                      svg_style::SvgStyle = SvgStyle())
    lag = build_unit_square_lag_mesh(n_per_axis)
    return Simulation(Int(n_per_axis), Int(eul_initial_depth),
                      flow_map, pixel_params,
                      (Float64(domain_lo[1]), Float64(domain_lo[2])),
                      (Float64(domain_hi[1]), Float64(domain_hi[2])),
                      svg_style, lag)
end

# ----------------------------------------------------------------------------
# Build a fresh uniformly-refined Eulerian quadtree
# ----------------------------------------------------------------------------

function _build_uniform_eul(depth::Int)
    eul = HierarchicalMesh{2}()
    for _ in 1:depth
        leaves = enumerate_leaves(eul)
        refine_cells!(eul, leaves)
    end
    return eul
end

# ----------------------------------------------------------------------------
# Single-frame step
# ----------------------------------------------------------------------------

"""
    step_to_time!(sim::Simulation, t::Real)
        -> (eul, frame, overlap, n_iters, refined_per_iter)

Advance the Lagrangian mesh to time `t`, build a fresh Eulerian quadtree,
run the AMR loop, and return the resulting Eulerian mesh, its frame, the
geometric overlap, and AMR diagnostics. Doesn't write any output — the
caller decides what to do with the returned state.
"""
function step_to_time!(sim::Simulation, t::Real)
    advance_lagrangian!(sim.lag, sim.flow_map, t)

    eul = _build_uniform_eul(sim.eul_initial_depth)
    frame = EulerianFrame(eul, sim.domain_lo, sim.domain_hi)

    ov, n_iters, refined_per_iter = pixelize!(eul, frame, sim.lag;
                                               params = sim.pixel_params)
    return (eul, frame, ov, n_iters, refined_per_iter)
end

# ----------------------------------------------------------------------------
# Multi-frame sequence
# ----------------------------------------------------------------------------

"""
    run_sequence!(sim::Simulation; n_frames=24, t_start=0.0, t_final=1.0,
                    output_dir="frames", verbose=true,
                    color_by=:density,
                    density_range=nothing)
        -> Vector{String}    # paths to written SVG files

Run a sequence of frames at uniformly-spaced times in `[t_start, t_final]`,
writing one SVG per frame to `output_dir/frame_NNNN.svg`. Returns the list
of written paths.

# Coloring

The Eulerian cells in each frame are colored according to `color_by`:

- `:density` (default) — physical mass density on each Eulerian cell,
  computed from the Lagrangian deformation under the assumption that
  each Lagrangian simplex carries a fixed mass equal to its reference
  area (uniform mass-per-reference-area). Compressed regions show up
  bright, stretched regions dim. See `eulerian_density` for details.
- `:overlap_count` — number of Lagrangian simplices overlapping each
  Eulerian leaf. Pure topological signal; doesn't reflect mass.
- `:none` — flat fill from `style.eul_fill`.

`density_range = (lo, hi)` fixes the colormap range to `[lo, hi]` so
the same density value gets the same color across all frames in the
sequence. If unset, a two-pass scheme is used: the first pass advances
the simulation through all frames to find the global density extrema
(over leaves with nonzero density), then the second pass writes the
frames with that range. The two-pass cost is roughly 2× the single-pass
runtime — set `density_range` explicitly to avoid it.

Frame `k ∈ 1:n_frames` corresponds to time `t = t_start + (k-1)/(n_frames-1)
* (t_final - t_start)` (for `n_frames > 1`); a single-frame sequence uses
`t_final`.
"""
function run_sequence!(sim::Simulation;
                        n_frames::Integer = 24,
                        t_start::Real = 0.0,
                        t_final::Real = 1.0,
                        output_dir::AbstractString = "frames",
                        verbose::Bool = true,
                        color_by::Symbol = :density,
                        density_range::Union{Nothing, Tuple{Real, Real}} = nothing)
    n_frames = Int(n_frames)
    n_frames >= 1 ||
        throw(ArgumentError("n_frames must be ≥ 1 (got $n_frames)"))
    color_by in (:density, :overlap_count, :none) ||
        throw(ArgumentError("color_by must be :density, :overlap_count, or :none (got $color_by)"))
    isdir(output_dir) || mkpath(output_dir)

    times = if n_frames == 1
        [Float64(t_final)]
    else
        t_span = Float64(t_final) - Float64(t_start)
        [Float64(t_start) + (k - 1) / (n_frames - 1) * t_span for k in 1:n_frames]
    end

    # Pass 1 (only if density coloring with auto-range): scan extrema.
    fixed_range = if color_by === :density && density_range === nothing
        verbose && println("scanning $(n_frames) frames to find density range...")
        lo_glob, hi_glob = Inf, -Inf
        for (k, t) in enumerate(times)
            eul, frame, ov, _, _ = step_to_time!(sim, t)
            ρ = eulerian_density(ov, sim.lag, frame)
            for ci in 1:n_cells(eul)
                is_leaf(eul.cells[ci]) || continue
                v = ρ[ci]
                v == 0 && continue
                v < lo_glob && (lo_glob = v)
                v > hi_glob && (hi_glob = v)
            end
            if verbose
                line = @sprintf("  scan %4d / %d   t=%.3f   density range so far: [%.3f, %.3f]",
                                 k, n_frames, t, lo_glob, hi_glob)
                println(line)
            end
        end
        if !isfinite(lo_glob) || !isfinite(hi_glob) || hi_glob <= lo_glob
            (0.0, 1.0)
        else
            (lo_glob, hi_glob)
        end
    elseif density_range !== nothing
        (Float64(density_range[1]), Float64(density_range[2]))
    else
        nothing
    end

    if verbose && fixed_range !== nothing && color_by === :density
        line = @sprintf("density colormap range: [%.3f, %.3f]", fixed_range[1], fixed_range[2])
        println(line)
    end

    # Pass 2 (or only pass): write frames.
    paths = String[]
    for (k, t) in enumerate(times)
        eul, frame, ov, n_iters, refined_per_iter = step_to_time!(sim, t)

        # Build cell_values per the chosen mode.
        cell_values = nothing
        if color_by === :density
            cell_values = eulerian_density(ov, sim.lag, frame)
        end
        # :overlap_count uses the legacy path (overlap → counts inside _write_eulerian)
        # :none falls through with no overlap → flat fill

        ov_arg = color_by === :overlap_count ? ov : nothing

        path = joinpath(output_dir, @sprintf("frame_%04d.svg", k))
        title = @sprintf("t = %.3f   (Eulerian leaves: %d, AMR iters: %d)",
                          t, count(c -> is_leaf(c), eul.cells), n_iters)
        write_svg(path, sim.lag, eul, frame;
                   overlap = ov_arg,
                   cell_values = cell_values,
                   cell_value_range = fixed_range,
                   title = title,
                   style = sim.svg_style)
        push!(paths, path)

        if verbose
            n_leaves = count(c -> is_leaf(c), eul.cells)
            tot_refined = sum(refined_per_iter; init=0)
            line = @sprintf("frame %4d / %d   t=%.3f   leaves=%d   AMR iters=%d   refined=%d",
                             k, n_frames, t, n_leaves, n_iters, tot_refined)
            println(line)
        end
    end

    return paths
end
