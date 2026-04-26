# ============================================================================
# SVG plotting: standalone vector output, no external dependencies
# ============================================================================
#
# The SVG layout is:
#
#   <svg viewBox="0 0 W H" ...>
#     <rect ... />                           # background
#     <g id="eulerian-cells">                # Eulerian leaves
#       <rect ... fill="…" />                # each leaf, color = overlap count
#       …
#     </g>
#     <g id="lagrangian-mesh">               # Lagrangian triangulation
#       <polygon points="..." />             # each triangle
#       …
#     </g>
#     <g id="annotations">                   # title, frame number, etc.
#       <text>…</text>
#     </g>
#   </svg>
#
# All coordinates are in physical (unit-square) space; the viewBox handles
# the world-to-pixel mapping. Higher overlap counts get warmer colors
# (grayscale by default; configurable via `SvgStyle`).

"""
    SvgStyle(; size_px=600, padding=20, lag_stroke="#3366aa",
             lag_stroke_width=0.6, eul_stroke="#999",
             eul_stroke_width=0.3, eul_fill="#f3f3f3", show_lag=true,
             show_eul=true, color_eul_by_count=true,
             count_color_min=0, count_color_max=10,
             count_color_palette=:grayscale_warm)

Visual style for SVG output.

- `size_px` — output image edge in pixels (square output: `size × size`).
- `padding` — pixel border around the unit square.
- `lag_stroke`, `lag_stroke_width` — Lagrangian triangle edge color and
  width (in unit-square coordinates; 0.6 of a pixel by default at 600px).
- `eul_stroke`, `eul_stroke_width`, `eul_fill` — Eulerian leaf rectangle
  styling. The fill is overridden per-cell when caller passes `cell_values`
  to `write_svg`, or when `color_eul_by_count = true` and an `overlap` is
  passed.
- `show_lag`, `show_eul` — toggle each layer.
- `color_eul_by_count` — when `cell_values` is not provided, fill each
  Eulerian leaf by its overlap count using `count_color_min`,
  `count_color_max`. Used by the simplest demos. Most callers pass
  `cell_values` explicitly (e.g. mass density) and ignore this setting.
- `count_color_palette` — `:grayscale_warm` (light → warm), `:viridis_lite`
  (a viridis-style colormap), or `:rd_pu` (ColorBrewer RdPu — light pink
  → magenta → deep purple). Used for both the count and `cell_values`
  paths.
"""
Base.@kwdef struct SvgStyle
    size_px::Int = 600
    padding::Int = 20
    lag_stroke::String = "#1f4f8c"
    lag_stroke_width::Float64 = 0.7
    eul_stroke::String = "#888"
    eul_stroke_width::Float64 = 0.25
    eul_fill::String = "#f3f3f3"
    show_lag::Bool = true
    show_eul::Bool = true
    color_eul_by_count::Bool = true
    count_color_min::Int = 0
    count_color_max::Int = 10
    count_color_palette::Symbol = :grayscale_warm
end

# ----------------------------------------------------------------------------
# Color palettes
# ----------------------------------------------------------------------------
#
# Both palettes return "#RRGGBB" hex strings. Input `t` is in [0, 1].

@inline function _palette_grayscale_warm(t::Real)
    # Light → warm: pale gray to red-orange. Looks good against the
    # Lagrangian's blue strokes.
    t = clamp(Float64(t), 0.0, 1.0)
    # 1.0 - t in white; t fades to a warm orange-red.
    r = round(Int, 255 * (1.0 - 0.10 * t))
    g = round(Int, 255 * (1.0 - 0.55 * t))
    b = round(Int, 255 * (1.0 - 0.85 * t))
    return @sprintf("#%02x%02x%02x", r, g, b)
end

@inline function _palette_viridis_lite(t::Real)
    # 4-stop piecewise viridis approximation. Purple → blue → green → yellow.
    t = clamp(Float64(t), 0.0, 1.0)
    if t < 1/3
        u = t * 3.0
        r = round(Int, 68  + (33  - 68 ) * u)
        g = round(Int, 1   + (145 - 1  ) * u)
        b = round(Int, 84  + (140 - 84 ) * u)
    elseif t < 2/3
        u = (t - 1/3) * 3.0
        r = round(Int, 33  + (94  - 33 ) * u)
        g = round(Int, 145 + (201 - 145) * u)
        b = round(Int, 140 + (98  - 140) * u)
    else
        u = (t - 2/3) * 3.0
        r = round(Int, 94  + (253 - 94 ) * u)
        g = round(Int, 201 + (231 - 201) * u)
        b = round(Int, 98  + (37  - 98 ) * u)
    end
    return @sprintf("#%02x%02x%02x", r, g, b)
end

# 4-stop piecewise approximation to ColorBrewer's RdPu sequential map.
# Anchors:
#   t=0.00  #fff7f3   very light pink (almost white)
#   t=0.33  #fa9fb5   pink
#   t=0.66  #dd3497   magenta
#   t=1.00  #49006a   deep purple
@inline function _palette_rd_pu(t::Real)
    t = clamp(Float64(t), 0.0, 1.0)
    if t < 1/3
        u = t * 3.0
        r = round(Int, 255 + (250 - 255) * u)
        g = round(Int, 247 + (159 - 247) * u)
        b = round(Int, 243 + (181 - 243) * u)
    elseif t < 2/3
        u = (t - 1/3) * 3.0
        r = round(Int, 250 + (221 - 250) * u)
        g = round(Int, 159 + (52  - 159) * u)
        b = round(Int, 181 + (151 - 181) * u)
    else
        u = (t - 2/3) * 3.0
        r = round(Int, 221 + (73  - 221) * u)
        g = round(Int, 52  + (0   - 52 ) * u)
        b = round(Int, 151 + (106 - 151) * u)
    end
    return @sprintf("#%02x%02x%02x", r, g, b)
end

@inline function _palette_color(palette::Symbol, t::Real)
    if palette === :grayscale_warm
        return _palette_grayscale_warm(t)
    elseif palette === :viridis_lite
        return _palette_viridis_lite(t)
    elseif palette === :rd_pu
        return _palette_rd_pu(t)
    else
        return _palette_grayscale_warm(t)
    end
end

# ----------------------------------------------------------------------------
# Geometry: project unit-square (x, y) to SVG pixel coordinates
# ----------------------------------------------------------------------------

@inline function _to_svg_xy(x::Real, y::Real, style::SvgStyle)
    inner = style.size_px - 2 * style.padding
    sx = style.padding + Float64(x) * inner
    # SVG y-axis points down; flip so y=0 is at the bottom of the picture.
    sy = style.padding + (1.0 - Float64(y)) * inner
    return (sx, sy)
end

# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

"""
    write_svg(path::AbstractString,
              lag::SimplicialMesh{2, Float64},
              eul::HierarchicalMesh{2},
              frame::EulerianFrame{2, Float64};
              overlap::Union{Nothing, GeometricOverlap} = nothing,
              title::Union{Nothing, AbstractString} = nothing,
              style::SvgStyle = SvgStyle())

Write an SVG visualization to `path`.

# Coloring the Eulerian cells

Three coloring modes, in priority order:

1. If `cell_values::Vector{Float64}` is provided, each Eulerian leaf is
   filled by mapping its value through `style.count_color_palette`,
   normalized into the range `cell_value_range` (defaults to the
   `[min, max]` of `cell_values` over leaves with nonzero values).
   Use this for plotting density, sampled fields, or any per-cell scalar.
2. Otherwise, if `overlap !== nothing` and `style.color_eul_by_count`,
   each leaf is colored by `length(overlap.eul_to_entries[i])` (the
   number of Lagrangian simplices overlapping it).
3. Otherwise, all leaves get the flat `style.eul_fill`.
"""
function write_svg(path::AbstractString,
                    lag::SimplicialMesh{2, Float64},
                    eul::HierarchicalMesh{2},
                    frame::EulerianFrame{2, Float64};
                    overlap::Union{Nothing, GeometricOverlap} = nothing,
                    cell_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
                    cell_value_range::Union{Nothing, Tuple{Real, Real}} = nothing,
                    title::Union{Nothing, AbstractString} = nothing,
                    style::SvgStyle = SvgStyle())
    open(path, "w") do io
        write_svg(io, lag, eul, frame;
                   overlap = overlap,
                   cell_values = cell_values,
                   cell_value_range = cell_value_range,
                   title = title,
                   style = style)
    end
    return path
end

function write_svg(io::IO,
                    lag::SimplicialMesh{2, Float64},
                    eul::HierarchicalMesh{2},
                    frame::EulerianFrame{2, Float64};
                    overlap::Union{Nothing, GeometricOverlap} = nothing,
                    cell_values::Union{Nothing, AbstractVector{<:Real}} = nothing,
                    cell_value_range::Union{Nothing, Tuple{Real, Real}} = nothing,
                    title::Union{Nothing, AbstractString} = nothing,
                    style::SvgStyle = SvgStyle())
    W = H = style.size_px
    println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" ",
                "viewBox=\"0 0 $W $H\" width=\"$W\" height=\"$H\">")
    # Background (white)
    println(io, "  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>")

    # Resolve the cell-value range if needed.
    cv_lo, cv_hi = _resolve_cell_value_range(cell_values, cell_value_range, eul)

    # --- Eulerian layer ---
    if style.show_eul
        _write_eulerian(io, eul, frame, overlap, cell_values, cv_lo, cv_hi, style)
    end

    # --- Lagrangian layer ---
    if style.show_lag
        _write_lagrangian(io, lag, style)
    end

    # --- Annotations ---
    if title !== nothing
        # Top-left corner, in axes-units padding
        x = style.padding
        y = style.padding * 0.7
        font_size = max(11, style.padding ÷ 2)
        # Use a slightly grayed-out text so it doesn't fight the figure
        line = @sprintf("  <text x=\"%d\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"%d\" fill=\"#444\">%s</text>",
                        x, y, font_size, _xml_escape(title))
        println(io, line)
    end

    println(io, "</svg>")
end

# ----------------------------------------------------------------------------
# Eulerian leaves: rectangles, optionally colored by per-cell scalar
# ----------------------------------------------------------------------------

# Auto-range from cell_values over leaves only. Skips zeros, since
# zero typically means "no overlap" / "no data" rather than the lower
# end of the data range.
function _resolve_cell_value_range(cell_values, cell_value_range, eul::HierarchicalMesh{2})
    cell_values === nothing && return (0.0, 1.0)   # unused, return placeholder
    cell_value_range !== nothing &&
        return (Float64(cell_value_range[1]), Float64(cell_value_range[2]))
    # Auto: compute over leaves with nonzero values.
    lo = Inf
    hi = -Inf
    @inbounds for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        ci > length(cell_values) && continue
        v = Float64(cell_values[ci])
        v == 0 && continue
        v < lo && (lo = v)
        v > hi && (hi = v)
    end
    if !isfinite(lo) || !isfinite(hi) || hi <= lo
        return (0.0, 1.0)
    end
    return (lo, hi)
end

function _write_eulerian(io::IO,
                          eul::HierarchicalMesh{2},
                          frame::EulerianFrame{2, Float64},
                          overlap::Union{Nothing, GeometricOverlap},
                          cell_values::Union{Nothing, AbstractVector{<:Real}},
                          cv_lo::Float64, cv_hi::Float64,
                          style::SvgStyle)
    println(io, "  <g id=\"eulerian-cells\" ",
                "stroke=\"$(style.eul_stroke)\" ",
                "stroke-width=\"$(style.eul_stroke_width)\" ",
                "stroke-linejoin=\"miter\">")
    cnt_min = style.count_color_min
    cnt_max = max(style.count_color_max, cnt_min + 1)
    cv_span = cv_hi - cv_lo

    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        sx_lo, sy_lo = _to_svg_xy(lo[1], hi[2], style)   # SVG: top-left
        sx_hi, sy_hi = _to_svg_xy(hi[1], lo[2], style)   # SVG: bottom-right
        w_px = sx_hi - sx_lo
        h_px = sy_hi - sy_lo

        # Choose fill: cell_values takes priority over count, both over flat fill.
        fill = style.eul_fill
        if cell_values !== nothing && ci <= length(cell_values)
            v = Float64(cell_values[ci])
            t = (v - cv_lo) / cv_span
            fill = _palette_color(style.count_color_palette, t)
        elseif style.color_eul_by_count && overlap !== nothing
            count = length(overlap.eul_to_entries[ci])
            t = (count - cnt_min) / (cnt_max - cnt_min)
            fill = _palette_color(style.count_color_palette, t)
        end

        line = @sprintf("    <rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"%s\"/>",
                        sx_lo, sy_lo, w_px, h_px, fill)
        println(io, line)
    end
    println(io, "  </g>")
end

# ----------------------------------------------------------------------------
# Lagrangian triangles: thin polygons
# ----------------------------------------------------------------------------

function _write_lagrangian(io::IO,
                            lag::SimplicialMesh{2, Float64},
                            style::SvgStyle)
    println(io, "  <g id=\"lagrangian-mesh\" fill=\"none\" ",
                "stroke=\"$(style.lag_stroke)\" ",
                "stroke-width=\"$(style.lag_stroke_width)\" ",
                "stroke-linejoin=\"round\">")
    for s in 1:n_simplices(lag)
        verts = simplex_vertex_positions(lag, s)
        p1 = _to_svg_xy(verts[1][1], verts[1][2], style)
        p2 = _to_svg_xy(verts[2][1], verts[2][2], style)
        p3 = _to_svg_xy(verts[3][1], verts[3][2], style)
        line = @sprintf("    <polygon points=\"%.2f,%.2f %.2f,%.2f %.2f,%.2f\"/>",
                        p1[1], p1[2], p2[1], p2[2], p3[1], p3[2])
        println(io, line)
    end
    println(io, "  </g>")
end

# ----------------------------------------------------------------------------
# XML-safe escapes
# ----------------------------------------------------------------------------

function _xml_escape(s::AbstractString)
    out = IOBuffer()
    for c in s
        if c == '&';      print(out, "&amp;")
        elseif c == '<';  print(out, "&lt;")
        elseif c == '>';  print(out, "&gt;")
        elseif c == '"';  print(out, "&quot;")
        elseif c == '\''; print(out, "&apos;")
        else              print(out, c)
        end
    end
    return String(take!(out))
end
