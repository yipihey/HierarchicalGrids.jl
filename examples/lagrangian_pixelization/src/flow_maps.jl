# ============================================================================
# Flow maps: pure functions (x, y, t) → (x', y')
# ============================================================================
#
# These are smooth, area-preserving (or near-so) deformations of the unit
# square that produce visually interesting Lagrangian distortions. All maps
# are evaluated as `apply_map(map, x, y, t)`; the underlying type carries
# whatever parameters the map needs.

"""
    FlowMap

Abstract supertype for time-dependent maps `(x, y, t) -> (x', y')`. A
concrete subtype implements `apply_map(map, x, y, t) :: NTuple{2, Float64}`.

The map should map the unit square `[0, 1]²` into itself for `t ∈ [0, 1]`,
or at least have its support stay inside `[0, 1]²` over the time range used
by the driver. The driver clamps vertices that escape, but the visual is
better if the map is well-behaved.
"""
abstract type FlowMap end

"""
    IdentityMap()

The trivial map that doesn't deform. Useful for sanity checks.
"""
struct IdentityMap <: FlowMap end

@inline apply_map(::IdentityMap, x::Real, y::Real, t::Real) =
    (Float64(x), Float64(y))

"""
    RigidRotation(ω::Float64; cx=0.5, cy=0.5)

Rigid rotation by angle `ω·t` about `(cx, cy)`. Linear in the spatial
coordinates, so Lagrangian triangles deform only by rotation (no shear,
no compression) — useful as a control case where the AMR shouldn't have
much to do.

`ω` is total turn over `t ∈ [0, 1]` in radians; `ω = 2π` gives one full turn.
"""
struct RigidRotation <: FlowMap
    ω::Float64
    cx::Float64
    cy::Float64
end

RigidRotation(ω::Real; cx::Real=0.5, cy::Real=0.5) =
    RigidRotation(Float64(ω), Float64(cx), Float64(cy))

rigid_rotation(ω::Real; kwargs...) = RigidRotation(ω; kwargs...)

@inline function apply_map(m::RigidRotation, x::Real, y::Real, t::Real)
    θ = m.ω * Float64(t)
    s, c = sincos(θ)
    dx = Float64(x) - m.cx
    dy = Float64(y) - m.cy
    return (m.cx + c*dx - s*dy, m.cy + s*dx + c*dy)
end

"""
    SwirlMap(; strength=1.5, cx=0.5, cy=0.5, radius=0.45, period=1.0)

A radial swirl: the angular velocity peaks at the center and falls off
smoothly to zero outside `radius`. Each Lagrangian point at distance `r`
from `(cx, cy)` rotates by an angle that depends on `r`, producing a
spiral deformation. Strongly distorts triangles near the center; the
boundary stays fixed (good for keeping vertices inside `[0, 1]²`).

`strength`: peak angular displacement (radians) at `t = 0.25·period`
and `t = 0.75·period` (the maxima of the sinusoidal time modulation).
The swirl reverses direction in the second half of the period to bring
vertices back to their starting positions at `t = period`.

# Inversion

The Lagrangian triangulation inverts (some triangles develop negative
signed area) when `strength` exceeds roughly `2.0` for typical mesh
densities — the differential rotation between adjacent radii becomes
large enough to fold the mesh. The default `1.5` keeps the deformation
visually striking while remaining inversion-free at `n_per_axis=24`.

`period`: the full forward-and-back cycle. Default is one period over
`t ∈ [0, 1]`.
"""
struct SwirlMap <: FlowMap
    strength::Float64
    cx::Float64
    cy::Float64
    radius::Float64
    period::Float64
end

function SwirlMap(; strength::Real=1.5, cx::Real=0.5, cy::Real=0.5,
                    radius::Real=0.45, period::Real=1.0)
    return SwirlMap(Float64(strength), Float64(cx), Float64(cy),
                    Float64(radius), Float64(period))
end

swirl_map(; kwargs...) = SwirlMap(; kwargs...)

@inline function apply_map(m::SwirlMap, x::Real, y::Real, t::Real)
    dx = Float64(x) - m.cx
    dy = Float64(y) - m.cy
    r2 = dx*dx + dy*dy
    R2 = m.radius * m.radius
    # Smoothly tapers to zero at r = radius. Use cos² of (πr/2R) on r ≤ R.
    if r2 >= R2
        return (Float64(x), Float64(y))
    end
    falloff = (1.0 - r2 / R2)^2     # smooth at r = radius
    # Time-modulation: forward-and-back with period
    phase = sin(2π * Float64(t) / m.period)
    θ = m.strength * falloff * phase
    s, c = sincos(θ)
    return (m.cx + c*dx - s*dy, m.cy + s*dx + c*dy)
end

"""
    TaylorGreenPulse(; amplitude=0.15, k=2.0)

A Taylor–Green-style pulsating shear (one full pulse over `t ∈ [0, 1]`).
The map is

    x' = x + a · sin(πt) · sin(kπx) · cos(kπy)
    y' = y - a · sin(πt) · cos(kπx) · sin(kπy)

which is incompressible to first order in `a`. For small `a` (≲ 0.2) the
map respects the unit square's boundary tightly; larger `a` may push some
vertices slightly out and the driver will clamp them.

`amplitude` is the displacement scale; `k` controls how many cells the
pulse decomposes into. The default `k = 2` gives the classic 4-cell
Taylor–Green pattern.
"""
struct TaylorGreenPulse <: FlowMap
    amplitude::Float64
    k::Float64
end

TaylorGreenPulse(; amplitude::Real=0.15, k::Real=2.0) =
    TaylorGreenPulse(Float64(amplitude), Float64(k))

taylor_green_pulse(; kwargs...) = TaylorGreenPulse(; kwargs...)

@inline function apply_map(m::TaylorGreenPulse, x::Real, y::Real, t::Real)
    xf, yf, tf = Float64(x), Float64(y), Float64(t)
    pulse = sin(π * tf)
    kπx = m.k * π * xf
    kπy = m.k * π * yf
    return (xf + m.amplitude * pulse * sin(kπx) * cos(kπy),
            yf - m.amplitude * pulse * cos(kπx) * sin(kπy))
end

"""
    GaussianCompression(; amplitude=0.6, cx=0.35, cy=0.55, sigma=0.18,
                          period=1.0)

A radial compression toward a Gaussian-weighted center. Each point is
pulled toward `(cx, cy)` with a strength that's a Gaussian in distance
from the center, modulated sinusoidally in time so that the map returns
to identity at `t = 0` and `t = period`:

    r_new = r · (1 - amplitude · exp(-r²/(2σ²)) · sin(πt/period))

with `r = distance from (cx, cy)`. Unlike `SwirlMap` and `RigidRotation`,
this map is **not area-preserving** — it genuinely compresses Lagrangian
triangles near the center and stretches them slightly outside. That's
exactly what creates strong AMR signal: the compressed simplices want
many Eulerian leaves to track them, while the surrounding region is
content with the base resolution.

# Parameters

- `amplitude` ∈ (0, 1): peak fractional compression at center. `0.5`
  means the center pulls in by half its distance at peak; `0.8` is
  much more aggressive. Above `~0.9` the compression can produce
  near-singular triangles.
- `cx`, `cy`: the compression center (the off-center default `(0.35,
  0.55)` keeps the boundary symmetric around it).
- `sigma`: width of the Gaussian. Default `0.18` makes the compression
  largely confined to ~2σ ≈ 0.36 around the center.
- `period`: full forward-and-back cycle.

# Composition

Compose with another flow map (e.g. `SwirlMap`) via `ComposedMap` to
get richer dynamics — swirl + compression gives both shear and density
variation, exercising the AMR more thoroughly than either alone.
"""
struct GaussianCompression <: FlowMap
    amplitude::Float64
    cx::Float64
    cy::Float64
    sigma::Float64
    period::Float64
end

function GaussianCompression(; amplitude::Real=0.6, cx::Real=0.35, cy::Real=0.55,
                               sigma::Real=0.18, period::Real=1.0)
    0.0 < amplitude < 1.0 ||
        throw(ArgumentError("amplitude must be in (0, 1) (got $amplitude)"))
    sigma > 0 || throw(ArgumentError("sigma must be > 0 (got $sigma)"))
    return GaussianCompression(Float64(amplitude), Float64(cx), Float64(cy),
                                Float64(sigma), Float64(period))
end

@inline function apply_map(m::GaussianCompression, x::Real, y::Real, t::Real)
    dx = Float64(x) - m.cx
    dy = Float64(y) - m.cy
    r2 = dx*dx + dy*dy
    pulse = sin(π * Float64(t) / m.period)
    # Compression factor in [1 - amplitude, 1]: at center pulls in,
    # at infinity tends to 1 (no displacement).
    compress = 1.0 - m.amplitude * exp(-r2 / (2 * m.sigma * m.sigma)) * pulse
    return (m.cx + compress * dx, m.cy + compress * dy)
end

"""
    ComposedMap(first::FlowMap, second::FlowMap)

Compose two flow maps: `apply_map(ComposedMap(f, g), x, y, t)` evaluates
`f` at `(x, y, t)` first, then `g` at the result with the same `t`.

Used to chain effects: e.g. swirl + Gaussian compression =

```julia
ComposedMap(SwirlMap(strength=2.0),
            GaussianCompression(amplitude=0.6))
```

The order matters: `ComposedMap(swirl, compress)` first swirls the
point, then compresses the swirled result toward the Gaussian center;
`ComposedMap(compress, swirl)` compresses first, then swirls. For most
demos the difference is subtle.
"""
struct ComposedMap{A <: FlowMap, B <: FlowMap} <: FlowMap
    first::A
    second::B
end

@inline function apply_map(m::ComposedMap, x::Real, y::Real, t::Real)
    x1, y1 = apply_map(m.first, x, y, t)
    return apply_map(m.second, x1, y1, t)
end

# Operator-style composition: g ∘ f means "apply f, then g"
# (same as function composition convention).
Base.:∘(g::FlowMap, f::FlowMap) = ComposedMap(f, g)

"""
    CustomMap(f)

Wrap a user-provided callable `f(x, y, t) -> (x', y')` as a `FlowMap`.
Useful for one-off experiments without defining a new struct.
"""
struct CustomMap{F} <: FlowMap
    f::F
end

@inline apply_map(m::CustomMap, x::Real, y::Real, t::Real) =
    let r = m.f(Float64(x), Float64(y), Float64(t))
        (Float64(r[1]), Float64(r[2]))
    end
