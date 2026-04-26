"""
    BoundaryConditions

Top-level submodule providing a unified per-axis-half boundary-condition
specification for Eulerian and Lagrangian meshes.

The framework defines five kinds of boundary condition (`BCKind`):

- `PERIODIC`   — wraps to the opposite side of the same axis.
- `INFLOW`     — externally-prescribed state enters the domain.
- `OUTFLOW`    — characteristic outflow (zero gradient / extrapolation).
- `REFLECTING` — mirror state across the boundary.
- `DIRICHLET`  — pin the boundary value (clamped Lagrangian motion, etc.).

A `BoundarySpec{D}` is a `D`-tuple whose entries are 2-tuples of `BCKind`
(lo side, hi side). The companion struct `FrameBoundaries{D}` (defined in
`src/Overlap/frame_boundaries.jl`) attaches a `BoundarySpec` to an
`EulerianFrame` without altering the frame's two-parameter type signature.

Periodic axes are special: the lo and hi sides must agree (both
`PERIODIC` or neither). The validator rejects half-periodic axes.

The non-periodic kinds are advisory at this layer; downstream PDE code
consumes them when assembling face fluxes, boundary integrals, or
clamping Lagrangian motion. The geometric overlap computation is
unaffected when no `FrameBoundaries` is supplied. The neighbor graph
exposes a thin wrap-around helper, `face_neighbors_with_bcs`, that
post-processes boundary entries for periodic axes.

# Example

```julia
spec = ((PERIODIC, PERIODIC), (REFLECTING, REFLECTING))   # 2D, x-periodic
fb   = FrameBoundaries(spec)
is_periodic_axis(fb, 1)   # true
is_periodic_axis(fb, 2)   # false
```
"""
module BoundaryConditions

export BCKind, PERIODIC, INFLOW, OUTFLOW, REFLECTING, DIRICHLET
export BoundarySpec
export default_bc, is_periodic_axis, validate

"""
    BCKind

Enumeration of supported boundary-condition kinds:
`PERIODIC`, `INFLOW`, `OUTFLOW`, `REFLECTING`, `DIRICHLET`.
"""
@enum BCKind PERIODIC INFLOW OUTFLOW REFLECTING DIRICHLET

"""
    BoundarySpec{D} = NTuple{D, NTuple{2, BCKind}}

Per-axis, per-side boundary kinds. `spec[axis][1]` is the lo-side BC,
`spec[axis][2]` is the hi-side BC.
"""
const BoundarySpec{D} = NTuple{D, NTuple{2, BCKind}} where D

"""
    default_bc(::Val{D}) where D -> BoundarySpec{D}

Return a default `BoundarySpec` for dimension `D` with all sides
`REFLECTING`. Used when no spec is supplied.
"""
@inline default_bc(::Val{D}) where {D} =
    ntuple(_ -> (REFLECTING, REFLECTING), Val(D))

"""
    is_periodic_axis(spec::BoundarySpec, axis::Integer) -> Bool

Whether the given axis is periodic in `spec` (both sides PERIODIC).
"""
@inline function is_periodic_axis(spec::NTuple{D, NTuple{2, BCKind}},
                                   axis::Integer) where {D}
    return spec[axis][1] === PERIODIC && spec[axis][2] === PERIODIC
end

"""
    validate(spec::BoundarySpec) -> spec

Validate a boundary spec. Currently enforces:

- Periodicity is symmetric: an axis with `PERIODIC` on one side must
  also have it on the other side.

Throws `ArgumentError` on violation; otherwise returns `spec` unchanged
so it can be used in expression position.
"""
function validate(spec::NTuple{D, NTuple{2, BCKind}}) where {D}
    for d in 1:D
        lo, hi = spec[d]
        ((lo === PERIODIC) == (hi === PERIODIC)) ||
            throw(ArgumentError(
                "boundary axis $d has PERIODIC on only one side; " *
                "PERIODIC must apply to both sides of an axis or neither"))
    end
    return spec
end

end # module BoundaryConditions
