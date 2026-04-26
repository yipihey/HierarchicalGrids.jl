# Boundary conditions

Boundary handling crosses several layers — geometry, neighbor graphs,
overlap, and PDE-level flux assembly — so the framework provides a
single small vocabulary that those layers can all read. The vocabulary
is kept advisory at the geometric layer: the mesh, the overlap
computation, and the polynomial remap do not change behavior when a
boundary spec is supplied. Only neighbor wiring and downstream PDE
code consume the spec.

## BCKind

```julia
@enum BCKind PERIODIC INFLOW OUTFLOW REFLECTING DIRICHLET
```

The five supported kinds:

| Kind         | Meaning |
|--------------|---------|
| `PERIODIC`   | Wraps to the opposite side of the same axis. |
| `INFLOW`     | Externally-prescribed state enters the domain. |
| `OUTFLOW`    | Characteristic outflow / zero-gradient. |
| `REFLECTING` | Mirror state across the boundary. |
| `DIRICHLET`  | Pin the boundary value (clamped Lagrangian motion, etc.). |

## BoundarySpec

```julia
const BoundarySpec{D} = NTuple{D, NTuple{2, BCKind}}
```

A `D`-tuple whose entries are `(lo_side, hi_side)` pairs of `BCKind`.
`spec[axis][1]` is the lo-side BC, `spec[axis][2]` is the hi-side BC.
Construct one as a literal:

```julia
spec = ((PERIODIC,   PERIODIC),     # x: periodic
        (REFLECTING, REFLECTING))   # y: reflecting walls
```

A default `default_bc(Val(D))` returns the all-`REFLECTING` spec.

Periodicity is symmetric by construction: an axis with `PERIODIC` on
one side must have it on the other. The constructor of the companion
`FrameBoundaries` validates this and throws `ArgumentError` otherwise.

## FrameBoundaries

`FrameBoundaries{D}` attaches a `BoundarySpec{D}` to an
`EulerianFrame{D, T}` without changing the frame's two-parameter type
signature. Construction is light:

```julia
fb1 = FrameBoundaries(2)                 # all REFLECTING (defaults)
fb2 = FrameBoundaries(Val(3))            # type-stable default
fb3 = FrameBoundaries(spec)              # explicit
```

Accessors:

- `bc(fb, axis, side)` returns the `BCKind` on a given axis-half
  (`side == 1` is lo, `side == 2` is hi).
- `is_periodic_axis(fb, axis)` is the common test.

## Where the spec is consumed

Geometric and remap operators are unchanged when boundaries are not
supplied. The places the framework currently reads a spec are:

- **Periodic neighbor wiring**:
  [`face_neighbors_with_bcs(mesh, i, frame_bcs)`](neighbors_and_halos.md)
  rewires lo/hi entries on every periodic axis to the wrap-around
  neighbor. Other kinds leave the entry as `0` for PDE code to
  handle.
- **Periodic Lagrangian wraps**:
  `periodic!(mesh::SimplicialMesh, axes, bounds)` rewrites the
  simplex-neighbor table so opposite-side faces become interior pairs.
  Useful for cosmological volumes or any periodic Lagrangian flow.
- **Periodic ghost overlaps**: `compute_overlap(...; frame_bcs = fb)`
  accepts the spec for forward compatibility. Wrap-around ghost
  entries against periodic axes are scheduled for a follow-up; until
  then the geometric overlap behaves identically with or without the
  argument. Use `face_neighbors_with_bcs` to walk halos.

## Pinned simplices (Dirichlet)

For Lagrangian meshes, `pin_boundary_simplices!(mesh, indices)` marks
the listed simplices as Dirichlet-pinned. Pinned simplices participate
in geometry as usual but are flagged for downstream motion code: a
solver that integrates vertex velocities should clamp velocities of
pinned simplices to zero (or to a prescribed Dirichlet value). The
flag is queried with `is_pinned(mesh, i)`. The geometric overlap and
remap operators do not consume this flag.

## A periodic-cosmology example

```julia
using HierarchicalGrids

# 2D, x and y both periodic on [0, L]
L    = 1.0
spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
fb   = FrameBoundaries(spec)

eul   = HierarchicalMesh{2}()
frame = EulerianFrame(eul, (0.0, 0.0), (L, L))

# Wrap the Lagrangian mesh so opposite-side faces are interior pairs.
lag = SimplicialMesh{2, Float64}(...)         # user mesh
periodic!(lag, (true, true), ((0.0, L), (0.0, L)))

# Stencil walk: face_neighbors_with_bcs returns the wrap-around leaf
# instead of 0 on periodic-side faces.
i_test = first(enumerate_leaves(eul))
nbrs   = face_neighbors_with_bcs(eul, i_test, fb)

# Pin a curated set of boundary simplices (e.g. on a non-periodic
# axis in a more complex setup).
pin_boundary_simplices!(lag, [1, 2, 3])
@assert is_pinned(lag, 1)
```

## See also

- [neighbors and halos](neighbors_and_halos.md) for
  `face_neighbors_with_bcs` and `HaloView`.
- [overlap](overlap.md) for the `frame_bcs` keyword on
  `compute_overlap`.
- The `BoundaryConditions` submodule re-exports `is_periodic_axis`
  and `validate` for code that wants to work directly with a
  `BoundarySpec` rather than a `FrameBoundaries`.
