# Topology-only AMR driver

`step_with_amr!` is a small interleaver that drives a user-supplied
physics step against an indicator-driven refinement schedule. It does
not own per-cell state, does not compute CFL/dt, and does not
re-implement refinement — it just calls `step!`, then periodically
calls `refine_by_indicator!`. Per-cell state stays consistent across
refinement via the [refinement-listener](refinement_events.md)
mechanism.

## Signature

```julia
step_with_amr!(state, frame::EulerianFrame{D, T},
               step!, indicator,
               n_steps::Integer;
               refine_threshold::Real,
               coarsen_threshold::Real = refine_threshold / 4,
               hysteresis_steps::Integer = 3,
               max_level::Integer = typemax(Int),
               isotropic::Bool = true)::Int
```

- `state` — opaque user data passed unchanged to `step!`. Typical
  payloads: a `PolynomialFieldSet`, a particle list, a hydro scratch
  struct.
- `frame::EulerianFrame{D, T}` — frame whose `mesh` is the target of
  refinement.
- `step!(state, frame)` — user callback. Mutates `state` (and optionally
  the frame's mesh) in place.
- `indicator(mesh)` — user callback. Returns anything
  `refine_by_indicator!` accepts: an iterable of length `n_cells(mesh)`
  or a callable `f(cell_index)`.
- `refine_threshold` / `coarsen_threshold` — flagged sibling groups all
  below `coarsen_threshold` coarsen; any leaf above `refine_threshold`
  refines. Default coarsen threshold is a quarter of the refine
  threshold (hysteresis).
- `hysteresis_steps` — AMR fires every `hysteresis_steps`-th call. Set
  to `0` to disable AMR cycles entirely (a benchmarking baseline).
- `max_level`, `isotropic` — passed through to `refine_by_indicator!`.

The driver runs exactly `n_steps` step calls and returns that count.
It deliberately does not loop on time or convergence — wrap it in
your own outer time loop.

## A complete example

```julia
using HierarchicalGrids

# Mesh + frame
eul   = HierarchicalMesh{2}()
frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

# Per-cell state. Keep it in sync with the mesh by registering a listener.
field = allocate_polynomial_fields(SoA(), BernsteinBasis{2, 1}(),
                                    n_cells(eul); rho = Float64)
register_refinement_listener!(eul) do event
    resize_fields!(field, n_cells(eul))
    # ... copy parent values to new children, etc.
end

# Physics callback (a placeholder identity step)
step!(state, frame) = nothing

# Indicator callback: refine where rho's leading coefficient is large
function indicator(m)
    return [Float64(field.rho[i][1]) for i in 1:n_cells(m)]
end

n_done = step_with_amr!(field, frame, step!, indicator, 30;
                        refine_threshold = 1.5,
                        coarsen_threshold = 0.4,
                        hysteresis_steps = 5)
@assert n_done == 30
```

## Composition with refinement listeners

The driver itself does not register listeners. It assumes that any
state passed in via `state` has already arranged for refinement
updates — typically by calling
[`register_refinement_listener!`](refinement_events.md) on
`frame.mesh` once at setup. This keeps the driver agnostic about
which fields exist and what interpolation policy applies on
refinement.

## See also

- [Refinement events](refinement_events.md) — how to keep state
  consistent across each AMR cycle.
- `refine_by_indicator!` — the underlying refinement primitive that
  the driver invokes.
