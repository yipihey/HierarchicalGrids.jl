# Cell-based scalar advection (CFD example)

A minimal 2D linear scalar advection solver that exercises the Phase-2
cell-based AMR stack of HierarchicalGrids.jl.

## What this demonstrates

This worked example shows how the pieces of the cell-based AMR stack
fit together end-to-end:

- **`PolynomialFieldSet{2}` (Bernstein storage)** for the scalar field.
- **`AdaptiveField`** wraps the field-set so refinement events
  automatically resize and remap coefficients.
- **`for_each_face!`** dispatches an upwind-flux kernel over interior
  faces; periodic boundary fluxes are handled separately via the
  BC-aware neighbor wiring (`face_neighbors_with_bcs`).
- **`for_each_cell!`** is exercised indirectly through the IC
  initialization (`init_field_from!`), and the structure is set up so
  a degree-1 follow-up could swap in a per-cell update kernel without
  touching the orchestration scaffold.
- **`step_with_amr!`** drives the time loop with periodic
  refine-by-indicator cycles.
- **`RemapDiagnostics`** is repurposed as a small mass-conservation
  + monotonicity summary.

## Numerics

- **Scheme**: first-order finite-volume upwind on cell means.
- **Storage**: `BernsteinBasis{2, 0}` (one coefficient per cell = the
  cell mean). The user spec asked for "degree-1 Bernstein, FV-like"
  — those two requests are in tension; we resolve it by using a
  degree-0 Bernstein so the single coefficient is exactly the FV
  state variable, the L²-projection IC reduces to the cell-mean
  integral, and `AdaptiveField` coarsening is exact (no higher-degree
  warning).
- **Time integration**: explicit forward Euler.
- **BCs**: periodic on both axes.
- **AMR**: `step_with_amr!` invokes `refine_by_indicator!` every
  `amr_every` steps, using a finite-difference proxy of `|∇c|` (the
  max absolute difference between a cell and its leaf neighbors) as
  the indicator.

## Running

```julia
cd examples/cfd_cell_advection
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Expected output

The test suite verifies:

1. Initial mass matches the integral of the Gaussian IC over the box.
2. Mass conservation drift after a short run is below `1e-12`.
3. Mass conservation drift after one full period is below `1e-10`.
4. L¹ error after one full period is below `0.10` (first-order upwind
   on a 16×16 grid over a full period is highly diffusive; this is
   the expected magnitude).
5. `step_with_amr!` exercising the refine-by-indicator pathway still
   conserves mass to within `1e-10`.
6. The gradient indicator is finite, nonneg, and nontrivial.

## Refinement pattern

With AMR enabled (`amr_every > 0`), the example refines around
gradient features (the leading and trailing edges of the advected
Gaussian). Coarsening fires on sibling groups whose indicator is
below the coarsen threshold; with the default hysteresis ratio
(refine / 4), the pattern stabilizes after a few cycles.

## Files

- `src/CFDCellAdvection.jl` — the entire solver in one module
  (configuration, state construction, flux kernel, time step, AMR
  driver wiring, and run-loop entry point).
- `test/runtests.jl` — six smoke tests covering construction,
  conservation, tracking, AMR, and the indicator.
