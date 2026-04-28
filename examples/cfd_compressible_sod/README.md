# Sod-tube compressible flow (CFD example)

A 2D Sod shock-tube solver that exercises the Phase-2 cell-based AMR
stack of HierarchicalGrids.jl against a problem with an analytic
Riemann reference solution.

## What this demonstrates

This worked example shows how the orchestration primitives compose for
a hyperbolic conservation law with discontinuous solutions:

- **`PolynomialFieldSet{2}` (Bernstein)** for the 4 conserved variables
  `(ρ, ρu, ρv, E)` stored as cell-mean coefficients.
- **`AdaptiveField`** wraps the field-set so refinement events
  automatically resize and remap the conservative variables (degree-0
  coarsening is exact — no projection error).
- **`for_each_face!`** dispatches an HLL-flux kernel over every interior
  face. The `flux_kernel_boundary` argument handles OUTFLOW BCs by
  zeroth-order extrapolation (interior state mirrored to the ghost),
  and PERIODIC y-faces are wired through the orchestrator's BC-aware
  neighbor graph.
- **`step_with_amr!`** drives the time loop with a refine-by-indicator
  pass every `amr_every` steps; the indicator is `|Δρ|/ρ` over leaf
  face-neighbors.
- **`RemapDiagnostics`** is repurposed as a small conservation summary.

## Numerics

- **Equations**: 2D compressible Euler with γ = 1.4.
- **Scheme**: first-order finite-volume Godunov with HLL fluxes.
- **State**: 4 named fields `(rho, rhou, rhov, E)` on
  `BernsteinBasis{2, 0}` — one coefficient per cell = the cell mean.
- **Reconstruction**: piecewise-constant (no slope reconstruction in
  this PR; future work).
- **Time integration**: forward Euler with adaptive CFL-controlled `dt`.
  CFL = 0.4 by default (HLL is stable up to CFL ≈ 0.5 in 2D).
- **BCs**: OUTFLOW on x-axes, PERIODIC on y-axes.
- **AMR**: every `amr_every` steps, the gradient indicator
  `|Δρ_neighbor|/ρ` flags cells for refinement; coarsening fires on
  sibling groups whose indicator is below `coarsen_threshold`.

## Setup

Standard Sod tube extruded to 2D:

- Domain: `[0, 1] × [0, 0.1]`.
- `t = 0`: `(ρ, u, v, p) = (1.0, 0, 0, 1.0)` for x < 0.5;
  `(0.125, 0, 0, 0.1)` for x ≥ 0.5.
- Final time: `t = 0.2`.

The analytic solution at `t = 0.2` consists of:

- Left-going rarefaction (head at x ≈ 0.26, tail at x ≈ 0.49).
- Contact discontinuity at x ≈ 0.69.
- Right-going shock at x ≈ 0.85.

We ship a closed-form exact-Riemann sampler ([`exact_riemann_at`](@ref);
Toro Algorithm 4.1) so the L¹-error test against the analytic ρ is
meaningful.

## Running

```julia
cd examples/cfd_compressible_sod
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or, the standalone driver:

```sh
julia --project=. examples/sod.jl
```

## Expected output

The test suite verifies:

1. Construction: initial mass matches `ρL · 0.5 · 0.1 + ρR · 0.5 · 0.1`.
2. Mass + energy conservation drift after a short run is below `1e-10`.
3. Shock position at `t = 0.2` falls in `(0.80, 0.92)` (analytic ≈ 0.85).
4. AMR engages: `n_leaves_final > n_leaves_initial`.
5. L¹ error vs the analytic Riemann ρ is below 5% on a 64×64 base run.
6. The exact-Riemann sampler reproduces the IC at `t = 0`.
7. cons ↔ prim round-trip is exact.

## Conservation properties

- **Mass / energy**: exactly conserved across PERIODIC and OUTFLOW
  faces under the HLL update. Drift in the test suite is bounded by
  floating-point round-off (≤ `1e-10`).
- **Momentum**: conserved across PERIODIC faces but NOT across OUTFLOW
  faces — by construction, OUTFLOW is a flux sink. The example reports
  the x-momentum drift as a diagnostic (not an invariant).

## Refinement pattern

With AMR enabled, the gradient indicator concentrates refinement
around the contact discontinuity and the shock. The rarefaction fan
has weaker gradients and is refined less aggressively — exactly the
qualitative pattern one wants for a shock-fitting AMR run.

## Files

- `src/CFDCompressibleSod.jl` — the entire solver in one module
  (config, state, HLL, exact Riemann, AMR driver, run-loop entry).
- `test/runtests.jl` — seven tests covering construction,
  conservation, shock-position tracking, AMR engagement, L¹ error,
  and the analytic sampler.
- `examples/sod.jl` — standalone driver.

## References

- Toro, E. F. *Riemann Solvers and Numerical Methods for Fluid Dynamics*
  (3rd ed., Springer 2009). §4.3-4.5 for the exact Riemann solver;
  §10.3 for the HLL scheme.
- Sod, G. A. (1978). *J. Comput. Phys.* 27:1-31.
