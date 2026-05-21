# Cell-mode incompressible Navier-Stokes (scaffold)

**Step 1 of the cell-mode incompressible-NS punch list** —
`HierarchicalMesh{2}` + `AdaptiveField` carrying cell-centered
`(:u, :v)`, with 2nd-order C/F-aware advection driven by
`for_each_face!`. RK2 (Heun) time integration. This is the
scaffolding that the implicit viscous step (step 2), cell-native
multigrid Poisson (step 3), and projection (step 4) will plug into.

## What's here

Pure passive vector advection by a constant carrier velocity:

```
∂_t (u, v) + (carrier · ∇) (u, v) = 0
```

with periodic BCs. The carrier is a fixed `NTuple{2, Float64}` from
the config; once viscous + projection are wired in (later steps), it
becomes the field's own velocity (self-advection).

## Why this is the right step 1

* **Uses the AMR-native orchestrator.** `for_each_face!` dispatches
  per-fine-sub-face at every C/F interface — refluxing is free.
* **Inherits the 2nd-order C/F kernel from `cfd_amr_advection_2o`.**
  Same Martin-Colella transverse correction:
  `face = (2/3)·ρ_F + (1/3)·ρ_C + (1/3)·slope_t·t_offset`,
  applied per scalar component.
* **Builds the right state shape for steps 2–4.** The implicit
  viscous step (JFNK + GMRES + Helmholtz precond) and the projection
  will read / write the same `(:u, :v)` field.
* **Empirically verifies 2nd-order at C/F for the vector advection
  on this state shape** — not the same as the scalar test, since
  here two components are coupled through `for_each_face!`'s
  per-fine-sub-face dispatch on a vector field.

## Empirical convergence

Taylor-Green velocity advected by carrier `(1.0, 0.5)`, t = 0.05,
dt = 1e-3, RK2:

|   | single-level | AMR (refined `[0.25, 0.75]²` block) |
|---|---|---|
| N=8  | 2.48e-2 | 2.23e-2 |
| N=16 | order 1.97 | order 1.91 |
| N=32 | order 1.99 | order 1.90 |
| N=64 | order 2.01 | order 1.94 |

Momentum conserved to `< 1e-11` in both u and v components in both
single-level and AMR setups.

## Code structure

```
src/CFDIncompressibleNSCell.jl
  ├── NSCellConfig                   — mesh, carrier velocity, time scheme
  ├── NSCellState                    — mesh + frame + AdaptiveField + leaves
  ├── taylor_green_ic / _solution    — IC + analytic reference
  ├── build_state                    — uniform mesh + optional center refine
  ├── _cf_face_value                  — Martin-Colella transverse correction
  ├── _accumulate_flux_divergence!    — for_each_face! + periodic-axis pass
  ├── step!                          — Euler or RK2 (Heun)
  ├── run!                           — drive to t_final
  └── total_momentum / kinetic_energy / l2_error — diagnostics
```

## Tests

`test/test_cfd_incompressible_ns_cell.jl` (5 testsets / 14 assertions):

1. **Build + Taylor-Green IC sanity** — mesh size, zero mean momentum,
   `KE(0) = 0.25` for U₀ = 1, k = 2π on `[0, 1]²`.
2. **Momentum conservation under pure advection (uniform)** —
   `|Δ(mu, mv)| < 1e-11` after 100 steps.
3. **Momentum conservation under pure advection (AMR)** — same on a
   center-refined mesh.
4. **Single-level spatial convergence ≥ 2** — three doubling pairs.
5. **AMR (refined block) convergence ≥ 2**.

## Next steps on this path

* **Step 2** — implicit viscous via JFNK on per-component Helmholtz.
  Pattern: `cfd_implicit_ns`. ~300 LoC. Tests: viscous Taylor-Green
  decay matches `exp(-2 k² ν t)`.
* **Step 3** — cell-native multigrid Poisson (the gate). Port the
  FAC V-cycle from `GeometricMultigrid` to operate directly on
  `HierarchicalMesh`, with tree-parent restriction / linear
  prolongation, GS red/black smoother, AMG bottom, and the same
  Martin-Colella flux fix at C/F faces. ~800 LoC. Tests: 2D periodic
  MMS converges in ≤ 8 V-cycles to 1e-10 on AMR mesh; matches
  `solve_poisson!` on a uniform single-patch reference.
* **Step 4** — MAC-on-averaged-faces (or nodal) projection +
  full incompressible NS step + AMR re-refinement via
  `step_with_amr!`. ~250 LoC. Tests: Taylor-Green decay on 3-level
  AMR mesh, lid-driven cavity at Re=100, mass conservation.

## Scope (this step)

* **2D only**, **periodic BCs only** — the punch list adds Dirichlet
  walls in step 4.
* **Static AMR.** The mesh is set up once at `build_state` and not
  re-refined during a run. `step_with_amr!` integration is step 4.
* **Passive vector advection by a constant carrier** — self-advection
  awaits step 4's full incompressible step.
* **No limiter.** Centered + RK2 is stable for smooth flows. Shock-
  capturing would need a TVD slope limiter on top of `_cf_face_value`.
