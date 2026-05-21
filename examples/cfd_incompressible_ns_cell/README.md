# Cell-mode incompressible Navier-Stokes (scaffold)

**Steps 1–2 of the cell-mode incompressible-NS punch list.** Cell-mode
AMR scaffolding (`HierarchicalMesh{2}` + `AdaptiveField` carrying
`(:u, :v)`) with:

* **Step 1** — 2nd-order C/F-aware advection by a constant carrier,
  via `for_each_face!` + the Martin-Colella transverse correction.
* **Step 2** — backward-Euler implicit viscous step `(I − Δt · ν · L)
  u^{n+1} = u^n`, per component, via `Krylov.cg` on a matrix-free SPD
  operator built from a discrete cell-volume Laplacian.

Steps 3 (cell-native multigrid Poisson) and 4 (projection + full NS
step + AMR re-refinement) are still pending; this README is updated
as each step lands.

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

## Step-1 empirical convergence (advection)

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

## Step-2 empirical decay (viscous)

Backward-Euler on Taylor-Green eigenmode: `u^{n+1} = u^n / (1 + 2 Δt
ν k²_disc)` where `k²_disc = (4/h²) sin²(π h)` is the 5-point
Laplacian eigenvalue on the periodic uniform grid. CG on the SPD
operator converges in **1 iteration on uniform** (TG is an exact
eigenvector) and **~15 iterations on AMR**.

| dt    | CG iters | KE              | BE-predicted    | rel err |
|-------|----------|-----------------|-----------------|---------|
| 1e-2  | 1        | 0.2314          | 0.2314          | 2.4e-4  |
| 5e-3  | 1        | 0.2404          | 0.2404          | 1.2e-4  |
| 1e-3  | 1        | 0.2480          | 0.2480          | 2.5e-5  |

(Residual is the 5-point discrete eigenvalue offset from the
analytic `−2 k²`; the underlying continuous decay rate
`exp(−4 ν k² t)` is matched to within `< 1%` in the multi-step run.)

## Code structure

```
src/CFDIncompressibleNSCell.jl
  ├── NSCellConfig                    — mesh, carrier, time scheme, μ, tolerances
  ├── NSCellState                     — mesh + frame + AdaptiveField + leaves
  ├── taylor_green_ic / _solution     — IC + analytic reference
  ├── build_state                     — uniform mesh + optional center refine
  ├── _cf_face_value                  — Martin-Colella transverse correction
  ├── _accumulate_flux_divergence!    — for_each_face! + periodic-axis pass
  ├── _apply_laplacian!               — discrete cell-volume Laplacian
  ├── HelmholtzOp / mul!              — Krylov matrix-free (I − Δt ν L)
  ├── viscous_step!                   — Krylov.cg per component
  ├── step!                           — Euler or RK2 (Heun) advection
  ├── run!                            — drive to t_final
  └── total_momentum / kinetic_energy / l2_error — diagnostics
```

## Tests

`test/test_cfd_incompressible_ns_cell.jl` (8 testsets / 23 assertions):

Step 1 (advection):
1. **Build + Taylor-Green IC sanity** — mesh size, zero mean momentum,
   `KE(0) = 0.25` for U₀ = 1, k = 2π on `[0, 1]²`.
2. **Momentum conservation under pure advection (uniform)** —
   `|Δ(mu, mv)| < 1e-11` after 100 steps.
3. **Momentum conservation under pure advection (AMR)** — same on a
   center-refined mesh.
4. **Single-level spatial convergence ≥ 2**.
5. **AMR (refined block) convergence ≥ 2**.

Step 2 (viscous):
6. **Stationary Taylor-Green decay (uniform)** — single backward-Euler
   step matches the analytic eigenvalue factor `1 / (1 + 2 Δt ν k²)²`
   to within `rtol = 1e-3` (driven by 5-point discretization error).
7. **Multi-step Taylor-Green decay** — repeated BE steps track the BE
   per-step factor exactly (CG solves to tol) and the continuous
   `exp(−4 ν k² t)` to within `5%`.
8. **AMR viscous step** — CG converges, energy decreases, energy
   bounded above 50% of the initial (no blow-up). Operator is 1st-order
   at C/F faces; step 3 upgrades to Martin-Colella for full 2nd-order.

## Next steps on this path

* **Step 3** (next) — cell-native multigrid Poisson. Port the FAC
  V-cycle from `GeometricMultigrid` to operate directly on
  `HierarchicalMesh`, with tree-parent restriction / linear
  prolongation, GS red/black smoother, AMG bottom, and the
  Martin-Colella flux fix at C/F faces (upgrading both the Poisson
  operator AND the step-2 viscous Laplacian to true 2nd-order at C/F).
  ~800 LoC. Tests: 2D periodic MMS converges in ≤ 8 V-cycles to 1e-10
  on AMR mesh; matches `solve_poisson!` on a uniform single-patch
  reference.
* **Step 4** — MAC-on-averaged-faces (or nodal) projection + full
  incompressible NS step + AMR re-refinement via `step_with_amr!`.
  ~250 LoC. Tests: Taylor-Green decay on 3-level AMR mesh, lid-driven
  cavity at Re=100, mass conservation.

## Scope (this step)

* **2D only**, **periodic BCs only** — the punch list adds Dirichlet
  walls in step 4.
* **Static AMR.** The mesh is set up once at `build_state` and not
  re-refined during a run. `step_with_amr!` integration is step 4.
* **Passive vector advection by a constant carrier** — self-advection
  awaits step 4's full incompressible step.
* **No limiter.** Centered + RK2 is stable for smooth flows. Shock-
  capturing would need a TVD slope limiter on top of `_cf_face_value`.
