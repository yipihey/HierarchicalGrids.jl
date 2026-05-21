# 2nd-order AMR scalar advection

A 2D constant-velocity scalar-advection mini-app demonstrating
**genuine 2nd-order accuracy across coarse-fine (C/F) interfaces** on a
Berger-Oliger style adaptive mesh.

## What this demonstrates

The companion mini-apps (`cfd_compressible_sod`, `cfd_implicit_ns`)
already advect on AMR meshes via the `for_each_face!` orchestrator,
which dispatches the flux kernel per-fine-sub-face at C/F interfaces
and gets refluxing for free. They are 1st-order in space, however,
because the HLL flux only sees the two adjacent cells.

For 2nd-order across C/F we need a transverse correction: the sub-face
center is offset from the coarse cell center along the **normal**
direction (by h_coarse/2) *and* the **transverse** direction (by
h_fine/2). A naive 1D linear interpolant captures the normal offset
but misses the transverse — leaving the scheme only 1st-order globally.

This mini-app demonstrates the Martin-Colella style fine-ghost
reconstruction that recovers full 2nd-order. The result, verified by
convergence tests:

| N  | leaves | single-level error | order | AMR error | order |
|----|--------|--------------------|-------|-----------|-------|
| 8  | 64     | 2.51e-2             | —     | 2.25e-2   | —     |
| 16 | 256    | 6.41e-3             | 1.97  | 5.94e-3   | 1.92  |
| 32 | 1024   | 1.61e-3             | 1.99  | 1.57e-3   | 1.92  |
| 64 | 4096   | 4.00e-4             | 2.01  | 4.04e-4   | 1.96  |

(AMR mesh refines the central `[0.25, 0.75]²` block once more, so the
leaf count is ~7× the single-level count at the same `N`.)

## Algorithm

Per face, the flux kernel computes a face-centred ρ value and the flux
`F = u · n · ρ_face`. The orchestrator handles the geometry (face
area, per-fine-sub-face dispatch); the kernel just picks the right
interpolation:

**Same-level face** (most faces):

```
ρ_face = (1/2)·(ρ_L + ρ_R)        ← standard centered, 2nd-order
```

**C/F face** (one of L/R is finer than the other):

1. Identify the coarse cell C, the fine cell F, the face axis, and the
   transverse offset `t = ρ_F's transverse coord − ρ_C's transverse coord`
   (i.e., ±h_fine/2 for a 2:1 refinement in 2D).
2. Find the coarse transverse neighbor T (in the direction of F's
   transverse offset) via `face_neighbors_with_bcs(mesh, i_coarse, bcs)` —
   so periodic wrap is honoured.
3. Compute the transverse slope at C:
   `slope_t = (ρ_T − ρ_C) / |center(T) − center(C)|`.
4. Evaluate the linear function `f(x, y) = a + b·x + c·y` (fit to
   ρ_C, ρ_F, ρ_T) at the sub-face center (0, t):

   ```
   ρ_face = (2/3)·ρ_F + (1/3)·ρ_C + (1/3)·slope_t·t
   ```

The orchestrator fires this kernel once per fine sub-face, so the two
sub-faces touching a coarse face contribute independently — and each
sub-face has its own `t` (one positive, one negative), giving distinct
ρ_face values. The coarse cell's flux balance accumulates both
contributions: exact refluxing for free.

Time integration is **RK2 (Heun)** for 2nd-order in time, so the
overall scheme is 2nd-order in space *and* time on a mesh with C/F
interfaces.

## Code structure

```
src/CFDAMRAdvection2O.jl
  ├── AdvectionConfig             — velocity, refinement, time scheme
  ├── AdvectionState              — mesh + frame + AdaptiveField + leaves
  ├── sinusoidal_ic / _solution   — smooth periodic profile + analytic ρ(x, t)
  ├── build_state                 — uniform mesh + optional central-block refine
  ├── _cf_face_value              — Martin-Colella transverse correction
  ├── _accumulate_flux_divergence! — for_each_face! + periodic-axis pass
  ├── step!                       — RK2 or forward-Euler one step
  ├── run!                        — time-stepping loop
  ├── total_mass / l2_error       — diagnostics
  └── ...
```

## Tests

`test/test_cfd_amr_advection_2o.jl` (5 testsets / 10 assertions):

1. **Build + IC sanity** — leaves, IC mass.
2. **Mass conservation, single-level** — `|m(T) − m(0)| < 1e-11`.
3. **Single-level spatial convergence ≥ 2** — three consecutive
   doubling pairs, observed slopes 1.97–2.01.
4. **AMR convergence ≥ 2** — same with centered refined block,
   observed slopes 1.92–1.96 (the transverse Martin-Colella correction
   is essential; without it the slope drops to ~1.0).
5. **Mass conservation, AMR** — refluxing handled by the per-fine-sub-face
   orchestrator dispatch; `|m(T) − m(0)| < 1e-11`.

## Scope

* **Single HierarchicalMesh; up to 2:1 refinement** (one extra level
  inside the refined block). Multiple levels of refinement nested
  inside each other are not exercised but should work because the
  formula is generic in `t_offset`.
* **2D only.** The same formula generalises to 3D, but requires
  estimating two transverse slopes (one per transverse axis) and
  evaluating the linear function at the sub-face center in 3 axes.
* **Periodic BCs only.** Non-periodic outer boundaries would need a
  boundary kernel; the structure exists in `for_each_face!` via
  `flux_kernel_boundary`.
* **Linear reconstruction** (3-point fit). No slope limiting — fine
  for smooth flows; for shock-capturing the formula would need to be
  combined with a TVD limiter on the cell-centered fluxes (independent
  of the C/F handling).
* **Centered scheme without dissipation.** Stable with RK2 at moderate
  CFL on smooth flows; not for high-Re turbulence.

## Why this matters

This is the building block needed to upgrade the existing CFD
mini-apps to 2nd-order across AMR boundaries:

* **`cfd_compressible_sod`, `cfd_implicit_ns`** — replace HLL with
  a Martin-Colella-reconstructed flux (or add MUSCL slopes + this
  C/F-aware face value), gain 2nd-order at C/F.
* **`cfd_incompressible_ns`** — same upgrade for the advection term,
  paired with a multi-level FAC MAC projection (Tier 2 follow-on) to
  give a fully 2nd-order incompressible AMR mini-app.

The hard part (the transverse correction) is just an ~80-line helper
that any kernel can call. The orchestrator handles everything else
(face-area weighting, per-fine-sub-face dispatch, refluxing).
