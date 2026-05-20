# 2D incompressible Navier-Stokes (combined Option A + Option B)

A worked example exercising the AMReX-port elliptic stack
(`mac_project!`, `solve_vector_abec!`, `MGWorkspace`, `Krylov.jl`) on the
incompressible Navier-Stokes problem. Implements *both* schemes from
`docs/INCOMPRESSIBLE_NS_OPTIONS.md` behind a single `time_scheme` knob:

* **`:cn`** — explicit centered advection + Crank-Nicolson implicit
  viscous via `solve_vector_abec!` + MAC-averaged approximate
  projection. The IAMR-Lite v1 (no Godunov upwinding yet).
* **`:sdirk2`** — 2-stage L-stable SDIRK2 backward implicit on the
  velocity-only system. Each stage is solved by JFNK
  (Jacobian-free Newton-Krylov, Pernice-Walker ε directional FD) with
  GMRES via `Krylov.jl`. The per-component Helmholtz inverse
  `(I + γΔt ν L)⁻¹` is the right-preconditioner. Divergence is enforced
  post-stage by the MAC-averaged approximate projection.

## What this demonstrates

Per `:cn` step:

```
u^* = u^n - Δt · advec(u^n) + Δt · ν · Δu^n / 2         (explicit half-step)
(I - Δt ν / 2 · Δ) u^{**} = u^*                           (CN viscous, VectorABec)
u_face = avg(u^{**} to faces)                              (cell → face)
mac_project!(u_face, β = 1/ρ)                              (MAC projection)
u^{n+1} = avg(u_face to cells)                             (face → cell)
```

Per `:sdirk2` stage (γ = 1 − 1/√2):

```
F(u) := u + γΔt (advec(u) − ν Δu) − rhs  = 0              (JFNK on 2N-vector W=(u,v))
  J · v ≈ (F(W + ε v) − F(W)) / ε                          (Pernice-Walker FD)
  GMRES with right-preconditioner (I + γΔt ν L)⁻¹           (HelmholtzPrecond)
Apply MAC-averaged projection to enforce ∇·u ≈ 0.          (Same as CN.)
```

## Equation

Constant-density incompressible NS:

```
∂_t u + (u · ∇) u = −∇p + ν Δu
∇·u = 0
```

with ρ = 1 throughout, ν = config.μ. Pressure is recovered via the
projection step (not stored as a primary state).

## Code structure

```
src/CFDIncompressibleNS.jl
  ├── Config + State                  — knobs (dt, scheme, ν, tols) and patch/MG storage
  ├── taylor_green_ic / _solution      — analytic IC + reference solution
  ├── build_state                      — PatchHierarchy + MGWorkspace + scratch
  ├── compute_cell_divergence! /        — centered 2nd-order stencils on cells
      _gradient! / _advection! /         (periodic via mod1 wrap)
      _laplacian!
  ├── project_cell_velocity!           — avg cells → faces, mac_project!, avg back
  ├── cn_viscous_step!                 — solve_vector_abec! for (I − Δt ν/2 · L) u = rhs
  ├── step_cn!                         — Option A driver
  ├── JFNKMatvec                       — Pernice-Walker J · v via directional FD
  ├── HelmholtzPrecond                 — right-precond via solve_vector_abec!
  ├── sdirk2_stage!                    — Newton outer + GMRES inner + Armijo + project
  ├── step_sdirk2!                     — 2-stage SDIRK2 driver
  └── run!                              — time-stepping loop
```

## Verification

`test/test_cfd_incompressible_ns.jl`:

1. **Build + IC** — Taylor-Green is initialised exactly at cell centers;
   both face and cell divergence start at machine precision.
2. **`:cn` one-step face divergence** — MAC-averaged projection drives
   face divergence to MG tolerance after each step.
3. **`:cn` Taylor-Green decay** — 4 steps at moderate ν decay the
   kinetic energy to within 15% of the analytic `KE₀ · exp(−4 k² ν t)`.
4. **`:sdirk2` Newton + GMRES + projection converge** — Newton hits its
   target residual within `newton_maxiter`, GMRES converges per
   iteration.
5. **`:sdirk2` Taylor-Green still decays** — qualitative decay check
   only (see *Limitations* below).

## Scope (v1)

* **Single patch, single level.** The Berger-Oliger AMR machinery is
  available via `PatchHierarchy` + the refinement listener, but this
  v1 doesn't enable it. Adding multi-level requires either (a) FAC
  multi-level MAC projection (Tier 2 follow-on) or (b) explicit
  Schwarz-style patch coupling.
* **2D only**, **periodic BCs only**. The implementation is shaped to
  generalise (3D adds an axis to the centered stencils; non-periodic
  BCs need wall / inflow / outflow ghost handling in the projection
  Poisson — `mac_project!` already supports DIRICHLET / NEUMANN via the
  `FrameBoundaries` spec, only the cell-centered advection here
  hard-codes periodic wrap).
* **Constant density, constant viscosity.** The MGWorkspace + ABec
  infrastructure supports variable ρ (β = 1/ρ at faces) and variable μ
  trivially; only the mini-app's CN solver hard-codes constants for
  brevity.
* **Centered (2nd-order) advection.** No slope limiter, no Godunov
  upwinding. Stable for low-Re / smooth flows like Taylor-Green;
  insufficient for high-Re or steep gradient problems. A Godunov
  predictor following the `cfd_compressible_sod` HLL pattern is the
  obvious follow-up.

## Limitations of the `:sdirk2` path (v1)

The cell-centered velocity representation cannot exactly hold a
face-divergence-free field — the discrete `avg` operator (cells →
faces) has a nontrivial kernel that injects O(h²) divergence on the
"checkerboard-like" mode. Consequences:

* **Operator-split projection has order reduction.** After each SDIRK2
  stage, post-projection cell velocities are not the exact saddle-point
  solution. The single-step error is roughly 1st-order in time despite
  SDIRK2's underlying 2nd-order accuracy. The natural v2 fix is to
  couple pressure as an explicit JFNK unknown (saddle-point system) and
  use a Schur-complement / PCD / LSC preconditioner — but that requires
  a *consistent* `D ∘ G = L` discretisation at cell centers (which the
  centered-finite-difference layout here doesn't provide). The
  AMReX-style fix is a **nodal projection** with the proper `D ∘ G`
  adjoint pair — that's a separate Tier 2 follow-up.
* **Residual face divergence after `:sdirk2` step is O(h²)**, not at
  MG tolerance. The MAC projection completes cleanly inside the stage,
  but the subsequent average-back-to-cells loses the precise
  face-div-free property.

The `:cn` path does not suffer the same order reduction because its
explicit advection step on a smooth (Taylor-Green-like) field leaves the
cell-centered velocity essentially equal to the analytic field plus a
pure-gradient correction; the projection removes the gradient exactly,
and the average round-trip is benign.

## Follow-ups (v2 candidates)

* **Saddle-point JFNK with nodal projection** — proper 2nd-order
  SDIRK2 in time, with pressure as an explicit JFNK unknown. Needs the
  nodal projection adjoint pair to be wired up (uses
  `solve_node_laplacian!` already shipped + new cell-to-node divergence
  + node-to-cell gradient operators).
* **Godunov edge-velocity predictor** — 2nd-order upwind advection,
  slope-limited Taylor extrapolation, enables high-Re lid-driven
  cavity. Mirror `cfd_compressible_sod`'s HLL approach.
* **Variable density** — pass β = 1/ρ to `mac_project!`; advect a
  scalar density via the existing cell-advection pattern. Falling-drop
  / Rayleigh-Taylor demos.
* **AMR** — `PatchHierarchy` + refresh listener pattern from
  `cfd_implicit_ns`; requires a multi-level MAC projection
  (Tier 2 follow-on).
* **Dirichlet / inflow BCs** — extend the cell-centered advection /
  diffusion stencils to honour `FrameBoundaries` BC kinds; enables
  the lid-driven cavity benchmark.

## Why "combined A + B"

Both schemes share the same `IncompressibleNSState`, the same
`MGWorkspace`, the same advection / Laplacian / divergence / gradient
stencils, and the same projection. Only the time integrator differs —
`:cn` drives the explicit + implicit fractional steps directly,
`:sdirk2` wraps the implicit advection-diffusion in a JFNK Newton-Krylov
loop. This makes the two paths directly comparable and stress-tests the
shared infrastructure.

See `docs/INCOMPRESSIBLE_NS_OPTIONS.md` for the design rationale and the
broader option comparison (this mini-app implements options A + B; the
third option, ABCH low-Mach reactive, remains a separate prospect).
