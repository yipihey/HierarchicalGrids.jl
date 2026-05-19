# Fully implicit compressible Navier-Stokes (Path A: JFNK)

A 2D worked example that takes the same conservative-variable layout as
`cfd_compressible_sod` and replaces forward-Euler + HLL with **backward-Euler
+ Jacobian-Free Newton-Krylov (JFNK)**. Unconditionally stable so the time
step is not bounded by the acoustic CFL — a clean test of the implicit-solver
infrastructure landed in the Tier-1 / Tier-2 AMReX port.

## What this demonstrates

Per backward-Euler step:

```
F(U^{n+1}) := U^{n+1} - U^n - Δt · R(U^{n+1}) = 0
```

where `R(U) = -∇·F(U)/V` is the per-cell net flux (convective HLL + optional
viscous/heat-conduction). The nonlinear equation `F(U)=0` is solved by
Newton iteration, with the Jacobian-vector product

```
J · v  ≈  (F(U + ε v) - F(U)) / ε              (JFNK)
```

fed into GMRES via the Krylov.jl bridge. Optional block-Jacobi preconditioner
(per-cell frozen 4×4 inverse).

## Equation

Compressible Navier-Stokes, ideal gas (γ = 1.4):

```
∂_t ρ      + ∇·(ρu)             = 0
∂_t (ρu)   + ∇·(ρu⊗u + pI - τ)  = 0
∂_t E      + ∇·((E+p)u - τ·u - κ∇T) = 0
```

with stress tensor `τ = μ(∇u + ∇uᵀ) - (2/3)μ (∇·u) I` (Stokes hypothesis).

Inviscid mode (`viscous = false`) drops the τ and κ terms — reducing to
implicit compressible Euler.

## Code structure

```
src/CFDImplicitNS.jl
  ├── EOS + flux helpers       (cons_to_prim, hll_flux, viscous_flux, …)
  ├── ImplicitNSConfig         — knobs (dt, tols, viscous, μ/λ/κ, precond)
  ├── ImplicitNSState          — mesh + adaptive field + JFNK scratch
  ├── compute_residual!        — R(U) = -∇·F(U) / V over the patch
  ├── compute_F!               — Newton residual F(U) = U - U^n - dt R(U)
  ├── JFNKMatvec               — J·v via Pernice-Walker ε directional FD
  ├── BlockJacobiOp            — optional per-cell 4×4 inverse precond.
  └── implicit_ns_step!        — Newton outer + GMRES inner + line search
```

## Tests

`test/runtests.jl` (and the top-level `test/test_cfd_implicit_ns.jl` smoke):

1. **Construction**: build_state initializes ρ = 1 everywhere.
2. **Uniform stationary flow**: backward-Euler step preserves (ρ, 0, 0, p).
3. **Uniform translation**: backward-Euler step preserves (ρ, u, 0, p).
4. **Viscous mode**: a Gaussian density pulse smears more under viscous +
   heat-conducting integration than under pure inviscid.
5. **`refresh_state!` after manual refinement**: refine one cell, verify
   the scratch buffers are resized, and the implicit step still converges
   on the heterogeneous mesh.
6. **AMR-enabled `run!`**: Sod-like initial discontinuity triggers AMR
   every step; mesh ends with strictly more leaves than the initial
   uniform setup, ρ stays positive.
7. **Sod tube**: discontinuous IC, dt ≈ 5× the explicit CFL bound, Newton
   converges every step, positivity preserved in ρ and p.

## AMR support

`for_each_face!` already handles hanging-node (C/F) faces in the
residual function — no JFNK-side changes needed once the mesh is
heterogeneous.  What the implicit step needs around AMR events is:

* `refresh_state!(state)` — rebuilds `state.leaves`, `state.cell_to_idx`,
  and resizes the flat-vector scratch (`U_n`, `U_iter`, `U_pert`,
  `F_iter`, `F_pert`) and the per-cell flux-divergence buffers after a
  refinement event.
* `implicit_ns_step_with_amr!(state, dt; refine_now = …)` — runs one
  Newton-Krylov step on the current mesh, then (if requested) fires
  `refine_by_indicator!` and refreshes the state.
* AMR knobs on `ImplicitNSConfig`: `amr_every`, `refine_threshold`,
  `coarsen_threshold`, `max_level`.  Setting `amr_every = 0` keeps the
  mesh static (same as the v1 behaviour).
* Conservative remapping of (ρ, ρu, ρv, E) at refinement / coarsening
  events is handled by the existing `AdaptiveField` machinery —
  degree-0 BernsteinBasis is mean-preserving, which matches the FV
  conservation law.

The default density-gradient indicator (`gradient_indicator(field,
mesh)`) is the same one used by `cfd_compressible_sod`: per-cell
`|Δρ|/max(ρ, ρ_nbr)` over leaf face-neighbours.

## Limitations / follow-ups

* **Single patch per level.** Mesh refinement adds leaves within the
  single AdaptiveField; there's no second `PatchHierarchy`-style patch
  yet.
* **First-order spatial accuracy.** Same Godunov + HLL as the Sod tube;
  no slope reconstruction.  Backward-Euler is first-order in time.
  BDF2 / DIRK for higher accuracy is left as a follow-up.
* **Block-Jacobi preconditioner only.** Good for moderate `dt`.  For
  stiff cases (large `dt`, low Mach, fine grids) a physics-based block
  precond (Schur on the pressure block) is the standard next step.
* **Viscous handling at periodic boundaries is zero-gradient.** The
  interior viscous flux uses full tangential gradients via the
  `face_neighbors` graph; periodic boundary face viscous fluxes are
  dropped.  For low-Re flow this is fine; for Re ~ O(1) at the boundary
  one would wire the periodic neighbor-graph through.

## Why this is "Path A"

The brainstorming notes describe three implicit-NS designs:

* **Path A — fully implicit JFNK** (this mini-app): one nonlinear solve
  covers advection + viscosity + pressure. Best for stiff cases; the
  preconditioner is the hard part.
* Path B — fractional-step projection for incompressible flow.
* Path C — IMEX (explicit hyperbolic + implicit parabolic).

Path A is the most ambitious. It validates the JFNK + GMRES + (optional)
block-Jacobi stack against a real fluid problem; the same machinery then
plugs into Path C and Path B as needed.
