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
5. **Sod tube**: discontinuous IC, dt ≈ 5× the explicit CFL bound, Newton
   converges every step, positivity preserved in ρ and p.

## Limitations / follow-ups

* **Single level, single patch.** The residual routine consumes the
  `AdaptiveField`'s `for_each_face!`, so adapting to AMR is mostly
  plumbing — the JFNK loop doesn't care about hierarchy depth.
* **First-order spatial accuracy.** Same Godunov + HLL as the Sod tube; no
  slope reconstruction. Backward-Euler is first-order in time. BDF2 / DIRK
  for higher accuracy is left as a follow-up.
* **Block-Jacobi preconditioner only.** Good for moderate `dt`. For stiff
  cases (large `dt`, low Mach, fine grids) a physics-based block precond
  (Schur on the pressure block) is the standard next step.
* **Viscous handling at periodic boundaries is zero-gradient.** The
  interior viscous flux uses full tangential gradients via the
  `face_neighbors` graph; periodic boundary face viscous fluxes are
  dropped. For low-Re flow this is fine; for Re ~ O(1) at the boundary
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
