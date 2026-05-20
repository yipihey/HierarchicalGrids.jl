# Incompressible / low-Mach Navier-Stokes — design options

After the AMReX-port wave and the fully-implicit compressible mini-app
(`cfd_implicit_ns`, Path A / JFNK), the natural follow-on demo is a
high-fidelity *incompressible* (or low-Mach) NS solver. We have every
elliptic + projection + AMR primitive needed; the open question is
which scheme to put on top.

This doc compares three candidate schemes that span the design space.
None is "wrong" — each lights up a different subset of the new
infrastructure and serves a different audience.

## What we already have (incompressible-NS perspective)

| Need | Have | File |
|------|------|------|
| MAC (face) velocity projection | ✓ | `Solver/MACProjection.jl` |
| Approximate (nodal) projection | ✓ | `Solver/NodeLaplacian.jl`, `Solver/NodeLaplacianML.jl` |
| Variable-coefficient Helmholtz | ✓ | `Solver/ABecLaplacian.jl` |
| Per-component vector Helmholtz | ✓ | `Solver/VectorABec.jl` |
| Full tensor viscosity (∇·τ) | ✓ | `Solver/TensorOp.jl` |
| Edge-centered MAC velocity storage | ✓ | `Solver/EdgeFields.jl` |
| BoomerAMG / AMG bottom solver | ✓ | `Solver/HYPREBottom.jl`, `Solver/AMGBottom.jl` |
| Krylov methods (CG/GMRES/BiCGStab/FGMRES) | ✓ | `Solver/KrylovBridge.jl` |
| JFNK matvec + GMRES + block-Jacobi precond | ✓ | `examples/cfd_implicit_ns/src/CFDImplicitNS.jl` |
| Stiff chemistry per-cell integrator | ✓ | `Solver/StiffChemistry.jl` |
| Berger-Oliger patch AMR | ✓ | `Solver/PatchHierarchy.jl` |
| Conservative remap (mean-preserving) | ✓ | `Solver/AdaptiveField.jl` |
| Refinement listener + `step_with_amr!` | ✓ | `src/Mesh/Mesh.jl`, `Solver/PatchHierarchy.jl` |

The list above also serves as the test of "Path B" (incompressible
projection) and "Path C" (IMEX) from the brainstorm — every elliptic
piece either of them needs is already shipped.

## Option A — Faithful IAMR / ABC port

**Reference.** Almgren, Bell, Colella, Howell, Welcome 1998, "A
conservative adaptive projection method for the variable-density
incompressible Navier-Stokes equations" (JCP 142:1, the IAMR /
VARDEN scheme). 25+ years of literature and code to compare against.

**Time integration.** Per (sub-)step, on each level:

1. **Godunov edge-velocity predictor** — second-order upwind Taylor
   extrapolation from cell centres to edge-centred values at `t^{n+1/2}`,
   with slope limiting.
2. **MAC project** — solve a cell-centred Poisson so the *advective*
   velocity at `t^{n+1/2}` is discretely divergence-free at edges. Uses
   `mac_project!` directly.
3. **Scalar / density update** — `ρ^{n+1} = ρ^n − Δt ∇·(ρU)^{n+1/2}` with
   the projected MAC velocity.
4. **Crank-Nicolson viscous momentum solve** — Helmholtz solve per
   velocity component (or full tensor) for `u^{n+1}`:
   ```
   (ρu)^{n+1} − Δt/2 · ∇·τ(u^{n+1}) = (ρu)^n + Δt/2 · ∇·τ(u^n)
                                      − Δt · ∇·(ρu⊗u)^{n+1/2} − Δt · ∇p
   ```
   Drives `VectorABec` or `TensorOp` + multigrid + HYPRE bottom.
5. **Approximate projection** — solve a nodal Poisson on `(ρu)^{n+1}` to
   project to (approximately) divergence-free. Uses
   `solve_node_laplacian!` or the multi-level variant.

**AMR.** Subcycled in time on finer levels (typically 2:1 refinement,
2:1 in time). Synchronisation at the end of a coarse step re-fluxes the
momentum advection at C/F faces and re-projects to enforce composite
divergence-free.

**What we'd need to build.** Mostly glue:

- Godunov edge-velocity predictor with slope limiter (~150 LoC, mirrors
  the compressible Sod predictor).
- Sync re-flux + sync projection driver (~100 LoC).
- Mini-app config + IC plumbing (~150 LoC).

**What it demonstrates.**

- 2nd-order accuracy in space *and* time.
- Variable-density incompressible NS (falling-drop, Rayleigh-Taylor).
- Lid-driven cavity at Re ∈ {100, 400, 1000} matching IAMR.
- Double shear layer with AMR refinement on `|∇u|`.
- The MAC + approximate-projection pairing in production.

**Pros.** Battle-tested algorithm. Plenty of reference solutions.
Decoupled solves — each elliptic solve is moderate-sized. The "classic"
incompressible-AMR mini-app; immediately recognisable.

**Cons.** Crank-Nicolson on viscous is only conditionally stable for
truly stiff diffusion (high `μ`). Synchronisation re-fluxing is
intricate to get right. Velocity components are *decoupled* in the
viscous solve, which is fine for `μ = const` but loses cross-component
coupling for tensor viscosity.

## Option B — SDIRK2 + JFNK extension

**Reference.** Knoll & Keyes 2004 review (JCP 193), Persson & Peraire
2008 (SISC 30) for SDIRK-IMEX with Newton-Krylov. This is the modern
"fully implicit incompressible" line of work — what people are
publishing in the late 2010s / early 2020s for stiff low-Re flows,
steady RANS, and very-large-`Δt` transient simulations.

**Time integration.** Replace Option A's CN viscous + projection with an
SDIRK2 (or BDF2) coupled momentum + projection solve:

```
For each SDIRK2 stage s:
    F(u_s, π_s) = [ M(ρ) u_s + γΔt · (A_adv(u_s) − ν Δu_s + ∇π_s) − rhs_s
                  , ∇·u_s ]
    Solve F = 0 by Newton; J·v by Pernice-Walker JFNK; GMRES bridge.
```

The unknown vector `(u, π)` is a saddle-point system. The right block
preconditioner is the differentiator: PCD (pressure
convection-diffusion, Elman-Silvester-Wathen), LSC (least-squares
commutator), or "approximate projection as preconditioner" all work.
PCD/LSC are the SOTA references; they exploit our existing ABec + nodal
Laplacian as building blocks.

**AMR.** Same Berger-Oliger framework; sub-cycling becomes optional
because we can take huge `Δt` on every level (each Newton step is
internally well-conditioned).

**What we'd need to build.**

- SDIRK2 / BDF2 stage driver (~80 LoC; trivial Butcher tableau).
- Coupled (u, π) JFNK matvec — extends the cfd_implicit_ns matvec to
  include the divergence-constraint row (~150 LoC).
- Block preconditioner: PCD or LSC (~250 LoC, the real engineering).
  Both can be expressed as "apply ABec on velocity block, apply nodal
  Laplacian on pressure block, with a Schur-style sandwich".
- Mini-app harness (~150 LoC).

**What it demonstrates.**

- High-order time accuracy (2nd-order L-stable; trivial extension to
  3rd-order ESDIRK3).
- Genuinely unconditional time-step (`Δt` bounded only by accuracy, not
  stability).
- Direct showcase of the JFNK + GMRES + multigrid-preconditioner stack
  shipped in `cfd_implicit_ns` and the `Krylov.jl` bridge.
- Steady-state lid cavity in O(10) Newton solves regardless of mesh.
- Verification via Taylor-Green decay-rate convergence sweep
  (`order = 2.0 ± 0.05` in both space and time).

**Pros.** Modern, publishable algorithm. Stresses every implicit piece
we shipped — JFNK, multigrid preconditioner, GMRES, AMG bottom. For
steady / quasi-steady or stiff transient flows, can be 10×+ faster than
explicit advection. Sets up cleanly for adding a velocity-pressure
turbulence model (k-ω, SA, RANS).

**Cons.** The block preconditioner is the hard part — get it wrong and
GMRES counts blow up to thousands per Newton step. Roughly 2× the
engineering of Option A. For high-Re transient turbulence, the cost
of solving the full coupled implicit system per step may exceed the
classical IAMR explicit-advection cost, so the "wins" are concentrated
in stiff or steady regimes.

## Option C — Low-Mach reactive (ABCH)

**Reference.** Day & Bell 2000 "Numerical simulation of laminar
reacting flows with complex chemistry" (CTM 4:4) — the LMC algorithm,
later evolved into PeleLM. Also: Almgren, Bell, Crutchfield, Howell,
Pember 1998 (the "ABCH" letters).

**Equations.** Density obeys an EOS, `p_0 = ρ R T(Y)`, *not* a pure
advection; the velocity divergence is *non-zero* and driven by heat
release:

```
∂ρ/∂t + ∇·(ρu) = 0
∂(ρu)/∂t + ∇·(ρu⊗u) = −∇π + ∇·τ
∂(ρT)/∂t + ∇·(ρuT) = ∇·(κ∇T) + Σ_k h_k ω̇_k
∂(ρY_k)/∂t + ∇·(ρuY_k) = ∇·(ρ D_k ∇Y_k) + ω̇_k W_k
∇·u = S(T, Y, ω̇)        (constraint, replaces ∇·u = 0)
π = dynamic pressure perturbation (the projection unknown)
```

**Time integration.** Operator-split per step:

1. Advect (ρ, ρu, ρT, ρY) via Godunov edge-velocity predictor + MAC
   project (same as IAMR).
2. **Stiff chemistry sub-step** — call our `StiffChemistry.solve_cell!`
   per cell to integrate `Y, T → Y^*, T^*` over `[t^n, t^{n+1}]` with the
   frozen advection-diffusion source.
3. Implicit diffusion of (T, Y) — `solve_abec!` per scalar.
4. Recompute the velocity-divergence constraint `S = S(T^{n+1}, Y^{n+1},
   ω̇^{n+1})`.
5. Crank-Nicolson viscous momentum solve (same as IAMR).
6. **Constrained projection** — solve `∇²π = ∇·u^* − S` (i.e. project
   onto `∇·u = S` instead of `∇·u = 0`). The Poisson RHS just gets a
   non-zero term; the multigrid solver is unchanged.

**AMR.** Refinement indicator on `|∇T|` (flame surface) or
`|∇Y_fuel|`. AMReX's PeleLM is the production reference.

**What we'd need to build.**

- Multi-species advection (~200 LoC; extends scalar advection pattern).
- Constrained-divergence projection RHS hookup (~50 LoC; trivial).
- Mini-mechanism: 1-step methane–air (Westbrook & Dryer) or 2-step
  H₂–O₂ (~150 LoC of rate-coefficient tables). A real CHEMKIN parser
  is out of scope; we ship the mechanism as a Julia function.
- Test cases: 1D laminar flame speed, 2D vortex roll-up with heat
  release, optional 2D Rayleigh-Taylor with reactive front (~200 LoC).

**What it demonstrates.**

- The full stack: AMR + projection + variable density + stiff chemistry
  + implicit diffusion, *all together*.
- The chemistry integrator we shipped, finally exercised in coupled
  form rather than as a stand-alone test.
- Genuinely novel territory for the Julia ecosystem — there's no
  production-quality low-Mach reactive-flow code in Julia today (Trixi.jl
  does compressible; nothing does AMR low-Mach reacting).
- Connects to the gravity-solver line of work: stellar convection,
  type-Ia ignition (the original ABCH targets).

**Pros.** Lights up every piece of the AMReX port simultaneously.
Combustion is a real application area with industrial pull. Hardest
demo to dismiss as "yet another lid cavity".

**Cons.** Most moving parts; biggest verification surface. Mechanism
choice is a rabbit hole — even a 9-species H₂–O₂ mechanism is non-trivial
to verify cell-by-cell. Easy to scope-creep into a "real" PeleLM clone.

## Side-by-side

| Aspect | A (IAMR) | B (SDIRK2 + JFNK) | C (ABCH low-Mach) |
|---|---|---|---|
| Time order | 2 (CN) | 2 (SDIRK2), 3 (ESDIRK3) trivial | 2 (split CN) |
| Space order | 2 (Godunov) | 2 (Godunov in adv part) | 2 (Godunov) |
| Stability | CFL-bounded advection; CN viscous | Unconditional | CFL-bounded advection |
| Couples velocity components | No (per-component Helmholtz) | Yes (full tensor + π) | No (per-component) |
| Engineering size | ~500 LoC | ~700–900 LoC | ~700 LoC |
| Hard part | Sync re-fluxing | Block precond (PCD/LSC) | Mechanism + verification |
| Reference codes | IAMR, VARDEN, BoxLib | Persson-Peraire, deal.II | PeleLM, MAESTROeX |
| Demo headline | "Falling drop with AMR" | "Steady cavity at Re=10⁴ in 12 Newton solves" | "Spark-ignited methane-air flame on adaptive mesh" |
| Best fit if priority is | Coverage / textbook reference | Showcasing new JFNK + multigrid-precond stack | Novelty + chemistry integration |

## Recommendation

If we ship one: **Option A (IAMR / ABC)** is the obvious first deliverable
— it gets a complete, recognisable, citation-able incompressible-NS
demo out the door, and the engineering risk is bounded (everything but
sync-reflux is plumbing).

If we ship two: **A then B**. B's SDIRK2 + JFNK block-preconditioner
piece directly tests every implicit primitive we shipped in the AMReX
wave and the compressible JFNK work — it's the natural follow-on that
*proves the stack* on a saddle-point problem.

Option C is the most ambitious and the most novel, but it's also the
one most likely to overshoot scope. It makes the most sense as a
*third* mini-app once A has validated the projection + AMR loop and B
has validated the implicit coupling — at that point ABCH is "compose
A's spatial discretisation with B's implicit treatment, add chemistry".

Open question: do we want A and B sharing a single mini-app
(`cfd_incompressible_ns`) with a `time_scheme = :cn | :sdirk2` knob, or
two separate examples? Sharing reduces duplication but couples the two
verification surfaces; splitting is cleaner but duplicates ~200 LoC of
mesh/IC/BC scaffolding.
