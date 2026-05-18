# AMReX → HierarchicalGrids.jl porting roadmap

This roadmap tracks the AMReX-compatible elliptic-solver capabilities being
ported to HierarchicalGrids.jl. The strategic survey behind it is in commit
`acc110c` of the development log.

## Shipped (Tier 1 + start of Tier 2)

| # | Item | AMReX equivalent | Files |
|---|------|------------------|-------|
| 1 | Variable-coefficient ABec operator | `MLABecLaplacian` | `src/Solver/ABecLaplacian.jl` |
| 2 | MAC (face-centered) velocity projection | `MacProjector` | `src/Solver/MACProjection.jl` |
| 3 | Krylov.jl flat-vector bridge | (CG/GMRES/BiCGStab/FGMRES/MINRES) | `src/Solver/KrylovBridge.jl` |
| 6 | AlgebraicMultigrid bottom solver | HYPRE BoomerAMG (CPU-only) | `src/Solver/AMGBottom.jl` |

What you can now solve, end-to-end on the patch hierarchy:

* `∇²φ = ρ` and `(I - Δt·D ∇²)φ = rhs` (heat) via either `solve_poisson!`
  or the variable-coefficient `solve_abec!`.
* `∇·(β ∇φ) = ∇·u*` and the matching MAC correction `u^{n+1} = u* - β ∇φ`,
  single level.
* Any of the above through `solve_with_krylov!` with `method =
  :cg|:gmres|:bicgstab|:fgmres|:minres` and an optional `precond`
  callback (Jacobi, AMG, …).
* Single-level AMG-preconditioned solves: build the sparse matrix once
  with `assemble_abec_matrix`, hand it to `amg_preconditioner`, pass the
  result to `solve_with_krylov!`. ~7 iters to converge on variable-β
  Helmholtz vs ~50 with Jacobi.

## Remaining (Tier 2 + Tier 3)

Effort estimates are rough — each line is a focused future session.

### Tier 2 (medium lift, opens application classes)

| # | Item | Effort | Unlocks |
|---|------|--------|---------|
| 4 | `MLNodeLaplacian` equivalent — node-centered variable-coefficient Poisson with C/F at AMR boundaries (Martin-Colella-Almgren) | 1500–2000 LoC | Nodal pressure projection (IAMR / incflo / MAESTROeX / PeleLMeX); a real incompressible driver. |
| 5 | `MLTensorOp` equivalent — multi-component cell-centered tensor viscosity with face-averaged μ | 800–1200 LoC | Implicit-viscosity Navier-Stokes; rounds out the IAMR/PeleLMeX trio. |
| 6+ | HYPRE.jl (MPI BoomerAMG) and AMGX.jl (CUDA BoomerAMG) bottom plug-ins | 300 LoC each | Scaling beyond geometric-coarsening floor and GPU. Use the AlgebraicMultigrid-bottom (already shipped) for the CPU-non-MPI case. |

### Tier 3 (specialty, application-driven)

| # | Item | Effort | Unlocks |
|---|------|--------|---------|
| 7 | Gray + multigroup radiation diffusion | 600 LoC on top of Tier 1 | Castro MGFLD, Quokka rad-hydro. Gray = ABec + Newton on κ(T); multigroup = block ABec per group. |
| 8 | Stiff chemistry per-cell integrator | 400 LoC | PelePhysics-style reactor. Wrap CVODE via Sundials.jl with `OhMyThreads` cell-batching (Sundials is a soft dep). |
| 9 | EB / cut-cell support across all operators | 5000+ LoC, multi-month | MFIX-Exa, incflo-EB, WarpX-EB. Geometry generation, EB stencils, EB-aware C/F. |
| 10 | `MLCurlCurl` equivalent — edge-centered ∇×μ⁻¹∇× | 2000 LoC | WarpX magnetostatic; resistive MHD. |

## Architectural notes carried into Tier 2

* The flat-vector layout (`FlatLayout`, `pack!`, `unpack!`) used by the
  Krylov bridge generalizes to any operator that acts on uncovered cells
  across `level_range`. Tier-2 #4/#5 should expose the same flat view
  for their own multi-component / node-centered storage.

* AlgebraicMultigrid.jl's `aspreconditioner` only takes a `SparseMatrixCSC`.
  Multi-level FAC matrix assembly (covering C/F entries) is the natural
  next step — needed for AMG to drive composite AMR solves rather than
  just single-level bottoms.

* `pcg_composite_abec_solve!` and `solve_with_krylov!` both consume the
  same operator + scratch interface. New operators (TensorOp, NodeLaplacian)
  should follow the same `apply!`, `gs_sweep!`, `compute_residual!`
  three-function contract.

* Sign convention: `L_ABec φ = A·α·φ - B·∇·(β ∇φ)` matches AMReX. For
  SPD-positive operators (so PCG works directly), use `A ≥ 0` with `α ≥ 0`
  and `B ≥ 0` with `β ≥ 0`. For pure Poisson `∇²φ = ρ`, set
  `A=0, B=1, β=1` and supply `-ρ` as RHS (the negative of the standard
  convention). The existing const-coef `solve_poisson!` keeps the
  `∇²φ = ρ` sign convention for backward compatibility.
