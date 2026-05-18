# AMReX → HierarchicalGrids.jl porting roadmap

This roadmap tracks the AMReX-compatible elliptic-solver capabilities
being ported to HierarchicalGrids.jl. Strategic survey behind it is in
the development-session log.

## Shipped (Tier 1 + most of Tier 2 + Tier 3 #7 #8)

| # | Item | AMReX equivalent | File |
|---|------|------------------|------|
| 1 | Variable-coefficient ABec operator | `MLABecLaplacian` | `src/Solver/ABecLaplacian.jl` |
| 2 | MAC (face) velocity projection | `MacProjector` (single level) | `src/Solver/MACProjection.jl` |
| 3 | Krylov.jl flat-vector bridge | CG/GMRES/BiCGStab/FGMRES/MINRES | `src/Solver/KrylovBridge.jl` |
| 4 (single-level) | Node-centered variable-σ Poisson | `MLNodeLaplacian` (single-level) | `src/Solver/NodeLaplacian.jl` |
| 5 (subset) | Multi-component decoupled diffusion | `MLTensorOp` (scalar-per-component, isotropic) | `src/Solver/VectorABec.jl` |
| 6 (CPU) | AlgebraicMultigrid bottom solver | HYPRE BoomerAMG (CPU, non-MPI) | `src/Solver/AMGBottom.jl` |
| 7 | Gray + multigroup radiation diffusion (linear) | Castro MGFLD inner ABec | `src/Solver/RadiationDiffusion.jl` |
| 8 | Stiff chemistry per-cell integrator | PelePhysics CVODE reactor (pure-Julia BDF1) | `src/Solver/StiffChemistry.jl` |

End-to-end capability now:

* **Scalar elliptic** — `∇²φ = ρ`, variable-coef Helmholtz, MAC and nodal
  pressure projection (single level), implicit thermal / species
  diffusion, gray + multigroup radiation backward-Euler step. All drive
  via `solve_abec!`, `solve_node_laplacian!`, `mac_project!`, or the
  Krylov bridge with `:cg | :gmres | :bicgstab | :fgmres | :minres` and
  optional Jacobi or AMG preconditioning.
* **Multi-component** — `solve_vector_abec!` for K decoupled scalar
  systems; sufficient for incompressible-NS isotropic viscosity. Full
  tensor coupling is still TODO (#5 full).
* **Stiff source terms** — per-cell `step_reaction!` with damped Newton
  / analytical-or-FD Jacobian, parallelised over leaves. Suitable for
  O(10)-species networks; large mechanisms should wrap CVODE externally.

## Remaining

### Tier 2 continuation

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 4 (multi-level) | `MLNodeLaplacian` with FAC at AMR C/F boundaries (Martin-Colella-Almgren) | 800–1200 LoC | Needed for AMR nodal pressure projection (IAMR / incflo / MAESTROeX). Storage primitives are in place; main work is the C/F nodal restrictor/prolongator and a proper composite operator. |
| 5 (full) | `MLTensorOp` with cross-component coupling μ(∇u + ∇uᵀ) + λ ∇·u · I | 600–900 LoC | Builds on `VectorABecProblem`; adds a joint smoother that updates all D components with cross terms. |
| 6+ | HYPRE.jl (MPI BoomerAMG) and AMGX.jl (CUDA BoomerAMG) | 300 LoC each | External deps. Use the in-tree `AMGBottom` for the pure-Julia / single-node CPU case (already shipped). |
| -- | Multi-level FAC matrix assembly (for AMG-on-composite) | 200–400 LoC | Adds C/F coupling rows to `assemble_abec_matrix`. Enables AMG to be used as a composite bottom solver. |

### Tier 3 continuation

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 7 (nonlinear) | Newton-Krylov coupling of κ(T), B(T) for fully nonlinear radiation | 200 LoC + NonlinearSolve.jl | Linear inner solve already shipped; outer Newton is mechanical. |
| 9 | EB / cut-cell support across all operators | 5000+ LoC, multi-month | Geometry generation, EB stencils, EB-aware C/F, EB-aware boundary conditions. Multi-month — should be its own design phase. |
| 10 | `MLCurlCurl` edge-centered ∇×(μ⁻¹∇×) + σ identity | 2000 LoC | Requires edge-centered field storage. WarpX magnetostatic, resistive MHD. |

### Architecture / infrastructure

| Item | Effort | Notes |
|------|--------|-------|
| Composite (multi-level) node Laplacian via FAC | 600 LoC | Mirrors the existing multi-level FAC for cell-centered ABec. |
| Edge-centered field type (`EdgeField{D,T}`) | 300 LoC | Foundation for `MLCurlCurl` and tensor-coupling. |
| GPU dispatch via KernelAbstractions | 1000+ LoC | Lift kernels to KA; the Krylov bridge already supports CUDA/AMD via Krylov.jl + KA. |

## Architectural notes for future sessions

* The flat-vector layout (`FlatLayout`, `pack!`, `unpack!`) generalises
  to any operator on uncovered cells. Tier-2 #4 multi-level / #5 full
  should reuse it for AMG preconditioning.

* `apply_*`, `gs_sweep_*`, `compute_*_residual!` is the three-function
  contract every new operator should follow — that's what
  `pcg_composite_abec_solve!`, `solve_with_krylov!`, and the AMG
  preconditioner all consume.

* Sign convention: `L_ABec φ = A·α·φ - B·∇·(β ∇φ)` matches AMReX. For
  PCG to converge directly, use SPD-positive forms (`A ≥ 0`, `α ≥ 0`,
  `B ≥ 0`, `β ≥ 0`). Pure Poisson `∇²φ = ρ` is `A=0, B=1, β=1, f=-ρ`
  in this convention. The original `solve_poisson!` keeps the
  `∇²φ = ρ` form for backward compatibility.

* `NodeLaplacian` uses node-centered storage `(N+1, ..., N+1)`. The
  natural convention identifies node `1` with node `N+1` along
  periodic axes; the apply / GS kernels do this internally.

* `RadiationDiffusion` and `MAC projection` are thin wrappers on the
  ABec operator. Treat new "physics" modules the same way unless they
  introduce a new field-storage type.
