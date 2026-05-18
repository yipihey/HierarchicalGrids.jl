# AMReX → HierarchicalGrids.jl porting roadmap

This roadmap tracks the AMReX-compatible elliptic-solver capabilities
being ported to HierarchicalGrids.jl. Strategic survey behind it is in
the development-session log.

## Shipped

| # | Item | AMReX equivalent | File |
|---|------|------------------|------|
| 1 | Variable-coefficient ABec operator | `MLABecLaplacian` | `src/Solver/ABecLaplacian.jl` |
| 2 | MAC (face) velocity projection | `MacProjector` (single level) | `src/Solver/MACProjection.jl` |
| 3 | Krylov.jl flat-vector bridge | CG/GMRES/BiCGStab/FGMRES/MINRES | `src/Solver/KrylovBridge.jl` |
| 4 (single-level) | Node-centered variable-σ Poisson | `MLNodeLaplacian` (single-level) | `src/Solver/NodeLaplacian.jl` |
| 4 (nested-MG) | Multi-level V-cycle on nested grids | `MLNodeLaplacian` (nested coarsening) | `src/Solver/NodeLaplacianML.jl` |
| 5 (subset) | Multi-component decoupled diffusion | `MLTensorOp` scalar-per-component | `src/Solver/VectorABec.jl` |
| 5 (full) | Tensor viscosity with cross-component coupling | `MLTensorOp` full | `src/Solver/TensorOp.jl` |
| 6 (CPU) | AlgebraicMultigrid bottom solver | HYPRE BoomerAMG (CPU, non-MPI) | `src/Solver/AMGBottom.jl` |
| 6 (MPI) | HYPRE BoomerAMG bottom solver | HYPRE BoomerAMG (MPI, via HYPRE.jl) | `src/Solver/HYPREBottom.jl` |
| 7 | Gray + multigroup radiation diffusion (linear) | Castro MGFLD inner ABec | `src/Solver/RadiationDiffusion.jl` |
| 8 | Stiff chemistry per-cell integrator | PelePhysics CVODE reactor (pure-Julia BDF1) | `src/Solver/StiffChemistry.jl` |
| 10 (foundation) | Edge-centered vector fields + component-wise Laplacian | `MLCurlCurl` storage primitives | `src/Solver/EdgeFields.jl` |
| 10 (2D op) | 2D curl-curl operator with σ identity | `MLCurlCurl` (2D, single-level) | `src/Solver/CurlCurl.jl` |

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
| 4 (full FAC) | `MLNodeLaplacian` with FAC at true AMR C/F boundaries (Martin-Colella-Almgren) | 600–1000 LoC | Nested-grid multi-level V-cycle is shipped; the remaining work is the proper C/F coupling when fine patches cover only a *sub-region* of coarse. |
| 6+ | AMGX.jl (CUDA BoomerAMG) | 300 LoC | NVIDIA-only. HYPRE.jl is already shipped (CPU + MPI). |
| -- | Multi-level FAC matrix assembly (for AMG-on-composite) | 200–400 LoC | Adds C/F coupling rows to `assemble_abec_matrix`. Enables AMG to be used as a composite bottom solver. |

### Tier 3 continuation

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 7 (nonlinear) | Newton-Krylov coupling of κ(T), B(T) for fully nonlinear radiation | 200 LoC + NonlinearSolve.jl | Linear inner solve already shipped; outer Newton is mechanical. |
| 9 | EB / cut-cell support across all operators | 5000+ LoC, multi-month | Geometry generation, EB stencils, EB-aware C/F, EB-aware boundary conditions. Multi-month — should be its own design phase. |
| 10 (3D + smoother) | 3D MLCurlCurl + Hiptmair smoother | 800–1200 LoC | 2D shipped (`CurlCurl.jl`); 3D adds full edge-edge cross-coupling and a tailored smoother. |

### Architecture / infrastructure

| Item | Effort | Notes |
|------|--------|-------|
| Composite (multi-level) node Laplacian via FAC | 600 LoC | Mirrors the existing multi-level FAC for cell-centered ABec. |
| Edge-centered field type (`EdgeField{D,T}`) | -- (shipped) | Foundation for `MLCurlCurl` and tensor-coupling. |
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
