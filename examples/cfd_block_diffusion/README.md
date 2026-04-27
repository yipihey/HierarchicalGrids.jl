# Block-based scalar diffusion (PR-15)

A small worked example demonstrating the **block-based AMR stack** in
HierarchicalGrids.jl using **Path B** (point-sample blocks). It solves
the 2-D heat equation

    ∂u/∂t = D ∇²u

on `[0, 1]²` with periodic BCs, starting from a centred Gaussian.

## What this demonstrates

- **`PointSampleFieldSet{2, N, Float64}`** — each leaf cell carries an
  N×N grid of point samples on the equispaced Lagrange node set, where
  node `(i, j)` sits at reference coordinate `((i-1)/(N-1), (j-1)/(N-1))`
  in the cell's unit cube.
- **`for_each_block!`** — drives the time-stepping loop over leaf
  blocks in parallel, with a single per-block kernel that computes a
  centred-difference Laplacian and an RK2 update.
- **`BlockHaloView`** — gives the kernel cross-block stencil reads via
  `bhv[Val(:u), off, (i, j)]`. Periodic boundary wrapping is handled
  transparently by the orchestrator.

## Layout: 8×8 point samples per block

This example uses `N = 8`, so each leaf cell carries 8×8 = 64 point
values. With `n_levels = 2` (the default) the mesh has 4×4 = 16 leaf
cells, giving an effective 32×32 grid (with shared boundary nodes).

## In-block vs cross-block stencil

The Laplacian at point `(i, j)` of a block needs the four
neighbours `(i±1, j)` and `(i, j±1)` at one-grid-spacing offsets. The
kernel branches per-point:

  * **Interior point** (`2 ≤ i, j ≤ N-1`): all four neighbours sit in
    the same block. Use `bv[Val(:u), (i±1, j)]`.
  * **Block-boundary point** (`i = 1`, `i = N`, `j = 1`, or `j = N`):
    one of the neighbours is in a different block. Use
    `bhv[Val(:u), (off_x, off_y), (i', j')]`.

A subtle but critical detail: nodes `i = 1` and `i = N` of adjacent
blocks SHARE the same physical cell-face point (both blocks store
`u(x_face, y)` at that node). Therefore the left-of-`(1, j)` neighbour
in physical space is NOT the left block's `(N, j)` (which is the same
shared edge point), but the left block's `(N-1, j)` — one grid spacing
further to the left, skipping the duplicated edge. Analogously,
right-of-`(N, j)` is the right block's `(2, j)`. The kernel codes this
explicitly:

```julia
if i == 1
    u_left = bhv[Val(:u), (-1, 0), (N - 1, j)]   # skip shared edge
elseif i == N
    u_right = bhv[Val(:u), (+1, 0), (2, j)]      # skip shared edge
else
    u_left  = bv[Val(:u), (i - 1, j)]
    u_right = bv[Val(:u), (i + 1, j)]
end
```

## Time-stepping

Standard explicit RK2 (Heun's method, two stages):

    u^*    = u^n   + (dt/2) RHS(u^n)
    u^{n+1} = u^n  + dt    RHS(u^*)

implemented as two `for_each_block!` passes per step. The kernel receives
the RK2 base state (`u^n`) through `ctx.base_in_views` so it can
compute `base + α dt RHS(read)` in one fused pass per stage.

The CFL limit for explicit centred-difference 2-D diffusion is
`dt < dx² / (4 D)` where `dx = (cell extent) / (N - 1)`.

## AMR caveat

`AdaptiveField` (the refinement-event listener that auto-resizes a
field on mesh changes) currently supports only `PolynomialFieldSet`
(Path A). For `PointSampleFieldSet` (Path B) the example refines the
mesh BEFORE allocating fields and skips the AMR loop. Adding adaptive
support for Path B is a follow-up PR.

## Diagnostics

`run!` returns a `DiagnosticsResult` with:

  * `mass_drift` — ∫ u dV change over the run. Should be ≈ 0 to
    round-off (the trapezoidal rule with periodic BCs is mass-exact
    for the discrete diffusion stencil).
  * `peak_decay_error` — relative error between the measured maximum
    and the analytic 2-D Gaussian-on-the-plane prediction
    `σ² / (σ² + 2 D t)`. Stays below ~1 % for the validation problem.

## Running

```julia
cd examples/cfd_block_diffusion
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Expected runtime

The default validation problem (16 blocks × 8×8, 50 RK2 steps) runs in
≤ 2 s on a recent laptop core. The smoke test wired into the main
`test/runtests.jl` (1 level / 4 blocks, 25 steps) runs in well under
30 s.

## Files

  * `src/CFDBlockDiffusion.jl` — module: config, kernel, `run!` driver, diagnostics.
  * `test/runtests.jl`         — validation tests (mass conservation, peak decay).
  * `Project.toml`             — uses `Pkg.develop("../..")` for HG.
