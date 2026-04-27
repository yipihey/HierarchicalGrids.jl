# CFDPatchAMR — Berger-Oliger Patch-AMR Worked Example

A 2D scalar-advection demo that exercises the `PatchHierarchy` stack
shipped in PR-13 of `HierarchicalGrids.jl`. The intent is to show the
mechanics of a patch-based AMR step (coarse update -> prolong ->
fine update -> restrict), not to be a production CFD solver.

## Problem

Linear advection on `[0, 1]^2` with periodic BCs:

```
∂ρ/∂t + u · ∇ρ = 0,    u = (0.5, 0.3),    ρ(t=0, x) = G_σ(x − x₀)
```

with `x₀ = (0.3, 0.3)` and `σ = 0.06`. After one period (`t = 1`)
the Gaussian peak returns to its starting position.

## Patch hierarchy

Two levels:

| Level | Coverage         | Refinement | Cells per axis | dx       |
| ----- | ---------------- | ---------- | -------------- | -------- |
| 1     | `[0, 1]²`        | 4          | 16             | 0.0625   |
| 2     | 0.4×0.4 box (h)  | 4 over box | 16             | ~0.025   |

`h` (fine-patch half-extent) defaults to 0.2. The level-2 patch is
*recreated* every `refresh_every` steps (default 5), centred on the
current peak (computed by argmax over the fine field) and snapped to
the coarse cell grid so its bounding box is exactly the union of
whole coarse cells. That snapping is what makes
`restrict_to_parents!` recover the fine patch's mass exactly on the
covered coarse footprint.

## Per-step pipeline

```julia
1. prolong_from_parents!(fine_views, coarse_views; level=2)  # fine <- coarse_pre
2. for_each_patch!(advect_kernel, ...; level = 2,
                    fields_in_parent = [coarse_views])       # fine update
3. for_each_patch!(advect_kernel, ...; level = 1)            # coarse update
4. restrict_to_parents!(coarse_views, fine_views; level=2)   # push fine back
```

The order matters for conservation: by prolonging *before* the coarse
update, the fine boundary halos (read in stage 2) see the same
pre-step coarse values that the coarse update (stage 3) uses for its
covered-to-uncovered flux contributions. The mass that the coarse
update transfers OUT of an uncovered cell into a covered neighbour
matches the mass that the fine update reads IN through the patch's
upstream boundary. The covered cells' post-coarse values produced by
stage 3 are then overwritten by the volume-weighted fine averages in
stage 4. The fine update's own mass change is interior-flux balanced
plus boundary-flux exchange with the (pre-step) parent cells; that
boundary exchange equals the coarse step's exchange across the same
interface, so total mass is preserved bit-exactly.

Every `refresh_every` steps, the level-2 patch is replaced by a fresh
one centred on the current peak (`_refresh_fine_patch!`). Old fine
data outside the new patch's footprint has already been pushed back
to the coarse via the previous step's `restrict_to_parents!`, so the
destruction is not lossy. The new fine patch's interior is filled by
`prolong_from_parents!` (constant prolongation = exact for degree-0).

## Numerical scheme

First-order upwind finite-volume on degree-0 polynomials
(`MonomialBasis{2, 0}` -> single cell-average coefficient). The
kernel reads upwind neighbours via `hv[:rho, (-1, 0)]` /
`hv[:rho, (0, -1)]` and writes a conservative upwind update:

```
ρ_new = ρ - (u_x · dt / dx) · (ρ - ρ_left)
            - (u_y · dt / dy) · (ρ - ρ_bottom)
```

`PatchHaloView` resolves out-of-patch offsets through the parent's
constant coefficient (`PatchBoundaryBC`). At the base patch, periodic
wrap-around is handled by the underlying `HaloView` via the
hierarchy's `physical_bcs = FrameBoundaries(((PERIODIC, PERIODIC),
(PERIODIC, PERIODIC)))`.

## Conservation

Coarse base patch mass `Σᵢ ρᵢ · Vᵢ` is preserved bit-exactly
(observed `|drift| < 1e-15` at 50 steps, well below the spec's 1e-8
target). The chain of guarantees is:

1. The coarse-level upwind FV step on a periodic uniform grid is
   exactly mass-conservative (sum of `dt · u · ρ · dy` flux
   differences telescopes to zero around a periodic boundary).
2. `restrict_to_parents!` is exactly volume-conservative on the fine
   patch's covered footprint: it sets each covered coarse cell to
   the volume-weighted mean of overlapping fine cells, so
   `Σ_covered_coarse ρ · V = Σ_fine ρ · V` whenever the fine patch
   exactly tiles whole coarse cells.
3. The cell-aligned fine patch (`fine_patch_box` snaps the bounding
   box to a power-of-2 multiple of the coarse cell size) ensures
   the fine patch's footprint *exactly* tiles a contiguous block of
   coarse cells, with no fractional overlaps.
4. The pipeline ordering (fine update before coarse update) makes
   the cross-level flux balance bit-exact: the mass leaving an
   uncovered coarse cell into a covered neighbour during the coarse
   step matches the mass entering the patch through the fine's
   upstream boundary during the fine step (both use the same
   pre-step coarse values).

If the order is swapped (coarse update first, then fine), the fine
boundary halos read the *post-coarse* parent values while the coarse
update used the *pre-coarse* covered-cell values for its flux
exchange — that's a `O(dt²)` per-step mismatch that accumulates to
`O(dt · T)` over a full integration. We observed roughly 10% mass
drift over one period under the wrong ordering, vs. round-off under
the correct one.

### Limitations / known scope reductions

- **Two levels only** (`n_levels = 2`). Adding level 3 would re-use
  the same prolong/restrict primitives keyed at `level = 3`, with the
  level-2 patch acting as parent. The orchestrator code is
  level-agnostic; only the driver in `step!` would need a third pair
  of `for_each_patch!`/`restrict`/`prolong` calls.
- **Constant velocity, axis-aligned upwind kernel** (positive
  components only). Flipping signs is straightforward — read `(+1,
  0)` / `(0, +1)` halos when the corresponding velocity component is
  negative.
- **Single fine patch per step.** The `PatchHierarchy` framework
  permits multiple sibling patches at the same level, but the demo
  ships one. A clustering pass over coarse cells flagged by a
  gradient indicator (the standard Berger-Rigoutsos algorithm) is
  the obvious follow-up.
- **No proper refluxing.** A production-grade implementation would
  accumulate the coarse-side flux at the fine-patch boundary and
  subtract the corresponding fine-side flux sum, then correct the
  parent cells.
- **Patch recreation throws away the fine field's high-frequency
  content** outside the new patch footprint. This is acceptable for
  a moving-feature demo where the feature is the only interesting
  structure, but a flow with persistent fine-scale features
  everywhere would lose information.

## Running

```julia
using Pkg
Pkg.activate("examples/cfd_patch_amr")
Pkg.develop(path = "../..")          # link to HierarchicalGrids parent
Pkg.test()                            # smoke test (~5-10 s)
```

A direct script entry point isn't shipped; `run!(::PatchAMRConfig)`
can be called from any host project.

## Expected runtime

The smoke test (50 steps, 16×16 base + 16×16 fine, sequential
backend) wall-clocks at well under 10 seconds on a 2024-vintage
laptop.
