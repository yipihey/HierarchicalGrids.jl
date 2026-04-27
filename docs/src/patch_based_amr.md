# Patch-based AMR

`PatchHierarchy{D, T}` is a Berger-Oliger style patch hierarchy: a
level-indexed list of `EulerianFrame`s where each fine patch lies inside
a single coarse parent. The library ships four primitives that compose
into a time step:

- `for_each_patch!` — run a user kernel over every leaf of every patch
  at one level, with parent-aware halo resolution.
- `prolong_from_parents!` — fill a level's fields by constant-projecting
  from its parent (degree-0 path).
- `restrict_to_parents!` — push volume-weighted fine averages back onto
  the covered parent cells (conservative degree-0 restriction).
- `step_patch_pipeline!` — orchestrator that wires the four primitives
  together in the correct order. Recommended entry point.

See [the worked example](https://github.com/yipihey/HierarchicalGrids.jl/tree/main/examples/cfd_patch_amr)
for an end-to-end 2D scalar-advection demo.

## Time-step ordering

The order in which the four sub-steps run matters for conservation.
The recommended sequence is:

1. **Prolong** — for every fine level, fill the input buffer's
   ghost-zone cells from the parent's pre-step input.
2. **Update fine** — advance every fine level (finest first), reading
   parent halos through `fields_in_parent` (= the pre-step parent
   input).
3. **Update coarse** — advance the coarse base on the full domain.
   Covered coarse cells are updated too; their values will be
   overwritten in step 4.
4. **Restrict** — push volume-weighted fine averages back onto the
   covered parent cells (finest first).

### Why this ordering, not the textbook one

A naive Berger-Oliger description suggests "coarse first, then fine".
That ordering leaks mass: the fine patch's boundary halos read the
*post-coarse* parent values, while the coarse update used the
*pre-coarse* covered-cell values for its flux exchange across the
covered/uncovered interface. The two flux contributions across the
same physical face no longer match, producing an `O(dt)` per-step
mismatch that accumulates to `O(dt · T)` over a full integration.

Reversing the order — prolong, fine update, coarse update, restrict —
makes the cross-level flux balance bit-exact. The mass leaving an
uncovered coarse cell into a covered neighbour during the coarse step
uses the same pre-step covered value that the fine update read at its
upstream boundary, so the two contributions across the patch-edge
interface cancel exactly. With a fine patch that exactly tiles whole
coarse cells (so the conservative restrict is exact) and a
mass-conservative kernel, the integrated mass on the base patch is
preserved to round-off. The worked example observes `|drift| < 1e-15`
over one period under this ordering versus roughly 10% drift under
the textbook one.

## Recommended pattern

`step_patch_pipeline!` bakes the ordering in:

```julia
using HierarchicalGrids

ph = PatchHierarchy(base_frame; physical_bcs = bcs)
add_patches!(ph, 2, [fine_frame])

# fields_in[ℓ] / fields_out[ℓ] are Vector{NamedTuple} over patches at level ℓ.
fields_in  = [[base_in_views],  [fine_in_views]]
fields_out = [[base_out_views], [fine_out_views]]

step_patch_pipeline!(my_kernel, fields_out, fields_in, ph;
                      ctx = my_ctx, backend = Sequential())
```

The same `kernel(pv, hv, ctx)` runs at every level. The orchestrator
hands each cell its `PatchView` (which carries `pv.level`), so the
kernel can dispatch on level if the numerics differ; for a uniform
first-order scheme the kernel is level-agnostic.

The helper does not swap input/output buffers — callers that
double-buffer should swap externally between time steps.

## Manual recipe

For users who want fine-grained control (e.g., per-level sub-cycling,
per-level kernels, or refluxing accumulation), call the four
primitives directly in the same order:

```julia
# Step 1: prolong (top-down).
prolong_from_parents!(fields_in[2], fields_in[1], ph;
                       level = 2, fieldname = :rho)

# Step 2: update fine levels, finest first.
for_each_patch!(my_kernel, fields_out[2], fields_in[2], ph;
                 level = 2, ghost_depth = 1,
                 fields_in_parent = fields_in[1],
                 ctx = my_ctx, backend = Sequential())

# Step 3: update the coarse base.
for_each_patch!(my_kernel, fields_out[1], fields_in[1], ph;
                 level = 1, ghost_depth = 1,
                 ctx = my_ctx, backend = Sequential())

# Step 4: restrict fine outputs onto parent outputs (finest first).
restrict_to_parents!(fields_out[1], fields_out[2], ph;
                      level = 2, fieldname = :rho)
```

This is the exact sequence `step_patch_pipeline!` performs internally
for a two-level hierarchy. For more than two levels the helper repeats
steps 1, 2, and 4 across all `2..n_levels(ph)` (top-down for prolong,
finest-first for fine updates and restricts).

## See also

- `for_each_patch!` — per-level orchestrator with parent-aware halos.
- `restrict_to_parents!` / `prolong_from_parents!` — the conservative
  degree-0 transfer operators.
- `examples/cfd_patch_amr/` — a self-contained 2D advection demo with
  conservation diagnostics and a hand-written version of the same
  pipeline.
