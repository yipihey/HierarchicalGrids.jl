# Anisotropic Refinement

Most AMR codes refine cells isotropically: a 3D cell becomes 8 children, each half the size along all axes. This is fine for problems with spherical symmetry or no preferred direction. But many problems have strong directional structure where isotropic refinement wastes resolution.

HierarchicalGrids supports anisotropic refinement as a first-class operation: a cell can split along any non-empty subset of its axes.

## When you'd want it

**Pancake collapses (cosmology)**: in the Zel'dovich approximation and its nonlinear evolution, structure forms first as 2D sheets, then filaments, then halos. A pancake collapsing along the z axis has fine structure in z but is smooth in x and y. Refining only along z gets you the resolution you need with a quarter the cells of full isotropic refinement.

**Boundary layers (fluid mechanics)**: near a wall, gradients are normal to the wall. Refining only normal to the wall captures the boundary layer without wasting resolution along the wall.

**Disk geometries (accretion, planetary atmospheres)**: thin disks have fine structure in the disk normal direction but smooth structure in the disk plane. Refining only in the normal direction is the obvious win.

**Shocks and contact discontinuities**: refining only in the direction of the gradient catches the discontinuity without overresolving the smooth flow on either side.

## Split masks

A split mask is an unsigned integer where bit `i` indicates "split along axis i". For 3D:

| Mask    | Binary | Splits axes      | Children |
|---------|--------|------------------|----------|
| 1       | 0b001  | x                | 2        |
| 2       | 0b010  | y                | 2        |
| 3       | 0b011  | x, y             | 4        |
| 4       | 0b100  | z                | 2        |
| 5       | 0b101  | x, z             | 4        |
| 6       | 0b110  | y, z             | 4        |
| 7       | 0b111  | x, y, z (full)   | 8        |

The convenience constant `FULLY_ISOTROPIC_MASK(Val(D))` gives you the all-ones mask for dimension `D`.

## Using it

Pass a per-cell split mask to `refine_cells!`:

```julia
mesh = HierarchicalMesh{3}()

# Refine only along the z axis
refine_cells!(mesh, [1], [0b100])

n_cells(mesh)              # → 3 (root + 2 children)

# The children are stacked along z
for ci in find_children(mesh, 1)
    pos = position_in_parent(mesh.cells[ci])
    @show pos    # (0, 0, 0) and (0, 0, 1) — only z varies
end
```

Mixed batches: each cell in the batch can have its own mask.

```julia
# Refine cell 1 along x, cell 5 along y
refine_cells!(mesh, [1, 5], [0b001, 0b010])
```

## How it interacts with the rest of the framework

**Children inherit the parent's split mask** (encoded in their own `split_mask` field). This is the framework's record of "how was this cell created". It matters because the per-axis level of a child is determined by which axes its ancestors split along.

**Per-axis level** computes how many refinement steps along each axis there are between this cell and the canonical reference:

```julia
# Internal helper, but illustrative
levels = HierarchicalGrids.Geometry.per_axis_level_of(mesh, cell_idx)
# levels[1] is the x level, levels[2] is y, levels[3] is z
```

For a cell that's been refined twice along z but never along x or y, `levels = (0, 0, 2)`.

**Volume** is correct automatically: a cell with `levels = (0, 0, 2)` has volume `1/(2^0 * 2^0 * 2^2) = 1/4`. The exact-rational volume computation handles arbitrary anisotropic refinement without special cases.

**Position in parent** uses PDEP to scatter the sibling index across the split-axis bits:

```julia
# Cell at sibling_index 1 with split_mask 0b101 has position
# (1, 0, 0)  — bit 0 of sibling_index goes to axis 0
# Cell at sibling_index 2 with split_mask 0b101 has position
# (0, 0, 1)  — bit 1 of sibling_index goes to axis 2 (the next set bit)
```

The position along non-split axes is always 0 (the cell occupies the full parent extent in those directions).

## Performance considerations

The framework detects fully-isotropic refinement (the common case) and takes a fast path where possible. For mixed isotropic + anisotropic meshes, the framework handles both consistently; you don't need to opt in or out.

The PDEP/PEXT operations used for position computation are 3-cycle operations on x86 hardware with BMI2. Without BMI2, the software fallback runs at maybe 20 cycles — still fast enough that it's not a bottleneck.

## Limitations

**Once you pick a split mask for a cell, all children are stuck with it.** A cell that was split only along z cannot have one of its z-children further split along x — well, it can, but the result is a different mask, and the framework treats each cell independently. There's no notion of "this z-axis split should be inherited by descendants forever". If you want to refine differently in different parts of the tree, just refine differently.

**Fully-isotropic refinement is still the most common case.** Anisotropic is a tool for specific problems; don't use it for every cell. Its main benefit is when there's a clear physical reason for one direction being different.

**Refinement masks are per-cell, not per-region.** If you want a region of the mesh refined anisotropically, you need to specify the mask for each cell you refine. There's no broad-stroke "refine this region with this mask" operator (yet); you're expected to compute which cells need refining and what their masks should be, then call `refine_cells!`.

## Example: a pancake-collapse refinement criterion

Here's a sketch of how you'd use anisotropic refinement in a real solver:

```julia
function refine_for_pancake(mesh, fields)
    cells_to_refine = Int[]
    masks = UInt8[]

    for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue

        # Compute density gradient
        grad = compute_gradient(mesh, fields, i)

        # If the gradient is large in a specific direction, refine that direction
        threshold = 0.1
        mask = UInt8(0)
        if abs(grad[1]) > threshold; mask |= 0b001; end
        if abs(grad[2]) > threshold; mask |= 0b010; end
        if abs(grad[3]) > threshold; mask |= 0b100; end

        if mask != 0
            push!(cells_to_refine, i)
            push!(masks, mask)
        end
    end

    if !isempty(cells_to_refine)
        refine_cells!(mesh, cells_to_refine, masks)
    end
end
```

In a pancake collapse, this would naturally refine only along the collapse axis where the gradient is large, giving you tens of times more effective resolution per byte than full isotropic refinement.
