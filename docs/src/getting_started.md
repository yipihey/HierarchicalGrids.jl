# Getting Started

This tutorial walks through the basics of HierarchicalGrids.jl: building a mesh, refining it, attaching field storage, and using the framework's parallelism. By the end you'll have a clear sense of how the layers compose.

## Installation

```julia
] add https://github.com/yipihey/HierarchicalGrids.jl
```

Or develop locally:

```julia
] dev /path/to/HierarchicalGrids.jl
```

## A first mesh

Construct a 3D mesh containing just the root cell:

```julia
using HierarchicalGrids

mesh = HierarchicalMesh{3}()
n_cells(mesh)            # → 1
is_leaf(mesh.cells[1])   # → true
```

The `{3}` parameter is the dimension. The framework supports arbitrary dimension; you can equally write `HierarchicalMesh{1}()`, `HierarchicalMesh{2}()`, or `HierarchicalMesh{4}()`. The 1D case is fully supported end-to-end: `compute_overlap`, polynomial remap, neighbor graph, halos, and AMR all work for `D = 1`.

## Refinement

Refine the root cell into 8 children (in 3D, fully isotropic refinement gives 2^3 children):

```julia
refine_cells!(mesh, [1])
n_cells(mesh)            # → 9 (root + 8 children)
is_leaf(mesh.cells[1])   # → false (root is no longer a leaf)
is_leaf(mesh.cells[2])   # → true (children are leaves)
```

Refine multiple cells at once. Refinement is a batch operation — all the work happens together with a single rebuild of the cells array:

```julia
# Refine the first child and the fourth child of root
# Note: indices may shift after refinement; query find_children to be safe
children_of_root = find_children(mesh, 1)
refine_cells!(mesh, [Int(children_of_root[1]), Int(children_of_root[4])])
n_cells(mesh)   # → 25 (1 + 8 + 8 + 8 - 2 since two refined cells become parents)
```

## Anisotropic refinement

By default, refinement is fully isotropic (split along all axes). You can specify a per-cell split mask to refine along a subset:

```julia
mesh = HierarchicalMesh{3}()

# Split only along the x axis
refine_cells!(mesh, [1], [0b001])
n_cells(mesh)   # → 3 (root + 2 children)

# Split a child along the z axis
refine_cells!(mesh, [2], [0b100])
n_cells(mesh)   # → 5
```

The split mask uses bit `i` to indicate "split along axis i". So `0b001` is x-only, `0b010` is y-only, `0b100` is z-only, `0b011` is x and y, etc.

This is useful for problems with strong directional structure: pancake collapses, accretion disks, atmospheric layers — anywhere isotropic refinement wastes resolution.

## Tree navigation

```julia
mesh = HierarchicalMesh{3}()
refine_cells!(mesh, [1])
refine_cells!(mesh, [2])

# Parent of any cell
find_parent(mesh, 3)    # → 2

# Children of a cell (returns indices in the cells array)
find_children(mesh, 1)  # → [2, 11, ...] (the 8 children of root)

# Lowest common ancestor of two cells
find_lca(mesh, 3, 11)   # → 1 (both are descendants of root)

# Scalar level (depth from canonical reference)
level_of(mesh, 1)       # → 0
level_of(mesh, 3)       # → 2
```

These functions use the lazy caches under the hood. The first call after a structural change (refine/coarsen) triggers a single O(N) cache rebuild; subsequent calls are O(1) lookups.

## Field storage

Fields are stored separately from the mesh — they're parallel arrays indexed by cell index. The Storage layer gives you the layout flexibility:

```julia
mesh = HierarchicalMesh{3}()
refine_cells!(mesh, [1])

# SoA: each field is a separate Vector
fields = allocate_fields(SoA(), n_cells(mesh);
                        density = Float32,
                        momentum = NTuple{3, Float32},
                        energy = Float32)

# Initialize
for i in 1:n_cells(mesh)
    fields.density[i] = 1.0f0
    fields.momentum[i] = (0.0f0, 0.0f0, 0.0f0)
    fields.energy[i] = 1.5f0
end
```

Switching to AoS is a one-line change. Kernel code is unchanged:

```julia
fields_aos = allocate_fields(AoS(), n_cells(mesh);
                            density = Float32,
                            momentum = NTuple{3, Float32},
                            energy = Float32)

for i in 1:n_cells(mesh)
    fields_aos.density[i] = 1.0f0       # same access syntax
    fields_aos.momentum[i] = (0.0f0, 0.0f0, 0.0f0)
    fields_aos.energy[i] = 1.5f0
end
```

Or block-structured for spatial-locality kernels:

```julia
fields_blk = allocate_fields(Blocked{8, SoA}(), n_cells(mesh);
                             density = Float32, ...)
```

When the mesh is refined, you resize the field storage. Existing data is preserved at unchanged indices; new cells are uninitialized:

```julia
old_n = n_cells(mesh)
refine_cells!(mesh, [3])
new_n = n_cells(mesh)
resize_fields!(fields, new_n)

# Initialize new cells (you'll typically interpolate from parents)
for i in (old_n + 1):new_n
    fields.density[i] = 1.0f0  # or interpolate from parent
end
```

(In a real solver you'd plug in a refinement-aware initialization that interpolates from parent values; the framework gives you the indices, you supply the policy. For batched updates, register a listener once and the framework will dispatch a `RefinementEvent` after every refine/coarsen — see [Refinement events](refinement_events.md).)

## Initializing a polynomial field from a function

For polynomial-coefficient field storage, `init_field_from!` runs a
per-cell L² projection of an analytical function onto the field's
basis. Two methods cover both Eulerian and Lagrangian setups:

```julia
using HierarchicalGrids

# Eulerian: project onto every cell of an EulerianFrame
mesh  = HierarchicalMesh{2}()
refine_cells!(mesh, [1])
frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))

field = allocate_polynomial_fields(SoA(), BernsteinBasis{2, 2}(),
                                    n_cells(mesh); rho = Float64)
init_field_from!(field, frame, x -> 1.0 + 0.5 * sinpi(2 * x[1]))

# Lagrangian: project onto every simplex of a SimplicialMesh
lag = SimplicialMesh{2, Float64}(...)
lag_field = allocate_polynomial_fields(SoA(), BernsteinBasis{2, 1}(),
                                        n_simplices(lag); rho = Float64)
init_field_from!(lag_field, lag, x -> 1.0 + 0.1 * x[2])
```

The default quadrature order (`2P + 1`, exact for two degree-`P` basis
functions) is sufficient for polynomial inputs; pass a larger
`quadrature_order` keyword for trigonometric or otherwise nonsmooth
functions.

## Volumes and conservation

Cell volumes are exact rationals. Sum of leaf volumes equals 1 (the root volume) regardless of refinement pattern:

```julia
using HierarchicalGrids.Geometry

mesh = HierarchicalMesh{3}()
refine_cells!(mesh, [1])
refine_cells!(mesh, [2])
refine_cells!(mesh, [3])

total = 0 // 1
for i in 1:n_cells(mesh)
    if is_leaf(mesh.cells[i])
        num, den = cell_volume(mesh, i)
        total += num // den
    end
end
@assert total == 1 // 1   # exact!
```

This is a property worth taking advantage of: a conservative scheme that maintains physical conservation laws on a refining mesh has zero geometric drift. Any conservation error is from the physics scheme, not the framework.

## Parallelism

The threading layer partitions cells into chunks and processes them in parallel:

```julia
parallel_for_cells(mesh) do m, i
    fields.density[i] = some_initialization(i)
end
```

For reductions:

```julia
total_mass = parallel_reduce_cells(+, mesh; init=0.0) do m, i
    if is_leaf(m.cells[i])
        num, den = cell_volume(m, i)
        Float64(fields.density[i]) * (num / den)
    else
        0.0
    end
end
```

For more control, work with chunks directly:

```julia
parallel_for_chunks(mesh) do m, chunk
    # `chunk.cell_range` is the contiguous range of cells this thread owns
    for i in chunk.cell_range
        # ... per-cell work
    end
end
```

The chunk-based pattern is intentionally similar to how MPI domain decomposition would work, so code written this way will be easier to MPI-parallelize later.

## Memory pools

For long-running simulations, route short-lived buffer allocations through a pool to prevent fragmentation:

```julia
pool = FieldBufferPool{Float64}()

# In your timestep loop:
function step!(...)
    flux_buffer = acquire_buffer!(pool, n_cells(mesh))
    try
        # ... use flux_buffer
    finally
        release_buffer!(pool, flux_buffer)
    end
end
```

Or for stack-discipline workspace:

```julia
scratch = ScratchBuffer{Float64}(10_000)

with_scratch(scratch, n_cells(mesh)) do workspace
    # workspace is a view of length n_cells(mesh)
    # automatically released on exit
end
```

## Where to go next

- Read `architecture.md` for the why behind these design choices.
- Read `layouts.md` for a deeper look at the Storage layer and how to add new layouts.
- Read `neighbors_and_halos.md` for face adjacency, sparsity patterns, and the `HaloView` stencil wrapper.
- Read `boundary_conditions.md` for `BCKind`, `FrameBoundaries`, and the periodic Lagrangian wrap helper.
- Read `refinement_events.md` and `amr_driver.md` for keeping state in sync across AMR cycles, and for the topology-only driver `step_with_amr!`.
- Look at `examples/dsmc/` for a real mini-application that uses everything described above.
