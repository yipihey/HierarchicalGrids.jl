# HierarchicalGrids.jl

[![CI](https://github.com/yipihey/HierarchicalGrids.jl/actions/workflows/test.yml/badge.svg)](https://github.com/yipihey/HierarchicalGrids.jl/actions/workflows/test.yml)

A Julia framework for hierarchical adaptive grids in arbitrary dimensions, with
integer-exact geometry, decoupled memory layouts, and an exact polyhedral
overlap layer for Lagrangian↔Eulerian remap.

## Status

**v0.1 — pre-release.** All foundation layers built and tested:
10,872 tests passing across mesh, geometry, storage, threading, memory,
quadrature, polynomial bases, and the geometric overlap stack. Two
worked examples (DSMC, Lagrangian pixelization) demonstrate building
real applications on top of the framework.

The framework is intentionally a *substrate*, not a turnkey simulation
tool — solvers go in user code, examples, or downstream packages.

## What it gives you

- **A hierarchical mesh** in any dimension `D` (1D, 2D, 3D, 4D, ...) with
  anisotropic refinement, integer-exact relative coordinates, and lazy
  parent/level/subtree caches.
- **A simplicial mesh** for Lagrangian triangulations or tetrahedral
  meshes, with reference and current vertex positions, deformation
  gradients, and distortion metrics.
- **Layout-flexible field storage** (`SoA`, `AoS`, `Blocked{B}`) — switch
  layouts with a one-line constructor change; kernel code is unchanged.
- **A geometric overlap layer** wrapping
  [r3djl](https://github.com/yipihey/r3djl): exact 2D/3D clipping of
  Lagrangian simplices against Eulerian cells, with polynomial moments
  up to user-specified order, in both sequential and parallel
  (OhMyThreads.jl) modes.
- **Polynomial fields and remap** — represent quantities as Bernstein-Bézier
  polynomials per cell; conservatively remap between Lagrangian and
  Eulerian descriptions in either direction with per-cell streaming or
  two-phase scatter–gather.
- **Threading and memory primitives** — chunk-based parallelism that
  mirrors future MPI structure; pool, scratch, and arena allocators for
  long-running simulations.

## What it deliberately doesn't have

- **No solvers.** No hydro, no Poisson, no I/O. These belong in user code
  or downstream modules. The DSMC and Lagrangian-pixelization examples
  show how a real application sits on top.
- **No MPI yet.** The chunk-based threading and process-independent
  cell paths are designed for distribution; we'll add MPI when there's
  a concrete use case.
- **No GPU yet.** Same: the layout abstraction is the foundation; a
  `BlockedAoSoA{32}` layout plus a CUDA/Metal/Vulkan backend for the
  threading layer is an additive change, not a rewrite.
- **No documentation generator.** Docs are hand-written Markdown in
  `docs/src/`. Documenter.jl integration can be added later.

## Design goals

- **Mesh purity.** The mesh holds *only* structural data — no fields,
  no physics, no I/O. This is the hard-learned lesson from frameworks
  (Enzo, FLASH, others) where the mesh class accumulated hundreds of
  members over decades. Composition over accretion.
- **Hierarchical relative coordinates.** Cells are stored relative to
  their parents, never as absolute coordinates. Bit widths in arithmetic
  depend on depth difference between cells and their LCA, not on absolute
  depth. Deep zoom doesn't widen integer types.
- **Integer-exact geometry.** Vertices are signed integers,
  cell-center-relative. Predicates are exact. Volumes sum exactly under
  refinement.
- **Layout decoupling.** Field storage uses Taichi-inspired layout
  abstraction: kernel code is layout-agnostic; switching SoA ↔ AoS ↔
  Blocked is a one-line change.
- **Anisotropic refinement first-class.** Cells can split along any subset
  of axes via a per-cell split mask. Fully-isotropic refinement gets
  fast-path treatment.
- **Future-MPI-ready, not MPI-coupled.** Cell paths are process-independent
  identities; threading uses chunk-based partitioning that mirrors how
  MPI domains would split. We don't pay the MPI tax until needed.

## Architecture

```
Layer 0   — BitPrimitives    POPCNT, PDEP, PEXT, integer type selection
Layer 1   — Mesh             cells, hierarchical tree, refinement,
                              SimplicialMesh, RefineByIndicator
Layer 2   — Geometry         predicates, volumes, axis-aligned ops
Layer 2   — Bases            Bernstein-Bézier polynomial bases
Layer 2   — Quadrature       Gauss-Legendre and simplex quadrature
Layer 2.5 — Storage          layout-flexible field storage; PolynomialFieldSet
Layer 3   — Threading        chunk-based parallelism (OhMyThreads.jl)
Layer 3   — Memory           pools, scratch buffers, arenas
Layer 4   — Overlap          Lagrangian↔Eulerian geometric overlap and
                              polynomial remap, via r3djl
Layer 4   — Diagnostics      mesh diagnostics, conservation checks
```

Each layer depends only on layers below. See `docs/src/architecture.md`
for the rationale.

## Quick start

### Installing

This package depends on [R3D](https://github.com/yipihey/r3djl) (the
Julia port of r3d), which is hosted on GitHub but not yet registered
in the General registry. The framework's `Manifest.toml` pins R3D to
a specific commit, so a fresh clone resolves with:

```bash
git clone https://github.com/yipihey/HierarchicalGrids.jl.git
cd HierarchicalGrids.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

If you want to add this package to your own project (rather than
develop it in place), declare both packages by URL:

```julia
using Pkg
Pkg.add(url="https://github.com/yipihey/r3djl", subdir="R3D.jl")
Pkg.add(url="https://github.com/yipihey/HierarchicalGrids.jl")
```

### A first mesh

```julia
using HierarchicalGrids

# Build a 3D mesh, refine some cells
mesh = HierarchicalMesh{3}()
refine_cells!(mesh, [1])           # 8 children of root
refine_cells!(mesh, [2, 5])        # refine first and fourth child

# Allocate per-cell field storage (SoA layout)
fields = allocate_fields(SoA(), n_cells(mesh);
                         density = Float32,
                         momentum = NTuple{3, Float32})

# Initialize in parallel
parallel_for_cells(mesh) do m, i
    fields.density[i] = 1.0f0
    fields.momentum[i] = (0.0f0, 0.0f0, 0.0f0)
end

# Switching to AoS is a one-line change — kernel code unchanged
fields_aos = allocate_fields(AoS(), n_cells(mesh);
                             density = Float32,
                             momentum = NTuple{3, Float32})
```

### A first overlap

```julia
using HierarchicalGrids

# A 2D Lagrangian triangulation and an Eulerian quadtree, both over the
# unit square. Compute their exact geometric overlap and check
# mass conservation.
lag = SimplicialMesh{2, Float64}(
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],   # vertices
    Int32[1 2; 2 4; 3 3]',                                # two triangles
    Int32[0 0; 0 0; 0 0]',                                # no neighbors
)
eul = HierarchicalMesh{2}()
for _ in 1:3; refine_cells!(eul, enumerate_leaves(eul)); end   # depth-3 grid
frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

ov = compute_overlap(lag, frame; moment_order=1, leaf_size=8)
@show n_entries(ov)                  # number of nonzero overlap pairs
@show total_overlap_volume(ov)       # 1.0 to round-off
```

## Examples

Each example is a standalone Julia package under `examples/`, depending
on the framework via `Pkg.develop`. They demonstrate different aspects
of building applications on top.

- **`examples/dsmc/`** — A Direct Simulation Monte Carlo solver for
  hard-sphere gas relaxation. Demonstrates particle storage with
  layout-flexible `FieldSet`, time-splitting (move/sort/collide/sample),
  per-cell collision binning, and time-averaged macroscopic quantities.
- **`examples/lagrangian_pixelization/`** — A 2D Lagrangian-to-Eulerian
  pixelization with adaptive refinement. The Lagrangian triangulation
  deforms under a flow map (swirl + Gaussian compression); the Eulerian
  quadtree refines to keep approximately equal Eulerian cells per
  Lagrangian patch. Shows `compute_overlap`, AMR via
  `refine_by_indicator!`, and physical density visualization.
  Comes with a standalone SVG renderer (no plotting dependencies).

## Tests

```bash
julia --project=. test/runtests.jl
```

10,872 tests; ~50 s on 2 threads. With multithreading:

```bash
JULIA_NUM_THREADS=2 julia --project=. test/runtests.jl
```

## Documentation

- `docs/src/architecture.md` — design rationale, layer-by-layer overview
- `docs/src/getting_started.md` — tutorial walking through mesh, fields,
  refinement
- `docs/src/layouts.md` — the layout abstraction in depth
- `docs/src/anisotropic_refinement.md` — split masks and how to use them
- `docs/src/overlap.md` — the Overlap layer and Lagrangian↔Eulerian remap
- `docs/src/extending.md` — adding new layouts, new geometry operations

## License

MIT. See `LICENSE`.

## Citation

If you use this framework in published work, please cite both
HierarchicalGrids.jl (this repository) and r3djl, plus the underlying
r3d paper:

> Powell, Devon, and Tom Abel. "An exact general remeshing scheme
> applied to physically conservative voxelization." *Journal of
> Computational Physics* 297 (2015): 340–356.
> [doi:10.1016/j.jcp.2015.05.022](https://doi.org/10.1016/j.jcp.2015.05.022)
