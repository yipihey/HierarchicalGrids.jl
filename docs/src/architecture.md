# Architecture

This document explains why HierarchicalGrids.jl is structured the way it is. The structure is driven by lessons from existing simulation frameworks (especially Enzo, FLASH, Trixi.jl) and by some specific bets about what's coming next in computational science.

## The core bet

Simulation frameworks tend to fail in one of two ways. Either they accumulate complexity for decades until the central data structures carry hundreds of fields and thousands of methods (Enzo's `Grid` class is the canonical example), or they specialize so tightly to one access pattern that adapting to new physics or new hardware requires rewriting from scratch.

HierarchicalGrids takes a third path: aggressive separation of concerns, with the mesh holding only structural data and everything else built as composable layers above it. Adding a new physics module doesn't touch the mesh. Switching memory layouts doesn't touch the kernels. Going from 3D to 4D doesn't touch the algorithms.

The bet is that this separation pays off over the lifetime of a research code, even if it costs a little upfront in indirection.

## Layered architecture

Each layer depends only on layers below. Crossing layers in the wrong direction is a code-review red flag.

| Layer | Module(s) | Purpose |
|-------|-----------|---------|
| 0     | BitPrimitives | Hardware bit ops, integer type selection |
| 1     | Mesh, SimplicialMesh, RefineByIndicator, CompositeMesh | Cells, hierarchical tree, refinement; simplicial meshes for Lagrangian work |
| 2     | Geometry | Predicates, volumes, axis-aligned ops |
| 2     | Bases | Bernstein-Bézier polynomial bases |
| 2     | Quadrature | Gauss-Legendre and simplex quadrature |
| 2.5   | Storage, PolynomialFieldSet | Layout-flexible field storage |
| 3     | Threading | Chunk-based parallelism (OhMyThreads.jl) |
| 3     | Memory | Pools, scratch buffers, arenas |
| 4     | Overlap | Lagrangian↔Eulerian geometric overlap and polynomial remap |
| 4     | Diagnostics | Mesh diagnostics, conservation checks |

### Layer 0 — BitPrimitives

The hardware bit operations and integer type selection. Most code in the framework doesn't touch this layer directly; it's the foundation that the mesh and geometry build on.

The interesting decisions here are:

**PDEP/PEXT for sibling/mask scatter-gather.** When a cell can split along an arbitrary subset of its D axes, encoding which child is which becomes a bit-manipulation problem: given a sibling index (0..2^k-1 where k is the popcount of the split mask) and the split mask itself, where does each axis's bit live? PDEP scatters bits across mask positions in a few cycles; PEXT does the inverse. On hardware without BMI2, we fall back to software loops — the rest of the framework is unaffected.

**Type selection by dimension.** For D ≤ 7, sibling indices and split masks fit in UInt8. For D ≤ 15, UInt16. The framework selects automatically via `sibling_index_type(Val(D))`. This is why per-cell metadata is 3 bytes for typical 3D meshes: we're not paying for bit width we don't need.

### Layer 1 — Mesh

The canonical per-cell data and the hierarchical tree.

**Mesh purity.** This is the most important architectural decision. The mesh contains only:
- Cell sibling indices and split masks (the structural data).
- Flags (leaf, boundary, dirty).
- Lazy caches (parent indices, per-cell levels, subtree sizes).

It does not contain fields, particles, gravity solvers, I/O methods, or any other concern. Those go in separate modules and are associated with the mesh by parallel-array indexing.

**DFS storage order.** Cells are stored in depth-first traversal order. This means subtrees are contiguous, which matters for cache-friendly iteration over a region and for future MPI domain decomposition (a chunk of cells in DFS order is naturally a complete subtree or a small number of them).

**Hierarchical relative coordinates.** A cell's identity is its sibling index plus its parent's identity (recursively, until you hit the canonical reference cell). Absolute coordinates are derivable but rarely needed — most operations between two cells use their LCA-relative frame. This matters because the bit width of any operation is bounded by the depth difference between the cells and their LCA, not by the absolute depth of either cell. Adding a 60-level zoom-in doesn't widen integer types; you just have more cells.

**Anisotropic refinement first-class.** A cell can be split along any subset of axes. The split mask carries which. Children inherit the split mask of their parent (so they know how they were created); a child's own split mask describes how *its* children would be created if it were refined. Fully-isotropic refinement is the common case and gets a fast-path constant (`FULLY_ISOTROPIC_MASK`).

**Lazy caches.** Parent indices, per-cell scalar levels, and subtree sizes are derived data. They're stored as parallel arrays alongside the cells but are invalidated by any structural change and rebuilt on first access. This is a tradeoff: some operations pay a one-time O(N) rebuild cost after refinement; in exchange, all subsequent operations are O(1) lookups instead of O(depth) walks.

### Layer 2 — Geometry

Integer-exact geometric operations.

**Why integer?** Floating point is fine for almost all simulation work. But for geometric *predicates* — does this point lie inside this cell, on which side of this plane, is this triangle degenerate — float arithmetic introduces inconsistencies that can cause topological errors. Integer arithmetic is exact, so predicates are exact. For cell-relative coordinates with reasonable bit depth, integer arithmetic is also as fast or faster than floating-point.

**LCA-relative frames.** When computing the relative position of cell A with respect to cell B, we don't go through absolute coordinates. We find the LCA, walk down to A and B from it, and accumulate per-axis position bits. The result is exact integer offsets in the LCA's local frame. The bit width of the result depends on the depth from the LCA to each cell — which for nearby cells is small, regardless of how deep the absolute zoom is.

**Volumes as exact rationals.** A cell's volume in canonical-reference units is 1/(2^total_level) for axis-aligned refinement. We return (numerator, denominator) tuples. Sum of volumes under refinement is exactly 1 (or whatever the parent volume was) — no floating-point drift, no conservation errors creeping in over many timesteps.

### Layer 2.5 — Storage

The layout abstraction. This is the Taichi-inspired piece.

**Layout as a type parameter.** A `FieldSet` is parameterized by its layout type (`SoA`, `AoS`, `Blocked{B}`). The user-facing access syntax (`fields.density[i]`) is the same regardless of layout. Layout-specific methods dispatch via multiple dispatch, so the indexed access compiles to direct memory operations.

**Why this matters.** Different physics has different access patterns:
- Streaming kernels (apply one operation to one field over all cells) want SoA — adjacent cells of the same field are adjacent in memory.
- Per-cell update kernels (read all fields, compute, write all fields) want AoS — all fields of the same cell are adjacent in memory.
- Stencil operations want Blocked layouts — cells in a spatial neighborhood are nearby in memory.

In a traditional framework, picking the wrong layout for a hot path can cost 2-5x in cache performance. Switching layouts requires rewriting kernels. With layout abstraction, switching is a one-line change in the constructor; you can benchmark all three and pick the best one.

**Why "Layer 2.5"?** It conceptually sits between geometry and the higher-level threading/memory layers — fields are an extension of mesh cells, but they're more general (you can have field storage that isn't per-cell, like per-particle storage in the DSMC example). The naming captures that it's a foundational utility used throughout, not just a single solver concern.

### Layer 3 — Threading

Chunk-based parallelism that mirrors future MPI structure.

**Chunks are the unit of work.** Whether you're using shared-memory threads or future MPI processes, the partitioning of cells into independent work units is the same logical operation. The current implementation uses Julia's `Threads.@threads` and equal-size partitioning; switching to OhMyThreads.jl or Polyester.jl is a one-place change here.

**No blocks, no MPI in the core.** Block structures (in the GPU sense) and MPI are explicit choices that this framework defers. Single-node shared-memory machines with hundreds of cores and terabytes of RAM are increasingly the norm; designs that pay overhead for distribution they don't need are paying that overhead forever. The chunk-based design preserves the option to add MPI later without restructuring.

### Layer 3 — Memory

Pool allocation, scratch buffers, and arenas.

**Why explicit memory management in Julia?** Julia's GC handles short-lived allocations well, but long-running scientific simulations stress allocators in ways that consumer workloads don't. Repeated allocation and freeing of buffers of similar but varying sizes leads to memory fragmentation that the GC can't fix (because nothing is leaked, the heap just gets fragmented). Frameworks like Enzo have hit this in production. Explicit pools mean: you allocate once during warmup, and after that, your "allocations" are just metadata changes against pre-existing buffers.

**Three patterns:**
- `FieldBufferPool` — recycles buffers by power-of-two size class. Best for short-lived field storage that's reused across timesteps.
- `ScratchBuffer` — stack-style allocator. Best for workspace that has clear scope (LIFO discipline, no fragmentation possible by construction).
- `Arena` — bulk allocation, freed all at once. Best for per-operation temporaries where individual lifetimes don't matter.

### Layer 2 — Bases

Bernstein-Bézier polynomial bases over simplices and boxes.

**Why Bernstein-Bézier?** When you need to represent a polynomial field over a cell — for higher-order remap, for limiter detection, for visualization — the Bernstein basis has properties no other polynomial basis matches: nonnegativity (a coefficient ≥ 0 means a positive polynomial somewhere in the cell), partition of unity (coefficients sum to 1, useful for averaging), convex-hull containment (the polynomial value lies in the convex hull of its coefficients). For limiter detection in particular, Bernstein nonnegativity is *the* standard certificate — you can prove a polynomial doesn't go negative anywhere in a cell by checking finitely many coefficients.

The framework provides Bernstein bases for both simplex (Lagrangian) and box (Eulerian) cells, conversions between Bernstein and monomial coefficients, and predicates like `is_positive_certificate`.

### Layer 2 — Quadrature

Gauss-Legendre quadrature on intervals and boxes; quadrature on simplices.

The Overlap layer's polynomial remap doesn't need quadrature — it uses analytic moment integrals over polytopes via Koehl's recursion, which is exact for polynomials. Quadrature is here for code that does need to evaluate integrals of arbitrary integrands (limiters using nonlinear functions of the polynomial state, for instance).

### Layer 4 — Overlap

Geometric overlap between Lagrangian (simplicial) and Eulerian (hierarchical) meshes, plus conservative polynomial remap built on top.

**The bedrock is r3djl.** All actual polytope clipping is delegated to [r3djl](https://github.com/yipihey/r3djl), the pure-Julia port of Devon Powell's r3d. r3djl handles the per-pair "intersect this Lagrangian simplex with this Eulerian box and compute moments up to order N" operation. Our layer is the *organization* on top: the BVH that finds candidate pairs, the CSR-like sparse data structure that holds all nonzero pairs, the per-Lagrangian and per-Eulerian indexing, the parallel scheduling, the polynomial remap that uses the moments.

**Two-phase vs streaming.** Conservative polynomial remap on a polynomial field — projecting from Lagrangian polynomial coefficients onto Eulerian polynomial coefficients with mass and higher-moment conservation — has two valid schedules:

- **Two-phase**: compute all overlap moments first, store them in a sparse data structure, then run a separate pass that uses those moments to do the projection. Simple, parallelizable, and lets you query the geometry independently of any remap. Cost: O(memory) for the stored moments.
- **Streaming**: never store moments; for each pair, compute moments on the fly and immediately accumulate the projection contribution. Fixed memory, but coupling the geometry and remap means a different access pattern for each direction (L→E vs E→L) and you can't reuse the geometry for multiple fields without recomputing.

Both are implemented. The two-phase API is `compute_overlap` + `polynomial_remap_*`; the streaming API is the `polynomial_remap_streaming_*` family. Use two-phase for clarity and multi-field reuse; use streaming when memory is a hard constraint.

**Anti-feature: no curved-edge support.** The current code assumes straight-edged Lagrangian simplices. Curved-edge support — for higher-order Lagrangian meshes — requires a "lift to flat polytope in higher dimension" trick that's still under design. See the README's "What we deliberately don't have" section.

### Layer 4 — Diagnostics

Conservation checks, mesh statistics, sanity diagnostics. Cheap to enable in development; cheap to disable in production. Doesn't extend any concept; just consumes them.

## What we deliberately don't have (yet)

**Solvers.** No hydro solver, no Poisson solver, no nothing. Solvers go in Layer 5+, built on top of this framework. The DSMC and Lagrangian-pixelization examples show the pattern.

**MPI.** Designed for, not implemented. Cell paths give us process-independent identities; chunks give us the partitioning unit. Adding MPI is straightforward; doing so prematurely would lock in design decisions we don't have data to make yet.

**GPU kernels.** Same as MPI — designed for, not implemented. The layout abstraction is the key piece; once we have GPU-friendly layouts (e.g., `BlockedAoSoA{32}` for warp-aligned access), kernels written against the layout abstraction will work without modification.

**Curved-edge polytopes.** The Overlap layer handles flat-faced (straight-edged) Lagrangian and Eulerian cells. Cubic-edge or general-curved-edge clipping would lift cubic-edge triangles into a higher-dimensional flat polytope, then project moments back down. The lift construction is on the roadmap; the current code computes overlaps for triangles and tetrahedra with straight edges only.

## What changes vs Enzo / FLASH / similar

The biggest single change is the mesh purity. In Enzo, `Grid.h` has 180+ data members covering hydrodynamics state, gravity, particles, refinement metadata, I/O state, and so on. Adding a new physics module typically meant adding members to `Grid`. After 30 years, the class is a swamp.

In HierarchicalGrids, adding a new physics module means adding a new struct that holds *its* state, with parallel-array indexing into the mesh. The mesh stays unchanged. Modules don't know about each other unless they need to.

The cost is some indirection: instead of `grid->density[i]`, you write `fields.density[i]`. The benefit is decades of clean evolution.

## Cross-cutting topics

A few features cut across several layers and are documented in their
own files:

- [Refinement events](refinement_events.md) — `RefinementEvent` and
  the `register_refinement_listener!` / `unregister_refinement_listener!`
  observer pattern that keeps external state (fields, particle bins,
  derived caches) consistent across batched mesh changes.
- [Neighbors and halos](neighbors_and_halos.md) — the lazily-built
  face-neighbor graph (`face_neighbors`, `face_fine_neighbors`,
  `cell_adjacency_sparsity`) and the `HaloView` stencil-indexing
  wrapper. A balanced `HierarchicalMesh` (`balanced::Bool` constructor
  flag) limits the level gap on any face to 1.
- [Boundary conditions](boundary_conditions.md) — the `BCKind` enum,
  `BoundarySpec`, `FrameBoundaries{D}`, the periodic Lagrangian wrap
  (`periodic!`), and the Dirichlet pin (`pin_boundary_simplices!`,
  `is_pinned`).
- [Topology-only AMR driver](amr_driver.md) — `step_with_amr!` is a
  small interleaver that calls a user `step!` callback and periodically
  invokes `refine_by_indicator!` on a hysteresis schedule. Kept
  intentionally minimal; per-cell state stays consistent via the
  refinement-listener mechanism.
- [Exact-rational overlap backend](exact_backend.md) — `IntegerLattice`
  quantization, `overlap_simplex_box_exact!`, and the `audit_overlap`
  harness. Bit-exact, deterministic, robust on degenerate orientations,
  and the only path that reaches `D = 4` (volume only) today.

## What we expect to add

- **Patch-based AMR (Berger-Oliger).** Some physics works better with patches than cell-trees. The geometry and mesh layers are sufficient to support patches as a parallel mesh type; we'll add this when we have a use case.
- **Voronoi/moving-mesh support.** The integer-exact predicates here are designed with future polytope clipping in mind. Adding a Voronoi mesh type that uses these predicates is a reasonable next phase.
- **GPU layouts.** `BlockedAoSoA{32}` plus a CUDA/Metal/Vulkan backend for the threading layer.
- **Solvers.** Hydro, Poisson, particles. Each as a Layer 4 module that uses the framework but doesn't extend it.
