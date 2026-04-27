# Lagrangian↔Eulerian overlap and polynomial remap

This document covers the Overlap layer (Layer 4) — the framework's
machinery for computing exact geometric intersections between Lagrangian
simplicial meshes and Eulerian hierarchical meshes, and for using those
intersections to do conservative polynomial-field remap in either
direction.

The per-pair clip ships in two interchangeable backends — `Float64`
(the default, covered here) and an integer-exact `Rational` path
(`overlap_simplex_box_exact!` plus the `IntegerLattice` quantization
helpers, covered in [Exact-rational overlap backend](exact_backend.md)).
The exact backend additionally unlocks `D = 4` (volume only); the float
path errors at `D ≥ 4`.

## What problem this solves

You have a Lagrangian description of some quantity — a deformed
triangulation in 2D, a tetrahedral mesh in 3D, with a polynomial
representation of (density, momentum, energy, …) per simplex. You want
to project that onto a regular or adaptive Cartesian grid (the Eulerian
quadtree/octree), preserving mass, momentum, and higher moments
exactly. Or you have the inverse problem: an Eulerian polynomial field
that you want to project back onto a Lagrangian mesh.

Both cases need the same geometric primitive — for every (Lagrangian
simplex, Eulerian leaf) pair that has nonzero overlap, you need the
exact polynomial moments of the overlap region up to some order. The
moments are everything; once you have them, the projection is linear
algebra.

## The pieces

```
HierarchicalGrids.Overlap
│
├── EulerianFrame           Wraps a HierarchicalMesh{D} with a physical
│                            bounding box. Cheap to construct; doesn't
│                            change as the mesh refines (only the cell
│                            list does).
│
├── compute_overlap          The geometric core. Given a Lagrangian
│                            SimplicialMesh and an EulerianFrame, return
│                            a GeometricOverlap with one entry per
│                            nonzero (lag, eul) pair.
│
├── GeometricOverlap         Sparse data structure: entries (each a
│                            volume + centroid + moments), plus CSR-like
│                            indices `lag_to_entries` (per Lagrangian
│                            simplex) and `eul_to_entries` (per Eulerian
│                            leaf). Either direction is O(1) per query.
│
├── polynomial_remap         Two-phase L→E remap: input is the overlap
│                            and per-Lagrangian polynomial coefficients;
│                            output is per-Eulerian coefficients.
│
├── polynomial_remap_E_to_L  Two-phase E→L remap (the inverse direction).
│
├── polynomial_remap_streaming_*
│                            Streaming versions of the above. Don't
│                            store moments; instead compute and
│                            accumulate on the fly. Constant memory.
│
└── PolynomialFieldSet       A FieldSet specialization where each
                              "field" is a per-cell vector of polynomial
                              coefficients in a chosen basis (Bernstein
                              or monomial). Used as the carrier for both
                              ends of a remap.
```

## A complete example: 2D mass conservation

```julia
using HierarchicalGrids

# Lagrangian: two triangles covering the unit square
lag = SimplicialMesh{2, Float64}(
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    Int32[1 2; 2 4; 3 3],   # triangle 1: 1-2-3; triangle 2: 2-4-3
    Int32[0 0; 0 0; 0 0],
)

# Eulerian: a depth-3 quadtree over [0,1]² (64 leaves)
eul = HierarchicalMesh{2}()
for _ in 1:3
    refine_cells!(eul, enumerate_leaves(eul))
end
frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

# Compute geometric overlap (degree-1 moments)
ov = compute_overlap(lag, frame; moment_order=1, leaf_size=8)

@assert n_entries(ov) > 0
@assert total_overlap_volume(ov) ≈ 1.0   # exact mass conservation
```

The same call works for `D = 1` (intervals against a 1D
hierarchical mesh) and `D = 3` (tetrahedra against an octree). For
1D, the underlying clip is `Geometry.interval_intersection`; the rest
of the API — moments, polynomial remap, the streaming variants —
follows the higher-D code path unchanged.

That's it for the geometry. From here, polynomial remap is a few lines:

```julia
# Mock per-simplex constant coefficients (polynomial order 0):
# every simplex has density = 2.0
ρ_lag = PolynomialFieldSet{2, Float64}(SoA(), n_simplices(lag), 0;
                                        density = Float64)
fill!(ρ_lag.density.coeffs, 2.0)

# Allocate output, run remap
ρ_eul = PolynomialFieldSet{2, Float64}(SoA(), n_cells(eul), 0;
                                        density = Float64)
polynomial_remap!(ρ_eul, ρ_lag, ov)

# Mass conservation (constant density × area = mass):
mass_lag = sum(ρ_lag.density.coeffs[s] * abs(simplex_volume(lag, s))
                for s in 1:n_simplices(lag))
mass_eul = 0.0
for ci in 1:n_cells(eul)
    is_leaf(eul.cells[ci]) || continue
    lo, hi = cell_physical_box(frame, ci)
    mass_eul += ρ_eul.density.coeffs[ci] * (hi[1] - lo[1]) * (hi[2] - lo[2])
end
@assert mass_lag ≈ mass_eul
```

## The data structure: GeometricOverlap

```
struct OverlapEntry{D, T}
    lag_idx::Int32
    eul_idx::Int32
    volume::T
    centroid::NTuple{D, T}
    moments::Vector{T}        # graded-lex order, origin = physical origin
end

struct GeometricOverlap{D, T}
    entries::Vector{OverlapEntry{D, T}}
    lag_to_entries::Vector{UnitRange{Int}}     # CSR row pointers
    eul_to_entries::Vector{Vector{Int}}        # vector-of-vectors (sparse)
    moment_order::Int
    n_lag::Int
    n_eul::Int
end
```

Two indexing structures because the access patterns are asymmetric:

- **Per-Lagrangian-simplex**: contiguous `entries` are sorted by
  `lag_idx`, so `lag_to_entries[s]` is a single `UnitRange` — a `view`
  is enough. This is the cache-friendly side and the one used by
  L→E remap.
- **Per-Eulerian-leaf**: not contiguous because Eulerian leaves are
  visited in BVH-traversal order during compute, not in cell-storage
  order. So `eul_to_entries[i]` is a `Vector{Int}` of indices into
  `entries`. This is the side used by E→L remap and by the AMR
  indicator in the Lagrangian-pixelization example.

Both invariants hold post-construction: `entries` is sorted lexicographically
by `(lag_idx, eul_idx)`, `lag_to_entries` is a valid CSR row-pointer
structure (every range is contiguous and in storage order), all volumes
are strictly positive.

## Sequential vs parallel compute

`compute_overlap` is parallelized via OhMyThreads.jl. The schedule:

1. Build a Lagrangian BVH (fast for typical mesh densities).
2. Partition the Eulerian leaves into chunks.
3. For each chunk, in parallel, traverse the BVH and accumulate
   per-thread `OverlapEntry` lists.
4. Merge per-thread lists into the global `entries` array, then build
   the index structures.

For very small problems the parallel version's overhead dominates; the
implementation auto-falls-through to the sequential path below a
threshold (controllable via `parallel = :auto | :always | :never`).

The parallelization is **race-free by construction** — each thread
writes to its own per-thread buffer; the merge happens serially. There
were two race conditions caught during the audit (mesh-cache
construction needed to happen before the parallel section, and an
`init` value passed to `tmapreduce` was being mutated); both are now
fixed and tested.

## Two-phase vs streaming polynomial remap

The two-phase approach computes all overlap moments first
(`compute_overlap`), stores them in `GeometricOverlap`, and then a
separate pass uses them to do the projection. Pros: simple, parallelizable,
geometry is reusable across multiple fields. Cons: memory cost is
O(n_entries × moments_per_entry).

The streaming approach (`polynomial_remap_streaming_L_to_E!` and
`_E_to_L!`) computes per-pair moments on the fly and immediately
accumulates the projection contribution. Pros: O(1) extra memory beyond
the input/output FieldSets. Cons: each direction has a different
access pattern (the L→E version walks Lagrangian simplices outer,
Eulerian leaves inner; the E→L version walks Eulerian leaves outer);
recomputing for multiple fields is wasteful.

Pick two-phase by default. Switch to streaming when memory is the
constraint, or when the per-pair moment computation dominates and the
field count is one.

## Multi-field remap

`PolynomialFieldSet` supports multiple polynomial fields per cell.
A `polynomial_remap!` call with a multi-field source and destination
remaps all fields in one pass — the geometric work is done once and
shared across fields. This is the right pattern for hydro state
(density + momentum + energy as four polynomial fields) or for
combined density + velocity remap in cosmology.

## Conservation guarantees

Polynomial remap conserves the constant-mode mass exactly (to round-off).
Higher-order modes are exact for polynomials of order ≤ `moment_order`,
and the limiting error is the truncation error of the moment order, not
arithmetic. For `moment_order = 1` (linear), all linear functions
remap exactly. For `moment_order = 2` (quadratic), quadratics remap
exactly. And so on.

Mass on inverted simplices: when a Lagrangian simplex inverts (negative
signed volume), the remap treats its absolute volume as the mass-bearing
region. In two-stream regions of cosmological flows, this gives the
correct multi-stream density; for general flows where inversion is an
artifact, you probably want to detect inversion separately and decide
how to handle it.

## Diagnostics

`RemapDiagnostics{T}` is a mutable accumulator passed through to the
polynomial remap routines for shell-crossing surveillance and kernel
sanity checks:

- `liouville_min`, `liouville_max` — extrema of the per-pair Jacobian
  proxy `entry.volume / source_physical_volume`. Values close to zero
  flag candidate shell-crossings; values far from one flag unusual
  stretch.
- `total_volume_in`, `total_volume_out` — accumulated overlap volumes
  on the source and target sides. Each entry contributes `entry.volume`
  to both, so the running totals must agree.
- `n_negative_jacobian_cells` — count of entries with non-positive
  Jacobian proxy. `compute_overlap` guarantees positive volumes, so a
  non-zero count signals upstream corruption (e.g. a Lagrangian simplex
  that inverted past the builder's checks).

```julia
diag = RemapDiagnostics(Float64)
polynomial_remap_l_to_e!(eul_field, lag_field, ov; diagnostics = diag)
@show diag
@assert diag.total_volume_in ≈ diag.total_volume_out
@assert diag.n_negative_jacobian_cells == 0

# Reuse across passes:
reset!(diag)
# Or reduce thread-local copies via Base.merge!(d_acc, d_thread).
```

`total_overlap_volume(ov)` and `n_entries(ov)` cover the lighter
geometry-only summary; `RemapDiagnostics` is the heavier per-remap
sanity check.

## Performance notes

- The BVH build dominates large meshes (millions of simplices). The
  current implementation uses a top-down median split with `leaf_size`
  controlling when to stop subdividing. `leaf_size = 8` is the empirical
  sweet spot for typical 2D/3D problems.
- For typical 2D problems with ~1000 Lagrangian simplices and ~1000
  Eulerian leaves, sequential `compute_overlap` runs in <10 ms;
  `polynomial_remap!` is comparable.
- Parallel speedup levels off around 4 threads on most machines for
  problems below ~10⁵ pairs. Above that, the parallel version scales
  linearly with thread count up to memory bandwidth.
- For hot loops (per-frame remap in a long simulation), allocate
  `GeometricOverlap` and `PolynomialFieldSet` once at warmup and reuse;
  see `Memory` layer for the buffer pool patterns.

## See also

- `examples/lagrangian_pixelization/` — a worked 2D example using
  `compute_overlap` and AMR-driven Eulerian refinement.
- `docs/src/architecture.md` — overall framework structure.
- [r3djl](https://github.com/yipihey/r3djl) — the underlying clipping
  library.
- Powell, D. & Abel, T. (2015) — the foundational paper for r3d's
  algorithm.
