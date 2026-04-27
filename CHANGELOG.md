# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **1D parity for `compute_overlap`**: closed-form interval intersection
  in `src/Overlap/r3d_adapter.jl`; `HierarchicalMesh{1}` and
  `SimplicialMesh{1, T}` are now first-class across the overlap and
  polynomial-remap stack. `Geometry.interval_intersection` powers the
  D=1 path with no external dependency.
- **`bernstein_positivity_certificate(coeffs, basis; atol)`** and
  **`is_strictly_positive(field; atol)`**: convex-hull positivity
  certificate returning the offending Bernstein multi-index on failure.
  Hot-path `all_bernstein_coeffs_positive` is preserved.
- **`RemapDiagnostics{T}`**: per-remap accumulator (Liouville Jacobian
  extrema, total volumes, negative-Jacobian cell count) plumbed
  through `polynomial_remap_l_to_e!` and `polynomial_remap_e_to_l!`
  via an optional `diagnostics` kwarg. Zero overhead when omitted.
- **`init_field_from!(field, frame_or_mesh, f; quadrature_order)`**:
  per-cell L²-projection of an analytical function onto a
  `PolynomialFieldSet`, using existing Gauss quadrature.
- **`RefinementEvent` + `register_refinement_listener!` /
  `unregister_refinement_listener!`**: observer pattern on
  `HierarchicalMesh` that fires after `refine_cells!` /
  `coarsen_cells!`. Each event carries the full old→new cell-index
  permutation so downstream per-cell storage can resize and permute
  in lockstep.
- **Face-neighbor graph and halo view**: `face_neighbors`,
  `face_fine_neighbors`, `cell_adjacency_sparsity` (returns
  `SparseMatrixCSC{Bool, Int32}`), and `halo_view` for zero-allocation
  per-cell stencil access. Built lazily, invalidated automatically by
  the refinement listener. Path-walking neighbor finder runs in
  O(N · D · depth) and is ~2.7× faster than the bucket variant on
  256×256 leaf meshes.
- **`balanced::Bool` field on `HierarchicalMesh`**: optional 2:1
  balance enforcement — when set, `refine_cells!` runs a balance pass
  that auto-refines coarser face-neighbors until the level gap on
  every face is ≤ 1.
- **Boundary-condition framework**: `BCKind` enum (`PERIODIC`,
  `INFLOW`, `OUTFLOW`, `REFLECTING`, `DIRICHLET`), `BoundarySpec{D}`,
  `FrameBoundaries{D}` companion to `EulerianFrame` (no breaking
  type-parameter change), `face_neighbors_with_bcs` for neighbor-graph
  wrap-around, `periodic!(mesh::SimplicialMesh, axes, bounds)` to
  rewire wrap-around simplex neighbors, and
  `pin_boundary_simplices!` / `is_pinned` for advisory Dirichlet
  pinning.
- **Periodic ghost-overlap in `compute_overlap`**: when
  `frame_bcs::FrameBoundaries{D}` has periodic axes, the AABB query
  generates ghost-image entries via Lagrangian-translation. Up to
  `3^k - 1` ghost shifts for `k` periodic axes; round-trip
  `total_overlap_volume == 1` holds bit-stably for tile-decomposed
  Lagrangian meshes that wrap.
- **`step_with_amr!(state, frame, step_fn, indicator, n_steps; ...)`**:
  topology-only AMR driver that interleaves user-supplied physics
  steps with `refine_by_indicator!` on a hysteresis schedule. CFL/dt
  remain the caller's responsibility.
- **`R3D.IntExact` adapter for exact-rational overlap**: new file
  `src/Overlap/r3d_int_adapter.jl` with `IntPairScratch{D, T}` and
  `overlap_simplex_box_exact!` returning `Rational{R}` volume,
  centroid, and full polynomial moments at D ∈ {2, 3, 4}.
  `_default_accumulator` follows IntExact's documented type-promotion
  guidance.
- **`IntegerLattice{D, T}` quantization helpers**: equal-scale lattice
  built from an `EulerianFrame`. `quantize`, `quantize_strict`,
  `dequantize`, `lat_resolution`, `unscale_volume`, `unscale_moment`
  bridge float vertices and integer coordinates round-trip.
- **`audit_overlap`** + **`OverlapAuditReport`**: side-by-side float vs
  IntExact comparison on a canonical polytope battery; max relative
  diff is at the machine-precision floor (~1.33e-16). Module-load
  consistency check runs in `Diagnostics.__init__()`, gated by
  `ENV["HG_INTEXACT_VERIFY"]` (default on).
- **`compute_overlap(..., backend = :exact)`**: `backend::Symbol` kwarg
  routes the per-pair clip through `overlap_simplex_box_exact!` with
  an auto-derived `IntegerLattice` and accumulator. Result is a
  drop-in `GeometricOverlap{D, Float64}`. Default `:float` backend is
  byte-identical to today. Performance: ~4.6× slowdown vs `:float` at
  D=2 with `moment_order=1` and the BigInt-promoted default
  accumulator. `D ∈ {2, 3, 4}` supported with full polynomial moments;
  `D=1` is rejected (use `:float`, already exact); periodic ghosts
  under `:exact` are deferred.
- **D=4 dispatch under the `:exact` backend** with full polynomial
  moments: integer pentachoron, Kuhn 24-pentachoron tile
  decomposition, P=1 centroid, and P=2 second-moment tests all pass
  bit-exactly via `R3D.IntExact.moments_exact!`. Float
  `compute_overlap` continues to error at D≥4 by design.
- **Comprehensive D=3 test coverage** for `compute_overlap` and
  `polynomial_remap`: identity, refined-Eulerian, translated,
  empty/partial-overlap, P=0 and P=1 round-trip, RemapDiagnostics
  variants. Surfaces and documents the positive-orientation
  requirement for tetrahedral simplices.

### Changed

- **r3djl pin bumped** from `b08c363` to `ff23853f` (as of 2026-04-27).
  Unlocks `R3D.IntExact.moments_exact!` at D ∈ {4, 5, 6} via the new
  `_moments_exact_dgeneric_4plus!` simplex-decomposition path (landed
  upstream in commit `943135f1`). HG surfaces this for `D = 4` only
  (D=5/6 remain research-grade upstream). The D=4 adapter dispatch
  in `src/Overlap/r3d_int_adapter.jl` now mirrors D=2/D=3 exactly:
  clip + `moments_exact!`, full polynomial-moment output, centroid
  populated from first-order moments. The `compute_overlap(..., backend
  = :exact)` `D = 4 ⇒ moment_order = 0` guard is removed; any
  `moment_order ≥ 0` is now accepted at D = 4. No public-API breakage:
  signatures (`init_simplex!`, `clip!`, `moments_exact!`,
  `volume_exact`, `overlap_simplex_box_exact!`, `compute_overlap`)
  are unchanged.

### Fixed

- **aarch64 / Apple Silicon precompile**: gate `llvm.x86.bmi.pdep` /
  `pext` ccalls with `@static if Sys.ARCH ∈ (:x86_64, :i686)` so the
  intrinsic is eliminated at parse time on non-x86 hosts.
- **`_find_coarsen_candidates` root-cell `BoundsError`**: the
  `parent <= 0` guard didn't catch `find_parent`'s `ROOT_PARENT ==
  typemax(UInt32)` sentinel. Now compares against `ROOT_PARENT`
  directly; `refine_by_indicator!` on a single-root mesh works.

### Documentation

- New: `refinement_events.md`, `neighbors_and_halos.md`,
  `boundary_conditions.md`, `amr_driver.md`, `exact_backend.md`.
  Updated: `architecture.md`, `getting_started.md`, `extending.md`,
  `overlap.md` to cover the additions above.

### Changed

- **r3djl pin bumped** from `ff23853f` to `154b346b` (2026-04-27).
  The two D = 2 upstream bugs HG had documented as known limitations
  (`_moments_exact_d2!` `0//0` throw on degenerate clips, and
  `clip!` producing negative-volume polytopes on refined-Eulerian
  geometry) are fully fixed. The 7 historical data-loss drops on
  the canonical two-triangle tile / 1-refined Eulerian geometry are
  gone.
- **IntExact-adapter drop classification** corrected to distinguish
  `vol == 0` (legitimate zero-area boundary intersection) from
  `vol < 0` (orientation-flipped clip). Two diagonal-corner pairs
  on the canonical two-triangle / 1-refined geometry — where the
  Lagrangian triangle's hypotenuse meets an Eulerian quadrant
  boundary at a single point — now classify as `:empty`, not
  `:negative_volume`.
- **`audit_overlap` and drop-audit tests updated** to act as
  regression detectors. They now assert the post-fix
  `n_negative_volume == 0` and `n_moments_throw == 0` on the
  canonical multi-tile / refined-Eulerian geometry; residual
  per-pair float-vs-exact differences on this geometry are
  sub-lattice-resolution (~1.5e-5 at `bits = 16`).

### Known limitations

- `D = 2` `bits ≥ 24` overflow in r3djl's shared-denominator
  storage remains; stay at `bits = 16` (the default) unless you've
  audited the geometry.

## [0.1.0] — 2026-04-26

Initial public release.

### Added

- **Layer 0 — BitPrimitives**: hardware bit operations (POPCNT, PDEP,
  PEXT) and dimension-keyed integer type selection.
- **Layer 1 — Mesh**: hierarchical mesh with anisotropic refinement,
  integer-exact relative coordinates, lazy parent/level/subtree
  caches; simplicial mesh for Lagrangian work; refinement by indicator;
  composite/paired mesh.
- **Layer 2 — Geometry**: integer-exact predicates, exact rational
  volumes under refinement, axis-aligned operations.
- **Layer 2 — Bases**: Bernstein-Bézier polynomial bases over simplices
  and boxes, conversions, positivity certificates.
- **Layer 2 — Quadrature**: Gauss-Legendre and simplex quadrature.
- **Layer 2.5 — Storage**: layout-flexible field storage (`SoA`, `AoS`,
  `Blocked{B}`); `PolynomialFieldSet` for per-cell polynomial
  coefficients.
- **Layer 3 — Threading**: chunk-based parallelism via OhMyThreads.jl,
  designed to mirror future MPI structure.
- **Layer 3 — Memory**: pool, scratch, and arena allocators for
  long-running simulations.
- **Layer 4 — Overlap**: Lagrangian↔Eulerian geometric overlap via
  r3djl, with sequential and parallel `compute_overlap`; CSR-like
  sparse data structure indexed in both directions; conservative
  polynomial remap (two-phase and streaming) in both L→E and E→L
  directions; multi-field remap.
- **Layer 4 — Diagnostics**: mesh diagnostics, conservation checks,
  Welford running statistics.
- **Examples**: two worked mini-apps demonstrating real applications
  on top of the framework — `examples/dsmc/` (Direct Simulation Monte
  Carlo for hard-sphere relaxation) and `examples/lagrangian_pixelization/`
  (2D Lagrangian↔Eulerian pixelization with adaptive refinement and
  density visualization, including an off-center Gaussian compression
  composed with a swirl).
- **10,872 framework tests** + 2,233 DSMC tests + 564 LagrangianPixelization
  tests = **13,669 tests** total, all passing.

### Dependencies

- [R3D](https://github.com/yipihey/r3djl) — pure-Julia port of
  devonmpowell/r3d, pinned via the top-level `Manifest.toml` to a
  GitHub commit. Until R3D is registered in the General registry,
  `Manifest.toml` is committed so `Pkg.instantiate()` resolves on a
  fresh clone.

[Unreleased]: https://github.com/yipihey/HierarchicalGrids.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yipihey/HierarchicalGrids.jl/releases/tag/v0.1.0
