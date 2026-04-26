# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
