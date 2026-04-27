# Exact-rational overlap backend

The Overlap layer ships two backends for the per-pair simplex-box clip
that powers `compute_overlap`:

- **Float** — `overlap_simplex_box!` over `Float64` vertex coordinates.
  The default; fast, well-tested, supports `D = 1, 2, 3`.
- **Exact** — `overlap_simplex_box_exact!` over integer-coordinate
  vertices and `Rational{R}` accumulators. Bit-exact, deterministic,
  unlocks `D = 4` with full polynomial moments, available via the
  `IntegerLattice` quantization helpers in this document.

This page covers the exact path: the lattice abstraction, the
quantization round-trip, the direct adapter, the D=4 capability, the
audit harness, and what to expect performance-wise.

## Why an exact backend

- **Bit-exact reproducibility.** `overlap_simplex_box_exact!` returns
  `Rational{R}` volumes, centroids, and moments. The same integer input
  gives the same integer output across machines, threads, build flags,
  and Julia versions — no `fastmath`, no `fma`, no platform-dependent
  reduction order.
- **Conservation on integer-lattice geometry.** When both the
  Lagrangian simplices and the Eulerian boxes live on a common
  `IntegerLattice`, the sum of per-pair overlap volumes equals the
  source-simplex volume *exactly* (as `Rational`), independent of the
  number of clipping events.
- **Robustness on degenerate orientations.** Coplanar faces, exact
  vertex coincidences, and near-tangent intersections are decided by
  integer comparisons rather than by float predicates. Inputs that the
  float path would handle with sliver-rejection heuristics terminate
  cleanly.
- **D=4 capability with full moments.** The exact path's
  pentachoron-vs-box clip and `R3D.IntExact.moments_exact!` give
  4-volumes and full polynomial moments today (unlocked by upstream
  r3djl commit `943135f1`, polynomial moments at D ∈ {4, 5, 6} via
  simplex decomposition). The float path still errors at `D ≥ 4` (its
  `r3djl` polynomial-moment recursion is not yet 4D-ready), so the
  exact backend remains the only D=4-capable path through HG.

## The lattice abstraction

`IntegerLattice{D, T}` is the equal-scale lattice that maps physical
`[lo, hi]` to integers `[0, 2^bits - 1]` along the longest physical
axis. The same `scale` is applied to every axis (anisotropic lattices
would complicate moment rescaling).

```julia
using HierarchicalGrids

# Construct from an existing EulerianFrame:
frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
lat   = IntegerLattice(frame)                # bits = 16, int_type = Int32

# Or directly from physical bounds:
lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 20, int_type = Int64)

@show lat.lo, lat.hi, lat.bits, lat.scale, lat.int_type
@show lat_resolution(lat)        # physical Δ between adjacent lattice points
```

Defaults: `bits = 16` (longest axis spans `65 535` integer steps),
`int_type = Int32`. The constructor enforces `2^bits - 1 ≤
typemax(int_type)` and rejects degenerate axes (`hi[d] ≤ lo[d]`).

The lattice origin is at `lat.lo`. Integer coordinate `p` along axis
`d` represents the physical point `lat.lo[d] + p / lat.scale`.

## Quantization round-trip

`quantize` snaps a float vertex to its nearest lattice integer;
`dequantize` is the inverse.

```julia
lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 16)

p   = quantize((0.25, 0.75), lat)    # ::NTuple{2, Int32}
v   = dequantize(p, lat)             # ::NTuple{2, Float64}, |v - input| ≤ Δ/2

# Strict variant — throws if the input isn't on the lattice within atol:
ps  = quantize_strict((0.25, 0.75), lat;
                       atol = lat_resolution(lat) / 4)
```

Round-trip error is bounded by `lat_resolution(lat) / 2` per axis.
`quantize` clamps out-of-range inputs to the integer range so callers
never see an `InexactError`; use `quantize_strict` when you expect the
input to already be lattice-aligned (e.g. synthesized from integer
geometry).

### Volumes back to physical units

`unscale_volume` is the 0th-moment converter. It is exact and
self-contained — the 0th moment is translation-invariant, so no offset
shift is needed:

```julia
using HierarchicalGrids
using HierarchicalGrids.Overlap: moments_length

lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 16, int_type = Int32)
scratch = IntPairScratch(Val(2), Int32; capacity = 32)
out_moments = Vector{Rational{Int128}}(undef, moments_length(2, 0))

simplex_int = (quantize((0.0, 0.0), lat),
               quantize((1.0, 0.0), lat),
               quantize((0.0, 1.0), lat))
box_lo = quantize((0.0, 0.0), lat)
box_hi = quantize((1.0, 1.0), lat)

vol_int, _, _ = overlap_simplex_box_exact!(out_moments, scratch,
                                            simplex_int, box_lo, box_hi, 0;
                                            accumulator = Int128)

physical_volume = unscale_volume(vol_int, lat)   # ≈ 0.5
```

### Higher moments and the offset-shift caveat

`unscale_moment(m, lat, k)` rescales a moment of total monomial
degree `k` by `(1/scale)^(D + k)`. **For `k ≥ 1` the result is in the
lattice's coordinate frame**, i.e. with origin at `lat.lo`, NOT in the
physical-frame coordinates. Composing into physical-frame moments
requires applying the offset shift `Δ = lat.lo` via `shift_moments!`:

```julia
using HierarchicalGrids
using HierarchicalGrids.Overlap: moments_length, moment_multiindices,
                                  shift_moments!

lat = IntegerLattice((0.0, 0.0), (1.0, 1.0); bits = 16, int_type = Int32)

P       = 2
D       = 2
nmom    = moments_length(D, P)
multi   = moment_multiindices(D, P)
scratch = IntPairScratch(Val(2), Int32; capacity = 32)
out_int = Vector{Rational{Int128}}(undef, nmom)

simplex_int = (quantize((0.0, 0.0), lat),
               quantize((1.0, 0.0), lat),
               quantize((0.0, 1.0), lat))
box_lo = quantize((0.0, 0.0), lat)
box_hi = quantize((1.0, 1.0), lat)

overlap_simplex_box_exact!(out_int, scratch, simplex_int, box_lo, box_hi, P;
                           accumulator = Int128)

# 1) Scale-only conversion: lattice-frame, physical units.
moments_lat = [unscale_moment(out_int[k], lat, sum(multi[k])) for k in 1:nmom]

# 2) Translate to the physical frame (origin at 0, not at lat.lo).
moments_phys = similar(moments_lat)
shift_moments!(moments_phys, moments_lat, D, P, lat.lo)
```

`unscale_volume(m, lat)` is identical to `unscale_moment(m, lat, 0)`
and skips the offset step (the 0th moment is offset-invariant). For
`k ≥ 1` always pair `unscale_moment` with `shift_moments!`.

## Direct adapter use

`overlap_simplex_box_exact!` clips one simplex against one
axis-aligned box and writes the volume, centroid, and full moment
vector up to `moment_order` into `out_moments`. The `IntPairScratch`
holds the persistent integer polytope and plane buffer; allocate one
per thread, reuse across many pairs.

### D = 2

```julia
using HierarchicalGrids
using HierarchicalGrids.Overlap: moments_length

P       = 2
nmom    = moments_length(2, P)
scratch = IntPairScratch(Val(2), Int32; capacity = 32)
out     = Vector{Rational{Int128}}(undef, nmom)

simplex = ((Int32(0), Int32(0)),
           (Int32(1024), Int32(0)),
           (Int32(0), Int32(1024)))
box_lo  = (Int32(0), Int32(0))
box_hi  = (Int32(1024), Int32(1024))

vol, centroid, _ = overlap_simplex_box_exact!(out, scratch,
                                               simplex, box_lo, box_hi, P;
                                               accumulator = Int128)
@assert vol == 1024 * 1024 // 2
```

### D = 3

```julia
P       = 1
nmom    = moments_length(3, P)
scratch = IntPairScratch(Val(3), Int32; capacity = 32)
out     = Vector{Rational{Int128}}(undef, nmom)

simplex = ((Int32(0),    Int32(0),    Int32(0)),
           (Int32(1024), Int32(0),    Int32(0)),
           (Int32(0),    Int32(1024), Int32(0)),
           (Int32(0),    Int32(0),    Int32(1024)))
box_lo  = (Int32(0),    Int32(0),    Int32(0))
box_hi  = (Int32(1024), Int32(1024), Int32(1024))

vol, centroid, _ = overlap_simplex_box_exact!(out, scratch,
                                               simplex, box_lo, box_hi, P;
                                               accumulator = Int128)
@assert vol == 1024^3 // 6
```

Empty / outside / degenerate overlaps return `vol = 0 // 1`, a
zero-tuple centroid, and zero-filled `out_moments`.

### Accumulator selection

The `accumulator` keyword is the integer width used by
`R3D.IntExact.moments_exact!` and the exact-volume fan triangulation.
The default comes from `_default_accumulator(T, Val(D))`, mirroring the
type-promotion guidance in `R3D.IntExact`'s docstring:

| input `T` | D ∈ {2, 3} | D = 4 |
|-----------|------------|-------|
| `Int16`   | `Int128`   | `BigInt` |
| `Int32`   | `Int128`   | `BigInt` |
| `Int64`   | `BigInt`   | `BigInt` |
| `Int128`  | `BigInt`   | `BigInt` |
| `BigInt`  | `BigInt`   | `BigInt` |

The element type of `out_moments` (a `Rational{R}` vector) must match
whichever accumulator you pass; the adapter throws `ArgumentError`
otherwise. `D = 4` always uses `BigInt` regardless of `T` because the
4×4 determinant in the fan triangulation overflows `Int128` even for
`T = Int16` after a few clips. Callers that know their geometry stays
axis-aligned can pass a tighter accumulator (`Int64` for `T = Int16`,
e.g.) — the default is conservative.

## D = 4

The exact backend supports `D = 4` with full polynomial moments:

```julia
P       = 1
nmom    = moments_length(4, P)  # 5 at P=1
scratch = IntPairScratch(Val(4), Int32; capacity = 64)
out     = Vector{Rational{BigInt}}(undef, nmom)

# Standard 4-simplex (pentachoron) at the origin.
simplex = ((Int32(0),    Int32(0),    Int32(0),    Int32(0)),
           (Int32(1024), Int32(0),    Int32(0),    Int32(0)),
           (Int32(0),    Int32(1024), Int32(0),    Int32(0)),
           (Int32(0),    Int32(0),    Int32(1024), Int32(0)),
           (Int32(0),    Int32(0),    Int32(0),    Int32(1024)))
box_lo  = (Int32(0),    Int32(0),    Int32(0),    Int32(0))
box_hi  = (Int32(1024), Int32(1024), Int32(1024), Int32(1024))

vol, centroid, _ = overlap_simplex_box_exact!(out, scratch,
                                                simplex, box_lo, box_hi, P;
                                                accumulator = BigInt)
@assert vol == big(1024)^4 // 24
@assert centroid == ntuple(_ -> big(1024) // 5, 4)
```

What works at D=4:

- `IntPairScratch{4, T}` allocation.
- `overlap_simplex_box_exact!(...; accumulator = BigInt)` with any
  `moment_order` ≥ 0. Polynomial moments are routed upstream through
  `R3D.IntExact.moments_exact!`'s `_moments_exact_dgeneric_4plus!`
  (landed in r3djl commit `943135f1`).
- Integer-exact 4-volume, centroid, and full second-and-higher moments
  in graded-lex order.

What does NOT work at D=4:

- The float path. `compute_overlap` and `overlap_simplex_box!` still
  error at `D ≥ 4`. D=4 is reachable today only through the exact
  adapter (the float r3djl path does not yet ship a D ≥ 4 moment
  recursion).

## Audit harness

`audit_overlap()` runs both backends side-by-side on a fixed canonical
polytope battery (five 2D triangles, four 3D tetrahedra) and returns
an `OverlapAuditReport`:

```julia
using HierarchicalGrids

report = audit_overlap()
# OverlapAuditReport(checked=9, passed=9, failed=0,
#                    max_vol_rel_diff=…, max_moment_rel_diff=…)

@show report.max_volume_relative_diff   # ~1.33e-16 on the canonical battery
@show report.max_moment_relative_diff
@assert report.n_failed == 0
```

`OverlapAuditReport` fields:

| Field | Meaning |
|-------|---------|
| `n_polytopes_checked` | Battery size. |
| `n_passed`, `n_failed` | Per-polytope pass/fail counts (`atol`-bounded). |
| `max_volume_relative_diff` | Largest float-vs-dequantized-exact volume diff. |
| `max_moment_relative_diff` | Largest float-vs-dequantized-exact moment diff (over all entries). |
| `failures` | Vector of NamedTuples for each failing polytope: `(polytope_name, dim, expected, got_float, got_exact)`. |

Interpreting `max_volume_relative_diff`: on the canonical battery the
floor sits at roughly `1.33e-16` (a few ULPs of `Float64`), reflecting
the rounding the float path picks up across plane construction,
clipping, and moment integration. Values an order of magnitude or
more above this floor on your geometry are a signal that either
input quantization is losing precision or the float path is hitting a
near-degenerate configuration the exact path is sailing through.

`audit_overlap` accepts:

- `verbose = true` — prints a per-polytope `[pass]` / `[FAIL]` summary
  as the audit runs.
- `atol = 1e-10` — relative-tolerance threshold for the per-entry
  comparison.

### Module-load consistency check

`Diagnostics.__init__()` calls `_verify_intexact_consistency()` once
when `HierarchicalGrids` finishes loading. This is the same battery
`audit_overlap()` runs; failure throws an `ErrorException` with the
report attached as a logged `@error`. The check is gated by the
`HG_INTEXACT_VERIFY` environment variable:

```sh
# Default — run the consistency check on every load.
julia --project=. -e 'using HierarchicalGrids'

# Skip the check (latency-sensitive context, CI image without R3D, etc.).
HG_INTEXACT_VERIFY=0 julia --project=. -e 'using HierarchicalGrids'
```

Any value other than `"1"` (e.g. `"0"`, `"no"`, `"off"`) skips the
check. The battery is small (9 polytopes); on a warm Julia process it
completes in well under 10 ms.

## Performance expectation

The exact path is meaningfully slower than the float path. Rough
guidance on warm caches with the default `Int128` accumulator at
`D ≤ 3`: expect a modest 2–4× slowdown per pair, varying with clip
count (every retained clip hits the rational arithmetic path) and
moment order (higher orders compound the per-pair `Rational`
allocations in `R3D.IntExact.moments_exact!`). At D=4 the `BigInt`
accumulator dominates and the gap widens further.

Don't take these numbers as a guarantee; benchmark on your geometry.
The exact path's value isn't speed — it's reproducibility and
robustness. Reach for it when you need bit-exact agreement, or when
you've isolated a degenerate-clip bug that the float path can't
resolve, or when you need a 4D volume.

Allocation profile: per-pair, the `IntPairScratch` itself is reused
(zero alloc on `clip!`); `R3D.IntExact.moments_exact!` still allocates
internally for its per-call polynomial scratch and the `Rational{R}`
output values. That is a property of the upstream sqrt-free recursion
in exact-rational form, not of this adapter.

## `compute_overlap(..., backend = :exact)`

`compute_overlap` accepts a `backend::Symbol` keyword (default
`:float`). Setting `backend = :exact` routes each per-pair clip
through `overlap_simplex_box_exact!` with an `IntegerLattice` and
accumulator chosen automatically:

```julia
overlap = compute_overlap(lag, frame; moment_order = 1, backend = :exact)
```

Optional kwargs `lattice::IntegerLattice` and `accumulator::Type{<:Signed}`
override the auto-derived defaults. The default lattice uses
`bits = 16` and `int_type = Int32` (or `Int64` for `D = 4`); the
default accumulator is promoted to `BigInt` when `moment_order ≥ 1`
since `Int128` overflows for many real-mesh clip configurations on
the bits=16 lattice. Pass `accumulator = Int128` explicitly to opt
into the smaller accumulator if your geometry stays within its
envelope.

The float and exact backends are organized identically at the
per-pair boundary, so the resulting `GeometricOverlap{D, Float64}` is
a drop-in for downstream consumers.

### Constraints

- `backend = :exact` requires `T == Float64` for both the
  `SimplicialMesh` and the `EulerianFrame`.
- `D = 1` is rejected; the float path's closed-form interval
  intersection is already exact, so use `backend = :float`.
- `D ∈ {2, 3, 4}` are supported with full polynomial moments. `D = 4`
  is reachable only through `:exact` (the float path errors at D ≥ 4).
- Periodic-ghost overlap (`frame_bcs` with periodic axes) is not yet
  supported under `backend = :exact`. Use `:float`, which handles
  periodic ghosts correctly — backend agreement stays at the
  machine-precision floor on all canonical test polytopes.

## Known limitations of the upstream IntExact path

The `D = 2` path in `R3D.IntExact` had two upstream bugs that HG
encountered as the first consumer; both are fully fixed upstream
as of r3djl commit `154b346` (2026-04-27):

1. **`R3D.IntExact._moments_exact_d2!` `0//0` throw on degenerate
   clips** — fully fixed upstream.

2. **`D = 2` `clip!` returning negative-volume polytopes on
   refined-Eulerian configurations** — fully fixed upstream. On the
   canonical "two-triangle tile / 1-refined Eulerian" geometry that
   exposed the bug most aggressively, all 7 historical data-loss
   drops are gone. Two of the original 7 (lag, eul) pairs now
   produce a clipped polygon with `vol == 0`: legitimate zero-area
   boundary intersections (the Lagrangian-triangle hypotenuse meets
   the Eulerian quadrant boundary at a single point). HG's adapter
   classifies these as `:empty`, not `:negative_volume`, so they
   are not data loss. Residual disagreements with the float backend
   on this geometry are sub-lattice-resolution (~1.5e-5 at
   `bits = 16`).

3. **`D = 2` storage overflow at high bit counts**: at lattice
   scales corresponding to `bits ≥ 24`, the polygon clip's
   shared-denominator integer storage can overflow `Int64` even with
   `accumulator = BigInt` — the storage type, not the accumulator,
   is the bound. Stay at `bits = 16` (the default) unless you have a
   specific reason to increase resolution and have audited the
   geometry.

The audit harness (`audit_overlap`) and the load-time
`_verify_intexact_consistency` check pass on the canonical polytope
battery. For applications using the exact backend on multi-simplex /
refined-Eulerian geometry, `audit_overlap(lag, frame; per_pair=true)`
or `compute_overlap(...; audit_drops=true)` give per-pair visibility
into any residual disagreements with the float backend.

## See also

- [Overlap and polynomial remap](overlap.md) — the float-path overview
  and the BVH/CSR organization that wraps both backends.
- [Architecture](architecture.md) — overall framework structure.
- `src/Overlap/quantize.jl` — `IntegerLattice`, `quantize`,
  `dequantize`, `unscale_volume`, `unscale_moment`.
- `src/Overlap/r3d_int_adapter.jl` — `IntPairScratch`,
  `overlap_simplex_box_exact!`, `_default_accumulator`.
- `src/Diagnostics/exact_audit.jl` — `OverlapAuditReport`,
  `audit_overlap`, `_verify_intexact_consistency`.
- [r3djl's `IntExact` submodule](https://github.com/yipihey/r3djl) —
  the underlying integer-exact polytope-clipping library.
