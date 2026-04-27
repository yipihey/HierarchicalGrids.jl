"""
IntExact audit harness — diagnostic comparison between the float and
integer-exact polytope-clipping backends in `src/Overlap/`.

Two consumers:

  1. **Module-load consistency check** (`_verify_intexact_consistency`)
     runs once when HierarchicalGrids is loaded (analogous to
     `_verify_moment_ordering` in `r3d_adapter.jl`). Bails with an error
     if the upstream IntExact backend has drifted.
  2. **User-callable diagnostic** (`audit_overlap`) returns a structured
     report and is intended for manual inspection.

The audit runs both `overlap_simplex_box!` (Float64 path) and
`overlap_simplex_box_exact!` (Int + Rational path) on a small fixed
battery of canonical 2D triangles and 3D tetrahedra. Each polytope has
a known analytic volume; the audit compares the float and exact answers
against each other and (where applicable) against the analytic value.

The two backends are compared on a shared integer lattice with scale
`SCALE = 2^10 = 1024`. The float path operates in physical units
`[0, 1]`; the integer path operates on lattice integers `[0, SCALE]`.
The integer volume is in lattice units `int_vol`; the corresponding
physical-unit volume is `Float64(int_vol) / SCALE^D`. Higher-order
moments scale by `SCALE^(D + |α|)`.

# Reference dependencies

This file lives in the Diagnostics module but its `audit_overlap` /
`_verify_intexact_consistency` functions reach across to the Overlap
submodule for `overlap_simplex_box!`, `overlap_simplex_box_exact!`,
`PairScratch`, `IntPairScratch`, and `moments_length`. The reference is
made via the parent-module path `HierarchicalGrids.Overlap` resolved at
call time (function bodies are late-bound), so this file safely loads
even though Diagnostics is included before Overlap in
`src/HierarchicalGrids.jl`. The module-load consistency check is
triggered from `Diagnostics.__init__`, which runs after the entire
HierarchicalGrids package finishes loading.
"""

# ============================================================================
# Report struct
# ============================================================================

"""
    OverlapAuditReport

Side-by-side comparison of the float and integer-exact polytope-clipping
backends on a fixed set of canonical polytopes. Returned by
[`audit_overlap`](@ref).

# Fields

- `n_polytopes_checked` — number of polytopes in the battery.
- `n_passed`, `n_failed` — pass/fail counts (a polytope passes if every
  comparison — volume, centroid, every moment — is within `atol`).
- `max_volume_relative_diff` — the largest relative difference observed
  between the float volume and the dequantized exact volume.
- `max_moment_relative_diff` — the largest relative difference observed
  between any pair of corresponding moment entries.
- `failures` — vector of NamedTuples summarizing each failing polytope.
  Each entry has fields
  `(polytope_name, dim, expected, got_float, got_exact)`.
"""
struct OverlapAuditReport
    n_polytopes_checked::Int
    n_passed::Int
    n_failed::Int
    max_volume_relative_diff::Float64
    max_moment_relative_diff::Float64
    failures::Vector{NamedTuple}
end

function Base.show(io::IO, r::OverlapAuditReport)
    print(io, "OverlapAuditReport(")
    print(io, "checked=", r.n_polytopes_checked)
    print(io, ", passed=", r.n_passed)
    print(io, ", failed=", r.n_failed)
    print(io, ", max_vol_rel_diff=", r.max_volume_relative_diff)
    print(io, ", max_moment_rel_diff=", r.max_moment_relative_diff, ")")
    if r.n_failed > 0
        for f in r.failures
            # Per-pair audits (Item B) emit a different schema than the
            # canonical-polytope path; render whichever fields exist.
            fnames = propertynames(f)
            if :polytope_name in fnames
                print(io, "\n  - ", f.polytope_name,
                      " (D=", f.dim, "): expected=", f.expected,
                      ", float=", f.got_float, ", exact=", f.got_exact)
            elseif :lag_idx in fnames
                # User-mesh per-pair entry.
                print(io, "\n  - lag=", f.lag_idx,
                      " eul=", f.eul_idx,
                      "  kind=", f.kind,
                      "  vol_diff=", f.volume_diff,
                      "  centroid_diff=", f.centroid_diff)
            else
                # Fallback: print whatever fields are present.
                print(io, "\n  - ", f)
            end
        end
    end
end

# ============================================================================
# OverlapDropReport — :exact-backend drop audit
# ============================================================================

"""
    OverlapDropReport

Records pairs that the `:exact` backend dropped on the way through
[`compute_overlap`](@ref). Drops have three sources, all upstream:

1. `R3D.IntExact.clip!` produces (or `R3D.IntExact.moments_exact!`
   integrates) a polytope with a non-positive volume — orientation
   flip, known upstream bug at `D = 2`. The HG adapter zeros the
   moments and the entry is suppressed; this is data loss.
2. `R3D.IntExact.moments_exact!` throws on a degenerate polytope (e.g.
   shared denominators of zero) — known upstream bug at `D = 2`.
   Surfaced via try/catch in `compute_overlap`'s exact path; this is
   also data loss.
3. The clip is empty (no overlap) — not a bug, but counted for
   completeness so callers can sanity-check broad-phase pruning.

Each entry is `(lag_idx::Int32, eul_idx::Int32, kind::Symbol)` where
`kind ∈ (:negative_volume, :moments_throw, :empty)`.

`n_negative_volume` and `n_moments_throw` together count actual data
loss; `n_empty` is informational.

# Fields

- `n_negative_volume::Int` — count of pairs dropped because upstream
  produced a non-positive `vol_rat` for a non-empty clip polytope.
- `n_moments_throw::Int` — count of pairs dropped because
  `R3D.IntExact.moments_exact!` raised an exception on a degenerate
  clip.
- `n_empty::Int` — count of pairs whose clip was geometrically empty.
- `drops::Vector{NamedTuple}` — per-pair records, in encounter order.

See also [`compute_overlap`](@ref) (with `audit_drops = true`),
[`audit_overlap`](@ref) (the per-mesh user audit, also reports drops).
"""
struct OverlapDropReport
    n_negative_volume::Int
    n_moments_throw::Int
    n_empty::Int
    drops::Vector{NamedTuple{(:lag_idx, :eul_idx, :kind),
                             Tuple{Int32, Int32, Symbol}}}
end

function Base.show(io::IO, r::OverlapDropReport)
    n_loss = r.n_negative_volume + r.n_moments_throw
    if n_loss == 0
        print(io, "OverlapDropReport: 0 data-loss drops",
                  " (", r.n_empty, " empty)")
        return
    end
    print(io, "OverlapDropReport: ", n_loss, " drops (",
              r.n_negative_volume, " negative-volume, ",
              r.n_moments_throw, " moments-throw, ",
              r.n_empty, " empty)")
    # List the first few data-loss drops.
    loss_drops = NamedTuple[]
    for d in r.drops
        if d.kind === :negative_volume || d.kind === :moments_throw
            push!(loss_drops, d)
            length(loss_drops) >= 5 && break
        end
    end
    if !isempty(loss_drops)
        n_show = length(loss_drops)
        print(io, "\n  first ", n_show, " data-loss drops:")
        for d in loss_drops
            print(io, "\n    lag=", d.lag_idx, " eul=", d.eul_idx,
                  "  :", d.kind)
        end
        if n_loss > n_show
            print(io, "\n    ... (", n_loss - n_show, " more)")
        end
    end
end

# ============================================================================
# Canonical polytope battery
# ============================================================================

# Lattice scale used to embed the float-domain canonical polytopes into
# integer coordinates. 2^10 = 1024 keeps every product comfortably inside
# Int64 even at moment_order = 2 in 3D (max coordinate^4 = 2^40).
const _AUDIT_LATTICE_BITS = 10
const _AUDIT_SCALE = 1 << _AUDIT_LATTICE_BITS  # 1024

# Each entry is a NamedTuple with fields:
#   name             :: String
#   dim              :: Int
#   vertices_float   :: tuple of NTuple{D, Float64}
#   vertices_int     :: tuple of NTuple{D, Int64}
#   box_float        :: (NTuple{D, Float64}, NTuple{D, Float64})  -- (lo, hi)
#   box_int          :: (NTuple{D, Int64}, NTuple{D, Int64})
#   expected_volume  :: Float64   -- analytic physical-unit volume of the
#                                    overlap (0.0 ⇒ empty case)

function _canonical_polytopes_2d()
    s = _AUDIT_SCALE
    sf = Float64(s)
    out = NamedTuple[]

    # 1. Unit triangle inside [0, 1]^2.  Overlap = the triangle itself.
    push!(out, (
        name = "unit_triangle",
        dim = 2,
        vertices_float = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        vertices_int = ((Int64(0), Int64(0)), (Int64(s), Int64(0)),
                        (Int64(0), Int64(s))),
        box_float = ((0.0, 0.0), (1.0, 1.0)),
        box_int = ((Int64(0), Int64(0)), (Int64(s), Int64(s))),
        expected_volume = 0.5,
    ))

    # 2. Off-axis triangle entirely inside the box. Vertices chosen to
    # have multiples of 1/4 so the integer encoding is exact.
    # Triangle (1/4, 1/4)-(3/4, 1/4)-(1/4, 3/4): area = 0.5 * 0.5 * 0.5 = 0.125
    push!(out, (
        name = "off_axis_triangle",
        dim = 2,
        vertices_float = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75)),
        vertices_int = ((Int64(s ÷ 4), Int64(s ÷ 4)),
                        (Int64(3 * s ÷ 4), Int64(s ÷ 4)),
                        (Int64(s ÷ 4), Int64(3 * s ÷ 4))),
        box_float = ((0.0, 0.0), (1.0, 1.0)),
        box_int = ((Int64(0), Int64(0)), (Int64(s), Int64(s))),
        expected_volume = 0.125,
    ))

    # 3. Partial overlap: triangle (0,0)-(1,0)-(0,1) clipped by box
    # [0, 1/2]^2.  The clipped region is the smaller triangle
    # (0, 0)-(1/2, 0)-(0, 1/2) ∪ ... actually it's the triangle below
    # the line x + y ≤ 1 intersected with [0, 1/2]^2 — the entire
    # square is below x + y = 1, so the clipped region IS the square.
    # Wait, x + y ≤ 1 is satisfied everywhere in [0, 1/2]^2 since
    # max(x+y) = 1. So the overlap is the full quarter-box, area = 1/4.
    push!(out, (
        name = "triangle_clipped_by_quarter_box",
        dim = 2,
        vertices_float = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        vertices_int = ((Int64(0), Int64(0)), (Int64(s), Int64(0)),
                        (Int64(0), Int64(s))),
        box_float = ((0.0, 0.0), (0.5, 0.5)),
        box_int = ((Int64(0), Int64(0)), (Int64(s ÷ 2), Int64(s ÷ 2))),
        expected_volume = 0.25,
    ))

    # 4. Partial overlap with non-trivial intersection: triangle
    # (0, 0)-(1, 0)-(0, 1) clipped by box [1/4, 3/4] x [0, 1/2].
    # The triangle interior is x + y ≤ 1, x, y ≥ 0. Intersect with the
    # box. In the box [1/4, 3/4] x [0, 1/2], the constraint x + y ≤ 1
    # is binding only where x + y > 1, i.e. x > 1 - y. With y ≤ 1/2 we
    # have 1 - y ≥ 1/2, so x = 3/4 (the right edge) intersects x + y = 1
    # at y = 1/4. Region:
    #   - rectangle [1/4, 3/4] x [0, 1/4]                  area = 1/2 * 1/4 = 1/8
    #   - trapezoid [1/4, 3/4] x [1/4, 1/2] minus triangle above x+y=1
    #     trapezoid area = 1/2 * 1/4 = 1/8
    #     above-the-line triangle has corners (3/4, 1/4), (3/4, 1/2), (1/2, 1/2)
    #       area = 1/2 * 1/4 * 1/4 = 1/32
    # Total: 1/8 + 1/8 - 1/32 = 8/32 - 1/32 = 7/32.
    push!(out, (
        name = "triangle_partial_overlap",
        dim = 2,
        vertices_float = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        vertices_int = ((Int64(0), Int64(0)), (Int64(s), Int64(0)),
                        (Int64(0), Int64(s))),
        box_float = ((0.25, 0.0), (0.75, 0.5)),
        box_int = ((Int64(s ÷ 4), Int64(0)),
                   (Int64(3 * s ÷ 4), Int64(s ÷ 2))),
        expected_volume = 7 / 32,
    ))

    # 5. Empty case: triangle entirely outside the box.
    push!(out, (
        name = "empty_2d",
        dim = 2,
        vertices_float = ((2.0, 2.0), (3.0, 2.0), (2.0, 3.0)),
        vertices_int = ((Int64(2 * s), Int64(2 * s)),
                        (Int64(3 * s), Int64(2 * s)),
                        (Int64(2 * s), Int64(3 * s))),
        box_float = ((0.0, 0.0), (1.0, 1.0)),
        box_int = ((Int64(0), Int64(0)), (Int64(s), Int64(s))),
        expected_volume = 0.0,
    ))

    return out
end

function _canonical_polytopes_3d()
    s = _AUDIT_SCALE
    out = NamedTuple[]

    # 1. Unit tetrahedron (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1) inside [0,1]^3.
    # Volume = 1/6.
    push!(out, (
        name = "unit_tetrahedron",
        dim = 3,
        vertices_float = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        vertices_int = ((Int64(0), Int64(0), Int64(0)),
                        (Int64(s), Int64(0), Int64(0)),
                        (Int64(0), Int64(s), Int64(0)),
                        (Int64(0), Int64(0), Int64(s))),
        box_float = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        box_int = ((Int64(0), Int64(0), Int64(0)),
                   (Int64(s), Int64(s), Int64(s))),
        expected_volume = 1 / 6,
    ))

    # 2. Kuhn-cube-tet: (0,0,0)-(1,0,0)-(1,1,0)-(1,1,1). One of the six
    # tetrahedra in the canonical Kuhn decomposition; volume = 1/6.
    push!(out, (
        name = "kuhn_tetrahedron",
        dim = 3,
        vertices_float = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)),
        vertices_int = ((Int64(0), Int64(0), Int64(0)),
                        (Int64(s), Int64(0), Int64(0)),
                        (Int64(s), Int64(s), Int64(0)),
                        (Int64(s), Int64(s), Int64(s))),
        box_float = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        box_int = ((Int64(0), Int64(0), Int64(0)),
                   (Int64(s), Int64(s), Int64(s))),
        expected_volume = 1 / 6,
    ))

    # 3. Partial overlap: unit tet clipped by [0, 1/2]^3. Inside the
    # quarter-box, the constraint x+y+z ≤ 1 is satisfied everywhere
    # (max sum = 3/2, so it's binding). Need a careful integration.
    # In the box [0, 1/2]^3, the constraint x+y+z ≤ 1 cuts off the
    # corner where x = y = z = 1/2 (sum = 3/2). The kept region:
    #   V = ∫∫∫_{[0,1/2]^3 ∩ x+y+z ≤ 1} 1
    # By symmetry, this is the box volume minus the tet
    # x+y+z ≥ 1, all coords in [0, 1/2]. That tet has corners at
    # (1/2,0,1/2), (1/2,1/2,0), (0,1/2,1/2), (1/2,1/2,1/2) — wait, which
    # corners satisfy x+y+z ≥ 1? Only (1/2, 1/2, 1/2) gives sum 3/2 > 1.
    # The cut x+y+z = 1 enters the box at (1/2, 1/2, 0), (1/2, 0, 1/2),
    # (0, 1/2, 1/2). The cut region above is a tet with apex
    # (1/2, 1/2, 1/2) and base triangle in plane x+y+z=1; legs of length
    # 1/2 along each axis. Volume = (1/6) * (1/2)^3 = 1/48.
    # So overlap = box_vol - 1/48 = 1/8 - 1/48 = 6/48 - 1/48 = 5/48.
    push!(out, (
        name = "tet_partial_overlap",
        dim = 3,
        vertices_float = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        vertices_int = ((Int64(0), Int64(0), Int64(0)),
                        (Int64(s), Int64(0), Int64(0)),
                        (Int64(0), Int64(s), Int64(0)),
                        (Int64(0), Int64(0), Int64(s))),
        box_float = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        box_int = ((Int64(0), Int64(0), Int64(0)),
                   (Int64(s ÷ 2), Int64(s ÷ 2), Int64(s ÷ 2))),
        expected_volume = 5 / 48,
    ))

    # 4. Empty 3D: tet entirely outside the box.
    push!(out, (
        name = "empty_3d",
        dim = 3,
        vertices_float = ((2.0, 2.0, 2.0), (3.0, 2.0, 2.0),
                          (2.0, 3.0, 2.0), (2.0, 2.0, 3.0)),
        vertices_int = ((Int64(2 * s), Int64(2 * s), Int64(2 * s)),
                        (Int64(3 * s), Int64(2 * s), Int64(2 * s)),
                        (Int64(2 * s), Int64(3 * s), Int64(2 * s)),
                        (Int64(2 * s), Int64(2 * s), Int64(3 * s))),
        box_float = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        box_int = ((Int64(0), Int64(0), Int64(0)),
                   (Int64(s), Int64(s), Int64(s))),
        expected_volume = 0.0,
    ))

    return out
end

# ============================================================================
# Audit core
# ============================================================================

# Relative-diff helper: |a - b| / max(|a|, |b|, eps), guarded for the
# empty case (both 0 ⇒ 0 diff).
@inline function _rel_diff(a::Real, b::Real)
    den = max(abs(a), abs(b), eps(Float64))
    return abs(a - b) / den
end

# Run a single polytope through both backends and compare. Returns
# `(passed::Bool, vol_rel::Float64, max_moment_rel::Float64, got_float, got_exact)`.
function _audit_one(pt::NamedTuple, atol::Real)
    # Pull Overlap symbols at call time — Diagnostics is loaded before
    # Overlap, but `__init__` deferral makes these resolvable.
    Olap = parentmodule(@__MODULE__).Overlap

    D = pt.dim
    moment_order = 2  # cover volume, centroid, second moments
    nmom = Olap.moments_length(D, moment_order)

    # ------------ Float path ------------
    if D == 2
        scratch_f = Olap.PairScratch(Val(2), Float64; capacity = 32)
    else
        scratch_f = Olap.PairScratch(Val(3), Float64; capacity = 32)
    end
    m_float = Vector{Float64}(undef, nmom)
    vol_f, cent_f, _ = Olap.overlap_simplex_box!(
        m_float, scratch_f, pt.vertices_float,
        pt.box_float[1], pt.box_float[2], moment_order)

    # ------------ Integer path ------------
    Rint = Int128
    if D == 2
        scratch_i = Olap.IntPairScratch(Val(2), Int64; capacity = 32)
    else
        scratch_i = Olap.IntPairScratch(Val(3), Int64; capacity = 32)
    end
    m_int = Vector{Rational{Rint}}(undef, nmom)
    vol_i, cent_i, _ = Olap.overlap_simplex_box_exact!(
        m_int, scratch_i, pt.vertices_int,
        pt.box_int[1], pt.box_int[2], moment_order;
        accumulator = Rint)

    # Dequantize: integer volume is in lattice^D units; physical = / scale^D.
    # For each moment of multi-index α, scale by SCALE^(D + sum(α)).
    scale = Float64(_AUDIT_SCALE)
    scaleD = scale^D

    vol_int_phys = Float64(vol_i) / scaleD
    vol_rel = _rel_diff(vol_f, vol_int_phys)

    # Compare moments index-by-index.
    multi = Olap.moment_multiindices(D, moment_order)
    max_mom_rel = 0.0
    for k in eachindex(m_float, m_int)
        αsum = sum(multi[k])
        m_int_phys = Float64(m_int[k]) / scale^(D + αsum)
        rel = _rel_diff(m_float[k], m_int_phys)
        max_mom_rel = max(max_mom_rel, rel)
    end

    # Pass condition: both vol_rel and max_mom_rel within atol; AND the
    # exact (or float, if the integer path returns 0) volume matches the
    # analytic expected_volume within atol.
    expected = Float64(pt.expected_volume)
    expected_rel = _rel_diff(vol_int_phys, expected)
    passed = (vol_rel <= atol) && (max_mom_rel <= atol) && (expected_rel <= atol)

    got_float = (volume = vol_f, centroid = cent_f)
    got_exact = (volume = vol_int_phys,
                 centroid = ntuple(d -> Float64(cent_i[d]) / scale, D))

    return (passed, vol_rel, max_mom_rel, expected, got_float, got_exact)
end

"""
    audit_overlap(; verbose::Bool = false, atol::Real = 1e-10) -> OverlapAuditReport

Run the canonical-polytope battery against both float
(`overlap_simplex_box!`) and integer-exact (`overlap_simplex_box_exact!`)
backends and return an [`OverlapAuditReport`](@ref).

# Battery

The battery covers (see `_canonical_polytopes_2d` /
`_canonical_polytopes_3d` in this file for definitions):

  * **D = 2**: unit triangle, off-axis triangle, two partial-overlap
    cases (quarter-box clip and asymmetric-box clip), empty case.
  * **D = 3**: unit tetrahedron, Kuhn-cube tetrahedron, partial-overlap
    tet (quarter-box clip), empty case.

# Method

For each polytope, the audit:

  1. Computes float volume, centroid, and moments via `overlap_simplex_box!`.
  2. Embeds the polytope on a lattice of scale `SCALE = 2^10` and
     computes exact volume, centroid, and moments via
     `overlap_simplex_box_exact!`. The exact integer answer is
     dequantized by dividing volume by `SCALE^D` and each moment of
     multi-index α by `SCALE^(D + |α|)`.
  3. Compares float vs. dequantized-exact entry-by-entry using
     `|a - b| / max(|a|, |b|, eps)` against the `atol` tolerance
     (default `1e-10`).
  4. Records mismatches in the report's `failures` field.

# Arguments

- `verbose=true` prints a per-polytope summary as the audit runs.
- `atol` is the relative-tolerance threshold for declaring a mismatch.

# Returns

The `OverlapAuditReport`. The maximum observed relative differences
(`max_volume_relative_diff`, `max_moment_relative_diff`) are useful
even when nothing fails: they tell downstream PRs (PR-3, PR-5) what
realistic floor the float-vs-exact comparison sits at on this geometry.
"""
# Per-polytope worker. Returns a "partial" report (with at most one
# polytope counted) so per-task partials can be merged via
# [`_merge_audit_reports`](@ref).
function _audit_one_partial(pt::NamedTuple, atol::Real, verbose::Bool)
    passed, vol_rel, mom_rel, expected, got_float, got_exact =
        _audit_one(pt, atol)
    if passed
        verbose && println("  [pass] ", pt.name,
                            "  vol_rel=", vol_rel,
                            "  mom_rel=", mom_rel)
        return OverlapAuditReport(1, 1, 0, vol_rel, mom_rel, NamedTuple[])
    else
        verbose && println("  [FAIL] ", pt.name,
                            "  vol_rel=", vol_rel,
                            "  mom_rel=", mom_rel)
        f = (polytope_name = pt.name,
             dim = pt.dim,
             expected = expected,
             got_float = got_float,
             got_exact = got_exact)
        return OverlapAuditReport(1, 0, 1, vol_rel, mom_rel, NamedTuple[f])
    end
end

# Associative merge for two partial OverlapAuditReports. Used as the
# reducer in `parallel_mapreduce`. Since the canonical-polytope audit
# does not cap `failures`, the merge simply concatenates them; the
# user-mesh audit caps post-hoc.
function _merge_audit_reports(a::OverlapAuditReport, b::OverlapAuditReport)
    return OverlapAuditReport(
        a.n_polytopes_checked + b.n_polytopes_checked,
        a.n_passed + b.n_passed,
        a.n_failed + b.n_failed,
        max(a.max_volume_relative_diff, b.max_volume_relative_diff),
        max(a.max_moment_relative_diff, b.max_moment_relative_diff),
        vcat(a.failures, b.failures),
    )
end

function audit_overlap(; verbose::Bool = false, atol::Real = 1e-10,
                         backend = nothing)
    polytopes = vcat(_canonical_polytopes_2d(), _canonical_polytopes_3d())

    if isempty(polytopes)
        return OverlapAuditReport(0, 0, 0, 0.0, 0.0, NamedTuple[])
    end

    HG = parentmodule(@__MODULE__)
    eff_backend = backend === nothing ? HG.Threading.default_backend() : backend

    init = OverlapAuditReport(0, 0, 0, 0.0, 0.0, NamedTuple[])

    # parallel_mapreduce: each polytope produces a unit partial report,
    # then `_merge_audit_reports` combines them associatively. Verbose
    # printing is sequential under `Sequential()` and best-effort under
    # parallel backends (interleaved per-task output is acceptable for a
    # diagnostic).
    return HG.Threading.parallel_mapreduce(
        eff_backend,
        pt -> _audit_one_partial(pt, atol, verbose),
        _merge_audit_reports,
        polytopes;
        init = init,
    )
end

# ============================================================================
# User-mesh per-pair audit — Item B
# ============================================================================

"""
    audit_overlap(lag::SimplicialMesh{D, Float64},
                  frame::EulerianFrame{D, Float64};
                  bits::Int = 16,
                  accumulator::Type{<:Signed} = Int128,
                  per_pair::Bool = true,
                  max_pair_diffs::Int = 32,
                  atol::Real = 1e-10) -> OverlapAuditReport

Run BOTH the `:float` and `:exact` backends on the user's actual mesh
and report agreement. Useful for verifying the exact backend on a
specific geometry before relying on it for production work — the
canonical-polytope audit (`audit_overlap()` with no arguments) covers
clean test polytopes, but real meshes can trip upstream `IntExact`
edge cases that the canonical battery doesn't cover.

Returns the same [`OverlapAuditReport`](@ref) struct used by the
canonical-polytope audit, with `n_polytopes_checked` interpreted as
the number of `(lag_idx, eul_idx)` pairs compared (the union of the
keys produced by either backend). The `failures` vector schema for
this overload is

    NamedTuple{(:lag_idx, :eul_idx, :kind, :volume_diff, :centroid_diff,
                :got_float, :got_exact)}

where `kind ∈ (:exact_dropped, :volume_mismatch, :centroid_mismatch)`.
The list is sorted by descending `volume_diff` and capped at
`max_pair_diffs` entries.

# Arguments

- `lag` — Lagrangian mesh.
- `frame` — Eulerian frame.
- `bits` — lattice bit budget for the exact backend (default `16`).
- `accumulator` — exact-rational accumulator type (default `Int128`;
  pass `BigInt` for guaranteed safety on heavily refined meshes).
- `per_pair` — when `true` (default), populate the `failures` field
  with sorted per-pair disagreements; when `false`, only the summary
  counts are reported (failures left empty).
- `max_pair_diffs` — cap on the number of per-pair entries in the
  report's `failures` field (default `32`).
- `atol` — absolute tolerance for declaring a per-pair disagreement.
  A pair is flagged when the volume difference or any centroid
  component difference exceeds `atol` (default `1e-10`).

# Drop surfacing

When the `:exact` backend drops pairs (upstream `IntExact` known bugs
at `D = 2` — see [`OverlapDropReport`](@ref)), those pairs appear in
`failures` with `kind = :exact_dropped`. The `:exact` total volume
report is then strictly less than `:float`'s and the audit's
`max_volume_relative_diff` is bounded below by the dropped volume.
"""
# Per-key audit. Returns a partial OverlapAuditReport (with at most one
# pair counted, exactly one failure entry if the pair disagreed). Used as
# the map function in `parallel_mapreduce` for the user-mesh audit.
function _audit_pair_partial(k::Tuple{Int32, Int32},
                              f_by_key::Dict, e_by_key::Dict, D::Int,
                              atol::Real)
    ef = get(f_by_key, k, nothing)
    ee = get(e_by_key, k, nothing)
    if ef === nothing && ee !== nothing
        vol = Float64(ee.volume)
        rel = _rel_diff(0.0, vol)
        f = (lag_idx = Int(k[1]), eul_idx = Int(k[2]),
             kind = :float_missing,
             volume_diff = vol,
             centroid_diff = 0.0,
             got_float = nothing,
             got_exact = (volume = ee.volume, centroid = ee.centroid))
        return OverlapAuditReport(1, 0, 1, rel, 0.0, NamedTuple[f])
    elseif ef !== nothing && ee === nothing
        vol = Float64(ef.volume)
        rel = _rel_diff(vol, 0.0)
        f = (lag_idx = Int(k[1]), eul_idx = Int(k[2]),
             kind = :exact_dropped,
             volume_diff = vol,
             centroid_diff = 0.0,
             got_float = (volume = ef.volume, centroid = ef.centroid),
             got_exact = nothing)
        return OverlapAuditReport(1, 0, 1, rel, 0.0, NamedTuple[f])
    else
        vol_diff = abs(Float64(ef.volume) - Float64(ee.volume))
        cent_diff = 0.0
        for d in 1:D
            cent_diff = max(cent_diff,
                            abs(Float64(ef.centroid[d]) -
                                Float64(ee.centroid[d])))
        end
        vol_rel = _rel_diff(Float64(ef.volume), Float64(ee.volume))
        if vol_diff > atol || cent_diff > atol
            kind = vol_diff > atol ? :volume_mismatch : :centroid_mismatch
            f = (lag_idx = Int(k[1]), eul_idx = Int(k[2]),
                 kind = kind,
                 volume_diff = vol_diff,
                 centroid_diff = cent_diff,
                 got_float = (volume = ef.volume, centroid = ef.centroid),
                 got_exact = (volume = ee.volume, centroid = ee.centroid))
            return OverlapAuditReport(1, 0, 1, vol_rel, vol_rel, NamedTuple[f])
        else
            return OverlapAuditReport(1, 1, 0, vol_rel, vol_rel, NamedTuple[])
        end
    end
end

function audit_overlap(lag, frame;
                       bits::Int = 16,
                       accumulator::Type{<:Signed} = Int128,
                       per_pair::Bool = true,
                       max_pair_diffs::Int = 32,
                       atol::Real = 1e-10,
                       backend = nothing)
    # Pull Overlap symbols at call time — Diagnostics is loaded BEFORE
    # Overlap, but `__init__` deferral makes these resolvable.
    HG = parentmodule(@__MODULE__)
    Olap = HG.Overlap

    # Type-check the inputs at the call boundary so the error message is
    # clear (an alternative would be to constrain via `where {D}` on the
    # method signature, but the canonical-polytope `audit_overlap()`
    # method shares the same name and dispatches on `()` — the
    # mesh-typed method has to take untyped args + an explicit check).
    lag isa Olap.SimplicialMesh ||
        throw(ArgumentError(
            "audit_overlap(lag, frame): `lag` must be a SimplicialMesh, " *
            "got $(typeof(lag))"))
    frame isa Olap.EulerianFrame ||
        throw(ArgumentError(
            "audit_overlap(lag, frame): `frame` must be an EulerianFrame, " *
            "got $(typeof(frame))"))

    # Build a lattice on the requested bit budget.
    D = Olap.spatial_dimension(frame)
    lat = Olap.IntegerLattice(frame; bits = bits,
                              int_type = (D == 4 ? Int64 : Int32))

    # Run both backends. moment_order = 1 covers volume + centroid; the
    # per-pair audit doesn't need higher moments.
    o_float = Olap.compute_overlap(lag, frame;
                                    backend = :float,
                                    moment_order = 1)
    o_exact, drops = Olap.compute_overlap(lag, frame;
                                           backend = :exact,
                                           moment_order = 1,
                                           lattice = lat,
                                           accumulator = accumulator,
                                           audit_drops = true)

    # Index entries by (lag_idx, eul_idx).
    f_by_key = Dict{Tuple{Int32, Int32}, Any}()
    for e in o_float.entries
        f_by_key[(e.lag_idx, e.eul_idx)] = e
    end
    e_by_key = Dict{Tuple{Int32, Int32}, Any}()
    for e in o_exact.entries
        e_by_key[(e.lag_idx, e.eul_idx)] = e
    end

    all_keys = union(keys(f_by_key), keys(e_by_key))
    # Sort to a deterministic order so the parallel reduction is associative
    # over a stable sequence and Sequential-vs-parallel results are
    # byte-identical (tasks may interleave, but the merge of partial
    # reports is associative on counts/maxes; the failure list is
    # canonicalized via sort below).
    keys_vec = sort!(collect(all_keys))
    n_total = length(keys_vec)

    if n_total == 0
        return OverlapAuditReport(0, 0, 0, 0.0, 0.0, NamedTuple[])
    end

    eff_backend = backend === nothing ? HG.Threading.default_backend() : backend

    init = OverlapAuditReport(0, 0, 0, 0.0, 0.0, NamedTuple[])

    report = HG.Threading.parallel_mapreduce(
        eff_backend,
        k -> _audit_pair_partial(k, f_by_key, e_by_key, D, atol),
        _merge_audit_reports,
        keys_vec;
        init = init,
    )

    # Sort failures by descending volume_diff and cap. The secondary sort
    # key is `(lag_idx, eul_idx)` so the result is deterministic across
    # backends even when several pairs tie on `volume_diff` (which is
    # common — many `:exact_dropped` pairs land on the same lattice
    # quantum).
    pair_failures = report.failures
    if per_pair
        sort!(pair_failures,
              by = f -> (-f.volume_diff, f.lag_idx, f.eul_idx))
        if length(pair_failures) > max_pair_diffs
            resize!(pair_failures, max_pair_diffs)
        end
    else
        empty!(pair_failures)
    end

    return OverlapAuditReport(n_total, report.n_passed, report.n_failed,
                              report.max_volume_relative_diff,
                              report.max_moment_relative_diff,
                              pair_failures)
end

# ============================================================================
# Module-load consistency check
# ============================================================================

# Mirrors the spirit of `_verify_moment_ordering` in `r3d_adapter.jl`:
# run a small canonical battery and bail with a clear error if anything
# disagrees. Because this file lives in the Diagnostics module (which
# loads BEFORE Overlap), this function cannot run at file include time.
# Instead, it is invoked from `Diagnostics.__init__()`, which runs after
# the entire HierarchicalGrids package is loaded — guaranteeing that
# both `Overlap.overlap_simplex_box!` and
# `Overlap.overlap_simplex_box_exact!` are resolvable.
#
# The check is gated behind `ENV["HG_INTEXACT_VERIFY"]`: set the var to
# any value other than "1" (e.g. "0", "no", "off") to skip. Default is
# "1" (run on every load). The audit battery is small (9 polytopes); on
# a warm Julia process it completes in well under 10 ms.
function _verify_intexact_consistency()
    if get(ENV, "HG_INTEXACT_VERIFY", "1") != "1"
        return nothing
    end
    # Force Sequential() for the module-load consistency check: the
    # battery is tiny (9 polytopes), so the parallel overhead would
    # dominate, and running parallel during `__init__` adds an
    # opportunity for precompile-time task scheduling weirdness with no
    # upside. Users who explicitly call `audit_overlap()` still get the
    # default-backend parallel path.
    HG = parentmodule(@__MODULE__)
    report = audit_overlap(; backend = HG.Threading.Sequential())
    if report.n_failed > 0
        @error "IntExact consistency check failed at module load" report
        throw(ErrorException(
            "IntExact consistency check found $(report.n_failed) " *
            "discrepancies on the canonical polytope battery; the " *
            "float (`overlap_simplex_box!`) and exact " *
            "(`overlap_simplex_box_exact!`) backends disagree. " *
            "See `audit_overlap()` and the OverlapAuditReport in the " *
            "preceding @error log for details."))
    end
    return nothing
end
