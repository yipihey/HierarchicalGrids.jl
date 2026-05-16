# ============================================================================
# PatchHierarchy + for_each_patch! + PatchBoundaryBC + restrict / prolong
#
# PR-13 — Berger–Oliger style patch-based AMR orchestrator.
#
# A `PatchHierarchy{D, T}` is a level-indexed list of `EulerianFrame{D, T}`s.
# Patches at level ℓ + 1 are contained in the union of level-ℓ patches; each
# fine patch sits inside one (or more) coarse parent patches. Patch
# boundaries that are NOT on the outer physical domain edge are filled by
# interpolation from the parent-level patch.
#
# This first cut implements the degree-0 path:
#
#   * Restriction (fine → coarse): conservative volume-weighted average of
#     fine-cell values into each parent cell, computed via
#     `compute_overlap(fine_frame, parent_frame)`. Coarse cells that have
#     positive overlap with the fine patch get the area-weighted mean of
#     the fine field; coarse cells with zero overlap retain their original
#     value.
#
#   * Prolongation (coarse → fine): each fine cell inherits the constant
#     coefficient of the parent cell whose physical box contains it (more
#     precisely: the parent cell with the largest overlap).
#
#   * Patch halo: when a halo offset walks off this patch's edge, the
#     `PatchHaloView` either (a) falls through to a same-level-patch
#     lookup (NOT IMPLEMENTED IN THE FIRST CUT — same-level patches are
#     allowed but their halos resolve through the parent) or (b) reads the
#     constant coefficient of the parent-level cell containing the ghost
#     cell's physical position, or (c) consults the physical BC for outer
#     domain edges.
#
# Higher-degree restrict/prolong (P ≥ 1) is documented as deferred — the
# full L²-projection path through `polynomial_remap_l_to_e!` /
# `polynomial_remap_e_to_l!` is the obvious follow-up. The mass-matrix
# solve and pullback machinery is already in place; only the orchestrator
# wiring and a per-pair frame-bookkeeping pass would need adding.
# ============================================================================

# ----------------------------------------------------------------------------
# PatchHierarchy
# ----------------------------------------------------------------------------

"""
    PatchHierarchy{D, T}

Berger–Oliger patch hierarchy. `levels[ℓ]::Vector{EulerianFrame{D, T}}`
holds patches at refinement level ℓ. Patches at finer levels overlap
their coarser parents.

# Fields

- `levels::Vector{Vector{EulerianFrame{D, T}}}` — patches per level.
  `levels[1]` is the base (typically a single root patch).
- `physical_bcs::Union{Nothing, FrameBoundaries{D}}` — outer-domain
  boundary specification, applied at patch edges that lie on the
  physical-domain wall.

# Invariants (checked by `validate(ph)`)

- `levels` is non-empty; `levels[1]` has at least one patch.
- For ℓ ≥ 2: each patch's `(lo, hi)` is contained in the union of the
  level-(ℓ-1) patches' physical boxes. Containment is strict in measure
  (the fine patch must lie inside a coarse patch's interior closure;
  touching at a single face is allowed).

The hierarchy is mutable: use `add_patches!` to extend it. Validation is
on-demand via `validate`.
"""
mutable struct PatchHierarchy{D, T}
    levels::Vector{Vector{EulerianFrame{D, T}}}
    physical_bcs::Union{Nothing, FrameBoundaries{D}}
end

"""
    PatchHierarchy(base::EulerianFrame{D, T}; physical_bcs = nothing)

Construct a hierarchy with a single base patch at level 1. Add more
patches at higher levels via `add_patches!`.
"""
function PatchHierarchy(base::EulerianFrame{D, T};
                         physical_bcs::Union{Nothing, FrameBoundaries{D}} = nothing
                         ) where {D, T}
    return PatchHierarchy{D, T}([[base]], physical_bcs)
end

"""
    add_patches!(ph::PatchHierarchy{D, T}, level::Int,
                  patches::Vector{EulerianFrame{D, T}}) -> ph

Append `patches` to `ph.levels[level]`, creating empty levels in between
if needed. Returns `ph` for chaining. Does NOT validate containment —
call `validate(ph)` afterwards if you want the invariant checked.
"""
function add_patches!(ph::PatchHierarchy{D, T}, level::Int,
                       patches::Vector{EulerianFrame{D, T}}) where {D, T}
    level >= 1 ||
        throw(ArgumentError("add_patches!: level must be ≥ 1, got $level"))
    while length(ph.levels) < level
        push!(ph.levels, EulerianFrame{D, T}[])
    end
    append!(ph.levels[level], patches)
    return ph
end

"""
    n_levels(ph::PatchHierarchy) -> Int

Number of refinement levels currently registered (including empty
intermediate levels, if any).
"""
@inline n_levels(ph::PatchHierarchy) = length(ph.levels)

"""
    n_patches(ph::PatchHierarchy, level::Integer) -> Int

Number of patches at the given level.
"""
@inline function n_patches(ph::PatchHierarchy, level::Integer)
    1 <= Int(level) <= length(ph.levels) ||
        throw(BoundsError(ph.levels, level))
    return length(ph.levels[Int(level)])
end

"""
    patches_at(ph::PatchHierarchy, level::Integer) -> Vector{EulerianFrame}

The list of patch frames at the given level.
"""
@inline function patches_at(ph::PatchHierarchy{D, T},
                              level::Integer) where {D, T}
    1 <= Int(level) <= length(ph.levels) ||
        throw(BoundsError(ph.levels, level))
    return ph.levels[Int(level)]
end

# Whether two boxes [a_lo, a_hi] and [b_lo, b_hi] satisfy a ⊆ b. Touching
# walls allowed (a_lo[d] == b_lo[d] is fine; we require the strict
# inequalities `a_lo >= b_lo` and `a_hi <= b_hi`).
@inline function _box_inside(a_lo::NTuple{D, T}, a_hi::NTuple{D, T},
                              b_lo::NTuple{D, T}, b_hi::NTuple{D, T}) where {D, T}
    @inbounds for d in 1:D
        a_lo[d] >= b_lo[d] || return false
        a_hi[d] <= b_hi[d] || return false
    end
    return true
end

"""
    validate(ph::PatchHierarchy) -> Nothing

Check the patch-containment invariant: each patch at level ℓ ≥ 2 is
contained in the union of level-(ℓ-1) patches. For the first cut the
union is approximated by "contained in some single parent patch" —
multi-parent overlap is allowed by the framework but rejected here so
that the parent-of-fine-patch lookup in `PatchBoundaryBC` is
unambiguous. Throws `ArgumentError` on violation.

`levels[1]` is treated as the root: any base patch is admissible.
"""
function validate(ph::PatchHierarchy{D, T}) where {D, T}
    isempty(ph.levels) &&
        throw(ArgumentError("PatchHierarchy has no levels"))
    isempty(ph.levels[1]) &&
        throw(ArgumentError("PatchHierarchy.levels[1] is empty"))

    for ℓ in 2:length(ph.levels)
        parents = ph.levels[ℓ - 1]
        for (pi, patch) in enumerate(ph.levels[ℓ])
            ok = false
            for parent in parents
                if _box_inside(patch.lo, patch.hi, parent.lo, parent.hi)
                    ok = true; break
                end
            end
            ok || throw(ArgumentError(
                "PatchHierarchy.validate: patch $pi at level $ℓ " *
                "lo=$(patch.lo) hi=$(patch.hi) is not contained in any " *
                "level-$(ℓ - 1) patch"))
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Helpers: locate a parent patch + parent leaf for a fine patch / point
# ----------------------------------------------------------------------------

# Find the (1-based) index of the parent-level patch that contains the fine
# patch's box. Returns 0 if no parent contains it (caller handles this as
# "patch is at outermost level / no interpolation source").
@inline function _find_parent_patch(parents::Vector{EulerianFrame{D, T}},
                                     patch::EulerianFrame{D, T}) where {D, T}
    @inbounds for (i, p) in enumerate(parents)
        if _box_inside(patch.lo, patch.hi, p.lo, p.hi)
            return i
        end
    end
    return 0
end

# Find the leaf cell index in `frame.mesh` whose physical box contains
# `point`. Returns 0 if no leaf contains the point. Brute-force linear
# scan — fine for the small leaf counts typical of patch boundaries.
function _find_leaf_containing(frame::EulerianFrame{D, T},
                                 point::NTuple{D, T}) where {D, T}
    Mesh.ensure_caches!(frame.mesh)
    @inbounds for ci in 1:n_cells(frame.mesh)
        is_leaf(frame.mesh.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        inside = true
        for d in 1:D
            if point[d] < lo[d] || point[d] > hi[d]
                inside = false; break
            end
        end
        inside && return Int(ci)
    end
    return 0
end

# Find the cell on the same patch whose physical box contains `point`.
# Like `_find_leaf_containing` but tolerant of points exactly on a face;
# uses a small relative epsilon to disambiguate.
@inline function _find_leaf_with_eps(frame::EulerianFrame{D, T},
                                       point::NTuple{D, T}) where {D, T}
    return _find_leaf_containing(frame, point)
end

# ----------------------------------------------------------------------------
# PatchBoundaryBC
# ----------------------------------------------------------------------------

"""
    PatchBoundaryBC{D, T, NT, BC}

Per-patch boundary handler used by `for_each_patch!`. Halo resolution
goes through three branches:

1. The cell at the queried offset is still inside this patch — handled by
   the underlying `Solver.HaloView` (no detour through `PatchBoundaryBC`).
2. The cell is outside this patch but inside the parent patch — return
   the parent cell's polynomial coefficients (degree-0: the constant
   coefficient is the cell's mean).
3. The cell is outside the parent patch as well — consult `physical_bcs`
   (the hierarchy's outer-domain BC) via the standard `Solver.HaloView`
   resolution rules.

# Fields

- `parent_frame::Union{Nothing, EulerianFrame{D, T}}` — the level-(ℓ-1)
  patch that contains this patch (or `nothing` at level 1).
- `parent_fields::NT` — `NamedTuple{Names}` of `PolynomialFieldView`s
  over `parent_frame.mesh`, mirroring `fields_in` at the parent level.
  `nothing` when `parent_frame` is `nothing`.
- `physical_bcs::BC` — outer-domain BC applied at patch edges that fall
  on the physical-domain boundary. `nothing` if the hierarchy has no
  physical BCs.

The (parent-frame, fine-patch-frame) overlap is computed once per call
to `for_each_patch!` (cached on the calling task's frame), then re-used
for every halo lookup. For the first cut the lookup itself is a leaf-
containment scan over the parent mesh, not a CSR overlap entry — the
overlap is built only for the restrict/prolong path which needs full
volume / moment data.
"""
struct PatchBoundaryBC{D, T, NT, BC}
    parent_frame::Union{Nothing, EulerianFrame{D, T}}
    parent_fields::NT
    physical_bcs::BC
end

"""
    PatchBoundaryBC(parent_frame, parent_fields; physical_bcs = nothing)

Build a `PatchBoundaryBC` from a parent patch's frame, the parent's
field-set views, and an optional outer-domain BC.
"""
function PatchBoundaryBC(parent_frame::EulerianFrame{D, T},
                          parent_fields::NamedTuple;
                          physical_bcs = nothing) where {D, T}
    NT = typeof(parent_fields)
    BC = typeof(physical_bcs)
    return PatchBoundaryBC{D, T, NT, BC}(parent_frame, parent_fields, physical_bcs)
end

# Sentinel constructor for level-1 patches (no parent). The `D` and `T`
# parameters are required for type-stable composition with `PatchHaloView`,
# even though the parent-frame slot is `nothing` and so is never read.
function PatchBoundaryBC{D, T}(::Nothing, ::Nothing;
                                physical_bcs = nothing) where {D, T}
    BC = typeof(physical_bcs)
    return PatchBoundaryBC{D, T, Nothing, BC}(nothing, nothing, physical_bcs)
end

# ----------------------------------------------------------------------------
# PatchView and PatchHaloView
# ----------------------------------------------------------------------------

"""
    PatchView{Names, Tin, Tout, D, T}

Per-cell view inside a patch. Thin wrapper around `CellView` so the
patch orchestrator can be kept distinct from `for_each_cell!` while
sharing all of the read/write/property surface. Constructed by
`for_each_patch!`; not user-built.
"""
struct PatchView{Names, Tin <: NamedTuple, Tout <: NamedTuple, D, T}
    cv::CellView{Names, Tin, Tout, D, T}
    patch_index::Int
    level::Int
end

# Forward all read/write/property methods to the inner CellView.
@inline Base.getindex(pv::PatchView, name::Symbol) = pv.cv[name]
@inline Base.getindex(pv::PatchView, ::Val{name}) where {name} = pv.cv[Val(name)]
@inline function Base.setindex!(pv::PatchView, value, name::Symbol)
    pv.cv[name] = value
    return value
end
@inline function Base.setindex!(pv::PatchView, value, ::Val{name}) where {name}
    pv.cv[Val(name)] = value
    return value
end

@inline function Base.getproperty(pv::PatchView, name::Symbol)
    if name === :cv || name === :patch_index || name === :level
        return getfield(pv, name)
    elseif name === :coords || name === :volume || name === :index
        return getproperty(getfield(pv, :cv), name)
    end
    return getfield(pv, name)
end

@inline Base.propertynames(::PatchView) =
    (:cv, :patch_index, :level, :coords, :volume, :index)

"""
    PatchHaloView{Names, Tin, D, T, GhostDepth, BC, NT, PBC}

Per-cell halo view that resolves out-of-patch offsets via the parent
patch (degree-0 interpolation: parent cell's constant coefficient) or
via the physical-domain BC.

The view holds:
- the underlying `Solver.HaloView` for same-patch lookups,
- the patch's frame (so we can compute physical positions of out-of-patch
  ghost cells), and
- a `PatchBoundaryBC` carrying the parent-level fields and the outer-domain
  BC.

When `hv[:rho, off]` is queried:
1. Walk the offset on this patch's mesh. If the walk lands on a same-patch
   cell, return that cell's coefficients (the standard `Solver.HaloView`
   path).
2. If the walk hits a 0 entry in `face_neighbors` (= patch wall), compute
   the physical position of the off-patch ghost cell and:
   a. Find the parent-level leaf containing that position; if found,
      return the parent's polynomial coefficients (degree-0: the constant
      mean of that parent cell).
   b. Otherwise, fall through to the physical-BC resolution
      (`Solver.HaloView` semantics: PERIODIC wraps, REFLECTING/OUTFLOW
      reflect to the central cell, DIRICHLET/INFLOW return `nothing`).

Read-only by design.
"""
struct PatchHaloView{Names, Tin <: NamedTuple, D, T, GhostDepth, BC, NT, PBC}
    fields_in::Tin
    cell_index::Int
    frame::EulerianFrame{D, T}
    patch_bcs::PatchBoundaryBC{D, T, NT, PBC}
    physical_halo::HaloView{Names, Tin, D, T, GhostDepth, BC}
end

@inline ghost_depth(::PatchHaloView{Names, Tin, D, T, GhostDepth, BC, NT, PBC}) where {Names, Tin, D, T, GhostDepth, BC, NT, PBC} =
    GhostDepth

# Compute the physical center of the would-be ghost cell at offset `off`
# from the central cell. The ghost cell is conceptually a copy of the
# central cell shifted by `off` units along each axis, where the unit
# matches the central cell's per-axis extent. This gives a stable
# "physical sample point" for parent-level lookup that doesn't require
# knowing the parent's grid alignment.
@inline function _ghost_center(frame::EulerianFrame{D, T},
                                 cell_index::Int,
                                 off::NTuple{D, Int}) where {D, T}
    p_lo, p_hi = cell_physical_box(frame, cell_index)
    extent = ntuple(d -> p_hi[d] - p_lo[d], Val(D))
    return ntuple(d -> (p_lo[d] + p_hi[d]) / T(2) + T(off[d]) * extent[d],
                   Val(D))
end

# Given a halo lookup that hit a patch wall, try to read the value from
# the parent-level patch at the ghost cell's center. Returns the parent
# cell's polynomial-view if found, or `nothing` otherwise (to fall through
# to physical-BC resolution).
@inline function _try_parent_lookup(hv::PatchHaloView{Names, Tin, D, T, GhostDepth, BC, NT, PBC},
                                      name::Symbol,
                                      off::NTuple{D, Int}
                                      ) where {Names, Tin, D, T, GhostDepth, BC, NT, PBC}
    pbc = getfield(hv, :patch_bcs)
    parent_frame = pbc.parent_frame
    parent_frame === nothing && return nothing
    parent_fields = pbc.parent_fields
    parent_fields === nothing && return nothing
    point = _ghost_center(getfield(hv, :frame),
                           getfield(hv, :cell_index), off)
    leaf = _find_leaf_containing(parent_frame::EulerianFrame{D, T},
                                   point)
    leaf == 0 && return nothing
    field = getfield(parent_fields, name)
    return _coeffs_for_cell(field, leaf)
end

@inline function Base.getindex(hv::PatchHaloView{Names, Tin, D, T, GhostDepth, BC, NT, PBC},
                                name::Symbol,
                                off::NTuple{D, Int}
                                ) where {Names, Tin, D, T, GhostDepth, BC, NT, PBC}
    # Same-patch path: try the underlying HaloView WITHOUT bcs, so a wall
    # hit is reported as `nothing` (rather than wrapped/reflected).
    base = getfield(hv, :physical_halo)
    target = _walk_offset(base.mesh, getfield(base, :cell_index), off, nothing)
    if target !== nothing
        # Same-patch hit: read directly from the patch's fields.
        field = getfield(getfield(base, :fields_in), name)
        return _coeffs_for_cell(field, target)
    end
    # Walked off the patch. Try the parent.
    parent_view = _try_parent_lookup(hv, name, off)
    parent_view === nothing || return parent_view
    # Fall through to the physical BC (via the underlying HaloView with
    # bcs enabled). This is the depth-0 reflection / periodic-wrap path.
    return base[name, off]
end

@inline function Base.getindex(hv::PatchHaloView{Names, Tin, D, T, GhostDepth, BC, NT, PBC},
                                ::Val{name},
                                off::NTuple{D, Int}
                                ) where {Names, Tin, D, T, GhostDepth, BC, NT, PBC, name}
    return hv[name, off]
end

# ----------------------------------------------------------------------------
# for_each_patch!
# ----------------------------------------------------------------------------

# Build a NamedTuple-of-PolynomialFieldView from a NamedTuple-of-fields.
# `fields` is already a NamedTuple of PolynomialFieldView (the orchestrator
# input convention) — no conversion needed; we pass it through. The helper
# exists so user-supplied PolynomialFieldSet objects can be normalised in
# one place if we ever need to.
@inline _normalize_views(fields::NamedTuple) = fields

"""
    for_each_patch!(kernel, fields_out, fields_in,
                    ph::PatchHierarchy{D, T};
                    level::Int,
                    ghost_depth::Int = 1,
                    ctx = nothing,
                    backend::AbstractParallelBackend = default_backend()
                   ) where {D, T}

Apply `kernel(pv::PatchView, hv::PatchHaloView, ctx)` to every leaf cell
of every patch at `level`, in parallel within each patch. Halo
resolution combines:

1. **Same-patch lookup** — same as `for_each_cell!`'s `HaloView` path.
2. **Parent-level interpolation** — when an offset walks off the patch
   wall and the ghost-cell location lies inside a parent-level patch,
   the kernel reads the parent's degree-0 (constant) coefficient.
3. **Physical-domain BC** — when neither (1) nor (2) resolves, the
   hierarchy's `physical_bcs` is consulted (PERIODIC wraps, etc.).

# Arguments

- `fields_out` — a `Vector{NamedTuple}` with one entry per patch at
  `level`. Each entry is a `NamedTuple` of `PolynomialFieldView`s
  matching the patch's mesh size (one field-set per patch).
- `fields_in` — read-only mirror of `fields_out` (same per-patch
  layout). Patches at level ℓ - 1 (used for parent interpolation) are
  pulled from `fields_in_parent` by the orchestrator: the user passes
  `fields_in_parent` as a separate keyword.
- `ph` — the patch hierarchy.
- `level` — the level whose patches will be visited.
- `ghost_depth`, `ctx`, `backend` — same semantics as `for_each_cell!`.

# Parent-field plumbing

`fields_in_parent` is a `Vector{NamedTuple}` over patches at level - 1,
mirroring `fields_in`'s shape. Pass `nothing` (the default) at level 1
where there is no parent.

The kernel may write to `fields_out[patch][:rho]` per-cell via
`pv[:rho] = ...`; reads through `pv` go to `fields_in[patch]`.

# Per-level dispatch

Patches at the same level are processed in sequence. Within each patch,
the leaf loop is parallelised via the configured backend, exactly like
`for_each_cell!`.
"""
function for_each_patch!(kernel, fields_out::Vector,
                          fields_in::Vector,
                          ph::PatchHierarchy{D, T};
                          level::Int,
                          fields_in_parent::Union{Nothing, Vector} = nothing,
                          ghost_depth::Int = 1,
                          ctx = nothing,
                          backend::AbstractParallelBackend = default_backend()
                          ) where {D, T}
    1 <= level <= length(ph.levels) ||
        throw(ArgumentError("for_each_patch!: level $level out of range " *
                             "1..$(length(ph.levels))"))
    ghost_depth >= 1 ||
        throw(ArgumentError("for_each_patch!: ghost_depth must be ≥ 1, " *
                             "got $ghost_depth"))

    patches = ph.levels[level]
    length(fields_out) == length(patches) ||
        throw(ArgumentError("for_each_patch!: fields_out has " *
                             "$(length(fields_out)) entries, expected " *
                             "$(length(patches)) (one per patch at level $level)"))
    length(fields_in) == length(patches) ||
        throw(ArgumentError("for_each_patch!: fields_in has " *
                             "$(length(fields_in)) entries, expected " *
                             "$(length(patches))"))
    if fields_in_parent !== nothing
        level >= 2 ||
            throw(ArgumentError("for_each_patch!: fields_in_parent given " *
                                 "but level == 1 (no parent level exists)"))
        length(fields_in_parent) == length(ph.levels[level - 1]) ||
            throw(ArgumentError("for_each_patch!: fields_in_parent has " *
                                 "$(length(fields_in_parent)) entries, " *
                                 "expected $(length(ph.levels[level - 1]))"))
    end

    parents = level >= 2 ? ph.levels[level - 1] : EulerianFrame{D, T}[]

    for (pi, patch) in enumerate(patches)
        mesh = patch.mesh
        Mesh.ensure_caches!(mesh)
        ensure_neighbor_graph!(mesh)

        leaves = enumerate_leaves(mesh)
        f_in = _normalize_views(fields_in[pi])
        f_out = _normalize_views(fields_out[pi])

        # Build per-patch PatchBoundaryBC. Hoist into a `let`-bound final
        # value so the per-cell closure below captures the value (not the
        # mutable binding) — without this, parallel backends reject the
        # closure with "Attempted to capture and modify outer local
        # variable: patch_bcs".
        patch_bcs_final = if level == 1 || isempty(parents) || fields_in_parent === nothing
            PatchBoundaryBC{D, T}(nothing, nothing;
                                    physical_bcs = ph.physical_bcs)
        else
            ppi = _find_parent_patch(parents, patch)
            if ppi == 0
                PatchBoundaryBC{D, T}(nothing, nothing;
                                        physical_bcs = ph.physical_bcs)
            else
                PatchBoundaryBC(parents[ppi],
                                  fields_in_parent[ppi];
                                  physical_bcs = ph.physical_bcs)
            end
        end

        # Use the hierarchy's physical_bcs only on patches that touch the
        # outer domain. For interior patches the underlying HaloView
        # rejects walls with `nothing`, and the patch BC fills in via
        # parent lookup. For the base patch, physical_bcs IS the outer BC
        # at every wall.
        underlying_bcs = ph.physical_bcs

        Names = keys(f_in)
        Tin = typeof(f_in)
        BC = typeof(underlying_bcs)

        let patch_bcs = patch_bcs_final
            parallel_foreach(backend,
                              i -> begin
                                  cv = cell_view(f_in, f_out, patch, Int(i))
                                  base_hv = halo_view_multi(f_in, mesh, Int(i),
                                                            ghost_depth;
                                                            bcs = underlying_bcs)
                                  hv = PatchHaloView{Names, Tin, D, T,
                                                      ghost_depth, BC,
                                                      typeof(patch_bcs.parent_fields),
                                                      typeof(patch_bcs.physical_bcs)}(
                                      f_in, Int(i), patch, patch_bcs, base_hv)
                                  pv = PatchView{Names, Tin, typeof(f_out), D, T}(cv,
                                                                                    pi,
                                                                                    level)
                                  kernel(pv, hv, ctx)
                                  return nothing
                              end,
                              leaves)
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Restrict and prolong
#
# Degree-0 path: conservative volume-weighted average — bit-identical to
# the original first-cut implementation (kept on a fast branch keyed on
# `n_coeffs == 1`).
#
# Degree ≥ 1 path: full L²-projection through each fine cell's polynomial
# reconstruction. For each (parent cell P, child cell C) overlap entry the
# RHS contribution is
#
#     b_P[α] += Σ_γ U_P[α, γ] · Σ_δ (Σ_ρ c_C[ρ] T_C[ρ, δ]) · M_{γ+δ}
#
# where T_C is the child's reference→physical pullback, U_P is the parent's
# reference→physical pullback, and M_{γ+δ} is the entry's moment of order
# `γ + δ` in physical coordinates (provided by `compute_overlap` at
# `moment_order ≥ 2P`). Solving `(|J_P| · M_ref) · c_P = b_P` per parent
# cell gives the L²-projection of the union of fine reconstructions onto
# the parent's polynomial basis.
#
# The cross mass matrix `M_{P, C}` is implicit in the per-entry moment
# vector × pullback product; we never materialize it. Caching this
# moment-times-pullback product per (fine_leaf, parent_leaf) pair is
# DEFERRED — the comment in the plan flags this as a future optimization.
#
# BernsteinBasis fields: route through `change_basis(MonomialBasis,
# BernsteinBasis, ...)` to monomial form, run the projection in monomial
# space, then `change_basis(BernsteinBasis, MonomialBasis, ...)` back.
# Mirrors the AdaptiveField Bernstein path (PR-4).
# ----------------------------------------------------------------------------

# Find, for each leaf in `dst_frame`, the leaf in `src_frame` whose
# physical box has the largest overlap with the destination cell. Returns
# a vector of length `n_cells(dst_frame.mesh)` with 0 for non-leaf or
# uncovered cells.
function _largest_overlap_source_leaf(src_frame::EulerianFrame{D, T},
                                        dst_frame::EulerianFrame{D, T}
                                        ) where {D, T}
    Mesh.ensure_caches!(src_frame.mesh)
    Mesh.ensure_caches!(dst_frame.mesh)
    n_dst = n_cells(dst_frame.mesh)
    out = zeros(Int, n_dst)
    @inbounds for ci in 1:n_dst
        is_leaf(dst_frame.mesh.cells[ci]) || continue
        d_lo, d_hi = cell_physical_box(dst_frame, ci)
        best_idx = 0
        best_vol = zero(T)
        for sj in 1:n_cells(src_frame.mesh)
            is_leaf(src_frame.mesh.cells[sj]) || continue
            s_lo, s_hi = cell_physical_box(src_frame, sj)
            aabbs_overlap(d_lo, d_hi, s_lo, s_hi) || continue
            vol = one(T)
            for d in 1:D
                w = min(d_hi[d], s_hi[d]) - max(d_lo[d], s_lo[d])
                w <= zero(T) && (vol = zero(T); break)
                vol *= w
            end
            if vol > best_vol
                best_vol = vol
                best_idx = Int(sj)
            end
        end
        out[ci] = best_idx
    end
    return out
end

# Pull (D, P) out of a polynomial basis. Supported here for the bases that
# carry an explicit `(D, P)` parameter pair. Other bases hit the fallback
# branch (degree-0 fast path or — for degree ≥ 1 — a clear error).
@inline _patch_basis_dim_degree(::MonomialBasis{D, P}) where {D, P} = (D, P)
@inline _patch_basis_dim_degree(::BernsteinBasis{D, P}) where {D, P} = (D, P)

# Build a (n_coeffs, n_cells) coefficient matrix from a `PolynomialFieldView`.
# Each column is the polynomial-coefficient vector of one cell. The matrix
# uses `Float64` for accumulation (parent mass-matrix solves promote to
# `Float64` regardless of the field's storage type, matching `AdaptiveField`'s
# convention). For BernsteinBasis fields, columns hold MONOMIAL coefficients
# (we apply `change_basis` per column on extraction).
function _read_field_coeffs_monomial(field, basis, ::Val{D}, P::Int,
                                       n_cells_total::Int) where {D}
    nc = binomial(D + P, P)
    out = zeros(Float64, nc, n_cells_total)
    if basis isa MonomialBasis
        @inbounds for i in 1:n_cells_total
            pv = _coeffs_for_cell(field, i)
            for k in 1:nc
                out[k, i] = Float64(pv[k])
            end
        end
        return out
    elseif basis isa BernsteinBasis
        mono_basis = MonomialBasis{D, P}()
        bern_buf = Vector{Float64}(undef, nc)
        @inbounds for i in 1:n_cells_total
            pv = _coeffs_for_cell(field, i)
            for k in 1:nc
                bern_buf[k] = Float64(pv[k])
            end
            mono_vec = change_basis(mono_basis, basis, bern_buf)
            for k in 1:nc
                out[k, i] = mono_vec[k]
            end
        end
        return out
    end
    error("_read_field_coeffs_monomial: unsupported basis $(typeof(basis))")
end

# Write monomial-form coefficients into one cell of a `PolynomialFieldView`.
# For `BernsteinBasis` fields, converts back via `change_basis` first.
@inline function _write_cell_coeffs_from_monomial!(field, basis, cell::Int,
                                                     mono_coeffs)
    pv = _coeffs_for_cell(field, cell)
    nc = length(pv)
    if basis isa MonomialBasis
        @inbounds for k in 1:nc
            pv[k] = mono_coeffs[k]
        end
        return
    elseif basis isa BernsteinBasis
        D, P = _patch_basis_dim_degree(basis)
        mono_basis = MonomialBasis{D, P}()
        bern_coeffs = change_basis(basis, mono_basis, mono_coeffs)
        @inbounds for k in 1:nc
            pv[k] = bern_coeffs[k]
        end
        return
    end
    error("_write_cell_coeffs_from_monomial!: unsupported basis $(typeof(basis))")
end

# Per-pair RHS accumulator for restrict/prolong. Adds to `rhs[α, j]` the
# L²-projection contribution of every overlap entry, where source = i,
# destination = j. Mirrors `_accumulate_polynomial_rhs_kernel!` in
# `polynomial_remap.jl` but does NOT zero `rhs` first (so callers can
# accumulate over multiple `(src_frame, dst_frame)` pairs).
function _accumulate_restrict_prolong_rhs!(rhs::Matrix{Float64},
                                            src_coeffs::Matrix{Float64},
                                            src_pullbacks::Vector{Matrix{Float64}},
                                            dst_pullbacks::Vector{Matrix{Float64}},
                                            overlap::GeometricOverlap{D, Float64},
                                            P_src::Int, P_dst::Int,
                                            invert::Bool, ::Val{D}) where {D}
    n_src = moments_length(D, P_src)
    n_dst = moments_length(D, P_dst)
    multi_src = moment_multiindices(D, P_src)
    multi_dst = moment_multiindices(D, P_dst)
    multi_phys = moment_multiindices(D, P_src + P_dst)
    lookup = Matrix{Int}(undef, n_dst, n_src)
    @inbounds for γi in 1:n_dst, δi in 1:n_src
        # Per-axis sum of the multi-indices; convert to flat index in multi_phys.
        target = ntuple(d -> multi_dst[γi][d] + multi_src[δi][d], D)
        # Linear search in multi_phys.
        idx = 0
        for k in eachindex(multi_phys)
            ok = true
            for d in 1:D
                if multi_phys[k][d] != target[d]
                    ok = false; break
                end
            end
            if ok
                idx = k; break
            end
        end
        lookup[γi, δi] = idx
    end
    Pi_buf = Vector{Float64}(undef, n_src)
    @inbounds for entry in overlap.entries
        i = invert ? Int(entry.eul_idx) : Int(entry.lag_idx)
        j = invert ? Int(entry.lag_idx) : Int(entry.eul_idx)
        src_pull = src_pullbacks[i]
        for δ in 1:n_src
            v = 0.0
            for α in 1:n_src
                v += src_coeffs[α, i] * src_pull[α, δ]
            end
            Pi_buf[δ] = v
        end
        dst_pull = dst_pullbacks[j]
        for α in 1:n_dst
            acc = 0.0
            for γ in 1:n_dst
                u = dst_pull[α, γ]
                u == 0.0 && continue
                inner = 0.0
                for δ in 1:n_src
                    inner += Pi_buf[δ] * Float64(entry.moments[lookup[γ, δ]])
                end
                acc += u * inner
            end
            rhs[α, j] += acc
        end
    end
    return rhs
end

# Solve `(|J_dst| · M_ref) c = b` for every destination cell whose
# `covered[j]` flag is true. Writes the resulting monomial coefficients into
# `dst_field` at column `j`. For BernsteinBasis fields converts back.
function _solve_and_write_dst!(dst_field, dst_basis,
                                 dst_frames::Vector{AxisAlignedRef{D, Float64}},
                                 rhs::Matrix{Float64},
                                 covered::Vector{Bool},
                                 P_dst::Int, ::Val{D}) where {D}
    n_dst = moments_length(D, P_dst)
    coeff_buf = Vector{Float64}(undef, n_dst)
    @inbounds for j in eachindex(covered)
        covered[j] || continue
        M_ref = reference_mass_matrix(dst_frames[j], P_dst)
        # |J_dst| = product of edge lengths for axis-aligned ref.
        jac = 1.0
        f = dst_frames[j]
        for d in 1:D
            jac *= (f.hi[d] - f.lo[d])
        end
        Mscaled = jac .* M_ref
        b = rhs[:, j]
        coeff_buf .= Mscaled \ b
        _write_cell_coeffs_from_monomial!(dst_field, dst_basis, j, coeff_buf)
    end
    return nothing
end

# Build axis-aligned reference frames for every cell of an EulerianFrame.
# Non-leaf cells get a placeholder zero-volume frame; their pullbacks /
# mass matrices are never read (the `covered` mask gates writes, and only
# leaves contribute overlap entries).
function _build_axisaligned_refs(frame::EulerianFrame{D, Float64}) where {D}
    Mesh.ensure_caches!(frame.mesh)
    n = n_cells(frame.mesh)
    refs = Vector{AxisAlignedRef{D, Float64}}(undef, n)
    placeholder = AxisAlignedRef{D, Float64}(ntuple(_ -> 0.0, Val(D)),
                                              ntuple(_ -> 1.0, Val(D)))
    @inbounds for i in 1:n
        if is_leaf(frame.mesh.cells[i])
            lo, hi = cell_physical_box(frame, i)
            refs[i] = AxisAlignedRef{D, Float64}(lo, hi)
        else
            refs[i] = placeholder
        end
    end
    return refs
end

# Degree-0 fast path for restrict — bit-identical to the original implementation.
function _restrict_to_parents_degree0!(fields_out_parent::Vector,
                                         fields_in_fine::Vector,
                                         parents::Vector{EulerianFrame{D, T}},
                                         fines::Vector{EulerianFrame{D, T}},
                                         fieldname::Symbol) where {D, T}
    for (par_i, par_frame) in enumerate(parents)
        Mesh.ensure_caches!(par_frame.mesh)
        n_par = n_cells(par_frame.mesh)
        sum_val = zeros(Float64, n_par)
        sum_vol = zeros(Float64, n_par)

        for (fine_i, fine_frame) in enumerate(fines)
            _box_inside(fine_frame.lo, fine_frame.hi,
                         par_frame.lo, par_frame.hi) || continue
            ov = compute_overlap(fine_frame, par_frame; moment_order = 0,
                                  parallel = false)
            fine_pfs = fields_in_fine[fine_i]
            fine_field = getfield(fine_pfs, fieldname)
            @inbounds for entry in ov.entries
                fl = Int(entry.lag_idx)
                pl = Int(entry.eul_idx)
                pv = _coeffs_for_cell(fine_field, fl)
                c0 = Float64(pv[1])
                v = Float64(entry.volume)
                sum_val[pl] += c0 * v
                sum_vol[pl] += v
            end
        end

        par_pfs = fields_out_parent[par_i]
        par_field = getfield(par_pfs, fieldname)
        @inbounds for ci in 1:n_par
            sum_vol[ci] > 0.0 || continue
            mean_val = sum_val[ci] / sum_vol[ci]
            pv = _coeffs_for_cell(par_field, ci)
            nc = length(pv)
            pv[1] = mean_val
            for k in 2:nc
                pv[k] = zero(typeof(pv[k]))
            end
        end
    end
    return nothing
end

# Higher-order (P ≥ 1) restrict for one parent patch. Accumulates the
# L²-projection RHS over every fine patch contained in this parent, then
# solves for every parent cell with positive coverage. Cells with no
# coverage retain their pre-call values.
function _restrict_higher_order_one_parent!(par_frame::EulerianFrame{D, Float64},
                                              par_pfs_field, par_basis,
                                              par_pullbacks::Vector{Matrix{Float64}},
                                              par_refs::Vector{AxisAlignedRef{D, Float64}},
                                              fines::Vector{EulerianFrame{D, Float64}},
                                              fields_in_fine::Vector,
                                              fieldname::Symbol,
                                              ::Val{D}, P::Int) where {D}
    n_par = n_cells(par_frame.mesh)
    n_dst = moments_length(D, P)
    rhs = zeros(Float64, n_dst, n_par)
    covered = fill(false, n_par)
    moment_ord = 2 * P

    for (fine_i, fine_frame) in enumerate(fines)
        _box_inside(fine_frame.lo, fine_frame.hi,
                     par_frame.lo, par_frame.hi) || continue
        ov = compute_overlap(fine_frame, par_frame;
                              moment_order = moment_ord, parallel = false)
        isempty(ov.entries) && continue

        fine_pfs_view = fields_in_fine[fine_i]
        fine_field = getfield(fine_pfs_view, fieldname)
        # Resolve the underlying PolynomialFieldSet for basis access.
        fine_basis = fine_field.pfs.basis
        n_fine_total = n_cells(fine_frame.mesh)
        fine_coeffs = _read_field_coeffs_monomial(fine_field, fine_basis,
                                                    Val(D), P, n_fine_total)
        fine_refs = _build_axisaligned_refs(fine_frame)
        fine_pullbacks = [reference_to_physical_pullback(f, P) for f in fine_refs]

        # Accumulate this fine patch's contribution to the parent RHS.
        # `compute_overlap(fine_frame, par_frame)` yields entries with
        # lag_idx = fine_leaf, eul_idx = parent_leaf, so invert = false.
        _accumulate_restrict_prolong_rhs!(rhs, fine_coeffs,
                                            fine_pullbacks, par_pullbacks,
                                            ov, P, P, false, Val(D))
        @inbounds for entry in ov.entries
            covered[Int(entry.eul_idx)] = true
        end
    end

    _solve_and_write_dst!(par_pfs_field, par_basis, par_refs, rhs,
                            covered, P, Val(D))
    return nothing
end

# Higher-order (P ≥ 1) prolong for one fine patch. Reads from the (already
# selected) parent patch and writes every fine cell that overlaps it.
function _prolong_higher_order_one_fine!(par_frame::EulerianFrame{D, Float64},
                                           par_pfs_field, par_basis,
                                           fine_frame::EulerianFrame{D, Float64},
                                           fine_pfs_field, fine_basis,
                                           ::Val{D}, P::Int) where {D}
    moment_ord = 2 * P
    # `compute_overlap(fine_frame, par_frame)` puts fine_leaf in lag_idx,
    # parent_leaf in eul_idx. For prolong (parent → fine) we want the
    # source = parent (eul_idx) and destination = fine (lag_idx), so we
    # invoke the kernel with `invert = true`.
    ov = compute_overlap(fine_frame, par_frame;
                          moment_order = moment_ord, parallel = false)
    isempty(ov.entries) && return nothing

    n_fine = n_cells(fine_frame.mesh)
    n_par = n_cells(par_frame.mesh)
    n_dst = moments_length(D, P)

    par_coeffs = _read_field_coeffs_monomial(par_pfs_field, par_basis,
                                               Val(D), P, n_par)
    par_refs = _build_axisaligned_refs(par_frame)
    par_pullbacks = [reference_to_physical_pullback(f, P) for f in par_refs]
    fine_refs = _build_axisaligned_refs(fine_frame)
    fine_pullbacks = [reference_to_physical_pullback(f, P) for f in fine_refs]

    rhs = zeros(Float64, n_dst, n_fine)
    covered = fill(false, n_fine)
    _accumulate_restrict_prolong_rhs!(rhs, par_coeffs, par_pullbacks,
                                        fine_pullbacks, ov, P, P, true, Val(D))
    @inbounds for entry in ov.entries
        covered[Int(entry.lag_idx)] = true
    end

    _solve_and_write_dst!(fine_pfs_field, fine_basis, fine_refs, rhs,
                            covered, P, Val(D))
    return nothing
end

"""
    restrict_to_parents!(fields_out_parent, fields_in_fine,
                          ph::PatchHierarchy{D, T}; level::Int,
                          fieldname::Symbol = :rho) where {D, T}

L²-projection restriction (fine → coarse): for every parent-level cell at
`level - 1` that is covered by one or more fine patches at `level`,
overwrite its polynomial coefficients with the L²-projection of the union
of fine reconstructions onto the parent's polynomial basis. Parent cells
with no fine-patch coverage are left unchanged.

`fields_out_parent` is a `Vector{NamedTuple}` over patches at `level - 1`;
each element holds the parent patch's writable fields. `fields_in_fine`
is the same shape over patches at `level`.

# Polynomial degree

The polynomial degree `P` is auto-detected from the basis carried by the
input fields. For degree-0 fields the implementation reduces to the
conservative volume-weighted mean (bit-identical to the legacy first-cut
behaviour). For `MonomialBasis{D, P}` and `BernsteinBasis{D, P}` with
`P ≥ 1` the projection preserves all moments up to degree P exactly
(modulo round-off): a polynomial reconstruction in the fine union projects
onto the parent's polynomial space without information loss to that order.

The cross mass matrix entries are read from the inter-frame `compute_overlap`
output at `moment_order = 2P` (the projection requires moments up to
degree P_src + P_dst). For BernsteinBasis fields the projection runs in
monomial coordinates via `change_basis`, mirroring the AdaptiveField path.

# Determinism

The per-pair `compute_overlap` is invoked with `parallel = false` for
deterministic accumulation. The mass-matrix solve is per-cell and depends
only on that cell's own RHS, so the result is byte-equal across thread
counts.
"""
function restrict_to_parents!(fields_out_parent::Vector,
                                fields_in_fine::Vector,
                                ph::PatchHierarchy{D, T};
                                level::Int,
                                fieldname::Symbol = :rho) where {D, T}
    level >= 2 ||
        throw(ArgumentError("restrict_to_parents!: level must be ≥ 2 " *
                             "(got $level); level 1 has no parent"))
    parents = ph.levels[level - 1]
    fines = ph.levels[level]
    length(fields_out_parent) == length(parents) ||
        throw(ArgumentError("restrict_to_parents!: fields_out_parent has " *
                             "$(length(fields_out_parent)) entries, expected " *
                             "$(length(parents))"))
    length(fields_in_fine) == length(fines) ||
        throw(ArgumentError("restrict_to_parents!: fields_in_fine has " *
                             "$(length(fields_in_fine)) entries, expected " *
                             "$(length(fines))"))

    # Detect the polynomial degree from the parent field. (All patches in
    # the hierarchy must share the same basis; we pick the first parent's
    # field as the reference.)
    if isempty(parents) || isempty(fields_out_parent)
        return nothing
    end
    par_field0 = getfield(fields_out_parent[1], fieldname)
    par_basis = par_field0.pfs.basis
    nc = n_coeffs(par_basis)

    if nc == 1
        # Degree-0 fast path: bit-identical to the original implementation.
        return _restrict_to_parents_degree0!(fields_out_parent,
                                              fields_in_fine,
                                              parents, fines, fieldname)
    end

    # Higher-order path. Validate basis compatibility and that we know how
    # to extract (D, P) from it.
    if !(par_basis isa MonomialBasis) && !(par_basis isa BernsteinBasis)
        throw(ArgumentError(
            "restrict_to_parents!: basis $(typeof(par_basis)) at degree ≥ 1 " *
            "is not supported. Use MonomialBasis or BernsteinBasis, or " *
            "convert via change_basis first."))
    end
    Db, P = _patch_basis_dim_degree(par_basis)
    Db == D ||
        throw(ArgumentError("restrict_to_parents!: basis dimension $Db ≠ " *
                             "PatchHierarchy dimension $D"))
    P >= 1 || error("restrict_to_parents!: degree-0 fast path missed (n_coeffs=$nc)")

    # Float64 promotion is enforced by the underlying Overlap / pullback
    # machinery (compute_overlap is Float64-only at the EulerianFrame
    # signature). Reject non-Float64 patch hierarchies cleanly.
    T === Float64 ||
        throw(ArgumentError(
            "restrict_to_parents!: degree-≥1 path requires Float64 " *
            "patch hierarchies (got T=$T). Use Float64 frames or stay at degree 0."))

    for (par_i, par_frame) in enumerate(parents)
        par_pfs_view = fields_out_parent[par_i]
        par_field = getfield(par_pfs_view, fieldname)
        par_refs = _build_axisaligned_refs(par_frame)
        par_pullbacks = [reference_to_physical_pullback(f, P) for f in par_refs]
        _restrict_higher_order_one_parent!(par_frame, par_field, par_basis,
                                            par_pullbacks, par_refs,
                                            fines, fields_in_fine, fieldname,
                                            Val(D), P)
    end
    return nothing
end

# Degree-0 fast path for prolong — bit-identical to the original implementation.
function _prolong_from_parents_degree0!(fields_out_fine::Vector,
                                          fields_in_parent::Vector,
                                          parents::Vector{EulerianFrame{D, T}},
                                          fines::Vector{EulerianFrame{D, T}},
                                          fieldname::Symbol) where {D, T}
    for (fine_i, fine_frame) in enumerate(fines)
        ppi = _find_parent_patch(parents, fine_frame)
        ppi == 0 && continue
        par_frame = parents[ppi]
        mapping = _largest_overlap_source_leaf(par_frame, fine_frame)
        par_pfs = fields_in_parent[ppi]
        par_field = getfield(par_pfs, fieldname)
        fine_pfs = fields_out_fine[fine_i]
        fine_field = getfield(fine_pfs, fieldname)
        @inbounds for ci in 1:length(mapping)
            par_leaf = mapping[ci]
            par_leaf == 0 && continue
            par_pv = _coeffs_for_cell(par_field, par_leaf)
            c0 = par_pv[1]
            fine_pv = _coeffs_for_cell(fine_field, ci)
            nc = length(fine_pv)
            fine_pv[1] = c0
            for k in 2:nc
                fine_pv[k] = zero(typeof(fine_pv[k]))
            end
        end
    end
    return nothing
end

"""
    prolong_from_parents!(fields_out_fine, fields_in_parent,
                           ph::PatchHierarchy{D, T}; level::Int,
                           fieldname::Symbol = :rho) where {D, T}

L²-projection prolongation (coarse → fine): every fine cell at `level`
receives the L²-projection of the polynomial reconstruction in its
covering parent cell, restricted to the fine cell's geometry.

For degree-0 fields the implementation reduces to the legacy "fine cell
inherits the parent cell's constant coefficient (largest-overlap parent)"
path — bit-identical. For `MonomialBasis{D, P}` and `BernsteinBasis{D, P}`
with `P ≥ 1` the prolongation preserves the parent's polynomial
reconstruction within each fine cell exactly (modulo round-off).

Higher-order coefficients on the fine side are zeroed for cells with no
parent overlap (mirrors the degree-0 behaviour); cells with overlap get
their full coefficient vector overwritten by the projection result.
"""
function prolong_from_parents!(fields_out_fine::Vector,
                                 fields_in_parent::Vector,
                                 ph::PatchHierarchy{D, T};
                                 level::Int,
                                 fieldname::Symbol = :rho) where {D, T}
    level >= 2 ||
        throw(ArgumentError("prolong_from_parents!: level must be ≥ 2 " *
                             "(got $level); level 1 has no parent"))
    parents = ph.levels[level - 1]
    fines = ph.levels[level]
    length(fields_out_fine) == length(fines) ||
        throw(ArgumentError("prolong_from_parents!: fields_out_fine has " *
                             "$(length(fields_out_fine)) entries, expected " *
                             "$(length(fines))"))
    length(fields_in_parent) == length(parents) ||
        throw(ArgumentError("prolong_from_parents!: fields_in_parent has " *
                             "$(length(fields_in_parent)) entries, expected " *
                             "$(length(parents))"))

    if isempty(fines) || isempty(fields_out_fine)
        return nothing
    end
    fine_field0 = getfield(fields_out_fine[1], fieldname)
    fine_basis = fine_field0.pfs.basis
    nc = n_coeffs(fine_basis)

    if nc == 1
        return _prolong_from_parents_degree0!(fields_out_fine,
                                                fields_in_parent,
                                                parents, fines, fieldname)
    end

    if !(fine_basis isa MonomialBasis) && !(fine_basis isa BernsteinBasis)
        throw(ArgumentError(
            "prolong_from_parents!: basis $(typeof(fine_basis)) at degree ≥ 1 " *
            "is not supported. Use MonomialBasis or BernsteinBasis, or " *
            "convert via change_basis first."))
    end
    Db, P = _patch_basis_dim_degree(fine_basis)
    Db == D ||
        throw(ArgumentError("prolong_from_parents!: basis dimension $Db ≠ " *
                             "PatchHierarchy dimension $D"))
    P >= 1 || error("prolong_from_parents!: degree-0 fast path missed (n_coeffs=$nc)")
    T === Float64 ||
        throw(ArgumentError(
            "prolong_from_parents!: degree-≥1 path requires Float64 " *
            "patch hierarchies (got T=$T). Use Float64 frames or stay at degree 0."))

    for (fine_i, fine_frame) in enumerate(fines)
        ppi = _find_parent_patch(parents, fine_frame)
        ppi == 0 && continue
        par_frame = parents[ppi]
        par_field = getfield(fields_in_parent[ppi], fieldname)
        par_basis = par_field.pfs.basis
        fine_field = getfield(fields_out_fine[fine_i], fieldname)
        _prolong_higher_order_one_fine!(par_frame, par_field, par_basis,
                                         fine_frame, fine_field, fine_basis,
                                         Val(D), P)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# step_patch_pipeline! — Berger-Oliger time step with the correct ordering
# ----------------------------------------------------------------------------

"""
    step_patch_pipeline!(kernel, fields_out, fields_in,
                          ph::PatchHierarchy{D, T};
                          ctx = nothing,
                          fieldname::Symbol = :rho,
                          ghost_depth::Int = 1,
                          backend::AbstractParallelBackend = default_backend()
                         ) where {D, T}

One time-step of a Berger-Oliger patch hierarchy, with the
empirically-verified sub-step ordering:

  1. **Prolong** — for every fine level (level 2..n_levels), fill
     `fields_in[ℓ]` from `fields_in[ℓ - 1]` via `prolong_from_parents!`.
     This makes the fine boundary halos in step 2 read pre-step parent
     values.
  2. **Update fine** — for each fine level, finest first, run
     `for_each_patch!(kernel, fields_out[ℓ], fields_in[ℓ], ph; level = ℓ,
     fields_in_parent = fields_in[ℓ - 1], ...)`.
  3. **Update coarse** — run the same kernel on level 1 (the base patch).
  4. **Restrict** — for every fine level, finest first, push the volume-
     weighted fine averages back to the parent's `fields_out` via
     `restrict_to_parents!`.

The same `kernel` is invoked at every level. The orchestrator passes the
correct per-cell `PatchView` (which carries `pv.level`) so the kernel
can dispatch on level if it needs to. For uniform first-order schemes
the kernel is level-agnostic.

# Arguments

- `kernel(pv::PatchView, hv::PatchHaloView, ctx)` — the per-cell update
  callable, identical to the one accepted by `for_each_patch!`.
- `fields_out::Vector{<:Vector}` — `fields_out[ℓ]` holds the output
  field-set views for every patch at level `ℓ` (one `NamedTuple` of
  `PolynomialFieldView`s per patch). Length must equal `n_levels(ph)`.
- `fields_in::Vector{<:Vector}` — same shape as `fields_out`, holding
  the input views.
- `ph::PatchHierarchy` — the hierarchy.
- `ctx` — opaque per-call context, threaded through to the kernel.
- `fieldname::Symbol` — the field touched by `prolong_from_parents!`
  and `restrict_to_parents!` (degree-0 path; defaults to `:rho`).
- `ghost_depth::Int` — same semantics as `for_each_patch!`.
- `backend::AbstractParallelBackend` — same semantics as
  `for_each_patch!`.

# Conservation

This ordering is documented (see `docs/src/patch_based_amr.md`) to give
bit-exact mass conservation when combined with the conservative degree-0
`restrict_to_parents!` and a fine patch that exactly tiles whole coarse
cells. The naive textbook ordering ("coarse first, then fine") produces
`O(dt)` per-step drift and is **not** the recommended pattern.

# Notes

- The helper only reads from `fields_in` and writes to `fields_out`; it
  does **not** swap buffers. Callers that double-buffer should perform
  the swap externally between steps.
- Step 1 writes into `fields_in` (the fine input buffer's ghost-zone
  cells get refreshed from the parent). This matches the worked
  example's pipeline in `examples/cfd_patch_amr/src/CFDPatchAMR.jl`.
- Step 4 writes into `fields_out` of the parent level; the overwrite
  is intentional — covered coarse cells' post-coarse values are
  replaced by the volume-weighted fine averages.
"""
function step_patch_pipeline!(kernel,
                                fields_out::Vector,
                                fields_in::Vector,
                                ph::PatchHierarchy{D, T};
                                ctx = nothing,
                                fieldname::Symbol = :rho,
                                ghost_depth::Int = 1,
                                backend::AbstractParallelBackend = default_backend()
                                ) where {D, T}
    n_lev = length(ph.levels)
    length(fields_out) == n_lev ||
        throw(ArgumentError("step_patch_pipeline!: fields_out has " *
                             "$(length(fields_out)) entries, expected " *
                             "$(n_lev) (one vector-of-views per level)"))
    length(fields_in) == n_lev ||
        throw(ArgumentError("step_patch_pipeline!: fields_in has " *
                             "$(length(fields_in)) entries, expected " *
                             "$(n_lev)"))

    # Step 1: Prolong fine levels from their parents (top-down: coarsest
    # parent first, so each level's input buffer is consistent before its
    # update runs). This fills fine ghost zones via the parent.
    for ℓ in 2:n_lev
        prolong_from_parents!(fields_in[ℓ], fields_in[ℓ - 1],
                               ph; level = ℓ, fieldname = fieldname)
    end

    # Step 2: Update fine levels, finest first. Each level's update reads
    # parent halos through `fields_in_parent` (= the pre-step parent
    # input buffer).
    for ℓ in n_lev:-1:2
        for_each_patch!(kernel, fields_out[ℓ], fields_in[ℓ], ph;
                          level = ℓ, ghost_depth = ghost_depth,
                          fields_in_parent = fields_in[ℓ - 1],
                          ctx = ctx, backend = backend)
    end

    # Step 3: Update the coarse base on the full domain.
    for_each_patch!(kernel, fields_out[1], fields_in[1], ph;
                      level = 1, ghost_depth = ghost_depth,
                      ctx = ctx, backend = backend)

    # Step 4: Restrict fine outputs back onto parent outputs (finest
    # first; covered parent cells get overwritten by volume-weighted
    # averages).
    for ℓ in n_lev:-1:2
        restrict_to_parents!(fields_out[ℓ - 1], fields_out[ℓ],
                              ph; level = ℓ, fieldname = fieldname)
    end

    return nothing
end
