# ============================================================================
# CellView and HaloView
#
# Phase 2 foundation: the read/write accessors that solver kernels consume.
# Both views are constructed by the orchestrator (PR-7) on a per-cell basis
# and handed to user kernels. PR-6 ships them with constructors that take
# the raw bindings explicitly so the test suite can exercise them without
# the orchestrator.
#
# Design notes
# ------------
#
# 1. `CellView` carries TWO field-set bindings:
#    - `fields_in::NamedTuple{Names, Tin}` — read-only inputs.
#    - `fields_out::NamedTuple{Names, Tout}` — write target.
#    Reading via `cv[:rho]` goes to `fields_in`; writing via `cv[:rho] = v`
#    goes to `fields_out`. The orchestrator is responsible for swapping
#    bindings between time steps so writes never alias reads within a step.
#
# 2. `getindex(cv, :name)` returns the existing `PolynomialView` from
#    `Storage.PolynomialFieldSet`. That view is a tiny non-allocating
#    struct that supports per-coefficient indexing (`pv[k]`). Returning it
#    avoids materializing a coefficient vector on every read.
#
# 3. `HaloView` is read-only by design — there is no `setindex!` method.
#    Any attempt to write through the halo is a `MethodError`. Out-of-
#    domain offsets that hit a non-resolvable BC return `nothing`.
#
# 4. BC handling in PR-6 is intentionally minimal:
#    - PERIODIC   → wrap via `face_neighbors_with_bcs`.
#    - REFLECTING → return central cell's coefficients (zeroth-order
#                   reflection; sign-flipping is the kernel's job).
#    - OUTFLOW    → return central cell's coefficients (zeroth-order
#                   extrapolation).
#    - DIRICHLET  → return `nothing` (BC-value source pending PR-13).
#    - INFLOW     → return `nothing` (BC-value source pending PR-13).
#    - no `bcs`   → return `nothing` for any boundary hop.
# ============================================================================

# ----------------------------------------------------------------------------
# Helpers: per-cell read of polynomial coefficients (returns a PolynomialView)
# ----------------------------------------------------------------------------

# `_coeffs_for_cell(field, i)` returns a `PolynomialView` for cell `i` in the
# named field. The PolynomialView is a tiny zero-allocation wrapper that
# supports `pv[k]` for the k-th coefficient and `pv(point)` for evaluation.
@inline function _coeffs_for_cell(field::PolynomialFieldView, i::Integer)
    return field[Int(i)]
end

# `_set_coeffs_for_cell!(field, i, value)` writes a coefficient vector / tuple
# into cell `i` of `field`. Forwards to `PolynomialFieldView`'s indexed
# bulk-set, which validates the length and dispatches to the layout-specific
# write path.
@inline function _set_coeffs_for_cell!(field::PolynomialFieldView, i::Integer, value)
    field[Int(i)] = value
    return value
end

# ----------------------------------------------------------------------------
# CellView
# ----------------------------------------------------------------------------

"""
    CellView{Names, Tin, Tout, D, T}

Per-cell read/write view of a tuple of polynomial-field-sets.

Indexing:

    cv[:rho]              # read field by name, returns a PolynomialView
    cv[:rho] = new_rho    # write a coefficient column to the output buffer

Property access (precomputed at construction, O(1) reads):

    cv.coords             # cell-center physical coords :: NTuple{D, T}
    cv.volume             # physical volume of the cell :: T
    cv.level              # refinement level :: Int
    cv.index              # cell index in the mesh :: Int

Read-write contract:

- `cv[name]` reads from `fields_in[name]` for this cell.
- `cv[name] = v` writes to `fields_out[name]` for this cell.

Writes through `cv` do NOT affect `fields_in` — the orchestrator is
responsible for handing in distinct input/output bindings within a step.
"""
struct CellView{Names, Tin <: NamedTuple, Tout <: NamedTuple, D, T}
    fields_in::Tin
    fields_out::Tout
    index::Int
    coords::NTuple{D, T}
    volume::T
    level::Int
end

# ----------------------------------------------------------------------------
# CellView construction
# ----------------------------------------------------------------------------

"""
    cell_view(fields_in, fields_out, frame::EulerianFrame{D, T}, i::Integer)

Construct a `CellView` for cell `i` in the given Eulerian frame. The
metadata (`coords`, `volume`, `level`, `index`) is computed once at
construction so subsequent property reads are O(1).

Both `fields_in` and `fields_out` must be `NamedTuple`s of
`PolynomialFieldSet`s with the same set of names (the type parameter
`Names` is taken from `fields_in`).
"""
function cell_view(fields_in::NamedTuple{Names},
                   fields_out::NamedTuple{Names},
                   frame::EulerianFrame{D, T},
                   i::Integer) where {Names, D, T}
    p_lo, p_hi = cell_physical_box(frame, Int(i))
    coords = ntuple(d -> (p_lo[d] + p_hi[d]) / T(2), Val(D))
    extent = ntuple(d -> p_hi[d] - p_lo[d], Val(D))
    vol = one(T)
    @inbounds for d in 1:D
        vol = vol * extent[d]
    end
    lvl = Int(level_of(frame.mesh, Int(i)))
    Tin = typeof(fields_in)
    Tout = typeof(fields_out)
    return CellView{Names, Tin, Tout, D, T}(fields_in, fields_out, Int(i),
                                              coords, vol, lvl)
end

# Property access — read precomputed metadata. The Names type parameter
# enforces statically-known field names so dispatch on `cv[:name]` is type-
# stable.
@inline function Base.getproperty(cv::CellView, name::Symbol)
    return getfield(cv, name)
end

@inline function Base.propertynames(::CellView)
    return (:fields_in, :fields_out, :index, :coords, :volume, :level)
end

# Indexed read: returns a PolynomialView from the input field-set.
#
# Two methods are provided:
#
#  * `cv[name::Symbol]` — ergonomic; the compiler constant-folds the
#    literal symbol in monomorphic call sites, so this is fast in
#    practice. Type inference requires the symbol be a statically-known
#    constant.
#  * `cv[Val(name)]` — explicit `Val`-key form, fully type-stable under
#    `@inferred`. The orchestrator (PR-7) is encouraged to use this
#    inside hot loops where the field name is fixed at codegen time.
@inline function Base.getindex(cv::CellView{Names, Tin, Tout, D, T},
                                name::Symbol) where {Names, Tin, Tout, D, T}
    field = getfield(getfield(cv, :fields_in), name)
    return _coeffs_for_cell(field, getfield(cv, :index))
end

@inline function Base.getindex(cv::CellView{Names, Tin, Tout, D, T},
                                ::Val{name}) where {Names, Tin, Tout, D, T, name}
    field = getfield(getfield(cv, :fields_in), name)
    return _coeffs_for_cell(field, getfield(cv, :index))
end

# Indexed write: writes to the output field-set. The orchestrator binds
# `fields_out` to a distinct buffer than `fields_in`, so this never
# clobbers a read for any kernel within a step.
@inline function Base.setindex!(cv::CellView{Names, Tin, Tout, D, T},
                                value, name::Symbol) where {Names, Tin, Tout, D, T}
    field = getfield(getfield(cv, :fields_out), name)
    _set_coeffs_for_cell!(field, getfield(cv, :index), value)
    return value
end

@inline function Base.setindex!(cv::CellView{Names, Tin, Tout, D, T},
                                value, ::Val{name}) where {Names, Tin, Tout, D, T, name}
    field = getfield(getfield(cv, :fields_out), name)
    _set_coeffs_for_cell!(field, getfield(cv, :index), value)
    return value
end

function Base.show(io::IO, cv::CellView{Names, Tin, Tout, D, T}) where {Names, Tin, Tout, D, T}
    print(io, "CellView{D=", D, ", T=", T, "}(index=", cv.index,
              ", level=", cv.level, ", fields=", Names, ")")
end

# ----------------------------------------------------------------------------
# HaloView
# ----------------------------------------------------------------------------

"""
    HaloView{Names, Tin <: NamedTuple, D, T, GhostDepth, BC}

Per-cell halo view. Indexing returns the polynomial coefficients of the
cell at a given face-hop offset from the central cell, with BC resolution
at domain boundaries.

    hv[:rho, (1, 0, 0)]     # +x neighbor (depth 1)
    hv[:rho, (1, 0, 0), 2]  # depth-2 form (depth is informational; the
                            # offset itself encodes the hop count)

For periodic axes, wraps. For REFLECTING/OUTFLOW boundaries, returns the
central cell's own coefficients. For DIRICHLET/INFLOW boundaries, returns
`nothing` — the BC-value source is a follow-up (PR-13). For boundaries
with no `bcs` provided, returns `nothing`.

Read-only: no `setindex!` is defined; any attempt to write is a
`MethodError`.
"""
struct HaloView{Names, Tin <: NamedTuple, D, T, GhostDepth, BC}
    fields_in::Tin
    cell_index::Int
    mesh::HierarchicalMesh{D}
    bcs::BC      # ::Union{Nothing, FrameBoundaries{D}}
end

"""
    ghost_depth(::HaloView{Names, Tin, D, T, GhostDepth, BC}) -> Int

Return the declared ghost depth of the halo view (the maximum total
face-hop distance the orchestrator promised to resolve).
"""
@inline ghost_depth(::HaloView{Names, Tin, D, T, GhostDepth, BC}) where {Names, Tin, D, T, GhostDepth, BC} =
    GhostDepth

@inline spatial_dimension(::HaloView{Names, Tin, D, T, GhostDepth, BC}) where {Names, Tin, D, T, GhostDepth, BC} =
    D

# ----------------------------------------------------------------------------
# HaloView construction
# ----------------------------------------------------------------------------

"""
    halo_view_multi(fields_in, mesh, cell_index, ghost_depth;
                     bcs = nothing, scalar_type = Float64)

Construct a multi-field `HaloView` for the given central cell. The
`fields_in` argument is a `NamedTuple` of `PolynomialFieldSet`s. `bcs`
is either `nothing` or a `FrameBoundaries{D}`; the union is type-stable
because both branches have a concrete singleton-friendly type.

Internal use is via the orchestrator; tests construct directly with this
helper.
"""
function halo_view_multi(fields_in::NamedTuple{Names},
                          mesh::HierarchicalMesh{D},
                          cell_index::Integer,
                          ghost_depth::Integer;
                          bcs = nothing,
                          scalar_type::Type{T} = Float64) where {Names, D, T}
    ghost_depth >= 1 || throw(ArgumentError(
        "halo_view_multi: ghost_depth must be ≥ 1, got $ghost_depth"))
    if bcs !== nothing && !(bcs isa FrameBoundaries{D})
        throw(ArgumentError(
            "halo_view_multi: bcs must be ::FrameBoundaries{$D} or nothing"))
    end
    Tin = typeof(fields_in)
    BC = typeof(bcs)
    return HaloView{Names, Tin, D, T, Int(ghost_depth), BC}(
        fields_in, Int(cell_index), mesh, bcs)
end

# ----------------------------------------------------------------------------
# Offset walking with BC resolution
# ----------------------------------------------------------------------------

# Resolution kinds for a single boundary-axis hop.
#   :wrap     — periodic; treat as a normal neighbor hop after wrap.
#   :central  — return the central cell's own coefficients.
#   :unknown  — return `nothing` (no BC-value source available).
@inline function _bc_hop_kind(bcs::FrameBoundaries{D}, axis::Int, side::Int) where {D}
    # side: 1 = lo, 2 = hi
    kind = bcs.spec[axis][side]
    if kind === PERIODIC
        return :wrap
    elseif kind === REFLECTING || kind === OUTFLOW
        return :central
    else
        # DIRICHLET or INFLOW: no BC-value source in PR-6.
        return :unknown
    end
end

# Walk the offset axis-by-axis from `start`, returning either:
#   * a positive cell index (the resolved target cell for read), or
#   * `nothing` if a boundary hop hit DIRICHLET/INFLOW or no-BC, or
#   * the original `start` if a hop was REFLECTING/OUTFLOW (we treat the
#     reflection as "use the central cell's coefficients" — the central
#     cell here is the cell that hit the boundary, which for depth=1 is
#     the orchestrator's central cell).
#
# The return convention deliberately encodes "use central cell" as
# returning the cell index AT WHICH the boundary was hit (which is the
# correct cell for zeroth-order reflection / extrapolation), and
# encodes "unresolved" as `nothing`.
function _walk_offset(mesh::HierarchicalMesh{D}, start::Integer,
                       off::NTuple{D, Int}, bcs) where {D}
    cur = UInt32(start)
    @inbounds for d in 1:D
        steps = off[d]
        steps == 0 && continue
        sign = steps > 0 ? 1 : -1
        s = abs(steps)
        face_idx = sign > 0 ? 2*d : 2*d - 1     # lo = 2d-1, hi = 2d
        side = sign > 0 ? 2 : 1                  # 1 = lo, 2 = hi
        for _ in 1:s
            tup = face_neighbors(mesh, Int(cur))
            nxt = tup[face_idx]
            if nxt == 0
                # Hit a domain boundary. Consult bcs.
                if bcs === nothing
                    return nothing
                end
                kind = _bc_hop_kind(bcs::FrameBoundaries{D}, d, side)
                if kind === :wrap
                    # Use the periodic-aware neighbor wiring to resolve the
                    # opposite-wall partner. This is more expensive than a
                    # plain face_neighbors lookup, but only triggered on
                    # boundary cells and only on periodic axes.
                    wtup = face_neighbors_with_bcs(mesh, Int(cur),
                                                    bcs::FrameBoundaries{D})
                    nxt2 = wtup[face_idx]
                    if nxt2 == 0
                        # Defensive: periodic wiring couldn't resolve.
                        return nothing
                    end
                    cur = nxt2
                elseif kind === :central
                    # Return the cell at which the boundary was hit; the
                    # caller reads its coefficients (zeroth-order reflection
                    # / extrapolation).
                    return Int(cur)
                else
                    return nothing
                end
            else
                cur = nxt
            end
        end
    end
    return Int(cur)
end

# ----------------------------------------------------------------------------
# HaloView indexing
# ----------------------------------------------------------------------------

@inline function _check_offset(off::NTuple{D, Int}, depth::Int) where {D}
    s = 0
    @inbounds for d in 1:D
        s += abs(off[d])
    end
    s <= depth ||
        throw(ArgumentError(
            "HaloView offset $off exceeds declared GhostDepth $depth"))
    return nothing
end

"""
    hv[name::Symbol, off::NTuple{D, Int}] -> PolynomialView | Nothing

Read the coefficients of the cell at offset `off` from the central cell,
or `nothing` if the offset hits an unresolved domain boundary.

A `Val{name}`-keyed form is also provided for fully type-stable access
inside hot loops.
"""
@inline function Base.getindex(hv::HaloView{Names, Tin, D, T, GhostDepth, BC},
                                name::Symbol,
                                off::NTuple{D, Int}) where {Names, Tin, D, T, GhostDepth, BC}
    _check_offset(off, GhostDepth)
    target = _walk_offset(getfield(hv, :mesh),
                           getfield(hv, :cell_index),
                           off,
                           getfield(hv, :bcs))
    target === nothing && return nothing
    field = getfield(getfield(hv, :fields_in), name)
    return _coeffs_for_cell(field, target)
end

@inline function Base.getindex(hv::HaloView{Names, Tin, D, T, GhostDepth, BC},
                                ::Val{name},
                                off::NTuple{D, Int}) where {Names, Tin, D, T, GhostDepth, BC, name}
    _check_offset(off, GhostDepth)
    target = _walk_offset(getfield(hv, :mesh),
                           getfield(hv, :cell_index),
                           off,
                           getfield(hv, :bcs))
    target === nothing && return nothing
    field = getfield(getfield(hv, :fields_in), name)
    return _coeffs_for_cell(field, target)
end

# Three-argument form with an explicit depth — depth is informational
# (must not exceed GhostDepth or the offset magnitude).
@inline function Base.getindex(hv::HaloView{Names, Tin, D, T, GhostDepth, BC},
                                name::Symbol,
                                off::NTuple{D, Int},
                                depth::Integer) where {Names, Tin, D, T, GhostDepth, BC}
    Int(depth) <= GhostDepth ||
        throw(ArgumentError(
            "HaloView depth $depth exceeds declared GhostDepth $GhostDepth"))
    s = 0
    @inbounds for d in 1:D
        s += abs(off[d])
    end
    s <= Int(depth) ||
        throw(ArgumentError(
            "HaloView offset $off exceeds requested depth $depth"))
    return hv[name, off]
end

# ----------------------------------------------------------------------------
# Hanging-node fine-neighbor accessor
# ----------------------------------------------------------------------------

"""
    hv[name::Symbol, :face_fine_neighbors, axis::Int, side::Int] -> Vector

For an unbalanced face where the central cell sees multiple fine
neighbors, return a `Vector` of `PolynomialView`s — one per fine
neighbor, in the order produced by `face_fine_neighbors`. For a face
with a single neighbor (or a domain boundary), returns at most a
one-element vector. Returns an empty `Vector` for unresolved boundaries.
"""
function Base.getindex(hv::HaloView{Names, Tin, D, T, GhostDepth, BC},
                        name::Symbol,
                        ::Val{:face_fine_neighbors},
                        axis::Integer,
                        side::Integer) where {Names, Tin, D, T, GhostDepth, BC}
    1 <= Int(axis) <= D ||
        throw(ArgumentError("axis $axis out of range 1..$D"))
    Int(side) in (1, 2) ||
        throw(ArgumentError("side must be 1 (lo) or 2 (hi), got $side"))
    face = Int(side) == 1 ? 2*Int(axis) - 1 : 2*Int(axis)
    fines = face_fine_neighbors(getfield(hv, :mesh),
                                 getfield(hv, :cell_index),
                                 face)
    field = getfield(getfield(hv, :fields_in), name)
    out = Vector{typeof(field[1])}(undef, length(fines))
    @inbounds for k in eachindex(fines)
        out[k] = _coeffs_for_cell(field, Int(fines[k]))
    end
    return out
end

# Convenience: forward `(name, :face_fine_neighbors, axis, side)` (a Symbol
# rather than Val{:face_fine_neighbors}) for ergonomics in tests.
@inline function Base.getindex(hv::HaloView, name::Symbol,
                                tag::Symbol, axis::Integer, side::Integer)
    tag === :face_fine_neighbors ||
        throw(ArgumentError(
            "HaloView 4-arg indexing only supports :face_fine_neighbors tag, got :$tag"))
    return hv[name, Val(:face_fine_neighbors), axis, side]
end

function Base.show(io::IO, hv::HaloView{Names, Tin, D, T, GhostDepth, BC}) where {Names, Tin, D, T, GhostDepth, BC}
    print(io, "HaloView{D=", D, ", GhostDepth=", GhostDepth,
              ", bcs=", BC === Nothing ? "nothing" : "FrameBoundaries{$D}",
              "}(cell=", hv.cell_index, ", fields=", Names, ")")
end
