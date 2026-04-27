# ============================================================================
# BlockView and BlockHaloView (PR-10 — Path A: polynomial blocks)
#
# A "block" in Path A is a single HG cell carrying a polynomial in some
# basis B. The kernel-facing accessor exposes:
#
#   1. The same coefficient read/write API as PR-6's `CellView` —
#      `bv[Val(:rho)]` returns the underlying `PolynomialView`,
#      `bv[Val(:rho)] = new_coeffs` writes through `fields_out`.
#
#   2. A point-evaluation API — `bv(Val(:rho), ξ)` evaluates the block's
#      polynomial at the reference point `ξ ∈ [0,1]^D` via the basis's
#      `evaluate` operation. PR-11 will add a sibling specialization for
#      `PointSampleFieldSet` that interpolates instead of evaluates a
#      polynomial; the kernel signature is identical, so swapping the
#      backing field-type swaps the implementation cleanly.
#
# `BlockHaloView` is the per-block analog of PR-6's `HaloView`, with the
# same coefficient-read interface plus a `bhv(Val(:rho), off, ξ)`
# point-evaluation that lands in the *neighbor* block's reference frame.
# This is the primitive a high-order DG flux reconstruction needs at
# block boundaries.
#
# Design note — orientation in neighbour reference frames:
# `bhv(:rho, off, ξ)` evaluates the offset block's polynomial at `ξ`
# expressed in that neighbour's own reference cube `[0,1]^D`. Translating
# a face-shared interface point between the two reference frames is the
# kernel's responsibility (it knows which axis the neighbour sits on); the
# halo view itself is intentionally orientation-agnostic.
#
# Design note — `_walk_offset` reuse:
# The same offset-walking helper as `HaloView` is needed here. PR-6
# defined `_walk_offset` as a non-exported file-local helper inside
# `Views.jl`. Since both views live in the same submodule (`Solver`),
# we reuse PR-6's helper directly rather than duplicating the BC-handling
# logic — there is exactly one place where the periodic-wrap / reflecting-
# fallback / dirichlet-nothing decision tree is encoded.
# ============================================================================

# ----------------------------------------------------------------------------
# Inline polynomial evaluators (zero-allocation for concrete bases)
#
# `Bases.evaluate` is correct but goes through a lookup-cached exponent
# table (`_cached_monomial_exponents`, `_cached_bernstein_multiindices`)
# whose Vector return wraps in a Julia type assertion that may register a
# tiny allocation on warm-up. The orchestrator's hot path needs zero-
# allocation point evaluation, so we provide `@generated` inline
# evaluators that fully unroll the (small, compile-time-known) coefficient
# loop. Falls back to `Bases.evaluate` for any other basis.
# ----------------------------------------------------------------------------

# Generate the canonical graded-lex exponent ordering at codegen time.
function _gen_monomial_exponents(D::Int, P::Int)
    function _gen(N::Int, d::Int)
        if N == 1
            return [(d,)]
        end
        out = NTuple{N, Int}[]
        for k in d:-1:0
            for r in _gen(N-1, d-k)
                push!(out, (k, r...))
            end
        end
        return out
    end
    result = NTuple{D, Int}[]
    for d in 0:P
        append!(result, _gen(D, d))
    end
    return result
end

# Multinomial coefficient at codegen time (P! / α_0! ... α_D!).
function _gen_multinomial(P::Int, α::NTuple{N, Int}) where {N}
    @assert sum(α) == P
    coef = factorial(big(P))
    for k in α
        coef ÷= factorial(big(k))
    end
    return Int(coef)
end

"""
    _eval_block_at(basis, coeffs::NTuple{N, T}, point::NTuple{D, T2}) -> T

Inline polynomial evaluation. For `MonomialBasis{D, P}` and
`BernsteinBasis{D, P}` this is a `@generated` function that fully unrolls
the coefficient loop, giving zero-allocation evaluation when the basis is
concrete. For any other basis type, falls back to `Bases.evaluate`.
"""
@generated function _eval_block_at(::MonomialBasis{D, P},
                                    coeffs::NTuple{N, T},
                                    point::NTuple{D, T2}) where {D, P, N, T, T2}
    exps = _gen_monomial_exponents(D, P)
    @assert length(exps) == N
    if N == 0
        return :(zero(promote_type(T, T2)))
    end
    terms = Expr[]
    for k in 1:N
        e = exps[k]
        prod_expr = :(coeffs[$k])
        for i in 1:D
            if e[i] == 1
                prod_expr = :($prod_expr * point[$i])
            elseif e[i] > 1
                prod_expr = :($prod_expr * point[$i]^$(e[i]))
            end
        end
        push!(terms, prod_expr)
    end
    return Expr(:call, :+, terms...)
end

@generated function _eval_block_at(::BernsteinBasis{D, P},
                                    coeffs::NTuple{N, T},
                                    point::NTuple{D, T2}) where {D, P, N, T, T2}
    # Multi-indices α with sum P, length D+1.
    inds = NTuple{D+1, Int}[]
    function _gen(M::Int, d::Int)
        if M == 1
            return [(d,)]
        end
        out = NTuple{M, Int}[]
        for k in d:-1:0
            for r in _gen(M-1, d-k)
                push!(out, (k, r...))
            end
        end
        return out
    end
    inds = _gen(D+1, P)
    @assert length(inds) == N
    if N == 0
        return :(zero(promote_type(T, T2)))
    end
    terms = Expr[]
    # λ_0 = 1 - sum_{i=1..D} point[i]; λ_i = point[i] for i = 1..D.
    # We compute λ_0 inline via repeated subtraction: emit the expression
    # `(1 - point[1] - point[2] - ... - point[D])`.
    λ0_expr = :(one(promote_type(T, T2)))
    for i in 1:D
        λ0_expr = :($λ0_expr - point[$i])
    end
    for k in 1:N
        α = inds[k]   # length D+1 ; α[1]=α_0, α[2]=α_1=x_1, ...
        m = _gen_multinomial(P, α)
        prod_expr = :(coeffs[$k] * $(m))
        # λ_0 power
        if α[1] == 1
            prod_expr = :($prod_expr * ($λ0_expr))
        elseif α[1] > 1
            prod_expr = :($prod_expr * ($λ0_expr)^$(α[1]))
        end
        # λ_i = x_i powers (i = 1..D)
        for i in 1:D
            ai = α[i+1]
            if ai == 1
                prod_expr = :($prod_expr * point[$i])
            elseif ai > 1
                prod_expr = :($prod_expr * point[$i]^$(ai))
            end
        end
        push!(terms, prod_expr)
    end
    return Expr(:call, :+, terms...)
end

# Generic fallback — any other AbstractBasis: defer to Bases.evaluate.
@inline function _eval_block_at(basis::AbstractBasis, coeffs, point)
    return evaluate(basis, coeffs, point)
end

# Materialize a PolynomialView's coefficients as an NTuple for the
# generated evaluator.
@inline function _materialize_coeffs(pv::PolynomialView, ::Val{nc}) where {nc}
    return ntuple(k -> pv[k], Val(nc))
end

# ----------------------------------------------------------------------------
# Helper: extract the basis from a NamedTuple of PolynomialFieldView's
# ----------------------------------------------------------------------------

# All fields in a `PolynomialFieldSet`-backed NamedTuple share the same
# basis; pull it from the first field. The basis type is a compile-time
# parameter of the field-set, so this is fully type-stable.
@inline function _basis_from_fields(fields::NamedTuple{Names}) where {Names}
    return getfield(fields, Names[1]).pfs.basis
end

# ----------------------------------------------------------------------------
# BlockView
# ----------------------------------------------------------------------------

"""
    BlockView{Names, Tin, Tout, D, T, B <: AbstractBasis{D}}

Per-block read/write view (Path A — polynomial blocks). A "block" is a
single HG cell whose state is encoded as a polynomial of degree `P` in the
basis `B`. The kernel sees:

```
bv[Val(:rho)]                # raw coefficient view (PolynomialView)
bv[Val(:rho)] = new_coeffs   # write to fields_out
bv(Val(:rho), ξ)             # evaluate the polynomial at reference point ξ
```

Property access:

```
bv.coords       # cell-center physical coords     :: NTuple{D, T}
bv.volume       # physical volume of the cell     :: T
bv.level        # refinement level                :: Int
bv.index        # cell index in the mesh          :: Int
bv.basis        # the polynomial basis            :: B
bv.degree       # polynomial degree (compile-time-known)
bv.n_coeffs     # number of coefficients per block
```

Reads come from `fields_in[name]`; writes land in `fields_out[name]`.

Path B (PR-11) introduces `PointSampleFieldSet`-backed blocks. The
constructor `block_view` dispatches on field type, so PR-11 adds a
parallel specialization without touching this method.
"""
struct BlockView{Names, Tin <: NamedTuple, Tout <: NamedTuple, D, T, B}
    fields_in::Tin
    fields_out::Tout
    index::Int
    coords::NTuple{D, T}
    volume::T
    level::Int
    basis::B
end

# ----------------------------------------------------------------------------
# BlockView construction
# ----------------------------------------------------------------------------

"""
    block_view(fields_in, fields_out, frame::EulerianFrame{D, T}, i::Integer)

Construct a `BlockView` for cell `i` (Path A — polynomial blocks). Both
`fields_in` and `fields_out` must be `NamedTuple`s of `PolynomialFieldView`s
that share the same basis (the orchestrator enforces this at the field-set
level — all fields in one `PolynomialFieldSet` share its `B`).

Path B (PR-11) will add a parallel method dispatching on
`PointSampleFieldView`s; the call signature is identical so kernels need
no changes.
"""
function block_view(fields_in::NamedTuple{Names},
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
    basis = _basis_from_fields(fields_in)
    Tin = typeof(fields_in)
    Tout = typeof(fields_out)
    B = typeof(basis)
    return BlockView{Names, Tin, Tout, D, T, B}(fields_in, fields_out, Int(i),
                                                  coords, vol, lvl, basis)
end

# ----------------------------------------------------------------------------
# BlockView property access
# ----------------------------------------------------------------------------

# Property access exposes precomputed metadata + derived basis queries.
# `degree` and `n_coeffs` are read from the basis type parameters so they
# are compile-time-known.
@inline function Base.getproperty(bv::BlockView{Names, Tin, Tout, D, T, B},
                                   name::Symbol) where {Names, Tin, Tout, D, T, B}
    if name === :degree
        return _basis_degree(B)
    elseif name === :n_coeffs
        return n_coeffs(getfield(bv, :basis))
    else
        return getfield(bv, name)
    end
end

@inline function Base.propertynames(::BlockView)
    return (:fields_in, :fields_out, :index, :coords, :volume, :level,
            :basis, :degree, :n_coeffs)
end

# Compile-time degree extraction from the basis type parameter.
@inline _basis_degree(::Type{<:AbstractBasis{D, P}}) where {D, P} = P

# ----------------------------------------------------------------------------
# BlockView indexing — coefficient read/write (delegates to PR-6 helpers)
# ----------------------------------------------------------------------------

@inline function Base.getindex(bv::BlockView{Names, Tin, Tout, D, T, B},
                                name::Symbol) where {Names, Tin, Tout, D, T, B}
    field = getfield(getfield(bv, :fields_in), name)
    return _coeffs_for_cell(field, getfield(bv, :index))
end

@inline function Base.getindex(bv::BlockView{Names, Tin, Tout, D, T, B},
                                ::Val{name}) where {Names, Tin, Tout, D, T, B, name}
    field = getfield(getfield(bv, :fields_in), name)
    return _coeffs_for_cell(field, getfield(bv, :index))
end

@inline function Base.setindex!(bv::BlockView{Names, Tin, Tout, D, T, B},
                                 value, name::Symbol) where {Names, Tin, Tout, D, T, B}
    field = getfield(getfield(bv, :fields_out), name)
    _set_coeffs_for_cell!(field, getfield(bv, :index), value)
    return value
end

@inline function Base.setindex!(bv::BlockView{Names, Tin, Tout, D, T, B},
                                 value, ::Val{name}) where {Names, Tin, Tout, D, T, B, name}
    field = getfield(getfield(bv, :fields_out), name)
    _set_coeffs_for_cell!(field, getfield(bv, :index), value)
    return value
end

# ----------------------------------------------------------------------------
# BlockView point evaluation — `bv(Val(:rho), ξ)`
# ----------------------------------------------------------------------------

"""
    bv(Val(:rho), ξ::NTuple{D, T}) -> T
    bv(:rho, ξ::NTuple{D, T})      -> T

Evaluate the block's `:rho` polynomial at the reference point `ξ` in the
unit cube `[0, 1]^D`. The basis-specific evaluator is selected at codegen
time and (for `MonomialBasis{D, P}` / `BernsteinBasis{D, P}`) is fully
unrolled, giving zero-allocation evaluation when called inside a hot loop.
"""
@inline function (bv::BlockView{Names, Tin, Tout, D, T, B})(
        ::Val{name}, ξ::NTuple{D, T2}) where {Names, Tin, Tout, D, T, B, T2, name}
    field = getfield(getfield(bv, :fields_in), name)
    pv = _coeffs_for_cell(field, getfield(bv, :index))
    nc = n_coeffs(getfield(bv, :basis))
    coeffs = _materialize_coeffs(pv, Val(nc))
    return _eval_block_at(getfield(bv, :basis), coeffs, ξ)
end

@inline function (bv::BlockView{Names, Tin, Tout, D, T, B})(
        name::Symbol, ξ::NTuple{D, T2}) where {Names, Tin, Tout, D, T, B, T2}
    field = getfield(getfield(bv, :fields_in), name)
    pv = _coeffs_for_cell(field, getfield(bv, :index))
    nc = n_coeffs(getfield(bv, :basis))
    coeffs = _materialize_coeffs(pv, Val(nc))
    return _eval_block_at(getfield(bv, :basis), coeffs, ξ)
end

function Base.show(io::IO, bv::BlockView{Names, Tin, Tout, D, T, B}) where {Names, Tin, Tout, D, T, B}
    print(io, "BlockView{D=", D, ", basis=", B, "}(index=", bv.index,
              ", level=", bv.level, ", fields=", Names, ")")
end

# ----------------------------------------------------------------------------
# BlockHaloView
# ----------------------------------------------------------------------------

"""
    BlockHaloView{Names, Tin, D, T, GhostDepth, BC, B}

Per-block halo view (Path A — polynomial blocks). Same shape as PR-6's
`HaloView`, with two extra capabilities:

```
bhv[Val(:rho), off]             # neighbor block's coefficient view
bhv(Val(:rho), off, ξ)          # evaluate neighbor at reference point ξ
                                # (in the NEIGHBOR's own reference frame)
bhv[Val(:rho), :face_fine_neighbors, axis, side]  # coarse-fine list
```

BC handling exactly matches `HaloView` (PR-6):

  - `PERIODIC`             → wraps via the periodic-aware neighbor wiring.
  - `REFLECTING / OUTFLOW` → returns the central block's own coefficients
                              (zeroth-order extrapolation; the kernel is
                              responsible for any sign-flip).
  - `DIRICHLET / INFLOW`   → returns `nothing` for the coefficient form
                              (BC-value source pending PR-13). The
                              point-evaluation form similarly returns
                              `nothing`.
  - no `bcs` provided      → returns `nothing` for any boundary hop.

PR-11 (`PointSampleFieldSet`) will add a sibling type and an
overloaded constructor `block_halo_view` that returns it; the kernel's
indexing API is preserved.
"""
struct BlockHaloView{Names, Tin <: NamedTuple, D, T, GhostDepth, BC, B}
    fields_in::Tin
    cell_index::Int
    mesh::HierarchicalMesh{D}
    bcs::BC
    basis::B
end

@inline ghost_depth(::BlockHaloView{Names, Tin, D, T, GhostDepth, BC, B}
                     ) where {Names, Tin, D, T, GhostDepth, BC, B} = GhostDepth

@inline spatial_dimension(::BlockHaloView{Names, Tin, D, T, GhostDepth, BC, B}
                            ) where {Names, Tin, D, T, GhostDepth, BC, B} = D

# ----------------------------------------------------------------------------
# BlockHaloView construction
# ----------------------------------------------------------------------------

"""
    block_halo_view(fields_in, mesh, cell_index, ghost_depth;
                     bcs = nothing, scalar_type = Float64)

Construct a `BlockHaloView` for the given central cell. Path A only:
`fields_in` must be a `NamedTuple` of `PolynomialFieldView`s sharing a
common basis (i.e. coming from a single `PolynomialFieldSet`).

PR-11 will add a `block_halo_view` method that dispatches on
`PointSampleFieldView`s and returns a sibling `BlockHaloView`-shaped
type backed by point samples.
"""
function block_halo_view(fields_in::NamedTuple{Names},
                          mesh::HierarchicalMesh{D},
                          cell_index::Integer,
                          ghost_depth::Integer;
                          bcs = nothing,
                          scalar_type::Type{T} = Float64) where {Names, D, T}
    ghost_depth >= 1 || throw(ArgumentError(
        "block_halo_view: ghost_depth must be ≥ 1, got $ghost_depth"))
    if bcs !== nothing && !(bcs isa FrameBoundaries{D})
        throw(ArgumentError(
            "block_halo_view: bcs must be ::FrameBoundaries{$D} or nothing"))
    end
    basis = _basis_from_fields(fields_in)
    Tin = typeof(fields_in)
    BC = typeof(bcs)
    B = typeof(basis)
    return BlockHaloView{Names, Tin, D, T, Int(ghost_depth), BC, B}(
        fields_in, Int(cell_index), mesh, bcs, basis)
end

# ----------------------------------------------------------------------------
# BlockHaloView indexing — coefficient read (reuses PR-6's `_walk_offset`)
# ----------------------------------------------------------------------------

@inline function _bhv_check_offset(off::NTuple{D, Int}, depth::Int) where {D}
    s = 0
    @inbounds for d in 1:D
        s += abs(off[d])
    end
    s <= depth ||
        throw(ArgumentError(
            "BlockHaloView offset $off exceeds declared GhostDepth $depth"))
    return nothing
end

"""
    bhv[name::Symbol, off::NTuple{D, Int}] -> PolynomialView | Nothing

Read the polynomial coefficients of the block at offset `off` from the
central block, or `nothing` for an unresolved domain boundary. Mirrors
`HaloView`'s indexing exactly.
"""
@inline function Base.getindex(bhv::BlockHaloView{Names, Tin, D, T, GhostDepth, BC, B},
                                name::Symbol,
                                off::NTuple{D, Int}) where {Names, Tin, D, T, GhostDepth, BC, B}
    _bhv_check_offset(off, GhostDepth)
    target = _walk_offset(getfield(bhv, :mesh),
                           getfield(bhv, :cell_index),
                           off,
                           getfield(bhv, :bcs))
    target === nothing && return nothing
    field = getfield(getfield(bhv, :fields_in), name)
    return _coeffs_for_cell(field, target)
end

@inline function Base.getindex(bhv::BlockHaloView{Names, Tin, D, T, GhostDepth, BC, B},
                                ::Val{name},
                                off::NTuple{D, Int}) where {Names, Tin, D, T, GhostDepth, BC, B, name}
    _bhv_check_offset(off, GhostDepth)
    target = _walk_offset(getfield(bhv, :mesh),
                           getfield(bhv, :cell_index),
                           off,
                           getfield(bhv, :bcs))
    target === nothing && return nothing
    field = getfield(getfield(bhv, :fields_in), name)
    return _coeffs_for_cell(field, target)
end

# Three-arg form with explicit depth — informational, must match offset/GhostDepth.
@inline function Base.getindex(bhv::BlockHaloView{Names, Tin, D, T, GhostDepth, BC, B},
                                name::Symbol,
                                off::NTuple{D, Int},
                                depth::Integer) where {Names, Tin, D, T, GhostDepth, BC, B}
    Int(depth) <= GhostDepth ||
        throw(ArgumentError(
            "BlockHaloView depth $depth exceeds declared GhostDepth $GhostDepth"))
    s = 0
    @inbounds for d in 1:D
        s += abs(off[d])
    end
    s <= Int(depth) ||
        throw(ArgumentError(
            "BlockHaloView offset $off exceeds requested depth $depth"))
    return bhv[name, off]
end

# ----------------------------------------------------------------------------
# BlockHaloView point-evaluation
# ----------------------------------------------------------------------------

"""
    bhv(Val(:rho), off::NTuple{D, Int}, ξ::NTuple{D, T}) -> T | Nothing
    bhv(:rho, off, ξ)                                   -> T | Nothing

Evaluate the polynomial of the block at offset `off` from the central
block, at reference point `ξ` in the *neighbor's* unit-cube reference
frame. Returns `nothing` if the offset hits an unresolved domain boundary
(DIRICHLET/INFLOW or no-`bcs` boundary).

For REFLECTING/OUTFLOW the BC handling matches `HaloView`: returns the
central block's own polynomial evaluation at `ξ` (zeroth-order
extrapolation). PERIODIC wraps via the periodic-aware neighbor wiring.
"""
@inline function (bhv::BlockHaloView{Names, Tin, D, T, GhostDepth, BC, B})(
        ::Val{name}, off::NTuple{D, Int}, ξ::NTuple{D, T2}
        ) where {Names, Tin, D, T, GhostDepth, BC, B, T2, name}
    _bhv_check_offset(off, GhostDepth)
    target = _walk_offset(getfield(bhv, :mesh),
                           getfield(bhv, :cell_index),
                           off,
                           getfield(bhv, :bcs))
    target === nothing && return nothing
    field = getfield(getfield(bhv, :fields_in), name)
    pv = _coeffs_for_cell(field, target)
    basis = getfield(bhv, :basis)
    nc = n_coeffs(basis)
    coeffs = _materialize_coeffs(pv, Val(nc))
    return _eval_block_at(basis, coeffs, ξ)
end

@inline function (bhv::BlockHaloView{Names, Tin, D, T, GhostDepth, BC, B})(
        name::Symbol, off::NTuple{D, Int}, ξ::NTuple{D, T2}
        ) where {Names, Tin, D, T, GhostDepth, BC, B, T2}
    _bhv_check_offset(off, GhostDepth)
    target = _walk_offset(getfield(bhv, :mesh),
                           getfield(bhv, :cell_index),
                           off,
                           getfield(bhv, :bcs))
    target === nothing && return nothing
    field = getfield(getfield(bhv, :fields_in), name)
    pv = _coeffs_for_cell(field, target)
    basis = getfield(bhv, :basis)
    nc = n_coeffs(basis)
    coeffs = _materialize_coeffs(pv, Val(nc))
    return _eval_block_at(basis, coeffs, ξ)
end

# ----------------------------------------------------------------------------
# Hanging-node fine-neighbor accessor (mirrors HaloView's API)
# ----------------------------------------------------------------------------

"""
    bhv[name::Symbol, :face_fine_neighbors, axis::Int, side::Int] -> Vector

Returns a `Vector` of `PolynomialView`s — one per fine neighbor on the
given face — in the order produced by `face_fine_neighbors`. For a
conforming face this is a one-element vector.
"""
function Base.getindex(bhv::BlockHaloView{Names, Tin, D, T, GhostDepth, BC, B},
                        name::Symbol,
                        ::Val{:face_fine_neighbors},
                        axis::Integer,
                        side::Integer) where {Names, Tin, D, T, GhostDepth, BC, B}
    1 <= Int(axis) <= D ||
        throw(ArgumentError("axis $axis out of range 1..$D"))
    Int(side) in (1, 2) ||
        throw(ArgumentError("side must be 1 (lo) or 2 (hi), got $side"))
    face = Int(side) == 1 ? 2*Int(axis) - 1 : 2*Int(axis)
    fines = face_fine_neighbors(getfield(bhv, :mesh),
                                  getfield(bhv, :cell_index),
                                  face)
    field = getfield(getfield(bhv, :fields_in), name)
    out = Vector{typeof(field[1])}(undef, length(fines))
    @inbounds for k in eachindex(fines)
        out[k] = _coeffs_for_cell(field, Int(fines[k]))
    end
    return out
end

@inline function Base.getindex(bhv::BlockHaloView, name::Symbol,
                                tag::Symbol, axis::Integer, side::Integer)
    tag === :face_fine_neighbors ||
        throw(ArgumentError(
            "BlockHaloView 4-arg indexing only supports :face_fine_neighbors tag, got :$tag"))
    return bhv[name, Val(:face_fine_neighbors), axis, side]
end

function Base.show(io::IO, bhv::BlockHaloView{Names, Tin, D, T, GhostDepth, BC, B}
                    ) where {Names, Tin, D, T, GhostDepth, BC, B}
    print(io, "BlockHaloView{D=", D, ", GhostDepth=", GhostDepth,
              ", basis=", B,
              ", bcs=", BC === Nothing ? "nothing" : "FrameBoundaries{$D}",
              "}(cell=", bhv.cell_index, ", fields=", Names, ")")
end
