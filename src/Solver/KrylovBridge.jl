# ============================================================================
# KrylovBridge.jl — flat-vector wrapping of the FAC composite operator and
# of the variable-coefficient ABec operator, so that Krylov.jl methods
# (GMRES, BiCGStab, FGMRES, MINRES …) can drive the solve.
#
# Why: our PCG-on-composite handles the SPD case. Non-SPD operators
# (tensor viscosity with cross-terms, advection-diffusion, the curl-curl
# of magnetostatics) need GMRES/BiCGStab/FGMRES. Rather than reinvent
# these, we expose the operator over a flat vector view of the
# hierarchical-grid storage and delegate to Krylov.jl.
#
# Tier 1 #3 of the AMReX-port roadmap.
# ============================================================================

using Krylov: gmres, bicgstab, cg, minres, fgmres
using LinearAlgebra: LinearAlgebra
import LinearAlgebra: mul!

"""
    FlatLayout{D, T}

Bidirectional mapping between the patch-hierarchy storage (active
uncovered cells across `level_range`) and a flat `Vector{T}`. Stores the
offset table once so `pack!` / `unpack!` are alloc-free.

Fields:
  * `n::Int` — total length of the flat vector.
  * `level_range::UnitRange{Int}`.
  * `cells::Vector{NTuple{3, Int}}` — `(level, patch, cell_index)` for each
    of the `n` flat positions, in iteration order.
"""
struct FlatLayout{D, T}
    n::Int
    level_range::UnitRange{Int}
    cells::Vector{NTuple{3, Int}}
end

"""
    flat_layout(ws::MGWorkspace{D, T}; level_range = ws.level_range) -> FlatLayout

Build a `FlatLayout` enumerating the active (uncovered) cells of every
patch in `level_range`. Storage order is `(level, patch, cell)` with cells
in `ws.patch_leaves[ℓ][pi]` order.
"""
function flat_layout(ws::MGWorkspace{D, T};
                      level_range::UnitRange{Int} = ws.level_range) where {D, T}
    cells = NTuple{3, Int}[]
    for ℓ in level_range
        for pi in 1:length(ws.patch_leaves[ℓ])
            covered = ws.covered_by_finer[ℓ][pi]
            for c in ws.patch_leaves[ℓ][pi]
                covered[c] && continue
                push!(cells, (ℓ, pi, c))
            end
        end
    end
    return FlatLayout{D, T}(length(cells), level_range, cells)
end

"""
    pack!(flat, fields, layout; field = :phi)

Copy `fields[ℓ][pi].<field>[c]` for each `(ℓ, pi, c)` in `layout.cells`
into `flat`. `field` defaults to `:phi`; pass `:rho` for the RHS slot.
"""
function pack!(flat::AbstractVector{T},
                fields::Vector{Vector{NamedTuple}},
                layout::FlatLayout{D, T};
                field::Symbol = :phi) where {D, T}
    length(flat) >= layout.n || throw(ArgumentError("flat too small"))
    getter = field === :phi ? _raw_phi : _raw_rho
    @inbounds for i in 1:layout.n
        ℓ, pi, c = layout.cells[i]
        flat[i] = getter(fields[ℓ][pi])[c]
    end
    return flat
end

"""
    unpack!(fields, flat, layout; field = :phi)

Inverse of `pack!`: writes each `flat[i]` into
`fields[ℓ][pi].<field>[c]`.
"""
function unpack!(fields::Vector{Vector{NamedTuple}},
                  flat::AbstractVector{T},
                  layout::FlatLayout{D, T};
                  field::Symbol = :phi) where {D, T}
    length(flat) >= layout.n || throw(ArgumentError("flat too small"))
    getter = field === :phi ? _raw_phi : _raw_rho
    @inbounds for i in 1:layout.n
        ℓ, pi, c = layout.cells[i]
        getter(fields[ℓ][pi])[c] = flat[i]
    end
    return fields
end

# ----------------------------------------------------------------------------
# Operator wrappers (callable structs implementing `*` and `mul!`).
# ----------------------------------------------------------------------------

"""
    FACCompositeOp{D, T}

Function-style linear operator: `op * x_flat` returns `(L_FAC · phi)_flat`
where `phi` is unpacked from `x_flat`. Used as the matvec for the
const-coef FAC operator (matches `apply_composite_laplacian!`).
"""
mutable struct FACCompositeOp{D, T}
    ws::MGWorkspace{D, T}
    layout::FlatLayout{D, T}
    # Scratch fields shared across matvecs.
    phi_scratch::Vector{Vector{NamedTuple}}
    Lphi_scratch::Vector{Vector{NamedTuple}}
end

function FACCompositeOp(ws::MGWorkspace{D, T};
                          level_range::UnitRange{Int} = ws.level_range) where {D, T}
    layout = flat_layout(ws; level_range = level_range)
    phi_scratch  = allocate_phi_rho(ws.ph)
    Lphi_scratch = allocate_phi_rho(ws.ph)
    return FACCompositeOp{D, T}(ws, layout, phi_scratch, Lphi_scratch)
end

Base.size(op::FACCompositeOp) = (op.layout.n, op.layout.n)
Base.size(op::FACCompositeOp, dim::Int) = op.layout.n
Base.eltype(op::FACCompositeOp{D, T}) where {D, T} = T

function LinearAlgebra.mul!(y::AbstractVector{T}, op::FACCompositeOp{D, T},
                              x::AbstractVector{T}) where {D, T}
    unpack!(op.phi_scratch, x, op.layout)
    apply_composite_laplacian!(op.Lphi_scratch, op.phi_scratch, op.ws;
                                 level_range = op.layout.level_range)
    pack!(y, op.Lphi_scratch, op.layout)
    # Composite operator: PCG sees A = -L_FAC for SPD-positive convention.
    @inbounds @simd for i in eachindex(y)
        y[i] = -y[i]
    end
    return y
end

Base.:*(op::FACCompositeOp{D, T}, x::AbstractVector{T}) where {D, T} =
    mul!(similar(x), op, x)

"""
    ABecOp{D, T}

Variable-coefficient ABec operator wrapped as a flat-vector linear
operator. `op * x_flat` returns `(L_ABec · phi)_flat`.
"""
mutable struct ABecOp{D, T}
    ws::MGWorkspace{D, T}
    coefs::ABecCoefs{D, T}
    layout::FlatLayout{D, T}
    phi_scratch::Vector{Vector{NamedTuple}}
    Lphi_scratch::Vector{Vector{NamedTuple}}
end

function ABecOp(ws::MGWorkspace{D, T}, coefs::ABecCoefs{D, T};
                  level_range::UnitRange{Int} = ws.level_range) where {D, T}
    layout = flat_layout(ws; level_range = level_range)
    return ABecOp{D, T}(ws, coefs, layout,
                          allocate_phi_rho(ws.ph),
                          allocate_phi_rho(ws.ph))
end

Base.size(op::ABecOp) = (op.layout.n, op.layout.n)
Base.size(op::ABecOp, dim::Int) = op.layout.n
Base.eltype(op::ABecOp{D, T}) where {D, T} = T

function LinearAlgebra.mul!(y::AbstractVector{T}, op::ABecOp{D, T},
                              x::AbstractVector{T}) where {D, T}
    unpack!(op.phi_scratch, x, op.layout)
    apply_abec!(op.Lphi_scratch, op.phi_scratch, op.coefs, op.ws;
                 level_range = op.layout.level_range)
    pack!(y, op.Lphi_scratch, op.layout)
    return y
end

Base.:*(op::ABecOp{D, T}, x::AbstractVector{T}) where {D, T} =
    mul!(similar(x), op, x)

# ----------------------------------------------------------------------------
# Driver: pick a Krylov.jl method and solve.
# ----------------------------------------------------------------------------

"""
    solve_with_krylov!(phi, rho, op, ws; method=:cg, tol=1e-9, maxiter=200,
                        precond=nothing, verbose=false, level_range=op.layout.level_range)

Solve `op * phi = rho` using a Krylov.jl method (`:cg`, `:gmres`,
`:bicgstab`, `:fgmres`, `:minres`). `phi`, `rho` are the hierarchical
field containers; the bridge packs/unpacks to flat vectors internally.

`precond` is an optional left-preconditioner — a callable `M(z, r)` that
writes `z := M^{-1} r` in flat-vector form. Pass `nothing` for no
preconditioning.

Returns an `MGResult` mirroring the convention used elsewhere.
"""
function solve_with_krylov!(phi::Vector{Vector{NamedTuple}},
                              rho::Vector{Vector{NamedTuple}},
                              op,
                              ws::MGWorkspace{D, T};
                              method::Symbol = :cg,
                              tol::Float64 = 1e-9,
                              maxiter::Int = 200,
                              precond = nothing,
                              verbose::Bool = false,
                              rhs_sign::T = one(T)) where {D, T}
    layout = op.layout
    b = Vector{T}(undef, layout.n)
    x0 = Vector{T}(undef, layout.n)
    pack!(b, rho, layout; field = :rho)
    pack!(x0, phi, layout; field = :phi)

    # Apply rhs_sign if the operator was negated (e.g. FACCompositeOp uses
    # A = -L_FAC, so the user's "L φ = ρ" becomes A φ = -ρ).
    if rhs_sign != one(T)
        @inbounds @simd for i in eachindex(b)
            b[i] = rhs_sign * b[i]
        end
    end

    r0 = _flat_norm(b, op, x0)

    if precond === nothing
        x, stats = _krylov_solve(method, op, b;
                                  itmax = maxiter, atol = tol, rtol = tol,
                                  history = true,
                                  verbose = verbose ? 1 : 0)
    else
        # Wrap user's precond as a LinearOperators-compatible callable.
        M = _PrecondOp{D, T}(precond, layout, allocate_phi_rho(ws.ph),
                              allocate_phi_rho(ws.ph))
        x, stats = _krylov_solve(method, op, b;
                                  M = M, itmax = maxiter,
                                  atol = tol, rtol = tol,
                                  history = true,
                                  verbose = verbose ? 1 : 0)
    end

    unpack!(phi, x, layout)
    converged = stats.solved
    iters = stats.niter
    history = isempty(stats.residuals) ? [r0] : copy(stats.residuals)
    res_final = history[end]
    return MGResult(iters, r0, res_final, converged, history)
end

# Compute the initial residual norm.  We do this with one matvec.
function _flat_norm(b::Vector{T}, op, x0::Vector{T}) where {T}
    Lx0 = similar(b)
    mul!(Lx0, op, x0)
    s = zero(T)
    @inbounds @simd for i in eachindex(b)
        d = b[i] - Lx0[i]
        s += d * d
    end
    return Float64(sqrt(s))
end

# Tag-dispatched Krylov-method selector.
function _krylov_solve(::Val{:cg}, op, b; kwargs...)
    return cg(op, b; kwargs...)
end
_krylov_solve(::Val{:gmres}, op, b; kwargs...) = gmres(op, b; kwargs...)
_krylov_solve(::Val{:bicgstab}, op, b; kwargs...) = bicgstab(op, b; kwargs...)
_krylov_solve(::Val{:fgmres}, op, b; kwargs...) = fgmres(op, b; kwargs...)
_krylov_solve(::Val{:minres}, op, b; kwargs...) = minres(op, b; kwargs...)
_krylov_solve(method::Symbol, op, b; kwargs...) =
    _krylov_solve(Val(method), op, b; kwargs...)

# Preconditioner wrapper: M⁻¹ r in flat-vector form, by unpacking r,
# calling the user callback, then re-packing.
mutable struct _PrecondOp{D, T}
    callback                       # function (z_fields, r_fields) -> z_fields
    layout::FlatLayout{D, T}
    z_scratch::Vector{Vector{NamedTuple}}
    r_scratch::Vector{Vector{NamedTuple}}
end

Base.size(p::_PrecondOp) = (p.layout.n, p.layout.n)
Base.eltype(p::_PrecondOp{D, T}) where {D, T} = T

function LinearAlgebra.mul!(z::AbstractVector{T}, p::_PrecondOp{D, T},
                              r::AbstractVector{T}) where {D, T}
    unpack!(p.r_scratch, r, p.layout)
    p.callback(p.z_scratch, p.r_scratch)
    pack!(z, p.z_scratch, p.layout)
    return z
end

Base.:*(p::_PrecondOp{D, T}, r::AbstractVector{T}) where {D, T} =
    mul!(similar(r), p, r)

# Convenience: build a Jacobi preconditioner for the ABec operator.
"""
    abec_jacobi_precond(ws, coefs; level_range = ws.level_range) -> callback

Return a callable `(z_fields, r_fields) -> z_fields` that applies the
Jacobi preconditioner `diag(L_ABec)^{-1}` cell-by-cell. Suitable to pass
as `precond` to `solve_with_krylov!`.
"""
function abec_jacobi_precond(ws::MGWorkspace{D, T}, coefs::ABecCoefs{D, T};
                               level_range::UnitRange{Int} = ws.level_range) where {D, T}
    return (z_fields, r_fields) -> begin
        _abec_jacobi_precond!(z_fields, r_fields, coefs, ws;
                                level_range = level_range)
    end
end
