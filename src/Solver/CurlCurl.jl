# ============================================================================
# CurlCurl.jl — 2D edge-centered curl-curl operator with cross-component
# coupling. Foundation for magnetostatic (WarpX) and resistive MHD.
#
# 2D in-plane operator on `EdgeField`:
#
#     (L A)_x = -∂_y² A_x + ∂_y ∂_x A_y + σ A_x
#     (L A)_y = -∂_x² A_y + ∂_x ∂_y A_x + σ A_y
#
# Equivalently, with curl_z = ∂_x A_y - ∂_y A_x living at cell centers,
#     (L A)_x = +∂_y (curl_z) + σ A_x
#     (L A)_y = -∂_x (curl_z) + σ A_y
# which is the standard ∇ × (∇ × A) + σ A.
#
# This matches AMReX MLCurlCurl in 2D with μ = 1. Variable μ (μ⁻¹
# weighting of the inner curl) is a straightforward generalisation by
# multiplying the cell-centered curl_z by μ⁻¹_cell before taking the
# outer divergence.
#
# Edge layout (matches `EdgeFields.jl`):
#   A_x[i, j]  for i = 1..N_x,    j = 1..N_y + 1
#   A_y[i, j]  for i = 1..N_x + 1, j = 1..N_y
# curl_z[i, j] lives at cell (i, j), i = 1..N_x, j = 1..N_y:
#   curl_z = (A_y[i+1, j] - A_y[i, j])/h_x  -  (A_x[i, j+1] - A_x[i, j])/h_y
#
# Scope: single-level / single-patch, periodic / Dirichlet / Neumann BCs,
# constant μ.
# ============================================================================

"""
    CurlCurlCoefs{T}

Coefficients for the 2D curl-curl operator
    (L A)_d = (∇ × (∇ × A))_d + σ A_d.

Fields:
  * `mu_inv::T` — inverse permeability (multiplies the inner curl); the
    full operator becomes `∇ × (μ⁻¹ ∇ × A) + σ A`. For constant μ = 1,
    `mu_inv = 1`.
  * `sigma::Vector{Vector{Vector{T}}}` — cell-centered identity weight
    `σ(x)`. Same layout as `ABecCoefs.alpha`.
"""
struct CurlCurlCoefs{T}
    mu_inv::T
    sigma::Vector{Vector{Vector{T}}}
end

"""
    allocate_curlcurl_coefs(ws::MGWorkspace{D,T}; mu_inv=1, sigma_init=1)
        -> CurlCurlCoefs

Allocate cell-centered σ storage initialised to `sigma_init`. Caller can
fill non-uniform σ afterwards via index assignment.
"""
function allocate_curlcurl_coefs(ws::MGWorkspace{D, T};
                                  mu_inv::T = one(T),
                                  sigma_init::T = one(T)) where {D, T}
    @assert D == 2 "CurlCurlCoefs: only D=2 supported (in-plane curl)"
    nL = length(ws.patch_N)
    sigma = Vector{Vector{Vector{T}}}(undef, nL)
    for ℓ in 1:nL
        nP = length(ws.patch_N[ℓ])
        per_patch = Vector{Vector{T}}(undef, nP)
        for pi in 1:nP
            n_storage = length(_raw_phi(ws.residual[ℓ][pi]))
            per_patch[pi] = fill(sigma_init, n_storage)
        end
        sigma[ℓ] = per_patch
    end
    return CurlCurlCoefs{T}(mu_inv, sigma)
end

"""
    apply_curl_curl!(LA::EdgeField, A::EdgeField, coefs, ws; level=1, patch=1)

Compute `(L A)_d = (∇ × (μ⁻¹ ∇ × A))_d + σ A_d` for each edge component.
Boundary handling follows `ws.axis_kinds.kinds` axis-by-axis.
"""
function apply_curl_curl!(LA::EdgeField{2, T},
                            A::EdgeField{2, T},
                            coefs::CurlCurlCoefs{T},
                            ws::MGWorkspace{2, T};
                            level::Int = 1, patch::Int = 1) where {T}
    Nc = ws.patch_N[level][patch]
    dx = ws.patch_dx[level][patch]
    c2c = ws.cart_to_cell[level][patch]
    kinds = ws.axis_kinds.kinds
    σ = coefs.sigma[level][patch]
    μ_inv = coefs.mu_inv

    Ax = A.u[1]; Ay = A.u[2]
    LAx = LA.u[1]; LAy = LA.u[2]
    inv_hx = one(T) / dx[1]
    inv_hy = one(T) / dx[2]

    # Pre-compute curl_z at every cell (with BC-extended ghost reads).
    # curl_z[i, j] = (Ay[i+1, j] - Ay[i, j])/h_x  -  (Ax[i, j+1] - Ax[i, j])/h_y
    curl_z = Matrix{T}(undef, Nc[1], Nc[2])
    @inbounds for j in 1:Nc[2], i in 1:Nc[1]
        ay_high = Ay[i + 1, j]
        ay_low  = Ay[i,     j]
        ax_high = Ax[i, j + 1]
        ax_low  = Ax[i, j]
        curl_z[i, j] = μ_inv * ((ay_high - ay_low) * inv_hx -
                                  (ax_high - ax_low) * inv_hy)
    end

    # (L A)_x[i, j] for i = 1..Nx, j = 1..Ny+1.
    # ∂_y (curl_z) evaluated at edge (i, j) needs curl_z[i, j-1] and
    # curl_z[i, j] (the two cells flanking the x-edge in the y-direction).
    @inbounds for j in 1:Nc[2] + 1, i in 1:Nc[1]
        cz_low  = _curl_z_at(curl_z, i, j - 1, Nc, kinds, 2)
        cz_high = _curl_z_at(curl_z, i, j,     Nc, kinds, 2)
        # σ at the edge: average σ of the two adjacent cells.
        σ_avg = _sigma_at_x_edge(σ, c2c, i, j, Nc, kinds)
        LAx[i, j] = (cz_high - cz_low) * inv_hy + σ_avg * Ax[i, j]
    end
    # (L A)_y[i, j] for i = 1..Nx+1, j = 1..Ny.
    # ∂_x(curl_z) evaluated at edge (i, j) needs curl_z[i-1, j] and curl_z[i, j].
    # Sign: (L A)_y = -∂_x(curl_z) + σ A_y. The "-" is implicit in the
    # ∂_x sign convention (curl_z = ∂_x A_y - ∂_y A_x ⇒ ∂_x curl_z carries
    # the natural direction; we negate it).
    @inbounds for j in 1:Nc[2], i in 1:Nc[1] + 1
        cz_low  = _curl_z_at(curl_z, i - 1, j, Nc, kinds, 1)
        cz_high = _curl_z_at(curl_z, i,     j, Nc, kinds, 1)
        σ_avg = _sigma_at_y_edge(σ, c2c, i, j, Nc, kinds)
        LAy[i, j] = -(cz_high - cz_low) * inv_hx + σ_avg * Ay[i, j]
    end
    return LA
end

# Read curl_z at cell (i, j) with BC handling along the perpendicular
# axis `paxis`. (For x-edges we evaluate ∂_y curl_z, so out-of-bounds
# along y triggers BC fallback in axis 2; for y-edges similarly axis 1.)
@inline function _curl_z_at(curl_z::Matrix{T}, i::Int, j::Int,
                              Nc::NTuple{2, Int},
                              kinds::NTuple{2, Symbol},
                              paxis::Int) where {T}
    in_x = 1 <= i <= Nc[1]
    in_y = 1 <= j <= Nc[2]
    if in_x && in_y
        return @inbounds curl_z[i, j]
    end
    # Reflect / wrap.
    ii = clamp(i, 1, Nc[1])
    jj = clamp(j, 1, Nc[2])
    # Periodic wrap if applicable on the out-of-range axis.
    if !in_x
        if kinds[1] === :periodic
            ii = i < 1 ? Nc[1] : 1
        elseif kinds[1] === :dirichlet
            # σ A_x term + ∂y curl_z absorbs the dirichlet via boundary
            # condition automatically; return 0 for the curl ghost.
            return zero(T)
        end
    end
    if !in_y
        if kinds[2] === :periodic
            jj = j < 1 ? Nc[2] : 1
        elseif kinds[2] === :dirichlet
            return zero(T)
        end
    end
    return @inbounds curl_z[ii, jj]
end

# σ averaged at an x-edge (i, j) — between cells (i, j-1) and (i, j),
# both along axis 2 = y.
@inline function _sigma_at_x_edge(σ::Vector{T}, c2c::Array{Int, 2},
                                    i::Int, j::Int, Nc::NTuple{2, Int},
                                    kinds::NTuple{2, Symbol}) where {T}
    s = zero(T); n = 0
    for ds in (-1, 0)
        jj = j + ds
        if 1 <= jj <= Nc[2]
            c = @inbounds c2c[i, jj]
            c == 0 && continue
            s += σ[c]; n += 1
        end
    end
    return n == 0 ? zero(T) : s / T(n)
end

# σ averaged at a y-edge (i, j) — between cells (i-1, j) and (i, j),
# along axis 1 = x.
@inline function _sigma_at_y_edge(σ::Vector{T}, c2c::Array{Int, 2},
                                    i::Int, j::Int, Nc::NTuple{2, Int},
                                    kinds::NTuple{2, Symbol}) where {T}
    s = zero(T); n = 0
    for ds in (-1, 0)
        ii = i + ds
        if 1 <= ii <= Nc[1]
            c = @inbounds c2c[ii, j]
            c == 0 && continue
            s += σ[c]; n += 1
        end
    end
    return n == 0 ? zero(T) : s / T(n)
end

"""
    solve_curl_curl!(A::EdgeField, RHS::EdgeField, coefs, ws;
                       method=:cg, tol=1e-9, maxiter=200,
                       level=1, patch=1) -> MGResult

Solve `L A = RHS` for the in-plane curl-curl operator. SPD-positive for
σ > 0 (the σ identity ensures the operator is SPD); CG is the natural
choice. For σ = 0 the operator has a (gradient) nullspace; the solve
returns a member of the orthogonal complement.

Uses the Krylov bridge with a per-call matvec wrapper. Single-level
single-patch only in this initial port.
"""
function solve_curl_curl!(A::EdgeField{2, T},
                            RHS::EdgeField{2, T},
                            coefs::CurlCurlCoefs{T},
                            ws::MGWorkspace{2, T};
                            method::Symbol = :cg,
                            tol::Float64 = 1e-9,
                            maxiter::Int = 200,
                            level::Int = 1, patch::Int = 1) where {T}
    n1 = length(A.u[1]); n2 = length(A.u[2])
    n_total = n1 + n2
    op = _CurlCurlOp{T}(ws, coefs, level, patch, n_total, n1, n2,
                          allocate_edge_field(ws, level, patch),
                          allocate_edge_field(ws, level, patch))

    b = Vector{T}(undef, n_total)
    x0 = Vector{T}(undef, n_total)
    @inbounds for i in 1:n1
        b[i] = RHS.u[1][i]
        x0[i] = A.u[1][i]
    end
    @inbounds for i in 1:n2
        b[n1 + i] = RHS.u[2][i]
        x0[n1 + i] = A.u[2][i]
    end

    r0 = _flat_norm_curl(op, b, x0)
    x, stats = _krylov_solve(method, op, b;
                              itmax = maxiter, atol = tol, rtol = tol,
                              history = true, verbose = 0)

    @inbounds for i in 1:n1; A.u[1][i] = x[i]; end
    @inbounds for i in 1:n2; A.u[2][i] = x[n1 + i]; end
    history = isempty(stats.residuals) ? [r0] : copy(stats.residuals)
    res_final = history[end]
    return MGResult(stats.niter, r0, res_final, stats.solved, history)
end

mutable struct _CurlCurlOp{T}
    ws::MGWorkspace{2, T}
    coefs::CurlCurlCoefs{T}
    level::Int
    patch::Int
    n::Int
    n1::Int
    n2::Int
    A_scratch::EdgeField{2, T}
    LA_scratch::EdgeField{2, T}
end
Base.size(op::_CurlCurlOp) = (op.n, op.n)
Base.size(op::_CurlCurlOp, ::Int) = op.n
Base.eltype(::_CurlCurlOp{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector{T}, op::_CurlCurlOp{T},
                              x::AbstractVector{T}) where {T}
    @inbounds for i in 1:op.n1; op.A_scratch.u[1][i] = x[i]; end
    @inbounds for i in 1:op.n2; op.A_scratch.u[2][i] = x[op.n1 + i]; end
    apply_curl_curl!(op.LA_scratch, op.A_scratch, op.coefs, op.ws;
                       level = op.level, patch = op.patch)
    @inbounds for i in 1:op.n1; y[i] = op.LA_scratch.u[1][i]; end
    @inbounds for i in 1:op.n2; y[op.n1 + i] = op.LA_scratch.u[2][i]; end
    return y
end

Base.:*(op::_CurlCurlOp{T}, x::AbstractVector{T}) where {T} =
    mul!(similar(x), op, x)

function _flat_norm_curl(op::_CurlCurlOp{T}, b::AbstractVector{T},
                          x0::AbstractVector{T}) where {T}
    Lx = similar(b)
    mul!(Lx, op, x0)
    s = zero(T)
    @inbounds @simd for i in eachindex(b)
        d = b[i] - Lx[i]
        s += d * d
    end
    return Float64(sqrt(s))
end
