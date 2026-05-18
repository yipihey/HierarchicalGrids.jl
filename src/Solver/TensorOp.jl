# ============================================================================
# TensorOp.jl — full tensor viscosity operator with cross-component
# coupling, matching the structure of AMReX's MLTensorOp.
#
# Operator (constant μ, λ — variable μ is a future extension):
#
#     (L u)_k = A · α(x) · u_k  -  B [ μ ∇² u_k  +  (μ + λ) ∂_k (∇·u) ]
#
# In 2D (k = 1):
#     (L u)_1 = A α u_1
#                - B μ (∂_x² u_1 + ∂_y² u_1)
#                - B (μ + λ) (∂_x² u_1 + ∂_x ∂_y u_2)
#
# Per-cell stencil at (i,j) for component 1:
#   diagonal       :  A α  +  2 B μ (1/h_x² + 1/h_y²)  +  2 B (μ+λ) / h_x²
#   ±x nbr (u_1)   : −B μ / h_x²                     −  B (μ+λ) / h_x²   (folded)
#   ±y nbr (u_1)   : −B μ / h_y²
#   corner (u_2)   : ∓B (μ+λ) / (4 h_x h_y)   (4 corner nbrs of OTHER component)
#
# This is the canonical Newtonian-fluid tensor viscosity. For
# incompressible flow with ∇·u = 0, the (μ+λ) ∂_k(∇·u) term vanishes
# and the operator decouples into a per-component scalar Laplacian
# (equivalent to `VectorABecProblem` with constant μ).
#
# Scope: cell-centered velocity, constant μ and λ, single-level /
# single-patch, periodic / Dirichlet / Neumann BCs.  Variable μ and
# multi-level FAC are future iterations.  Tier 2 #5 full of the
# AMReX-port roadmap.
# ============================================================================

"""
    TensorCoefs{D, T}

Coefficients for the tensor viscosity operator:
    (L u)_k = A · α(x) · u_k  -  B [ μ ∇² u_k  +  (μ + λ) ∂_k (∇·u) ]

Fields:
  * `A::T`, `B::T` — scalar weights, typically `A=1`, `B=Δt/ρ` for an
    implicit-viscosity backward-Euler step.
  * `mu::T`, `lambda::T` — Lamé-type constants. `lambda = 0` for
    incompressible (then the cross-coupling term vanishes).
  * `alpha::Vector{Vector{Vector{T}}}` — cell-centered α; same layout
    as `ABecCoefs.alpha`. Shared across velocity components.
"""
struct TensorCoefs{D, T}
    A::T
    B::T
    mu::T
    lambda::T
    alpha::Vector{Vector{Vector{T}}}
end

"""
    allocate_tensor_coefs(ws; A=1, B=1, mu=1, lambda=0, alpha_init=1)
        -> TensorCoefs

Allocate `TensorCoefs` matching the workspace hierarchy. `α` is filled
with `alpha_init`.
"""
function allocate_tensor_coefs(ws::MGWorkspace{D, T};
                                 A::T = one(T),
                                 B::T = one(T),
                                 mu::T = one(T),
                                 lambda::T = zero(T),
                                 alpha_init::T = one(T)) where {D, T}
    nL = length(ws.patch_N)
    alpha = Vector{Vector{Vector{T}}}(undef, nL)
    for ℓ in 1:nL
        nP = length(ws.patch_N[ℓ])
        per_patch = Vector{Vector{T}}(undef, nP)
        for pi in 1:nP
            n_storage = length(_raw_phi(ws.residual[ℓ][pi]))
            per_patch[pi] = fill(alpha_init, n_storage)
        end
        alpha[ℓ] = per_patch
    end
    return TensorCoefs{D, T}(A, B, mu, lambda, alpha)
end

"""
    TensorVelocity{D, T}

D-component cell-centered velocity field over the patch hierarchy.
Stored as `NTuple{D, Vector{Vector{NamedTuple}}}` so each component
reuses the existing `Vector{Vector{NamedTuple}}` infrastructure
(`fill_field!`, residuals, etc.).
"""
struct TensorVelocity{D, T}
    u::NTuple{D, Vector{Vector{NamedTuple}}}
end

"""
    allocate_tensor_velocity(ws::MGWorkspace{D,T}) -> TensorVelocity

Allocate a D-component cell-centered velocity field over the hierarchy.
"""
function allocate_tensor_velocity(ws::MGWorkspace{D, T}) where {D, T}
    comps = ntuple(d -> allocate_phi_rho(ws.ph), Val(D))
    return TensorVelocity{D, T}(comps)
end

"""
    fill_tensor_velocity!(U, ws, fs::NTuple{D,Function})

Fill each velocity component `d`'s `.phi` slot from `fs[d](x)` at
cell centers across all patches.
"""
function fill_tensor_velocity!(U::TensorVelocity{D, T},
                                 ws::MGWorkspace{D, T},
                                 fs::NTuple{D, Function}) where {D, T}
    for d in 1:D
        fill_field!(U.u[d], ws.ph, :phi, fs[d])
    end
    return U
end

# ----------------------------------------------------------------------------
# Apply: (L u)_k for all components, single level / single patch.
# ----------------------------------------------------------------------------

"""
    apply_tensor!(LU, U, coefs, ws; level=1, patch=1)

Compute `(L u)_k` for each component `k = 1..D` and write into
`LU.u[k][level][patch].phi`. Cross-component coupling via the
`(μ + λ) ∂_k (∇·u)` term is handled inline; periodic / Dirichlet /
Neumann BCs follow `ws.axis_kinds.kinds`.
"""
function apply_tensor!(LU::TensorVelocity{D, T},
                        U::TensorVelocity{D, T},
                        coefs::TensorCoefs{D, T},
                        ws::MGWorkspace{D, T};
                        level::Int = 1, patch::Int = 1) where {D, T}
    N = ws.patch_N[level][patch]
    dx = ws.patch_dx[level][patch]
    c2c = ws.cart_to_cell[level][patch]
    kinds = ws.axis_kinds.kinds
    A = coefs.A; B = coefs.B
    μ = coefs.mu; λ = coefs.lambda
    α = coefs.alpha[level][patch]
    invh = ntuple(d -> one(T) / dx[d], Val(D))

    u_raws = ntuple(d -> _raw_phi(U.u[d][level][patch])::Vector{T}, Val(D))
    L_raws = ntuple(d -> _raw_phi(LU.u[d][level][patch])::Vector{T}, Val(D))

    @inbounds for I in CartesianIndices(N)
        c = c2c[I]
        c == 0 && continue
        αc = α[c]
        # ∇·u at (i,j) — central differences over face-averaged
        # (component-centered) values. For cell-centered storage we use
        # the simple cell-difference div: ∇·u|_c = Σ_d (u_d|_{c+e_d} - u_d|_{c-e_d}) / (2 h_d)
        # We need this at the center cell and at axis-aligned neighbors
        # to evaluate ∂_k(∇·u) via central differences.
        for k in 1:D
            uk_c = u_raws[k][c]
            # Diffusion (Laplacian) term: -B μ ∇² u_k.
            lap_term = zero(T)
            for d in 1:D
                up = _read_cell_phi(u_raws[k], c2c, I, d, +1, N, kinds, uk_c)
                um = _read_cell_phi(u_raws[k], c2c, I, d, -1, N, kinds, uk_c)
                lap_term += (up - 2 * uk_c + um) * invh[d] * invh[d]
            end

            # Cross term: -B (μ + λ) ∂_k (∇·u). Compute as the second
            # derivative ∂_k ∂_d u_d (Einstein-summed over d).
            cross = zero(T)
            for d in 1:D
                if d == k
                    # ∂_k² u_k contribution.
                    up_k = _read_cell_phi(u_raws[k], c2c, I, k, +1, N, kinds, uk_c)
                    um_k = _read_cell_phi(u_raws[k], c2c, I, k, -1, N, kinds, uk_c)
                    cross += (up_k - 2 * uk_c + um_k) * invh[k] * invh[k]
                else
                    # ∂_k ∂_d u_d ≈ (u_d|_{++} - u_d|_{+-} - u_d|_{-+} + u_d|_{--})
                    #                 / (4 h_k h_d)
                    ud_pp = _read_cell_corner(u_raws[d], c2c, I, k, d, +1, +1, N, kinds, u_raws[d][c])
                    ud_pm = _read_cell_corner(u_raws[d], c2c, I, k, d, +1, -1, N, kinds, u_raws[d][c])
                    ud_mp = _read_cell_corner(u_raws[d], c2c, I, k, d, -1, +1, N, kinds, u_raws[d][c])
                    ud_mm = _read_cell_corner(u_raws[d], c2c, I, k, d, -1, -1, N, kinds, u_raws[d][c])
                    cross += (ud_pp - ud_pm - ud_mp + ud_mm) *
                              (invh[k] * invh[d] / 4)
                end
            end

            L_raws[k][c] = A * αc * uk_c - B * μ * lap_term -
                             B * (μ + λ) * cross
        end
    end
    return LU
end

# Read φ at (I + s·e_d) with BC handling. Falls back to the central
# value for Dirichlet (-φ_c), value (for Neumann / periodic-no-wrap).
@inline function _read_cell_phi(u::Vector{T}, c2c::Array{Int, D},
                                  I::CartesianIndex{D}, d::Int, s::Int,
                                  N::NTuple{D, Int},
                                  kinds::NTuple{D, Symbol},
                                  uc::T) where {D, T}
    inew = I[d] + s
    if 1 <= inew <= N[d]
        return @inbounds u[c2c[I + s * _unit_offsets(Val(D))[d]]]
    end
    kind = kinds[d]
    if kind === :periodic
        target = inew < 1 ? N[d] : 1
        return @inbounds u[c2c[CartesianIndex(ntuple(j -> j == d ? target : I[j], Val(D)))]]
    end
    kind === :dirichlet && return -uc
    return uc
end

# Read φ at (I + sk·e_k + sd·e_d) — used for mixed-derivative corner
# samples in the cross-component coupling. Periodic / Dirichlet /
# Neumann handled axis-by-axis.
@inline function _read_cell_corner(u::Vector{T}, c2c::Array{Int, D},
                                     I::CartesianIndex{D},
                                     k::Int, d::Int,
                                     sk::Int, sd::Int,
                                     N::NTuple{D, Int},
                                     kinds::NTuple{D, Symbol},
                                     uc::T) where {D, T}
    iknew = I[k] + sk
    idnew = I[d] + sd
    use_dirichlet = false
    target_k = iknew
    target_d = idnew

    if !(1 <= iknew <= N[k])
        kk = kinds[k]
        if kk === :periodic
            target_k = iknew < 1 ? N[k] : 1
        elseif kk === :dirichlet
            use_dirichlet = true
        else
            target_k = iknew < 1 ? 1 : N[k]
        end
    end
    if !(1 <= idnew <= N[d])
        kd = kinds[d]
        if kd === :periodic
            target_d = idnew < 1 ? N[d] : 1
        elseif kd === :dirichlet
            use_dirichlet = true
        else
            target_d = idnew < 1 ? 1 : N[d]
        end
    end
    use_dirichlet && return -uc
    I_target = CartesianIndex(ntuple(Val(D)) do j
        if j == k
            target_k
        elseif j == d
            target_d
        else
            I[j]
        end
    end)
    return @inbounds u[c2c[I_target]]
end

# ----------------------------------------------------------------------------
# Block-Jacobi smoother for the tensor operator.
#
# At each cell, solve the D × D block diagonal system
#     diag(L)[c]  ·  u_new[c]  =  rhs[c]  +  off-diagonal-fixed terms
# where off-diagonal terms are evaluated using the most recent component
# values. We use a Jacobi sweep (no red-black colouring of components —
# all D components are updated together at each cell). This is the
# AMReX-MLTensorOp convention.
# ----------------------------------------------------------------------------

"""
    gs_sweep_tensor!(U, RHS, coefs, ws; n_sweeps=1, level=1, patch=1)

Per-cell block-Jacobi smoother for the tensor operator. At each cell,
collect the off-diagonal contributions (from current iterate values of
all components), build the D × D diagonal block, and solve for
`u_new[c]`. Cells are visited once per sweep — no colouring (the cross-
coupling term reads from 4 corner cells of the OTHER component, which
makes a clean red/black colouring tricky).
"""
function gs_sweep_tensor!(U::TensorVelocity{D, T},
                            RHS::TensorVelocity{D, T},
                            coefs::TensorCoefs{D, T},
                            ws::MGWorkspace{D, T};
                            n_sweeps::Int = 1,
                            level::Int = 1, patch::Int = 1) where {D, T}
    N = ws.patch_N[level][patch]
    dx = ws.patch_dx[level][patch]
    c2c = ws.cart_to_cell[level][patch]
    kinds = ws.axis_kinds.kinds
    A = coefs.A; B = coefs.B
    μ = coefs.mu; λ = coefs.lambda
    α = coefs.alpha[level][patch]
    invh = ntuple(d -> one(T) / dx[d], Val(D))

    u_raws = ntuple(d -> _raw_phi(U.u[d][level][patch])::Vector{T}, Val(D))
    rhs_raws = ntuple(d -> _raw_phi(RHS.u[d][level][patch])::Vector{T}, Val(D))

    # Per-cell scratch (small, since D ≤ 3).
    block_diag = zeros(T, D)
    rhs_eff = zeros(T, D)

    for _ in 1:n_sweeps
        @inbounds for I in CartesianIndices(N)
            c = c2c[I]
            c == 0 && continue
            αc = α[c]
            for k in 1:D
                # Diagonal block coefficient (only the diagonal in the
                # D×D block at this cell — off-diagonal cross-component
                # mixed-derivative terms have zero diagonal contribution
                # at the center cell).
                diag_k = A * αc
                for d in 1:D
                    diag_k += 2 * B * μ * invh[d] * invh[d]
                    if d == k
                        # ∂_k² u_k extra term from (μ+λ) ∂_k(∇·u).
                        diag_k += 2 * B * (μ + λ) * invh[k] * invh[k]
                    end
                end
                block_diag[k] = diag_k

                # Off-diagonal contributions (everything except the
                # center cell's u_k). Build the effective RHS so that
                # diag_k · u_k_new = rhs_eff_k.
                uk_c = u_raws[k][c]
                lap_off = zero(T)
                for d in 1:D
                    up = _read_cell_phi(u_raws[k], c2c, I, d, +1, N, kinds, uk_c)
                    um = _read_cell_phi(u_raws[k], c2c, I, d, -1, N, kinds, uk_c)
                    lap_off += (up + um) * invh[d] * invh[d]
                end
                # ∂_k(∇·u) cross-coupling
                cross_off = zero(T)
                for d in 1:D
                    if d == k
                        up = _read_cell_phi(u_raws[k], c2c, I, k, +1, N, kinds, uk_c)
                        um = _read_cell_phi(u_raws[k], c2c, I, k, -1, N, kinds, uk_c)
                        cross_off += (up + um) * invh[k] * invh[k]
                    else
                        ud_pp = _read_cell_corner(u_raws[d], c2c, I, k, d, +1, +1, N, kinds, u_raws[d][c])
                        ud_pm = _read_cell_corner(u_raws[d], c2c, I, k, d, +1, -1, N, kinds, u_raws[d][c])
                        ud_mp = _read_cell_corner(u_raws[d], c2c, I, k, d, -1, +1, N, kinds, u_raws[d][c])
                        ud_mm = _read_cell_corner(u_raws[d], c2c, I, k, d, -1, -1, N, kinds, u_raws[d][c])
                        cross_off += (ud_pp - ud_pm - ud_mp + ud_mm) *
                                       (invh[k] * invh[d] / 4)
                    end
                end
                rhs_eff[k] = rhs_raws[k][c] + B * μ * lap_off +
                              B * (μ + λ) * cross_off
            end
            # Update all components from the per-cell block-Jacobi
            # solve. For purely-diagonal block this is one division per
            # component. (Cross-coupling within the same cell's u_k
            # vector is zero by construction — the (μ+λ) term only
            # creates cross terms at corner neighbours, not at the
            # center.)
            for k in 1:D
                u_raws[k][c] = rhs_eff[k] / block_diag[k]
            end
        end
    end
    return U
end

# ----------------------------------------------------------------------------
# Top-level solve via Krylov bridge.
# Flat layout has D × n_active entries (one block per cell, D entries per
# block), stored as [comp1_cells, comp2_cells, ..., compD_cells].
# ----------------------------------------------------------------------------

"""
    solve_tensor!(U, RHS, coefs, ws; method=:cg, tol=1e-9, maxiter=200,
                    level=1, patch=1) -> MGResult

Solve `L · U = RHS` for the tensor velocity field on a single patch
using the Krylov bridge. SPD-positive when `A·α ≥ 0`, `B·μ ≥ 0`, and
`B·(μ+λ) ≥ 0`, so CG is the default. The matrix-vector product runs
`apply_tensor!`.
"""
function solve_tensor!(U::TensorVelocity{D, T},
                        RHS::TensorVelocity{D, T},
                        coefs::TensorCoefs{D, T},
                        ws::MGWorkspace{D, T};
                        method::Symbol = :cg,
                        tol::Float64 = 1e-9,
                        maxiter::Int = 200,
                        level::Int = 1, patch::Int = 1) where {D, T}
    layout = flat_layout(ws; level_range = level:level)
    n_per_comp = layout.n
    n_total = D * n_per_comp
    op = _TensorOp{D, T}(ws, coefs, level, patch, layout, n_total,
                            allocate_tensor_velocity(ws),
                            allocate_tensor_velocity(ws))

    b = Vector{T}(undef, n_total)
    x0 = Vector{T}(undef, n_total)
    for d in 1:D
        rng = ((d - 1) * n_per_comp + 1):(d * n_per_comp)
        b_view = view(b, rng); x0_view = view(x0, rng)
        pack!(b_view, RHS.u[d], layout; field = :phi)
        pack!(x0_view, U.u[d], layout; field = :phi)
    end

    r0 = _flat_norm_tensor(op, b, x0)
    x, stats = _krylov_solve(method, op, b;
                              itmax = maxiter, atol = tol, rtol = tol,
                              history = true, verbose = 0)

    for d in 1:D
        rng = ((d - 1) * n_per_comp + 1):(d * n_per_comp)
        unpack!(U.u[d], view(x, rng), layout; field = :phi)
    end
    history = isempty(stats.residuals) ? [r0] : copy(stats.residuals)
    res_final = history[end]
    return MGResult(stats.niter, r0, res_final, stats.solved, history)
end

mutable struct _TensorOp{D, T}
    ws::MGWorkspace{D, T}
    coefs::TensorCoefs{D, T}
    level::Int
    patch::Int
    layout::FlatLayout{D, T}
    n::Int
    U_scratch::TensorVelocity{D, T}
    LU_scratch::TensorVelocity{D, T}
end
Base.size(op::_TensorOp) = (op.n, op.n)
Base.size(op::_TensorOp, ::Int) = op.n
Base.eltype(::_TensorOp{D, T}) where {D, T} = T

function LinearAlgebra.mul!(y::AbstractVector{T}, op::_TensorOp{D, T},
                              x::AbstractVector{T}) where {D, T}
    n_per_comp = op.layout.n
    for d in 1:D
        rng = ((d - 1) * n_per_comp + 1):(d * n_per_comp)
        unpack!(op.U_scratch.u[d], view(x, rng), op.layout; field = :phi)
    end
    apply_tensor!(op.LU_scratch, op.U_scratch, op.coefs, op.ws;
                    level = op.level, patch = op.patch)
    for d in 1:D
        rng = ((d - 1) * n_per_comp + 1):(d * n_per_comp)
        pack!(view(y, rng), op.LU_scratch.u[d], op.layout; field = :phi)
    end
    return y
end

Base.:*(op::_TensorOp{D, T}, x::AbstractVector{T}) where {D, T} =
    mul!(similar(x), op, x)

function _flat_norm_tensor(op::_TensorOp{D, T}, b::AbstractVector{T},
                             x0::AbstractVector{T}) where {D, T}
    Lx = similar(b)
    mul!(Lx, op, x0)
    s = zero(T)
    @inbounds @simd for i in eachindex(b)
        d = b[i] - Lx[i]
        s += d * d
    end
    return Float64(sqrt(s))
end
