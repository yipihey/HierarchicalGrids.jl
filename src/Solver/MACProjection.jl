# ============================================================================
# MACProjection.jl — MAC (face-centered) velocity projection
#
# Given a face-centered velocity u* and a cell-centered density ρ (or
# directly its face-averaged inverse 1/ρ), find a cell-centered scalar
# φ such that
#     u^{n+1} = u* - β ∇φ                with β = 1/ρ on each face
# satisfies the discrete divergence-free constraint
#     ∇ · u^{n+1} = 0.
# This becomes the variable-coefficient Poisson problem
#     ∇ · (β ∇φ) = ∇ · u*
# which we solve in SPD-positive ABec form  A=0, B=1, β=β, f = -∇·u*.
#
# Single-level / single-patch projection only for now (Tier 1 of the
# AMReX port roadmap).  Multi-level FAC projection is Tier 2.
# ============================================================================

"""
    FaceVelocity{D, T}

Face-centered vector field on a single patch. The d-th component is
sized `(N₁, …, N_d+1, …, N_D)`, indexed by face position along axis d
(`i_d ∈ 1:N_d+1`, `i_d=1` is the low-side boundary face, `i_d=N_d+1` the
high-side).
"""
struct FaceVelocity{D, T}
    u::NTuple{D, Array{T, D}}
end

"""
    allocate_face_velocity(ws, ℓ, pi=1; init=0) -> FaceVelocity

Allocate a face-centered velocity field on a single patch. Components are
filled with the constant `init`.
"""
function allocate_face_velocity(ws::MGWorkspace{D, T}, ℓ::Int, pi::Int = 1;
                                  init::T = zero(T)) where {D, T}
    N = ws.patch_N[ℓ][pi]
    comps = ntuple(Val(D)) do d
        fdims = ntuple(j -> j == d ? N[j] + 1 : N[j], Val(D))
        fill(init, fdims)
    end
    return FaceVelocity{D, T}(comps)
end

"""
    fill_face_velocity!(u, ws, ℓ, pi, fs::NTuple{D,Function})

Sample component `d` of the face velocity from `fs[d](x)`, evaluated at
the face center.
"""
function fill_face_velocity!(u::FaceVelocity{D, T},
                              ws::MGWorkspace{D, T},
                              ℓ::Int, pi::Int,
                              fs::NTuple{D, Function}) where {D, T}
    frame = patches_at(ws.ph, ℓ)[pi]
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    for d in 1:D
        ud = u.u[d]
        fd = fs[d]
        @inbounds for I in CartesianIndices(size(ud))
            x = ntuple(Val(D)) do j
                if j == d
                    frame.lo[d] + (I[d] - 1) * dx[d]
                else
                    frame.lo[j] + (I[j] - 0.5) * dx[j]
                end
            end
            ud[I] = T(fd(x))
        end
    end
    return u
end

"""
    face_divergence!(div_cell, u, ws, ℓ, pi)

Compute `div_cell[c] = ∑_d (u_d[face_high] - u_d[face_low]) / h_d` at each
cell of patch `(ℓ, pi)`. `div_cell` is a cell-centered field-view (with
`.phi` slot used as storage).
"""
function face_divergence!(div_cell::Vector{Vector{NamedTuple}},
                           u::FaceVelocity{D, T},
                           ws::MGWorkspace{D, T},
                           ℓ::Int, pi::Int) where {D, T}
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    div_raw = _raw_phi(div_cell[ℓ][pi])::Vector{T}
    invh = ntuple(d -> one(T) / dx[d], Val(D))
    @inbounds for I in CartesianIndices(N)
        c = c2c[I]
        c == 0 && continue
        s = zero(T)
        for d in 1:D
            u_low  = u.u[d][I]
            u_high = u.u[d][I + _unit_offsets(Val(D))[d]]
            s += (u_high - u_low) * invh[d]
        end
        div_raw[c] = s
    end
    return div_cell
end

"""
    subtract_grad!(u, phi, β, ws, ℓ, pi)

In-place update `u_d[face] -= β_d[face] · (φ_high - φ_low) / h_d` for each
interior face. Boundary faces (where one of `φ_high`, `φ_low` is outside
the patch) follow the workspace's BC kinds (periodic wrap or Dirichlet
ghost). This is the corrector step of MAC projection.
"""
function subtract_grad!(u::FaceVelocity{D, T},
                         phi_field::Vector{Vector{NamedTuple}},
                         β::NTuple{D, Array{T, D}},
                         ws::MGWorkspace{D, T},
                         ℓ::Int, pi::Int) where {D, T}
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    phi_raw = _raw_phi(phi_field[ℓ][pi])::Vector{T}
    kinds = ws.axis_kinds.kinds
    invh = ntuple(d -> one(T) / dx[d], Val(D))
    for d in 1:D
        ud = u.u[d]
        βd = β[d]
        @inbounds for I in CartesianIndices(size(ud))
            # Face position I along axis d corresponds to face between
            # cells (I[1], …, I[d]-1, …, I[D]) and (I[1], …, I[d], …, I[D]).
            i_low  = I[d] - 1
            i_high = I[d]
            φ_low  = _phi_at_or_bc(phi_raw, c2c, I, d, -1, N, kinds, i_low,  zero(T))
            φ_high = _phi_at_or_bc(phi_raw, c2c, I, d, +1, N, kinds, i_high, zero(T))
            grad = (φ_high - φ_low) * invh[d]
            ud[I] -= βd[I] * grad
        end
    end
    return u
end

# Return φ at a cell adjacent to a face, or the appropriate BC value if
# the index is outside the patch. `side` is ±1 (which side of the face we
# read), `i_target` is the desired index along axis d.
@inline function _phi_at_or_bc(phi_raw::Vector{T}, c2c::Array{Int, D},
                                I_face::CartesianIndex{D}, d::Int,
                                side::Int, N::NTuple{D, Int},
                                kinds::NTuple{D, Symbol},
                                i_target::Int, phi_default::T) where {D, T}
    if 1 <= i_target <= N[d]
        # Build the cell Cartesian index by replacing axis d.
        I_cell = CartesianIndex(ntuple(j -> j == d ? i_target : I_face[j], Val(D)))
        c = @inbounds c2c[I_cell]
        c == 0 && return phi_default
        return @inbounds phi_raw[c]
    end
    # i_target outside the patch — apply BC.
    if i_target < 1
        kinds[d] === :periodic && return _phi_wrap(phi_raw, c2c, I_face, d, N[d])
        kinds[d] === :dirichlet && return -_phi_inside(phi_raw, c2c, I_face, d, 1)
        return _phi_inside(phi_raw, c2c, I_face, d, 1)   # Neumann / reflect
    else  # i_target > N[d]
        kinds[d] === :periodic && return _phi_wrap(phi_raw, c2c, I_face, d, 1)
        kinds[d] === :dirichlet && return -_phi_inside(phi_raw, c2c, I_face, d, N[d])
        return _phi_inside(phi_raw, c2c, I_face, d, N[d])
    end
end

@inline function _phi_wrap(phi_raw::Vector{T}, c2c::Array{Int, D},
                            I_face::CartesianIndex{D}, d::Int,
                            i_target::Int) where {D, T}
    I_cell = CartesianIndex(ntuple(j -> j == d ? i_target : I_face[j], Val(D)))
    c = @inbounds c2c[I_cell]
    c == 0 && return zero(T)
    return @inbounds phi_raw[c]
end

@inline function _phi_inside(phi_raw::Vector{T}, c2c::Array{Int, D},
                              I_face::CartesianIndex{D}, d::Int,
                              i_target::Int) where {D, T}
    return _phi_wrap(phi_raw, c2c, I_face, d, i_target)
end

"""
    mac_project!(u, β, ws; tol=1e-9, maxiter=200, level=1, patch=1) -> MGResult

Project the face-centered velocity field `u` onto the discrete
divergence-free space, using face β = 1/ρ:
    1. Compute `rhs = -∇·u` at cell centers.
    2. Solve  -∇·(β ∇φ) = rhs  via ABec PCG.
    3. Update `u_d -= β_d · ∂_d φ` on every face.

Returns the `MGResult` from the inner solve. Single-level / single-patch
only in this Tier-1 implementation; multi-level / multi-patch projection
follows the MLNodeLaplacian Tier-2 port.
"""
function mac_project!(u::FaceVelocity{D, T},
                       β::NTuple{D, Array{T, D}},
                       ws::MGWorkspace{D, T};
                       tol::Float64 = 1e-9,
                       maxiter::Int = 200,
                       level::Int = 1,
                       patch::Int = 1,
                       verbose::Bool = false) where {D, T}
    @assert level == 1 "mac_project!: only single-level supported in Tier 1"
    @assert patch == 1 "mac_project!: only single-patch supported in Tier 1"

    # Scratch fields: φ (cell) + rhs (cell, via .rho slot).
    fields = allocate_phi_rho(ws.ph)
    face_divergence!(fields, u, ws, level, patch)
    # rhs = -div(u).  Stash into fields[1][1].rho and zero fields[1][1].phi.
    @inbounds for c in ws.patch_leaves[level][patch]
        fields[level][patch].rho[c] = (-fields[level][patch].phi[c][1],)
        fields[level][patch].phi[c] = (zero(T),)
    end

    coefs = allocate_abec_coefs(ws; A = zero(T), B = one(T))
    # Plug the user β into the ABec coefs (single level, single patch).
    @inbounds for d in 1:D
        copyto!(coefs.beta[level][patch][d], β[d])
    end

    r = pcg_composite_abec_solve!(fields, fields, coefs, ws;
                                    tol = tol, maxiter = maxiter,
                                    verbose = verbose,
                                    level_range = level:level)

    # Apply corrector: u -= β ∇φ on every face.
    subtract_grad!(u, fields, β, ws, level, patch)
    return r
end

"""
    face_divergence_l2(u, ws, ℓ=1, pi=1) -> Float64

Volume-weighted L² norm of `∇·u` evaluated at cell centers. Useful as a
diagnostic to confirm the projection achieved the target divergence.
"""
function face_divergence_l2(u::FaceVelocity{D, T},
                              ws::MGWorkspace{D, T},
                              ℓ::Int = 1, pi::Int = 1) where {D, T}
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    invh = ntuple(d -> one(T) / dx[d], Val(D))
    v_cell = prod(dx)
    s = zero(Float64)
    @inbounds for I in CartesianIndices(N)
        c = c2c[I]; c == 0 && continue
        d_sum = zero(T)
        for d in 1:D
            u_low  = u.u[d][I]
            u_high = u.u[d][I + _unit_offsets(Val(D))[d]]
            d_sum += (u_high - u_low) * invh[d]
        end
        s += Float64(d_sum)^2 * v_cell
    end
    return sqrt(s)
end
