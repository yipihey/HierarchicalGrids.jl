# ============================================================================
# EdgeFields.jl — edge-centered vector field storage and the
# scalar-Laplacian-per-component operator on edges. Foundation for the
# Tier-3 #10 MLCurlCurl port (magnetostatic / resistive MHD).
#
# In 2D, "edges" are x-edges (parallel to the x-axis) and y-edges. The
# storage layout matches `FaceVelocity` from `MACProjection.jl`:
#
#   E.u[1] :: Array{T, D}   sized (N₁,   N₂+1, …, N_D+1)  ← x-edges
#   E.u[2] :: Array{T, D}   sized (N₁+1, N₂,   …, N_D+1)  ← y-edges
#   …
#
# That is, axis `d`'s edge array has cell-count `N_d` along its own
# axis (since edges run parallel to it) and `N_j + 1` along every
# perpendicular axis `j ≠ d`.
#
# In 3D the storage extends naturally: x-edges have shape (N₁, N₂+1, N₃+1),
# y-edges (N₁+1, N₂, N₃+1), z-edges (N₁+1, N₂+1, N₃).
#
# Initial scope (Tier 3 #10 skeleton):
#   * `EdgeField{D, T}` storage + allocators.
#   * `apply_edge_laplacian!` — per-component scalar Laplacian on each
#     edge component (the LEADING TERM of the curl-curl operator under
#     the Coulomb gauge ∇·A=0).
#   * `solve_edge_laplacian!` via the Krylov bridge.
# Full curl-curl with cross-component coupling (and AMR FAC) is the
# next iteration.
# ============================================================================

"""
    EdgeField{D, T}

Edge-centered vector field on a single patch. Same layout as
`FaceVelocity`: `u[d]` is sized
`(N₁, …, N_d,  …, N_D + 1)`  with the `N_d` dimension along axis d
(edge running parallel to axis d) and `N_j + 1` for j ≠ d.
"""
struct EdgeField{D, T}
    u::NTuple{D, Array{T, D}}
end

"""
    allocate_edge_field(ws::MGWorkspace{D,T}, ℓ=1, pi=1; init=0) -> EdgeField

Allocate an edge-centered vector field on a single patch.
"""
function allocate_edge_field(ws::MGWorkspace{D, T},
                               ℓ::Int = 1, pi::Int = 1;
                               init::T = zero(T)) where {D, T}
    N = ws.patch_N[ℓ][pi]
    comps = ntuple(Val(D)) do d
        edims = ntuple(j -> j == d ? N[j] : N[j] + 1, Val(D))
        fill(init, edims)
    end
    return EdgeField{D, T}(comps)
end

"""
    fill_edge_field!(E::EdgeField, ws, ℓ, pi, fs::NTuple{D, Function})

Sample each component `d` from `fs[d](x)` at the corresponding edge
center. Edge `(i₁, …, i_d, …)` along axis d has physical position
`(x₁_node, …, x_d_center, …)`: cell-center along the d-axis,
node-position along the perpendiculars.
"""
function fill_edge_field!(E::EdgeField{D, T},
                            ws::MGWorkspace{D, T},
                            ℓ::Int, pi::Int,
                            fs::NTuple{D, Function}) where {D, T}
    frame = patches_at(ws.ph, ℓ)[pi]
    dx = ws.patch_dx[ℓ][pi]
    for d in 1:D
        comp = E.u[d]
        f = fs[d]
        @inbounds for I in CartesianIndices(size(comp))
            x = ntuple(Val(D)) do j
                if j == d
                    # cell-center along axis d
                    frame.lo[d] + (I[d] - 0.5) * dx[d]
                else
                    # node position along axis j
                    frame.lo[j] + (I[j] - 1) * dx[j]
                end
            end
            comp[I] = T(f(x))
        end
    end
    return E
end

"""
    apply_edge_laplacian!(LE::EdgeField, E::EdgeField, ws, ℓ=1, pi=1;
                            kinds = ws.axis_kinds.kinds)

Compute the per-component Laplacian: `(LE).u[d] = ∇² (E.u[d])` for each
component independently. Treats each edge component as a scalar field
on its own shifted grid; periodic / Dirichlet / Neumann BCs follow
`kinds`.

This is the leading term of the full curl-curl operator in the Coulomb
gauge ∇·A = 0; cross-component coupling is added in a future iteration.
"""
function apply_edge_laplacian!(LE::EdgeField{D, T},
                                 E::EdgeField{D, T},
                                 ws::MGWorkspace{D, T},
                                 ℓ::Int = 1, pi::Int = 1;
                                 kinds::NTuple{D, Symbol} =
                                     ws.axis_kinds.kinds) where {D, T}
    dx = ws.patch_dx[ℓ][pi]
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    for d_comp in 1:D
        u = E.u[d_comp]
        L = LE.u[d_comp]
        Nu = size(u)
        @inbounds for I in CartesianIndices(Nu)
            φ = u[I]
            val = zero(T)
            for d in 1:D
                # +d neighbour
                φp = _edge_neighbor(u, I, d, +1, Nu, kinds)
                φm = _edge_neighbor(u, I, d, -1, Nu, kinds)
                val += (φp - 2 * φ + φm) * invh2[d]
            end
            L[I] = val
        end
    end
    return LE
end

# Edge neighbor with periodic / Dirichlet / Neumann fallback. Note the
# edge index runs 1:N_d along its own axis (cell-center spacing) and
# 1:N_j+1 along perpendicular axes (node spacing).
@inline function _edge_neighbor(u::Array{T, D}, I::CartesianIndex{D},
                                  d::Int, s::Int, Nu::NTuple{D, Int},
                                  kinds::NTuple{D, Symbol}) where {D, T}
    inew = I[d] + s
    if 1 <= inew <= Nu[d]
        return @inbounds u[I + s * _unit_offsets(Val(D))[d]]
    end
    kind = kinds[d]
    if kind === :periodic
        # Endpoints depend on whether axis d is the edge's own axis or
        # a perpendicular axis. For the own axis (size N_d, cell-center
        # spacing) wrap to 1 ↔ N_d. For perpendiculars (size N_j+1,
        # node spacing) identify 1 with N_j+1.
        target = inew < 1 ? Nu[d] : 1
        return @inbounds u[CartesianIndex(ntuple(j -> j == d ? target : I[j], Val(D)))]
    end
    if kind === :dirichlet
        return -u[I]
    end
    return @inbounds u[I]   # Neumann fallback
end

"""
    solve_edge_laplacian!(E::EdgeField, RHS::EdgeField, ws, ℓ=1, pi=1;
                            method=:cg, tol=1e-9, maxiter=200)

Solve `∇² E_d = RHS_d` for each component independently (e.g.
component-wise vector Poisson for magnetostatic in the Coulomb gauge).

Returns an `NTuple{D, MGResult}`.
"""
function solve_edge_laplacian!(E::EdgeField{D, T},
                                 RHS::EdgeField{D, T},
                                 ws::MGWorkspace{D, T},
                                 ℓ::Int = 1, pi::Int = 1;
                                 method::Symbol = :cg,
                                 tol::Float64 = 1e-9,
                                 maxiter::Int = 200) where {D, T}
    return ntuple(Val(D)) do d_comp
        op = _EdgeLaplOp{D, T}(ws, ℓ, pi, d_comp, length(E.u[d_comp]),
                                  E.u[d_comp], similar(E.u[d_comp]))
        b = vec(RHS.u[d_comp])
        x0 = vec(E.u[d_comp])
        r0 = _flat_norm_edge(op, b, x0)
        x, stats = _krylov_solve(method, op, b;
                                    itmax = maxiter, atol = tol, rtol = tol,
                                    history = true, verbose = 0)
        copyto!(E.u[d_comp], reshape(x, size(E.u[d_comp])))
        history = isempty(stats.residuals) ? [r0] : copy(stats.residuals)
        res_final = history[end]
        MGResult(stats.niter, r0, res_final, stats.solved, history)
    end
end

mutable struct _EdgeLaplOp{D, T}
    ws::MGWorkspace{D, T}
    level::Int
    patch::Int
    d_comp::Int
    n::Int
    u_arr::Array{T, D}
    L_arr::Array{T, D}
end
Base.size(op::_EdgeLaplOp) = (op.n, op.n)
Base.size(op::_EdgeLaplOp, ::Int) = op.n
Base.eltype(::_EdgeLaplOp{D, T}) where {D, T} = T

function LinearAlgebra.mul!(y::AbstractVector{T}, op::_EdgeLaplOp{D, T},
                              x::AbstractVector{T}) where {D, T}
    copyto!(op.u_arr, reshape(x, size(op.u_arr)))
    kinds = op.ws.axis_kinds.kinds
    dx = op.ws.patch_dx[op.level][op.patch]
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    Nu = size(op.u_arr)
    @inbounds for I in CartesianIndices(Nu)
        φ = op.u_arr[I]
        val = zero(T)
        for d in 1:D
            φp = _edge_neighbor(op.u_arr, I, d, +1, Nu, kinds)
            φm = _edge_neighbor(op.u_arr, I, d, -1, Nu, kinds)
            val += (φp - 2 * φ + φm) * invh2[d]
        end
        op.L_arr[I] = -val  # SPD-positive ⇒ A = -∇²
    end
    copyto!(y, vec(op.L_arr))
    return y
end

Base.:*(op::_EdgeLaplOp{D, T}, x::AbstractVector{T}) where {D, T} =
    mul!(similar(x), op, x)

function _flat_norm_edge(op::_EdgeLaplOp{D, T}, b::AbstractVector{T},
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
