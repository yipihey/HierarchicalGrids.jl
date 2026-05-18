# ============================================================================
# NodeLaplacian.jl — node-centered variable-coefficient Poisson operator,
# used for variable-density nodal pressure projection (IAMR / incflo /
# MAESTROeX / PeleLMeX).
#
# Discretisation:  L φ = ∇·(σ(x) ∇φ)  with σ at cell centers and φ at
# nodes (cell corners). The stencil is the standard 5-point (D=2) or
# 7-point (D=3) node Laplacian, with σ averaged over the cells whose
# common edge connects the two adjacent nodes:
#
#   D=2:   L φ_n = (1/h²) Σ_d Σ_{s=±1} σ_avg(n, n+s·e_d) · (φ_{n+s·e_d} - φ_n)
#   D=3:   same, summed over 3 axes.
#
# This is the "approximate projection" stencil — SPD-positive for σ > 0,
# trivially compatible with periodic BCs (nodes 1 and N+1 are identified),
# and amenable to CG / GMRES with Jacobi or AMG preconditioning via the
# existing Krylov bridge.
#
# Scope: single level / single patch for now. Multi-level FAC nodal
# projection (with C/F coupling) is on the same Tier-2 timeline.
# ============================================================================

"""
    NodeField{D, T}

Node-centered scalar field over the patch hierarchy. For each (level,
patch) with `(N₁, …, N_D)` cells, the corresponding node array has
shape `(N₁+1, …, N_D+1)`. For periodic axes, the user convention is
that node `1` and node `N_d+1` along axis d are identified — the apply
kernel reads either as appropriate.
"""
mutable struct NodeField{D, T}
    data::Vector{Vector{Array{T, D}}}
end

"""
    allocate_node_field(ws::MGWorkspace{D,T}; init=0) -> NodeField

Allocate a node-centered field for every patch.
"""
function allocate_node_field(ws::MGWorkspace{D, T};
                              init::T = zero(T)) where {D, T}
    nL = length(ws.patch_N)
    data = Vector{Vector{Array{T, D}}}(undef, nL)
    for ℓ in 1:nL
        nP = length(ws.patch_N[ℓ])
        per_patch = Vector{Array{T, D}}(undef, nP)
        for pi in 1:nP
            N = ws.patch_N[ℓ][pi]
            ndims_n = ntuple(j -> N[j] + 1, Val(D))
            per_patch[pi] = fill(init, ndims_n)
        end
        data[ℓ] = per_patch
    end
    return NodeField{D, T}(data)
end

"""
    fill_node_field!(φ::NodeField, ws, f)

Sample `f(x)` at every node position of every (level, patch).
"""
function fill_node_field!(φ::NodeField{D, T},
                            ws::MGWorkspace{D, T}, f) where {D, T}
    for ℓ in 1:length(φ.data)
        for pi in 1:length(φ.data[ℓ])
            frame = patches_at(ws.ph, ℓ)[pi]
            dx = ws.patch_dx[ℓ][pi]
            arr = φ.data[ℓ][pi]
            @inbounds for I in CartesianIndices(size(arr))
                x = ntuple(d -> frame.lo[d] + (I[d] - 1) * dx[d], Val(D))
                arr[I] = T(f(x))
            end
        end
    end
    return φ
end

"""
    NodeCoefs{D, T}

Variable-coefficient node-Laplacian coefficients. `sigma[ℓ][p][c]` is the
cell-centered diffusion coefficient at cell `c` of patch (level, p).
Scalar `B` weights the diffusion term so that the operator is
    L φ = -B · ∇·(σ ∇φ)
(negative sign so L is SPD-positive for σ > 0 and the standard convention
∇²φ = ρ becomes `solve(L, -ρ)`).
"""
struct NodeCoefs{D, T}
    B::T
    sigma::Vector{Vector{Vector{T}}}
end

"""
    allocate_node_coefs(ws; B=1, sigma_init=1) -> NodeCoefs

Allocate σ storage matching the cell layout of the hierarchy.
"""
function allocate_node_coefs(ws::MGWorkspace{D, T};
                              B::T = one(T),
                              sigma_init::T = one(T)) where {D, T}
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
    return NodeCoefs{D, T}(B, sigma)
end

"""
    fill_node_coefs_sigma!(coefs, ws, f)

Sample σ(x) at each cell center.
"""
function fill_node_coefs_sigma!(coefs::NodeCoefs{D, T},
                                  ws::MGWorkspace{D, T}, f) where {D, T}
    for ℓ in 1:length(coefs.sigma)
        for pi in 1:length(coefs.sigma[ℓ])
            frame = patches_at(ws.ph, ℓ)[pi]
            σ = coefs.sigma[ℓ][pi]
            @inbounds for c in ws.patch_leaves[ℓ][pi]
                lo, hi = cell_physical_box(frame, c)
                x = ntuple(d -> 0.5 * (lo[d] + hi[d]), Val(D))
                σ[c] = T(f(x))
            end
        end
    end
    return coefs
end

# ----------------------------------------------------------------------------
# Apply, GS smoother, and matrix assembly for the node Laplacian.
# Single-level single-patch implementation; periodic / Dirichlet / Neumann
# at the patch boundary handled inline.
# ----------------------------------------------------------------------------

# σ averaged over the cells whose common edge contains the segment
# between nodes I_node and I_node + s·e_d. For the 5-point node stencil
# in D dimensions, this is the mean of 2^{D-1} cells on the "side"
# of the edge.
@inline function _sigma_edge(σ_raw::Vector{T}, c2c::Array{Int, D},
                              N_cell::NTuple{D, Int}, I_node::CartesianIndex{D},
                              s::Int, d::Int) where {D, T}
    # The edge between nodes I_node and (I_node + s e_d) sits at face
    # position I_node along axis d (if s = +1) or I_node - 1 (if s = -1).
    # The cells whose face this edge lies on are those with axis-d
    # coordinate equal to (I_node if s=+1; I_node-1 if s=-1).
    fc = s == +1 ? I_node[d] : I_node[d] - 1
    # Out-of-bounds cell index ⇒ skip (treat σ on that side as 0).
    (fc < 1 || fc > N_cell[d]) && return zero(T)

    # Sum σ over the 2^{D-1} cells whose axis-j coordinate for j≠d is
    # in {I_node[j]-1, I_node[j]} (i.e. cells touching the node).
    s_sum = zero(T)
    s_count = 0
    @inbounds for off in 0:((1 << (D - 1)) - 1)
        # Construct the cell Cartesian index.
        valid = true
        ci = ntuple(Val(D)) do j
            if j == d
                fc
            else
                # bit offset within axis dimension (excluding d).
                k = j > d ? j - 2 : j - 1
                bit = (off >> k) & 1
                ix = I_node[j] - 1 + bit
                if ix < 1 || ix > N_cell[j]
                    valid = false
                    1   # placeholder, won't be used
                else
                    ix
                end
            end
        end
        valid || continue
        c = c2c[CartesianIndex(ci)]
        c == 0 && continue
        s_sum += σ_raw[c]
        s_count += 1
    end
    return s_count == 0 ? zero(T) : s_sum / T(s_count)
end

"""
    apply_node_laplacian!(Lphi::NodeField, phi::NodeField, coefs, ws;
                            level=1, patch=1)

Compute `Lphi = -B · ∇·(σ ∇φ)` at every node of patch `(level, patch)`,
using the natural 5-point (D=2) / 7-point (D=3) nodal stencil with σ
averaged across the cells whose common edge connects two nodes.
"""
function apply_node_laplacian!(Lphi::NodeField{D, T},
                                 phi::NodeField{D, T},
                                 coefs::NodeCoefs{D, T},
                                 ws::MGWorkspace{D, T};
                                 level::Int = 1, patch::Int = 1) where {D, T}
    N = ws.patch_N[level][patch]
    dx = ws.patch_dx[level][patch]
    c2c = ws.cart_to_cell[level][patch]
    σ_raw = coefs.sigma[level][patch]
    B = coefs.B
    kinds = ws.axis_kinds.kinds
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    n_arr = phi.data[level][patch]
    L_arr = Lphi.data[level][patch]

    Nn = size(n_arr)
    @inbounds for I in CartesianIndices(Nn)
        φ_c = n_arr[I]
        val = zero(T)
        for d in 1:D
            for s in (+1, -1)
                σ_e = _sigma_edge(σ_raw, c2c, N, I, s, d)
                φ_n = _node_neighbor(n_arr, I, d, s, Nn, kinds)
                val += σ_e * (φ_n - φ_c) * invh2[d]
            end
        end
        L_arr[I] = -B * val
    end
    return Lphi
end

# Read node value at I + s·e_d, with periodic / Dirichlet / Neumann
# fallback when stepping out of range.
@inline function _node_neighbor(arr::Array{T, D}, I::CartesianIndex{D},
                                  d::Int, s::Int,
                                  Nn::NTuple{D, Int},
                                  kinds::NTuple{D, Symbol}) where {D, T}
    inew = I[d] + s
    if 1 <= inew <= Nn[d]
        return @inbounds arr[I + s * _unit_offsets(Val(D))[d]]
    end
    # Out-of-range. Apply BC.
    kind = kinds[d]
    if kind === :periodic
        # Nodes 1 and Nn[d] are identified.  Wrap from one boundary to
        # the interior on the opposite side.
        target = inew < 1 ? Nn[d] - 1 : 2
        return @inbounds arr[CartesianIndex(ntuple(j -> j == d ? target : I[j], Val(D)))]
    end
    if kind === :dirichlet
        # φ = 0 on the boundary nodes themselves; off-boundary ghost
        # uses -φ_c (reflection through zero).
        return -arr[I]
    end
    # Neumann or fallback: zero-flux ghost ≡ same value.
    return @inbounds arr[I]
end

"""
    gs_sweep_node!(phi::NodeField, rho::NodeField, coefs, ws; n_sweeps=1,
                     level=1, patch=1, reverse_colours=false)

Red-black Gauss-Seidel relaxation of `L φ = ρ` at the nodal level.
Update at a node:
    φ_n := (Σ_d σ_avg · (φ_low + φ_high) / h²  -  ρ_n / B) /
           (Σ_d σ_avg / h² · 2)
"""
function gs_sweep_node!(phi::NodeField{D, T},
                          rho::NodeField{D, T},
                          coefs::NodeCoefs{D, T},
                          ws::MGWorkspace{D, T};
                          n_sweeps::Int = 1,
                          level::Int = 1, patch::Int = 1,
                          reverse_colours::Bool = false) where {D, T}
    N = ws.patch_N[level][patch]
    dx = ws.patch_dx[level][patch]
    c2c = ws.cart_to_cell[level][patch]
    σ_raw = coefs.sigma[level][patch]
    B = coefs.B
    kinds = ws.axis_kinds.kinds
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    n_arr = phi.data[level][patch]
    rho_arr = rho.data[level][patch]
    Nn = size(n_arr)

    colour_order = reverse_colours ? (1:-1:0) : (0:1)
    for _ in 1:n_sweeps
        for colour in colour_order
            @inbounds for I in CartesianIndices(Nn)
                s = 0
                for d in 1:D; s += I[d]; end
                (s & 1) == colour || continue

                nbr_sum = zero(T)
                diag = zero(T)
                for d in 1:D
                    for s_dir in (+1, -1)
                        σ_e = _sigma_edge(σ_raw, c2c, N, I, s_dir, d)
                        φ_n = _node_neighbor(n_arr, I, d, s_dir, Nn, kinds)
                        nbr_sum += σ_e * φ_n * invh2[d]
                        diag    += σ_e * invh2[d]
                    end
                end
                # Solve  -B (nbr_sum - diag φ) = rho_arr[I]   for φ:
                #        -B nbr_sum + B diag φ = ρ
                #        φ = (ρ + B nbr_sum) / (B diag)
                # Guard against diag=0 (isolated boundary point with σ=0).
                if diag > 0
                    n_arr[I] = (rho_arr[I] + B * nbr_sum) / (B * diag)
                end
            end
        end
    end
    return phi
end

"""
    solve_node_laplacian!(phi::NodeField, rho::NodeField, coefs, ws;
                            method=:cg, tol=1e-9, maxiter=200,
                            level=1, patch=1)

Solve `L φ = ρ` at the nodal level via the Krylov bridge using a
flat-vector wrapper. SPD operator under σ > 0; CG is the default.

Returns an `MGResult`.
"""
function solve_node_laplacian!(phi::NodeField{D, T},
                                 rho::NodeField{D, T},
                                 coefs::NodeCoefs{D, T},
                                 ws::MGWorkspace{D, T};
                                 method::Symbol = :cg,
                                 tol::Float64 = 1e-9,
                                 maxiter::Int = 200,
                                 level::Int = 1, patch::Int = 1) where {D, T}
    nn = length(phi.data[level][patch])
    # Matvec callback.
    L_tmp = allocate_node_field(ws)
    op = _NodeLaplOp{D, T}(ws, coefs, level, patch, nn,
                              phi.data[level][patch],
                              L_tmp.data[level][patch])

    b = vec(rho.data[level][patch])
    x0 = vec(phi.data[level][patch])
    r0 = _flat_norm_node(op, b, x0)

    x, stats = _krylov_solve(method, op, b;
                              itmax = maxiter, atol = tol, rtol = tol,
                              history = true, verbose = 0)
    # Copy result back.
    copyto!(phi.data[level][patch], reshape(x, size(phi.data[level][patch])))
    history = isempty(stats.residuals) ? [r0] : copy(stats.residuals)
    res_final = history[end]
    return MGResult(stats.niter, r0, res_final, stats.solved, history)
end

# Function-object linear operator for the node Laplacian.
mutable struct _NodeLaplOp{D, T}
    ws::MGWorkspace{D, T}
    coefs::NodeCoefs{D, T}
    level::Int
    patch::Int
    n::Int
    phi_arr::Array{T, D}
    L_arr::Array{T, D}
end
Base.size(op::_NodeLaplOp) = (op.n, op.n)
Base.size(op::_NodeLaplOp, ::Int) = op.n
Base.eltype(::_NodeLaplOp{D, T}) where {D, T} = T

function LinearAlgebra.mul!(y::AbstractVector{T}, op::_NodeLaplOp{D, T},
                              x::AbstractVector{T}) where {D, T}
    copyto!(op.phi_arr, reshape(x, size(op.phi_arr)))
    # Run the per-patch apply. Construct two NodeFields temporarily —
    # share the storage.
    phi_field = NodeField{D, T}(Vector{Vector{Array{T, D}}}(undef, length(op.ws.patch_N)))
    Lphi_field = NodeField{D, T}(Vector{Vector{Array{T, D}}}(undef, length(op.ws.patch_N)))
    for ℓ in 1:length(op.ws.patch_N)
        nP = length(op.ws.patch_N[ℓ])
        phi_field.data[ℓ] = Vector{Array{T, D}}(undef, nP)
        Lphi_field.data[ℓ] = Vector{Array{T, D}}(undef, nP)
        for pi in 1:nP
            if ℓ == op.level && pi == op.patch
                phi_field.data[ℓ][pi] = op.phi_arr
                Lphi_field.data[ℓ][pi] = op.L_arr
            else
                Nl = op.ws.patch_N[ℓ][pi]
                ndims_n = ntuple(j -> Nl[j] + 1, Val(D))
                phi_field.data[ℓ][pi] = zeros(T, ndims_n)
                Lphi_field.data[ℓ][pi] = zeros(T, ndims_n)
            end
        end
    end
    apply_node_laplacian!(Lphi_field, phi_field, op.coefs, op.ws;
                            level = op.level, patch = op.patch)
    copyto!(y, vec(op.L_arr))
    return y
end

Base.:*(op::_NodeLaplOp{D, T}, x::AbstractVector{T}) where {D, T} =
    mul!(similar(x), op, x)

function _flat_norm_node(op::_NodeLaplOp{D, T}, b::AbstractVector{T},
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
