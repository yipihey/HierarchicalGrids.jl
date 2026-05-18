# ============================================================================
# StiffChemistry.jl — per-cell stiff ODE integrator for reactive source
# terms (combustion, network chemistry, cooling).
#
# Each cell carries a state vector y ∈ R^N (species mass fractions, plus
# optionally energy / temperature). The user supplies:
#   * `f!(dy, y, p, t)` — RHS of the ODE,
#   * optionally `J!(J, y, p, t)` — analytical Jacobian (otherwise we
#     finite-difference),
#   * `p` — per-cell parameters (struct or tuple).
#
# Each cell's ODE is integrated from `t = 0` to `t = dt` independently
# using a singly-diagonally-implicit (SDIRK) backward-Euler step with
# Newton iteration. For very stiff problems users should wrap CVODE via
# Sundials.jl externally; this in-tree integrator is the lightweight
# pure-Julia fallback (no external deps) for problems with O(10) species.
#
# Tier 3 #8 of the AMReX-port roadmap. The analogous AMReX construct is
# PelePhysics' per-cell CVODE reactor.
# ============================================================================

using LinearAlgebra: lu!, ldiv!, I

"""
    ReactionSystem{F, J, P, T}

Description of a per-cell stiff reaction network:
  * `nspecies::Int` — length of the state vector y per cell.
  * `rhs!::F` — `f!(dy::Vector{T}, y::Vector{T}, p, t::T)` writes ẏ.
  * `jac!::J` — `J!(J::Matrix{T}, y::Vector{T}, p, t::T)` or `nothing`
    (Jacobian then evaluated by finite differences).
  * `params_per_cell::P` — `nothing` (no per-cell params) or a
    `Vector{Vector{ParamType}}` indexed `params[ℓ][pi]` of vectors of
    length n_cells (matching the cell storage layout).
"""
struct ReactionSystem{F, J, P, T}
    nspecies::Int
    rhs!::F
    jac!::J
    params_per_cell::P
end

ReactionSystem(nspecies::Int, rhs!::F; jac! = nothing,
                 params_per_cell = nothing, T::Type = Float64) where {F} =
    ReactionSystem{F, typeof(jac!), typeof(params_per_cell), T}(
        nspecies, rhs!, jac!, params_per_cell)

"""
    SpeciesField{D, T}

Per-cell state of `nspecies` reactive species for every (level, patch).
Storage: `y[ℓ][pi][c, s]` where `c` is the cell index and `s ∈ 1:nspecies`.
"""
mutable struct SpeciesField{D, T}
    nspecies::Int
    data::Vector{Vector{Matrix{T}}}
end

"""
    allocate_species(ws::MGWorkspace{D,T}, nspecies; init=zero(T)) -> SpeciesField

Allocate a `nspecies`-component species field over the workspace hierarchy.
"""
function allocate_species(ws::MGWorkspace{D, T}, nspecies::Int;
                            init::T = zero(T)) where {D, T}
    nL = length(ws.patch_N)
    data = Vector{Vector{Matrix{T}}}(undef, nL)
    for ℓ in 1:nL
        nP = length(ws.patch_N[ℓ])
        per_patch = Vector{Matrix{T}}(undef, nP)
        for pi in 1:nP
            n_cells = length(_raw_phi(ws.residual[ℓ][pi]))
            per_patch[pi] = fill(init, n_cells, nspecies)
        end
        data[ℓ] = per_patch
    end
    return SpeciesField{D, T}(nspecies, data)
end

"""
    fill_species!(spec, ws, f)

Populate the species field: `spec[ℓ][pi][c, :] = f(x_c)`, returning a
length-`spec.nspecies` vector.
"""
function fill_species!(spec::SpeciesField{D, T},
                        ws::MGWorkspace{D, T}, f) where {D, T}
    for ℓ in 1:length(spec.data)
        for pi in 1:length(spec.data[ℓ])
            frame = patches_at(ws.ph, ℓ)[pi]
            mat = spec.data[ℓ][pi]
            @inbounds for c in ws.patch_leaves[ℓ][pi]
                lo, hi = cell_physical_box(frame, c)
                x = ntuple(d -> 0.5 * (lo[d] + hi[d]), Val(D))
                yc = f(x)
                for s in 1:spec.nspecies
                    mat[c, s] = T(yc[s])
                end
            end
        end
    end
    return spec
end

"""
    step_reaction!(spec, sys, ws, dt; newton_tol=1e-10, newton_maxiter=30,
                                       fd_eps=1e-8) -> n_failed

Advance `spec` over `[0, dt]` cell-by-cell using a backward-Euler step
solved by damped Newton iteration:
    y^{n+1} = y^n + dt · f(y^{n+1}, p, t^{n+1})

For each leaf cell, computes y^{n+1} via Newton-Raphson on
    G(y) := y - y^n - dt · f(y, p, t^{n+1})
with Jacobian   I - dt · ∂f/∂y. If `sys.jac!` is `nothing`, the Jacobian
is approximated by first-order forward differences.

Returns the number of cells for which Newton failed to converge to
`newton_tol` within `newton_maxiter` steps. Cells that fail retain
their `y^n` value; the caller can re-attempt with a smaller `dt`.
"""
function step_reaction!(spec::SpeciesField{D, T},
                          sys::ReactionSystem,
                          ws::MGWorkspace{D, T},
                          dt::T;
                          newton_tol::T = T(1e-10),
                          newton_maxiter::Int = 30,
                          fd_eps::T = T(1e-8)) where {D, T}
    n = sys.nspecies
    n_failed = Threads.Atomic{Int}(0)
    # Per-thread scratch (avoid contention).
    nthr = max(1, Threads.nthreads())
    scratch = [
        (y_old = Vector{T}(undef, n), y_new = Vector{T}(undef, n),
         y_pert = Vector{T}(undef, n),
         dy = Vector{T}(undef, n), G = Vector{T}(undef, n),
         G_pert = Vector{T}(undef, n),
         J = Matrix{T}(undef, n, n))
        for _ in 1:nthr
    ]

    for ℓ in 1:length(spec.data)
        for pi in 1:length(spec.data[ℓ])
            mat = spec.data[ℓ][pi]
            params = sys.params_per_cell === nothing ? nothing :
                     sys.params_per_cell[ℓ][pi]
            leaves = ws.patch_leaves[ℓ][pi]
            Threads.@threads :static for c in leaves
                tid = Threads.threadid()
                sc = scratch[tid]
                @inbounds for s in 1:n
                    sc.y_old[s] = mat[c, s]
                    sc.y_new[s] = mat[c, s]   # initial guess = y^n
                end
                p = params === nothing ? nothing : params[c]
                converged = _newton_be!(sc, sys, dt, p, newton_tol,
                                          newton_maxiter, fd_eps)
                if converged
                    @inbounds for s in 1:n
                        mat[c, s] = sc.y_new[s]
                    end
                else
                    Threads.atomic_add!(n_failed, 1)
                end
            end
        end
    end
    return n_failed[]
end

# Damped Newton iteration for the backward-Euler step
#     G(y) = y - y_old - dt · f(y, p, t) = 0
@inline function _newton_be!(sc, sys, dt::T, p, tol::T,
                               maxiter::Int, fd_eps::T) where {T}
    n = sys.nspecies
    @inbounds for iter in 1:maxiter
        # G(y) = y - y_old - dt · f(y)
        sys.rhs!(sc.G, sc.y_new, p, dt)
        for s in 1:n
            sc.G[s] = sc.y_new[s] - sc.y_old[s] - dt * sc.G[s]
        end
        normG = sqrt(sum(g -> g * g, sc.G))
        normG < tol && return true

        # Jacobian of G = I - dt · ∂f/∂y.
        if sys.jac! === nothing
            _fd_jacobian!(sc.J, sys, sc.y_new, sc.y_old, p, dt, sc.G,
                            sc.y_pert, sc.G_pert, fd_eps)
        else
            sys.jac!(sc.J, sc.y_new, p, dt)
            for j in 1:n, i in 1:n
                sc.J[i, j] = -dt * sc.J[i, j]
            end
            for s in 1:n
                sc.J[s, s] += one(T)
            end
        end

        # Solve J · δy = -G.
        for s in 1:n
            sc.dy[s] = -sc.G[s]
        end
        F = lu!(sc.J; check = false)
        if !LinearAlgebra.issuccess(F)
            return false
        end
        ldiv!(F, sc.dy)

        # Damped update: try full step; if RHS norm grows, halve until
        # decrease or step too small.
        α = one(T)
        for _ in 1:5
            for s in 1:n
                sc.y_pert[s] = sc.y_new[s] + α * sc.dy[s]
            end
            sys.rhs!(sc.G_pert, sc.y_pert, p, dt)
            ng = zero(T)
            for s in 1:n
                g = sc.y_pert[s] - sc.y_old[s] - dt * sc.G_pert[s]
                ng += g * g
            end
            ng = sqrt(ng)
            if ng < normG
                for s in 1:n
                    sc.y_new[s] = sc.y_pert[s]
                end
                break
            end
            α *= T(0.5)
        end
    end
    return false
end

# Finite-difference Jacobian of  G(y) = y - y_old - dt · f(y)  w.r.t. y.
# Diagonal column j gets contribution +1 from y, plus -dt · ∂f_i/∂y_j.
@inline function _fd_jacobian!(J::Matrix{T}, sys, y::Vector{T},
                                  y_old::Vector{T}, p,
                                  dt::T, G::Vector{T}, y_pert::Vector{T},
                                  G_pert::Vector{T}, eps::T) where {T}
    n = sys.nspecies
    for s in 1:n
        y_pert[s] = y[s]
    end
    for j in 1:n
        h = max(eps, eps * abs(y[j]))
        y_pert[j] = y[j] + h
        # G_pert_i = y_pert_i - y_old_i - dt · f(y_pert)_i.
        sys.rhs!(G_pert, y_pert, p, dt)
        for i in 1:n
            G_pert[i] = y_pert[i] - y_old[i] - dt * G_pert[i]
        end
        @inbounds for i in 1:n
            J[i, j] = (G_pert[i] - G[i]) / h
        end
        y_pert[j] = y[j]   # restore
    end
    return J
end
