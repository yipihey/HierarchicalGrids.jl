# ============================================================================
# VectorABec.jl — multi-component decoupled ABec solver.
#
# Solves K independent scalar ABec systems on the same hierarchy, e.g.
# the per-component velocity diffusion in incompressible Navier–Stokes
# with isotropic viscosity:
#
#       (I - Δt·ν·∇²) u_k = u_k^*       for k = 1, …, D
#
# Each component is independent — no cross-component coupling in the
# operator. The true tensor viscosity (μ(∇u + ∇uᵀ) + λ ∇·u·I) couples
# components and is a separate Tier-2 #5 follow-up.
#
# This is just orchestration over the existing `solve_abec!` to give a
# clean entry point for vector physics, and to centralise per-component
# workspaces if/when AMG hierarchies are reused.
# ============================================================================

"""
    VectorABecProblem{D, T, K}

A K-component decoupled ABec problem on the patch hierarchy. Each
component has its own `ABecCoefs` (so α, β, A, B can vary per component)
and its own field container.

  * `coefs::NTuple{K, ABecCoefs{D,T}}`
  * `fields::NTuple{K, Vector{Vector{NamedTuple}}}` — each is an
    `allocate_phi_rho(ph)` container.
"""
struct VectorABecProblem{D, T, K}
    coefs::NTuple{K, ABecCoefs{D, T}}
    fields::NTuple{K, Vector{Vector{NamedTuple}}}
end

"""
    allocate_vector_abec(ws::MGWorkspace{D,T}, K::Int; A=0, B=1) -> VectorABecProblem

Allocate `K` independent ABec coefficient sets and field containers on the
hierarchy. Use the standard `fill_abec_alpha!` / `fill_abec_beta!` on each
`coefs[k]` to populate physical coefficients; fill `fields[k].rho` for
RHS, `fields[k].phi` for initial guess.
"""
function allocate_vector_abec(ws::MGWorkspace{D, T}, K::Int;
                                A::T = zero(T), B::T = one(T)) where {D, T}
    coefs  = ntuple(_ -> allocate_abec_coefs(ws; A = A, B = B), K)
    fields = ntuple(_ -> allocate_phi_rho(ws.ph), K)
    return VectorABecProblem{D, T, K}(coefs, fields)
end

"""
    solve_vector_abec!(prob::VectorABecProblem, ws; tol, maxiter, verbose) ->
        NTuple{K, MGResult}

Solve all K systems independently. Returns a tuple of per-component
`MGResult`. Suitable for, e.g., per-component velocity diffusion in
incompressible flow when cross-coupling (μ ∇uᵀ + λ ∇·u·I) is treated
explicitly.
"""
function solve_vector_abec!(prob::VectorABecProblem{D, T, K},
                              ws::MGWorkspace{D, T};
                              tol::Float64 = ws.opts.tol,
                              maxiter::Int = ws.opts.maxiter,
                              verbose::Bool = ws.opts.verbose) where {D, T, K}
    return ntuple(K) do k
        solve_abec!(ws, prob.fields[k], prob.coefs[k];
                    tol = tol, maxiter = maxiter, verbose = verbose)
    end
end
