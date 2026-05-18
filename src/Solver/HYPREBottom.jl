# ============================================================================
# HYPREBottom.jl — HYPRE BoomerAMG / PCG / GMRES / BiCGSTAB bottom solvers
# via HYPRE.jl.
#
# This is the MPI-capable counterpart of `AMGBottom.jl` — the AMReX
# BoomerAMG bottom solver in pure-Julia wrapper form. HYPRE.jl ships
# HYPRE_jll, so no system install is required. Single-rank
# (no-MPI-init) usage works out of the box; multi-rank MPI usage
# requires the caller to set up MPI before calling these routines.
#
# Tier 2 #6+ of the AMReX-port roadmap.
# ============================================================================

using HYPRE: HYPRE, HYPREMatrix, HYPREVector,
              BoomerAMG, PCG, GMRES, BiCGSTAB, FlexGMRES,
              ParCSRPCG, ParCSRBiCGSTAB

# Track whether HYPRE.Init() has been called so we don't re-init.
const _HYPRE_INITED = Ref{Bool}(false)

"""
    init_hypre()

Lazy `HYPRE.Init()` — called automatically by the bottom solvers. Safe to
call multiple times. For MPI usage, the caller should `using MPI;
MPI.Init()` *before* `init_hypre()`; for single-rank use the default
behaviour is correct.
"""
function init_hypre()
    if !_HYPRE_INITED[]
        HYPRE.Init()
        _HYPRE_INITED[] = true
    end
    return nothing
end

"""
    HYPREPreconditioner{D, T}

HYPRE BoomerAMG (or other HYPRE solver-as-preconditioner) wrapping the
single-level ABec operator on the flat-vector layout. The assembled
HYPRE matrix is built once at construction; `amg_precond_callback` (or
the new `hypre_precond_callback`) returns a callable plug for use as
`precond` in `solve_with_krylov!`.
"""
mutable struct HYPREPreconditioner{D, T}
    layout::FlatLayout{D, T}
    A::SparseMatrixCSC{T, Int}
    solver::Any         # HYPRE.BoomerAMG / HYPRE.PCG / etc.
end

"""
    hypre_preconditioner(ws, coefs; level=1, patch=1,
                          solver = :boomeramg,
                          boomeramg_kwargs = NamedTuple()) -> HYPREPreconditioner

Build an HYPRE-based preconditioner for the single-patch ABec operator.

  * `solver ∈ (:boomeramg, :pcg, :gmres, :bicgstab, :flexgmres)` — picks
    the HYPRE solver to use as the M⁻¹ application.
  * `boomeramg_kwargs` is splatted into `HYPRE.BoomerAMG`; see HYPRE.jl
    docs for tuning (StrongThreshold, RelaxType, CoarsenType, …).
"""
function hypre_preconditioner(ws::MGWorkspace{D, T},
                                coefs::ABecCoefs{D, T};
                                level::Int = 1,
                                patch::Int = 1,
                                solver::Symbol = :boomeramg,
                                boomeramg_kwargs = NamedTuple(),
                                tol::Float64 = 1e-6,
                                maxiter::Int = 50) where {D, T}
    @assert level == 1 "hypre_preconditioner: only single-level supported"
    @assert patch == 1 "hypre_preconditioner: only single-patch supported"
    init_hypre()

    A_sp = assemble_abec_matrix(ws, coefs; level = level, patch = patch)

    s = solver === :boomeramg ? BoomerAMG(; boomeramg_kwargs...) :
        solver === :pcg       ? PCG(;     Tol = tol, MaxIter = maxiter) :
        solver === :gmres     ? GMRES(;   Tol = tol, MaxIter = maxiter) :
        solver === :bicgstab  ? BiCGSTAB(; Tol = tol, MaxIter = maxiter) :
        solver === :flexgmres ? FlexGMRES(; Tol = tol, MaxIter = maxiter) :
        throw(ArgumentError("unknown HYPRE solver :$solver"))

    layout = flat_layout(ws; level_range = level:level)
    return HYPREPreconditioner{D, T}(layout, A_sp, s)
end

"""
    hypre_precond_callback(hp::HYPREPreconditioner)

Return a `(z_fields, r_fields) -> z_fields` callback that applies the
HYPRE BoomerAMG / Krylov solver to `r_fields` and writes the result
into `z_fields`. Pass as `precond` to `solve_with_krylov!`.
"""
function hypre_precond_callback(hp::HYPREPreconditioner{D, T}) where {D, T}
    return (z_fields, r_fields) -> begin
        r_flat = Vector{T}(undef, hp.layout.n)
        pack!(r_flat, r_fields, hp.layout; field = :phi)
        # `HYPRE.solve(::HYPRESolver, ::SparseMatrixCSC, ::Vector)` returns
        # a Julia Vector directly. The internal HYPREMatrix is rebuilt
        # each call — fine when matvec dominates anyway, and avoids the
        # HYPREVector → Vector extraction problem.
        z_flat = HYPRE.solve(hp.solver, hp.A, r_flat)
        unpack!(z_fields, z_flat, hp.layout; field = :phi)
        return z_fields
    end
end

"""
    solve_abec_hypre!(phi, rho, coefs, ws; solver=:boomeramg,
                       boomeramg_kwargs=NamedTuple(),
                       tol=1e-9, maxiter=200) -> MGResult

End-to-end ABec solve using HYPRE directly (no outer Krylov wrap).
Useful as a one-shot solver when BoomerAMG alone gives sufficient
accuracy; for tighter tolerances combine with `solve_with_krylov!`
using `hypre_precond_callback` as the preconditioner.
"""
function solve_abec_hypre!(phi::Vector{Vector{NamedTuple}},
                            rho::Vector{Vector{NamedTuple}},
                            coefs::ABecCoefs{D, T},
                            ws::MGWorkspace{D, T};
                            solver::Symbol = :boomeramg,
                            boomeramg_kwargs = NamedTuple(),
                            tol::Float64 = 1e-9,
                            maxiter::Int = 200,
                            level::Int = 1,
                            patch::Int = 1) where {D, T}
    @assert level == 1 "solve_abec_hypre!: only single-level supported"
    @assert patch == 1 "solve_abec_hypre!: only single-patch supported"
    init_hypre()

    A_sp = assemble_abec_matrix(ws, coefs; level = level, patch = patch)

    layout = flat_layout(ws; level_range = level:level)
    b_flat = Vector{T}(undef, layout.n)
    x_flat = Vector{T}(undef, layout.n)
    pack!(b_flat, rho, layout; field = :rho)
    pack!(x_flat, phi, layout; field = :phi)

    # Initial residual.
    r0 = sqrt(sum(abs2, b_flat .- A_sp * x_flat))

    s = solver === :boomeramg ? BoomerAMG(; Tol = tol, MaxIter = maxiter,
                                             boomeramg_kwargs...) :
        solver === :pcg       ? PCG(;       Tol = tol, MaxIter = maxiter) :
        solver === :gmres     ? GMRES(;     Tol = tol, MaxIter = maxiter) :
        solver === :bicgstab  ? BiCGSTAB(;  Tol = tol, MaxIter = maxiter) :
        solver === :flexgmres ? FlexGMRES(; Tol = tol, MaxIter = maxiter) :
        throw(ArgumentError("unknown HYPRE solver :$solver"))

    x_out = HYPRE.solve(s, A_sp, b_flat)
    unpack!(phi, x_out, layout; field = :phi)

    rf = sqrt(sum(abs2, b_flat .- A_sp * x_out))
    return MGResult(maxiter, r0, rf, rf < tol * r0 || rf < tol, [r0, rf])
end
