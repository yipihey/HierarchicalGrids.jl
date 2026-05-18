# ============================================================================
# RadiationDiffusion.jl — flux-limited / classical radiation diffusion
# wrappers built on the ABec operator.
#
# Single-group (gray) radiation diffusion in the linear regime (fixed κ, T):
#     ∂_t E_r = ∇·(D ∇E_r) - c·κ·(E_r - a·T⁴)
# Backward Euler:
#     (I - Δt ∇·(D∇)) E_r^{n+1} + Δt·c·κ E_r^{n+1} = E_r^n + Δt·c·κ·a·T⁴
# In AMReX MLABecLaplacian form  A·α·φ - B·∇·(β ∇φ) = f:
#     A = 1 + Δt·c·κ_cell
#     α = 1
#     B = Δt
#     β = D_face = c / (3·κ_face)
#     f = E_r^n + Δt·c·κ·a·T⁴ + sources
#
# Multigroup (`G` groups): one ABec solve per group, optionally coupled via
# in-scattering / cross-group emission. The linear-decoupled case is just G
# independent ABec solves; the user supplies the coupling term as part of `f`.
#
# The fully nonlinear case (κ(T), B(T), gas-radiation energy exchange) is a
# Newton-Krylov outer iteration; left as a separate driver (Tier 3 #7 v2)
# that consumes the linear solvers exposed here.
#
# References:
#   * Castro MGFLD: arXiv:1207.3845
#   * Quokka two-moment VET: arXiv:2110.01792
# ============================================================================

"""
    setup_gray_radiation!(coefs, ws, kappa, dt, c_light=1.0)

Configure an `ABecCoefs` for a backward-Euler gray-radiation diffusion step.

Parameters:
  * `kappa(x)` — opacity function evaluated at cell centers. Same function
    is averaged at face centers for the diffusion coefficient D = c/(3κ).
  * `dt::T` — time step.
  * `c_light::T` — speed of light (= 1 in natural units).

Sets:
  * `coefs.A = 1`
  * `coefs.B = dt` (so that B·∇·(β ∇) accounts for the diffusion term).
  * `coefs.alpha[ℓ][p][c] = 1 + dt · c_light · kappa(x_c)` per cell
  * `coefs.beta[ℓ][p][d][face] = c_light / (3 · kappa(x_face))` per face

After calling this, build the RHS f = E_r^n + dt·c·κ·a·T⁴ + sources in
`fields[:].rho` and call `solve_abec!(ws, fields, coefs)`.
"""
function setup_gray_radiation!(coefs::ABecCoefs{D, T},
                                ws::MGWorkspace{D, T},
                                kappa_fun,
                                dt::T;
                                c_light::T = one(T)) where {D, T}
    coefs_new_A = one(T)
    coefs_new_B = dt
    # The struct is immutable but its arrays are mutable; we update the
    # arrays in place and return the user a fresh `ABecCoefs` with the
    # new scalar A, B.
    cdt = c_light * dt

    for ℓ in 1:length(coefs.alpha)
        for pi in 1:length(coefs.alpha[ℓ])
            frame = patches_at(ws.ph, ℓ)[pi]
            α = coefs.alpha[ℓ][pi]
            @inbounds for c in ws.patch_leaves[ℓ][pi]
                lo, hi = cell_physical_box(frame, c)
                x = ntuple(d -> 0.5 * (lo[d] + hi[d]), Val(D))
                α[c] = one(T) + cdt * T(kappa_fun(x))
            end
        end
    end

    for ℓ in 1:length(coefs.beta)
        for pi in 1:length(coefs.beta[ℓ])
            frame = patches_at(ws.ph, ℓ)[pi]
            N = ws.patch_N[ℓ][pi]
            dx = ws.patch_dx[ℓ][pi]
            for d in 1:D
                βd = coefs.beta[ℓ][pi][d]
                @inbounds for I in CartesianIndices(size(βd))
                    x = ntuple(Val(D)) do j
                        if j == d
                            frame.lo[d] + (I[d] - 1) * dx[d]
                        else
                            frame.lo[j] + (I[j] - 0.5) * dx[j]
                        end
                    end
                    κ_face = T(kappa_fun(x))
                    # D_face = c / (3 κ).  Cap κ from below to avoid 1/0.
                    κ_safe = max(κ_face, T(1e-30))
                    βd[I] = c_light / (3 * κ_safe)
                end
            end
        end
    end

    return ABecCoefs{D, T}(coefs_new_A, coefs_new_B,
                             coefs.alpha, coefs.beta)
end

"""
    solve_gray_radiation_step!(E_r_field, E_r_prev_field, coefs, ws;
                                 c_light=1.0, source_field=nothing,
                                 T_field=nothing, a_rad=1.0, tol=1e-9,
                                 maxiter=200)

One backward-Euler step of linear gray radiation diffusion. Builds the
RHS `f = E_r^n + dt·c·κ·a·T⁴ + source` into `E_r_field[:].rho` and solves
`(A α - B ∇·(β ∇)) E_r = f` via `solve_abec!`.

Inputs:
  * `coefs` — already configured by `setup_gray_radiation!`.
  * `E_r_field` — the field container; on exit `.phi` holds E_r^{n+1}.
  * `E_r_prev_field` — `.phi` holds E_r^n (typically same container).
  * `T_field` — if provided, `.phi` holds gas temperature for the κ·a·T⁴
    emission term.
  * `source_field` — optional extra source written into RHS.
  * `a_rad` — radiation constant; default 1 in natural units.

Returns the `MGResult` from the inner ABec solve.
"""
function solve_gray_radiation_step!(E_r_field::Vector{Vector{NamedTuple}},
                                      E_r_prev_field::Vector{Vector{NamedTuple}},
                                      coefs::ABecCoefs{D, T},
                                      ws::MGWorkspace{D, T};
                                      c_light::T = one(T),
                                      source_field::Union{Nothing,
                                          Vector{Vector{NamedTuple}}} = nothing,
                                      T_field::Union{Nothing,
                                          Vector{Vector{NamedTuple}}} = nothing,
                                      a_rad::T = one(T),
                                      tol::Float64 = 1e-9,
                                      maxiter::Int = 200) where {D, T}
    dt = coefs.B
    cdt = c_light * dt
    for ℓ in ws.level_range
        for pi in 1:length(E_r_field[ℓ])
            E_prev_phi = _raw_phi(E_r_prev_field[ℓ][pi])::Vector{T}
            E_rho = _raw_rho(E_r_field[ℓ][pi])::Vector{T}
            α = coefs.alpha[ℓ][pi]
            @inbounds for c in ws.patch_leaves[ℓ][pi]
                rhs = E_prev_phi[c]                            # E_r^n
                # κ·c·dt term came from α = 1 + κ·c·dt; on the RHS we add
                # the matching emission: c·dt·κ·a·T⁴ = (α - 1) · a · T⁴.
                if T_field !== nothing
                    Traw = _raw_phi(T_field[ℓ][pi])::Vector{T}
                    rhs += (α[c] - one(T)) * a_rad * Traw[c]^4
                end
                if source_field !== nothing
                    Srho = _raw_rho(source_field[ℓ][pi])::Vector{T}
                    rhs += dt * Srho[c]
                end
                E_rho[c] = rhs
            end
        end
    end
    return solve_abec!(ws, E_r_field, coefs; tol = tol, maxiter = maxiter)
end

"""
    MultigroupRadiation{D, T, G}

Container for `G` radiation groups. Each group `g` has its own ABec
coefficients (allocated to the same hierarchy) and field. The user is
responsible for filling κ_g and the per-group source.
"""
struct MultigroupRadiation{D, T, G}
    coefs::NTuple{G, ABecCoefs{D, T}}
    fields::NTuple{G, Vector{Vector{NamedTuple}}}
end

"""
    allocate_multigroup(ws, ngroups; A=1.0, B=1.0) -> MultigroupRadiation

Allocate a `ngroups`-group multigroup container: one `ABecCoefs` and one
field container per group, all on `ws.ph`. Initial α=0, β=1 (the caller
fills group-specific coefficients via `setup_gray_radiation!` per group,
or by directly setting `.alpha` / `.beta`).
"""
function allocate_multigroup(ws::MGWorkspace{D, T}, ngroups::Int;
                              A::T = one(T), B::T = one(T)) where {D, T}
    coefs = ntuple(ngroups) do g
        allocate_abec_coefs(ws; A = A, B = B)
    end
    fields = ntuple(ngroups) do g
        allocate_phi_rho(ws.ph)
    end
    return MultigroupRadiation{D, T, ngroups}(coefs, fields)
end

"""
    solve_multigroup_step!(mg, ws; tol=1e-9, maxiter=200, verbose=false) ->
        NTuple{G, MGResult}

Solve `G` independent ABec systems, one per group. For coupled multigroup
(e.g. with cross-group emissivity in the RHS), the caller is responsible
for assembling `mg.fields[g].rho` to include all coupling terms before
calling this. Returns a tuple of per-group `MGResult`.

This is the same orchestration AMReX's `MGFLDSolver` uses in the
"groups-decoupled" mode (each group's ABec solved sequentially); the
fully coupled case requires an outer iteration the caller drives.
"""
function solve_multigroup_step!(mg::MultigroupRadiation{D, T, G},
                                  ws::MGWorkspace{D, T};
                                  tol::Float64 = 1e-9,
                                  maxiter::Int = 200,
                                  verbose::Bool = false) where {D, T, G}
    results = ntuple(G) do g
        solve_abec!(ws, mg.fields[g], mg.coefs[g];
                    tol = tol, maxiter = maxiter, verbose = verbose)
    end
    return results
end
