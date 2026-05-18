# ============================================================================
# NodeLaplacianML.jl — multi-level extension of the node-centered Poisson
# operator. Provides restriction / prolongation operators on `NodeField`
# and a multi-level V-cycle.
#
# Scope: **nested grids only** (single patch per level, each level a
# uniform 2:1 coarsening of its parent). Truly-AMR nodal projection
# (with a fine subpatch covering only part of the coarse domain and
# Martin-Colella-Almgren C/F coupling) is a follow-up.
#
# Tier 2 #4 multi-level of the AMReX-port roadmap.
# ============================================================================

"""
    restrict_node!(coarse::NodeField, fine::NodeField, level_fine; level=1, patch=1)

Full-weighting nodal restriction from level `level_fine` (in `fine.data`)
to level `level_fine - 1` (in `coarse.data`). Assumes 2:1 nested
coarsening: coarse node `(I, J)` coincides with fine node `(2I-1, 2J-1)`.

Restriction stencil (2D interior coarse node):
    C[I, J] = (1/4)  · F[2I-1, 2J-1]
             + (1/8)  · sum(face neighbours)
             + (1/16) · sum(corner neighbours)
"""
function restrict_node!(coarse::NodeField{2, T},
                          fine::NodeField{2, T},
                          level_fine::Int;
                          patch::Int = 1) where {T}
    C = coarse.data[level_fine - 1][patch]
    F = fine.data[level_fine][patch]
    Cd = size(C); Fd = size(F)
    @assert (Fd[1] - 1) == 2 * (Cd[1] - 1) "fine grid must be 2:1 of coarse"
    @assert (Fd[2] - 1) == 2 * (Cd[2] - 1) "fine grid must be 2:1 of coarse"

    @inbounds for J in 1:Cd[2], I in 1:Cd[1]
        fi = 2 * I - 1
        fj = 2 * J - 1
        # Sample fine neighbours with boundary clamping (read F[i, j]
        # safely — out-of-range coordinates wrap or reflect according
        # to the user's BCs; for nested grids without AMR sub-patches
        # the simplest approach is to clamp to the grid).
        face_sum = zero(T)
        corner_sum = zero(T)
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii = clamp(fi + di, 1, Fd[1]); jj = clamp(fj + dj, 1, Fd[2])
            face_sum += F[ii, jj]
        end
        for (di, dj) in ((1, 1), (1, -1), (-1, 1), (-1, -1))
            ii = clamp(fi + di, 1, Fd[1]); jj = clamp(fj + dj, 1, Fd[2])
            corner_sum += F[ii, jj]
        end
        C[I, J] = T(0.25) * F[fi, fj] + T(1 / 8) * face_sum + T(1 / 16) * corner_sum
    end
    return coarse
end

"""
    prolong_node!(fine::NodeField, coarse::NodeField, level_fine; level=1, patch=1)

Bilinear prolongation from level `level_fine - 1` to `level_fine` on a
2:1 nested grid. Fine node `(2I-1, 2J-1)` copies coarse `(I, J)`; nodes
between coarse nodes use 1/2 and 1/4 weights.
"""
function prolong_node!(fine::NodeField{2, T},
                         coarse::NodeField{2, T},
                         level_fine::Int;
                         patch::Int = 1) where {T}
    C = coarse.data[level_fine - 1][patch]
    F = fine.data[level_fine][patch]
    Cd = size(C); Fd = size(F)

    @inbounds for J in 1:Cd[2], I in 1:Cd[1]
        fi = 2 * I - 1; fj = 2 * J - 1
        F[fi, fj] = C[I, J]
        # Right neighbour (i+1/2, j).
        if I < Cd[1]
            F[fi + 1, fj] = T(0.5) * (C[I, J] + C[I + 1, J])
        end
        # Top neighbour (i, j+1/2).
        if J < Cd[2]
            F[fi, fj + 1] = T(0.5) * (C[I, J] + C[I, J + 1])
        end
        # Corner (i+1/2, j+1/2).
        if I < Cd[1] && J < Cd[2]
            F[fi + 1, fj + 1] = T(0.25) * (C[I, J] + C[I + 1, J] +
                                              C[I, J + 1] + C[I + 1, J + 1])
        end
    end
    return fine
end

"""
    vcycle_node!(phi::NodeField, rho::NodeField, coefs, ws;
                   level_range = ws.level_range, n_pre = 2, n_post = 2)

Recursive multi-level V-cycle for the node-Laplacian on nested grids.
Smooths `n_pre` sweeps at the top, restricts residual to the next
coarser level, recurses, prolongs correction, smooths `n_post`. At the
coarsest level, runs `bottom_smooth_iters` GS sweeps (configured on
`ws.opts`).
"""
function vcycle_node!(phi::NodeField{2, T},
                        rho::NodeField{2, T},
                        coefs::NodeCoefs{2, T},
                        ws::MGWorkspace{2, T};
                        level_range::UnitRange{Int} = ws.level_range,
                        n_pre::Int = 2, n_post::Int = 2,
                        patch::Int = 1,
                        scratch_residual::Union{Nothing, NodeField{2, T}} = nothing,
                        scratch_correction::Union{Nothing, NodeField{2, T}} = nothing) where {T}
    ℓ_hi = last(level_range); ℓ_lo = first(level_range)

    if ℓ_hi == ℓ_lo
        # Bottom: many GS sweeps.
        gs_sweep_node!(phi, rho, coefs, ws;
                         n_sweeps = ws.opts.bottom_smooth_iters,
                         level = ℓ_hi, patch = patch)
        return phi
    end

    res = scratch_residual === nothing ? allocate_node_field(ws) : scratch_residual
    cor = scratch_correction === nothing ? allocate_node_field(ws) : scratch_correction

    # Pre-smooth at fine level.
    gs_sweep_node!(phi, rho, coefs, ws;
                     n_sweeps = n_pre, level = ℓ_hi, patch = patch)

    # Residual at fine level: r = ρ - L φ.
    Lphi_temp = scratch_residual === nothing ? allocate_node_field(ws) : res
    apply_node_laplacian!(Lphi_temp, phi, coefs, ws;
                            level = ℓ_hi, patch = patch)
    arr = Lphi_temp.data[ℓ_hi][patch]
    rho_arr = rho.data[ℓ_hi][patch]
    @inbounds for I in eachindex(arr)
        arr[I] = rho_arr[I] - arr[I]
    end

    # Restrict residual to coarser level.
    restrict_node!(Lphi_temp, Lphi_temp, ℓ_hi; patch = patch)
    # Zero correction on coarser levels.
    @inbounds for ℓ in ℓ_lo:(ℓ_hi - 1)
        fill!(cor.data[ℓ][patch], zero(T))
    end

    # Recurse on the coarse correction.
    vcycle_node!(cor, Lphi_temp, coefs, ws;
                   level_range = ℓ_lo:(ℓ_hi - 1),
                   n_pre = n_pre, n_post = n_post,
                   patch = patch,
                   scratch_residual = res,
                   scratch_correction = cor)

    # Prolong correction and add to fine φ.
    prolong_node!(cor, cor, ℓ_hi; patch = patch)
    phi_arr = phi.data[ℓ_hi][patch]
    cor_arr = cor.data[ℓ_hi][patch]
    @inbounds for I in eachindex(phi_arr)
        phi_arr[I] += cor_arr[I]
    end

    # Post-smooth at fine level.
    gs_sweep_node!(phi, rho, coefs, ws;
                     n_sweeps = n_post, level = ℓ_hi, patch = patch)
    return phi
end

"""
    solve_node_laplacian_ml!(phi::NodeField, rho::NodeField, coefs, ws;
                               tol=1e-9, maxiter=50, n_pre=2, n_post=2,
                               level_range = ws.level_range, patch=1)

Multi-level V-cycle iteration for the node Laplacian on nested grids.
Returns an `MGResult`. The iteration uses standalone V-cycles (not a
Krylov accelerator) — for tighter tolerances combine with
`solve_with_krylov!` using a vcycle preconditioner callback.
"""
function solve_node_laplacian_ml!(phi::NodeField{2, T},
                                    rho::NodeField{2, T},
                                    coefs::NodeCoefs{2, T},
                                    ws::MGWorkspace{2, T};
                                    tol::Float64 = 1e-9,
                                    maxiter::Int = 50,
                                    n_pre::Int = 2, n_post::Int = 2,
                                    level_range::UnitRange{Int} = ws.level_range,
                                    patch::Int = 1) where {T}
    # Initial residual.
    ℓ_top = last(level_range)
    Lphi = allocate_node_field(ws)
    apply_node_laplacian!(Lphi, phi, coefs, ws; level = ℓ_top, patch = patch)
    arr = Lphi.data[ℓ_top][patch]
    rho_arr = rho.data[ℓ_top][patch]
    r0 = zero(Float64)
    @inbounds for I in eachindex(arr)
        d = rho_arr[I] - arr[I]
        r0 += Float64(d * d)
    end
    r0 = sqrt(r0)
    history = [r0]

    iter = 0; r = r0; converged = false
    while iter < maxiter
        vcycle_node!(phi, rho, coefs, ws;
                       level_range = level_range,
                       n_pre = n_pre, n_post = n_post,
                       patch = patch)
        apply_node_laplacian!(Lphi, phi, coefs, ws;
                                level = ℓ_top, patch = patch)
        arr = Lphi.data[ℓ_top][patch]
        r_new = zero(Float64)
        @inbounds for I in eachindex(arr)
            d = rho_arr[I] - arr[I]
            r_new += Float64(d * d)
        end
        r_new = sqrt(r_new)
        push!(history, r_new)
        iter += 1
        if r_new <= tol * r0 || r_new <= tol
            converged = true; r = r_new; break
        end
        r = r_new
    end
    return MGResult(iter, r0, r, converged, history)
end
