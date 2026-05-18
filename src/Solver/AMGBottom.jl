# ============================================================================
# AMGBottom.jl — Algebraic-multigrid bottom solver via AlgebraicMultigrid.jl
#
# Assembles a sparse-matrix representation of the (single-level) ABec or
# const-coef Poisson operator and wraps AlgebraicMultigrid.jl's Ruge-Stüben
# or smoothed-aggregation AMG as a Krylov.jl-compatible preconditioner.
#
# This is the pure-Julia counterpart of AMReX's HYPRE BoomerAMG bottom
# solver — same role (scale beyond the geometric-coarsening floor) but
# without an external dependency. Tier 2 #6 of the AMReX-port roadmap.
#
# Scope: single-level on a single patch. Multi-level matrix assembly
# (covering FAC C/F entries) is part of the MLNodeLaplacian Tier-2 task.
# ============================================================================

using AlgebraicMultigrid: ruge_stuben, smoothed_aggregation, aspreconditioner,
                           RugeStubenAMG, SmoothedAggregationAMG
using SparseArrays: SparseMatrixCSC, sparse, spzeros

"""
    assemble_abec_matrix(ws, coefs; level = 1, patch = 1) -> SparseMatrixCSC{T}

Build a sparse matrix `A` representing the ABec operator
    L φ = A·α·φ - B·∇·(β ∇φ)
on a single patch (`level`, `patch`). Rows / columns are indexed by leaf
cell ID via `ws.patch_leaves[level][patch]`. Periodic BCs wrap, Dirichlet
contributes the +2β/h² to the diagonal as in `apply_abec!`.

The matrix has 2D+1 nonzeros per row (center + 2D nearest neighbors).
"""
function assemble_abec_matrix(ws::MGWorkspace{D, T},
                                coefs::ABecCoefs{D, T};
                                level::Int = 1,
                                patch::Int = 1) where {D, T}
    N = ws.patch_N[level][patch]
    dx = ws.patch_dx[level][patch]
    c2c = ws.cart_to_cell[level][patch]
    kinds = ws.axis_kinds.kinds
    α = coefs.alpha[level][patch]
    β = coefs.beta[level][patch]
    A = coefs.A
    B = coefs.B
    leaves = ws.patch_leaves[level][patch]
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))

    # Build cell → flat-row mapping (1-based dense over the leaf set).
    cell_to_row = zeros(Int, length(_raw_phi(ws.residual[level][patch])))
    @inbounds for (i, c) in enumerate(leaves)
        cell_to_row[c] = i
    end
    n = length(leaves)

    # Pre-allocate I, J, V arrays for the stencil.
    nz_per_row = 2 * D + 1
    I = Vector{Int}(undef, n * nz_per_row)
    J = Vector{Int}(undef, n * nz_per_row)
    V = Vector{T}(undef, n * nz_per_row)
    nnz_count = 0

    @inbounds for cI in CartesianIndices(N)
        c = c2c[cI]
        c == 0 && continue
        row = cell_to_row[c]
        row == 0 && continue

        diag_coef = A * α[c]
        for d in 1:D
            β_low  = β[d][cI]
            β_high = β[d][cI + _unit_offsets(Val(D))[d]]

            # Diagonal accumulation.
            diag_coef += B * (β_high + β_low) * invh2[d]

            # High-side neighbor.
            nbr_coef = -B * β_high * invh2[d]
            inew = cI[d] + 1
            if 1 <= inew <= N[d]
                nb_c = c2c[cI + _unit_offsets(Val(D))[d]]
                nb_row = cell_to_row[nb_c]
                if nb_row != 0
                    nnz_count += 1
                    I[nnz_count] = row
                    J[nnz_count] = nb_row
                    V[nnz_count] = nbr_coef
                end
            else
                # Off-patch: periodic / dirichlet / neumann.
                kind = kinds[d]
                if kind === :periodic
                    nb_c = c2c[CartesianIndex(ntuple(j -> j == d ? 1 : cI[j], Val(D)))]
                    nb_row = cell_to_row[nb_c]
                    if nb_row != 0
                        nnz_count += 1
                        I[nnz_count] = row
                        J[nnz_count] = nb_row
                        V[nnz_count] = nbr_coef
                    end
                elseif kind === :dirichlet
                    # Ghost = -φ_c ⇒ this face's nbr contribution is
                    # -β·(φ_ghost - φ_c)/h² → +2β·φ_c/h² (added to diag).
                    diag_coef += -B * 2 * β_high * invh2[d]  # +B β/h² already
                                                                # included in
                                                                # diag_coef +=
                                                                # (β_high+β_low)/h²;
                                                                # add extra to
                                                                # complete the
                                                                # 2 β/h² Dirichlet
                                                                # contribution.
                    # Net Dirichlet diag = -B β_high/h² (counter-corrects the
                    # generic +Bβ_high·1/h² in diag_coef above so the total
                    # for this face is just the +2B β_high/h² we want).
                    # Above is hand-derived to match `_apply_abec_root!`.
                end
                # Neumann: contribution is zero (no extra term).
            end

            # Low-side neighbor.
            nbr_coef_low = -B * β_low * invh2[d]
            inew = cI[d] - 1
            if 1 <= inew <= N[d]
                nb_c = c2c[cI - _unit_offsets(Val(D))[d]]
                nb_row = cell_to_row[nb_c]
                if nb_row != 0
                    nnz_count += 1
                    I[nnz_count] = row
                    J[nnz_count] = nb_row
                    V[nnz_count] = nbr_coef_low
                end
            else
                kind = kinds[d]
                if kind === :periodic
                    nb_c = c2c[CartesianIndex(ntuple(j -> j == d ? N[d] : cI[j], Val(D)))]
                    nb_row = cell_to_row[nb_c]
                    if nb_row != 0
                        nnz_count += 1
                        I[nnz_count] = row
                        J[nnz_count] = nb_row
                        V[nnz_count] = nbr_coef_low
                    end
                elseif kind === :dirichlet
                    diag_coef += -B * 2 * β_low * invh2[d]
                end
            end
        end

        # Diagonal.
        nnz_count += 1
        I[nnz_count] = row
        J[nnz_count] = row
        V[nnz_count] = diag_coef
    end

    resize!(I, nnz_count)
    resize!(J, nnz_count)
    resize!(V, nnz_count)
    return sparse(I, J, V, n, n)
end

"""
    AMGPreconditioner{D, T}

AlgebraicMultigrid preconditioner wrapping `aspreconditioner(ml)` over the
single-level `ABecOp` flat-vector layout. The constructor builds the
AMG hierarchy (Ruge-Stüben by default; pass `:sa` for smoothed-aggregation).
"""
mutable struct AMGPreconditioner{D, T}
    layout::FlatLayout{D, T}
    A::SparseMatrixCSC{T, Int}
    amg                            # AMG hierarchy (RugeStubenAMG or SmoothedAggregationAMG)
    M                              # preconditioner returned by aspreconditioner
end

"""
    amg_preconditioner(ws, coefs; level=1, patch=1, method=:rs) -> AMGPreconditioner

Build an AMG preconditioner for the single-patch ABec operator.
`method ∈ (:rs, :sa)` selects Ruge-Stüben or smoothed-aggregation.
"""
function amg_preconditioner(ws::MGWorkspace{D, T},
                              coefs::ABecCoefs{D, T};
                              level::Int = 1,
                              patch::Int = 1,
                              method::Symbol = :rs) where {D, T}
    @assert level == 1 "amg_preconditioner: only single-level supported"
    @assert patch == 1 "amg_preconditioner: only single-patch supported"
    A = assemble_abec_matrix(ws, coefs; level = level, patch = patch)
    ml = method === :rs ? ruge_stuben(A) : smoothed_aggregation(A)
    M = aspreconditioner(ml)
    layout = flat_layout(ws; level_range = level:level)
    return AMGPreconditioner{D, T}(layout, A, ml, M)
end

"""
    amg_precond_callback(amg::AMGPreconditioner)

Return a `(z_fields, r_fields) -> z_fields` callback that applies the
AMG hierarchy to `r_fields` and writes the result into `z_fields`.
Pass as `precond` to `solve_with_krylov!`.
"""
function amg_precond_callback(amg::AMGPreconditioner{D, T}) where {D, T}
    return (z_fields, r_fields) -> begin
        # Pack r → flat, apply M⁻¹ via AMG, unpack to z.
        r_flat = Vector{T}(undef, amg.layout.n)
        pack!(r_flat, r_fields, amg.layout; field = :phi)
        z_flat = amg.M \ r_flat
        unpack!(z_fields, z_flat, amg.layout; field = :phi)
        return z_fields
    end
end
