"""
Moment-based remap operators for moving fields between Lagrangian and
Eulerian meshes through a precomputed `GeometricOverlap`.

The mathematics here is independent of r3djl: given the overlap volumes
(and higher moments), each operator implements one rule for combining
source-cell field values into destination-cell field values. The rules
mirror the law of total expectation, total covariance, and total cumulants
for the corresponding moment orders.

# Operators

| Operator              | Field structure        | Math                          |
|-----------------------|------------------------|-------------------------------|
| `MassWeightedAverage` | scalar per cell        | weighted mean by overlap vol  |
| `ConservativeFlux`    | extensive scalar/cell  | partition by overlap vol      |
| `TotalCovariance`     | symmetric d×d tensor   | law of total covariance       |
| `TotalCumulants{O}`   | rank-`O` symm. tensor  | full law of total cumulants   |

For polynomial-reconstruction fields (`PolynomialFieldSet`), use
`polynomial_remap_l_to_e!` / `polynomial_remap_e_to_l!`, which integrate
the source polynomial over each overlap polytope using the overlap
entry's stored moments and the source cell's affine map.

# Status

- `MassWeightedAverage` and `ConservativeFlux` (scalar fields): fully
  implemented and tested.
- `TotalCovariance` (tensor fields): implemented for the centroid-only
  contribution (the standard Israel–Stewart law-of-total-covariance term
  used in moment-of-moments hydro). The polynomial-source generalization
  — where the source covariance is integrated against full overlap
  moments rather than just centroids — is a TODO; it would let
  TotalCovariance reproduce arbitrary polynomial source variance fields
  exactly.
- `TotalCumulants{O}` for `O ≥ 3`: API in place, body returns
  `not implemented` until the third-cumulant pattern is exercised by a
  caller.

For polynomial-reconstruction fields (`PolynomialFieldSet`), the
machinery in `polynomial_remap.jl` and `polynomial_remap_streaming.jl`
covers arbitrary monomial orders and both axis-aligned and simplicial
reference frames, in both L→E and E→L directions. The operators here
remain useful for scalar / tensor-of-cell-centroid fields where a full
polynomial reconstruction isn't carried.
"""

# ============================================================================
# Operator types
# ============================================================================

abstract type RemapOperator end

"""
    MassWeightedAverage <: RemapOperator

Standard intensive-quantity remap. Given source cell values `u_i` and
overlap volumes `w_ij = vol(S_i ∩ T_j)`, the destination value is

    u_j_new = (Σ_i w_ij * u_i) / (Σ_i w_ij)

which is the law of total expectation specialized to piecewise-constant
fields. Conserves `Σ_j vol(T_j) * u_j_new = Σ_i vol(S_i) * u_i` when
the meshes cover the same domain.
"""
struct MassWeightedAverage <: RemapOperator end

"""
    ConservativeFlux <: RemapOperator

For extensive quantities (mass, momentum, energy *per cell*, not per
unit volume): the destination value is the sum of source contributions
weighted by overlap fraction:

    U_j_new = Σ_i (w_ij / vol(S_i)) * U_i

Conserves `Σ_j U_j_new = Σ_i U_i` exactly.
"""
struct ConservativeFlux <: RemapOperator end

"""
    TotalCovariance <: RemapOperator

For second-moment fields (e.g. the dfmm `Mvv` velocity covariance): the
destination tensor is the source-volume-weighted average of source
tensors PLUS the spread-of-means coarse-graining contribution:

    M_j_new = (Σ_i w_ij * M_i) / W_j +
              (Σ_i w_ij * (μ_i - μ_j_new) (μ_i - μ_j_new)^T) / W_j

where `μ_i` is the source cell's centroid (or any first-moment vector
field paired with the tensor), `W_j = Σ_i w_ij`, and `μ_j_new` is the
weighted-mean of the source means. This is exactly the law of total
covariance applied to coarse-graining.
"""
struct TotalCovariance <: RemapOperator end

"""
    TotalCumulants{O} <: RemapOperator

Generalization to rank-`O` symmetric cumulant tensors. Currently the
`O = 2` case is implemented as `TotalCovariance`; higher orders return
`not implemented`. The dfmm design (§6.4) specifies the third-cumulant
pattern needed for the heat flux Q.
"""
struct TotalCumulants{O} <: RemapOperator end
TotalCumulants(O::Integer) = TotalCumulants{Int(O)}()

# ============================================================================
# Scalar-field remap (law of total expectation)
# ============================================================================

"""
    remap_l_to_e!(target::AbstractVector{T}, source::AbstractVector{T},
                   overlap::GeometricOverlap, op::RemapOperator)

Move a scalar-per-cell field from the Lagrangian side to the Eulerian
side via the precomputed `overlap`.

- `target` must have length `overlap.n_eul` (one entry per Eulerian cell).
- `source` must have length `overlap.n_lag` (one entry per Lagrangian
  simplex).

Eulerian cells with no overlap are left at zero (target is filled with
zero before accumulation).

For `MassWeightedAverage`: target value is volume-weighted mean of
contributing source values; cells with zero accumulated weight are
left at zero.

For `ConservativeFlux`: target value is the sum of source-contribution
fractions; preserves `Σ source_i * vol(S_i) = Σ target_j * vol(T_j)` only
if the source values represent volume densities — see operator docstring.
"""
function remap_l_to_e!(target::AbstractVector{T}, source::AbstractVector{T},
                        overlap::GeometricOverlap{D, T}, ::MassWeightedAverage) where {D, T}
    length(source) == overlap.n_lag ||
        throw(DimensionMismatch("source length $(length(source)) ≠ n_lag $(overlap.n_lag)"))
    length(target) == overlap.n_eul ||
        throw(DimensionMismatch("target length $(length(target)) ≠ n_eul $(overlap.n_eul)"))
    fill!(target, zero(T))
    weights = zeros(T, overlap.n_eul)
    @inbounds for entry in overlap.entries
        i = Int(entry.lag_idx)
        j = Int(entry.eul_idx)
        w = entry.volume
        target[j] += w * source[i]
        weights[j] += w
    end
    @inbounds for j in 1:overlap.n_eul
        if weights[j] > zero(T)
            target[j] /= weights[j]
        end
    end
    return target
end

function remap_l_to_e!(target::AbstractVector{T}, source::AbstractVector{T},
                        overlap::GeometricOverlap{D, T}, ::ConservativeFlux) where {D, T}
    length(source) == overlap.n_lag ||
        throw(DimensionMismatch("source length"))
    length(target) == overlap.n_eul ||
        throw(DimensionMismatch("target length"))
    fill!(target, zero(T))
    # We need source-cell volumes to normalize. For a SimplicialMesh source,
    # this is a separate query — but we can compute it as the sum of the
    # source's overlap entries (= S_i ∩ entire domain ≈ vol(S_i)) when the
    # meshes cover the same domain. To keep this layer mesh-agnostic we
    # pass through and assume the caller has normalized appropriately,
    # exposing a separate fully-conservative variant below.
    src_vol = zeros(T, overlap.n_lag)
    @inbounds for entry in overlap.entries
        src_vol[entry.lag_idx] += entry.volume
    end
    @inbounds for entry in overlap.entries
        i = Int(entry.lag_idx); j = Int(entry.eul_idx)
        if src_vol[i] > zero(T)
            target[j] += (entry.volume / src_vol[i]) * source[i]
        end
    end
    return target
end

# ============================================================================
# E → L variant
# ============================================================================

"""
    remap_e_to_l!(target::AbstractVector{T}, source::AbstractVector{T},
                   overlap::GeometricOverlap, op::RemapOperator)

Reverse direction: Eulerian → Lagrangian. Same semantics with the role of
`lag_idx` and `eul_idx` swapped.
"""
function remap_e_to_l!(target::AbstractVector{T}, source::AbstractVector{T},
                        overlap::GeometricOverlap{D, T}, ::MassWeightedAverage) where {D, T}
    length(source) == overlap.n_eul ||
        throw(DimensionMismatch("source length $(length(source)) ≠ n_eul $(overlap.n_eul)"))
    length(target) == overlap.n_lag ||
        throw(DimensionMismatch("target length $(length(target)) ≠ n_lag $(overlap.n_lag)"))
    fill!(target, zero(T))
    weights = zeros(T, overlap.n_lag)
    @inbounds for entry in overlap.entries
        i = Int(entry.lag_idx); j = Int(entry.eul_idx)
        w = entry.volume
        target[i] += w * source[j]
        weights[i] += w
    end
    @inbounds for i in 1:overlap.n_lag
        if weights[i] > zero(T)
            target[i] /= weights[i]
        end
    end
    return target
end

function remap_e_to_l!(target::AbstractVector{T}, source::AbstractVector{T},
                        overlap::GeometricOverlap{D, T}, ::ConservativeFlux) where {D, T}
    length(source) == overlap.n_eul ||
        throw(DimensionMismatch("source length"))
    length(target) == overlap.n_lag ||
        throw(DimensionMismatch("target length"))
    fill!(target, zero(T))
    dst_vol = zeros(T, overlap.n_eul)
    @inbounds for entry in overlap.entries
        dst_vol[entry.eul_idx] += entry.volume
    end
    @inbounds for entry in overlap.entries
        i = Int(entry.lag_idx); j = Int(entry.eul_idx)
        if dst_vol[j] > zero(T)
            target[i] += (entry.volume / dst_vol[j]) * source[j]
        end
    end
    return target
end

# ============================================================================
# Tensor-field remap: law of total covariance
# ============================================================================

"""
    remap_l_to_e_covariance!(target_M::AbstractArray{T, 3},
                              target_mu::AbstractMatrix{T},
                              source_M::AbstractArray{T, 3},
                              source_mu::AbstractMatrix{T},
                              overlap::GeometricOverlap{D, T}) where {D, T}

Apply the law of total covariance to remap a paired (mean, covariance)
field from Lagrangian to Eulerian.

# Inputs

- `source_M[a, b, i]` — covariance tensor of source cell `i`, symmetric in
  `(a, b)`, with `1 ≤ a, b ≤ D`. Length `n_lag` along the third axis.
- `source_mu[d, i]` — mean (e.g. velocity centroid) of source cell `i`,
  with `1 ≤ d ≤ D`. Length `n_lag` along the second axis.

# Outputs

- `target_M[a, b, j]` — destination covariance.
- `target_mu[d, j]` — destination mean (volume-weighted average).

The destination covariance has two contributions:

    target_M[a, b, j] = (Σ_i w_ij * source_M[a, b, i]) / W_j     (within-cell)
                      + (Σ_i w_ij * Δ[a] * Δ[b]) / W_j            (between-cell)

where `Δ = source_mu[:, i] - target_mu[:, j]` and `W_j = Σ_i w_ij`. The
between-cell term is the spread of source means and is what standard
remaps lose to numerical dissipation.

This is the centroid-only version; the full polynomial-source version
(where source data is a polynomial reconstruction integrated against the
overlap moments) is a TODO.
"""
function remap_l_to_e_covariance!(target_M::AbstractArray{T, 3},
                                    target_mu::AbstractMatrix{T},
                                    source_M::AbstractArray{T, 3},
                                    source_mu::AbstractMatrix{T},
                                    overlap::GeometricOverlap{D, T}) where {D, T}
    size(source_M) == (D, D, overlap.n_lag) ||
        throw(DimensionMismatch("source_M size"))
    size(source_mu) == (D, overlap.n_lag) ||
        throw(DimensionMismatch("source_mu size"))
    size(target_M) == (D, D, overlap.n_eul) ||
        throw(DimensionMismatch("target_M size"))
    size(target_mu) == (D, overlap.n_eul) ||
        throw(DimensionMismatch("target_mu size"))

    fill!(target_M, zero(T))
    fill!(target_mu, zero(T))
    weights = zeros(T, overlap.n_eul)

    # First pass: accumulate volume-weighted source means and within-cell M
    @inbounds for entry in overlap.entries
        i = Int(entry.lag_idx); j = Int(entry.eul_idx); w = entry.volume
        weights[j] += w
        for d in 1:D
            target_mu[d, j] += w * source_mu[d, i]
        end
        for a in 1:D, b in 1:D
            target_M[a, b, j] += w * source_M[a, b, i]
        end
    end

    # Normalize means
    @inbounds for j in 1:overlap.n_eul
        if weights[j] > zero(T)
            for d in 1:D
                target_mu[d, j] /= weights[j]
            end
        end
    end

    # Second pass: add between-cell (Δ Δ^T) contribution and normalize M
    @inbounds for entry in overlap.entries
        i = Int(entry.lag_idx); j = Int(entry.eul_idx); w = entry.volume
        for a in 1:D, b in 1:D
            Δa = source_mu[a, i] - target_mu[a, j]
            Δb = source_mu[b, i] - target_mu[b, j]
            target_M[a, b, j] += w * Δa * Δb
        end
    end

    @inbounds for j in 1:overlap.n_eul
        if weights[j] > zero(T)
            for a in 1:D, b in 1:D
                target_M[a, b, j] /= weights[j]
            end
        end
    end

    return (target_M, target_mu)
end

# Higher-order cumulants stub
function remap_l_to_e!(::Any, ::Any, ::GeometricOverlap, ::TotalCumulants{O}) where O
    O <= 1 && throw(ArgumentError("TotalCumulants{$O} is degenerate; use MassWeightedAverage"))
    O == 2 && throw(ArgumentError("TotalCumulants{2} = TotalCovariance; use remap_l_to_e_covariance!"))
    error("TotalCumulants{$O} remap is not yet implemented; see dfmm design §6.4 " *
          "for the third-cumulant pattern when needed.")
end
