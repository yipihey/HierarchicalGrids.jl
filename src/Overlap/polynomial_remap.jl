"""
Polynomial-aware remap: integrate a per-cell polynomial reconstruction
over each overlap polytope using the entry's stored physical-frame
moments, then L²-project onto each destination cell's polynomial basis.

This is what makes the moment integration in `r3d_adapter.jl` pay off:
without it, a `MassWeightedAverage` remap throws away all moments above
the 0th. With it, we get conservative, sub-cell-accurate transport of
piecewise polynomial reconstructions — the "fourth-order tracer
fidelity" the dfmm design talks about for cubic edges.

# Math sketch

Source cell `i` carries a polynomial reconstruction in its **reference
frame** ξ_i ∈ R^D, with coefficients c_{i,α} in the monomial basis:

    p_i(ξ_i) = Σ_α c_{i,α} ξ_i^α

The reference frame is described by an affine map φ_i: ξ_i → x. The
**pullback matrix** T_i expresses each reference monomial as a polynomial
in physical x:

    ξ_i^α  =  Σ_β T_i[α, β] x^β

so the source polynomial in physical coordinates is

    p_i(φ_i^{-1}(x))  =  Σ_β P_i[β] x^β        with P_i[β] = Σ_α c_{i,α} T_i[α, β]

Destination cell `j` is described by its own affine map φ_j: ξ_j → x,
with pullback U_j:

    ξ_j^β  =  Σ_γ U_j[β, γ] x^γ

L² projection of p_i onto the destination's monomial basis (degree P_dst)
asks for d_j such that for every test monomial ξ_j^β with |β| ≤ P_dst,

    Σ_α M^{(j)}_{αβ} d_{j,α}  =  Σ_i ∫_{S_i ∩ T_j} ξ_j^β · p_i(φ_i^{-1}(x)) dx

where M^{(j)} is the destination's reference-frame mass matrix and the
right-hand side becomes, after pullback:

    Σ_i Σ_γ Σ_δ U_j[β, γ] · P_i[δ] · ∫_{S_i ∩ T_j} x^{γ+δ} dx
                                       └────── overlap moments M_{γ+δ} ──────┘

Solving the per-cell mass system gives d_j. Each contribution requires
moments up to degree `P_src + P_dst` in physical coordinates — that's the
moment_order the GeometricOverlap must have been built with.

# What's implemented

- `AxisAlignedRef`, `SimplicialRef` — concrete reference-frame types and
  their pullback matrices.
- `reference_to_physical_pullback(frame, P)` — the matrix T such that
  ξ^α = Σ_β T[α, β] x^β, in graded-lex order. Cached per (frame, P).
- `reference_mass_matrix(frame, P)` — the M^{(j)} mass matrix; analytic
  for both AABB and simplex references.
- `eulerian_frame`, `lagrangian_frame` — convenience constructors that
  pull frames out of an `EulerianFrame`/`SimplicialMesh`.
- `integrate_polynomial_over_overlap` — the per-pair scalar
  ∫_Ω p_i(φ_i^{-1}(x)) dx.
- `accumulate_polynomial_rhs!` — assemble the RHS vector for the
  destination L² system. The kernel for both directions (L→E and E→L)
  is the same; only the source/destination roles swap.
- `polynomial_remap_l_to_e!`, `polynomial_remap_e_to_l!` — full
  L²-projection remap with destination mass-matrix solve.

# What's not yet (reasonable follow-ups)

- Bernstein-basis adapter: the `Bases` module already has
  `change_basis(BernsteinBasis, MonomialBasis, ...)` but only for D=1.
  The dfmm Bayesian remap will likely live in Bernstein for positivity;
  add multi-D bidirectional change_basis when needed.
- `PolynomialFieldSet` thin wrapper: take/return `PolynomialFieldSet`
  containers so dfmm code can call `polynomial_remap!(pfs_e, pfs_l,
  overlap)` directly. Easy to layer on top.
- Conservative variant: the L² projection conserves the integral
  ∫_{T_j} d_j(ξ_j) dξ_j by construction (since the constant is in the
  basis), but the integral over the **physical** target cell weights by
  the cell's volume. For exact mass conservation across hierarchy
  changes we'd want to track a separate conservative-flux contribution
  — same as the scalar `ConservativeFlux` operator in `remap.jl`. Add
  if/when needed.
"""

# ============================================================================
# Cell reference frames
# ============================================================================

"""
    CellReferenceFrame{D, T}

Abstract type for the affine map φ: ξ → x carried by one mesh cell. Each
concrete subtype implements `reference_to_physical_pullback(frame, P)`,
giving the matrix T such that ξ^α = Σ_β T[α, β] x^β for all multi-indices
|α| ≤ P, both in graded-lex order with first index most significant.
"""
abstract type CellReferenceFrame{D, T} end

"""
    AxisAlignedRef{D, T}(lo, hi) <: CellReferenceFrame{D, T}

Reference frame for an axis-aligned box, with reference coordinates
ξ ∈ [0, 1]^D mapped to physical x by `x_d = lo_d + (hi_d - lo_d) ξ_d`.

Used for Eulerian (HierarchicalMesh) cells.
"""
struct AxisAlignedRef{D, T} <: CellReferenceFrame{D, T}
    lo::NTuple{D, T}
    hi::NTuple{D, T}

    function AxisAlignedRef{D, T}(lo::NTuple{D, T}, hi::NTuple{D, T}) where {D, T}
        for d in 1:D
            hi[d] > lo[d] || throw(ArgumentError("hi[$d] = $(hi[d]) must exceed lo[$d] = $(lo[d])"))
        end
        return new{D, T}(lo, hi)
    end
end

AxisAlignedRef(lo::NTuple{D, T}, hi::NTuple{D, T}) where {D, T} =
    AxisAlignedRef{D, T}(lo, hi)

"""
    SimplicialRef{D, T}(anchor, edges) <: CellReferenceFrame{D, T}

Reference frame for a D-simplex, with reference coordinates ξ in the
standard unit simplex `{ξ_k ≥ 0, Σ_k ξ_k ≤ 1}` (origin + D unit basis
vectors as the D+1 vertices) mapped to physical x by
`x = anchor + Σ_k edges[k] ξ_k`.

`anchor` is the simplex's "anchor" vertex (vertex 1 in mesh ordering);
`edges[k]` is `vertex[k+1] - vertex[1]` for k = 1, …, D.

Used for Lagrangian (SimplicialMesh) cells.
"""
struct SimplicialRef{D, T} <: CellReferenceFrame{D, T}
    anchor::NTuple{D, T}
    edges::NTuple{D, NTuple{D, T}}

    function SimplicialRef{D, T}(anchor::NTuple{D, T}, edges::NTuple{D, NTuple{D, T}}) where {D, T}
        return new{D, T}(anchor, edges)
    end
end

SimplicialRef(anchor::NTuple{D, T}, edges::NTuple{D, NTuple{D, T}}) where {D, T} =
    SimplicialRef{D, T}(anchor, edges)

"""
    eulerian_frame(frame::EulerianFrame{D, T}, leaf_idx::Integer) -> AxisAlignedRef{D, T}

Build the reference frame for a single Eulerian leaf cell.
"""
@inline function eulerian_frame(frame::EulerianFrame{D, T}, leaf_idx::Integer) where {D, T}
    lo, hi = cell_physical_box(frame, leaf_idx)
    return AxisAlignedRef{D, T}(lo, hi)
end

"""
    lagrangian_frame(mesh::SimplicialMesh{D, T}, simplex_idx::Integer) -> SimplicialRef{D, T}

Build the reference frame for a single Lagrangian simplex.
"""
function lagrangian_frame(mesh::SimplicialMesh{D, T}, simplex_idx::Integer) where {D, T}
    verts = simplex_vertex_positions(mesh, simplex_idx)
    anchor = NTuple{D, T}(verts[1])
    edges = ntuple(k -> ntuple(d -> T(verts[k + 1][d]) - anchor[d], Val(D)), Val(D))
    return SimplicialRef{D, T}(anchor, edges)
end

# ============================================================================
# Pullback matrices: ξ^α = Σ_β T[α, β] x^β
# ============================================================================

"""
    reference_to_physical_pullback(frame::CellReferenceFrame{D, T}, P::Integer) -> Matrix{T}

Compute the pullback matrix T ∈ R^{n × n} where `n = moments_length(D, P)`,
such that for every reference multi-index α (|α| ≤ P) in graded-lex order,

    ξ^α = Σ_β T[α, β] x^β    (sum over physical multi-indices |β| ≤ P)

For axis-aligned references the pullback is sparse and the substitution
expands cheaply via single-axis binomial expansions. For simplicial
references the pullback is dense and we expand via repeated polynomial
multiplication on the per-axis affine maps ξ_d(x).

Allocates a fresh `n × n` matrix on each call. For tight inner loops, see
the cached helpers in the assembly routines.
"""
function reference_to_physical_pullback end

function reference_to_physical_pullback(frame::AxisAlignedRef{D, T}, P::Integer) where {D, T}
    P = Int(P)
    n = moments_length(D, P)
    T_mat = zeros(T, n, n)
    if P == 0
        @inbounds T_mat[1, 1] = one(T)
        return T_mat
    end
    multi = moment_multiindices(D, P)
    # ξ_d = (x_d - lo_d) / h_d where h_d = hi_d - lo_d
    # ξ^α = ∏_d (x_d - lo_d)^{α_d} / h_d^{α_d}
    #     = ∏_d (1/h_d^{α_d}) Σ_{k_d=0..α_d} C(α_d, k_d) x_d^{k_d} (-lo_d)^{α_d - k_d}
    h = ntuple(d -> frame.hi[d] - frame.lo[d], Val(D))
    @inbounds for (αi, α) in enumerate(multi)
        # Build the multi-index β by independent per-axis binomial expansion.
        # Each β has β_d ∈ 0..α_d. Multiply per-axis coefficients.
        # We enumerate via a "ranges" loop.
        _expand_axisaligned!(T_mat, αi, α, frame.lo, h, multi, T)
    end
    return T_mat
end

# Helper: fill row `αi` of the pullback for an axis-aligned ref.
function _expand_axisaligned!(T_mat::Matrix{T}, αi::Int, α::Vector{Int},
                                lo::NTuple{D, T}, h::NTuple{D, T},
                                multi::Vector{Vector{Int}}, ::Type{T}) where {D, T}
    # Product of per-axis 1/h^{α_d}
    inv_h_prod = one(T)
    for d in 1:D
        inv_h_prod *= one(T) / h[d]^α[d]
    end
    # We need to enumerate all β with β_d ≤ α_d and assemble the coefficient.
    # Use a recursive enumeration — small in practice (typically α_d ≤ 3).
    β = zeros(Int, D)
    _accumulate_axisaligned_coefs!(T_mat, αi, β, α, 1, lo, inv_h_prod, multi, T, one(T))
end

function _accumulate_axisaligned_coefs!(T_mat::Matrix{T}, αi::Int,
                                          β::Vector{Int}, α::Vector{Int},
                                          d::Int, lo::NTuple{D, T},
                                          prefix_coef::T,
                                          multi::Vector{Vector{Int}},
                                          ::Type{T}, inv_h_prod::T) where {D, T}
    if d > D
        # Look up β's flat index and add prefix_coef * inv_h_prod
        βi = _lookup_multi_index(multi, β, D)
        @inbounds T_mat[αi, βi] += prefix_coef * inv_h_prod
        return
    end
    # For axis d: β[d] runs from 0 to α[d]. Coefficient is C(α[d], β[d]) (-lo[d])^{α[d]-β[d]}.
    @inbounds for k in 0:α[d]
        β[d] = k
        sign_term = (-lo[d])^(α[d] - k)
        coef = T(binomial(α[d], k)) * sign_term
        _accumulate_axisaligned_coefs!(T_mat, αi, β, α, d + 1,
                                         lo, prefix_coef * coef, multi, T, inv_h_prod)
    end
    @inbounds β[d] = 0  # restore (not strictly needed since loop overwrites)
    return
end

function reference_to_physical_pullback(frame::SimplicialRef{D, T}, P::Integer) where {D, T}
    P = Int(P)
    n = moments_length(D, P)
    T_mat = zeros(T, n, n)
    multi = moment_multiindices(D, P)
    # Reference-to-physical: x = anchor + Σ_k edges[k] ξ_k
    # ⇒ x_e = anchor[e] + Σ_k edges[k][e] ξ_k    (per output axis e)
    # We need physical-to-reference: ξ = A^{-1} (x - anchor)
    # where A is the D×D matrix whose k-th column is edges[k].
    # ξ_k = Σ_e Ainv[k, e] (x_e - anchor[e])
    Amat = SMatrix{D, D, T}(ntuple(i -> begin
        e = ((i - 1) ÷ D) + 1   # column index 1..D corresponds to edges[e]
        r = ((i - 1) % D) + 1   # row index = output axis
        frame.edges[e][r]
    end, Val(D * D)))
    Ainv = inv(Amat)
    # P = 0 short-circuit: only the constant monomial ξ^(0,...,0) = 1, which is
    # also the constant 1 in physical x. The pullback is the 1×1 identity.
    if P == 0
        @inbounds T_mat[1, 1] = one(T)
        return T_mat
    end
    # For each ξ_k: it's a linear polynomial in x. Build its monomial-coefficient
    # vector once, then multiply for ξ^α = ∏_k (ξ_k)^{α_k}.
    # Linear polynomial representation: dense Vector{T} of length n indexed by the
    # graded-lex order. The constant and linear-in-x_d entries are populated;
    # everything else is zero.
    xi_coefs = Vector{Vector{T}}(undef, D)
    for k in 1:D
        v = zeros(T, n)
        # ξ_k = -Σ_e Ainv[k, e] anchor[e] + Σ_e Ainv[k, e] x_e
        const_term = zero(T)
        for e in 1:D
            const_term -= Ainv[k, e] * frame.anchor[e]
        end
        v[1] = const_term
        # Linear monomials live at indices 2..D+1; x_e is at index 1+e.
        for e in 1:D
            v[1 + e] = Ainv[k, e]
        end
        xi_coefs[k] = v
    end
    # Now build the row of T for each α: ξ^α = ∏_k (ξ_k)^{α_k}.
    # Multiply polynomials in graded-lex coefficient form, truncating at degree P.
    @inbounds for (αi, α) in enumerate(multi)
        # Start with the "1" polynomial (constant 1).
        prod_coefs = zeros(T, n)
        prod_coefs[1] = one(T)
        prod_deg = 0
        for k in 1:D
            for _ in 1:α[k]
                prod_coefs = _poly_multiply_truncated(prod_coefs, xi_coefs[k], prod_deg, 1, P, multi, D)
                prod_deg += 1
            end
        end
        @inbounds for βi in 1:n
            T_mat[αi, βi] = prod_coefs[βi]
        end
    end
    return T_mat
end

# Multiply two polynomials (each represented as a graded-lex coefficient vector)
# truncated to total degree ≤ P. Both inputs and output use the same multi-index list.
# `deg_a` and `deg_b` are upper bounds on total degree of nonzero entries in
# `a` and `b`; used to skip work when small.
function _poly_multiply_truncated(a::Vector{T}, b::Vector{T},
                                    deg_a::Int, deg_b::Int, P::Int,
                                    multi::Vector{Vector{Int}}, D::Int) where T
    n = length(multi)
    out = zeros(T, n)
    @inbounds for i in 1:n
        ai = a[i]
        ai == zero(T) && continue
        sum_i = sum(multi[i])
        sum_i > deg_a && continue
        for j in 1:n
            bj = b[j]
            bj == zero(T) && continue
            sum_j = sum(multi[j])
            sum_j > deg_b && continue
            total_deg = sum_i + sum_j
            total_deg > P && continue
            # Compute the product index γ = α + β (componentwise), look up in multi.
            γ = ntuple(d -> multi[i][d] + multi[j][d], D)
            γi = _lookup_multi_index(multi, collect(γ), D)
            out[γi] += ai * bj
        end
    end
    return out
end

# Lookup a multi-index in the graded-lex `multi` list. Linear scan; multi is small.
function _lookup_multi_index(multi::Vector{Vector{Int}}, target, D::Int)
    @inbounds for k in eachindex(multi)
        ok = true
        for d in 1:D
            if multi[k][d] != target[d]
                ok = false; break
            end
        end
        ok && return k
    end
    error("multi-index $target not found (out of range for the list)")
end

# ============================================================================
# Reference-frame mass matrices for L² projection
# ============================================================================

"""
    reference_mass_matrix(frame::CellReferenceFrame{D, T}, P::Integer) -> Symmetric{T}

Mass matrix M with `M[α, β] = ∫_{ref} ξ^α ξ^β dξ = ∫_{ref} ξ^{α+β} dξ` for
multi-indices α, β with `|α|, |β| ≤ P`, in graded-lex order.

Closed-form for both AABB (`∫_{[0,1]^D} ξ^γ dξ = ∏_d 1/(γ_d + 1)`) and
unit simplex (`∫_{Δ^D} ξ^γ dξ = ∏_d γ_d! / (Σγ_d + D)!`).

Symmetric and positive-definite.
"""
function reference_mass_matrix(frame::AxisAlignedRef{D, T}, P::Integer) where {D, T}
    P = Int(P)
    multi = moment_multiindices(D, P)
    n = length(multi)
    M = zeros(T, n, n)
    @inbounds for i in 1:n, j in i:n
        # γ = α + β
        val = one(T)
        for d in 1:D
            val *= one(T) / T(multi[i][d] + multi[j][d] + 1)
        end
        M[i, j] = val
        M[j, i] = val
    end
    return M
end

function reference_mass_matrix(frame::SimplicialRef{D, T}, P::Integer) where {D, T}
    P = Int(P)
    multi = moment_multiindices(D, P)
    n = length(multi)
    M = zeros(T, n, n)
    @inbounds for i in 1:n, j in i:n
        # γ = α + β; ∫_{Δ^D} ξ^γ dξ = ∏_d γ_d! / (|γ| + D)!
        γ = ntuple(d -> multi[i][d] + multi[j][d], D)
        numer = one(T)
        for d in 1:D
            numer *= T(factorial(γ[d]))
        end
        denom = T(factorial(sum(γ) + D))
        val = numer / denom
        M[i, j] = val
        M[j, i] = val
    end
    return M
end

# ============================================================================
# Per-pair scalar integration
# ============================================================================

"""
    integrate_polynomial_over_overlap(src_coeffs::AbstractVector,
                                        src_pullback::AbstractMatrix,
                                        entry::OverlapEntry,
                                        P_src::Integer) -> T

Compute `∫_{overlap} p_src(φ_src^{-1}(x)) dx` for one overlap entry, where
`p_src(ξ) = Σ_α src_coeffs[α] ξ^α` is the source polynomial in its
reference frame.

`src_pullback` is the precomputed `T` matrix from
`reference_to_physical_pullback(src_frame, P_src)`.

Requires `entry.moments` to have been computed at order ≥ `P_src`. Reads
moments up to total degree `P_src` only.
"""
function integrate_polynomial_over_overlap(src_coeffs::AbstractVector{T},
                                             src_pullback::AbstractMatrix{T},
                                             entry::OverlapEntry{D, S},
                                             P_src::Integer) where {D, T, S}
    n_src = moments_length(D, Int(P_src))
    length(src_coeffs) == n_src ||
        throw(DimensionMismatch("src_coeffs length $(length(src_coeffs)) ≠ $n_src for D=$D, P=$P_src"))
    size(src_pullback) == (n_src, n_src) ||
        throw(DimensionMismatch("src_pullback size $(size(src_pullback)) ≠ ($n_src, $n_src)"))
    length(entry.moments) >= n_src ||
        throw(DimensionMismatch("entry.moments length $(length(entry.moments)) < $n_src; rebuild overlap with moment_order ≥ $P_src"))

    # Compute physical-coordinate coefficients P[β] = Σ_α src_coeffs[α] · src_pullback[α, β]
    # then integrate: result = Σ_β P[β] · entry.moments[β]
    s = zero(T)
    @inbounds for β in 1:n_src
        Pβ = zero(T)
        for α in 1:n_src
            Pβ += src_coeffs[α] * src_pullback[α, β]
        end
        s += Pβ * T(entry.moments[β])
    end
    return s
end

# ============================================================================
# RHS assembly for L² projection
# ============================================================================

"""
    accumulate_polynomial_rhs!(rhs::AbstractMatrix{T},
                                 src_coeffs::AbstractMatrix{T},
                                 src_pullbacks::Vector{<:AbstractMatrix{T}},
                                 dst_pullbacks::Vector{<:AbstractMatrix{T}},
                                 overlap::GeometricOverlap{D, T},
                                 P_src::Integer, P_dst::Integer;
                                 invert::Bool = false) where {D, T}

Accumulate the right-hand side of the destination L² projection system
into `rhs[α, j] += Σ_i Σ_γ Σ_δ U_j[α, γ] · P_i[δ] · M_{γ+δ}` for every
overlap entry, where:
- `α` runs over destination monomials (|α| ≤ P_dst), so `rhs` is shaped
  `(n_dst, n_dst_cells)` with `n_dst = moments_length(D, P_dst)`.
- The "source" runs over Lagrangian simplices when `invert=false`; over
  Eulerian leaves when `invert=true`.

`src_pullbacks[i]` is the `(n_src, n_src)` pullback for source cell `i`;
`dst_pullbacks[j]` is the `(n_dst, n_dst)` pullback for destination cell `j`.

Each overlap entry must have `moments` up to order `P_src + P_dst` —
otherwise the upper-degree terms can't be computed and the routine
throws.
"""
function accumulate_polynomial_rhs!(rhs::AbstractMatrix{T},
                                      src_coeffs::AbstractMatrix{T},
                                      src_pullbacks::Vector{<:AbstractMatrix{T}},
                                      dst_pullbacks::Vector{<:AbstractMatrix{T}},
                                      overlap::GeometricOverlap{D, T},
                                      P_src::Integer, P_dst::Integer;
                                      invert::Bool = false) where {D, T}
    P_src = Int(P_src); P_dst = Int(P_dst)
    n_src = moments_length(D, P_src)
    n_dst = moments_length(D, P_dst)
    n_phys = moments_length(D, P_src + P_dst)
    moment_order(overlap) >= P_src + P_dst ||
        throw(ArgumentError("overlap was built with moment_order=$(moment_order(overlap)); needs ≥ $(P_src + P_dst) for P_src=$P_src, P_dst=$P_dst"))
    size(rhs, 1) == n_dst ||
        throw(DimensionMismatch("rhs first dim $(size(rhs, 1)) ≠ $n_dst"))
    size(src_coeffs, 1) == n_src ||
        throw(DimensionMismatch("src_coeffs first dim $(size(src_coeffs, 1)) ≠ $n_src"))

    multi_src = moment_multiindices(D, P_src)
    multi_dst = moment_multiindices(D, P_dst)
    multi_phys = moment_multiindices(D, P_src + P_dst)

    # Precompute the lookup index of γ + δ into multi_phys for every (γ, δ) pair
    # — this is O(n_dst * n_src) per build, used for every entry, much faster
    # than searching multi_phys per entry.
    lookup = Matrix{Int}(undef, n_dst, n_src)
    @inbounds for γi in 1:n_dst, δi in 1:n_src
        sum_idx = ntuple(d -> multi_dst[γi][d] + multi_src[δi][d], D)
        lookup[γi, δi] = _lookup_multi_index(multi_phys, collect(sum_idx), D)
    end

    fill!(rhs, zero(T))
    # Per-entry buffer: physical-coordinate coefficients of the source polynomial
    Pi_buf = Vector{T}(undef, n_src)

    @inbounds for entry in overlap.entries
        i = invert ? Int(entry.eul_idx) : Int(entry.lag_idx)
        j = invert ? Int(entry.lag_idx) : Int(entry.eul_idx)

        # P_i[δ] = Σ_α src_coeffs[α, i] · src_pullback_i[α, δ]
        src_pull = src_pullbacks[i]
        for δ in 1:n_src
            Piδ = zero(T)
            for α in 1:n_src
                Piδ += src_coeffs[α, i] * src_pull[α, δ]
            end
            Pi_buf[δ] = Piδ
        end

        # rhs[α, j] += Σ_γ U_j[α, γ] · Σ_δ P_i[δ] · M_{γ+δ}
        dst_pull = dst_pullbacks[j]
        for α in 1:n_dst
            acc = zero(T)
            for γ in 1:n_dst
                Uαγ = dst_pull[α, γ]
                Uαγ == zero(T) && continue
                inner = zero(T)
                for δ in 1:n_src
                    inner += Pi_buf[δ] * entry.moments[lookup[γ, δ]]
                end
                acc += Uαγ * inner
            end
            rhs[α, j] += acc
        end
    end
    return rhs
end

# ============================================================================
# High-level remap: full L² projection
# ============================================================================

"""
    polynomial_remap_l_to_e!(target_coeffs::AbstractMatrix{T},
                              source_coeffs::AbstractMatrix{T},
                              overlap::GeometricOverlap{D, T},
                              src_frames::Vector{<:CellReferenceFrame{D, T}},
                              dst_frames::Vector{<:CellReferenceFrame{D, T}},
                              P_src::Integer, P_dst::Integer)

L²-projection remap of a per-cell polynomial reconstruction from the
Lagrangian side to the Eulerian side via the precomputed `overlap`.

# Arguments

- `source_coeffs[α, i]` — coefficient of the `α`-th monomial (graded-lex,
  in the source cell's reference frame) for source cell `i`. Size
  `(moments_length(D, P_src), overlap.n_lag)`.
- `target_coeffs[α, j]` — destination coefficients in the destination
  cell's reference frame. Size `(moments_length(D, P_dst), overlap.n_eul)`.
  Filled in place.
- `src_frames`, `dst_frames` — reference frames for source and destination
  cells. Lengths `overlap.n_lag` and `overlap.n_eul`.
- `P_src`, `P_dst` — polynomial orders for source and destination.

The overlap's `moment_order` must be at least `P_src + P_dst`.

For destination cells with no overlap, `target_coeffs` is left at zero.

# Algorithm

For each destination cell `j`:
1. Compute the L² rhs `b_j[α] = Σ_i ∫_{S_i ∩ T_j} ξ_j^α p_i(φ_i^{-1}(x)) dx`
   via `accumulate_polynomial_rhs!`.
2. Solve the per-cell mass system `M_j d_j = b_j` where `M_j` is the
   reference-frame mass matrix.

Both steps are exact (modulo the moment-order requirement). The result
is the unique L² projection of the piecewise-polynomial source onto the
destination's piecewise-polynomial space.
"""
function polynomial_remap_l_to_e!(target_coeffs::AbstractMatrix{T},
                                    source_coeffs::AbstractMatrix{T},
                                    overlap::GeometricOverlap{D, T},
                                    src_frames::Vector{<:CellReferenceFrame{D, T}},
                                    dst_frames::Vector{<:CellReferenceFrame{D, T}},
                                    P_src::Integer, P_dst::Integer) where {D, T}
    _polynomial_remap!(target_coeffs, source_coeffs, overlap,
                        src_frames, dst_frames, P_src, P_dst; invert = false)
end

"""
    polynomial_remap_e_to_l!(target_coeffs, source_coeffs, overlap,
                              src_frames, dst_frames, P_src, P_dst)

Reverse direction: Eulerian → Lagrangian. Same as
`polynomial_remap_l_to_e!` with the roles of `lag_idx` and `eul_idx`
swapped in the overlap entries; `src_frames` here is for Eulerian leaves
and `dst_frames` is for Lagrangian simplices.
"""
function polynomial_remap_e_to_l!(target_coeffs::AbstractMatrix{T},
                                    source_coeffs::AbstractMatrix{T},
                                    overlap::GeometricOverlap{D, T},
                                    src_frames::Vector{<:CellReferenceFrame{D, T}},
                                    dst_frames::Vector{<:CellReferenceFrame{D, T}},
                                    P_src::Integer, P_dst::Integer) where {D, T}
    _polynomial_remap!(target_coeffs, source_coeffs, overlap,
                        src_frames, dst_frames, P_src, P_dst; invert = true)
end

function _polynomial_remap!(target_coeffs::AbstractMatrix{T},
                              source_coeffs::AbstractMatrix{T},
                              overlap::GeometricOverlap{D, T},
                              src_frames::Vector{<:CellReferenceFrame{D, T}},
                              dst_frames::Vector{<:CellReferenceFrame{D, T}},
                              P_src::Integer, P_dst::Integer;
                              invert::Bool) where {D, T}
    P_src = Int(P_src); P_dst = Int(P_dst)
    n_dst = moments_length(D, P_dst)

    n_src_cells = invert ? overlap.n_eul : overlap.n_lag
    n_dst_cells = invert ? overlap.n_lag : overlap.n_eul

    length(src_frames) == n_src_cells ||
        throw(ArgumentError("src_frames length $(length(src_frames)) ≠ $n_src_cells"))
    length(dst_frames) == n_dst_cells ||
        throw(ArgumentError("dst_frames length $(length(dst_frames)) ≠ $n_dst_cells"))
    size(source_coeffs, 2) == n_src_cells ||
        throw(DimensionMismatch("source_coeffs second dim $(size(source_coeffs, 2)) ≠ $n_src_cells"))
    size(target_coeffs, 2) == n_dst_cells ||
        throw(DimensionMismatch("target_coeffs second dim $(size(target_coeffs, 2)) ≠ $n_dst_cells"))
    size(target_coeffs, 1) == n_dst ||
        throw(DimensionMismatch("target_coeffs first dim $(size(target_coeffs, 1)) ≠ $n_dst"))
    size(source_coeffs, 1) == moments_length(D, P_src) ||
        throw(DimensionMismatch("source_coeffs first dim $(size(source_coeffs, 1)) ≠ $(moments_length(D, P_src))"))

    # Precompute pullback matrices per cell. These dominate setup cost but
    # are reused across every overlap entry that touches the cell.
    src_pullbacks = [reference_to_physical_pullback(f, P_src) for f in src_frames]
    dst_pullbacks = [reference_to_physical_pullback(f, P_dst) for f in dst_frames]

    # Assemble RHS for every destination cell
    rhs = zeros(T, n_dst, n_dst_cells)
    accumulate_polynomial_rhs!(rhs, source_coeffs, src_pullbacks, dst_pullbacks,
                                 overlap, P_src, P_dst; invert = invert)

    # Solve M_j d_j = b_j for every destination cell with nonzero rhs.
    # The destination mass matrix lives in the destination's *reference frame*
    # weighted by |det J_dst| (the destination cell's physical volume).
    # ∫_{T_j} ξ^α ξ^β dx = |det J_dst| · ∫_{ref} ξ^α ξ^β dξ
    # so we can either bake |det J_dst| into the mass matrix or factor it out
    # of both M_j and b_j (giving the same d_j). Cleaner: factor it out.
    # But the rhs we accumulated is in physical x; we have ∫ ξ^α p dx (no
    # cell-volume factor). The mass matrix should be ∫_{T_j} ξ^α ξ^β dx as well
    # so the volume factors cancel cleanly when we solve M_j d_j = b_j.
    #
    # ⇒ Use M_j = |det J_dst| · M_ref and use b_j as accumulated.
    fill!(target_coeffs, zero(T))
    for j in 1:n_dst_cells
        if any(rhs[α, j] != zero(T) for α in 1:n_dst)
            M_ref = reference_mass_matrix(dst_frames[j], P_dst)
            jac = _frame_jacobian(dst_frames[j])
            M_j = jac .* M_ref
            d = M_j \ view(rhs, :, j)
            for α in 1:n_dst
                target_coeffs[α, j] = d[α]
            end
        end
    end
    return target_coeffs
end

"""
    _frame_jacobian(frame::CellReferenceFrame) -> T

Volume of the physical cell divided by the volume of the reference cell.
For axis-aligned: ∏_d (hi_d - lo_d). For simplicial: |det A| (where A is
the edge matrix), since the unit simplex has volume 1/D! and a physical
simplex from edges A has volume |det A|/D!.
"""
@inline function _frame_jacobian(frame::AxisAlignedRef{D, T}) where {D, T}
    j = one(T)
    @inbounds for d in 1:D
        j *= (frame.hi[d] - frame.lo[d])
    end
    return j
end

@inline function _frame_jacobian(frame::SimplicialRef{D, T}) where {D, T}
    Amat = SMatrix{D, D, T}(ntuple(i -> begin
        e = ((i - 1) ÷ D) + 1
        r = ((i - 1) % D) + 1
        frame.edges[e][r]
    end, Val(D * D)))
    return abs(det(Amat))
end
