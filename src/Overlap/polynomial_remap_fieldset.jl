"""
Thin wrapper around `polynomial_remap_l_to_e!` / `polynomial_remap_e_to_l!`
that takes `PolynomialFieldSet` containers directly, so dfmm code can
remap a field by name without manually reshaping coefficient matrices.

# Layout fast path

For `SoA`-laid-out polynomial fields (the framework's default and the
right choice for hot loops), the wrapper exposes the underlying
contiguous storage as a `(n_coeffs, n_elements)` `Matrix{T}` view via
`reshape` — zero copies, the polynomial-remap kernel writes directly
into the target field's storage.

For `AoS` and `Blocked` layouts, the wrapper transparently allocates a
temporary `(n_coeffs, n_elements)` matrix, copies coefficients in,
runs the kernel, and copies back. Functionally equivalent but allocates
proportional to total coefficient count. Document this clearly so users
know to pick `SoA` for performance.

# Basis requirement

The polynomial-remap kernel works in the monomial basis (in each cell's
reference frame). The wrapper checks that the source and target field
sets carry `MonomialBasis{D, P}`-typed bases and throws clearly if not.
For non-monomial bases, the user must convert via `change_basis` first.

# Frame caches

If you remap multiple fields between the same pair of meshes, build the
source and destination frame vectors once and reuse them across calls.
The `polynomial_remap_field!` driver re-computes per-cell pullback
matrices each call (they're cheap relative to per-cell solves), but
the frames themselves don't change between fields.
"""

# ============================================================================
# Coefficient-matrix views
# ============================================================================

"""
    polynomial_coeffs_view(pfs::PolynomialFieldSet, fieldname::Symbol) -> Matrix{T}

Return a `(n_coeffs, n_elements)` matrix view of the underlying storage
for the named polynomial field. **Zero-copy** for `SoA` layout — writes
to the returned matrix directly modify the field set.

Throws `ArgumentError` for `AoS` and `Blocked` layouts (no contiguous
view exists). Use `polynomial_coeffs_matrix` for those, which always
allocates a copy.
"""
function polynomial_coeffs_view(pfs::PolynomialFieldSet, fieldname::Symbol)
    L = _layout_type_poly(pfs)
    return _polynomial_coeffs_view_impl(pfs.storage, L, Val(fieldname),
                                          n_coeffs(pfs.basis), pfs.n)
end

@inline function _polynomial_coeffs_view_impl(storage::NamedTuple, ::Type{SoA},
                                                ::Val{name}, nc::Int, n::Int) where name
    name in keys(storage) ||
        throw(ArgumentError("PolynomialFieldSet has no field :$name (available: $(keys(storage)))"))
    return reshape(getfield(storage, name), nc, n)
end

@inline function _polynomial_coeffs_view_impl(::Any, ::Type{L}, ::Val{name},
                                                ::Int, ::Int) where {L, name}
    throw(ArgumentError(
        "polynomial_coeffs_view requires SoA layout (got $L). " *
        "For AoS or Blocked layouts, use polynomial_coeffs_matrix to get an allocated copy " *
        "and copy back via set_polynomial_coeffs_matrix!."))
end

"""
    polynomial_coeffs_matrix(pfs::PolynomialFieldSet, fieldname::Symbol) -> Matrix{T}

Return a `(n_coeffs, n_elements)` matrix containing a **copy** of the
field's coefficients. Works for any layout. Use
`set_polynomial_coeffs_matrix!` to write modified coefficients back.

For SoA storage, prefer `polynomial_coeffs_view` (zero-copy).
"""
function polynomial_coeffs_matrix(pfs::PolynomialFieldSet, fieldname::Symbol)
    nc = n_coeffs(pfs.basis)
    n = pfs.n
    fv = getproperty(pfs, fieldname)  # PolynomialFieldView
    # Determine element type from the first coefficient
    T = typeof(fv[1][1])
    out = Matrix{T}(undef, nc, n)
    @inbounds for i in 1:n
        pv = fv[i]
        for k in 1:nc
            out[k, i] = pv[k]
        end
    end
    return out
end

"""
    set_polynomial_coeffs_matrix!(pfs::PolynomialFieldSet, fieldname::Symbol, coeffs::AbstractMatrix)

Write a `(n_coeffs, n_elements)` matrix of coefficients back into the
named field. Inverse of `polynomial_coeffs_matrix`. Works for any
layout. For SoA, this reduces to a `copyto!` on the underlying storage.
"""
function set_polynomial_coeffs_matrix!(pfs::PolynomialFieldSet,
                                         fieldname::Symbol,
                                         coeffs::AbstractMatrix)
    nc = n_coeffs(pfs.basis)
    n = pfs.n
    size(coeffs) == (nc, n) ||
        throw(DimensionMismatch("coeffs has size $(size(coeffs)), expected ($nc, $n)"))
    fv = getproperty(pfs, fieldname)
    @inbounds for i in 1:n
        pv = fv[i]
        for k in 1:nc
            pv[k] = coeffs[k, i]
        end
    end
    return pfs
end

# ============================================================================
# High-level field-set remap
# ============================================================================

"""
    polynomial_remap_field!(target_pfs::PolynomialFieldSet,
                              target_fieldname::Symbol,
                              source_pfs::PolynomialFieldSet,
                              source_fieldname::Symbol,
                              overlap::GeometricOverlap{D, T},
                              src_frames::Vector{<:CellReferenceFrame{D, T}},
                              dst_frames::Vector{<:CellReferenceFrame{D, T}};
                              direction::Symbol = :l_to_e) where {D, T}

L²-projection remap of one field, taking and returning `PolynomialFieldSet`
containers directly. Auto-detects polynomial orders from the bases and
reads/writes coefficients via the most efficient available access path.

# Arguments

- `target_pfs`, `target_fieldname` — destination field set and field
  name. The target field's basis determines `P_dst`.
- `source_pfs`, `source_fieldname` — source field set and field name.
  The source field's basis determines `P_src`.
- `overlap` — precomputed `GeometricOverlap`. Must have `moment_order
  ≥ P_src + P_dst` (the L²-projection requirement).
- `src_frames`, `dst_frames` — reference frames per cell, on the source
  and destination sides respectively. Build with `lagrangian_frame`
  / `eulerian_frame` once and reuse across multiple field remaps.
- `direction` — `:l_to_e` (Lagrangian source → Eulerian target) or
  `:e_to_l` (Eulerian source → Lagrangian target).

# Basis requirements

Both fields must use a `MonomialBasis{D, P}` with the same `D` (matching
the overlap dimension). The polynomial orders `P_src` and `P_dst` may
differ.

# Performance

For `SoA`-laid-out fields, the source coefficients are read and target
coefficients are written through zero-copy reshape views into the
underlying storage. For `AoS`/`Blocked` layouts, the wrapper allocates
temporary copy matrices.

The polynomial-remap kernel itself precomputes per-cell pullback
matrices each call. If you remap many fields between the same mesh
pair, the per-call setup cost is `O(n_cells × n_coeffs²)` per side; the
per-field assembly is `O(n_entries × n_coeffs²_src × n_coeffs_dst)`.
"""
function polynomial_remap_field!(target_pfs::PolynomialFieldSet,
                                   target_fieldname::Symbol,
                                   source_pfs::PolynomialFieldSet,
                                   source_fieldname::Symbol,
                                   overlap::GeometricOverlap{D, T},
                                   src_frames::Vector{<:CellReferenceFrame{D, T}},
                                   dst_frames::Vector{<:CellReferenceFrame{D, T}};
                                   direction::Symbol = :l_to_e) where {D, T}
    direction in (:l_to_e, :e_to_l) ||
        throw(ArgumentError("direction must be :l_to_e or :e_to_l, got $direction"))

    # Validate bases
    src_basis = basis_of(source_pfs)
    dst_basis = basis_of(target_pfs)
    src_basis isa MonomialBasis ||
        throw(ArgumentError("source field set must use MonomialBasis (got $(typeof(src_basis))). " *
                             "Convert via change_basis first."))
    dst_basis isa MonomialBasis ||
        throw(ArgumentError("target field set must use MonomialBasis (got $(typeof(dst_basis))). " *
                             "Convert via change_basis first."))
    P_src = _basis_order(src_basis)
    P_dst = _basis_order(dst_basis)
    D_src = _basis_dim(src_basis)
    D_dst = _basis_dim(dst_basis)
    D_src == D || throw(ArgumentError(
        "source basis dimension $D_src ≠ overlap dimension $D"))
    D_dst == D || throw(ArgumentError(
        "target basis dimension $D_dst ≠ overlap dimension $D"))

    # Validate field-set sizes against overlap
    n_src_expected = direction === :l_to_e ? overlap.n_lag : overlap.n_eul
    n_dst_expected = direction === :l_to_e ? overlap.n_eul : overlap.n_lag
    n_elements(source_pfs) == n_src_expected ||
        throw(ArgumentError(
            "source field set has $(n_elements(source_pfs)) elements, expected $n_src_expected for $direction"))
    n_elements(target_pfs) == n_dst_expected ||
        throw(ArgumentError(
            "target field set has $(n_elements(target_pfs)) elements, expected $n_dst_expected for $direction"))

    # Get source coefficients (view if SoA, copy otherwise)
    src_coeffs = _get_coeffs_for_remap(source_pfs, source_fieldname)

    # Get target coefficients buffer
    tgt_coeffs, target_is_view = _get_coeffs_for_remap_writable(target_pfs, target_fieldname)

    # Run the kernel
    if direction === :l_to_e
        polynomial_remap_l_to_e!(tgt_coeffs, src_coeffs, overlap,
                                   src_frames, dst_frames, P_src, P_dst)
    else
        polynomial_remap_e_to_l!(tgt_coeffs, src_coeffs, overlap,
                                   src_frames, dst_frames, P_src, P_dst)
    end

    # If we worked on a copy (AoS/Blocked), write back
    if !target_is_view
        set_polynomial_coeffs_matrix!(target_pfs, target_fieldname, tgt_coeffs)
    end

    return target_pfs
end

# Convenience: same field name on both sides
"""
    polynomial_remap_field!(target_pfs, source_pfs, fieldname, overlap,
                              src_frames, dst_frames; direction=:l_to_e)

Convenience overload when the source and target fields have the same name.
"""
function polynomial_remap_field!(target_pfs::PolynomialFieldSet,
                                   source_pfs::PolynomialFieldSet,
                                   fieldname::Symbol,
                                   overlap::GeometricOverlap{D, T},
                                   src_frames::Vector{<:CellReferenceFrame{D, T}},
                                   dst_frames::Vector{<:CellReferenceFrame{D, T}};
                                   direction::Symbol = :l_to_e) where {D, T}
    return polynomial_remap_field!(target_pfs, fieldname,
                                    source_pfs, fieldname,
                                    overlap, src_frames, dst_frames;
                                    direction = direction)
end

# ============================================================================
# Helpers
# ============================================================================

@inline _basis_order(::MonomialBasis{D, P}) where {D, P} = P
@inline _basis_dim(::MonomialBasis{D, P}) where {D, P} = D

# Get source coefficients in (n_coeffs, n_elements) shape. Uses view for SoA,
# copy otherwise.
function _get_coeffs_for_remap(pfs::PolynomialFieldSet, fieldname::Symbol)
    L = _layout_type_poly(pfs)
    if L === SoA
        return polynomial_coeffs_view(pfs, fieldname)
    else
        return polynomial_coeffs_matrix(pfs, fieldname)
    end
end

# Get target coefficients buffer. Returns (matrix, is_view::Bool). When
# is_view is false, caller must copy back via set_polynomial_coeffs_matrix!.
function _get_coeffs_for_remap_writable(pfs::PolynomialFieldSet, fieldname::Symbol)
    L = _layout_type_poly(pfs)
    if L === SoA
        return (polynomial_coeffs_view(pfs, fieldname), true)
    else
        nc = n_coeffs(pfs.basis)
        T = typeof(getproperty(pfs, fieldname)[1][1])
        return (zeros(T, nc, pfs.n), false)
    end
end
