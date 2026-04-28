"""
    AdaptiveField{F <: PolynomialFieldSet, M <: HierarchicalMesh}

Wrapper that auto-registers a refinement listener on `mesh`. When the
mesh refines or coarsens, the underlying `PolynomialFieldSet`'s
coefficient storage is resized and permuted via `event.index_remap`,
preserving cell-index alignment with the mesh.

# Refinement (constant prolongation)

Each child cell inherits the parent's polynomial coefficients verbatim.
The (now non-leaf) parent slot also retains its coefficients, since
`event.index_remap[parent_old_idx]` points at the parent's NEW slot.

# Coarsening

For degree-0 fields (basis with `n_coeffs == 1`), the parent receives
the volume-weighted average of children's coefficients. Because children
of any (an)isotropic split have equal volume by construction, this
reduces to the arithmetic mean — exact and conservative.

For degree ≥ 1 `MonomialBasis` fields, the parent receives the **exact
L²-projection** of the children's piecewise-polynomial reconstruction
onto the parent's polynomial space. Mathematically, with parent cell P,
children {C_k}, basis functions φ_α on the parent reference cell:

    ⟨p_P, φ_α⟩_P = Σ_k ⟨p_{C_k}, φ_α⟩_{C_k}    for all |α| ≤ P

Solved as a per-event mass-matrix system `M_P · c_P = Σ_k M_{P,C_k} · c_{C_k}`
where the cross mass matrices `M_{P,C_k}` are integrated analytically
over each child's sub-box in the parent's reference frame. The result
preserves all moments up to the basis degree — total mass, centroid×mass,
second moments, etc. — exactly (modulo floating-point round-off).

For `BernsteinBasis{D, P}` fields, the same exact L²-projection is
applied via `change_basis`: each child's Bernstein coefficients are
converted to monomial form, the projection is performed in monomial
space, and the resulting parent coefficients are converted back to
Bernstein. The conversion matrices are cached per `(D, P)`.

For other polynomial bases (e.g. future `LagrangeBasis{D, P}` in D > 1),
AdaptiveField falls back to constant-moment-only coarsening with a
single warning.

# Field replacement

`PolynomialFieldSet` is an immutable struct (`n` is set at construction
time), so on resize we mutate the underlying storage Vectors in place
and rebuild a fresh `PolynomialFieldSet` with the new `n`. The wrapper
type is `mutable`, so `af.field` is reassigned. User code that holds
a stale reference to `af.field` will see the OLD `n`; prefer accessing
through `parent(af)`.
"""
mutable struct AdaptiveField{F, M}
    field::F
    mesh::M
    listener_handle::ListenerHandle
    disposed::Bool
end

# ----------------------------------------------------------------------------
# Constructor / disposal
# ----------------------------------------------------------------------------

function AdaptiveField(field::PolynomialFieldSet, mesh::HierarchicalMesh)
    af = AdaptiveField{typeof(field), typeof(mesh)}(field, mesh, ListenerHandle(0), false)
    handle = register_refinement_listener!(mesh, ev -> _handle_event!(af, ev))
    af.listener_handle = handle
    finalizer(_finalize_adaptive_field!, af)
    return af
end

"""
    dispose!(af::AdaptiveField)

Unregister the refinement listener. Subsequent mesh refinements will
not modify the wrapped field. Safe to call multiple times.
"""
function dispose!(af::AdaptiveField)
    af.disposed && return af
    unregister_refinement_listener!(af.mesh, af.listener_handle)
    af.disposed = true
    return af
end

function _finalize_adaptive_field!(af::AdaptiveField)
    af.disposed && return
    # Best-effort cleanup. We avoid throwing in the finalizer.
    try
        unregister_refinement_listener!(af.mesh, af.listener_handle)
    catch
        # ignore — if the mesh has been GC'd or torn down, there's nothing to do
    end
    af.disposed = true
    return
end

"""
    Base.parent(af::AdaptiveField)

Return the underlying `PolynomialFieldSet`. Note: this reference
becomes stale after the next refinement event (the wrapper rebuilds
the inner `PolynomialFieldSet` to reflect the new `n`).
"""
Base.parent(af::AdaptiveField) = af.field

# ----------------------------------------------------------------------------
# Storage resize helpers
# ----------------------------------------------------------------------------

# Resize the storage Vectors of a PolynomialFieldSet's storage in place.
# Supports SoA and AoS layouts (the two layouts directly used by current
# Phase-2 wiring). Blocked layouts are explicitly rejected with a clear
# error: AdaptiveField support for Blocked storage is a follow-up.
@inline function _resize_poly_storage_inplace!(
    storage::NamedTuple, ::Type{SoA}, nc::Int, new_n::Int)
    for k in keys(storage)
        resize!(getfield(storage, k), nc * new_n)
    end
    return storage
end

@inline function _resize_poly_storage_inplace!(
    storage::Vector, ::Type{AoS}, ::Int, new_n::Int)
    resize!(storage, new_n)
    return storage
end

@inline function _resize_poly_storage_inplace!(
    storage, ::Type{<:Blocked}, ::Int, ::Int)
    throw(ArgumentError(
        "AdaptiveField: Blocked layout resize is not yet supported. " *
        "Use SoA or AoS for adaptive fields, or open a follow-up PR " *
        "to extend `_resize_poly_storage_inplace!` to Blocked."))
end

# Build a fresh PolynomialFieldSet sharing the same storage object (which we
# resize in place) but with the new `n`. The storage itself is mutable
# (NamedTuple of Vector / Vector of NamedTuple), so the new wrapper sees the
# same data — only the `n` field needs to differ.
function _rebuild_with_n(pfs::PolynomialFieldSet{L, B, Names, ST, S},
                          new_n::Int) where {L, B, Names, ST, S}
    return PolynomialFieldSet{L, B, Names, ST, S}(new_n, pfs.basis, pfs.storage)
end

# ----------------------------------------------------------------------------
# Coefficient copy helpers
# ----------------------------------------------------------------------------

# Copy all coefficients of all fields from old element `old_i` (in the
# pre-resize layout) into a saved per-element snapshot. We snapshot every
# OLD cell's coefficients up front because the resize-in-place will clobber
# them once `new_n != old_n` (SoA) or shift them out of position (AoS).
#
# Returns a Dict{Int, NamedTuple{Names, NTuple{nc, T}...}}-like structure.
# To avoid heavy allocations, we use a NamedTuple of matrices: one matrix
# per field, of size (nc, n_old), with column i holding the coefficients
# of cell i.
function _snapshot_coefficients(pfs::PolynomialFieldSet{L, B, Names, ST, S},
                                  n_old::Int) where {L, B, Names, ST, S}
    nc = n_coeffs(pfs.basis)
    snap = NamedTuple{Names}(ntuple(length(Names)) do k
        T = fieldtype(ST, k)
        M = Matrix{T}(undef, nc, n_old)
        name = Names[k]
        @inbounds for i in 1:n_old, j in 1:nc
            M[j, i] = _get_poly_coeff(pfs, Val(name), i, j)
        end
        M
    end)
    return snap
end

# Write coefficients (taken from a snapshot column) into element `new_i` of
# the (possibly resized) PolynomialFieldSet `pfs`.
@inline function _write_coeffs_from_column!(
    pfs::PolynomialFieldSet{L, B, Names, ST, S},
    snap, new_i::Int, src_col::Int) where {L, B, Names, ST, S}
    nc = n_coeffs(pfs.basis)
    @inbounds for name in Names
        M = getfield(snap, name)
        for k in 1:nc
            _set_poly_coeff!(pfs, Val(name), new_i, k, M[k, src_col])
        end
    end
    return nothing
end

# Write averaged coefficients from a set of source columns into element new_i.
# All columns are weighted equally (the exact volume-weighted average for any
# isotropic or anisotropic split, since a single split level produces children
# of equal volume).
#
# `parent_cell` is the (post-coarsen) parent's `CellMeta`; its `split_mask`
# tells us how the children were laid out in [0, 1]^D. This is needed for
# the higher-order L²-projection branch and ignored on the degree-0 fast path.
function _write_coeffs_average!(
    pfs::PolynomialFieldSet{L, B, Names, ST, S},
    snap, new_i::Int, src_cols, parent_cell,
    basis_degree_warned::Ref{Bool}) where {L, B, Names, ST, S}
    nc = n_coeffs(pfs.basis)
    n_src = length(src_cols)
    n_src > 0 || return nothing

    if nc == 1
        # Degree-0 fast path: exact volume-weighted average. Bit-identical
        # to the legacy implementation; the higher-order machinery is not
        # set up to keep this path branch-free and allocation-free.
        @inbounds for name in Names
            M = getfield(snap, name)
            T = eltype(M)
            s = zero(T)
            for c in src_cols
                s += M[1, c]
            end
            avg = s / T(n_src)
            _set_poly_coeff!(pfs, Val(name), new_i, 1, avg)
        end
        return nothing
    end

    # Degree ≥ 1: try the L²-projection path (exact recovery of all
    # moments up to degree P). MonomialBasis takes the path directly;
    # BernsteinBasis brackets the projection with `change_basis` from/to
    # monomial form (the conversion matrices are cached). Falls back to
    # constant-moment only, with a warning, for any other basis.
    if pfs.basis isa MonomialBasis
        _write_coeffs_l2_projection!(pfs, snap, new_i, src_cols, parent_cell)
        return nothing
    elseif pfs.basis isa BernsteinBasis
        _write_coeffs_l2_projection_bernstein!(pfs, snap, new_i, src_cols, parent_cell)
        return nothing
    end

    # Fallback: constant-moment only, with a once-per-AdaptiveField warning.
    # Reached by future non-monomial / non-Bernstein bases at degree ≥ 1.
    if !basis_degree_warned[]
        @warn "AdaptiveField: coarsening a degree>=1 field with " *
              "$(typeof(pfs.basis).name.name) uses the constant-moment-only " *
              "fallback (higher coefficients zeroed). Convert to MonomialBasis " *
              "via `change_basis` for the exact L²-projection path."
        basis_degree_warned[] = true
    end
    @inbounds for name in Names
        M = getfield(snap, name)
        T = eltype(M)
        s = zero(T)
        for c in src_cols
            s += M[1, c]
        end
        avg = s / T(n_src)
        _set_poly_coeff!(pfs, Val(name), new_i, 1, avg)
        for k in 2:nc
            _set_poly_coeff!(pfs, Val(name), new_i, k, zero(T))
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# L²-projection coarsening (degree ≥ 1, MonomialBasis)
# ----------------------------------------------------------------------------

# Compute the per-axis sub-box [a_d, b_d] ⊂ [0, 1] for child `child_sib`
# of a cell with `split_mask`. Mirrors `cell_unit_box` for one refinement
# step but parameterized by the (sibling_index, split_mask) pair directly,
# so we can recover the OLD children's geometry from the post-coarsen
# parent metadata alone.
function _child_unit_subbox(child_sib::Integer,
                             split_mask::Unsigned, ::Val{D}) where {D}
    # Walk axes serially, accumulating the (lo, hi) tuple. We can't use
    # `ntuple` cleanly here because the lookup into `child_sib`/`split_mask`
    # is stateful (we advance `bit_pos` only on split axes), so we build
    # arrays first and convert to tuples at the end.
    a = Vector{Float64}(undef, D)
    b = Vector{Float64}(undef, D)
    bit_pos = 0
    @inbounds for axis in 1:D
        axis_bit = oneunit(split_mask) << (axis - 1)
        if (split_mask & axis_bit) != 0
            lower_half = ((child_sib >> bit_pos) & 1) == 0
            if lower_half
                a[axis] = 0.0
                b[axis] = 0.5
            else
                a[axis] = 0.5
                b[axis] = 1.0
            end
            bit_pos += 1
        else
            a[axis] = 0.0
            b[axis] = 1.0
        end
    end
    return (NTuple{D, Float64}(a), NTuple{D, Float64}(b))
end

# Per-axis cross integration coefficient
#
#     A[i+1, k+1] = Σ_{j=0..i} C(i, j) · a^{i-j} · s^{j+1} / (j + k + 1)
#
# where `[a, a+s] ⊂ [0, 1]` is the child's sub-box on this axis. This is the
# 1-D cross integral
#
#     A[i+1, k+1] = ∫_{a}^{a+s} ξ^i · ((ξ - a) / s)^k dξ
#
# By the per-axis tensor-product structure, the multi-D cross mass matrix
# satisfies M_cross[α, β] = ∏_d A_d[α_d + 1, β_d + 1].
function _per_axis_cross_coefs(a::T, s::T, P::Int) where T
    A = Matrix{T}(undef, P + 1, P + 1)
    @inbounds for i in 0:P, k in 0:P
        acc = zero(T)
        spow = s
        api = a^i  # a^{i-j} starts at a^i (j=0); divide by a after — but a may be 0.
        # Compute Σ_{j=0..i} C(i,j) a^{i-j} s^{j+1} / (j+k+1) directly with stable
        # incremental updates would over-engineer this; the inner loop is tiny
        # (i ≤ P, P typically ≤ 3), so use the straightforward form.
        for j in 0:i
            cij = T(binomial(i, j))
            apow = a^(i - j)
            sjp1 = s^(j + 1)
            acc += cij * apow * sjp1 / T(j + k + 1)
        end
        A[i + 1, k + 1] = acc
    end
    return A
end

# Build the cross mass matrix M_cross[α, β] for one child sub-box, in
# graded-lex order matching `MonomialBasis{D, P}` coefficient layout.
function _cross_mass_matrix(a::NTuple{D, T}, b::NTuple{D, T},
                              P::Int) where {D, T}
    s = ntuple(d -> b[d] - a[d], Val(D))
    # Per-axis cross coefficients.
    A_axes = ntuple(d -> _per_axis_cross_coefs(a[d], s[d], P), Val(D))
    multi = moment_multiindices(D, P)
    n = length(multi)
    M = Matrix{T}(undef, n, n)
    @inbounds for αi in 1:n
        α = multi[αi]
        for βi in 1:n
            β = multi[βi]
            v = one(T)
            for d in 1:D
                v *= A_axes[d][α[d] + 1, β[d] + 1]
            end
            M[αi, βi] = v
        end
    end
    return M
end

# Parent reference mass matrix on the unit cube [0, 1]^D, in graded-lex order.
function _parent_reference_mass(::Type{T}, ::Val{D}, P::Int) where {T, D}
    multi = moment_multiindices(D, P)
    n = length(multi)
    M = Matrix{T}(undef, n, n)
    @inbounds for i in 1:n, j in i:n
        v = one(T)
        for d in 1:D
            v *= one(T) / T(multi[i][d] + multi[j][d] + 1)
        end
        M[i, j] = v
        M[j, i] = v
    end
    return M
end

# Core L²-projection step: given a snapshot column-matrix `Mfield` of children
# coefficients in monomial form, the children's source columns `src_cols`,
# and the parent split mask, return the parent's monomial coefficients.
# Falls back to constant-moment-only with a warning if the mass-matrix
# solve fails (singular, extremely ill-conditioned, etc.).
function _project_monomial_children_to_parent(
    Mfield::AbstractMatrix{T}, src_cols, parent_mask::Unsigned,
    ::Val{D}, P::Int, name::Symbol) where {T, D}
    nc = moments_length(D, P)
    n_children = length(src_cols)

    Mp = _parent_reference_mass(T, Val(D), P)
    rhs = zeros(T, nc)

    # Iterate children in DFS order; sibling indices are 0..n-1 (refine_cells!
    # emits children contiguously with the same split_mask as the parent).
    ci_offset = 0
    for c in src_cols
        child_sib = ci_offset
        a, b = _child_unit_subbox(child_sib, parent_mask, Val(D))
        a_T = ntuple(d -> T(a[d]), Val(D))
        b_T = ntuple(d -> T(b[d]), Val(D))
        Mcross = _cross_mass_matrix(a_T, b_T, P)
        for αi in 1:nc
            acc = zero(T)
            for βi in 1:nc
                acc += Mcross[αi, βi] * Mfield[βi, c]
            end
            rhs[αi] += acc
        end
        ci_offset += 1
    end

    return try
        Mp \ rhs
    catch err
        # Graceful degradation: constant moment only.
        s_const = zero(T)
        for col in src_cols
            s_const += Mfield[1, col]
        end
        avg = s_const / T(n_children)
        @warn "AdaptiveField: mass-matrix solve failed during coarsening " *
              "(field $name); falling back to constant-moment average." err
        fb = zeros(T, nc)
        fb[1] = avg
        fb
    end
end

# L²-projection coarsening for one cell. Performs the per-event mass-matrix
# solve and writes the result into element `new_i` of `pfs`. Falls back to
# constant-moment averaging if the mass-matrix solve fails (e.g. singular,
# extremely ill-conditioned). For typical degrees P ≤ 3 the parent mass
# matrix on the unit cube has condition number O(10^{2P}), which is fine.
function _write_coeffs_l2_projection!(
    pfs::PolynomialFieldSet{L, B, Names, ST, S},
    snap, new_i::Int, src_cols, parent_cell) where {L, B, Names, ST, S}
    basis = pfs.basis::MonomialBasis
    D, P = _basis_dim_degree(basis)
    nc = n_coeffs(basis)
    @assert nc == moments_length(D, P) "MonomialBasis n_coeffs mismatch"
    parent_mask = parent_cell.split_mask

    @inbounds for name in Names
        Mfield = getfield(snap, name)
        coeffs = _project_monomial_children_to_parent(
            Mfield, src_cols, parent_mask, Val(D), P, name)
        for k in 1:nc
            _set_poly_coeff!(pfs, Val(name), new_i, k, coeffs[k])
        end
    end
    return nothing
end

# L²-projection coarsening for `BernsteinBasis` cells. Brackets the
# `_project_monomial_children_to_parent` step with `change_basis` calls:
# children's Bernstein coefficients → monomial → projection → parent's
# Bernstein. The `change_basis` matrices are cached per `(D, P)`.
function _write_coeffs_l2_projection_bernstein!(
    pfs::PolynomialFieldSet{L, B, Names, ST, S},
    snap, new_i::Int, src_cols, parent_cell) where {L, B, Names, ST, S}
    basis = pfs.basis::BernsteinBasis
    D, P = _basis_dim_degree(basis)
    nc = n_coeffs(basis)
    parent_mask = parent_cell.split_mask
    mono_basis = MonomialBasis{D, P}()

    @inbounds for name in Names
        Mbern = getfield(snap, name)
        T = eltype(Mbern)

        # Convert each child's Bernstein coefficients to monomial form.
        # Build a per-field column-matrix in monomial space using the same
        # column indices as the original snapshot — this lets us reuse
        # `_project_monomial_children_to_parent` unchanged.
        n_old_cols = size(Mbern, 2)
        Mmono = Matrix{T}(undef, nc, n_old_cols)
        for c in src_cols
            bern_vec = @view Mbern[:, c]
            mono_vec = change_basis(mono_basis, basis, bern_vec)
            for k in 1:nc
                Mmono[k, c] = T(mono_vec[k])
            end
        end

        mono_parent = _project_monomial_children_to_parent(
            Mmono, src_cols, parent_mask, Val(D), P, name)
        bern_parent = change_basis(basis, mono_basis, mono_parent)

        for k in 1:nc
            _set_poly_coeff!(pfs, Val(name), new_i, k, T(bern_parent[k]))
        end
    end
    return nothing
end

# Tiny helper: pull (D, P) out of a MonomialBasis or BernsteinBasis type.
@inline _basis_dim_degree(::MonomialBasis{D, P}) where {D, P} = (D, P)
@inline _basis_dim_degree(::BernsteinBasis{D, P}) where {D, P} = (D, P)

# ----------------------------------------------------------------------------
# Event handler
# ----------------------------------------------------------------------------

# Build an inverse index_remap: inv[new_i] = old_i (or 0 if no preimage,
# i.e. `new_i` is a freshly created child slot).
function _build_inverse_remap(remap::Vector{UInt32}, new_n::Int)
    inv = zeros(UInt32, new_n)
    @inbounds for old_i in eachindex(remap)
        ni = remap[old_i]
        if ni != 0
            inv[Int(ni)] = UInt32(old_i)
        end
    end
    return inv
end

function _handle_event!(af::AdaptiveField, event::RefinementEvent)
    af.disposed && return  # defensive; shouldn't fire if unregistered

    pfs = af.field
    n_old = pfs.n
    new_n = Int(n_cells(af.mesh))  # mesh has already been rebuilt

    # Sanity: the listener fires after invalidate_caches!. n_cells(mesh) is
    # the new size. Nothing to do if the size and topology didn't change
    # (shouldn't normally happen, but be defensive).
    if isempty(event.refined_parents) && isempty(event.coarsened_parents)
        return
    end

    # Snapshot all OLD coefficients before we touch the storage. Cheap for
    # typical adaptive workloads (n_old is the cell count); avoids subtle
    # aliasing bugs when new_n != old_n.
    snap = _snapshot_coefficients(pfs, n_old)

    # Resize storage in place.
    L = _layout_type_poly(pfs)
    nc = n_coeffs(pfs.basis)
    _resize_poly_storage_inplace!(pfs.storage, L, nc, new_n)

    # Rebuild the immutable wrapper with the new n.
    pfs_new = _rebuild_with_n(pfs, new_n)
    af.field = pfs_new

    # Track which new slots have been written, to assert no slot is left
    # uninitialized at the end. (Cheap relative to the per-cell coefficient
    # writes already happening.)
    written = falses(new_n)

    # Phase 1: copy unchanged + refined-parent slots.
    # `index_remap[i] != 0` means cell i has a new home. Note that for a
    # refined parent, `index_remap[parent_old]` points at the new (now
    # non-leaf) parent slot — copy its old coefficients there too, so the
    # parent retains its data.
    remap = event.index_remap
    @inbounds for old_i in 1:n_old
        ni = remap[old_i]
        if ni != 0
            _write_coeffs_from_column!(pfs_new, snap, Int(ni), Int(old_i))
            written[Int(ni)] = true
        end
    end

    # Phase 2: refinement — constant prolongation. For each refined parent
    # k, copy its OLD coefficients to every new child slot in
    # event.new_children[k].
    @inbounds for k in eachindex(event.refined_parents)
        parent_old = Int(event.refined_parents[k])
        child_range = event.new_children[k]
        for new_child in child_range
            _write_coeffs_from_column!(pfs_new, snap, Int(new_child), parent_old)
            written[Int(new_child)] = true
        end
    end

    # Phase 3: coarsening — average. For each coarsened parent in NEW
    # indices, find its OLD parent index (via inverse remap) and the OLD
    # children indices (contiguous block immediately after the OLD parent).
    if !isempty(event.coarsened_parents)
        inv = _build_inverse_remap(remap, new_n)
        warned = Ref(false)
        @inbounds for parent_new in event.coarsened_parents
            pn = Int(parent_new)
            parent_old = Int(inv[pn])
            # OLD parent's split_mask was preserved on the NEW cell (the
            # coarsen path keeps split_mask; only the FLAG_LEAF bit changes).
            parent_cell = af.mesh.cells[pn]
            n_children = children_count(parent_cell)
            # OLD children are immediately after the OLD parent in DFS.
            src_cols = (parent_old + 1):(parent_old + n_children)
            _write_coeffs_average!(pfs_new, snap, pn, src_cols,
                                   parent_cell, warned)
            # The parent slot itself was written in Phase 1 from the OLD
            # parent's (non-leaf) coefficients; overwrite that with the
            # averaged children's coefficients.
            written[pn] = true
        end
    end

    # Defensive sanity check. If this ever fires, either the event is
    # malformed or our handling missed a case. Done in @debug mode by
    # construction — leave the assertion in for now since correctness here
    # is paramount and the cost is one allocation of a BitVector.
    if !all(written)
        unwritten = findall(!, written)
        error("AdaptiveField: $(length(unwritten)) new cell slot(s) left " *
              "unwritten after handling event (e.g. slot $(first(unwritten))). " *
              "This is a bug in AdaptiveField or RefinementEvent.")
    end

    return
end

# ----------------------------------------------------------------------------
# Forwarding for ergonomics
# ----------------------------------------------------------------------------

# Forward common queries to the underlying PolynomialFieldSet.
Storage.n_elements(af::AdaptiveField) = n_elements(af.field)
Storage.field_names(af::AdaptiveField) = field_names(af.field)
Storage.basis_of(af::AdaptiveField) = basis_of(af.field)
Storage.n_coeffs_per_element(af::AdaptiveField) = n_coeffs_per_element(af.field)
