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

# Coarsening (volume-weighted average)

For degree-0 fields (basis with `n_coeffs == 1`), the parent receives
the volume-weighted average of children's coefficients. Because children
of any (an)isotropic split have equal volume by construction, this
reduces to the arithmetic mean — exact and conservative.

For degree ≥ 1 fields, only the constant moment is averaged; higher-
order coefficients are zeroed and a single warning is emitted via
`@warn_once`. Full L²-projection coarsening (with cross mass matrices
between fine and coarse reference frames) is deferred to a follow-up
PR — see TODO marker below.

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
function _write_coeffs_average!(
    pfs::PolynomialFieldSet{L, B, Names, ST, S},
    snap, new_i::Int, src_cols, basis_degree_warned::Ref{Bool}) where {L, B, Names, ST, S}
    nc = n_coeffs(pfs.basis)
    n_src = length(src_cols)
    n_src > 0 || return nothing

    @inbounds for name in Names
        M = getfield(snap, name)
        T = eltype(M)
        if nc == 1
            # Degree-0: exact volume-weighted average (children have equal
            # volume from a single split level).
            s = zero(T)
            for c in src_cols
                s += M[1, c]
            end
            avg = s / T(n_src)
            _set_poly_coeff!(pfs, Val(name), new_i, 1, avg)
        else
            # Degree >= 1: only the constant moment is averaged; higher
            # coefficients are zeroed. Issue one warning per AdaptiveField
            # lifetime (across all fields), since the user only needs to be
            # told once that they're using a deferred path.
            if !basis_degree_warned[]
                @warn "AdaptiveField: coarsening a degree>=1 field uses " *
                      "constant-moment-only fallback (higher coefficients " *
                      "zeroed). Full L^2-projection coarsening with cross " *
                      "mass matrices is deferred to a follow-up PR."
                basis_degree_warned[] = true
            end
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
    end
    return nothing
end

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
            _write_coeffs_average!(pfs_new, snap, pn, src_cols, warned)
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
