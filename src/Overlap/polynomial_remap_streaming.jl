"""
Streaming (single-pass) polynomial remap from a Lagrangian simplicial
mesh to a **uniformly refined** Eulerian hierarchical mesh, using
`R3D.Flat.voxelize_fold!` to fuse the per-leaf RHS accumulation into
the polytope-vs-grid recursion.

# Why streaming?

The two-phase pipeline (`compute_overlap` + `polynomial_remap_field!`)
materializes a full `GeometricOverlap{D, T}` containing one
`OverlapEntry` per non-empty (simplex, leaf) pair, with a
`Vector{T}` of polynomial moments per entry. That's the right design
when one set of geometry feeds many field remaps — the geometric work
is amortized.

But when the geometry changes every timestep (the dfmm common case)
**and** only a handful of fields are remapped per geometry, the
per-entry allocation is pure overhead. The streaming variant skips it:
for each Lagrangian simplex, `voxelize_fold!` walks the polytope
recursion and at each non-empty leaf calls a callback that contracts
the per-leaf moment vector with the source polynomial's pulled-back
coefficients and accumulates into the destination RHS column for that
leaf. **No `OverlapEntry` allocation.**

The trade-offs:
- ✅ Zero per-entry allocations; the only matrix allocations are the
  per-cell pullback matrices and the destination RHS, which are always
  needed.
- ✅ Memory bandwidth proportional to `n_dst` (RHS columns) not
  `n_entries × n_phys` (full overlap moments).
- ⚠️ Requires the Eulerian mesh to be **uniformly refined**: every leaf
  at the same depth, every refinement fully isotropic.
- ⚠️ One pass per direction (L→E or E→L) per field. If you remap many
  fields, the per-simplex polytope traversal is repeated; the two-phase
  variant avoids that.

For dfmm Tier C / D:
- Use the **streaming** variant inside the per-step transport sweep.
- Use the **two-phase** variant for diagnostics, conservation checks, or
  when running many remaps through a fixed geometry.

# Math (same as `polynomial_remap.jl`, just streamed)

Per-leaf contribution to destination RHS column `j`:

    b_j[α] += Σ_γ U_j[α, γ] · (Σ_δ P_i[δ] · m[γ + δ])

where:
- `P_i = src_pullback_i^T · src_coeffs[:, i]` is the source polynomial in
  physical coordinates (precomputed per simplex, before the fold).
- `m` is the per-leaf moment vector `voxelize_fold!` provides (in
  physical-frame, graded-lex order, length `moments_length(D, P_src + P_dst)`).
- `U_j` is the destination cell's reference-to-physical pullback matrix.
- `lookup[γ, δ]` indexes `m` for the multi-index `γ + δ` (precomputed once).

After processing all simplices, solve `M_j d_j = b_j` per leaf to recover
the destination polynomial coefficients in the destination's reference
frame.

# Naming convention for direction selection

The polynomial-remap surface chose three different direction-selection
conventions across three layers; this is documented here so it doesn't
look like an oversight:

- **Low level** (`polynomial_remap.jl`, raw matrices): one function per
  direction — `polynomial_remap_l_to_e!` and `polynomial_remap_e_to_l!`.
  Direction is in the function name.
- **Two-phase field-set** (`polynomial_remap_fieldset.jl`, takes a
  `GeometricOverlap`): one function with a `direction = :l_to_e | :e_to_l`
  kwarg — `polynomial_remap_field!`. Direction is a runtime value.
- **Streaming field-set** (this file, takes a frame and a uniform-grid
  info): one function per direction — `polynomial_remap_l_to_uniform_e!`
  and `polynomial_remap_uniform_e_to_l!`. Direction is in the name; the
  word "uniform" tags the Eulerian side as the uniform-grid side.

The streaming variant uses separate functions because the two directions
have asymmetric internal state (different shapes for `Pi` precomputation,
different RHS access patterns) and dispatching on a runtime `Symbol`
would force a runtime branch through the inner loop. The two-phase
variant uses a kwarg because both directions traverse the same
`GeometricOverlap` data structure with only the `lag_idx`/`eul_idx`
roles swapped — a single implementation with one branch is natural.

# Status

Implemented and tested for D = 2 and D = 3. The framework's polynomial
remap consumes moments up to order `P_src + P_dst`, which r3djl currently
provides only at D = 2 and D = 3 (its D ≥ 4 `moments!` is order = 0
only). The previous D ≥ 4 sequential-clip-on-simplex bug was fixed
upstream (r3djl `c4496ab`), so the volume-only path is unblocked; the
remaining blocker for plumbing this module through D ≥ 4 is the
higher-order moments.

Status note in `src/Overlap/lifting.jl`.
"""

# ============================================================================
# Uniform-refinement detection
# ============================================================================

"""
    uniform_grid_dimensions(frame::EulerianFrame{D, T}) -> Union{Nothing, NamedTuple}

Returns a NamedTuple `(ibox_lo, ibox_hi, d, depth, leaf_index_map)`
describing the Eulerian frame as a uniform Cartesian grid, OR `nothing`
if the mesh is not uniformly refined.

The mesh is "uniformly refined" when:
1. Every leaf is at the same depth `L` (i.e. `level_of(mesh, ci) == L`
   for all leaves).
2. Every non-leaf cell uses fully-isotropic refinement
   (`split_mask == FULLY_ISOTROPIC_MASK`).

Returned fields:
- `ibox_lo, ibox_hi::NTuple{D, Int}` — `(0, ..., 0)` and `(N, ..., N)`
  with `N = 2^L`. Pass these directly to `voxelize_fold!`.
- `d::NTuple{D, T}` — physical cell spacing.
- `depth::Int` — the uniform refinement depth `L`.
- `leaf_index_map::Array{Int, D}` — `D`-dimensional array of size
  `(N, ..., N)`, where `leaf_index_map[i, j[, k]]` is the
  `HierarchicalMesh` cell index of the leaf at voxel `(i, j[, k])`
  (1-based). Use this to map `voxelize_fold!`'s `(i, j[, k])` callback
  arguments to destination-array column indices.

This check is `O(n_cells)` and should be done once per geometry change.
"""
function uniform_grid_dimensions(frame::EulerianFrame{D, T}) where {D, T}
    mesh = frame.mesh

    # Check 1: every non-leaf cell uses fully-isotropic refinement
    fully_iso = isotropic_mask(D)
    @inbounds for ci in 1:n_cells(mesh)
        cell = mesh.cells[ci]
        if !is_leaf(cell)
            children = find_children(mesh, ci)
            isempty(children) && continue
            # Check the first child's split_mask (siblings share it).
            # Compare numerically since split_mask may be a smaller uint type.
            first_child = mesh.cells[children[1]]
            if UInt32(first_child.split_mask) != fully_iso
                return nothing
            end
        end
    end

    # Check 2: every leaf at the same depth
    leaves = enumerate_leaves(mesh)
    isempty(leaves) && return nothing
    depth = level_of(mesh, leaves[1])
    @inbounds for ci in leaves
        if level_of(mesh, ci) != depth
            return nothing
        end
    end

    N = 1 << Int(depth)   # 2^depth
    ibox_lo = ntuple(_ -> 0, Val(D))
    ibox_hi = ntuple(_ -> N, Val(D))

    extent = ntuple(d -> frame.hi[d] - frame.lo[d], Val(D))
    cell_size = ntuple(d -> extent[d] / N, Val(D))

    # Build the leaf-index map: voxel (i, j[, k]) → HierarchicalMesh cell idx.
    # We get this by walking each leaf, computing its unit-box position,
    # and mapping to integer voxel coordinates.
    leaf_index_map = Array{Int, D}(undef, ntuple(_ -> N, Val(D))...)
    fill!(leaf_index_map, 0)
    @inbounds for ci in leaves
        u_lo, u_hi = cell_unit_box(mesh, ci)
        # Voxel (i, j[, k]) in 1-based indexing has u_lo[d] = (i-1)/N
        idx = ntuple(d -> round(Int, u_lo[d] * N) + 1, Val(D))
        leaf_index_map[idx...] = Int(ci)
    end
    # Sanity: every entry should be filled
    if any(==(0), leaf_index_map)
        return nothing
    end

    return (ibox_lo = ibox_lo,
            ibox_hi = ibox_hi,
            d = cell_size,
            depth = Int(depth),
            leaf_index_map = leaf_index_map)
end

# ============================================================================
# Streaming RHS-accumulation state
# ============================================================================

"""
    StreamingRemapState{D, T, NDST}

Mutable state passed through `voxelize_fold!`'s callback. Holds:

- `rhs::Matrix{T}` — destination RHS, shape `(n_dst, n_dst_cells)`,
  accumulated across all source simplices.
- `dst_pullbacks::Vector{Matrix{T}}` — per-destination-cell reference-
  to-physical pullback matrices, indexed by destination cell idx
  (`HierarchicalMesh` cell index, 1-based).
- `lookup::Matrix{Int}` — `(n_dst, n_src)` table of `lookup[γ, δ]` =
  flat physical-moment index for the multi-index `γ + δ`.
- `Pi::Vector{T}` — current source simplex's physical-frame coefficients
  `P_i = src_pullback_i^T · src_coeffs[:, i]`. Filled by the outer loop
  before each `voxelize_fold!` call.
- `leaf_index_map::Array{Int, D}` — voxel-coords → destination cell idx.
- `n_src::Int`, `n_dst::Int` — coefficient counts (cached).
- `frame_lo::NTuple{D, T}` — destination frame's lower bound. Voxelize_fold!
  expects polytope coordinates relative to the spatial origin (i.e. the
  cell at index `(i, j)` covers `[(i-1)*d, i*d]`); we shift the polytope
  to the frame's local origin before feeding it in.
"""
mutable struct StreamingRemapState{D, T}
    rhs::Matrix{T}
    dst_pullbacks::Vector{Matrix{T}}
    lookup::Matrix{Int}
    Pi::Vector{T}
    leaf_index_map::Array{Int, D}
    n_src::Int
    n_dst::Int
    frame_lo::NTuple{D, T}
end

"""
    StreamingRemapStateEtoL{D, T}

Mutable state for the Eulerian → Lagrangian streaming variant. Different
from `StreamingRemapState` because the destination is **fixed during one
voxelize sweep** (one Lagrangian simplex), while the source changes per
leaf.

Fields:
- `rhs::Matrix{T}` — destination RHS, shape `(n_dst, n_dst_cells)`
- `dst_simplex_idx::Int` — current outer-loop simplex index. Set by the
  caller before each `voxelize_fold!` sweep so the per-leaf callback
  knows which RHS column to write into.
- `dst_pullback_current::Matrix{T}` — pullback for the current outer
  simplex; refreshed by the caller per sweep (avoids per-callback
  index lookup).
- `Pi_per_leaf::Matrix{T}` — precomputed source physical-frame
  coefficients for **every Eulerian leaf**, shape `(n_src, n_eul_cells)`.
  Computed once before the outer loop. Per-leaf callback just looks up
  the column.
- `lookup::Matrix{Int}` — `(n_dst, n_src)` `(γ, δ)` → flat phys-mom index.
- `leaf_index_map::Array{Int, D}` — voxel-coords → Eulerian cell idx.
- `n_src::Int`, `n_dst::Int` — coefficient counts.
- `frame_lo::NTuple{D, T}` — Eulerian frame's lower bound.
"""
mutable struct StreamingRemapStateEtoL{D, T}
    rhs::Matrix{T}
    dst_simplex_idx::Int
    dst_pullback_current::Matrix{T}
    Pi_per_leaf::Matrix{T}
    lookup::Matrix{Int}
    leaf_index_map::Array{Int, D}
    n_src::Int
    n_dst::Int
    frame_lo::NTuple{D, T}
end

# ============================================================================
# Per-leaf callback (D=2 and D=3 dispatched explicitly)
# ============================================================================

# The callback contract: state, leaf indices, moment vector → state.
# For voxelize_fold!, signatures are (i, j, m) for D=2 and (i, j, k, m) for D=3.

@inline function _streaming_leaf_callback_2d(state::StreamingRemapState{2, T},
                                               i::Int, j::Int,
                                               m::AbstractVector{T}) where T
    leaf_idx = @inbounds state.leaf_index_map[i, j]
    @inbounds U = state.dst_pullbacks[leaf_idx]
    Pi = state.Pi
    lookup = state.lookup
    n_dst = state.n_dst
    n_src = state.n_src
    @inbounds for α in 1:n_dst
        acc = zero(T)
        for γ in 1:n_dst
            Uαγ = U[α, γ]
            Uαγ == zero(T) && continue
            inner = zero(T)
            for δ in 1:n_src
                inner += Pi[δ] * m[lookup[γ, δ]]
            end
            acc += Uαγ * inner
        end
        state.rhs[α, leaf_idx] += acc
    end
    return state
end

@inline function _streaming_leaf_callback_3d(state::StreamingRemapState{3, T},
                                               i::Int, j::Int, k::Int,
                                               m::AbstractVector{T}) where T
    leaf_idx = @inbounds state.leaf_index_map[i, j, k]
    @inbounds U = state.dst_pullbacks[leaf_idx]
    Pi = state.Pi
    lookup = state.lookup
    n_dst = state.n_dst
    n_src = state.n_src
    @inbounds for α in 1:n_dst
        acc = zero(T)
        for γ in 1:n_dst
            Uαγ = U[α, γ]
            Uαγ == zero(T) && continue
            inner = zero(T)
            for δ in 1:n_src
                inner += Pi[δ] * m[lookup[γ, δ]]
            end
            acc += Uαγ * inner
        end
        state.rhs[α, leaf_idx] += acc
    end
    return state
end

# E→L callbacks: source is per-leaf, destination is fixed (state.dst_simplex_idx).
# Cache-friendly: writes to state.rhs[:, dst_simplex_idx] only across the sweep.
# Pi is precomputed per leaf in state.Pi_per_leaf so the callback just looks it up.

@inline function _streaming_leaf_callback_etol_2d(state::StreamingRemapStateEtoL{2, T},
                                                    i::Int, j::Int,
                                                    m::AbstractVector{T}) where T
    leaf_idx = @inbounds state.leaf_index_map[i, j]
    j_dst = state.dst_simplex_idx
    U = state.dst_pullback_current
    Pi_col = view(state.Pi_per_leaf, :, leaf_idx)
    lookup = state.lookup
    n_dst = state.n_dst
    n_src = state.n_src
    @inbounds for α in 1:n_dst
        acc = zero(T)
        for γ in 1:n_dst
            Uαγ = U[α, γ]
            Uαγ == zero(T) && continue
            inner = zero(T)
            for δ in 1:n_src
                inner += Pi_col[δ] * m[lookup[γ, δ]]
            end
            acc += Uαγ * inner
        end
        state.rhs[α, j_dst] += acc
    end
    return state
end

@inline function _streaming_leaf_callback_etol_3d(state::StreamingRemapStateEtoL{3, T},
                                                    i::Int, j::Int, k::Int,
                                                    m::AbstractVector{T}) where T
    leaf_idx = @inbounds state.leaf_index_map[i, j, k]
    j_dst = state.dst_simplex_idx
    U = state.dst_pullback_current
    Pi_col = view(state.Pi_per_leaf, :, leaf_idx)
    lookup = state.lookup
    n_dst = state.n_dst
    n_src = state.n_src
    @inbounds for α in 1:n_dst
        acc = zero(T)
        for γ in 1:n_dst
            Uαγ = U[α, γ]
            Uαγ == zero(T) && continue
            inner = zero(T)
            for δ in 1:n_src
                inner += Pi_col[δ] * m[lookup[γ, δ]]
            end
            acc += Uαγ * inner
        end
        state.rhs[α, j_dst] += acc
    end
    return state
end

# ============================================================================
# Public streaming entry point
# ============================================================================

"""
    polynomial_remap_l_to_uniform_e!(target_pfs, target_fieldname,
                                       source_pfs, source_fieldname,
                                       lag, frame, src_frames, dst_frames,
                                       grid_info = nothing;
                                       workspace = nothing,
                                       src_pullbacks = nothing,
                                       dst_pullbacks = nothing)

Single-pass polynomial remap from a Lagrangian simplicial mesh to a
**uniformly-refined** Eulerian frame, using `R3D.Flat.voxelize_fold!`
for fused per-leaf accumulation. Skips the `GeometricOverlap`
materialization entirely — for one-field-per-geometry use this is
typically much faster than the two-phase pipeline.

# Arguments

- `target_pfs`, `target_fieldname` — destination field set / field name.
  Must use `MonomialBasis{D, P_dst}`. The target's coefficients are
  **overwritten** (not accumulated into).
- `source_pfs`, `source_fieldname` — source field set / field name.
  Must use `MonomialBasis{D, P_src}`. `D` and the polynomial orders
  may differ between source and target.
- `lag` — Lagrangian simplicial mesh (current configuration).
- `frame` — Eulerian frame; **must** be uniformly refined (same depth,
  fully-isotropic refinement everywhere). Use `uniform_grid_dimensions`
  to validate up front.
- `src_frames`, `dst_frames` — per-cell reference frames. As with the
  two-phase variant, build once and reuse across multiple field remaps
  done through the same geometry. `dst_frames` must be indexed by
  Eulerian `cell_idx` (only entries for actual leaves are used; the
  others may be placeholders).
- `grid_info` — the NamedTuple returned by `uniform_grid_dimensions(frame)`.
  Pass `nothing` to have it computed inside; pre-computing once is
  cheaper if you're calling this in a tight loop.

# Keyword arguments (all optional)

- `workspace::R3D.Flat.VoxelizeWorkspace` — pre-allocated workspace
  reused across all simplices. Pass to amortize allocation across many
  calls; one workspace is safe across one remap call.
- `src_pullbacks::Vector{Matrix{T}}` — precomputed reference-to-physical
  pullbacks for source frames (length matches `src_frames`). Computing
  these takes substantial time and allocation; passing them in saves
  ~80% of wall time and ~95% of allocations on dfmm-typical sizes.
- `dst_pullbacks::Vector{Matrix{T}}` — same, indexed by Eulerian
  `cell_idx`. Non-leaf cells need a placeholder matrix (any shape-
  compatible matrix is fine, it isn't read).

# Returns

`target_pfs` after writing the L²-projected coefficients in place.

# Performance notes

For dfmm-typical sizes (2048 simplices, 256-leaf grid, P=3) on a 2-thread
machine: ~5 ms with precomputed pullbacks, ~35 ms without. Compare to
~24 ms for the equivalent two-phase pipeline (compute_overlap + remap).

# Errors

Throws `ArgumentError` if:
- The Eulerian frame isn't uniformly refined.
- Source/target bases aren't `MonomialBasis`.
- Source/target dimensions don't match the frame.
- Source/target field-set element counts don't match the meshes.
- A precomputed pullback array has the wrong length.

Throws for `D ∉ (2, 3)`. r3djl's `voxelize_fold!` does support D ≥ 4
but only at moment order = 0; polynomial remap consumes orders up to
P_src + P_dst, so D ≥ 4 is blocked until r3djl higher-order moments at
D ≥ 4 land.
"""
function polynomial_remap_l_to_uniform_e!(target_pfs::PolynomialFieldSet,
                                            target_fieldname::Symbol,
                                            source_pfs::PolynomialFieldSet,
                                            source_fieldname::Symbol,
                                            lag::SimplicialMesh{D, T},
                                            frame::EulerianFrame{D, T},
                                            src_frames::Vector{<:CellReferenceFrame{D, T}},
                                            dst_frames::Vector{<:CellReferenceFrame{D, T}},
                                            grid_info = nothing;
                                            workspace::Union{Nothing, R3D.Flat.VoxelizeWorkspace{D, T}} = nothing,
                                            src_pullbacks::Union{Nothing, Vector{Matrix{T}}} = nothing,
                                            dst_pullbacks::Union{Nothing, Vector{Matrix{T}}} = nothing
                                            ) where {D, T}
    D in (2, 3) || throw(ArgumentError(
        "polynomial_remap_l_to_uniform_e! supports D=2 and D=3 (got D=$D). " *
        "r3djl now supports voxelize_fold! at D≥4 but only at moment order=0; " *
        "polynomial remap consumes orders up to P_src+P_dst, so D≥4 is blocked " *
        "until r3djl higher-order moments land at D≥4."))

    # Validate bases
    src_basis = basis_of(source_pfs)
    dst_basis = basis_of(target_pfs)
    src_basis isa MonomialBasis ||
        throw(ArgumentError("source field set must use MonomialBasis (got $(typeof(src_basis)))"))
    dst_basis isa MonomialBasis ||
        throw(ArgumentError("target field set must use MonomialBasis (got $(typeof(dst_basis)))"))
    P_src = _basis_order(src_basis)
    P_dst = _basis_order(dst_basis)
    _basis_dim(src_basis) == D ||
        throw(ArgumentError("source basis dimension ≠ frame dimension"))
    _basis_dim(dst_basis) == D ||
        throw(ArgumentError("target basis dimension ≠ frame dimension"))

    # Get or compute uniform-grid info
    info = grid_info === nothing ? uniform_grid_dimensions(frame) : grid_info
    info === nothing &&
        throw(ArgumentError("Eulerian frame is not uniformly refined; use the " *
                             "two-phase polynomial_remap_field! instead."))

    n_simplices(lag) == length(src_frames) ||
        throw(ArgumentError("src_frames length ≠ n_simplices(lag)"))
    n_cells(frame.mesh) == length(dst_frames) ||
        throw(ArgumentError("dst_frames length ≠ n_cells(frame.mesh)"))
    n_elements(source_pfs) == n_simplices(lag) ||
        throw(ArgumentError("source field set element count ≠ n_simplices(lag)"))
    n_elements(target_pfs) == n_cells(frame.mesh) ||
        throw(ArgumentError("target field set element count ≠ n_cells(frame.mesh)"))

    n_src = moments_length(D, P_src)
    n_dst = moments_length(D, P_dst)
    moment_order_required = P_src + P_dst

    # Source coefficients (zero-copy view if SoA)
    src_coeffs = _get_coeffs_for_remap(source_pfs, source_fieldname)

    # Per-source-cell physical-frame pullback. Use precomputed if provided.
    src_pulls = if src_pullbacks === nothing
        [reference_to_physical_pullback(f, P_src) for f in src_frames]
    else
        length(src_pullbacks) == length(src_frames) ||
            throw(ArgumentError("src_pullbacks length ≠ src_frames length"))
        src_pullbacks
    end

    # Per-destination-cell pullback. Use precomputed if provided.
    dst_pulls = if dst_pullbacks === nothing
        v = Vector{Matrix{T}}(undef, n_cells(frame.mesh))
        @inbounds for ci in 1:n_cells(frame.mesh)
            if is_leaf(frame.mesh.cells[ci])
                v[ci] = reference_to_physical_pullback(dst_frames[ci], P_dst)
            else
                v[ci] = zeros(T, n_dst, n_dst)   # unused; shaped for type stability
            end
        end
        v
    else
        length(dst_pullbacks) == n_cells(frame.mesh) ||
            throw(ArgumentError("dst_pullbacks length ≠ n_cells(frame.mesh)"))
        dst_pullbacks
    end

    # Lookup table: (γ, δ) → flat physical-moment index for γ + δ
    multi_src = moment_multiindices(D, P_src)
    multi_dst = moment_multiindices(D, P_dst)
    multi_phys = moment_multiindices(D, P_src + P_dst)
    lookup = Matrix{Int}(undef, n_dst, n_src)
    @inbounds for γi in 1:n_dst, δi in 1:n_src
        sum_idx = ntuple(d -> multi_dst[γi][d] + multi_src[δi][d], D)
        lookup[γi, δi] = _lookup_multi_index(multi_phys, collect(sum_idx), D)
    end

    # Destination RHS to be filled
    rhs = zeros(T, n_dst, n_cells(frame.mesh))

    # Pi buffer (reused per simplex)
    Pi = zeros(T, n_src)

    state = StreamingRemapState{D, T}(rhs, dst_pulls, lookup, Pi,
                                       info.leaf_index_map, n_src, n_dst,
                                       frame.lo)

    # Voxelize workspace
    ws = workspace === nothing ? R3D.Flat.VoxelizeWorkspace{D, T}(64) : workspace

    # The polytope buffer for shifting Lagrangian simplex into frame-local coords
    poly = R3D.Flat.FlatPolytope{D, T}(64)

    # Process each Lagrangian simplex
    @inbounds for s in 1:n_simplices(lag)
        # Compute Pi[δ] = Σ_α src_coeffs[α, s] * src_pullback_s[α, δ]
        src_pull = src_pulls[s]
        for δ in 1:n_src
            acc = zero(T)
            for α in 1:n_src
                acc += src_coeffs[α, s] * src_pull[α, δ]
            end
            Pi[δ] = acc
        end

        # Build polytope for this simplex, shifted to frame-local coords
        # (voxel (i, j) covers [(i-1)*d, i*d] in *frame-local* coords)
        verts = simplex_vertex_positions(lag, s)
        shifted_verts = ntuple(k -> ntuple(dd -> T(verts[k][dd]) - state.frame_lo[dd], Val(D)), Val(D + 1))
        R3D.Flat.init_simplex!(poly, collect(shifted_verts))

        # Stream through voxelize_fold!
        if D == 2
            R3D.Flat.voxelize_fold!(_streaming_leaf_callback_2d,
                                      state, poly,
                                      info.ibox_lo, info.ibox_hi, info.d,
                                      moment_order_required;
                                      workspace = ws)
        else  # D == 3
            R3D.Flat.voxelize_fold!(_streaming_leaf_callback_3d,
                                      state, poly,
                                      info.ibox_lo, info.ibox_hi, info.d,
                                      moment_order_required;
                                      workspace = ws)
        end
    end

    # Solve M_j d_j = b_j per leaf
    target_coeffs, target_is_view = _get_coeffs_for_remap_writable(target_pfs, target_fieldname)
    fill!(target_coeffs, zero(T))
    @inbounds for ci in 1:n_cells(frame.mesh)
        is_leaf(frame.mesh.cells[ci]) || continue
        # Skip cells with no contribution
        any_nonzero = false
        for α in 1:n_dst
            if rhs[α, ci] != zero(T)
                any_nonzero = true; break
            end
        end
        any_nonzero || continue
        M_ref = reference_mass_matrix(dst_frames[ci], P_dst)
        jac = _frame_jacobian(dst_frames[ci])
        M_j = jac .* M_ref
        d = M_j \ view(rhs, :, ci)
        for α in 1:n_dst
            target_coeffs[α, ci] = d[α]
        end
    end
    if !target_is_view
        set_polynomial_coeffs_matrix!(target_pfs, target_fieldname, target_coeffs)
    end

    return target_pfs
end

"""
    polynomial_remap_l_to_uniform_e!(target_pfs, source_pfs, fieldname,
                                       lag, frame, src_frames, dst_frames,
                                       grid_info = nothing; workspace = nothing)

Convenience overload with the same field name on both sides.
"""
function polynomial_remap_l_to_uniform_e!(target_pfs::PolynomialFieldSet,
                                            source_pfs::PolynomialFieldSet,
                                            fieldname::Symbol,
                                            lag::SimplicialMesh{D, T},
                                            frame::EulerianFrame{D, T},
                                            src_frames::Vector{<:CellReferenceFrame{D, T}},
                                            dst_frames::Vector{<:CellReferenceFrame{D, T}},
                                            grid_info = nothing;
                                            kwargs...) where {D, T}
    return polynomial_remap_l_to_uniform_e!(
        target_pfs, fieldname, source_pfs, fieldname,
        lag, frame, src_frames, dst_frames, grid_info; kwargs...)
end

# ============================================================================
# E→L streaming: Eulerian uniform grid → Lagrangian simplicial mesh
# ============================================================================

"""
    polynomial_remap_uniform_e_to_l!(target_pfs, target_fieldname,
                                       source_pfs, source_fieldname,
                                       lag, frame, src_frames, dst_frames,
                                       grid_info = nothing;
                                       workspace = nothing,
                                       src_pullbacks = nothing,
                                       dst_pullbacks = nothing)

Single-pass polynomial remap from a **uniformly-refined** Eulerian frame
to a Lagrangian simplicial mesh. Mirror of
`polynomial_remap_l_to_uniform_e!`: the geometric loop is the same
(voxelize each Lagrangian simplex against the Eulerian grid), but the
data flow is reversed — each leaf's source polynomial contributes to
the **fixed destination column** of the current outer simplex.

# Arguments

Same as `polynomial_remap_l_to_uniform_e!`, with role swaps:
- `source_pfs` is indexed by Eulerian cell index
  (`n_elements == n_cells(frame.mesh)`).
- `target_pfs` is indexed by Lagrangian simplex index
  (`n_elements == n_simplices(lag)`).
- `src_frames` is for Eulerian leaves; `dst_frames` is for Lagrangian
  simplices.
- `src_pullbacks` (kwarg) is per Eulerian cell; `dst_pullbacks` is per
  Lagrangian simplex.

The destination polynomial for each Lagrangian simplex is the L²
projection of the per-leaf-piecewise source polynomial onto that
simplex's monomial basis in its reference frame.

# Implementation note

For efficiency, the source-side physical-frame coefficients
`Pi[δ, leaf] = src_pullback[leaf]^T · src_coeffs[:, leaf]` are
**precomputed for every leaf once** before the outer simplex loop, so
the per-leaf callback is a pure dot product. Without this hoisting the
E→L variant would recompute the same source contraction many times.

# Performance characteristics

E→L is structurally similar to L→E but does more per-leaf work: each
callback fire reads from `Pi_per_leaf` (one column lookup) and performs
the destination contraction. On dfmm-typical sizes (2048 simplices,
256-leaf grid, P=3), measured wall time is roughly 2× L→E. Both are
much faster than the two-phase pipeline for one-field-per-geometry use.

# Agreement with two-phase

Produces the same destination coefficients as
`polynomial_remap_field!(...; direction = :e_to_l)` to floating-point
round-off (~1e-13 relative for P=3 on O(1)-magnitude coefficients).
The two implementations sum contributions in different orders, so the
final bits differ; the math is identical.

# Errors

Throws `ArgumentError` if the Eulerian frame isn't uniformly refined,
the bases aren't `MonomialBasis`, or argument shapes don't match.
Throws for `D ∉ (2, 3)`.
"""
function polynomial_remap_uniform_e_to_l!(target_pfs::PolynomialFieldSet,
                                            target_fieldname::Symbol,
                                            source_pfs::PolynomialFieldSet,
                                            source_fieldname::Symbol,
                                            lag::SimplicialMesh{D, T},
                                            frame::EulerianFrame{D, T},
                                            src_frames::Vector{<:CellReferenceFrame{D, T}},
                                            dst_frames::Vector{<:CellReferenceFrame{D, T}},
                                            grid_info = nothing;
                                            workspace::Union{Nothing, R3D.Flat.VoxelizeWorkspace{D, T}} = nothing,
                                            src_pullbacks::Union{Nothing, Vector{Matrix{T}}} = nothing,
                                            dst_pullbacks::Union{Nothing, Vector{Matrix{T}}} = nothing
                                            ) where {D, T}
    D in (2, 3) || throw(ArgumentError(
        "polynomial_remap_uniform_e_to_l! supports D=2 and D=3 (got D=$D). " *
        "Polynomial remap consumes moments up to order P_src+P_dst, which " *
        "isn't yet available in r3djl at D≥4."))

    src_basis = basis_of(source_pfs)
    dst_basis = basis_of(target_pfs)
    src_basis isa MonomialBasis ||
        throw(ArgumentError("source field set must use MonomialBasis (got $(typeof(src_basis)))"))
    dst_basis isa MonomialBasis ||
        throw(ArgumentError("target field set must use MonomialBasis (got $(typeof(dst_basis)))"))
    P_src = _basis_order(src_basis)
    P_dst = _basis_order(dst_basis)
    _basis_dim(src_basis) == D ||
        throw(ArgumentError("source basis dimension ≠ frame dimension"))
    _basis_dim(dst_basis) == D ||
        throw(ArgumentError("target basis dimension ≠ frame dimension"))

    info = grid_info === nothing ? uniform_grid_dimensions(frame) : grid_info
    info === nothing &&
        throw(ArgumentError("Eulerian frame is not uniformly refined; use the " *
                             "two-phase polynomial_remap_field! with direction=:e_to_l instead."))

    n_simplices(lag) == length(dst_frames) ||
        throw(ArgumentError("dst_frames length ≠ n_simplices(lag) (E→L: dst is Lagrangian)"))
    n_cells(frame.mesh) == length(src_frames) ||
        throw(ArgumentError("src_frames length ≠ n_cells(frame.mesh) (E→L: src is Eulerian)"))
    n_elements(source_pfs) == n_cells(frame.mesh) ||
        throw(ArgumentError("source field set element count ≠ n_cells(frame.mesh)"))
    n_elements(target_pfs) == n_simplices(lag) ||
        throw(ArgumentError("target field set element count ≠ n_simplices(lag)"))

    n_src = moments_length(D, P_src)
    n_dst = moments_length(D, P_dst)
    moment_order_required = P_src + P_dst

    src_coeffs = _get_coeffs_for_remap(source_pfs, source_fieldname)

    # Source-side (Eulerian) pullbacks
    src_pulls = if src_pullbacks === nothing
        v = Vector{Matrix{T}}(undef, n_cells(frame.mesh))
        @inbounds for ci in 1:n_cells(frame.mesh)
            if is_leaf(frame.mesh.cells[ci])
                v[ci] = reference_to_physical_pullback(src_frames[ci], P_src)
            else
                v[ci] = zeros(T, n_src, n_src)   # unused; type-stable placeholder
            end
        end
        v
    else
        length(src_pullbacks) == n_cells(frame.mesh) ||
            throw(ArgumentError("src_pullbacks length ≠ n_cells(frame.mesh)"))
        src_pullbacks
    end

    # Destination-side (Lagrangian) pullbacks
    dst_pulls = if dst_pullbacks === nothing
        [reference_to_physical_pullback(f, P_dst) for f in dst_frames]
    else
        length(dst_pullbacks) == length(dst_frames) ||
            throw(ArgumentError("dst_pullbacks length ≠ dst_frames length"))
        dst_pullbacks
    end

    multi_src = moment_multiindices(D, P_src)
    multi_dst = moment_multiindices(D, P_dst)
    multi_phys = moment_multiindices(D, P_src + P_dst)
    lookup = Matrix{Int}(undef, n_dst, n_src)
    @inbounds for γi in 1:n_dst, δi in 1:n_src
        sum_idx = ntuple(d -> multi_dst[γi][d] + multi_src[δi][d], D)
        lookup[γi, δi] = _lookup_multi_index(multi_phys, collect(sum_idx), D)
    end

    rhs = zeros(T, n_dst, n_simplices(lag))

    # Precompute Pi for every Eulerian leaf:
    #   Pi_per_leaf[δ, leaf_idx] = Σ_α src_coeffs[α, leaf_idx] * src_pulls[leaf_idx][α, δ]
    # This hoists the source-side contraction out of the inner callback,
    # turning ~10× per-leaf work into ~1× per-leaf lookup. Cost is one
    # n_src×n_src matrix-vector per leaf (computed once, reused across
    # all simplex sweeps that touch that leaf).
    Pi_per_leaf = zeros(T, n_src, n_cells(frame.mesh))
    @inbounds for ci in 1:n_cells(frame.mesh)
        is_leaf(frame.mesh.cells[ci]) || continue
        src_pull = src_pulls[ci]
        for δ in 1:n_src
            acc = zero(T)
            for α in 1:n_src
                acc += src_coeffs[α, ci] * src_pull[α, δ]
            end
            Pi_per_leaf[δ, ci] = acc
        end
    end

    state = StreamingRemapStateEtoL{D, T}(
        rhs,
        0,                            # dst_simplex_idx (set per outer)
        zeros(T, n_dst, n_dst),       # dst_pullback_current (set per outer)
        Pi_per_leaf,
        lookup,
        info.leaf_index_map,
        n_src, n_dst,
        frame.lo,
    )

    ws = workspace === nothing ? R3D.Flat.VoxelizeWorkspace{D, T}(64) : workspace
    poly = R3D.Flat.FlatPolytope{D, T}(64)

    @inbounds for s in 1:n_simplices(lag)
        # Fix the destination context for this sweep
        state.dst_simplex_idx = s
        state.dst_pullback_current = dst_pulls[s]

        # Build polytope for this simplex, shifted to frame-local coords
        verts = simplex_vertex_positions(lag, s)
        shifted_verts = ntuple(k -> ntuple(dd -> T(verts[k][dd]) - state.frame_lo[dd], Val(D)), Val(D + 1))
        R3D.Flat.init_simplex!(poly, collect(shifted_verts))

        if D == 2
            R3D.Flat.voxelize_fold!(_streaming_leaf_callback_etol_2d,
                                      state, poly,
                                      info.ibox_lo, info.ibox_hi, info.d,
                                      moment_order_required;
                                      workspace = ws)
        else
            R3D.Flat.voxelize_fold!(_streaming_leaf_callback_etol_3d,
                                      state, poly,
                                      info.ibox_lo, info.ibox_hi, info.d,
                                      moment_order_required;
                                      workspace = ws)
        end
    end

    # Solve M_j d_j = b_j per Lagrangian simplex
    target_coeffs, target_is_view = _get_coeffs_for_remap_writable(target_pfs, target_fieldname)
    fill!(target_coeffs, zero(T))
    @inbounds for j in 1:n_simplices(lag)
        any_nonzero = false
        for α in 1:n_dst
            if rhs[α, j] != zero(T)
                any_nonzero = true; break
            end
        end
        any_nonzero || continue
        M_ref = reference_mass_matrix(dst_frames[j], P_dst)
        jac = _frame_jacobian(dst_frames[j])
        M_j = jac .* M_ref
        d = M_j \ view(rhs, :, j)
        for α in 1:n_dst
            target_coeffs[α, j] = d[α]
        end
    end
    if !target_is_view
        set_polynomial_coeffs_matrix!(target_pfs, target_fieldname, target_coeffs)
    end

    return target_pfs
end

"""
    polynomial_remap_uniform_e_to_l!(target_pfs, source_pfs, fieldname,
                                       lag, frame, src_frames, dst_frames,
                                       grid_info = nothing; kwargs...)

Convenience overload with the same field name on both sides.
"""
function polynomial_remap_uniform_e_to_l!(target_pfs::PolynomialFieldSet,
                                            source_pfs::PolynomialFieldSet,
                                            fieldname::Symbol,
                                            lag::SimplicialMesh{D, T},
                                            frame::EulerianFrame{D, T},
                                            src_frames::Vector{<:CellReferenceFrame{D, T}},
                                            dst_frames::Vector{<:CellReferenceFrame{D, T}},
                                            grid_info = nothing;
                                            kwargs...) where {D, T}
    return polynomial_remap_uniform_e_to_l!(
        target_pfs, fieldname, source_pfs, fieldname,
        lag, frame, src_frames, dst_frames, grid_info; kwargs...)
end
