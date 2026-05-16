# ============================================================================
# GeometricMultigrid — Poisson solver on a PatchHierarchy
#
# Composite cell-centered 2nd-order finite-volume Laplacian + geometric V-cycle.
#
# v1 scope:
#   * One patch per level (FFT bottom requires it; AMR tests use it).
#   * Degree-0 polynomial fields (MonomialBasis{D, 0}).
#   * Red-black Gauss-Seidel smoother (level-restricted).
#   * Volume-weighted residual restriction (via existing
#     `restrict_to_parents!` degree-0 path).
#   * Multilinear correction prolongation (NEW; the existing degree-0
#     prolong would degrade MG to first order).
#   * FFT bottom solver on the level_range[1] root patch: periodic /
#     Dirichlet / Neumann via rfft and FFTW.r2r (DST-II / DCT-II).
#   * Partial-hierarchy support: `solve_poisson!(...; level_range = ℓ_lo:ℓ_hi)`
#     updates only those levels; levels outside the range stay frozen and
#     feed in via the existing `PatchBoundaryBC` parent-lookup path.
#   * `interior_solver = :schur` is reserved for v2 (see plan §G).
# ============================================================================

module GeometricMultigrid

using ..Mesh: HierarchicalMesh, n_cells, is_leaf, level_of,
              refine_cells!,
              face_neighbors, face_fine_neighbors,
              register_refinement_listener!, unregister_refinement_listener!,
              ListenerHandle
using ..BoundaryConditions: BCKind, BoundarySpec,
                            PERIODIC, DIRICHLET, REFLECTING, INFLOW, OUTFLOW,
                            is_periodic_axis
using ..Bases: MonomialBasis, n_coeffs
using ..Storage: allocate_polynomial_fields, SoA
using ..Overlap: EulerianFrame, FrameBoundaries, cell_physical_box,
                  enumerate_leaves
using ..Threading: AbstractParallelBackend, Sequential, default_backend,
                   parallel_foreach
using ..Solver: PatchHierarchy, PatchView, PatchHaloView,
                for_each_patch!, restrict_to_parents!, n_levels, n_patches,
                patches_at, add_patches!

using FFTW
using LinearAlgebra: norm, mul!, lu!, ldiv!, lu, cholesky
using SparseArrays: SparseMatrixCSC, sparse, spzeros, nnz

export MGWorkspace, MGOptions, MGResult,
       solve_poisson!, vcycle!,
       apply_laplacian!, compute_residual!, residual_l2,
       allocate_phi_rho, build_uniform_root_hierarchy,
       manufactured_rhs!, fill_field!

# ----------------------------------------------------------------------------
# Field allocation helpers
# ----------------------------------------------------------------------------

# Allocate a degree-0 (phi, rho) field set for every patch on every level.
# Returns a `Vector{Vector{NamedTuple{(:phi, :rho)}}}` (outer: level, inner:
# patch).
function allocate_phi_rho(ph::PatchHierarchy{D, T}) where {D, T}
    basis = MonomialBasis{D, 0}()
    fields = Vector{Vector{NamedTuple}}(undef, n_levels(ph))
    for ℓ in 1:n_levels(ph)
        patches = patches_at(ph, ℓ)
        per_patch = Vector{NamedTuple}(undef, length(patches))
        for (pi, p) in enumerate(patches)
            n = n_cells(p.mesh)
            pfs = allocate_polynomial_fields(SoA(), basis, n;
                                              phi = T, rho = T)
            for i in 1:length(pfs.phi); pfs.phi[i] = (zero(T),); end
            for i in 1:length(pfs.rho); pfs.rho[i] = (zero(T),); end
            per_patch[pi] = (phi = pfs.phi, rho = pfs.rho)
        end
        fields[ℓ] = per_patch
    end
    return fields
end

# Fill `field` (one symbol) by cell-center evaluation of `f`.
function fill_field!(fields::Vector{Vector{NamedTuple}},
                     ph::PatchHierarchy{D, T},
                     name::Symbol, f) where {D, T}
    for ℓ in 1:n_levels(ph)
        patches = patches_at(ph, ℓ)
        for (pi, frame) in enumerate(patches)
            leaves = enumerate_leaves(frame.mesh)
            view = fields[ℓ][pi]
            arr = getfield(view, name)
            @inbounds for c in leaves
                lo, hi = cell_physical_box(frame, c)
                center = ntuple(d -> (lo[d] + hi[d]) / 2, Val(D))
                arr[c] = (T(f(center)),)
            end
        end
    end
    return fields
end

# Zero one field across the whole hierarchy.
function zero_field!(fields::Vector{Vector{NamedTuple}}, name::Symbol)
    for level_fields in fields
        for view in level_fields
            f = getfield(view, name)
            @inbounds for i in 1:length(f)
                f[i] = (zero(eltype(f[i])),)
            end
        end
    end
    return fields
end

@inline _get_val(f, i) = @inbounds f[i][1]
@inline _set_val!(f, i, v) = (@inbounds f[i] = (v,); nothing)

# Local copy of `Solver._find_parent_patch` (not exported). Returns the
# 1-based index of the parent patch containing `child`, or 0 if none.
@inline function _local_find_parent_patch(parents::Vector{EulerianFrame{D, T}},
                                            child::EulerianFrame{D, T}) where {D, T}
    @inbounds for (i, p) in enumerate(parents)
        inside = true
        for d in 1:D
            if child.lo[d] < p.lo[d] || child.hi[d] > p.hi[d]
                inside = false; break
            end
        end
        inside && return i
    end
    return 0
end

# ----------------------------------------------------------------------------
# Uniform-patch geometry
# ----------------------------------------------------------------------------

# For a uniform-refined patch with `2^k` cells per axis, build a mapping
# `cell_idx -> NTuple{D,Int}` of 1-based Cartesian indices. Non-leaf cells
# map to a zero tuple. Returns `(grid_idx, N, dx, leaves)`.
function build_grid_map(frame::EulerianFrame{D, T}) where {D, T}
    mesh = frame.mesh
    leaves = enumerate_leaves(mesh)
    n = n_cells(mesh)
    grid_idx = fill(ntuple(_ -> 0, Val(D)), n)
    if isempty(leaves)
        return grid_idx, ntuple(_ -> 0, Val(D)), ntuple(_ -> zero(T), Val(D)), Int[]
    end
    lo0, hi0 = cell_physical_box(frame, leaves[1])
    dx = ntuple(d -> hi0[d] - lo0[d], Val(D))
    N = ntuple(d -> round(Int, (frame.hi[d] - frame.lo[d]) / dx[d]), Val(D))
    @inbounds for c in leaves
        c_lo, c_hi = cell_physical_box(frame, c)
        center = ntuple(d -> (c_lo[d] + c_hi[d]) / 2, Val(D))
        gi = ntuple(d -> Int(floor((center[d] - frame.lo[d]) / dx[d])) + 1, Val(D))
        grid_idx[c] = gi
    end
    return grid_idx, N, dx, leaves
end

# ----------------------------------------------------------------------------
# MGOptions / MGResult
# ----------------------------------------------------------------------------

# Per-axis BC summary used by `_read_phi` to resolve `nothing` ghost reads.
struct AxisBCKinds{D}
    kinds::NTuple{D, Symbol}   # :periodic | :dirichlet | :neumann
end

@inline function _axis_kinds_from_bcs(bcs::BoundarySpec{D}) where {D}
    kinds = ntuple(D) do d
        lo = bcs[d][1]; hi = bcs[d][2]
        if lo == PERIODIC && hi == PERIODIC
            :periodic
        elseif lo == DIRICHLET && hi == DIRICHLET
            :dirichlet
        elseif lo == REFLECTING && hi == REFLECTING
            :neumann
        else
            :neumann
        end
    end
    return AxisBCKinds{D}(kinds)
end

Base.@kwdef struct MGOptions
    n_pre::Int            = 2
    n_post::Int           = 2
    tol::Float64          = 1e-8
    maxiter::Int          = 50
    cycle::Symbol         = :vcycle
    interior_solver::Symbol = :rbgs
    verbose::Bool         = false
    project_mean::Bool    = true
    bottom_smooth_iters::Int = 40

    # Bottom-solver choice for single-level recursive calls when the FFT
    # plan is NOT applicable (typically fine-patch subcycling or
    # non-Cartesian root). `:cg` runs Jacobi-preconditioned conjugate
    # gradients against the matrix-free Laplacian — converges in O(N)
    # iterations vs O(N²) for raw GS sweeping. `:gs` keeps the old
    # smoother-only path (kept for compat / debugging).
    bottom_solver::Symbol = :cg
    bottom_cg_tol::Float64 = 1e-10
    bottom_cg_maxiter::Int = 500
end

struct MGResult
    iters::Int
    res_init::Float64
    res_final::Float64
    converged::Bool
    history::Vector{Float64}
end

# ----------------------------------------------------------------------------
# MGWorkspace
# ----------------------------------------------------------------------------

mutable struct MGWorkspace{D, T}
    ph::PatchHierarchy{D, T}
    bcs::BoundarySpec{D}
    level_range::UnitRange{Int}
    opts::MGOptions

    grid_idx::Vector{Vector{Vector{NTuple{D, Int}}}}
    patch_N::Vector{Vector{NTuple{D, Int}}}
    patch_dx::Vector{Vector{NTuple{D, T}}}
    patch_leaves::Vector{Vector{Vector{Int}}}

    # Inverse of grid_idx: cart_to_cell[ℓ][pi][i, j[, k]] → cell index.
    # Used by the fast direct-array smoother to read stencil neighbors
    # without going through `for_each_patch!` (which allocates per cell).
    cart_to_cell::Vector{Vector{Array{Int, D}}}

    # Cached per-axis BC summary.
    axis_kinds::AxisBCKinds{D}

    # Cached per-level, per-patch rho_flat for the smoother (only sized to
    # match each patch's n_cells; reused across V-cycles).
    rho_flat::Vector{Vector{Vector{T}}}

    # `covered_by_finer[ℓ][pi][c]` is `true` if coarse cell `c` at level ℓ
    # is covered by a patch at level ℓ+1 (i.e., a finer patch sits on top
    # of it). During AMR V-cycles, the residual on covered coarse cells is
    # overwritten by the volume-averaged fine residual via
    # `restrict_to_parents!` — so computing the coarse Laplacian on those
    # cells is wasted work. The Laplacian / GS kernels skip covered cells
    # when this mask is consulted. Top-level value is `false` everywhere.
    covered_by_finer::Vector{Vector{Vector{Bool}}}

    # `parent_cell[ℓ][pi][c, k]` is the parent-patch cell index containing
    # the +/-axis-aligned GHOST of fine cell `c` at level ℓ. The face index
    # `k ∈ 1:2D` follows the convention used by `face_neighbors`
    # (axis 1 lo, axis 1 hi, axis 2 lo, axis 2 hi, …). 0 means the ghost
    # is outside the parent patch (shouldn't happen if patches are
    # well-contained). Pre-computed once at workspace construction so the
    # fast level-ℓ Laplacian / smoother avoids the O(n_parent) scan that
    # `PatchHaloView._find_leaf_containing` does.
    parent_cell::Vector{Vector{Array{Int, 2}}}

    # FAC flux fix: inverted view of `parent_cell`. For each coarse cell `c`
    # at level ℓ-1 and each coarse face `k ∈ 1:2D`, lists the fine cells
    # at level ℓ whose ghost lookup lands inside coarse cell `c` on face `k`.
    # Indexed as `fac_fine_at_coarse_face[ℓ-1][pi_coarse][c, k] :: Vector{Int}`.
    # Empty list ⇒ coarse cell c's face k is NOT a C/F interface.
    # Built lazily — workspaces without finer levels just have empty
    # vectors. Used by `apply_fac_flux_fix!` to replace the coarse one-
    # sided flux at each C/F face with the area-weighted sum of the fine
    # sub-face fluxes (Martin-Colella's 2nd-order conservative composite
    # operator).
    fac_fine_at_coarse_face::Vector{Vector{Matrix{Vector{Int}}}}

    # Scratch fields, one per (level, patch).
    residual::Vector{Vector{NamedTuple}}
    correction::Vector{Vector{NamedTuple}}
    rhs_coarse::Vector{Vector{NamedTuple}}

    # FFT bottom solver state (configured at level_range[1] if single-patch
    # and tile-uniform; otherwise we fall back to many GS sweeps at bottom).
    fft_ok::Bool
    fft_level::Int
    fft_N::NTuple{D, Int}
    fft_dx::NTuple{D, T}
    fft_inv_eig::Array{T, D}
    fft_kind::Vector{Symbol}
    fft_buf::Array{T, D}
    fft_buf_complex::Array{Complex{T}, D}
    fft_plan_fwd::Any
    fft_plan_inv::Any

    # Schur factor cache, per level (lazy-built on first :schur bottom call).
    # Only the levels with a corresponding parent benefit. Invalidated by
    # the refinement listener.
    schur_factors::Dict{Int, Any}

    listener::Union{Nothing, ListenerHandle}
end

function MGWorkspace(ph::PatchHierarchy{D, T},
                     bcs::BoundarySpec{D};
                     level_range::UnitRange{Int} = 1:n_levels(ph),
                     opts::MGOptions = MGOptions()) where {D, T}
    nL = n_levels(ph)
    first(level_range) >= 1 && last(level_range) <= nL ||
        throw(ArgumentError("MGWorkspace: level_range $level_range out of 1:$nL"))

    grid_idx = Vector{Vector{Vector{NTuple{D, Int}}}}(undef, nL)
    patch_N  = Vector{Vector{NTuple{D, Int}}}(undef, nL)
    patch_dx = Vector{Vector{NTuple{D, T}}}(undef, nL)
    patch_leaves = Vector{Vector{Vector{Int}}}(undef, nL)

    for ℓ in 1:nL
        patches = patches_at(ph, ℓ)
        gp = Vector{Vector{NTuple{D, Int}}}(undef, length(patches))
        Np = Vector{NTuple{D, Int}}(undef, length(patches))
        dxp = Vector{NTuple{D, T}}(undef, length(patches))
        leavesp = Vector{Vector{Int}}(undef, length(patches))
        for (pi, frame) in enumerate(patches)
            gi, N, dx, leaves = build_grid_map(frame)
            gp[pi] = gi; Np[pi] = N; dxp[pi] = dx; leavesp[pi] = leaves
        end
        grid_idx[ℓ] = gp; patch_N[ℓ] = Np; patch_dx[ℓ] = dxp; patch_leaves[ℓ] = leavesp
    end

    residual   = allocate_phi_rho(ph)
    correction = allocate_phi_rho(ph)
    rhs_coarse = allocate_phi_rho(ph)

    axis_kinds = _axis_kinds_from_bcs(bcs)
    rho_flat = Vector{Vector{Vector{T}}}(undef, nL)
    cart_to_cell = Vector{Vector{Array{Int, D}}}(undef, nL)
    covered_by_finer = Vector{Vector{Vector{Bool}}}(undef, nL)
    for ℓ in 1:nL
        npatches = length(patches_at(ph, ℓ))
        rho_flat[ℓ] = [Vector{T}(undef, n_cells(patches_at(ph, ℓ)[pi].mesh))
                        for pi in 1:npatches]
        c2c = Vector{Array{Int, D}}(undef, npatches)
        covered = Vector{Vector{Bool}}(undef, npatches)
        for pi in 1:npatches
            N = patch_N[ℓ][pi]
            c2c_p = fill(0, N)
            for c in patch_leaves[ℓ][pi]
                gi = grid_idx[ℓ][pi][c]
                c2c_p[gi...] = c
            end
            c2c[pi] = c2c_p
            covered[pi] = fill(false, n_cells(patches_at(ph, ℓ)[pi].mesh))
        end
        cart_to_cell[ℓ] = c2c
        covered_by_finer[ℓ] = covered
    end
    # Mark coarse cells covered by any finer-level patch. A coarse cell is
    # covered if its physical cell box is contained inside the union of
    # finer patch boxes. Cheap O(n_coarse * n_fine_patches) build.
    for ℓ in 1:(nL - 1)
        finer_patches = patches_at(ph, ℓ + 1)
        isempty(finer_patches) && continue
        for pi in 1:length(patches_at(ph, ℓ))
            frame = patches_at(ph, ℓ)[pi]
            mask = covered_by_finer[ℓ][pi]
            for c in patch_leaves[ℓ][pi]
                c_lo, c_hi = cell_physical_box(frame, c)
                inside_any = false
                for fp in finer_patches
                    fp_lo, fp_hi = fp.lo, fp.hi
                    ok = true
                    @inbounds for d in 1:D
                        if c_lo[d] < fp_lo[d] || c_hi[d] > fp_hi[d]
                            ok = false; break
                        end
                    end
                    if ok
                        inside_any = true; break
                    end
                end
                mask[c] = inside_any
            end
        end
    end

    # Pre-compute the parent-cell index containing the ghost of every fine
    # cell at every face. Skips level 1 (no parent). Boundary cells get a
    # non-zero entry; interior fine cells get 0 (the same-patch face has
    # priority in the kernel and the parent slot isn't consulted).
    #
    # Simultaneously builds the inverted index `fac_fine_at_coarse_face`
    # used by the FAC flux fix on the coarse side.
    parent_cell = Vector{Vector{Array{Int, 2}}}(undef, nL)
    fac_fine_at_coarse_face = Vector{Vector{Matrix{Vector{Int}}}}(undef, nL)
    for ℓ in 1:nL
        npatches = length(patches_at(ph, ℓ))
        parent_cell[ℓ] = Vector{Array{Int, 2}}(undef, npatches)
        fac_fine_at_coarse_face[ℓ] = Vector{Matrix{Vector{Int}}}(undef, npatches)
        for pi in 1:npatches
            n = n_cells(patches_at(ph, ℓ)[pi].mesh)
            parent_cell[ℓ][pi] = fill(0, n, 2 * D)
            # Inverted index, one Vector{Int} per (coarse_cell, face) slot.
            # The coarse-side slot is allocated at level ℓ (acting as the
            # "parent of level ℓ+1"); level nL has no inverse index needed.
            tbl_inv = Matrix{Vector{Int}}(undef, n, 2 * D)
            for c in 1:n, k in 1:(2 * D)
                tbl_inv[c, k] = Int[]
            end
            fac_fine_at_coarse_face[ℓ][pi] = tbl_inv
        end
    end
    for ℓ in 2:nL
        coarse_patches = patches_at(ph, ℓ - 1)
        for (pi, fine_frame) in enumerate(patches_at(ph, ℓ))
            ppi = _local_find_parent_patch(coarse_patches, fine_frame)
            ppi == 0 && continue
            parent_frame = coarse_patches[ppi]
            parent_N = patch_N[ℓ - 1][ppi]
            parent_dx = patch_dx[ℓ - 1][ppi]
            parent_lo = parent_frame.lo
            parent_c2c = cart_to_cell[ℓ - 1][ppi]
            fine_dx = patch_dx[ℓ][pi]
            fine_lo = fine_frame.lo
            tbl = parent_cell[ℓ][pi]
            tbl_inv = fac_fine_at_coarse_face[ℓ - 1][ppi]
            for c in patch_leaves[ℓ][pi]
                gi = grid_idx[ℓ][pi][c]
                for d in 1:D, side in (-1, 1)
                    face = 2 * (d - 1) + (side == 1 ? 2 : 1)
                    if (side == -1 && gi[d] > 1) || (side == 1 && gi[d] < patch_N[ℓ][pi][d])
                        continue
                    end
                    g_d = fine_lo[d] + (gi[d] - T(0.5) + T(side)) * fine_dx[d]
                    pi_d = clamp(Int(floor((g_d - parent_lo[d]) / parent_dx[d])) + 1,
                                  1, parent_N[d])
                    par_gi_tup = ntuple(j -> begin
                        if j == d
                            pi_d
                        else
                            fine_center_j = fine_lo[j] + (gi[j] - T(0.5)) * fine_dx[j]
                            clamp(Int(floor((fine_center_j - parent_lo[j]) / parent_dx[j])) + 1,
                                   1, parent_N[j])
                        end
                    end, Val(D))
                    par_c = parent_c2c[par_gi_tup...]
                    tbl[c, face] = par_c
                    if par_c != 0
                        # On the COARSE side, the corresponding face is the
                        # MIRROR direction (a fine cell whose -x neighbor is
                        # in the parent means the parent's +x face is the C/F
                        # interface from c's perspective).
                        coarse_face = 2 * (d - 1) + (side == 1 ? 1 : 2)
                        push!(tbl_inv[par_c, coarse_face], c)
                    end
                end
            end
        end
    end

    # FFT bottom on level_range[1] (only if single-patch + tile-uniform).
    fft_ok = false
    fft_level = first(level_range)
    fft_N = ntuple(_ -> 0, Val(D)); fft_dx = ntuple(_ -> zero(T), Val(D))
    fft_inv_eig = Array{T, D}(undef, ntuple(_ -> 1, Val(D)))
    fft_kind   = Symbol[]
    fft_buf    = Array{T, D}(undef, ntuple(_ -> 1, Val(D)))
    fft_buf_complex = Array{Complex{T}, D}(undef, ntuple(_ -> 1, Val(D)))
    fft_plan_fwd = nothing
    fft_plan_inv = nothing

    if length(patches_at(ph, fft_level)) == 1
        frame = patches_at(ph, fft_level)[1]
        N  = patch_N[fft_level][1]
        dx = patch_dx[fft_level][1]
        nleaves = count(c -> is_leaf(frame.mesh.cells[c]), 1:n_cells(frame.mesh))
        # FFT bottom requires the patch to cover the full domain box (so
        # the physical_bcs are the boundary condition seen by every face).
        # For non-root patches the outer faces are Dirichlet-from-parent
        # and the periodic / DST / DCT eigenvalues don't apply.
        root_frame = patches_at(ph, 1)[1]
        covers_domain = all(d -> frame.lo[d] == root_frame.lo[d] &&
                                  frame.hi[d] == root_frame.hi[d], 1:D)
        if prod(N) == nleaves && covers_domain
            kinds = Symbol[]
            ok = true
            for d in 1:D
                lo_k = bcs[d][1]; hi_k = bcs[d][2]
                if lo_k == PERIODIC && hi_k == PERIODIC
                    push!(kinds, :periodic)
                elseif lo_k == DIRICHLET && hi_k == DIRICHLET
                    push!(kinds, :dirichlet)
                elseif lo_k == REFLECTING && hi_k == REFLECTING
                    push!(kinds, :neumann)
                else
                    ok = false; break
                end
            end
            if ok
                fft_ok = true; fft_N = N; fft_dx = dx; fft_kind = kinds
                fft_buf = Array{T, D}(undef, N)
                fft_inv_eig = _build_inv_eigenvalues(N, dx, kinds, T)
                fft_plan_fwd, fft_plan_inv, fft_buf_complex =
                    _build_fft_plans(fft_buf, kinds)
            end
        end
    end

    ws = MGWorkspace{D, T}(ph, bcs, level_range, opts,
                            grid_idx, patch_N, patch_dx, patch_leaves,
                            cart_to_cell, axis_kinds, rho_flat,
                            covered_by_finer, parent_cell,
                            fac_fine_at_coarse_face,
                            residual, correction, rhs_coarse,
                            fft_ok, fft_level, fft_N, fft_dx, fft_inv_eig,
                            fft_kind, fft_buf, fft_buf_complex,
                            fft_plan_fwd, fft_plan_inv,
                            Dict{Int, Any}(),
                            nothing)

    # Refinement listener: blow away FFT and Schur state on AMR.
    handle = register_refinement_listener!(patches_at(ph, 1)[1].mesh, _ -> begin
        ws.fft_ok = false
        empty!(ws.schur_factors)
    end)
    ws.listener = handle
    return ws
end

function _build_inv_eigenvalues(N::NTuple{D, Int}, dx::NTuple{D, T},
                                  kinds::Vector{Symbol}, ::Type{T}) where {D, T}
    eigs = Vector{Vector{T}}(undef, D)
    for d in 1:D
        Nd = N[d]; h2 = dx[d] * dx[d]
        vd = Vector{T}(undef, Nd)
        if kinds[d] === :periodic
            for k in 0:(Nd - 1)
                vd[k + 1] = (2 * cos(2π * k / Nd) - 2) / h2
            end
        elseif kinds[d] === :dirichlet
            for k in 0:(Nd - 1)
                vd[k + 1] = -4 * sin(π * (k + 1) / (2 * Nd))^2 / h2
            end
        elseif kinds[d] === :neumann
            for k in 0:(Nd - 1)
                vd[k + 1] = -4 * sin(π * k / (2 * Nd))^2 / h2
            end
        else
            error("_build_inv_eigenvalues: unknown axis BC :$(kinds[d])")
        end
        eigs[d] = vd
    end
    out = Array{T, D}(undef, N)
    @inbounds for I in CartesianIndices(N)
        s = zero(T)
        for d in 1:D; s += eigs[d][I[d]]; end
        out[I] = s == 0 ? zero(T) : one(T) / s
    end
    return out
end

function _build_fft_plans(buf::Array{T, D}, kinds::Vector{Symbol}) where {T, D}
    if all(==(:periodic), kinds)
        sz = size(buf)
        rsz = (sz[1] ÷ 2 + 1, sz[2:end]...)
        complex_buf = Array{Complex{T}, D}(undef, rsz)
        fwd = plan_rfft(buf; flags = FFTW.MEASURE)
        inv = plan_irfft(complex_buf, sz[1]; flags = FFTW.MEASURE)
        return fwd, inv, complex_buf
    end
    fwd_types = ntuple(d -> kinds[d] === :periodic ? FFTW.R2HC :
                              kinds[d] === :dirichlet ? FFTW.RODFT10 :
                              kinds[d] === :neumann ? FFTW.REDFT10 :
                              error("unknown kind"), D)
    inv_types = ntuple(d -> kinds[d] === :periodic ? FFTW.HC2R :
                              kinds[d] === :dirichlet ? FFTW.RODFT01 :
                              kinds[d] === :neumann ? FFTW.REDFT01 :
                              error("unknown kind"), D)
    fwd = FFTW.plan_r2r(buf, fwd_types; flags = FFTW.MEASURE)
    inv = FFTW.plan_r2r(buf, inv_types; flags = FFTW.MEASURE)
    return fwd, inv, Array{Complex{T}, D}(undef, ntuple(_ -> 1, Val(D)))
end

function release!(ws::MGWorkspace)
    if ws.listener !== nothing
        unregister_refinement_listener!(patches_at(ws.ph, 1)[1].mesh, ws.listener)
        ws.listener = nothing
    end
    return ws
end

# ----------------------------------------------------------------------------
# Composite Laplacian (matrix-free) and residual
# ----------------------------------------------------------------------------

# Read φ at a one-step stencil offset, handling three cases:
#   1. Same-patch neighbor (detected via `face_neighbors` on the mesh) —
#      read directly from the same-patch fields.
#   2. Off-patch but covered by a parent (detected because case 1 failed
#      AND the PatchHaloView returns a non-`nothing` value) — apply the
#      Martin–Colella linear ghost: φ_ghost = (1/3) φ_c + (2/3) φ_parent.
#      This restores 2nd-order accuracy at the C/F interface; the naive
#      degree-0 ghost stalls the V-cycle at ~3–7% relative residual.
#   3. Outer-domain wall with no parent — apply the BC ghost rule
#      (Dirichlet → −φ_c; Neumann → +φ_c; periodic is wrapped by the
#      HaloView at case 2 and treated as same-level data — DO NOT apply MC).
@inline function _read_phi(hv::PatchHaloView{Names, Tin, D, T, GD, BC, NT, PBC},
                            off::NTuple{D, Int}, phi_c::T,
                            axis_kinds::NTuple{D, Symbol}
                            ) where {Names, Tin, D, T, GD, BC, NT, PBC}
    base = getfield(hv, :physical_halo)
    mesh = base.mesh
    c = getfield(base, :cell_index)

    # Determine the active axis and side of the offset (one-step axis-aligned).
    d_active = 0
    sgn = 0
    @inbounds for d in 1:D
        if off[d] != 0
            d_active = d; sgn = off[d]
            break
        end
    end
    if d_active == 0
        return phi_c
    end

    # Step 1: same-patch (BC-unaware) face lookup. Non-zero ⇒ in-patch.
    face_idx = 2 * (d_active - 1) + (sgn == 1 ? 2 : 1)
    nbr = @inbounds face_neighbors(mesh, c)[face_idx]
    if nbr != 0
        fields_in = getfield(base, :fields_in)
        field = getfield(fields_in, :phi)
        return T(@inbounds field[Int(nbr)][1])
    end

    # Step 2: parent lookup or BC fallback (delegated to PatchHaloView).
    v = hv[:phi, off]
    if v === nothing
        # No parent and the BC reported `nothing` (DIRICHLET / INFLOW).
        return axis_kinds[d_active] === :dirichlet ? -phi_c : phi_c
    end

    # Two sub-cases: parent (apply MC) or root-level BC-wrap (skip MC).
    # We're in case 2 only if face_neighbors returned 0 (i.e., the cell is
    # at a patch wall). On the root patch a wall with a PERIODIC axis hits
    # the BC-resolved HaloView (not the parent). Test this by asking the
    # patch_bcs whether a parent exists.
    pbc = getfield(hv, :patch_bcs)
    if pbc.parent_frame === nothing
        # No parent in scope ⇒ this is a BC-wrapped read (e.g., periodic
        # wrap on the root patch). The wrapped neighbor is at "same-level"
        # distance, so DO NOT apply MC.
        return T(v[1])
    end
    # Parent lookup succeeded ⇒ Martin–Colella linear ghost.
    phi_p = T(v[1])
    return T(1//3) * phi_c + T(2//3) * phi_p
end

# Laplacian kernel: writes L φ into pv[:phi].
function _lapl_kernel(pv::PatchView, hv::PatchHaloView, ctx::AxisBCKinds{D}) where {D}
    p_lo, p_hi = cell_physical_box(hv.frame, pv.cv.index)
    T = eltype(p_lo)
    phi_c = T(pv[:phi][1])
    acc = zero(T)
    @inbounds for d in 1:D
        h = p_hi[d] - p_lo[d]
        invh2 = one(T) / (h * h)
        off_p = ntuple(j -> j == d ?  1 : 0, Val(D))
        off_m = ntuple(j -> j == d ? -1 : 0, Val(D))
        phi_p = _read_phi(hv, off_p, phi_c, ctx.kinds)
        phi_m = _read_phi(hv, off_m, phi_c, ctx.kinds)
        acc += (phi_p - 2 * phi_c + phi_m) * invh2
    end
    pv[:phi] = (acc,)
    return nothing
end

"""
    apply_laplacian!(Lphi, phi, ws; level_range = ws.level_range)

Apply the composite Laplacian L over `level_range`. Writes
`Lphi[ℓ][p][:phi] = (L φ)_cell`.
"""
function apply_laplacian!(Lphi::Vector{Vector{NamedTuple}},
                          phi::Vector{Vector{NamedTuple}},
                          ws::MGWorkspace{D, T};
                          level_range::UnitRange{Int} = ws.level_range,
                          skip_covered::Bool = false,
                          skip_cf::Bool = false,
                          backend = default_backend()) where {D, T}
    for ℓ in level_range
        if ℓ == 1
            apply_laplacian_root_fast!(Lphi, phi, ws;
                                         skip_covered = skip_covered,
                                         skip_cf = skip_cf)
        else
            apply_laplacian_fine_fast!(Lphi, phi, ws, ℓ)
        end
    end
    return Lphi
end

# Fast direct-array Laplacian for a fine (non-root) level. Uses the
# workspace's pre-cached cart_to_cell and parent_cell tables to avoid the
# O(n_parent) `_find_leaf_containing` scan that PatchHaloView does.
# Applies Martin–Colella linear ghost (1/3 φ_c + 2/3 φ_parent) for off-
# patch reads. Outer-domain reads from a fine patch are not expected for
# subcycling AMR; if a fine patch touches the domain wall, both `nb_c`
# and `par_c` are 0 and we fall back to the BC ghost rule (Dirichlet =
# −φ_c, periodic / Neumann = +φ_c).
function apply_laplacian_fine_fast!(Lphi::Vector{Vector{NamedTuple}},
                                     phi::Vector{Vector{NamedTuple}},
                                     ws::MGWorkspace{D, T}, ℓ::Int) where {D, T}
    pi = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    pcell = ws.parent_cell[ℓ][pi]
    phi_f = phi[ℓ][pi].phi
    Lphi_f = Lphi[ℓ][pi].phi
    kinds = ws.axis_kinds.kinds
    parent_phi = phi[ℓ - 1][1].phi
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    one_third = T(1//3); two_thirds = T(2//3)

    @inbounds for I in CartesianIndices(N)
        c = c2c[I]
        c == 0 && continue
        phi_c = T(phi_f[c][1])
        acc = zero(T)
        for d in 1:D
            for (side, face_idx) in ((1, 2 * (d - 1) + 2), (-1, 2 * (d - 1) + 1))
                # In-patch?
                inew = I[d] + side
                if 1 <= inew <= N[d]
                    nb_tuple = Base.setindex(Tuple(I), inew, d)
                    nb_c = c2c[CartesianIndex{D}(nb_tuple)]
                    phi_n = T(phi_f[nb_c][1])
                else
                    # Off-patch: parent lookup.
                    par_c = pcell[c, face_idx]
                    if par_c != 0
                        phi_p = T(parent_phi[par_c][1])
                        # Martin-Colella linear ghost
                        phi_n = one_third * phi_c + two_thirds * phi_p
                    else
                        # Outer wall (no parent). Apply BC.
                        phi_n = kinds[d] === :dirichlet ? -phi_c : phi_c
                    end
                end
                acc += (phi_n - phi_c) * invh2[d]
            end
        end
        Lphi_f[c] = (acc,)
    end
    return Lphi
end

# Read phi at a wrapped/clamped axis-d neighbor of cell `c` at grid index I.
# Returns the neighbor value, or the appropriate BC fall-back (`-phi_c` for
# Dirichlet, `phi_c` for Neumann). Specialised per D via @generated for
# allocation-free CartesianIndex arithmetic.
@inline function _neighbor_phi(phi_f, c2c::Array{Int, D},
                                I::CartesianIndex{D}, d::Int, sign::Int,
                                N::NTuple{D, Int}, kinds::NTuple{D, Symbol},
                                phi_c::T) where {D, T}
    Nd = N[d]
    i_new = I[d] + sign
    if i_new < 1
        kinds[d] === :periodic && (i_new = Nd)
        kinds[d] === :neumann && (i_new = 1)
        kinds[d] === :dirichlet && return -phi_c
    elseif i_new > Nd
        kinds[d] === :periodic && (i_new = 1)
        kinds[d] === :neumann && (i_new = Nd)
        kinds[d] === :dirichlet && return -phi_c
    end
    # Build new CartesianIndex by replacing axis d. Using `Base.setindex` on
    # an NTuple{D, Int} is allocation-free for a small D.
    new_tuple = Base.setindex(Tuple(I), i_new, d)
    nb_I = CartesianIndex{D}(new_tuple)
    nb_c = c2c[nb_I]
    return T(phi_f[nb_c][1])
end

# FAC flux fix on the COARSE side at C/F faces. Replaces the (invalid)
# coarse one-sided flux through each C/F face with the area-weighted sum
# of fine sub-face fluxes, using Martin–Colella's MC ghost. Should be
# called AFTER the regular coarse Laplacian has been computed with
# skip-C/F-faces; this routine ADDS the FAC contribution for each C/F
# face.
#
# The companion routine `apply_laplacian_root_fast!(...; skip_cf=true)`
# computes the coarse Laplacian on uncovered cells while OMITTING the
# flux contribution from any face whose other side is covered (so the
# covered coarse cells' garbage phi never enters the result).
#
# Together, they form the FAC composite operator of Almgren–Bell–Colella
# (1998), which drives the V-cycle residual to machine precision up to
# the C/F discretisation order.
function apply_fac_flux_fix!(Lphi::Vector{Vector{NamedTuple}},
                              phi::Vector{Vector{NamedTuple}},
                              ws::MGWorkspace{D, T},
                              coarse_level::Int) where {D, T}
    fine_level = coarse_level + 1
    fine_level > length(ws.ph.levels) && return Lphi
    pi_c = 1
    pi_f = 1
    dxc = ws.patch_dx[coarse_level][pi_c]
    dxf = ws.patch_dx[fine_level][pi_f]
    phi_coarse = phi[coarse_level][pi_c].phi
    phi_fine   = phi[fine_level][pi_f].phi
    Lphi_coarse = Lphi[coarse_level][pi_c].phi
    inv_table = ws.fac_fine_at_coarse_face[coarse_level][pi_c]

    two_thirds = T(2//3)

    @inbounds for c in ws.patch_leaves[coarse_level][pi_c]
        any_face = false
        for k in 1:(2 * D)
            fine_list = inv_table[c, k]
            isempty(fine_list) && continue
            any_face = true
            d = (k - 1) ÷ 2 + 1
            # Flux contribution per sub-face to L φ_c (per unit V_c):
            #   F_j_in_Lphi_c = (2/3) (φ_cf_j − φ_c) / (h_f * h_c)
            # The sub-face count is implicit in fine_list (each fine cell is
            # one sub-face). Sub-face area = h_f^{D-1}, divided by V_c =
            # prod(h_c) ⇒ per-sub-face contribution = (gradient) * (h_f^{D-1})
            # / prod(h_c) = (gradient) * 1 / (h_c[d] * prod(h_c/h_f over j!=d))
            # but since h_c/h_f = 2 (2:1), prod_{j!=d}(h_c/h_f) = 2^{D-1}.
            # Total over r^{D-1} = 2^{D-1} sub-faces ⇒ each sub-face per-unit-V
            # contribution divides by total sub-faces. Easier formulation:
            #     ΔL_c from face k = (1/h_c[d]) * (avg fine grad over sub-faces)
            #                       = (1/h_c[d]) * (1/n_sub) * Σ_j (φ_cf_j − ghost_j) / h_f[d]
            # With MC: (φ_cf_j − ghost_j) = (2/3)(φ_cf_j − φ_c).
            #     ⇒ ΔL_c = (1/h_c[d]) * (1/n_sub) * Σ_j (2/3)(φ_cf_j − φ_c) / h_f[d]
            #            = (2/3) / (h_c[d] * h_f[d] * n_sub) * Σ_j (φ_cf_j − φ_c)
            n_sub = length(fine_list)
            inv_factor = two_thirds / (dxc[d] * dxf[d] * n_sub)
            phi_c = T(phi_coarse[c][1])
            fac_acc = zero(T)
            for fcell in fine_list
                fac_acc += (T(phi_fine[fcell][1]) - phi_c)
            end
            Lphi_coarse[c] = (Lphi_coarse[c][1] + fac_acc * inv_factor,)
        end
    end
    return Lphi
end

# Fast direct-array Laplacian for the root level (no parent halos).
#   skip_covered=true → don't compute the Laplacian on cells covered by a
#     finer patch (result discarded by restrict_to_parents! anyway).
#   skip_cf=true → for an uncovered cell whose neighbor across some face
#     is covered, OMIT that face's flux contribution. `apply_fac_flux_fix!`
#     adds the correct fine-side contribution afterwards.
function apply_laplacian_root_fast!(Lphi::Vector{Vector{NamedTuple}},
                                     phi::Vector{Vector{NamedTuple}},
                                     ws::MGWorkspace{D, T};
                                     skip_covered::Bool = false,
                                     skip_cf::Bool = false) where {D, T}
    pi = 1; ℓ = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    phi_f = phi[ℓ][pi].phi
    Lphi_f = Lphi[ℓ][pi].phi
    kinds = ws.axis_kinds.kinds
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    covered = ws.covered_by_finer[ℓ][pi]
    fac_table = ws.fac_fine_at_coarse_face[ℓ][pi]
    has_fac = skip_cf && length(ws.ph.levels) > ℓ

    @inbounds for I in CartesianIndices(N)
        c = c2c[I]
        c == 0 && continue
        skip_covered && covered[c] && continue
        phi_c = T(phi_f[c][1])
        acc = zero(T)
        for d in 1:D
            face_p = 2 * (d - 1) + 2
            if !(has_fac && !isempty(fac_table[c, face_p]))
                phi_p = _neighbor_phi(phi_f, c2c, I, d, +1, N, kinds, phi_c)
                acc += (phi_p - phi_c) * invh2[d]
            end
            face_m = 2 * (d - 1) + 1
            if !(has_fac && !isempty(fac_table[c, face_m]))
                phi_m = _neighbor_phi(phi_f, c2c, I, d, -1, N, kinds, phi_c)
                acc += (phi_m - phi_c) * invh2[d]
            end
        end
        Lphi_f[c] = (acc,)
    end
    return Lphi
end

"""
    compute_residual!(r, phi, rho, ws; level_range = ws.level_range)

`r = rho - L phi`.
"""
function compute_residual!(r::Vector{Vector{NamedTuple}},
                            phi::Vector{Vector{NamedTuple}},
                            rho::Vector{Vector{NamedTuple}},
                            ws::MGWorkspace{D, T};
                            level_range::UnitRange{Int} = ws.level_range,
                            backend = default_backend()) where {D, T}
    apply_laplacian!(r, phi, ws; level_range = level_range, backend = backend)
    @inbounds for ℓ in level_range
        for pi in 1:length(r[ℓ])
            rf = r[ℓ][pi].phi
            rh = rho[ℓ][pi].rho
            for c in ws.patch_leaves[ℓ][pi]
                _set_val!(rf, c, _get_val(rh, c) - _get_val(rf, c))
            end
        end
    end
    return r
end

function residual_l2(r::Vector{Vector{NamedTuple}}, ws::MGWorkspace{D, T};
                     level_range::UnitRange{Int} = ws.level_range) where {D, T}
    s = 0.0
    @inbounds for ℓ in level_range
        for pi in 1:length(r[ℓ])
            f = r[ℓ][pi].phi
            v = prod(ws.patch_dx[ℓ][pi])
            for c in ws.patch_leaves[ℓ][pi]
                rc = Float64(_get_val(f, c))
                s += rc * rc * v
            end
        end
    end
    return sqrt(s)
end

# ----------------------------------------------------------------------------
# Red-black Gauss-Seidel smoother
# ----------------------------------------------------------------------------

struct GSCtx{D, T}
    target_colour::Int
    grid_idx::Vector{NTuple{D, Int}}
    rho_flat::Vector{T}
    axis_kinds::NTuple{D, Symbol}
end

function _gs_colour_kernel(pv::PatchView, hv::PatchHaloView, ctx::GSCtx{D, T}) where {D, T}
    c = pv.cv.index
    gi = @inbounds ctx.grid_idx[c]
    s = 0
    @inbounds for d in 1:D
        s += gi[d]
    end
    (s & 1) == ctx.target_colour || return nothing

    p_lo, p_hi = cell_physical_box(hv.frame, c)
    diag = zero(T)
    nbr_acc = zero(T)
    phi_c = T(pv[:phi][1])
    @inbounds for d in 1:D
        h = p_hi[d] - p_lo[d]
        invh2 = one(T) / (h * h)
        off_p = ntuple(j -> j == d ?  1 : 0, Val(D))
        off_m = ntuple(j -> j == d ? -1 : 0, Val(D))
        phi_p = _read_phi(hv, off_p, phi_c, ctx.axis_kinds)
        phi_m = _read_phi(hv, off_m, phi_c, ctx.axis_kinds)
        nbr_acc += (phi_p + phi_m) * invh2
        diag    += 2 * invh2
    end
    rho_c = ctx.rho_flat[c]
    pv[:phi] = ((nbr_acc - rho_c) / diag,)
    return nothing
end

# Fill ws.rho_flat[ℓ][pi] in place from rho[ℓ][pi].rho.
@inline function _refresh_rho_flat!(ws::MGWorkspace{D, T},
                                     rho::Vector{Vector{NamedTuple}},
                                     ℓ::Int) where {D, T}
    @inbounds for pi in 1:length(rho[ℓ])
        rf = rho[ℓ][pi].rho
        out = ws.rho_flat[ℓ][pi]
        for i in 1:length(rf)
            out[i] = T(rf[i][1])
        end
    end
    return nothing
end

# One full RB-GS sweep over a level (both colours). Single-patch-per-level.
function gs_sweep!(phi::Vector{Vector{NamedTuple}},
                   rho::Vector{Vector{NamedTuple}},
                   ws::MGWorkspace{D, T}, ℓ::Int;
                   n_sweeps::Int = 1,
                   skip_covered::Bool = false,
                   backend = default_backend()) where {D, T}
    _refresh_rho_flat!(ws, rho, ℓ)
    if ℓ == 1
        gs_sweep_root_fast!(phi, ws, ℓ; n_sweeps = n_sweeps,
                             skip_covered = skip_covered)
    else
        gs_sweep_fine_fast!(phi, ws, ℓ; n_sweeps = n_sweeps)
    end
    return phi
end

# Fast direct-array RB-GS for a fine (non-root) level. Same algorithm as
# `gs_sweep_root_fast!` but uses `parent_cell` to look up the parent's
# value at off-patch boundaries (with Martin–Colella correction).
function gs_sweep_fine_fast!(phi::Vector{Vector{NamedTuple}},
                              ws::MGWorkspace{D, T}, ℓ::Int;
                              n_sweeps::Int = 1) where {D, T}
    pi = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    pcell = ws.parent_cell[ℓ][pi]
    phi_f = phi[ℓ][pi].phi
    rho_flat = ws.rho_flat[ℓ][pi]
    kinds = ws.axis_kinds.kinds
    parent_phi = phi[ℓ - 1][1].phi
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    one_third = T(1//3); two_thirds = T(2//3)

    # Per-cell diagonal of the Laplacian acting on φ_c. With the MC ghost
    # `φ_ghost = (1/3) φ_c + (2/3) φ_parent`, an off-patch face contributes
    # (1/3 − 1) φ_c / h_d² = −(2/3) φ_c / h_d², so the EFFECTIVE diagonal at
    # a boundary cell DROPS by (2/3)/h_d² for each boundary face. We
    # compute it on the fly per cell — cheap.

    for _ in 1:n_sweeps
        for colour in 0:1
            @inbounds for I in CartesianIndices(N)
                s = 0
                for d in 1:D; s += I[d]; end
                (s & 1) == colour || continue
                c = c2c[I]
                c == 0 && continue

                # Sum the non-φ_c part of the stencil and compute the
                # effective diagonal.
                diag = zero(T)
                nbr_acc = zero(T)
                bc_diag_adj = zero(T)   # subtracts (1/3)/h² per MC face
                for d in 1:D
                    for (side, face_idx) in ((1, 2 * (d - 1) + 2), (-1, 2 * (d - 1) + 1))
                        inew = I[d] + side
                        if 1 <= inew <= N[d]
                            nb_tuple = Base.setindex(Tuple(I), inew, d)
                            nb_c = c2c[CartesianIndex{D}(nb_tuple)]
                            nbr_acc += T(phi_f[nb_c][1]) * invh2[d]
                            diag    += invh2[d]
                        else
                            par_c = pcell[c, face_idx]
                            if par_c != 0
                                # MC face: contributes (1/3) φ_c + (2/3) φ_p.
                                # Move (1/3) φ_c to the diagonal side.
                                nbr_acc += two_thirds * T(parent_phi[par_c][1]) * invh2[d]
                                diag    += invh2[d]
                                bc_diag_adj += one_third * invh2[d]
                            else
                                # BC ghost: Dirichlet = −φ_c → effectively the
                                # neighbor is `-φ_c`, so the diagonal gains
                                # an extra +invh2 (from -(-φ_c)).
                                if kinds[d] === :dirichlet
                                    diag += 2 * invh2[d]   # absorbs both
                                else
                                    # Neumann / Periodic-with-no-parent:
                                    # ghost = +φ_c → contributes nothing.
                                end
                            end
                        end
                    end
                end
                rho_c = rho_flat[c]
                # GS update: solve diag_eff * φ_c = nbr_acc - ρ
                # where diag_eff = diag - bc_diag_adj.
                diag_eff = diag - bc_diag_adj
                phi_f[c] = ((nbr_acc - rho_c) / diag_eff,)
            end
        end
    end
    return phi
end

# Fast direct-array RB-GS for the ROOT level (no parent halos).
# Reads stencil neighbors via `cart_to_cell` with explicit periodic /
# Dirichlet / Neumann handling on the outer faces. Bypasses
# `for_each_patch!` entirely — one allocation-free pass per sweep.
function gs_sweep_root_fast!(phi::Vector{Vector{NamedTuple}},
                              ws::MGWorkspace{D, T}, ℓ::Int;
                              n_sweeps::Int = 1,
                              skip_covered::Bool = false) where {D, T}
    @assert ℓ == 1
    pi = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    phi_f = phi[ℓ][pi].phi
    rho_flat = ws.rho_flat[ℓ][pi]
    kinds = ws.axis_kinds.kinds
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    diag = sum(d -> 2 * invh2[d], 1:D)
    covered = ws.covered_by_finer[ℓ][pi]

    for _ in 1:n_sweeps
        for colour in 0:1
            @inbounds for I in CartesianIndices(N)
                # parity test
                s = 0
                for d in 1:D; s += I[d]; end
                (s & 1) == colour || continue
                c = c2c[I]
                c == 0 && continue
                skip_covered && covered[c] && continue
                phi_c = T(phi_f[c][1])
                nbr_acc = zero(T)
                for d in 1:D
                    phi_p = _neighbor_phi(phi_f, c2c, I, d, +1, N, kinds, phi_c)
                    phi_m = _neighbor_phi(phi_f, c2c, I, d, -1, N, kinds, phi_c)
                    nbr_acc += (phi_p + phi_m) * invh2[d]
                end
                rho_c = rho_flat[c]
                phi_f[c] = ((nbr_acc - rho_c) / diag,)
            end
        end
    end
    return phi
end

# Fall-back smoother for non-root levels: uses `for_each_patch!` so that
# parent halos at the fine-patch boundary resolve correctly through
# PatchBoundaryBC.
function gs_sweep_via_orchestrator!(phi::Vector{Vector{NamedTuple}},
                                     ws::MGWorkspace{D, T}, ℓ::Int;
                                     n_sweeps::Int = 1,
                                     backend = default_backend()) where {D, T}
    parent_in = ℓ >= 2 ? phi[ℓ - 1] : nothing
    axis_kinds = ws.axis_kinds.kinds
    for _ in 1:n_sweeps
        for colour in (0, 1)
            ctx = GSCtx{D, T}(colour, ws.grid_idx[ℓ][1], ws.rho_flat[ℓ][1],
                              axis_kinds)
            for_each_patch!(_gs_colour_kernel,
                              phi[ℓ], phi[ℓ], ws.ph;
                              level = ℓ, ghost_depth = 1,
                              fields_in_parent = parent_in,
                              ctx = ctx, backend = backend)
        end
    end
    return phi
end

# ----------------------------------------------------------------------------
# Multilinear prolongation (additive: phi_fine += prolong(correction_parent))
# ----------------------------------------------------------------------------

function prolong_correction_add!(fine::NamedTuple,
                                  fine_frame::EulerianFrame{D, T},
                                  fine_grid_idx::Vector{NTuple{D, Int}},
                                  fine_N::NTuple{D, Int},
                                  fine_dx::NTuple{D, T},
                                  fine_leaves::Vector{Int},
                                  parent::NamedTuple,
                                  parent_frame::EulerianFrame{D, T},
                                  parent_grid_idx::Vector{NTuple{D, Int}},
                                  parent_N::NTuple{D, Int},
                                  parent_dx::NTuple{D, T},
                                  parent_leaves::Vector{Int}
                                  ) where {D, T}
    par_lookup = fill(0, parent_N)
    @inbounds for c in parent_leaves
        gi = parent_grid_idx[c]
        par_lookup[gi...] = c
    end
    fine_phi = fine.phi
    par_phi  = parent.phi

    # Boundary-axis info: which axes are periodic on the parent's domain.
    # We use this for wraparound in the corner loop.
    # (Read from the workspace's bcs — but we don't have a ws here; use
    # parent_frame extents to decide if a neighbor index wraps. The parent
    # patch is the full coarse domain in our single-patch-per-level design.
    # Caller passes the periodicity vector via fine_dx vs parent extent.)

    @inbounds for fc in fine_leaves
        f_lo, f_hi = cell_physical_box(fine_frame, fc)
        center = ntuple(d -> (f_lo[d] + f_hi[d]) / 2, Val(D))
        # Containing parent cell (1-based)
        gi = ntuple(d ->
            clamp(Int(floor((center[d] - parent_frame.lo[d]) / parent_dx[d])) + 1,
                   1, parent_N[d]), Val(D))
        par_center = ntuple(d -> parent_frame.lo[d] + (gi[d] - 0.5) * parent_dx[d], Val(D))
        frac = ntuple(d -> (center[d] - par_center[d]) / parent_dx[d], Val(D))
        dirn = ntuple(d -> frac[d] >= 0 ? 1 : -1, Val(D))
        absf = ntuple(d -> abs(frac[d]), Val(D))
        acc = zero(T)
        for corner_bits in 0:((1 << D) - 1)
            w = one(T)
            valid = true
            cell_gi = ntuple(d -> begin
                bit = (corner_bits >> (d - 1)) & 1
                if bit == 0
                    w *= (1 - absf[d])
                    return gi[d]
                else
                    w *= absf[d]
                    nb = gi[d] + dirn[d]
                    if nb < 1 || nb > parent_N[d]
                        # No periodicity assumed for the prolongation: use
                        # the central cell's value as a fallback (the
                        # correction is small there, and Dirichlet/Neumann
                        # BC will be re-applied in the post-smooth).
                        nb = gi[d]
                    end
                    return nb
                end
            end, Val(D))
            valid || continue
            pc = par_lookup[cell_gi...]
            pc == 0 && continue
            acc += w * T(par_phi[pc][1])
        end
        fine_phi[fc] = (fine_phi[fc][1] + acc,)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# FFT bottom solver
# ----------------------------------------------------------------------------

function fft_bottom_solve!(ws::MGWorkspace{D, T},
                            phi_view::NamedTuple, rho_view::NamedTuple) where {D, T}
    ws.fft_ok || error("fft_bottom_solve!: FFT plan not configured")
    ℓ = ws.fft_level
    leaves   = ws.patch_leaves[ℓ][1]
    grid_idx = ws.grid_idx[ℓ][1]
    N = ws.fft_N
    buf = ws.fft_buf

    fill!(buf, zero(T))
    rh = rho_view.rho
    @inbounds for c in leaves
        gi = grid_idx[c]
        buf[gi...] = T(rh[c][1])
    end

    if ws.opts.project_mean && all(==(:periodic), ws.fft_kind)
        m = sum(buf) / length(buf)
        @inbounds for I in eachindex(buf); buf[I] -= m; end
    end

    if all(==(:periodic), ws.fft_kind)
        kbuf = ws.fft_plan_fwd * buf
        @inbounds for I in CartesianIndices(kbuf)
            kbuf[I] = kbuf[I] * ws.fft_inv_eig[I]
        end
        mul!(buf, ws.fft_plan_inv, kbuf)
    else
        rbuf = ws.fft_plan_fwd * buf
        @inbounds for I in eachindex(rbuf)
            rbuf[I] *= ws.fft_inv_eig[I]
        end
        rbuf2 = ws.fft_plan_inv * rbuf
        normfac = one(T)
        for d in 1:D
            normfac *= ws.fft_kind[d] === :periodic ? N[d] : 2 * N[d]
        end
        @inbounds for I in eachindex(rbuf2)
            buf[I] = rbuf2[I] / normfac
        end
    end

    pf = phi_view.phi
    @inbounds for c in leaves
        gi = grid_idx[c]
        pf[c] = (buf[gi...],)
    end
    return phi_view
end

# ----------------------------------------------------------------------------
# CG bottom solver — Jacobi-preconditioned conjugate gradients against the
# matrix-free composite Laplacian. Operates on a SINGLE level (typically
# used as the bottom of a vcycle restricted to that level). Converges in
# O(N) iterations vs O(N²) for raw GS smoothing.
# ----------------------------------------------------------------------------

# Apply L to phi on a single level, writing result into Lphi.
function _apply_lapl_single_level!(Lphi, phi, ws::MGWorkspace{D, T}, ℓ::Int;
                                    backend = default_backend()) where {D, T}
    if ℓ == 1
        apply_laplacian_root_fast!(Lphi, phi, ws)
    else
        parent_in = phi[ℓ - 1]
        ctx = ws.axis_kinds
        for_each_patch!(_lapl_kernel,
                          Lphi[ℓ], phi[ℓ], ws.ph;
                          level = ℓ, ghost_depth = 1,
                          fields_in_parent = parent_in,
                          ctx = ctx, backend = backend)
    end
    return Lphi
end

# Diagonal of L (per cell) on a uniform patch is Σ 2/h_d². Same for every
# cell. Returns a scalar.
@inline function _lapl_diag(ws::MGWorkspace{D, T}, ℓ::Int, pi::Int) where {D, T}
    dx = ws.patch_dx[ℓ][pi]
    s = zero(T)
    @inbounds for d in 1:D
        s += 2 / (dx[d] * dx[d])
    end
    return s
end

# Jacobi-preconditioned CG bottom solver. Solves L φ = ρ on level ℓ with
# levels < ℓ frozen (their phi[ℓ-1] used as the parent halo for the
# Laplacian). Updates phi[ℓ][1] in place.
function cg_bottom_solve!(phi::Vector{Vector{NamedTuple}},
                          rho::Vector{Vector{NamedTuple}},
                          ws::MGWorkspace{D, T}, ℓ::Int;
                          tol::Float64 = 1e-10,
                          maxiter::Int = 500,
                          backend = default_backend()) where {D, T}
    pi = 1
    leaves = ws.patch_leaves[ℓ][pi]
    n = length(leaves)
    n == 0 && return phi

    diag_L = _lapl_diag(ws, ℓ, pi)
    # The discrete cell-center Laplacian L has eigenvalues ≤ 0 — i.e. L is
    # NEGATIVE semidefinite. CG requires SPD, so we solve the equivalent
    # system  M φ = b  with  M = −L,  b = −ρ.  Both sides are negated, so
    # the unknown φ is unchanged. `diag_L` is negative; `inv_diag` is the
    # corresponding Jacobi preconditioner entry for M.
    inv_diag = one(T) / abs(diag_L)

    phi_arr = phi[ℓ][pi].phi
    rho_arr = rho[ℓ][pi].rho

    # Scratch fields: we reuse residual / correction / rhs_coarse buffers
    # for r, p, Mp respectively. (Mp = M p where M = −L.)
    r_field  = ws.residual[ℓ][pi].phi
    p_field  = ws.correction[ℓ][pi].phi
    Mp_field = ws.rhs_coarse[ℓ][pi].phi

    # r₀ = b − M φ = −ρ − (−L φ) = −ρ + L φ = −(ρ − L φ).
    _apply_lapl_single_level!(ws.residual, phi, ws, ℓ; backend = backend)
    @inbounds for c in leaves
        # r_field currently holds L φ. We want r = L φ − ρ.
        r_field[c] = (r_field[c][1] - rho_arr[c][1],)
    end

    # z₀ = M⁻¹ r₀ (Jacobi).  p₀ = z₀.  rs₀ = ⟨r₀, z₀⟩.
    rs = zero(T)
    r_norm2 = zero(T)
    @inbounds for c in leaves
        rc = r_field[c][1]
        z = rc * inv_diag
        p_field[c] = (z,)
        rs += rc * z
        r_norm2 += rc * rc
    end
    r_norm0 = sqrt(r_norm2)
    r_norm0 == 0 && return phi

    iter = 0
    while iter < maxiter
        # Mp = M p = −L p.
        _apply_lapl_single_level!(ws.rhs_coarse, ws.correction, ws, ℓ; backend = backend)
        @inbounds for c in leaves
            Mp_field[c] = (-Mp_field[c][1],)
        end

        pMp = zero(T)
        @inbounds for c in leaves
            pMp += p_field[c][1] * Mp_field[c][1]
        end
        pMp == 0 && break
        α = rs / pMp

        # φ += α p ;  r -= α Mp.
        new_r_norm2 = zero(T)
        @inbounds for c in leaves
            phi_arr[c] = (phi_arr[c][1] + α * p_field[c][1],)
            r_field[c] = (r_field[c][1] - α * Mp_field[c][1],)
            new_r_norm2 += r_field[c][1] * r_field[c][1]
        end
        iter += 1
        if sqrt(new_r_norm2) <= tol * r_norm0
            break
        end

        # β = ⟨r_new, z_new⟩ / rs ;  p ← z_new + β p.
        rs_new = zero(T)
        @inbounds for c in leaves
            rs_new += r_field[c][1] * (r_field[c][1] * inv_diag)
        end
        β = rs_new / rs
        @inbounds for c in leaves
            z = r_field[c][1] * inv_diag
            p_field[c] = (z + β * p_field[c][1],)
        end
        rs = rs_new
    end
    return phi
end

# ----------------------------------------------------------------------------
# Schur bottom solver — direct dense Schur-complement elimination on a
# fine patch with Dirichlet-from-parent boundary. Opt-in via
# `MGOptions.bottom_solver = :schur`. Single fine-patch solves only.
#
# Block structure on the fine patch:
#     [ A_II  A_IΓ ] [φ_I]   [ b_I ]
#     [ A_ΓI  A_ΓΓ ] [φ_Γ] = [ b_Γ ]
#
# where I = cells fully interior to the fine patch (all stencil neighbors
# in-patch) and Γ = cells with at least one off-patch stencil neighbor
# (whose ghost reads the parent). We work with the SPD matrix
# M = −L (positive definite for Dirichlet-from-parent), solve M φ = −ρ.
#
# Direct elimination:
#   1. Compute z_I = A_II⁻¹ b_I (one Cholesky solve).
#   2. b̃_Γ = b_Γ − A_ΓI z_I.
#   3. S = A_ΓΓ − A_ΓI A_II⁻¹ A_IΓ (built explicitly, |Γ| Cholesky solves).
#   4. Solve S φ_Γ = b̃_Γ (dense LU).
#   5. φ_I = z_I − A_II⁻¹ A_IΓ φ_Γ.
#
# v1 uses degree-0 Dirichlet (φ_ghost = φ_parent at the patch wall);
# Martin–Colella's linear ghost couples φ_ghost = (1/3) φ_cf + (2/3) φ_parent
# back to φ_cf, which would modify A_ΓΓ's diagonal. Deferred to v2.
# ----------------------------------------------------------------------------

mutable struct SchurFactor{T}
    # Cell partitioning
    I_cells::Vector{Int}            # fine-patch cell index of each I row
    Γ_cells::Vector{Int}            # fine-patch cell index of each Γ row
    I_pos::Vector{Int}              # I_pos[c] = row in A_II if c ∈ I, else 0
    Γ_pos::Vector{Int}              # Γ_pos[c] = row in A_ΓΓ if c ∈ Γ, else 0
    # Operator blocks
    A_II_chol::Any                  # cholesky of A_II
    A_IΓ::SparseMatrixCSC{T, Int}
    A_ΓI::SparseMatrixCSC{T, Int}
    A_ΓΓ::SparseMatrixCSC{T, Int}
    S_lu::Any                       # LU factor of dense Schur complement
    # Parent-side contributions to the RHS at each Γ cell (Dirichlet flux).
    # Built fresh each solve from phi_parent — but the pattern (which Γ
    # cell touches which parent cell on which face, with what coefficient)
    # is cached.
    ghost_links::Vector{NTuple{3, Int}}   # (Γ_pos, parent_cell, face_idx)
    invh2::NTuple{1, T}             # 1/h² along each axis (we cache only
                                    # one entry for D=2/3 — see usage)
end

# Default constructor placeholder so MGWorkspace can hold one slot per
# fine patch. The actual factor is built on-demand by `_build_schur_factor`.
SchurFactor{T}() where {T} =
    SchurFactor{T}(Int[], Int[], Int[], Int[],
                    nothing, spzeros(T, 0, 0), spzeros(T, 0, 0),
                    spzeros(T, 0, 0), nothing,
                    NTuple{3, Int}[], (zero(T),))

# Build the Schur factorisation for a single fine patch (single-patch-per-
# level assumption). Reuses workspace `cart_to_cell`, `parent_cell`, and
# `axis_kinds`. O(N_fine + |Γ| × A_II_solve) up-front cost — amortised
# across many solves with the same operator.
function _build_schur_factor(ws::MGWorkspace{D, T}, ℓ::Int) where {D, T}
    pi = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    pcell = ws.parent_cell[ℓ][pi]
    leaves = ws.patch_leaves[ℓ][pi]
    n = length(leaves)
    # Uniform spacing assumed.
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    diag_M = sum(d -> 2 * invh2[d], 1:D)   # diagonal of M = -L

    # Partition: a cell is in I iff every face has a same-patch neighbor.
    # Otherwise Γ. Sized to the maximum fine-cell index so we can index by
    # the (sparse, DFS-order) cell id directly.
    max_cell = maximum(leaves)
    I_pos = zeros(Int, max_cell)
    Γ_pos = zeros(Int, max_cell)
    I_cells = Int[]
    Γ_cells = Int[]
    @inbounds for c in leaves
        is_boundary = false
        for k in 1:(2 * D)
            if pcell[c, k] != 0
                is_boundary = true; break
            end
        end
        if is_boundary
            push!(Γ_cells, c)
            Γ_pos[c] = length(Γ_cells)
        else
            push!(I_cells, c)
            I_pos[c] = length(I_cells)
        end
    end

    nI = length(I_cells); nΓ = length(Γ_cells)

    # Build sparse A_II, A_IΓ, A_ΓΓ, A_ΓI, and ghost_links.
    # Triplet-form builders. The diagonal of each cell ABSORBS its MC face
    # contributions (each off-patch face with MC ghost adds +invh2/3 to
    # the M-matrix diagonal, since MC ghost = (1/3) φ_cf + (2/3) φ_p
    # injects (1/3) φ_cf back into the stencil).
    II_I = Int[]; II_J = Int[]; II_V = T[]
    IG_I = Int[]; IG_J = Int[]; IG_V = T[]
    GI_I = Int[]; GI_J = Int[]; GI_V = T[]
    GG_I = Int[]; GG_J = Int[]; GG_V = T[]
    ghost_links = NTuple{3, Int}[]
    one_third = T(1//3); two_thirds = T(2//3)

    @inbounds for c in leaves
        Ic = ws.grid_idx[ℓ][pi][c]
        pos_c = I_pos[c]
        in_I = pos_c != 0
        row_c = in_I ? pos_c : Γ_pos[c]

        # Walk faces to compute the diagonal and off-diagonal contributions.
        diag = zero(T)
        for d in 1:D, side in (-1, 1)
            face_idx = 2 * (d - 1) + (side == 1 ? 2 : 1)
            inew = Ic[d] + side
            if 1 <= inew <= N[d]
                # Same-patch neighbor: standard M-matrix off-diagonal.
                nb_tuple = Base.setindex(Tuple(CartesianIndex(Ic)), inew, d)
                nb_c = c2c[CartesianIndex{D}(nb_tuple)]
                pos_nb_I = I_pos[nb_c]
                pos_nb_Γ = Γ_pos[nb_c]
                coeff = -invh2[d]
                if in_I
                    if pos_nb_I != 0
                        push!(II_I, row_c); push!(II_J, pos_nb_I); push!(II_V, coeff)
                    else
                        push!(IG_I, row_c); push!(IG_J, pos_nb_Γ); push!(IG_V, coeff)
                    end
                else
                    if pos_nb_I != 0
                        push!(GI_I, row_c); push!(GI_J, pos_nb_I); push!(GI_V, coeff)
                    else
                        push!(GG_I, row_c); push!(GG_J, pos_nb_Γ); push!(GG_V, coeff)
                    end
                end
                # Diagonal +invh2 from this face (standard).
                diag += invh2[d]
            else
                # Off-patch face. MC ghost: φ_ghost = (1/3) φ_cf + (2/3) φ_p.
                # In M = -L form, the diagonal contribution from this face
                # is +invh2 (from cf cell self-coupling via -L) MINUS the
                # (1/3) coupling back to φ_cf, i.e., effectively
                #   diag += (1 - 1/3) * invh2 = (2/3) * invh2
                # and the parent contributes (2/3) * invh2 * φ_p to the rhs.
                par_c = pcell[c, face_idx]
                if par_c != 0
                    diag += two_thirds * invh2[d]
                    push!(ghost_links, (row_c, par_c, d))
                else
                    # No parent: outer-wall BC. Defer to Dirichlet (-φ_c):
                    # diag += 2 * invh2.
                    diag += 2 * invh2[d]
                end
            end
        end
        if in_I
            push!(II_I, row_c); push!(II_J, row_c); push!(II_V, diag)
        else
            push!(GG_I, row_c); push!(GG_J, row_c); push!(GG_V, diag)
        end
    end

    A_II = sparse(II_I, II_J, II_V, nI, nI)
    A_IΓ = sparse(IG_I, IG_J, IG_V, nI, nΓ)
    A_ΓI = sparse(GI_I, GI_J, GI_V, nΓ, nI)
    A_ΓΓ = sparse(GG_I, GG_J, GG_V, nΓ, nΓ)

    # Factor A_II.
    A_II_chol = cholesky(A_II)

    # Build dense S = A_ΓΓ − A_ΓI A_II⁻¹ A_IΓ explicitly. For each column
    # j of A_IΓ, solve A_II * x = A_IΓ[:, j], then S[:, j] = A_ΓΓ[:, j]
    # − A_ΓI * x.
    S_dense = Matrix{T}(A_ΓΓ)
    A_IΓ_dense = Matrix{T}(A_IΓ)
    for j in 1:nΓ
        col = view(A_IΓ_dense, :, j)
        x = A_II_chol \ Vector{T}(col)
        # subtract A_ΓI * x from S[:, j]
        ax = A_ΓI * x
        S_dense[:, j] .-= ax
    end

    S_lu = lu!(S_dense)

    return SchurFactor{T}(I_cells, Γ_cells, I_pos, Γ_pos,
                           A_II_chol, A_IΓ, A_ΓI, A_ΓΓ, S_lu,
                           ghost_links, (invh2[1],))
end

# Solve `M φ = −ρ` on the fine patch using the precomputed Schur factor.
# Updates phi[ℓ][pi].phi in place.
function schur_bottom_solve!(phi::Vector{Vector{NamedTuple}},
                              rho::Vector{Vector{NamedTuple}},
                              ws::MGWorkspace{D, T},
                              ℓ::Int,
                              factor::SchurFactor{T}) where {D, T}
    pi = 1
    phi_arr = phi[ℓ][pi].phi
    rho_arr = rho[ℓ][pi].rho
    parent_phi = phi[ℓ - 1][1].phi
    dx = ws.patch_dx[ℓ][pi]
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))

    nI = length(factor.I_cells)
    nΓ = length(factor.Γ_cells)

    # Build b_I and b_Γ. b = −ρ (sign flip for M = −L).
    b_I = Vector{T}(undef, nI)
    b_Γ = Vector{T}(undef, nΓ)
    @inbounds for (i, c) in enumerate(factor.I_cells)
        b_I[i] = -T(rho_arr[c][1])
    end
    @inbounds for (i, c) in enumerate(factor.Γ_cells)
        b_Γ[i] = -T(rho_arr[c][1])
    end
    # Add ghost contributions to b_Γ. With MC ghost = (1/3) φ_cf + (2/3) φ_p,
    # the parent term moves to the rhs as +(2/3) invh2 · φ_p (sign-flipped
    # from L → M, with the (1/3) part absorbed into the matrix diagonal).
    two_thirds = T(2//3)
    @inbounds for (row, par_c, d) in factor.ghost_links
        b_Γ[row] += two_thirds * invh2[d] * T(parent_phi[par_c][1])
    end

    # z_I = A_II⁻¹ b_I
    z_I = factor.A_II_chol \ b_I
    # b̃_Γ = b_Γ − A_ΓI z_I
    bt_Γ = b_Γ - factor.A_ΓI * z_I
    # φ_Γ = S⁻¹ b̃_Γ
    φ_Γ = factor.S_lu \ bt_Γ
    # φ_I = z_I − A_II⁻¹ (A_IΓ φ_Γ)
    A_IΓ_φ_Γ = factor.A_IΓ * φ_Γ
    correction = factor.A_II_chol \ A_IΓ_φ_Γ
    φ_I = z_I - correction

    # Write back into phi.
    @inbounds for (i, c) in enumerate(factor.I_cells)
        phi_arr[c] = (φ_I[i],)
    end
    @inbounds for (i, c) in enumerate(factor.Γ_cells)
        phi_arr[c] = (φ_Γ[i],)
    end
    return phi
end

# ----------------------------------------------------------------------------
# V-cycle
# ----------------------------------------------------------------------------

"""
    vcycle!(phi, rho, ws; level_range = ws.level_range)

One FAC-style geometric V-cycle on `level_range`. Updates phi at every
active level. At each level the smoother is RB-GS; the coarsest level is
solved by the FFT bottom solver if it was configured, otherwise by many
GS sweeps.
"""
function vcycle!(phi::Vector{Vector{NamedTuple}},
                 rho::Vector{Vector{NamedTuple}},
                 ws::MGWorkspace{D, T};
                 level_range::UnitRange{Int} = ws.level_range,
                 backend = Sequential()) where {D, T}
    ℓ_hi = last(level_range)
    ℓ_lo = first(level_range)

    if ℓ_hi == ℓ_lo
        # Coarsest level of this V-cycle: direct solve via FFT (preferred,
        # exact for the discrete operator on root), Schur-complement direct
        # solve (exact for the fine-patch discrete operator with Dirichlet-
        # from-parent), Jacobi-PCG (matrix-free, no setup), or raw GS sweeps
        # (fallback / debugging). The phi at this level is the correction
        # we solve for; rho is the (restricted) residual.
        if ws.fft_ok && ℓ_hi == ws.fft_level
            fft_bottom_solve!(ws, phi[ℓ_hi][1], rho[ℓ_hi][1])
        elseif ws.opts.bottom_solver === :schur && ℓ_hi >= 2
            factor = get!(ws.schur_factors, ℓ_hi) do
                _build_schur_factor(ws, ℓ_hi)
            end
            schur_bottom_solve!(phi, rho, ws, ℓ_hi, factor)
        elseif ws.opts.bottom_solver === :cg
            cg_bottom_solve!(phi, rho, ws, ℓ_hi;
                              tol = ws.opts.bottom_cg_tol,
                              maxiter = ws.opts.bottom_cg_maxiter,
                              backend = backend)
        else
            gs_sweep!(phi, rho, ws, ℓ_hi;
                       n_sweeps = ws.opts.bottom_smooth_iters, backend = backend)
        end
        return phi
    end

    # Pre-smooth fine level (level ℓ_hi).
    gs_sweep!(phi, rho, ws, ℓ_hi; n_sweeps = ws.opts.n_pre, backend = backend)

    # Compute residuals on level ℓ_hi AND ℓ_hi-1. On level ℓ_hi-1, skip
    # coarse cells covered by a finer patch — the restrict_to_parents!
    # call below overwrites them with the volume-averaged fine residual,
    # so computing the Laplacian on them is wasted work. ALSO apply the
    # FAC flux fix on uncovered C/F-adjacent cells so the composite
    # operator matches the fine-side flux (this is what stops the V-cycle
    # from stagnating at the C/F interface error level).
    apply_laplacian!(ws.residual, phi, ws;
                      level_range = ℓ_hi:ℓ_hi, backend = backend)
    # Coarse-level Laplacian: skip covered cells AND omit C/F-face flux
    # contributions on uncovered cells (those faces are wrong because the
    # covered coarse phi is garbage). The FAC fix below adds the correct
    # fine-side flux through each C/F face.
    apply_laplacian!(ws.residual, phi, ws;
                      level_range = (ℓ_hi - 1):(ℓ_hi - 1),
                      skip_covered = true, skip_cf = true,
                      backend = backend)
    apply_fac_flux_fix!(ws.residual, phi, ws, ℓ_hi - 1)
    @inbounds for ℓ in (ℓ_hi - 1):ℓ_hi
        for pi in 1:length(ws.residual[ℓ])
            rf = ws.residual[ℓ][pi].phi
            rh = rho[ℓ][pi].rho
            covered = ws.covered_by_finer[ℓ][pi]
            for c in ws.patch_leaves[ℓ][pi]
                # On level ℓ_hi-1, leave covered cells' residual as-is —
                # restrict_to_parents! overwrites them. Without this skip
                # we'd compute `rh[c] - L_skipped[c]` where L_skipped is
                # uninitialised on covered cells.
                if ℓ == ℓ_hi - 1 && covered[c]
                    continue
                end
                _set_val!(rf, c, _get_val(rh, c) - _get_val(rf, c))
            end
        end
    end

    # Build the right-hand side for the coarser recursive call:
    # rhs_coarse[ℓ_hi-1] = residual[ℓ_hi-1] everywhere, with COVERED coarse
    # cells overwritten by the volume-weighted average of residual[ℓ_hi].
    @inbounds for pi in 1:length(ws.rhs_coarse[ℓ_hi - 1])
        src = ws.residual[ℓ_hi - 1][pi].phi
        dst = ws.rhs_coarse[ℓ_hi - 1][pi].rho
        # Also stash a copy into rhs_coarse.phi for restrict_to_parents! to overwrite.
        dst_phi = ws.rhs_coarse[ℓ_hi - 1][pi].phi
        for c in ws.patch_leaves[ℓ_hi - 1][pi]
            v = _get_val(src, c)
            _set_val!(dst, c, v)
            _set_val!(dst_phi, c, v)
        end
    end
    # Restrict fine residual onto rhs_coarse[:phi]; the covered cells get
    # overwritten with the volume average. (Uncovered cells are left alone
    # by restrict_to_parents! — they retain their level-(ℓ_hi-1) residual.)
    restrict_to_parents!(ws.rhs_coarse[ℓ_hi - 1],
                          ws.residual[ℓ_hi],
                          ws.ph; level = ℓ_hi, fieldname = :phi)
    # Copy :phi → :rho for the recursive call's "rho" slot.
    @inbounds for pi in 1:length(ws.rhs_coarse[ℓ_hi - 1])
        src = ws.rhs_coarse[ℓ_hi - 1][pi].phi
        dst = ws.rhs_coarse[ℓ_hi - 1][pi].rho
        for c in ws.patch_leaves[ℓ_hi - 1][pi]
            _set_val!(dst, c, _get_val(src, c))
        end
    end

    # Zero correction[lo..hi-1].phi (correction-form recursive call).
    @inbounds for ℓ in ℓ_lo:(ℓ_hi - 1)
        for pi in 1:length(ws.correction[ℓ])
            f = ws.correction[ℓ][pi].phi
            for c in ws.patch_leaves[ℓ][pi]
                _set_val!(f, c, zero(T))
            end
        end
    end

    # Recurse to compute correction[lo..hi-1] from rhs_coarse[lo..hi-1].
    # NB: the recursive call sees phi=correction, rho=rhs_coarse, so the
    # rhs_coarse.rho field carries the restricted residual.
    vcycle!(ws.correction, ws.rhs_coarse, ws;
             level_range = ℓ_lo:(ℓ_hi - 1), backend = backend)

    # Apply correction: φ[ℓ_hi-1] += correction[ℓ_hi-1]; φ[ℓ_hi] += P(correction).
    @inbounds for pi in 1:length(phi[ℓ_hi - 1])
        pf = phi[ℓ_hi - 1][pi].phi
        cf = ws.correction[ℓ_hi - 1][pi].phi
        for c in ws.patch_leaves[ℓ_hi - 1][pi]
            _set_val!(pf, c, _get_val(pf, c) + _get_val(cf, c))
        end
    end
    @inbounds for pi in 1:length(phi[ℓ_hi])
        prolong_correction_add!(phi[ℓ_hi][pi],
                                  patches_at(ws.ph, ℓ_hi)[pi],
                                  ws.grid_idx[ℓ_hi][pi],
                                  ws.patch_N[ℓ_hi][pi],
                                  ws.patch_dx[ℓ_hi][pi],
                                  ws.patch_leaves[ℓ_hi][pi],
                                  ws.correction[ℓ_hi - 1][1],
                                  patches_at(ws.ph, ℓ_hi - 1)[1],
                                  ws.grid_idx[ℓ_hi - 1][1],
                                  ws.patch_N[ℓ_hi - 1][1],
                                  ws.patch_dx[ℓ_hi - 1][1],
                                  ws.patch_leaves[ℓ_hi - 1][1])
    end

    # Post-smooth on the fine level.
    gs_sweep!(phi, rho, ws, ℓ_hi; n_sweeps = ws.opts.n_post, backend = backend)
    return phi
end

# ----------------------------------------------------------------------------
# solve_poisson!
# ----------------------------------------------------------------------------

"""
    solve_poisson!(phi, rho, ws; level_range = ws.level_range,
                                  backend = Sequential())

Solve L φ = ρ on `level_range`. Returns an `MGResult`.
"""
function solve_poisson!(phi::Vector{Vector{NamedTuple}},
                        rho::Vector{Vector{NamedTuple}},
                        ws::MGWorkspace{D, T};
                        level_range::UnitRange{Int} = ws.level_range,
                        backend = Sequential()) where {D, T}
    opts = ws.opts
    history = Float64[]
    compute_residual!(ws.residual, phi, rho, ws;
                       level_range = level_range, backend = backend)
    r0 = residual_l2(ws.residual, ws; level_range = level_range)
    push!(history, r0)

    if r0 == 0
        return MGResult(0, r0, r0, true, history)
    end

    r = r0
    iter = 0
    converged = false
    while iter < opts.maxiter
        vcycle!(phi, rho, ws; level_range = level_range, backend = backend)
        iter += 1
        compute_residual!(ws.residual, phi, rho, ws;
                           level_range = level_range, backend = backend)
        r = residual_l2(ws.residual, ws; level_range = level_range)
        push!(history, r)
        opts.verbose && @info "MG cycle $iter: residual = $r  (reduction = $(r/r0))"
        if r <= opts.tol * r0 || r <= opts.tol
            converged = true; break
        end
    end
    return MGResult(iter, r0, r, converged, history)
end

# ----------------------------------------------------------------------------
# Hierarchy construction helpers
# ----------------------------------------------------------------------------

function _uniform_mesh(::Val{D}, refines::Int) where {D}
    mesh = HierarchicalMesh{D}()
    for _ in 1:refines
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    return mesh
end

"""
    build_uniform_root_hierarchy(::Val{D}, refines, lo, hi;
                                  physical_bcs = nothing) -> PatchHierarchy

Build a PatchHierarchy with a single uniformly-refined root patch of size
`(2^refines)^D` on the box `[lo, hi]`.
"""
function build_uniform_root_hierarchy(::Val{D}, refines::Int,
                                       lo::NTuple{D, T},
                                       hi::NTuple{D, T};
                                       physical_bcs = nothing) where {D, T}
    mesh = _uniform_mesh(Val(D), refines)
    base = EulerianFrame(mesh, lo, hi)
    return PatchHierarchy(base; physical_bcs = physical_bcs)
end

"""
    manufactured_rhs!(rho_field_view, frame, leaves, f)

Project f(x::NTuple{D,T}) → T into rho_field_view.rho by cell-center
evaluation. Returns the field view.
"""
function manufactured_rhs!(fields_view::NamedTuple,
                            frame::EulerianFrame{D, T},
                            leaves::Vector{Int}, f) where {D, T}
    rf = fields_view.rho
    @inbounds for c in leaves
        lo, hi = cell_physical_box(frame, c)
        center = ntuple(d -> (lo[d] + hi[d]) / 2, Val(D))
        rf[c] = (T(f(center)),)
    end
    return fields_view
end

end # module GeometricMultigrid
