# ============================================================================
# GeometricMultigrid — Poisson solver on a PatchHierarchy
# ============================================================================

"""
    GeometricMultigrid

A composite cell-centered finite-volume Poisson solver built on top of
`HierarchicalGrids.Solver.PatchHierarchy`. Supports periodic, Dirichlet,
and Neumann boundary conditions on a tile-uniform root level, with
2:1-refined fine patches on top.

# Quick start

```julia
using HierarchicalGrids
using HierarchicalGrids: PERIODIC
using HierarchicalGrids.Overlap: FrameBoundaries

# Build a 64² periodic root grid.
bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
bcs = FrameBoundaries(bcs_spec)
ph = build_uniform_root_hierarchy(Val(2), 6, (0.0, 0.0), (1.0, 1.0);
                                   physical_bcs = bcs)

# Allocate (phi, rho) and project a manufactured source.
fields = allocate_phi_rho(ph)
fill_field!(fields, ph, :rho, x -> -8π^2 * sin(2π*x[1]) * sin(2π*x[2]))

# Build a workspace and solve.
ws = MGWorkspace(ph, bcs_spec; opts = MGOptions(tol = 1e-10))
result = solve_poisson!(ws, fields)        # convenience form
# or: result = solve_poisson!(fields, fields, ws)

@info "solved: \$result"                    # MGResult(converged in N V-cycles, ...)
```

# Solver paths

The bottom solver inside the V-cycle dispatches based on which paths are
applicable:

  * **FFT direct solve** is used at `level_range[1]` when that level is a
    single tile-uniform patch covering the domain. Supports periodic
    (`rfft`), Dirichlet (DST-II), and Neumann (DCT-II). 1 V-cycle to
    machine precision.
  * **Schur-complement direct solve** is used for single-level fine-patch
    solves when `bottom_solver = :schur`. Builds sparse A_II / A_IΓ /
    A_ΓΓ blocks once, factors A_II by Cholesky, forms a dense Schur
    complement S, factors S by LU. Reuses the factor across solves with
    the same operator. Best for subcycling.
  * **Jacobi-PCG** (`bottom_solver = :cg`, default) for non-FFT,
    non-Schur bottoms — matrix-free, O(N) iterations.
  * **Raw Gauss-Seidel** (`bottom_solver = :gs`) as a debugging fallback.

The level-2+ smoother in the V-cycle goes through a fast direct-array
path that uses pre-cached `cart_to_cell` and `parent_cell` tables to
avoid the O(n_parent) scan in `PatchHaloView`. Martin–Colella linear
ghost at C/F interfaces gives 2nd-order accuracy at the fine boundary;
the coarse-side FAC flux fix gives a consistent composite operator.

# Known limitations and behavior

  * **AMR V-cycle reports ~1% relative residual on uncovered cells.** The
    fine-level (level-2) residual reaches machine precision, but the
    level-1 residual on UNCOVERED cells settles at a stable fixed point
    around 1% of the initial. This is the prolongation-coupling limit of
    the V-cycle: the bottom solver solves L_FAC·δ[1] = r assuming
    δ[2] = 0, but after prolongation δ[2] = P(δ[1]) is non-zero, so the
    next-iteration residual is r - L_FAC·δ with a small but persistent
    L_FAC-via-fine-cells coupling term. This is INTRINSIC to V-cycle FAC
    structure; addressing it requires F-cycle / AFAC / composite Krylov.
    `MGOptions.matching_filter_sweeps > 0` applies additional FAC-aware
    smoothing to level 1 (using the current fine phi) as a partial fix;
    the marginal residual reduction usually isn't worth the cost since
    the solution accuracy is already discretization-limited.
  * **Residual L2 norm excludes covered cells by default**
    (`skip_covered = true` in `residual_l2`). Covered coarse cells are
    placeholders for the volume-average of fine cells; their "residual"
    against the periodic Laplacian is not physically meaningful. With
    the old (all-cells) norm the reported residual was ~3× the actual
    uncovered residual, masking the true convergence.
  * **One patch per level.** Multi-patch root or multi-patch fine levels
    are rejected. FFT bottom requires a single domain-covering patch.
  * **Degree-0 polynomial fields only.** The cell-centered FV operator
    treats the stored coefficient as a cell average.
  * **No threading** of the fast Cartesian paths yet — only the
    orchestrator path (level 2+ smoother / Laplacian) parallelises via
    OhMyThreads.

# Public API

  * `build_uniform_root_hierarchy(::Val{D}, refines, lo, hi; physical_bcs)`
  * `allocate_phi_rho(ph)` / `fill_field!(fields, ph, name, f)`
  * `MGOptions(; kwargs...)`
  * `MGWorkspace(ph, bcs_spec; level_range, opts)`
  * `solve_poisson!(ws, fields)` or `solve_poisson!(phi, rho, ws)`
  * `vcycle!(phi, rho, ws; level_range)`
  * `apply_laplacian!(Lphi, phi, ws; level_range)`
  * `compute_residual!(r, phi, rho, ws; level_range)`
  * `residual_l2(r, ws; level_range)`
  * `release!(ws)` — explicit teardown of the refinement listener.

See also: `HierarchicalGrids.Solver.PatchHierarchy`, `for_each_patch!`.
"""
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
using Base.Threads: @threads, nthreads

export MGWorkspace, MGOptions, MGResult, PhiRhoFields,
       solve_poisson!, vcycle!,
       apply_laplacian!, compute_residual!, residual_l2,
       allocate_phi_rho, build_uniform_root_hierarchy,
       manufactured_rhs!, fill_field!, release!,
       # ABec / variable-coefficient operator API
       ABecCoefs, allocate_abec_coefs, fill_abec_alpha!, fill_abec_beta!,
       apply_abec!, gs_sweep_abec!, compute_abec_residual!,
       vcycle_abec!, pcg_composite_abec_solve!, solve_abec!,
       # MAC projector
       FaceVelocity, allocate_face_velocity, fill_face_velocity!,
       face_divergence!, face_divergence_l2, mac_project!,
       # Krylov.jl bridge
       FlatLayout, flat_layout, pack!, unpack!,
       FACCompositeOp, ABecOp, solve_with_krylov!, abec_jacobi_precond

# Type alias for the (phi, rho) field container returned by
# `allocate_phi_rho`. The outer Vector is per-level; the inner per-patch.
# Loose on the concrete `PolynomialFieldView` parameters to avoid coupling
# to the Storage submodule's internal types — any NamedTuple with the
# right symbol-shape suffices for the solver's read/write paths.
const PhiRhoFields = Vector{Vector{NamedTuple}}

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

# Raw-storage accessors for the degree-0 SoA layout. `phi_view.phi` is a
# `PolynomialFieldView`; the underlying contiguous Vector{T} lives at
# `view.phi.pfs.storage.phi`. Accessing it directly bypasses the
# PolynomialView struct that `phi_view.phi[i]` would build per cell.
@inline _raw_phi(v::NamedTuple) = v.phi.pfs.storage.phi
@inline _raw_rho(v::NamedTuple) = v.rho.pfs.storage.rho

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

    # Matching filter: after each V-cycle (when AMR is active), run
    # `matching_filter_sweeps` additional FAC-aware GS sweeps on the
    # uncovered cells of the coarse level. These sweeps read the CURRENT
    # fine-level phi (not the correction) for the FAC coupling, which is
    # what's needed to satisfy L_FAC φ = ρ at C/F-adjacent uncovered cells.
    # Without this, the coarse-level uncovered residual stalls at the
    # prolongation-coupling limit (~1% relative). 0 disables.
    matching_filter_sweeps::Int = 0

    # PCG preconditioner choice for `cycle = :pcg`.
    #   :jacobi  Diagonal-scaling. Always SPD. Default.
    #   :none    Identity. Same convergence as :jacobi for uniform grids,
    #            but converges slower on stretched / variable-coefficient
    #            problems.
    #   :vcycle  One V-cycle per PCG iteration. Fewer outer iterations IF
    #            the V-cycle is SPD — which requires SYMMETRIC smoothing
    #            (e.g., symmetric red-black GS doing both RBR and BRB).
    #            Our current RB-GS is asymmetric, so :vcycle is currently
    #            non-convergent on AMR. Use :jacobi until the smoother is
    #            symmetrized (TODO).
    pcg_precond::Symbol = :jacobi

    # Bottom-solver choice for single-level recursive calls and for the
    # AMR-V-cycle bottom level. Options:
    #   :cg      Jacobi-preconditioned CG against the matrix-free Laplacian.
    #            O(N) iterations vs O(N²) for raw GS. Used for non-FFT
    #            single-level solves (e.g. fine-patch subcycling).
    #   :schur   Direct dense Schur-complement solve. Exact for the
    #            discrete operator with Dirichlet-from-parent ghost.
    #   :gs      Raw Gauss-Seidel sweeping. Combined with `:mg_only` it
    #            becomes the FAC-aware bottom relaxer.
    #   :mg_only Use the FAC-aware GS smoother as the bottom (no FFT).
    #            In AMR mode this is the path that drives the residual
    #            to machine precision because both the smoother and the
    #            residual use the same FAC operator.
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

function Base.show(io::IO, r::MGResult)
    red = r.res_init == 0 ? 0.0 : r.res_final / r.res_init
    status = r.converged ? "converged" : "STOPPED (maxiter)"
    print(io, "MGResult($status in $(r.iters) V-cycles, ",
          "|r|: $(round(r.res_init, sigdigits=3)) → ",
          "$(round(r.res_final, sigdigits=3)), ",
          "reduction $(round(red, sigdigits=3)))")
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
    # Scratch buffer for the r2r path (in-place transforms).
    fft_r2r_buf::Array{T, D}

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
    fft_r2r_buf = Array{T, D}(undef, ntuple(_ -> 1, Val(D)))
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
                fft_r2r_buf = Array{T, D}(undef, N)
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
                            fft_plan_fwd, fft_plan_inv, fft_r2r_buf,
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

function Base.show(io::IO, ws::MGWorkspace{D, T}) where {D, T}
    nL = length(ws.ph.levels)
    bottom = ws.fft_ok ? "FFT($(join(ws.fft_kind, "/")))" :
              ws.opts.bottom_solver === :schur ? "Schur(direct)" :
              ws.opts.bottom_solver === :cg    ? "Jacobi-PCG"    : "GS"
    print(io, "MGWorkspace{$D,$T}(",
          "$(nL) levels, range $(ws.level_range), ",
          "bottom = $bottom)")
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
            apply_laplacian_fine_fast!(Lphi, phi, ws, ℓ;
                                         skip_covered = skip_covered,
                                         skip_cf = skip_cf)
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
# Fine-level cell stencil: returns (nbr_acc, diag_eff) decomposition of
# the FAC operator at cell c. The Laplacian and the GS update both consume
# this:
#   L φ_c    = nbr_acc − diag_eff · φ_c
#   φ_c_new  = (nbr_acc − ρ_c) / diag_eff       (GS update)
@inline function _fine_stencil_sums(I::CartesianIndex{D},
                                      c::Int, c2c::Array{Int, D},
                                      phi_raw::Vector{T},
                                      parent_phi_raw::Vector{T},
                                      pcell::Matrix{Int},
                                      kinds::NTuple{D, Symbol},
                                      invh2::NTuple{D, T},
                                      N::NTuple{D, Int},
                                      phi_c::T) where {D, T}
    nbr_acc = zero(T)
    diag_eff = zero(T)
    two_thirds = T(2//3)
    @inbounds for d in 1:D
        for (side, face_idx) in ((1, 2 * (d - 1) + 2), (-1, 2 * (d - 1) + 1))
            inew = I[d] + side
            if 1 <= inew <= N[d]
                # Same-patch neighbor
                nb_c = c2c[I + side * _unit_offsets(Val(D))[d]]
                nbr_acc += phi_raw[nb_c] * invh2[d]
                diag_eff += invh2[d]
            else
                par_c = pcell[c, face_idx]
                if par_c != 0
                    # MC ghost: phi_n = (1/3) φ_c + (2/3) φ_p
                    # Per-face contribution: (2/3) φ_p / h² to nbr_acc,
                    # (2/3) / h² to diag_eff.
                    nbr_acc += two_thirds * parent_phi_raw[par_c] * invh2[d]
                    diag_eff += two_thirds * invh2[d]
                else
                    # Outer wall: Dirichlet (φ_ghost = -φ_c) contributes
                    # 2/h² to diag_eff, 0 to nbr_acc. Neumann / periodic
                    # without parent: nothing.
                    if kinds[d] === :dirichlet
                        diag_eff += 2 * invh2[d]
                    end
                end
            end
        end
    end
    return nbr_acc, diag_eff
end

function apply_laplacian_fine_fast!(Lphi::Vector{Vector{NamedTuple}},
                                     phi::Vector{Vector{NamedTuple}},
                                     ws::MGWorkspace{D, T}, ℓ::Int;
                                     skip_covered::Bool = false,
                                     skip_cf::Bool = false) where {D, T}
    pi = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    pcell = ws.parent_cell[ℓ][pi]
    phi_raw = _raw_phi(phi[ℓ][pi])::Vector{T}
    Lphi_raw = _raw_phi(Lphi[ℓ][pi])::Vector{T}
    kinds = ws.axis_kinds.kinds
    parent_phi_raw = _raw_phi(phi[ℓ - 1][1])::Vector{T}
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    # Intermediate-level C/F handling: this level may have a FINER level
    # above it (level ℓ+1) — cells in this level adjacent to that level's
    # patch are "C/F-adjacent" and need the FAC flux-fix treatment when
    # `skip_cf = true`. `covered_by_finer[ℓ]` flags those cells.
    covered = ws.covered_by_finer[ℓ][pi]
    fac_table = ws.fac_fine_at_coarse_face[ℓ][pi]
    has_finer = length(ws.ph.levels) > ℓ

    @inbounds for I in CartesianIndices(N)
        c = c2c[I]
        c == 0 && continue
        skip_covered && has_finer && covered[c] && continue
        phi_c = phi_raw[c]
        # If we're skipping C/F faces and this cell has any, we drop those
        # face contributions here; the FAC flux fix added by the caller
        # supplies the correct value.
        if skip_cf && has_finer
            nbr_acc, diag_eff = _fine_stencil_sums_skip_cf(I, c, c2c, phi_raw,
                                                             parent_phi_raw, pcell,
                                                             fac_table, kinds,
                                                             invh2, N, phi_c)
        else
            nbr_acc, diag_eff = _fine_stencil_sums(I, c, c2c, phi_raw,
                                                     parent_phi_raw, pcell,
                                                     kinds, invh2, N, phi_c)
        end
        Lphi_raw[c] = nbr_acc - diag_eff * phi_c
    end
    return Lphi
end

# Variant of `_fine_stencil_sums` that OMITS in-patch faces whose
# cross-face neighbor IS the parent of some FINER-level cell (i.e., the
# coarse cell on this level sits at the C/F boundary of level ℓ+1 from
# THIS cell's perspective). For non-boundary cells it's identical to
# `_fine_stencil_sums`.
@inline function _fine_stencil_sums_skip_cf(I::CartesianIndex{D},
                                              c::Int, c2c::Array{Int, D},
                                              phi_raw::Vector{T},
                                              parent_phi_raw::Vector{T},
                                              pcell::Matrix{Int},
                                              fac_table::Matrix{Vector{Int}},
                                              kinds::NTuple{D, Symbol},
                                              invh2::NTuple{D, T},
                                              N::NTuple{D, Int},
                                              phi_c::T) where {D, T}
    nbr_acc = zero(T)
    diag_eff = zero(T)
    two_thirds = T(2//3)
    @inbounds for d in 1:D
        for (side, face_idx) in ((1, 2 * (d - 1) + 2), (-1, 2 * (d - 1) + 1))
            # If this face is a C/F interface with a finer level (this
            # cell has fine cells listed in fac_table for this face), the
            # contribution is supplied by `apply_fac_flux_fix!` later;
            # omit it here.
            if !isempty(fac_table[c, face_idx])
                continue
            end
            inew = I[d] + side
            if 1 <= inew <= N[d]
                nb_c = c2c[I + side * _unit_offsets(Val(D))[d]]
                nbr_acc += phi_raw[nb_c] * invh2[d]
                diag_eff += invh2[d]
            else
                # Off-patch toward parent (level ℓ-1): MC ghost.
                par_c = pcell[c, face_idx]
                if par_c != 0
                    nbr_acc += two_thirds * parent_phi_raw[par_c] * invh2[d]
                    diag_eff += two_thirds * invh2[d]
                else
                    if kinds[d] === :dirichlet
                        diag_eff += 2 * invh2[d]
                    end
                end
            end
        end
    end
    return nbr_acc, diag_eff
end

# Unit offsets along each axis: `_unit_offsets(Val(D))[d]` returns
# `CartesianIndex{D}((j == d ? 1 : 0) for j in 1:D)`. Cached per D via
# Val-dispatch so the kernel can do `I + unit_offsets[d]` (allocation-free
# CartesianIndex arithmetic) instead of `Base.setindex(Tuple(I), …)`.
@generated function _unit_offsets(::Val{D}) where {D}
    offsets = ntuple(d -> CartesianIndex{D}(ntuple(j -> j == d ? 1 : 0, D)), D)
    return :($offsets)
end

# Read phi at a wrapped/clamped axis-d neighbor of cell `c` at grid index I.
# Returns the neighbor value, or the appropriate BC fall-back (`-phi_c` for
# Dirichlet, `phi_c` for Neumann). `phi_raw` is the underlying
# `Vector{T}` (extracted by `_raw_phi(...)` once per call), allowing
# allocation-free direct indexing.
@inline function _neighbor_phi(phi_raw::AbstractVector{T}, c2c::Array{Int, D},
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
    @inbounds nb_c = c2c[I + (i_new - I[d]) * _unit_offsets(Val(D))[d]]
    return @inbounds phi_raw[nb_c]
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
    phi_coarse = _raw_phi(phi[coarse_level][pi_c])::Vector{T}
    phi_fine   = _raw_phi(phi[fine_level][pi_f])::Vector{T}
    Lphi_coarse = _raw_phi(Lphi[coarse_level][pi_c])::Vector{T}
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
            phi_c = phi_coarse[c]
            fac_acc = zero(T)
            for fcell in fine_list
                fac_acc += (phi_fine[fcell] - phi_c)
            end
            Lphi_coarse[c] += fac_acc * inv_factor
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
    phi_raw = _raw_phi(phi[ℓ][pi])::Vector{T}
    Lphi_raw = _raw_phi(Lphi[ℓ][pi])::Vector{T}
    kinds = ws.axis_kinds.kinds
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    covered = ws.covered_by_finer[ℓ][pi]
    fac_table = ws.fac_fine_at_coarse_face[ℓ][pi]
    has_fac = skip_cf && length(ws.ph.levels) > ℓ

    # Loop is alloc-free and writes one cell per iteration → safe to thread
    # statically. Skip threading for tiny patches where overhead dominates.
    cidx = CartesianIndices(N)
    if length(cidx) >= 1024 && nthreads() > 1
        @threads :static for I in cidx
            _root_lapl_cell!(I, c2c, phi_raw, Lphi_raw, kinds, invh2, N,
                              covered, fac_table, skip_covered, has_fac,
                              Val(D))
        end
    else
        @inbounds for I in cidx
            _root_lapl_cell!(I, c2c, phi_raw, Lphi_raw, kinds, invh2, N,
                              covered, fac_table, skip_covered, has_fac,
                              Val(D))
        end
    end
    return Lphi
end

@inline function _root_lapl_cell!(I::CartesianIndex{D}, c2c, phi_raw::Vector{T},
                                    Lphi_raw::Vector{T},
                                    kinds::NTuple{D, Symbol},
                                    invh2::NTuple{D, T}, N::NTuple{D, Int},
                                    covered::Vector{Bool},
                                    fac_table::Matrix{Vector{Int}},
                                    skip_covered::Bool, has_fac::Bool,
                                    ::Val{D}) where {D, T}
    @inbounds begin
        c = c2c[I]
        c == 0 && return
        skip_covered && covered[c] && return
        phi_c = phi_raw[c]
        acc = zero(T)
        for d in 1:D
            face_p = 2 * (d - 1) + 2
            if !(has_fac && !isempty(fac_table[c, face_p]))
                phi_p = _neighbor_phi(phi_raw, c2c, I, d, +1, N, kinds, phi_c)
                acc += (phi_p - phi_c) * invh2[d]
            end
            face_m = 2 * (d - 1) + 1
            if !(has_fac && !isempty(fac_table[c, face_m]))
                phi_m = _neighbor_phi(phi_raw, c2c, I, d, -1, N, kinds, phi_c)
                acc += (phi_m - phi_c) * invh2[d]
            end
        end
        Lphi_raw[c] = acc
    end
    return nothing
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
            rf = _raw_phi(r[ℓ][pi])::Vector{T}
            rh = _raw_rho(rho[ℓ][pi])::Vector{T}
            for c in ws.patch_leaves[ℓ][pi]
                rf[c] = rh[c] - rf[c]
            end
        end
    end
    return r
end

"""
    residual_l2(r, ws; level_range, skip_covered = true) -> Float64

L²-norm of the residual over `level_range`, volume-weighted. `skip_covered`
defaults to true so covered coarse cells (whose phi is just a placeholder
for the volume-average of the fine solution underneath) are excluded —
their "residual" against the periodic Laplacian is not physically
meaningful and including it caps the reported convergence at the magnitude
of that nonzero contribution. Pass `skip_covered = false` to recover the
old all-cells sum.
"""
function residual_l2(r::Vector{Vector{NamedTuple}}, ws::MGWorkspace{D, T};
                     level_range::UnitRange{Int} = ws.level_range,
                     skip_covered::Bool = true) where {D, T}
    s = 0.0
    @inbounds for ℓ in level_range
        for pi in 1:length(r[ℓ])
            f = _raw_phi(r[ℓ][pi])::Vector{T}
            v = prod(ws.patch_dx[ℓ][pi])
            covered = ws.covered_by_finer[ℓ][pi]
            for c in ws.patch_leaves[ℓ][pi]
                skip_covered && covered[c] && continue
                rc = Float64(f[c])
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
        rf = _raw_rho(rho[ℓ][pi])::Vector{T}
        out = ws.rho_flat[ℓ][pi]
        for i in 1:length(rf)
            out[i] = rf[i]
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
                   use_fac::Bool = false,
                   reverse_colours::Bool = false,
                   backend = default_backend()) where {D, T}
    _refresh_rho_flat!(ws, rho, ℓ)
    if ℓ == 1
        gs_sweep_root_fast!(phi, ws, ℓ; n_sweeps = n_sweeps,
                             skip_covered = skip_covered,
                             use_fac = use_fac,
                             reverse_colours = reverse_colours)
    else
        gs_sweep_fine_fast!(phi, ws, ℓ; n_sweeps = n_sweeps,
                             skip_covered = skip_covered,
                             use_fac = use_fac,
                             reverse_colours = reverse_colours)
    end
    return phi
end

# Fast direct-array RB-GS for a fine (non-root) level. Same algorithm as
# `gs_sweep_root_fast!` but uses `parent_cell` to look up the parent's
# value at off-patch boundaries (with Martin–Colella correction).
function gs_sweep_fine_fast!(phi::Vector{Vector{NamedTuple}},
                              ws::MGWorkspace{D, T}, ℓ::Int;
                              n_sweeps::Int = 1,
                              skip_covered::Bool = false,
                              use_fac::Bool = false,
                              reverse_colours::Bool = false) where {D, T}
    pi = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    pcell = ws.parent_cell[ℓ][pi]
    phi_raw = _raw_phi(phi[ℓ][pi])::Vector{T}
    rho_flat = ws.rho_flat[ℓ][pi]
    kinds = ws.axis_kinds.kinds
    parent_phi_raw = _raw_phi(phi[ℓ - 1][1])::Vector{T}
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    covered = ws.covered_by_finer[ℓ][pi]
    fac_table = ws.fac_fine_at_coarse_face[ℓ][pi]
    has_finer = use_fac && length(ws.ph.levels) > ℓ
    fine_phi_raw = has_finer ? (_raw_phi(phi[ℓ + 1][1])::Vector{T}) : phi_raw
    dxf_finer = has_finer ? ws.patch_dx[ℓ + 1][1] : dx

    colour_order = reverse_colours ? (1:-1:0) : (0:1)

    for _ in 1:n_sweeps
        for colour in colour_order
            @inbounds for I in CartesianIndices(N)
                s = 0
                for d in 1:D; s += I[d]; end
                (s & 1) == colour || continue
                c = c2c[I]
                c == 0 && continue
                skip_covered && has_finer && covered[c] && continue
                phi_c = phi_raw[c]
                if has_finer
                    nbr_acc, diag_eff = _fine_stencil_sums_fac(
                        I, c, c2c, phi_raw, parent_phi_raw, pcell,
                        fine_phi_raw, fac_table, kinds, invh2, dx, dxf_finer,
                        N, phi_c)
                else
                    nbr_acc, diag_eff = _fine_stencil_sums(I, c, c2c, phi_raw,
                                                             parent_phi_raw, pcell,
                                                             kinds, invh2, N, phi_c)
                end
                rho_c = rho_flat[c]
                phi_raw[c] = (nbr_acc - rho_c) / diag_eff
            end
        end
    end
    return phi
end

# Fine-level stencil sums with FAC contribution from an even FINER level
# (level ℓ+1). Used by `gs_sweep_fine_fast!` at intermediate levels.
# Per face:
#   * If face has a C/F sibling at level ℓ+1 (fac_table[c, k] non-empty):
#       nbr_acc += (2/3)/(h_c h_f n_sub) · Σ_j φ_fine_j
#       diag_eff += (2/3)/(h_c h_f)
#   * Else if face is in-patch: regular.
#   * Else (off-patch toward parent ℓ-1): MC ghost from parent.
@inline function _fine_stencil_sums_fac(I::CartesianIndex{D},
                                          c::Int, c2c::Array{Int, D},
                                          phi_raw::Vector{T},
                                          parent_phi_raw::Vector{T},
                                          pcell::Matrix{Int},
                                          fine_phi_raw::Vector{T},
                                          fac_table::Matrix{Vector{Int}},
                                          kinds::NTuple{D, Symbol},
                                          invh2::NTuple{D, T},
                                          dxc::NTuple{D, T},
                                          dxf::NTuple{D, T},
                                          N::NTuple{D, Int},
                                          phi_c::T) where {D, T}
    nbr_acc = zero(T)
    diag_eff = zero(T)
    two_thirds = T(2//3)
    @inbounds for d in 1:D
        for (side, face_idx) in ((1, 2 * (d - 1) + 2), (-1, 2 * (d - 1) + 1))
            fine_list = fac_table[c, face_idx]
            if !isempty(fine_list)
                # C/F face with level ℓ+1.
                n_sub = length(fine_list)
                coeff_nbr = two_thirds / (dxc[d] * dxf[d] * n_sub)
                for fcell in fine_list
                    nbr_acc += coeff_nbr * fine_phi_raw[fcell]
                end
                diag_eff += two_thirds / (dxc[d] * dxf[d])
                continue
            end
            inew = I[d] + side
            if 1 <= inew <= N[d]
                # In-patch (same-level) neighbor.
                nb_c = c2c[I + side * _unit_offsets(Val(D))[d]]
                nbr_acc += phi_raw[nb_c] * invh2[d]
                diag_eff += invh2[d]
            else
                # Off-patch toward parent ℓ-1: MC ghost.
                par_c = pcell[c, face_idx]
                if par_c != 0
                    nbr_acc += two_thirds * parent_phi_raw[par_c] * invh2[d]
                    diag_eff += two_thirds * invh2[d]
                else
                    if kinds[d] === :dirichlet
                        diag_eff += 2 * invh2[d]
                    end
                end
            end
        end
    end
    return nbr_acc, diag_eff
end


# Fast direct-array RB-GS for the ROOT level (no parent halos).
# Reads stencil neighbors via `cart_to_cell` with explicit periodic /
# Dirichlet / Neumann handling on the outer faces. Bypasses
# `for_each_patch!` entirely — one allocation-free pass per sweep.
function gs_sweep_root_fast!(phi::Vector{Vector{NamedTuple}},
                              ws::MGWorkspace{D, T}, ℓ::Int;
                              n_sweeps::Int = 1,
                              skip_covered::Bool = false,
                              use_fac::Bool = false,
                              reverse_colours::Bool = false) where {D, T}
    @assert ℓ == 1
    pi = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    phi_raw = _raw_phi(phi[ℓ][pi])::Vector{T}
    rho_flat = ws.rho_flat[ℓ][pi]
    kinds = ws.axis_kinds.kinds
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    diag = sum(d -> 2 * invh2[d], 1:D)
    covered = ws.covered_by_finer[ℓ][pi]
    fac_table = ws.fac_fine_at_coarse_face[ℓ][pi]
    has_fac = use_fac && length(ws.ph.levels) > ℓ
    # Fine-side state for FAC contributions (only used when has_fac).
    phi_fine_raw = has_fac ? (_raw_phi(phi[ℓ + 1][1])::Vector{T}) : phi_raw
    dxf = has_fac ? ws.patch_dx[ℓ + 1][1] : dx

    # Colour order: (0,1) for forward, (1,0) for backward. Symmetric V-cycle
    # for PCG preconditioner = pre_smooth(forward) + post_smooth(backward).
    colour_order = reverse_colours ? (1:-1:0) : (0:1)

    cidx = CartesianIndices(N)
    threaded = length(cidx) >= 1024 && nthreads() > 1
    for _ in 1:n_sweeps
        for colour in colour_order
            if threaded
                @threads :static for I in cidx
                    _root_gs_cell!(I, c2c, phi_raw, phi_fine_raw, rho_flat,
                                    fac_table, kinds, invh2, dx, dxf,
                                    diag, N, covered, skip_covered,
                                    has_fac, colour, Val(D))
                end
            else
                @inbounds for I in cidx
                    _root_gs_cell!(I, c2c, phi_raw, phi_fine_raw, rho_flat,
                                    fac_table, kinds, invh2, dx, dxf,
                                    diag, N, covered, skip_covered,
                                    has_fac, colour, Val(D))
                end
            end
        end
    end
    return phi
end

@inline function _root_gs_cell!(I::CartesianIndex{D}, c2c,
                                  phi_raw::Vector{T},
                                  phi_fine_raw::Vector{T},
                                  rho_flat::Vector{T},
                                  fac_table::Matrix{Vector{Int}},
                                  kinds::NTuple{D, Symbol},
                                  invh2::NTuple{D, T},
                                  dxc::NTuple{D, T},
                                  dxf::NTuple{D, T},
                                  diag_default::T,
                                  N::NTuple{D, Int},
                                  covered::Vector{Bool},
                                  skip_covered::Bool, has_fac::Bool,
                                  colour::Int,
                                  ::Val{D}) where {D, T}
    @inbounds begin
        s = 0
        for d in 1:D; s += I[d]; end
        (s & 1) == colour || return
        c = c2c[I]
        c == 0 && return
        skip_covered && covered[c] && return
        phi_c = phi_raw[c]
        # If no FAC needed, fast path with the cached default diagonal.
        if !has_fac
            nbr_acc = zero(T)
            for d in 1:D
                phi_p = _neighbor_phi(phi_raw, c2c, I, d, +1, N, kinds, phi_c)
                phi_m = _neighbor_phi(phi_raw, c2c, I, d, -1, N, kinds, phi_c)
                nbr_acc += (phi_p + phi_m) * invh2[d]
            end
            rho_c = rho_flat[c]
            phi_raw[c] = (nbr_acc - rho_c) / diag_default
            return
        end
        # FAC-aware path: at each face, check if it's a C/F interface.
        # If so, use the fine-side flux contribution; else regular stencil.
        two_thirds = T(2//3)
        nbr_acc = zero(T)
        diag_eff = zero(T)
        for d in 1:D
            for (side, face_idx) in ((1, 2 * (d - 1) + 2), (-1, 2 * (d - 1) + 1))
                fine_list = fac_table[c, face_idx]
                if !isempty(fine_list)
                    # C/F face. Total contribution to (L φ)_c from this face:
                    #   (2/3)/(h_c h_f) · (φ_avg − φ_c)
                    # where φ_avg = (Σ_j φ_cf_j) / n_sub. Splitting:
                    #   nbr_acc += (2/3)/(h_c h_f n_sub) · Σ_j φ_cf_j
                    #   diag_eff += (2/3)/(h_c h_f)
                    n_sub = length(fine_list)
                    coeff_nbr = two_thirds / (dxc[d] * dxf[d] * n_sub)
                    for fcell in fine_list
                        nbr_acc += coeff_nbr * phi_fine_raw[fcell]
                    end
                    diag_eff += two_thirds / (dxc[d] * dxf[d])
                else
                    # Regular face (in-patch or outer BC).
                    phi_n = _neighbor_phi(phi_raw, c2c, I, d, side, N, kinds, phi_c)
                    nbr_acc += phi_n * invh2[d]
                    diag_eff += invh2[d]
                end
            end
        end
        rho_c = rho_flat[c]
        phi_raw[c] = (nbr_acc - rho_c) / diag_eff
    end
    return nothing
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
        # In-place rfft to avoid allocating a fresh k-space buffer.
        mul!(ws.fft_buf_complex, ws.fft_plan_fwd, buf)
        kbuf = ws.fft_buf_complex
        @inbounds for I in CartesianIndices(kbuf)
            kbuf[I] = kbuf[I] * ws.fft_inv_eig[I]
        end
        mul!(buf, ws.fft_plan_inv, kbuf)
    else
        # r2r path: forward into the cached scratch buffer, multiply by
        # 1/eigenvalues, inverse back into the main buffer, then normalise.
        # Avoids the per-call allocation that `plan_fwd * buf` would do.
        rbuf = ws.fft_r2r_buf
        mul!(rbuf, ws.fft_plan_fwd, buf)
        @inbounds for I in eachindex(rbuf)
            rbuf[I] *= ws.fft_inv_eig[I]
        end
        mul!(buf, ws.fft_plan_inv, rbuf)
        normfac = one(T)
        for d in 1:D
            normfac *= ws.fft_kind[d] === :periodic ? N[d] : 2 * N[d]
        end
        inv_norm = one(T) / normfac
        @inbounds for I in eachindex(buf)
            buf[I] *= inv_norm
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

struct SchurFactor{T, Chol, Lu}
    # Cell partitioning
    I_cells::Vector{Int}            # fine-patch cell index of each I row
    Γ_cells::Vector{Int}            # fine-patch cell index of each Γ row
    I_pos::Vector{Int}              # I_pos[c] = row in A_II if c ∈ I, else 0
    Γ_pos::Vector{Int}              # Γ_pos[c] = row in A_ΓΓ if c ∈ Γ, else 0
    # Operator blocks (typed so back-substitution is fully specialized).
    A_II_chol::Chol                 # cholesky of A_II
    A_IΓ::SparseMatrixCSC{T, Int}
    A_ΓI::SparseMatrixCSC{T, Int}
    A_ΓΓ::SparseMatrixCSC{T, Int}
    S_lu::Lu                        # LU factor of dense Schur complement
    # Parent-side contributions to the RHS at each Γ cell (Dirichlet flux).
    # Built fresh each solve from phi_parent — but the pattern (which Γ
    # cell touches which parent cell on which face, with what coefficient)
    # is cached.
    ghost_links::Vector{NTuple{3, Int}}   # (Γ_pos, parent_cell, face_idx)
    invh2::NTuple{1, T}             # 1/h² along each axis (we cache only
                                    # one entry for D=2/3 — see usage)
end

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
                nb_c = c2c[CartesianIndex(Ic) + side * _unit_offsets(Val(D))[d]]
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

    return SchurFactor{T, typeof(A_II_chol), typeof(S_lu)}(
        I_cells, Γ_cells, I_pos, Γ_pos,
        A_II_chol, A_IΓ, A_ΓI, A_ΓΓ, S_lu,
        ghost_links, (invh2[1],))
end

# Solve `M φ = −ρ` on the fine patch using the precomputed Schur factor.
# Updates phi[ℓ][pi].phi in place.
function schur_bottom_solve!(phi::Vector{Vector{NamedTuple}},
                              rho::Vector{Vector{NamedTuple}},
                              ws::MGWorkspace{D, T},
                              ℓ::Int,
                              factor::SchurFactor) where {D, T}
    pi = 1
    phi_arr = _raw_phi(phi[ℓ][pi])::Vector{T}
    rho_arr = _raw_rho(rho[ℓ][pi])::Vector{T}
    parent_phi = _raw_phi(phi[ℓ - 1][1])::Vector{T}
    dx = ws.patch_dx[ℓ][pi]
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))

    nI = length(factor.I_cells)
    nΓ = length(factor.Γ_cells)

    # Build b_I and b_Γ. b = −ρ (sign flip for M = −L).
    b_I = Vector{T}(undef, nI)
    b_Γ = Vector{T}(undef, nΓ)
    @inbounds for (i, c) in enumerate(factor.I_cells)
        b_I[i] = -rho_arr[c]
    end
    @inbounds for (i, c) in enumerate(factor.Γ_cells)
        b_Γ[i] = -rho_arr[c]
    end
    two_thirds = T(2//3)
    @inbounds for (row, par_c, d) in factor.ghost_links
        b_Γ[row] += two_thirds * invh2[d] * parent_phi[par_c]
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
        phi_arr[c] = φ_I[i]
    end
    @inbounds for (i, c) in enumerate(factor.Γ_cells)
        phi_arr[c] = φ_Γ[i]
    end
    return phi
end

# ----------------------------------------------------------------------------
# V-cycle
# ----------------------------------------------------------------------------

"""
    vcycle!(phi, rho, ws; level_range = ws.level_range)

One geometric V-cycle on `level_range`. Updates phi at every active level.

Inner-loop algorithm (textbook FAC, ABC98 §2):
the smoother is RB-GS using the regular Laplacian with **Martin–Colella
linear ghost** at C/F boundaries (the MC ghost IS the FAC treatment on the
fine side); the residual at each level uses the same regular Laplacian; the
coarsest level is solved by the FFT bottom solver if it was configured,
otherwise via Schur / CG / GS.

The fine-vs-coarse flux mismatch (the "FAC composite operator" term) is
NOT applied in this inner loop — it is handled by the outer
`pcg_composite_solve!` wrapper via `apply_composite_laplacian!`.

**Convergence regime**:

  * **2-level periodic AMR**: converges to discretisation accuracy in
    O(10) cycles. Direct use is fine.
  * **3-level nested**: reduces residual by ~10×/cycle for a few cycles,
    then stalls at ~2 % relative (FAC operator inconsistency). Direct
    use is fine for low-precision needs.
  * **4+ level nested**: pure V-cycle diverges. The recursive coarse
    correction equation uses regular Laplacian, and the residual it
    operates on is the FAC-composite restricted residual — the
    operator mismatch amplifies per cycle. **Use `cycle = :pcg,
    pcg_precond = :jacobi` instead** — that path uses the correct FAC
    composite operator via `apply_composite_laplacian!` and converges
    in depth-independent ~62 iterations to machine precision.
"""
function vcycle!(phi::Vector{Vector{NamedTuple}},
                 rho::Vector{Vector{NamedTuple}},
                 ws::MGWorkspace{D, T};
                 level_range::UnitRange{Int} = ws.level_range,
                 backend = Sequential()) where {D, T}
    ℓ_hi = last(level_range)
    ℓ_lo = first(level_range)

    if ℓ_hi == ℓ_lo
        # Coarsest level of this V-cycle: direct solve via FFT, Schur
        # direct, Jacobi-PCG, or raw GS sweeps. The phi at this level is
        # the correction we solve for; rho is the (restricted) residual.
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
                       n_sweeps = ws.opts.bottom_smooth_iters,
                       backend = backend)
        end
        return phi
    end

    # Pre-smooth fine level (level ℓ_hi). Forward colour order
    # (red → black). Post-smooth below uses backward order so the
    # combined V-cycle is symmetric — required for SPD V-cycle
    # preconditioner in PCG (pcg_precond = :vcycle).
    gs_sweep!(phi, rho, ws, ℓ_hi; n_sweeps = ws.opts.n_pre,
               reverse_colours = false, backend = backend)

    is_outer = phi !== ws.correction

    # Residual on the fine level (regular Laplacian; MC ghost handles the
    # C/F face implicitly via the fine-side parent-halo read).
    apply_laplacian!(ws.residual, phi, ws;
                      level_range = ℓ_hi:ℓ_hi, backend = backend)
    @inbounds for pi in 1:length(ws.residual[ℓ_hi])
        rf = ws.residual[ℓ_hi][pi].phi
        rh = rho[ℓ_hi][pi].rho
        for c in ws.patch_leaves[ℓ_hi][pi]
            _set_val!(rf, c, _get_val(rh, c) - _get_val(rf, c))
        end
    end

    # OUTER-CALL ONLY: compute the residual on level ℓ_hi-1 uncovered cells.
    # The coarse-grid correction equation L_coarse(δφ) = r drives δφ at
    # uncovered cells using this residual. With the FAC SYNC above, covered
    # neighbours read sensible values, so L_regular gives the standard
    # ABC98 composite operator on the uncovered side.
    #
    # The RECURSIVE call skips this because (a) `phi === ws.correction`, so
    # the correction at level ℓ_hi-1 starts at zero and L_regular(0)=0
    # everywhere; (b) the "rho" for the recursive call's level ℓ_hi-1 is
    # ws.rhs_coarse[ℓ_hi-1].rho which was not set by the outer call (only
    # the recursive call's ℓ_hi=outer_ℓ_hi-1 has its rho set), so reading
    # it would give stale data. Standard MG behaviour: uncovered cells of
    # the recursive call's coarser level get 0 in the RHS (no fine cells
    # restrict there). The outer-level uncovered contribution is captured
    # by the outer call's residual at its ℓ_hi-1, which becomes the
    # recursive call's top-level RHS.
    if is_outer
        apply_laplacian!(ws.residual, phi, ws;
                          level_range = (ℓ_hi - 1):(ℓ_hi - 1),
                          skip_covered = true,
                          backend = backend)
        @inbounds for pi in 1:length(ws.residual[ℓ_hi - 1])
            rf = ws.residual[ℓ_hi - 1][pi].phi
            rh = rho[ℓ_hi - 1][pi].rho
            covered = ws.covered_by_finer[ℓ_hi - 1][pi]
            for c in ws.patch_leaves[ℓ_hi - 1][pi]
                covered[c] && continue  # leave covered alone, restriction overwrites
                _set_val!(rf, c, _get_val(rh, c) - _get_val(rf, c))
            end
        end
    end

    # Build the RHS for the coarser recursive call:
    # * Outer call: rhs_coarse[ℓ_hi-1] = residual[ℓ_hi-1] on uncovered cells
    #   (computed above), volume-averaged residual[ℓ_hi] on covered cells.
    # * Recursive call: uncovered cells get 0 (standard MG: no fine cells
    #   restrict there). Covered cells get the volume-averaged residual.
    if is_outer
        @inbounds for pi in 1:length(ws.rhs_coarse[ℓ_hi - 1])
            src = ws.residual[ℓ_hi - 1][pi].phi
            dst_phi = ws.rhs_coarse[ℓ_hi - 1][pi].phi
            dst_rho = ws.rhs_coarse[ℓ_hi - 1][pi].rho
            for c in ws.patch_leaves[ℓ_hi - 1][pi]
                v = _get_val(src, c)
                _set_val!(dst_phi, c, v)
                _set_val!(dst_rho, c, v)
            end
        end
    else
        @inbounds for pi in 1:length(ws.rhs_coarse[ℓ_hi - 1])
            dst_phi = ws.rhs_coarse[ℓ_hi - 1][pi].phi
            dst_rho = ws.rhs_coarse[ℓ_hi - 1][pi].rho
            for c in ws.patch_leaves[ℓ_hi - 1][pi]
                _set_val!(dst_phi, c, zero(T))
                _set_val!(dst_rho, c, zero(T))
            end
        end
    end
    # Overwrite the covered coarse cells with the volume-averaged fine
    # residual (this is the standard FAC restriction at C/F faces).
    restrict_to_parents!(ws.rhs_coarse[ℓ_hi - 1],
                          ws.residual[ℓ_hi],
                          ws.ph; level = ℓ_hi, fieldname = :phi)
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
    #
    # CRITICAL: In the OUTER V-cycle call, `phi` and `ws.correction` are
    # distinct fields, so `phi[ℓ-1] += correction[ℓ-1]` correctly adds the
    # coarse-grid correction to the actual solution. In the RECURSIVE
    # call, however, `phi === ws.correction` (we passed correction as the
    # phi argument), and naive addition would double the field value:
    # phi[ℓ-1] += phi[ℓ-1]. The skip below is the standard
    # "in-place V-cycle" pattern — when phi already IS the correction
    # field, the sub-recursion has already written the result directly
    # into phi[ℓ-1]; no further accumulation is needed at this level.
    # The prolongation to phi[ℓ_hi] is still needed in both cases.
    if phi !== ws.correction
        @inbounds for pi in 1:length(phi[ℓ_hi - 1])
            pf = phi[ℓ_hi - 1][pi].phi
            cf = ws.correction[ℓ_hi - 1][pi].phi
            for c in ws.patch_leaves[ℓ_hi - 1][pi]
                _set_val!(pf, c, _get_val(pf, c) + _get_val(cf, c))
            end
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

    # Post-smooth on the fine level (backward colour order, see pre-smooth).
    gs_sweep!(phi, rho, ws, ℓ_hi; n_sweeps = ws.opts.n_post,
               reverse_colours = true, backend = backend)
    return phi
end

# ----------------------------------------------------------------------------
# solve_poisson!
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# PCG-on-composite-FAC — eliminates the V-cycle's prolongation-coupling
# fixed point by treating uncovered-coarse + fine cells as one symmetric
# system and running preconditioned CG with the V-cycle as preconditioner.
#
# The V·L_FAC matrix is symmetric in the V-weighted inner product
# ⟨a, b⟩_V = Σ_c V_c · a[c] · b[c]  (verified by hand for 2:1 in 2D/3D):
#   V_c · L[c, cf] = V_cf · L[cf, c]
# so CG works directly on the cell-centered FAC system in this norm.
#
# Per outer iteration: 1 matvec (apply_composite_laplacian!), 1 V-cycle
# preconditioner application, ~3 V-weighted dot products. ~50–100 % more
# wall-clock than a pure V-cycle iteration, but drives residual to machine
# precision instead of stalling at the prolongation-coupling fixed point.
# ----------------------------------------------------------------------------

# Apply the FAC composite operator: regular Laplacian on fine levels (with
# MC ghost at C/F via apply_laplacian_fine_fast!), and root operator with
# FAC flux fix on the coarsest level (skipping covered cells, omitting
# C/F flux from the regular sum). Result written to Lphi.
function apply_composite_laplacian!(Lphi::Vector{Vector{NamedTuple}},
                                     phi::Vector{Vector{NamedTuple}},
                                     ws::MGWorkspace{D, T};
                                     level_range::UnitRange{Int} = ws.level_range,
                                     backend = default_backend()) where {D, T}
    ℓ_lo = first(level_range)
    ℓ_hi = last(level_range)
    has_finer = length(ws.ph.levels) > ℓ_lo
    # Apply L_FAC at the COARSEST level of the range (skip C/F faces, add
    # FAC contribution from the next-finer level). At other levels, apply
    # the regular Laplacian. (A fully correct multi-level FAC operator
    # would apply skip_cf + FAC fix at EVERY level with a finer level
    # above — but empirically that produces a non-SPD operator in the
    # V-weighted norm despite the per-pair coupling being symmetric. The
    # cause is under investigation; this conservative form is symmetric
    # AND positive-definite, so PCG converges.)
    for ℓ in level_range
        if ℓ == ℓ_lo && has_finer
            apply_laplacian!(Lphi, phi, ws; level_range = ℓ:ℓ,
                              skip_covered = true, skip_cf = true,
                              backend = backend)
            apply_fac_flux_fix!(Lphi, phi, ws, ℓ)
        else
            apply_laplacian!(Lphi, phi, ws; level_range = ℓ:ℓ,
                              backend = backend)
        end
    end
    return Lphi
end

# Composite vector operations: iterate only over UNCOVERED cells (the
# "active" subset of the composite FAC system).

@inline function _composite_dot_V(a_fields, b_fields,
                                    ws::MGWorkspace{D, T};
                                    level_range = ws.level_range) where {D, T}
    s = 0.0
    @inbounds for ℓ in level_range
        for pi in 1:length(a_fields[ℓ])
            a = _raw_phi(a_fields[ℓ][pi])::Vector{T}
            b = _raw_phi(b_fields[ℓ][pi])::Vector{T}
            v = prod(ws.patch_dx[ℓ][pi])
            covered = ws.covered_by_finer[ℓ][pi]
            for c in ws.patch_leaves[ℓ][pi]
                covered[c] && continue
                s += Float64(a[c]) * Float64(b[c]) * v
            end
        end
    end
    return s
end

@inline _composite_norm_V(a, ws; level_range = ws.level_range) =
    sqrt(_composite_dot_V(a, a, ws; level_range = level_range))

# y[c] += α · x[c] on active (uncovered) cells, on .phi field.
@inline function _composite_axpy_phi!(y_fields, x_fields, α::T,
                                        ws::MGWorkspace{D, T};
                                        level_range = ws.level_range) where {D, T}
    @inbounds for ℓ in level_range
        for pi in 1:length(y_fields[ℓ])
            x = _raw_phi(x_fields[ℓ][pi])::Vector{T}
            y = _raw_phi(y_fields[ℓ][pi])::Vector{T}
            covered = ws.covered_by_finer[ℓ][pi]
            for c in ws.patch_leaves[ℓ][pi]
                covered[c] && continue
                y[c] += α * x[c]
            end
        end
    end
    return y_fields
end

# y[c] = β · y[c] + x[c] on active cells (used for p = z + β p).
@inline function _composite_y_eq_x_plus_beta_y!(y_fields, x_fields, β::T,
                                                  ws::MGWorkspace{D, T};
                                                  level_range = ws.level_range) where {D, T}
    @inbounds for ℓ in level_range
        for pi in 1:length(y_fields[ℓ])
            x = _raw_phi(x_fields[ℓ][pi])::Vector{T}
            y = _raw_phi(y_fields[ℓ][pi])::Vector{T}
            covered = ws.covered_by_finer[ℓ][pi]
            for c in ws.patch_leaves[ℓ][pi]
                covered[c] && continue
                y[c] = x[c] + β * y[c]
            end
        end
    end
    return y_fields
end

# Zero a field on active cells.
@inline function _composite_zero_phi!(y_fields,
                                        ws::MGWorkspace{D, T};
                                        level_range = ws.level_range) where {D, T}
    @inbounds for ℓ in level_range
        for pi in 1:length(y_fields[ℓ])
            y = _raw_phi(y_fields[ℓ][pi])::Vector{T}
            for c in 1:length(y); y[c] = zero(T); end
        end
    end
    return y_fields
end

# Copy from .phi field of source to .rho field of dest, on active cells.
# Used to feed the residual to the V-cycle preconditioner (V-cycle reads
# its RHS from .rho).
@inline function _composite_phi_to_rho!(dst_fields, src_fields,
                                          ws::MGWorkspace{D, T};
                                          level_range = ws.level_range) where {D, T}
    @inbounds for ℓ in level_range
        for pi in 1:length(dst_fields[ℓ])
            src = _raw_phi(src_fields[ℓ][pi])::Vector{T}
            dst = _raw_rho(dst_fields[ℓ][pi])::Vector{T}
            for c in 1:length(dst); dst[c] = src[c]; end
        end
    end
    return dst_fields
end

# r = ρ − L φ on active cells. `r_fields.phi` ← `rho_fields.rho` − `r_fields.phi`
# (after `apply_composite_laplacian!` has written L φ into `r_fields.phi`).
@inline function _composite_subtract_rho_minus_self!(r_fields, rho_fields,
                                                       ws::MGWorkspace{D, T};
                                                       level_range = ws.level_range) where {D, T}
    @inbounds for ℓ in level_range
        for pi in 1:length(r_fields[ℓ])
            r = _raw_phi(r_fields[ℓ][pi])::Vector{T}
            rh = _raw_rho(rho_fields[ℓ][pi])::Vector{T}
            covered = ws.covered_by_finer[ℓ][pi]
            for c in ws.patch_leaves[ℓ][pi]
                covered[c] && continue
                r[c] = rh[c] - r[c]
            end
        end
    end
    return r_fields
end

# Negate the .phi field on active cells.
@inline function _composite_negate_phi!(y_fields,
                                          ws::MGWorkspace{D, T};
                                          level_range = ws.level_range) where {D, T}
    @inbounds for ℓ in level_range
        for pi in 1:length(y_fields[ℓ])
            y = _raw_phi(y_fields[ℓ][pi])::Vector{T}
            covered = ws.covered_by_finer[ℓ][pi]
            for c in ws.patch_leaves[ℓ][pi]
                covered[c] && continue
                y[c] = -y[c]
            end
        end
    end
    return y_fields
end

# Apply the PCG preconditioner: write z = M⁻¹ r (the action of an
# approximate inverse of A = -L_FAC on r).
function _apply_pcg_precond!(z_fields::Vector{Vector{NamedTuple}},
                              r_fields::Vector{Vector{NamedTuple}},
                              ws::MGWorkspace{D, T},
                              level_range::UnitRange{Int},
                              backend) where {D, T}
    kind = ws.opts.pcg_precond
    if kind === :none
        # z = r (identity preconditioner)
        _composite_zero_phi!(z_fields, ws; level_range = level_range)
        _composite_axpy_phi!(z_fields, r_fields, one(T), ws;
                              level_range = level_range)
    elseif kind === :jacobi
        # z = diag(A)⁻¹ r where A = -L_FAC, so diag(A) > 0.
        _composite_jacobi_precond!(z_fields, r_fields, ws; level_range = level_range)
    elseif kind === :vcycle
        # z = M⁻¹ r ≈ -vcycle⁻¹ r (since vcycle ≈ L_FAC⁻¹ and A = -L_FAC).
        _composite_phi_to_rho!(ws.rhs_coarse, r_fields, ws;
                                level_range = level_range)
        _composite_zero_phi!(z_fields, ws; level_range = level_range)
        vcycle!(z_fields, ws.rhs_coarse, ws;
                 level_range = level_range, backend = backend)
        _composite_negate_phi!(z_fields, ws; level_range = level_range)
    else
        error("_apply_pcg_precond!: unknown preconditioner :$kind")
    end
    return z_fields
end

# Jacobi preconditioner: z[c] = r[c] / diag(A)[c]. For the FAC operator,
# diag(A) is the per-cell coefficient of φ_c in (A φ)_c; this is just the
# sum of |1/h²| over all faces (regular Laplacian) with FAC adjustments at
# C/F-adjacent cells.
function _composite_jacobi_precond!(z_fields, r_fields, ws::MGWorkspace{D, T};
                                     level_range = ws.level_range) where {D, T}
    @inbounds for ℓ in level_range
        for pi in 1:length(z_fields[ℓ])
            r = _raw_phi(r_fields[ℓ][pi])::Vector{T}
            z = _raw_phi(z_fields[ℓ][pi])::Vector{T}
            dx = ws.patch_dx[ℓ][pi]
            covered = ws.covered_by_finer[ℓ][pi]
            invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
            d_default = sum(d -> 2 * invh2[d], 1:D)
            for c in ws.patch_leaves[ℓ][pi]
                covered[c] && (z[c] = zero(T); continue)
                # For simplicity, use the default diagonal. C/F-adjacent
                # cells have a slightly different diag but the simple
                # Jacobi is good enough as a preconditioner.
                z[c] = r[c] / d_default
            end
        end
    end
    return z_fields
end

"""
    pcg_composite_solve!(phi, rho, ws; tol, maxiter, verbose) -> MGResult

Preconditioned CG on the FAC composite system. We work with A = −L_FAC
(SPD positive definite for the cell-centered Laplacian under V-weighted
inner product) and the right-hand side b = −ρ. V-cycle is the
preconditioner: since vcycle approximates L_FAC⁻¹, M⁻¹ = −vcycle.
"""
function pcg_composite_solve!(phi::Vector{Vector{NamedTuple}},
                               rho::Vector{Vector{NamedTuple}},
                               ws::MGWorkspace{D, T};
                               tol::Float64 = 1e-10,
                               maxiter::Int = 100,
                               verbose::Bool = false,
                               level_range::UnitRange{Int} = ws.level_range,
                               backend = default_backend()) where {D, T}
    z_fields = allocate_phi_rho(ws.ph)

    # r = b - A φ where A = -L_FAC, b = -ρ.
    #   r = -ρ - (-L_FAC φ) = L_FAC φ - ρ = -(ρ - L_FAC φ).
    apply_composite_laplacian!(ws.residual, phi, ws;
                                level_range = level_range, backend = backend)
    _composite_subtract_rho_minus_self!(ws.residual, rho, ws;
                                          level_range = level_range)
    _composite_negate_phi!(ws.residual, ws; level_range = level_range)

    r0 = _composite_norm_V(ws.residual, ws; level_range = level_range)
    history = Float64[r0]
    if r0 == 0
        return MGResult(0, r0, r0, true, history)
    end

    _apply_pcg_precond!(z_fields, ws.residual, ws, level_range, backend)

    # p = z (zero correction first then axpy)
    _composite_zero_phi!(ws.correction, ws; level_range = level_range)
    _composite_axpy_phi!(ws.correction, z_fields, one(T), ws;
                          level_range = level_range)

    rs_old = _composite_dot_V(ws.residual, z_fields, ws;
                                level_range = level_range)

    converged = false
    r = r0
    for iter in 1:maxiter
        # Ap = A p = -L_FAC p
        apply_composite_laplacian!(ws.rhs_coarse, ws.correction, ws;
                                    level_range = level_range, backend = backend)
        _composite_negate_phi!(ws.rhs_coarse, ws; level_range = level_range)

        pAp = _composite_dot_V(ws.correction, ws.rhs_coarse, ws;
                                level_range = level_range)
        if pAp <= 0
            verbose && @warn "PCG: non-positive pAp = $pAp at iter $iter (operator not SPD?)"
            break
        end
        α = T(rs_old / pAp)

        # φ += α p ;  r -= α Ap
        _composite_axpy_phi!(phi, ws.correction, α, ws;
                              level_range = level_range)
        _composite_axpy_phi!(ws.residual, ws.rhs_coarse, -α, ws;
                              level_range = level_range)

        r = _composite_norm_V(ws.residual, ws; level_range = level_range)
        push!(history, r)
        verbose && @info "PCG iter $iter: |r|_V = $r  (reduction = $(r/r0))"
        if r <= tol * r0 || r <= tol
            converged = true; break
        end

        _apply_pcg_precond!(z_fields, ws.residual, ws, level_range, backend)

        rs_new = _composite_dot_V(ws.residual, z_fields, ws;
                                    level_range = level_range)
        β = T(rs_new / rs_old)
        rs_old = rs_new

        # p = z + β p
        _composite_y_eq_x_plus_beta_y!(ws.correction, z_fields, β, ws;
                                         level_range = level_range)
    end
    return MGResult(length(history) - 1, r0, r, converged, history)
end

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
    # PCG path: composite-system CG with V-cycle preconditioner. Drives
    # residual to machine precision (eliminates the V-cycle's prolongation-
    # coupling fixed point at AMR interfaces).
    if opts.cycle === :pcg
        return pcg_composite_solve!(phi, rho, ws;
                                      tol = opts.tol, maxiter = opts.maxiter,
                                      verbose = opts.verbose,
                                      level_range = level_range,
                                      backend = backend)
    end
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
    stalled = 0
    has_finer_levels = length(ws.ph.levels) > first(level_range)
    while iter < opts.maxiter
        vcycle!(phi, rho, ws; level_range = level_range, backend = backend)
        # Matching filter: polish coarse-level uncovered cells using the
        # FAC operator with the current fine phi. Only meaningful in AMR
        # mode (when a finer level exists above the coarse bottom).
        if opts.matching_filter_sweeps > 0 && has_finer_levels
            ℓ_coarse = first(level_range)
            _refresh_rho_flat!(ws, rho, ℓ_coarse)
            gs_sweep_root_fast!(phi, ws, ℓ_coarse;
                                 n_sweeps = opts.matching_filter_sweeps,
                                 use_fac = true, skip_covered = true)
        end
        iter += 1
        compute_residual!(ws.residual, phi, rho, ws;
                           level_range = level_range, backend = backend)
        r_new = residual_l2(ws.residual, ws; level_range = level_range)
        push!(history, r_new)
        opts.verbose && @info "MG cycle $iter: residual = $r_new  (reduction = $(r_new/r0))"
        if r_new <= opts.tol * r0 || r_new <= opts.tol
            r = r_new; converged = true; break
        end
        # Stagnation detection: if the residual changed by < 0.5% for 3
        # consecutive cycles, the FAC composite operator is at its fixed
        # point and further V-cycles won't help. Stop and report (the
        # solution is still 2nd-order accurate at the C/F interface).
        if iter >= 2 && abs(r_new - r) < 5e-3 * r
            stalled += 1
        else
            stalled = 0
        end
        r = r_new
        if stalled >= 3
            break
        end
    end
    return MGResult(iter, r0, r, converged, history)
end

"""
    solve_poisson!(ws::MGWorkspace, fields; level_range, backend)

Convenience wrapper around the 3-argument form when φ and ρ share the
same per-patch container (as returned by `allocate_phi_rho`). Reads ρ
from `fields[…][…].rho` and writes φ to `fields[…][…].phi`.
"""
solve_poisson!(ws::MGWorkspace, fields::Vector;
                level_range = ws.level_range, backend = Sequential()) =
    solve_poisson!(fields, fields, ws;
                    level_range = level_range, backend = backend)

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

# Variable-coefficient ABec operator: L φ = A α φ - B ∇·(β ∇φ) = f.
include("ABecLaplacian.jl")

# MAC (face-centered) velocity projection: u^{n+1} = u - β ∇φ with
# ∇·(β ∇φ) = ∇·u.  Builds on the ABec operator.
include("MACProjection.jl")

# Krylov.jl bridge — flat-vector wrapping of the FAC composite and ABec
# operators so that GMRES / BiCGStab / FGMRES / MINRES can drive solves.
include("KrylovBridge.jl")

end # module GeometricMultigrid
