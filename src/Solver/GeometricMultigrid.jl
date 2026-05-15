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
using LinearAlgebra: norm, mul!

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
        if prod(N) == nleaves
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
                            residual, correction, rhs_coarse,
                            fft_ok, fft_level, fft_N, fft_dx, fft_inv_eig,
                            fft_kind, fft_buf, fft_buf_complex,
                            fft_plan_fwd, fft_plan_inv, nothing)

    # Refinement listener: blow away FFT state on AMR.
    handle = register_refinement_listener!(patches_at(ph, 1)[1].mesh, _ -> begin
        ws.fft_ok = false
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

@inline function _read_phi(hv::PatchHaloView{Names, Tin, D, T, GD, BC, NT, PBC},
                            off::NTuple{D, Int}, default::T
                            ) where {Names, Tin, D, T, GD, BC, NT, PBC}
    v = hv[:phi, off]
    return v === nothing ? default : T(v[1])
end

# Laplacian kernel: writes L φ into pv[:phi].
function _lapl_kernel(pv::PatchView, hv::PatchHaloView, ::Nothing)
    p_lo, p_hi = cell_physical_box(hv.frame, pv.cv.index)
    D = length(p_lo)
    T = eltype(p_lo)
    phi_c = T(pv[:phi][1])
    acc = zero(T)
    @inbounds for d in 1:D
        h = p_hi[d] - p_lo[d]
        invh2 = one(T) / (h * h)
        off_p = ntuple(j -> j == d ?  1 : 0, D)
        off_m = ntuple(j -> j == d ? -1 : 0, D)
        phi_p = _read_phi(hv, off_p, phi_c)
        phi_m = _read_phi(hv, off_m, phi_c)
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
                          backend = Sequential()) where {D, T}
    for ℓ in level_range
        parent_in = ℓ >= 2 ? phi[ℓ - 1] : nothing
        for_each_patch!(_lapl_kernel,
                          Lphi[ℓ], phi[ℓ], ws.ph;
                          level = ℓ, ghost_depth = 1,
                          fields_in_parent = parent_in,
                          ctx = nothing, backend = backend)
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
                            backend = Sequential()) where {D, T}
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
        off_p = ntuple(j -> j == d ?  1 : 0, D)
        off_m = ntuple(j -> j == d ? -1 : 0, D)
        phi_p = _read_phi(hv, off_p, phi_c)
        phi_m = _read_phi(hv, off_m, phi_c)
        nbr_acc += (phi_p + phi_m) * invh2
        diag    += 2 * invh2
    end
    rho_c = ctx.rho_flat[c]
    pv[:phi] = ((nbr_acc - rho_c) / diag,)
    return nothing
end

# Flatten rho[c] (a polynomial-field cell tuple) into a plain Vector{T}.
function _flatten_rho(rho_view::NamedTuple, ::Type{T}) where {T}
    rf = rho_view.rho
    n  = length(rf)
    out = Vector{T}(undef, n)
    @inbounds for i in 1:n
        out[i] = T(rf[i][1])
    end
    return out
end

# One full RB-GS sweep over a level (both colours). Single-patch-per-level.
function gs_sweep!(phi::Vector{Vector{NamedTuple}},
                   rho::Vector{Vector{NamedTuple}},
                   ws::MGWorkspace{D, T}, ℓ::Int;
                   n_sweeps::Int = 1, backend = Sequential()) where {D, T}
    parent_in = ℓ >= 2 ? phi[ℓ - 1] : nothing
    npatches = length(phi[ℓ])
    rho_flats = [_flatten_rho(rho[ℓ][pi], T) for pi in 1:npatches]
    for _ in 1:n_sweeps
        for colour in (0, 1)
            # We exploit the single-patch-per-level invariant by building a
            # ctx that references patch 1. (Multi-patch support is a v2 item;
            # see plan §G "Root uniformity".)
            ctx = GSCtx{D, T}(colour, ws.grid_idx[ℓ][1], rho_flats[1])
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
# V-cycle
# ----------------------------------------------------------------------------

"""
    vcycle!(phi, rho, ws; level_range = ws.level_range)

One geometric V-cycle on `level_range`.
"""
function vcycle!(phi::Vector{Vector{NamedTuple}},
                 rho::Vector{Vector{NamedTuple}},
                 ws::MGWorkspace{D, T};
                 level_range::UnitRange{Int} = ws.level_range,
                 backend = Sequential()) where {D, T}
    if length(level_range) == 1
        ℓ = first(level_range)
        if ws.fft_ok && ℓ == ws.fft_level
            fft_bottom_solve!(ws, phi[ℓ][1], rho[ℓ][1])
        else
            gs_sweep!(phi, rho, ws, ℓ;
                       n_sweeps = ws.opts.bottom_smooth_iters, backend = backend)
        end
        return phi
    end

    ℓ_hi = last(level_range)
    ℓ_lo = first(level_range)

    # Pre-smooth on fine level
    gs_sweep!(phi, rho, ws, ℓ_hi; n_sweeps = ws.opts.n_pre, backend = backend)

    # Residual on ℓ_hi  (ws.residual)
    apply_laplacian!(ws.residual, phi, ws; level_range = ℓ_hi:ℓ_hi, backend = backend)
    @inbounds for pi in 1:length(ws.residual[ℓ_hi])
        rf = ws.residual[ℓ_hi][pi].phi
        rh = rho[ℓ_hi][pi].rho
        for c in ws.patch_leaves[ℓ_hi][pi]
            _set_val!(rf, c, _get_val(rh, c) - _get_val(rf, c))
        end
    end

    # Restrict residual onto rhs_coarse[ℓ_hi-1].phi (volume-weighted average).
    restrict_to_parents!(ws.rhs_coarse[ℓ_hi - 1],
                          ws.residual[ℓ_hi],
                          ws.ph; level = ℓ_hi, fieldname = :phi)
    # Move phi → rho on the coarse rhs (so recursive call sees rhs as :rho).
    @inbounds for pi in 1:length(ws.rhs_coarse[ℓ_hi - 1])
        src = ws.rhs_coarse[ℓ_hi - 1][pi].phi
        dst = ws.rhs_coarse[ℓ_hi - 1][pi].rho
        for c in ws.patch_leaves[ℓ_hi - 1][pi]
            _set_val!(dst, c, _get_val(src, c))
        end
    end

    # Zero correction on all coarser levels (correction-form V-cycle).
    @inbounds for ℓ in ℓ_lo:(ℓ_hi - 1)
        for pi in 1:length(ws.correction[ℓ])
            f = ws.correction[ℓ][pi].phi
            for c in ws.patch_leaves[ℓ][pi]
                _set_val!(f, c, zero(T))
            end
        end
    end

    # Recurse on coarser range
    vcycle!(ws.correction, ws.rhs_coarse, ws;
             level_range = ℓ_lo:(ℓ_hi - 1), backend = backend)

    # Prolong correction[ℓ_hi - 1] onto phi[ℓ_hi] (multilinear, additive)
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

    # Post-smooth
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
