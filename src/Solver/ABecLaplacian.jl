# ============================================================================
# ABecLaplacian.jl — variable-coefficient elliptic operator
#
# Solves   L φ = A · α(x) · φ - B · ∇·(β(x) ∇φ) = f
# matching the AMReX MLABecLaplacian convention.
#
# Use cases (see ABecCoefs docstring for the canonical settings of A, B,
# α, β for each):
#   * Pure Poisson    ∇²φ = ρ                     A=0, B=-1, β=1
#   * Implicit heat   (I - Δt D∇²)φ = rhs         A=1, α=1, B=Δt·D, β=1
#   * Variable diff   -∇·(D∇φ) = s                A=0, B=1, β=D
#   * MAC projection  ∇·(β∇φ) = ∇·u*              A=0, B=-1, β=1/ρ
#
# The discretisation is 2nd-order finite-volume with face-centered β.
# The smoother is red/black Gauss–Seidel; the bottom solver reuses the
# existing FFT path for pure Poisson cases (degenerates correctly when
# β is constant), and a Jacobi-PCG fallback otherwise.
# ============================================================================

"""
    ABecCoefs{D, T}

Variable-coefficient elliptic operator coefficients for the equation

    A · α(x) · φ - B · ∇·(β(x) ∇φ) = f

Fields:
  * `A::T`, `B::T` — scalar weights.
  * `alpha::Vector{Vector{Vector{T}}}` — `alpha[ℓ][p][c]`, cell-centered α.
  * `beta::Vector{Vector{NTuple{D, Array{T,D}}}}` — `beta[ℓ][p][d]` is a
    D-dim array sized `(N₁, …, N_d+1, …, N_D)`; element
    `[i₁, …, i_d, …, i_D]` is the β value at the face between cells
    `(i₁, …, i_d-1, …, i_D)` and `(i₁, …, i_d, …, i_D)`. Face index
    `i_d` ranges `1:N_d+1`; `i_d=1` is the low-side patch boundary face
    and `i_d=N_d+1` is the high-side boundary face. For periodic axes
    the wrap face value should be stored at both endpoints.

Construct with `allocate_abec_coefs(ws; A, B, alpha_init, beta_init)`
and populate with `fill_abec_alpha!` / `fill_abec_beta!`.
"""
struct ABecCoefs{D, T}
    A::T
    B::T
    alpha::Vector{Vector{Vector{T}}}
    beta::Vector{Vector{NTuple{D, Array{T, D}}}}
end

"""
    allocate_abec_coefs(ws; A=0, B=1, alpha_init=0, beta_init=1) -> ABecCoefs

Allocate ABec coefficient storage matching `ws.ph`. Cell-centered α and
face-centered β are filled with the constants `alpha_init` and
`beta_init`; the user fills in non-uniform values via
`fill_abec_alpha!` / `fill_abec_beta!`.
"""
function allocate_abec_coefs(ws::MGWorkspace{D, T};
                              A::T = zero(T), B::T = one(T),
                              alpha_init::T = zero(T),
                              beta_init::T = one(T)) where {D, T}
    nL = length(ws.patch_N)
    alpha = Vector{Vector{Vector{T}}}(undef, nL)
    beta  = Vector{Vector{NTuple{D, Array{T, D}}}}(undef, nL)
    for ℓ in 1:nL
        nP = length(ws.patch_N[ℓ])
        αℓ = Vector{Vector{T}}(undef, nP)
        βℓ = Vector{NTuple{D, Array{T, D}}}(undef, nP)
        for pi in 1:nP
            N = ws.patch_N[ℓ][pi]
            # α is cell-indexed: size matches the full mesh (leaves + tree
            # parents). Only leaf indices are read by the apply / GS
            # kernels, but the storage must be at least as large as the
            # largest leaf cell index returned by `c2c`.
            n_storage = length(_raw_phi(ws.residual[ℓ][pi]))
            αℓ[pi] = fill(alpha_init, n_storage)
            β_d = ntuple(Val(D)) do d
                fdims = ntuple(j -> j == d ? N[j] + 1 : N[j], Val(D))
                fill(beta_init, fdims)
            end
            βℓ[pi] = β_d
        end
        alpha[ℓ] = αℓ
        beta[ℓ]  = βℓ
    end
    return ABecCoefs{D, T}(A, B, alpha, beta)
end

"""
    fill_abec_alpha!(coefs, ws, f)

Sample `α(x) = f(x)` at each cell center of every (level, patch).
"""
function fill_abec_alpha!(coefs::ABecCoefs{D, T},
                           ws::MGWorkspace{D, T}, f) where {D, T}
    for ℓ in 1:length(coefs.alpha)
        for pi in 1:length(coefs.alpha[ℓ])
            frame = patches_at(ws.ph, ℓ)[pi]
            α = coefs.alpha[ℓ][pi]
            @inbounds for c in ws.patch_leaves[ℓ][pi]
                lo, hi = cell_physical_box(frame, c)
                x = ntuple(d -> 0.5 * (lo[d] + hi[d]), Val(D))
                α[c] = T(f(x))
            end
        end
    end
    return coefs
end

"""
    fill_abec_beta!(coefs, ws, f)

Sample `β(x) = f(x)` at each face center along each axis of every
(level, patch). For face position `(i₁, …, i_d, …)` along axis d, the
physical location is `(x₁_center, …, lo_d + (i_d-1)·h_d, …)`.
"""
function fill_abec_beta!(coefs::ABecCoefs{D, T},
                          ws::MGWorkspace{D, T}, f) where {D, T}
    for ℓ in 1:length(coefs.beta)
        for pi in 1:length(coefs.beta[ℓ])
            frame = patches_at(ws.ph, ℓ)[pi]
            N = ws.patch_N[ℓ][pi]
            dx = ws.patch_dx[ℓ][pi]
            for d in 1:D
                βd = coefs.beta[ℓ][pi][d]
                fdims = size(βd)
                @inbounds for I in CartesianIndices(fdims)
                    x = ntuple(Val(D)) do j
                        if j == d
                            frame.lo[d] + (I[d] - 1) * dx[d]
                        else
                            frame.lo[j] + (I[j] - 0.5) * dx[j]
                        end
                    end
                    βd[I] = T(f(x))
                end
            end
        end
    end
    return coefs
end

# ----------------------------------------------------------------------------
# Apply: L φ = A·α·φ - B·∇·(β·∇φ)
# ----------------------------------------------------------------------------

"""
    apply_abec!(Lphi, phi, coefs, ws; level_range, skip_covered)

In-place apply: `Lphi := A·α·phi - B·∇·(β·∇phi)` on each cell of the
patches in `level_range`. Reads neighbor φ via the existing
`_neighbor_phi` / parent-halo machinery (so BCs and the Martin–Colella
C/F ghost are handled identically to the const-coef path).
"""
function apply_abec!(Lphi::Vector{Vector{NamedTuple}},
                     phi::Vector{Vector{NamedTuple}},
                     coefs::ABecCoefs{D, T},
                     ws::MGWorkspace{D, T};
                     level_range::UnitRange{Int} = ws.level_range,
                     skip_covered::Bool = false) where {D, T}
    for ℓ in level_range
        if ℓ == 1
            _apply_abec_root!(Lphi, phi, coefs, ws; skip_covered = skip_covered)
        else
            _apply_abec_fine!(Lphi, phi, coefs, ws, ℓ; skip_covered = skip_covered)
        end
    end
    return Lphi
end

function _apply_abec_root!(Lphi::Vector{Vector{NamedTuple}},
                            phi::Vector{Vector{NamedTuple}},
                            coefs::ABecCoefs{D, T},
                            ws::MGWorkspace{D, T};
                            skip_covered::Bool = false) where {D, T}
    pi = 1; ℓ = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    phi_raw = _raw_phi(phi[ℓ][pi])::Vector{T}
    Lphi_raw = _raw_phi(Lphi[ℓ][pi])::Vector{T}
    α = coefs.alpha[ℓ][pi]
    β = coefs.beta[ℓ][pi]
    A = coefs.A
    B = coefs.B
    kinds = ws.axis_kinds.kinds
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    covered = ws.covered_by_finer[ℓ][pi]

    @inbounds for I in CartesianIndices(N)
        c = c2c[I]
        c == 0 && continue
        skip_covered && covered[c] && continue
        phi_c = phi_raw[c]
        val = A * α[c] * phi_c
        for d in 1:D
            β_low  = β[d][I]
            β_high = β[d][I + _unit_offsets(Val(D))[d]]
            phi_high = _neighbor_phi(phi_raw, c2c, I, d, +1, N, kinds, phi_c)
            phi_low  = _neighbor_phi(phi_raw, c2c, I, d, -1, N, kinds, phi_c)
            div_d = (β_high * (phi_high - phi_c) -
                     β_low  * (phi_c    - phi_low)) * invh2[d]
            val -= B * div_d
        end
        Lphi_raw[c] = val
    end
    return Lphi
end

function _apply_abec_fine!(Lphi::Vector{Vector{NamedTuple}},
                            phi::Vector{Vector{NamedTuple}},
                            coefs::ABecCoefs{D, T},
                            ws::MGWorkspace{D, T}, ℓ::Int;
                            skip_covered::Bool = false) where {D, T}
    pi = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    pcell = ws.parent_cell[ℓ][pi]
    phi_raw = _raw_phi(phi[ℓ][pi])::Vector{T}
    Lphi_raw = _raw_phi(Lphi[ℓ][pi])::Vector{T}
    parent_phi_raw = _raw_phi(phi[ℓ - 1][1])::Vector{T}
    α = coefs.alpha[ℓ][pi]
    β = coefs.beta[ℓ][pi]
    A = coefs.A
    B = coefs.B
    kinds = ws.axis_kinds.kinds
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    covered = ws.covered_by_finer[ℓ][pi]
    has_finer = length(ws.ph.levels) > ℓ
    two_thirds = T(2//3)

    @inbounds for I in CartesianIndices(N)
        c = c2c[I]
        c == 0 && continue
        skip_covered && has_finer && covered[c] && continue
        phi_c = phi_raw[c]
        val = A * α[c] * phi_c
        for d in 1:D
            β_low  = β[d][I]
            β_high = β[d][I + _unit_offsets(Val(D))[d]]

            # High-d neighbor (face_idx for parent_cell lookup uses
            # convention: +d face is 2(d-1)+2, -d face is 2(d-1)+1).
            phi_high = _abec_fine_neighbor(I, c, +1, d, N, c2c, phi_raw,
                                            parent_phi_raw, pcell,
                                            kinds, phi_c, two_thirds)
            phi_low  = _abec_fine_neighbor(I, c, -1, d, N, c2c, phi_raw,
                                            parent_phi_raw, pcell,
                                            kinds, phi_c, two_thirds)

            div_d = (β_high * (phi_high - phi_c) -
                     β_low  * (phi_c    - phi_low)) * invh2[d]
            val -= B * div_d
        end
        Lphi_raw[c] = val
    end
    return Lphi
end

# Fine-level neighbor read: same-patch direct read, off-patch goes via
# Martin–Colella ghost (1/3 φ_c + 2/3 φ_parent), outer wall uses BC kind.
@inline function _abec_fine_neighbor(I::CartesianIndex{D}, c::Int,
                                       side::Int, d::Int,
                                       N::NTuple{D, Int},
                                       c2c::Array{Int, D},
                                       phi_raw::Vector{T},
                                       parent_phi_raw::Vector{T},
                                       pcell::Matrix{Int},
                                       kinds::NTuple{D, Symbol},
                                       phi_c::T, two_thirds::T) where {D, T}
    inew = I[d] + side
    if 1 <= inew <= N[d]
        nb_c = @inbounds c2c[I + side * _unit_offsets(Val(D))[d]]
        return @inbounds phi_raw[nb_c]
    end
    face_idx = side == +1 ? 2 * (d - 1) + 2 : 2 * (d - 1) + 1
    par_c = @inbounds pcell[c, face_idx]
    if par_c != 0
        # MC ghost: φ_n = (1/3) φ_c + (2/3) φ_parent.  Return that as the
        # effective neighbor value; the caller multiplies by β_face.
        return (one(T) - two_thirds) * phi_c +
               two_thirds * (@inbounds parent_phi_raw[par_c])
    end
    # Outer-domain wall.
    kinds[d] === :dirichlet && return -phi_c
    return phi_c   # Neumann or periodic-without-parent → reflect
end

# ----------------------------------------------------------------------------
# Red-black Gauss-Seidel smoother for the ABec operator.
# ----------------------------------------------------------------------------

"""
    gs_sweep_abec!(phi, rho, coefs, ws, ℓ; n_sweeps, reverse_colours, skip_covered)

In-place RB-GS relaxation of `L φ = ρ` on level ℓ, where `L` is the ABec
operator defined by `coefs`. The relaxed cell-center update is

    φ_c := (B/h² Σ_d (β₊ φ₊ + β₋ φ₋) + ρ_c) /
           (A α_c + B/h² Σ_d (β₊ + β₋))      —  rearranged from
           A α_c φ_c - B (β₊(φ₊-φ_c) - β₋(φ_c-φ₋))/h² = ρ_c

Boundary handling (same-patch / parent halo / outer wall) and covered-cell
skipping follow the const-coef GS path.
"""
function gs_sweep_abec!(phi::Vector{Vector{NamedTuple}},
                         rho::Vector{Vector{NamedTuple}},
                         coefs::ABecCoefs{D, T},
                         ws::MGWorkspace{D, T}, ℓ::Int;
                         n_sweeps::Int = 1,
                         reverse_colours::Bool = false,
                         skip_covered::Bool = false) where {D, T}
    _refresh_rho_flat!(ws, rho, ℓ)
    if ℓ == 1
        _gs_abec_root!(phi, coefs, ws; n_sweeps = n_sweeps,
                        reverse_colours = reverse_colours,
                        skip_covered = skip_covered)
    else
        _gs_abec_fine!(phi, coefs, ws, ℓ; n_sweeps = n_sweeps,
                        reverse_colours = reverse_colours,
                        skip_covered = skip_covered)
    end
    return phi
end

function _gs_abec_root!(phi::Vector{Vector{NamedTuple}},
                         coefs::ABecCoefs{D, T},
                         ws::MGWorkspace{D, T};
                         n_sweeps::Int = 1,
                         reverse_colours::Bool = false,
                         skip_covered::Bool = false) where {D, T}
    pi = 1; ℓ = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    phi_raw = _raw_phi(phi[ℓ][pi])::Vector{T}
    rho_flat = ws.rho_flat[ℓ][pi]
    α = coefs.alpha[ℓ][pi]
    β = coefs.beta[ℓ][pi]
    A = coefs.A; B = coefs.B
    kinds = ws.axis_kinds.kinds
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    covered = ws.covered_by_finer[ℓ][pi]

    colour_order = reverse_colours ? (1:-1:0) : (0:1)

    for _ in 1:n_sweeps
        for colour in colour_order
            @inbounds for I in CartesianIndices(N)
                s = 0
                for d in 1:D; s += I[d]; end
                (s & 1) == colour || continue
                c = c2c[I]
                c == 0 && continue
                skip_covered && covered[c] && continue
                phi_c = phi_raw[c]
                num = rho_flat[c]
                den = A * α[c]
                for d in 1:D
                    β_low  = β[d][I]
                    β_high = β[d][I + _unit_offsets(Val(D))[d]]
                    phi_high = _neighbor_phi(phi_raw, c2c, I, d, +1, N, kinds, phi_c)
                    phi_low  = _neighbor_phi(phi_raw, c2c, I, d, -1, N, kinds, phi_c)
                    num += B * (β_high * phi_high + β_low * phi_low) * invh2[d]
                    den += B * (β_high + β_low) * invh2[d]
                end
                phi_raw[c] = num / den
            end
        end
    end
    return phi
end

function _gs_abec_fine!(phi::Vector{Vector{NamedTuple}},
                         coefs::ABecCoefs{D, T},
                         ws::MGWorkspace{D, T}, ℓ::Int;
                         n_sweeps::Int = 1,
                         reverse_colours::Bool = false,
                         skip_covered::Bool = false) where {D, T}
    pi = 1
    N = ws.patch_N[ℓ][pi]
    dx = ws.patch_dx[ℓ][pi]
    c2c = ws.cart_to_cell[ℓ][pi]
    pcell = ws.parent_cell[ℓ][pi]
    phi_raw = _raw_phi(phi[ℓ][pi])::Vector{T}
    parent_phi_raw = _raw_phi(phi[ℓ - 1][1])::Vector{T}
    rho_flat = ws.rho_flat[ℓ][pi]
    α = coefs.alpha[ℓ][pi]
    β = coefs.beta[ℓ][pi]
    A = coefs.A; B = coefs.B
    kinds = ws.axis_kinds.kinds
    invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
    covered = ws.covered_by_finer[ℓ][pi]
    has_finer = length(ws.ph.levels) > ℓ
    two_thirds = T(2//3)

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
                num = rho_flat[c]
                den = A * α[c]
                for d in 1:D
                    β_low  = β[d][I]
                    β_high = β[d][I + _unit_offsets(Val(D))[d]]
                    # high-side neighbor
                    inew_h = I[d] + 1
                    if 1 <= inew_h <= N[d]
                        nb_c = c2c[I + _unit_offsets(Val(D))[d]]
                        num += B * β_high * phi_raw[nb_c] * invh2[d]
                        den += B * β_high * invh2[d]
                    else
                        face_idx = 2 * (d - 1) + 2
                        par_c = pcell[c, face_idx]
                        if par_c != 0
                            # MC ghost: ghost = (1/3) phi_c + (2/3) phi_par
                            # Contribution: B β_high · ghost / h².
                            # Split: (B β_high / h²) · (2/3) φ_p → num,
                            # plus (B β_high / h²) · (1/3) φ_c → effectively
                            # diag−den. But we put +(2/3) phi_p in num
                            # and decrement den by the (1/3) phi_c part.
                            # Net effect on the GS equation:
                            #   ... A α φ_c - B β_high (φ_ghost - φ_c)/h² ...
                            # = ... A α φ_c - B β_high ((1/3 φ_c + 2/3 φ_p) - φ_c)/h² ...
                            # = ... A α φ_c - B β_high (2/3) (φ_p - φ_c)/h² ...
                            # ⇒ coefficient on φ_c gains (2/3) B β_high / h²,
                            #   constant (RHS) gains -(2/3) B β_high φ_p / h².
                            # Rearranging for GS update (num + ... = den · φ_c):
                            #   num += (2/3) B β_high φ_p / h²
                            #   den += (2/3) B β_high / h²
                            num += two_thirds * B * β_high * parent_phi_raw[par_c] * invh2[d]
                            den += two_thirds * B * β_high * invh2[d]
                        else
                            # Outer wall.  Dirichlet: ghost = -φ_c, so
                            # -B β_high(-φ_c - φ_c)/h² = 2 B β_high φ_c/h²
                            # ⇒ den += 2 B β_high / h².  Neumann/periodic:
                            # ghost = φ_c, contribution is zero.
                            if kinds[d] === :dirichlet
                                den += 2 * B * β_high * invh2[d]
                            end
                        end
                    end
                    # low-side neighbor
                    inew_l = I[d] - 1
                    if 1 <= inew_l <= N[d]
                        nb_c = c2c[I - _unit_offsets(Val(D))[d]]
                        num += B * β_low * phi_raw[nb_c] * invh2[d]
                        den += B * β_low * invh2[d]
                    else
                        face_idx = 2 * (d - 1) + 1
                        par_c = pcell[c, face_idx]
                        if par_c != 0
                            num += two_thirds * B * β_low * parent_phi_raw[par_c] * invh2[d]
                            den += two_thirds * B * β_low * invh2[d]
                        else
                            if kinds[d] === :dirichlet
                                den += 2 * B * β_low * invh2[d]
                            end
                        end
                    end
                end
                phi_raw[c] = num / den
            end
        end
    end
    return phi
end

# ----------------------------------------------------------------------------
# Residual: r = ρ - L φ
# ----------------------------------------------------------------------------

"""
    compute_abec_residual!(r, phi, rho, coefs, ws; level_range)

`r := rho - L_ABec φ` in-place. Uses `apply_abec!` for the operator.
"""
function compute_abec_residual!(r::Vector{Vector{NamedTuple}},
                                  phi::Vector{Vector{NamedTuple}},
                                  rho::Vector{Vector{NamedTuple}},
                                  coefs::ABecCoefs{D, T},
                                  ws::MGWorkspace{D, T};
                                  level_range::UnitRange{Int} = ws.level_range) where {D, T}
    apply_abec!(r, phi, coefs, ws; level_range = level_range)
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

# ----------------------------------------------------------------------------
# V-cycle for the ABec operator.  Same recursion structure as the const-coef
# vcycle! (textbook FAC, ABC98) — pre-smooth, FAC SYNC on phi (outer only),
# residual, restrict, recurse, prolong, post-smooth.  Bottom solver is a
# Jacobi-preconditioned CG since FFT only works for the const-coef Poisson
# case.
# ----------------------------------------------------------------------------

"""
    vcycle_abec!(phi, rho, coefs, ws; level_range, backend)

One geometric V-cycle for the ABec operator.  Same shape as `vcycle!` but
the apply / smooth / bottom paths consume `coefs`.  Convergence regime
matches `vcycle!`: stable on 2-level AMR (textbook), stagnates on 3+
without an outer Krylov wrap.  For deep hierarchies, use
`pcg_composite_abec_solve!`.
"""
function vcycle_abec!(phi::Vector{Vector{NamedTuple}},
                       rho::Vector{Vector{NamedTuple}},
                       coefs::ABecCoefs{D, T},
                       ws::MGWorkspace{D, T};
                       level_range::UnitRange{Int} = ws.level_range,
                       backend = Sequential()) where {D, T}
    ℓ_hi = last(level_range)
    ℓ_lo = first(level_range)

    if ℓ_hi == ℓ_lo
        # Bottom solver: many GS sweeps (no FFT for variable-coef ABec).
        gs_sweep_abec!(phi, rho, coefs, ws, ℓ_hi;
                        n_sweeps = ws.opts.bottom_smooth_iters,
                        backend = backend)
        return phi
    end

    gs_sweep_abec!(phi, rho, coefs, ws, ℓ_hi; n_sweeps = ws.opts.n_pre,
                    reverse_colours = false, backend = backend)

    is_outer = phi !== ws.correction
    if is_outer
        # FAC SYNC: phi[ℓ_hi-1][covered] := volume-average phi[ℓ_hi].
        restrict_to_parents!(phi[ℓ_hi - 1], phi[ℓ_hi], ws.ph;
                              level = ℓ_hi, fieldname = :phi)
    end

    apply_abec!(ws.residual, phi, coefs, ws; level_range = ℓ_hi:ℓ_hi)
    @inbounds for pi in 1:length(ws.residual[ℓ_hi])
        rf = ws.residual[ℓ_hi][pi].phi
        rh = rho[ℓ_hi][pi].rho
        for c in ws.patch_leaves[ℓ_hi][pi]
            _set_val!(rf, c, _get_val(rh, c) - _get_val(rf, c))
        end
    end

    if is_outer
        apply_abec!(ws.residual, phi, coefs, ws;
                     level_range = (ℓ_hi - 1):(ℓ_hi - 1),
                     skip_covered = true)
        @inbounds for pi in 1:length(ws.residual[ℓ_hi - 1])
            rf = ws.residual[ℓ_hi - 1][pi].phi
            rh = rho[ℓ_hi - 1][pi].rho
            covered = ws.covered_by_finer[ℓ_hi - 1][pi]
            for c in ws.patch_leaves[ℓ_hi - 1][pi]
                covered[c] && continue
                _set_val!(rf, c, _get_val(rh, c) - _get_val(rf, c))
            end
        end
    end

    # Build rhs_coarse[ℓ_hi-1].
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
    restrict_to_parents!(ws.rhs_coarse[ℓ_hi - 1], ws.residual[ℓ_hi],
                          ws.ph; level = ℓ_hi, fieldname = :phi)
    @inbounds for pi in 1:length(ws.rhs_coarse[ℓ_hi - 1])
        src = ws.rhs_coarse[ℓ_hi - 1][pi].phi
        dst = ws.rhs_coarse[ℓ_hi - 1][pi].rho
        for c in ws.patch_leaves[ℓ_hi - 1][pi]
            _set_val!(dst, c, _get_val(src, c))
        end
    end

    # Zero correction at lower levels.
    @inbounds for ℓ in ℓ_lo:(ℓ_hi - 1)
        for pi in 1:length(ws.correction[ℓ])
            f = ws.correction[ℓ][pi].phi
            for c in ws.patch_leaves[ℓ][pi]
                _set_val!(f, c, zero(T))
            end
        end
    end

    vcycle_abec!(ws.correction, ws.rhs_coarse, coefs, ws;
                  level_range = ℓ_lo:(ℓ_hi - 1), backend = backend)

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

    gs_sweep_abec!(phi, rho, coefs, ws, ℓ_hi; n_sweeps = ws.opts.n_post,
                    reverse_colours = true, backend = backend)
    return phi
end

# ----------------------------------------------------------------------------
# PCG on the ABec composite system (Jacobi-preconditioned).
#
# A := +L_ABec  (which is SPD positive-definite when A>0 with α>0 and B>0
# with β>0; reduces to the existing -L_FAC = -∇² SPD-positive case for
# pure Poisson with A=0, B=-1).  In either sign convention, the operator
# is SPD in the V-weighted inner product, so CG works.
# ----------------------------------------------------------------------------

"""
    pcg_composite_abec_solve!(phi, rho, coefs, ws; tol, maxiter, level_range)

Preconditioned CG on the ABec composite system.  Returns `MGResult`.
Same V-weighted-norm structure as `pcg_composite_solve!`.
"""
function pcg_composite_abec_solve!(phi::Vector{Vector{NamedTuple}},
                                     rho::Vector{Vector{NamedTuple}},
                                     coefs::ABecCoefs{D, T},
                                     ws::MGWorkspace{D, T};
                                     tol::Float64 = 1e-10,
                                     maxiter::Int = 200,
                                     verbose::Bool = false,
                                     level_range::UnitRange{Int} = ws.level_range,
                                     backend = default_backend()) where {D, T}
    z_fields = allocate_phi_rho(ws.ph)

    # r := ρ - L φ
    apply_abec!(ws.residual, phi, coefs, ws; level_range = level_range)
    @inbounds for ℓ in level_range
        for pi in 1:length(ws.residual[ℓ])
            rf = _raw_phi(ws.residual[ℓ][pi])::Vector{T}
            rh = _raw_rho(rho[ℓ][pi])::Vector{T}
            covered = ws.covered_by_finer[ℓ][pi]
            for c in ws.patch_leaves[ℓ][pi]
                covered[c] && continue
                rf[c] = rh[c] - rf[c]
            end
        end
    end
    r0 = _composite_norm_V(ws.residual, ws; level_range = level_range)
    history = Float64[r0]
    if r0 == 0
        return MGResult(0, r0, r0, true, history)
    end

    _abec_jacobi_precond!(z_fields, ws.residual, coefs, ws;
                           level_range = level_range)

    _composite_zero_phi!(ws.correction, ws; level_range = level_range)
    _composite_axpy_phi!(ws.correction, z_fields, one(T), ws;
                          level_range = level_range)
    rs_old = _composite_dot_V(ws.residual, z_fields, ws;
                                level_range = level_range)

    converged = false
    r = r0
    for iter in 1:maxiter
        # Ap = L p
        apply_abec!(ws.rhs_coarse, ws.correction, coefs, ws;
                     level_range = level_range)
        pAp = _composite_dot_V(ws.correction, ws.rhs_coarse, ws;
                                 level_range = level_range)
        if pAp <= 0
            verbose && @warn "PCG-ABec: non-positive pAp=$pAp at iter $iter"
            break
        end
        α = T(rs_old / pAp)

        _composite_axpy_phi!(phi, ws.correction, α, ws;
                              level_range = level_range)
        _composite_axpy_phi!(ws.residual, ws.rhs_coarse, -α, ws;
                              level_range = level_range)
        r = _composite_norm_V(ws.residual, ws; level_range = level_range)
        push!(history, r)
        verbose && @info "PCG-ABec iter $iter: |r|_V = $r  (reduction = $(r/r0))"
        if r <= tol * r0 || r <= tol
            converged = true; break
        end

        _abec_jacobi_precond!(z_fields, ws.residual, coefs, ws;
                               level_range = level_range)
        rs_new = _composite_dot_V(ws.residual, z_fields, ws;
                                    level_range = level_range)
        β = T(rs_new / rs_old)
        rs_old = rs_new
        _composite_y_eq_x_plus_beta_y!(ws.correction, z_fields, β, ws;
                                         level_range = level_range)
    end
    return MGResult(length(history) - 1, r0, r, converged, history)
end

# Jacobi preconditioner: z[c] = r[c] / diag(L)[c].
function _abec_jacobi_precond!(z_fields, r_fields,
                                 coefs::ABecCoefs{D, T},
                                 ws::MGWorkspace{D, T};
                                 level_range = ws.level_range) where {D, T}
    A = coefs.A; B = coefs.B
    @inbounds for ℓ in level_range
        for pi in 1:length(z_fields[ℓ])
            r = _raw_phi(r_fields[ℓ][pi])::Vector{T}
            z = _raw_phi(z_fields[ℓ][pi])::Vector{T}
            dx = ws.patch_dx[ℓ][pi]
            covered = ws.covered_by_finer[ℓ][pi]
            invh2 = ntuple(d -> one(T) / (dx[d] * dx[d]), Val(D))
            α = coefs.alpha[ℓ][pi]
            β = coefs.beta[ℓ][pi]
            c2c = ws.cart_to_cell[ℓ][pi]
            N = ws.patch_N[ℓ][pi]
            for I in CartesianIndices(N)
                c = c2c[I]
                c == 0 && continue
                covered[c] && (z[c] = zero(T); continue)
                diag = A * α[c]
                for d in 1:D
                    β_low = β[d][I]
                    β_high = β[d][I + _unit_offsets(Val(D))[d]]
                    # Conservative diagonal estimate: A·α + B/h² Σ (β_low+β_high)
                    diag += B * (β_high + β_low) * invh2[d]
                end
                z[c] = diag == 0 ? zero(T) : r[c] / diag
            end
        end
    end
    return z_fields
end

# ----------------------------------------------------------------------------
# Top-level solve.
# ----------------------------------------------------------------------------

"""
    solve_abec!(phi, rho, coefs, ws; tol, maxiter, verbose, level_range)

Solve `A α φ - B ∇·(β ∇φ) = ρ` to relative tolerance `tol`.  Returns an
`MGResult`.  Internally calls `pcg_composite_abec_solve!` which handles
both pure Poisson (A=0, B=-1) and positive-definite Helmholtz (A>0, B>0)
cases via the V-weighted inner product.
"""
function solve_abec!(phi::Vector{Vector{NamedTuple}},
                      rho::Vector{Vector{NamedTuple}},
                      coefs::ABecCoefs{D, T},
                      ws::MGWorkspace{D, T};
                      tol::Float64 = ws.opts.tol,
                      maxiter::Int = ws.opts.maxiter,
                      verbose::Bool = ws.opts.verbose,
                      level_range::UnitRange{Int} = ws.level_range,
                      backend = Sequential()) where {D, T}
    return pcg_composite_abec_solve!(phi, rho, coefs, ws;
                                       tol = tol, maxiter = maxiter,
                                       verbose = verbose,
                                       level_range = level_range,
                                       backend = backend)
end

# Convenience wrapper for the case where φ and ρ share a `fields` container.
solve_abec!(ws::MGWorkspace, fields::Vector, coefs::ABecCoefs;
             kwargs...) = solve_abec!(fields, fields, coefs, ws; kwargs...)
