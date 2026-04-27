using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells,
    EulerianFrame, FrameBoundaries, enumerate_leaves,
    BCKind, PERIODIC, REFLECTING,
    MonomialBasis, n_coeffs, allocate_polynomial_fields, SoA,
    Sequential, OhMyThreadsBackend,
    PatchHierarchy, PatchBoundaryBC, PatchView, PatchHaloView,
    add_patches!, for_each_patch!,
    restrict_to_parents!, prolong_from_parents!,
    step_patch_pipeline!,
    cell_physical_box, is_leaf

# Pull the unexported accessor + validator from the Solver submodule so
# we can run them directly from tests.
const Solver = HierarchicalGrids.Solver
const ph_validate = Solver.validate
const ph_n_levels = Solver.n_levels
const ph_n_patches = Solver.n_patches
const ph_patches_at = Solver.patches_at

# ============================================================================
# Helpers: build a uniform NxN mesh on a box [lo, hi] in 2D
# ============================================================================

function _make_uniform_2d_frame(levels::Int,
                                  lo::NTuple{2, Float64},
                                  hi::NTuple{2, Float64})
    mesh = HierarchicalMesh{2}()
    for _ in 1:levels
        leaves = enumerate_leaves(mesh)
        refine_cells!(mesh, leaves)
    end
    return EulerianFrame(mesh, lo, hi)
end

function _make_field(frame::EulerianFrame{2, Float64};
                      basis = MonomialBasis{2, 0}(),
                      fields = (:rho,))
    n = n_cells(frame.mesh)
    pairs = [nm => Float64 for nm in fields]
    pfs = allocate_polynomial_fields(SoA(), basis, n; pairs...)
    # Zero-initialise — the underlying SoA Vector is allocated `undef`.
    nc = n_coeffs(basis)
    for nm in fields
        fv = getproperty(pfs, nm)
        for i in 1:n
            fv[i] = ntuple(_ -> 0.0, nc)
        end
    end
    return pfs
end

function _views(pfs, names = (:rho,))
    return NamedTuple{names}(map(n -> getproperty(pfs, n), names))
end

# Bulk-set every cell's :rho constant to a given value.
function _fill_constant!(pfs, value::Float64; fieldname::Symbol = :rho)
    field = getproperty(pfs, fieldname)
    nc = n_coeffs(pfs.basis)
    for i in 1:length(field)
        field[i] = ntuple(k -> k == 1 ? value : 0.0, nc)
    end
    return pfs
end

# Set per-cell values from a function of cell index.
function _fill_per_index!(pfs, f; fieldname::Symbol = :rho)
    field = getproperty(pfs, fieldname)
    nc = n_coeffs(pfs.basis)
    for i in 1:length(field)
        field[i] = ntuple(k -> k == 1 ? Float64(f(i)) : 0.0, nc)
    end
    return pfs
end

# ============================================================================
# 1. Construction + validate
# ============================================================================

@testset "PatchHierarchy: construction and validate" begin
    base = _make_uniform_2d_frame(2, (0.0, 0.0), (1.0, 1.0))
    ph = PatchHierarchy(base)
    @test ph_n_levels(ph) == 1
    @test ph_n_patches(ph, 1) == 1

    fine = _make_uniform_2d_frame(1, (0.25, 0.25), (0.75, 0.75))
    add_patches!(ph, 2, [fine])
    @test ph_n_levels(ph) == 2
    @test ph_n_patches(ph, 2) == 1
    @test ph_validate(ph) === nothing
end

@testset "PatchHierarchy: validate rejects out-of-base patch" begin
    base = _make_uniform_2d_frame(0, (0.0, 0.0), (1.0, 1.0))
    bad  = _make_uniform_2d_frame(0, (0.5, 0.5), (1.5, 1.5))   # extends past hi=1
    ph = PatchHierarchy(base)
    add_patches!(ph, 2, [bad])
    @test_throws ArgumentError ph_validate(ph)
end

@testset "PatchHierarchy: validate accepts patch sharing base wall" begin
    base = _make_uniform_2d_frame(0, (0.0, 0.0), (1.0, 1.0))
    edge = _make_uniform_2d_frame(0, (0.0, 0.25), (0.5, 0.75))   # touches lo wall
    ph = PatchHierarchy(base)
    add_patches!(ph, 2, [edge])
    @test ph_validate(ph) === nothing
end

# ============================================================================
# 2. Conservative restriction (degree-0)
# ============================================================================

@testset "restrict_to_parents!: fine constant 1.0 fills covered coarse cells" begin
    # Base: 2×2 (4 leaves) on [0, 1]^2; fine: 2×2 on [0.25, 0.75]^2.
    # The fine patch sits inside ALL four coarse cells (each coarse cell's
    # quadrant of the unit square has nonzero overlap with the centred
    # half-square).
    base_frame = _make_uniform_2d_frame(1, (0.0, 0.0), (1.0, 1.0))
    fine_frame = _make_uniform_2d_frame(1, (0.25, 0.25), (0.75, 0.75))

    base_pfs = _make_field(base_frame)
    fine_pfs = _make_field(fine_frame)

    # Initialise base cells to a known pattern (so we can observe whether
    # uncovered cells stay put; in this geometry every base leaf IS covered).
    _fill_per_index!(base_pfs, i -> 100 + i)
    _fill_constant!(fine_pfs, 1.0)

    ph = PatchHierarchy(base_frame)
    add_patches!(ph, 2, [fine_frame])

    restrict_to_parents!([_views(base_pfs)], [_views(fine_pfs)],
                          ph; level = 2, fieldname = :rho)

    # Every base leaf is partially covered by the centred fine patch, so
    # every base leaf should now hold the fine field's constant value 1.0.
    # The root cell (non-leaf, index 1) is not a leaf and is left at its
    # initial value.
    n_covered_leaves = 0
    for ci in 1:n_cells(base_frame.mesh)
        if HierarchicalGrids.is_leaf(base_frame.mesh.cells[ci])
            @test base_pfs.rho[ci][1] ≈ 1.0
            n_covered_leaves += 1
        end
    end
    @test n_covered_leaves == 4
end

@testset "restrict_to_parents!: uncovered coarse cells keep original value" begin
    # Base: 4×4 (16 leaves) on [0, 1]^2; fine: 1 leaf on [0.0, 0.25]^2,
    # which sits entirely inside the lower-left coarse cell at level-1 (only
    # one base leaf is covered).
    base_frame = _make_uniform_2d_frame(2, (0.0, 0.0), (1.0, 1.0))   # 16 leaves
    fine_frame = _make_uniform_2d_frame(0, (0.0, 0.0), (0.25, 0.25))  # 1 leaf

    base_pfs = _make_field(base_frame)
    fine_pfs = _make_field(fine_frame)

    # Distinct per-leaf values in the base.
    _fill_per_index!(base_pfs, i -> 100 + i)
    _fill_constant!(fine_pfs, 7.0)

    ph = PatchHierarchy(base_frame)
    add_patches!(ph, 2, [fine_frame])

    # Snapshot pre-restrict values for every base leaf.
    pre = [base_pfs.rho[i][1] for i in 1:n_cells(base_frame.mesh)]

    restrict_to_parents!([_views(base_pfs)], [_views(fine_pfs)],
                          ph; level = 2, fieldname = :rho)

    # Find the base leaves the fine patch covers (only the lower-left quadrant
    # of the 4×4 grid). The covered base cells are exactly those whose
    # physical box overlaps [0.0, 0.25]^2 with positive measure.
    covered = Int[]
    for ci in 1:n_cells(base_frame.mesh)
        HierarchicalGrids.is_leaf(base_frame.mesh.cells[ci]) || continue
        lo, hi = HierarchicalGrids.cell_physical_box(base_frame, ci)
        ovx = max(0.0, min(hi[1], 0.25) - max(lo[1], 0.0))
        ovy = max(0.0, min(hi[2], 0.25) - max(lo[2], 0.0))
        if ovx > 0.0 && ovy > 0.0
            push!(covered, ci)
        end
    end
    @test !isempty(covered)
    # Covered cells should have value 7.0 (constant fine field).
    for ci in covered
        @test base_pfs.rho[ci][1] ≈ 7.0
    end
    # Uncovered leaves should retain pre-restrict value.
    for ci in 1:n_cells(base_frame.mesh)
        HierarchicalGrids.is_leaf(base_frame.mesh.cells[ci]) || continue
        ci in covered && continue
        @test base_pfs.rho[ci][1] == pre[ci]
    end
end

# ============================================================================
# 3. Constant prolongation
# ============================================================================

@testset "prolong_from_parents!: fine cells inherit covering parent value" begin
    base_frame = _make_uniform_2d_frame(1, (0.0, 0.0), (1.0, 1.0))    # 4 leaves
    fine_frame = _make_uniform_2d_frame(1, (0.0, 0.0), (1.0, 1.0))    # 4 leaves at level 2

    base_pfs = _make_field(base_frame)
    fine_pfs = _make_field(fine_frame)

    # Distinct values per coarse leaf.
    _fill_per_index!(base_pfs, i -> 10.0 * i)
    _fill_constant!(fine_pfs, -1.0)

    ph = PatchHierarchy(base_frame)
    add_patches!(ph, 2, [fine_frame])

    prolong_from_parents!([_views(fine_pfs)], [_views(base_pfs)],
                           ph; level = 2, fieldname = :rho)

    # Every fine leaf should now equal its covering base leaf's value.
    for fci in 1:n_cells(fine_frame.mesh)
        HierarchicalGrids.is_leaf(fine_frame.mesh.cells[fci]) || continue
        f_lo, f_hi = HierarchicalGrids.cell_physical_box(fine_frame, fci)
        # Find the base leaf containing the fine cell's centre.
        cx = 0.5 * (f_lo[1] + f_hi[1]); cy = 0.5 * (f_lo[2] + f_hi[2])
        base_match = 0
        for bci in 1:n_cells(base_frame.mesh)
            HierarchicalGrids.is_leaf(base_frame.mesh.cells[bci]) || continue
            b_lo, b_hi = HierarchicalGrids.cell_physical_box(base_frame, bci)
            if b_lo[1] <= cx <= b_hi[1] && b_lo[2] <= cy <= b_hi[2]
                base_match = bci; break
            end
        end
        @test base_match != 0
        @test fine_pfs.rho[fci][1] ≈ base_pfs.rho[base_match][1]
    end
end

# ============================================================================
# 4. Round-trip restrict → prolong preserves a constant fine field
# ============================================================================

@testset "round-trip restrict → prolong: constant fine field preserved" begin
    base_frame = _make_uniform_2d_frame(1, (0.0, 0.0), (1.0, 1.0))    # 4 leaves
    fine_frame = _make_uniform_2d_frame(1, (0.0, 0.0), (1.0, 1.0))    # 4 leaves

    base_pfs = _make_field(base_frame)
    fine_pfs = _make_field(fine_frame)

    c = 3.14
    _fill_constant!(fine_pfs, c)

    ph = PatchHierarchy(base_frame)
    add_patches!(ph, 2, [fine_frame])

    restrict_to_parents!([_views(base_pfs)], [_views(fine_pfs)],
                          ph; level = 2, fieldname = :rho)

    # Now zap the fine and prolong from the (just-restricted) parent.
    _fill_constant!(fine_pfs, -999.0)
    prolong_from_parents!([_views(fine_pfs)], [_views(base_pfs)],
                           ph; level = 2, fieldname = :rho)

    for fci in 1:n_cells(fine_frame.mesh)
        HierarchicalGrids.is_leaf(fine_frame.mesh.cells[fci]) || continue
        @test fine_pfs.rho[fci][1] ≈ c
    end
end

# ============================================================================
# 5. for_each_patch!: kernel runs at given level only
# ============================================================================

@testset "for_each_patch!: doubles every cell's :rho at the chosen level" begin
    base_frame = _make_uniform_2d_frame(1, (0.0, 0.0), (1.0, 1.0))    # 4 leaves
    fine_frame = _make_uniform_2d_frame(1, (0.25, 0.25), (0.75, 0.75)) # 4 leaves

    base_pfs_in  = _make_field(base_frame)
    base_pfs_out = _make_field(base_frame)
    fine_pfs_in  = _make_field(fine_frame)
    fine_pfs_out = _make_field(fine_frame)

    _fill_per_index!(base_pfs_in,  i -> 1.0 * i)
    _fill_per_index!(fine_pfs_in,  i -> 100.0 * i)

    ph = PatchHierarchy(base_frame)
    add_patches!(ph, 2, [fine_frame])

    kernel = function (pv::PatchView, hv::PatchHaloView, ctx)
        old = pv[:rho]
        pv[:rho] = ntuple(k -> 2.0 * old[k], length(old))
        return nothing
    end

    # Run at level 1: only the base patch is updated.
    for_each_patch!(kernel,
                     [_views(base_pfs_out)], [_views(base_pfs_in)],
                     ph; level = 1, ghost_depth = 1,
                     backend = Sequential())

    for ci in 1:n_cells(base_frame.mesh)
        HierarchicalGrids.is_leaf(base_frame.mesh.cells[ci]) || continue
        @test base_pfs_out.rho[ci][1] ≈ 2.0 * ci
    end
    # Fine outputs untouched.
    for ci in 1:n_cells(fine_frame.mesh)
        @test fine_pfs_out.rho[ci][1] == 0.0
    end

    # Run at level 2 with the parent-fields plumbed in.
    for_each_patch!(kernel,
                     [_views(fine_pfs_out)], [_views(fine_pfs_in)],
                     ph; level = 2, ghost_depth = 1,
                     fields_in_parent = [_views(base_pfs_in)],
                     backend = Sequential())

    for ci in 1:n_cells(fine_frame.mesh)
        HierarchicalGrids.is_leaf(fine_frame.mesh.cells[ci]) || continue
        @test fine_pfs_out.rho[ci][1] ≈ 200.0 * ci
    end
end

# ============================================================================
# 6. Patch halo at internal edges
# ============================================================================

@testset "PatchHaloView: internal-edge offset reads parent value" begin
    # Base: a 2×2 grid on [0, 1]^2 with distinct per-leaf values.
    # Fine patch: a 2×2 grid on [0.25, 0.75]^2 (interior — does not touch
    # any base wall). The +x face of the fine patch's right column lies
    # inside the base's right column; the halo offset (1, 0) from a
    # fine cell on that face must report the parent's value.
    base_frame = _make_uniform_2d_frame(1, (0.0, 0.0), (1.0, 1.0))
    fine_frame = _make_uniform_2d_frame(1, (0.25, 0.25), (0.75, 0.75))

    base_pfs = _make_field(base_frame)
    fine_pfs = _make_field(fine_frame)

    # Distinct, recognisable per-base-leaf values.
    _fill_per_index!(base_pfs, i -> 10.0 * i)
    _fill_constant!(fine_pfs, 0.0)

    # Find the fine leaf whose physical box lies in the upper-right of the
    # patch (so a +x hop walks off the patch into the base's right side).
    fine_target = 0
    for ci in 1:n_cells(fine_frame.mesh)
        HierarchicalGrids.is_leaf(fine_frame.mesh.cells[ci]) || continue
        lo, hi = HierarchicalGrids.cell_physical_box(fine_frame, ci)
        if hi[1] >= 0.74 && lo[2] <= 0.5  # right column, lower row
            fine_target = ci; break
        end
    end
    @test fine_target != 0

    ph = PatchHierarchy(base_frame)
    add_patches!(ph, 2, [fine_frame])

    # Find the parent base leaf the +x ghost lands in. The ghost cell
    # centre is the right-column fine cell's centre + extent_x. With the
    # fine patch at [0.25, 0.75]² the right-column fine cell centre is
    # (0.625, ...). Extent is 0.25, so the ghost centre is at (0.875, ...),
    # which lies inside the base's right column.
    f_lo, f_hi = HierarchicalGrids.cell_physical_box(fine_frame, fine_target)
    extent_x = f_hi[1] - f_lo[1]
    ghost_cx = (f_lo[1] + f_hi[1]) / 2 + extent_x
    ghost_cy = (f_lo[2] + f_hi[2]) / 2
    expected_base = 0
    for bci in 1:n_cells(base_frame.mesh)
        HierarchicalGrids.is_leaf(base_frame.mesh.cells[bci]) || continue
        b_lo, b_hi = HierarchicalGrids.cell_physical_box(base_frame, bci)
        if b_lo[1] <= ghost_cx <= b_hi[1] && b_lo[2] <= ghost_cy <= b_hi[2]
            expected_base = bci; break
        end
    end
    @test expected_base != 0
    expected_value = base_pfs.rho[expected_base][1]

    captured_values = Float64[]
    captured_indices = Int[]
    kernel = function (pv::PatchView, hv::PatchHaloView, ctx)
        if pv.cv.index == fine_target
            pview = hv[:rho, (1, 0)]
            push!(captured_values, pview[1])
            push!(captured_indices, pv.cv.index)
        end
        return nothing
    end

    for_each_patch!(kernel,
                     [_views(fine_pfs)], [_views(fine_pfs)],
                     ph; level = 2, ghost_depth = 1,
                     fields_in_parent = [_views(base_pfs)],
                     backend = Sequential())

    @test length(captured_values) == 1
    @test captured_values[1] ≈ expected_value
end

@testset "PatchHaloView: same-patch offset behaves like CellView neighbour" begin
    # A 2×2 fine patch (4 leaves). For a cell whose +x neighbour is also
    # in the patch, hv[:rho, (1, 0)] must return that neighbour's value
    # (NOT a parent lookup).
    base_frame = _make_uniform_2d_frame(0, (0.0, 0.0), (1.0, 1.0))   # 1 base leaf
    fine_frame = _make_uniform_2d_frame(1, (0.25, 0.25), (0.75, 0.75)) # 4 fine leaves

    base_pfs = _make_field(base_frame)
    fine_pfs = _make_field(fine_frame)

    _fill_per_index!(fine_pfs, i -> 10.0 + i)
    _fill_constant!(base_pfs, 999.0)

    ph = PatchHierarchy(base_frame)
    add_patches!(ph, 2, [fine_frame])

    # Find a fine leaf that is in the LEFT column of the patch (so its +x
    # neighbour is also a fine leaf on the same patch).
    leaves = Int[]
    for ci in 1:n_cells(fine_frame.mesh)
        HierarchicalGrids.is_leaf(fine_frame.mesh.cells[ci]) || continue
        lo, hi = HierarchicalGrids.cell_physical_box(fine_frame, ci)
        if hi[1] <= 0.51   # left column (right edge x = 0.5)
            push!(leaves, ci)
        end
    end
    @test !isempty(leaves)
    target = leaves[1]

    captured = Ref{Float64}(NaN)
    kernel = function (pv::PatchView, hv::PatchHaloView, ctx)
        if pv.cv.index == target
            pview = hv[:rho, (1, 0)]
            captured[] = pview[1]
        end
        return nothing
    end

    for_each_patch!(kernel,
                     [_views(fine_pfs)], [_views(fine_pfs)],
                     ph; level = 2, ghost_depth = 1,
                     fields_in_parent = [_views(base_pfs)],
                     backend = Sequential())

    # Find the +x same-patch neighbour and check the read matches it (NOT
    # the base's 999.0).
    f_lo, f_hi = HierarchicalGrids.cell_physical_box(fine_frame, target)
    nb_cx = (f_lo[1] + f_hi[1]) / 2 + (f_hi[1] - f_lo[1])
    nb_cy = (f_lo[2] + f_hi[2]) / 2
    nb_idx = 0
    for ci in 1:n_cells(fine_frame.mesh)
        HierarchicalGrids.is_leaf(fine_frame.mesh.cells[ci]) || continue
        lo, hi = HierarchicalGrids.cell_physical_box(fine_frame, ci)
        if lo[1] <= nb_cx <= hi[1] && lo[2] <= nb_cy <= hi[2]
            nb_idx = ci; break
        end
    end
    @test nb_idx != 0
    @test captured[] ≈ fine_pfs.rho[nb_idx][1]
    @test captured[] != 999.0
end

# ============================================================================
# 7. step_patch_pipeline! — Berger-Oliger correct ordering
# ============================================================================

# Snapshot every cell's :rho constant coefficient into a Vector{Float64}.
function _snapshot_rho(pfs)
    return [pfs.rho[i][1] for i in 1:length(pfs.rho)]
end

# Doubling kernel: rho_out = 2 * rho_in. Level-agnostic.
const _double_kernel = function (pv::PatchView, hv::PatchHaloView, ctx)
    old = pv[:rho]
    pv[:rho] = ntuple(k -> 2.0 * old[k], length(old))
    return nothing
end

# Identity kernel: rho_out = rho_in. Used for mass-conservation tests.
const _identity_kernel = function (pv::PatchView, hv::PatchHaloView, ctx)
    pv[:rho] = pv[:rho]
    return nothing
end

@testset "step_patch_pipeline!: matches manual prolong/fine/coarse/restrict" begin
    # Two-level hierarchy: coarse 2×2 base, fine 2×2 patch on the centred
    # half-square. The fine patch covers all four base leaves with positive
    # but not full overlap.
    base_frame = _make_uniform_2d_frame(1, (0.0, 0.0), (1.0, 1.0))
    fine_frame = _make_uniform_2d_frame(1, (0.25, 0.25), (0.75, 0.75))

    # Build TWO copies of the field-sets so we can run the pipeline helper on
    # one and the manual sequence on the other, then byte-compare.
    function _build_fields()
        base_in  = _make_field(base_frame); _fill_per_index!(base_in,  i -> 1.0 + 0.5 * i)
        base_out = _make_field(base_frame)
        fine_in  = _make_field(fine_frame); _fill_per_index!(fine_in,  i -> 10.0 + 0.25 * i)
        fine_out = _make_field(fine_frame)
        return (base_in, base_out, fine_in, fine_out)
    end

    A_base_in, A_base_out, A_fine_in, A_fine_out = _build_fields()
    B_base_in, B_base_out, B_fine_in, B_fine_out = _build_fields()

    ph = PatchHierarchy(base_frame)
    add_patches!(ph, 2, [fine_frame])

    # Manual sequence on copy A — replicates the example's `step!`.
    prolong_from_parents!([_views(A_fine_in)], [_views(A_base_in)],
                           ph; level = 2, fieldname = :rho)
    for_each_patch!(_double_kernel,
                     [_views(A_fine_out)], [_views(A_fine_in)],
                     ph; level = 2, ghost_depth = 1,
                     fields_in_parent = [_views(A_base_in)],
                     backend = Sequential())
    for_each_patch!(_double_kernel,
                     [_views(A_base_out)], [_views(A_base_in)],
                     ph; level = 1, ghost_depth = 1,
                     backend = Sequential())
    restrict_to_parents!([_views(A_base_out)], [_views(A_fine_out)],
                          ph; level = 2, fieldname = :rho)

    # Helper on copy B.
    fields_in  = [[_views(B_base_in)],  [_views(B_fine_in)]]
    fields_out = [[_views(B_base_out)], [_views(B_fine_out)]]
    step_patch_pipeline!(_double_kernel, fields_out, fields_in, ph;
                          ghost_depth = 1, backend = Sequential())

    # Byte-equal field-set comparison (rho only).
    @test _snapshot_rho(A_base_in)  == _snapshot_rho(B_base_in)
    @test _snapshot_rho(A_base_out) == _snapshot_rho(B_base_out)
    @test _snapshot_rho(A_fine_in)  == _snapshot_rho(B_fine_in)
    @test _snapshot_rho(A_fine_out) == _snapshot_rho(B_fine_out)
end

@testset "step_patch_pipeline!: identity kernel preserves total mass" begin
    # An identity kernel (rho_out = rho_in) on a hierarchy where the fine
    # patch exactly tiles a contiguous block of coarse cells. Total mass
    # on the coarse base patch must be preserved to round-off after one
    # pipeline step (the prolong/restrict are degree-0 conservative on a
    # cell-aligned fine patch; the identity update changes nothing).
    base_frame = _make_uniform_2d_frame(2, (0.0, 0.0), (1.0, 1.0))   # 4×4 = 16 leaves
    # Fine patch on the lower-left quadrant: covers exactly 4 coarse cells.
    fine_frame = _make_uniform_2d_frame(2, (0.0, 0.0), (0.5, 0.5))   # 4×4 = 16 fine leaves

    base_in  = _make_field(base_frame); _fill_per_index!(base_in,  i -> Float64(i))
    base_out = _make_field(base_frame); _fill_per_index!(base_out, i -> Float64(i))
    fine_in  = _make_field(fine_frame)
    fine_out = _make_field(fine_frame)

    ph = PatchHierarchy(base_frame)
    add_patches!(ph, 2, [fine_frame])

    # Seed the fine input from the parent (so we start in a consistent
    # state — same value as constant prolongation would produce).
    prolong_from_parents!([_views(fine_in)], [_views(base_in)],
                           ph; level = 2, fieldname = :rho)
    # Mirror into fine_out so the post-step restrict sees the consistent value.
    for i in 1:n_cells(fine_frame.mesh)
        fine_out.rho[i] = fine_in.rho[i]
    end

    # Total coarse mass = Σ rho · V over base leaves.
    function _coarse_mass(pfs)
        s = 0.0
        for ci in 1:n_cells(base_frame.mesh)
            is_leaf(base_frame.mesh.cells[ci]) || continue
            lo, hi = cell_physical_box(base_frame, ci)
            v = (hi[1] - lo[1]) * (hi[2] - lo[2])
            s += pfs.rho[ci][1] * v
        end
        return s
    end

    mass_before = _coarse_mass(base_in)

    fields_in  = [[_views(base_in)],  [_views(fine_in)]]
    fields_out = [[_views(base_out)], [_views(fine_out)]]
    step_patch_pipeline!(_identity_kernel, fields_out, fields_in, ph;
                          ghost_depth = 1, backend = Sequential())

    mass_after = _coarse_mass(base_out)

    # Identity update + cell-aligned conservative restrict ⇒ mass invariant
    # to round-off.
    @test isapprox(mass_after, mass_before; atol = 1e-12,
                    rtol = 1e-12)
end
