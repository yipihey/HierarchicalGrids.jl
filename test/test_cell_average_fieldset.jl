using Test
using HierarchicalGrids
using HierarchicalGrids: SoA, AoS, Blocked,
    CellAverageFieldSet, allocate_cell_average_fields, CellAverageFieldView,
    PointSampleFieldSet, allocate_point_sample_fields,
    n_elements, field_names,
    HierarchicalMesh, split_mask_type, refine_cells!, coarsen_cells!,
    AdaptiveField, dispose!, n_cells, is_leaf, enumerate_leaves,
    EulerianFrame, cell_physical_box,
    for_each_cell!, Sequential, halo_view_multi,
    FrameBoundaries, PERIODIC

# Sum of value × volume over the leaf cells — the finite-volume conserved
# quantity (total mass when the field is a density).
function _total_mass(af, frame)
    m = 0.0
    mesh = af.mesh
    for i in 1:n_cells(mesh)
        is_leaf(mesh.cells[i]) || continue
        lo, hi = cell_physical_box(frame, i)
        vol = prod(hi .- lo)
        m += af.field.rho[i] * vol
    end
    return m
end

# A NamedTuple of per-field views, the binding the orchestrator / halo
# accessors consume.
_views(fs, names) = NamedTuple{names}(map(n -> getproperty(fs, n), names))

# ============================================================================
# 1. Allocation, queries, and storage size across layouts
# ============================================================================

@testset "CellAverageFieldSet: allocation and queries" begin
    fields = allocate_cell_average_fields(SoA(), Val(2), 4; rho = Float64)
    @test fields isa CellAverageFieldSet
    @test n_elements(fields) == 4
    @test field_names(fields) == (:rho,)

    # SoA storage: a NamedTuple with one Vector of length 4 (one scalar/cell).
    @test fields.storage isa NamedTuple
    @test length(fields.storage.rho) == 4

    # Multiple fields of mixed scalar types (NamedTuple keyword interface).
    fields2 = allocate_cell_average_fields(SoA(), Val(3), 5; rho = Float64, p = Float32)
    @test field_names(fields2) == (:rho, :p)
    @test eltype(fields2.storage.rho) == Float64
    @test eltype(fields2.storage.p)   == Float32

    # D must be ≥ 1.
    @test_throws ArgumentError allocate_cell_average_fields(SoA(), Val(0), 4; rho = Float64)
end

# ============================================================================
# 2. Per-cell read/write and bulk assign across SoA / AoS / Blocked
# ============================================================================

@testset "CellAverageFieldSet: read/write and bulk assign (SoA, AoS, Blocked{4})" begin
    n = 16
    for layout in (SoA(), AoS(), Blocked{4}())
        fields = allocate_cell_average_fields(layout, Val(2), n; rho = Float64)
        @test n_elements(fields) == n

        # Per-cell write of the scalar cell average.
        for i in 1:n
            fields.rho[i] = Float64(10 * i + 1)
        end
        for i in 1:n
            @test fields.rho[i] == Float64(10 * i + 1)
        end

        # Single-cell overwrite leaves neighbors untouched.
        fields.rho[7] = -3.0
        @test fields.rho[7] == -3.0
        @test fields.rho[6] == Float64(61)
        @test fields.rho[8] == Float64(81)

        # Bulk readback via collect / iteration.
        c = collect(fields.rho)
        @test length(c) == n
        @test c[6] == 61.0
        @test sum(fields.rho) == sum(c)
    end
end

# ============================================================================
# 3. Layout invariance — the same kernel gives identical results
# ============================================================================

@testset "CellAverageFieldSet: layout-invariant kernel results" begin
    # A simple per-cell update applied through the public accessor; the result
    # must be byte-identical regardless of the underlying memory layout.
    kernel!(fields) = for i in 1:n_elements(fields)
        fields.rho[i] = 2.0 * fields.rho[i] + 1.0
    end

    n = 32
    results = Vector{Vector{Float64}}()
    for layout in (SoA(), AoS(), Blocked{8}())
        fields = allocate_cell_average_fields(layout, Val(3), n; rho = Float64)
        for i in 1:n
            fields.rho[i] = Float64(i) / 7
        end
        kernel!(fields)
        push!(results, collect(fields.rho))
    end
    @test results[1] == results[2]
    @test results[2] == results[3]
end

# ============================================================================
# 4. Property-access errors
# ============================================================================

@testset "CellAverageFieldSet: property-access errors" begin
    fields = allocate_cell_average_fields(SoA(), Val(2), 4; rho = Float64)
    @test_throws KeyError fields.nonexistent
    @test_throws ArgumentError fields.rho = [1.0]   # whole-field assign forbidden
end

# ============================================================================
# 5. Distinction from PointSampleFieldSet{…, N=1}
# ============================================================================

@testset "CellAverageFieldSet: distinct type from a 1-point PointSampleFieldSet" begin
    ca  = allocate_cell_average_fields(SoA(), Val(2), 4; rho = Float64)
    ps1 = allocate_point_sample_fields(SoA(), Val(2), Val(1), 4; rho = Float64)
    # Same storage footprint (one scalar per cell) but semantically different
    # models: a volume average vs. a point value, with conservative-mean vs.
    # interpolatory coarsening. The types must NOT be conflated.
    @test ca isa CellAverageFieldSet
    @test ps1 isa PointSampleFieldSet
    @test !(ca isa PointSampleFieldSet)
    @test typeof(ca) !== typeof(ps1)
end

# ============================================================================
# 6. Conservative refinement: prolongation (injection) + restriction (mean)
# ============================================================================

@testset "CellAverageFieldSet: prolongation is injection (D=2)" begin
    mesh = HierarchicalMesh{2}()
    field = allocate_cell_average_fields(SoA(), Val(2), 1; rho = Float64)
    field.rho[1] = 3.5
    af = AdaptiveField(field, mesh)

    refine_cells!(mesh, [1])
    @test n_cells(mesh) == 5            # parent + 4 isotropic children
    @test af.field.n == 5
    @test af.field.rho[1] == 3.5        # parent slot retains its value
    for child in 2:5
        @test af.field.rho[child] == 3.5   # each child inherits by injection
    end
    dispose!(af)
end

@testset "CellAverageFieldSet: restriction is the volume-weighted mean (D=2)" begin
    mesh = HierarchicalMesh{2}()
    field = allocate_cell_average_fields(AoS(), Val(2), 1; rho = Float64)
    field.rho[1] = 0.0
    af = AdaptiveField(field, mesh)

    refine_cells!(mesh, [1])
    af.field.rho[2] = 2.0
    af.field.rho[3] = 4.0
    af.field.rho[4] = 6.0
    af.field.rho[5] = 8.0

    coarsen_cells!(mesh, [1])
    @test n_cells(mesh) == 1
    @test af.field.n == 1
    @test af.field.rho[1] == 5.0       # (2+4+6+8)/4, equal-volume children
    dispose!(af)
end

# ============================================================================
# 7. Conservation invariant: Σ value × volume preserved to round-off
# ============================================================================

@testset "CellAverageFieldSet: refine/coarsen conserves Σ value×volume" begin
    for layout in (SoA(), AoS())
        mesh = HierarchicalMesh{2}()
        field = allocate_cell_average_fields(layout, Val(2), 1; rho = Float64)
        # Non-unit, anisotropic physical spacing so per-cell volumes genuinely
        # differ across levels and the volume weighting is exercised.
        frame = EulerianFrame(mesh, (0.0, 0.0), (2.0, 3.0))
        af = AdaptiveField(field, mesh)
        af.field.rho[1] = 1.0

        # Build a multi-level leaf set, then assign distinct averages.
        refine_cells!(mesh, [1])          # cells 2..5 (level 1)
        refine_cells!(mesh, [2])          # cells 6..9 (level 2)
        for (k, i) in enumerate(enumerate_leaves(mesh))
            af.field.rho[i] = Float64(k) + 0.25
        end
        total0 = _total_mass(af, frame)

        # Coarsen the level-2 block back: parent = mean of its 4 children.
        coarsen_cells!(mesh, [2])
        @test isapprox(_total_mass(af, frame), total0; rtol = 1e-13)

        # Refine a different leaf and coarsen again — still conserved.
        first_leaf = enumerate_leaves(mesh)[1]
        refine_cells!(mesh, [first_leaf])
        @test isapprox(_total_mass(af, frame), total0; rtol = 1e-13)

        dispose!(af)
    end
end

@testset "CellAverageFieldSet: anisotropic-split restriction conserves mass" begin
    mesh = HierarchicalMesh{2}()
    M = split_mask_type(Val(2))
    field = allocate_cell_average_fields(SoA(), Val(2), 1; rho = Float64)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    af = AdaptiveField(field, mesh)
    af.field.rho[1] = 0.0

    # Anisotropic split along axis 1 only → exactly 2 equal-volume children.
    refine_cells!(mesh, [1], [M(0b001)])
    @test n_cells(mesh) == 3
    af.field.rho[2] = 4.0
    af.field.rho[3] = 10.0
    total0 = _total_mass(af, frame)

    coarsen_cells!(mesh, [1])
    @test n_cells(mesh) == 1
    @test af.field.rho[1] == 7.0          # mean over the 2 children present
    @test isapprox(_total_mass(af, frame), total0; rtol = 1e-13)
    dispose!(af)
end

# ============================================================================
# 8. Orchestrator round-trip — for_each_cell! over a cell-average field
# ============================================================================

@testset "CellAverageFieldSet: for_each_cell! round-trip" begin
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    refine_cells!(mesh, enumerate_leaves(mesh))
    n = n_cells(mesh)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))

    fin  = allocate_cell_average_fields(SoA(), Val(2), n; rho = Float64)
    fout = allocate_cell_average_fields(SoA(), Val(2), n; rho = Float64)
    for i in 1:n
        fin.rho[i]  = 0.0
        fout.rho[i] = 0.0
    end
    leaves = enumerate_leaves(mesh)
    for (pos, i) in enumerate(leaves)
        fin.rho[i] = Float64(pos)
    end

    fin_v  = _views(fin,  (:rho,))
    fout_v = _views(fout, (:rho,))

    # The kernel sees the cell average as a plain scalar (no sub-view).
    kernel = function (cv, hv, ctx)
        cv[Val(:rho)] = 2.0 * cv[Val(:rho)]
        return nothing
    end

    for_each_cell!(kernel, fout_v, fin_v, frame; ghost_depth = 0,
                   backend = Sequential())

    for (pos, i) in enumerate(leaves)
        @test fout.rho[i] == 2.0 * Float64(pos)
    end
end

# ============================================================================
# 9. Halo / BC-resolved neighbor read over a cell-average field
# ============================================================================

@testset "CellAverageFieldSet: halo BC-resolved neighbor reads (1D periodic)" begin
    mesh = HierarchicalMesh{1}()
    refine_cells!(mesh, [1])              # cells 2, 3 are the two leaves
    n = n_cells(mesh)
    fields = allocate_cell_average_fields(SoA(), Val(1), n; rho = Float64)
    fields.rho[2] = 10.0
    fields.rho[3] = 20.0
    fv = _views(fields, (:rho,))

    bcs = FrameBoundaries(((PERIODIC, PERIODIC),))

    hv = halo_view_multi(fv, mesh, 2, 1; bcs = bcs)
    @test hv[Val(:rho), (1,)] == 20.0     # interior +x neighbor
    @test hv[Val(:rho), (-1,)] == 20.0    # -x hits boundary → periodic wrap

    hv3 = halo_view_multi(fv, mesh, 3, 1; bcs = bcs)
    @test hv3[Val(:rho), (-1,)] == 10.0   # interior -x neighbor
    @test hv3[Val(:rho), (1,)] == 10.0    # +x hits boundary → periodic wrap

    # Without BCs, a boundary hop is unresolved (returns nothing).
    hv_nobc = halo_view_multi(fv, mesh, 2, 1; bcs = nothing)
    @test hv_nobc[Val(:rho), (1,)] == 20.0
    @test hv_nobc[Val(:rho), (-1,)] === nothing
end

# ============================================================================
# 10. show() is informative
# ============================================================================

@testset "CellAverageFieldSet: show is non-empty" begin
    fields = allocate_cell_average_fields(SoA(), Val(2), 4; rho = Float64)
    s = sprint(show, fields)
    @test occursin("CellAverageFieldSet", s)
    @test occursin("D=2", s)
    @test occursin("rho", s)
end
