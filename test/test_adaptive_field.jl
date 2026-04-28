using Test
using Logging
using Random
using HierarchicalGrids
using HierarchicalGrids: MonomialBasis, BernsteinBasis, allocate_polynomial_fields, SoA,
                          HierarchicalMesh, refine_cells!, coarsen_cells!,
                          AdaptiveField, dispose!, n_cells, n_coeffs,
                          register_refinement_listener!,
                          unregister_refinement_listener!,
                          EulerianFrame, init_field_from!,
                          cell_physical_box, evaluate, is_leaf,
                          enumerate_leaves, moment_multiindices, moments_length

@testset "AdaptiveField" begin

    @testset "Construction registers a listener" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = (1.0,)

        n_listeners_before = length(mesh._listeners)
        af = AdaptiveField(field, mesh)
        @test length(mesh._listeners) == n_listeners_before + 1
        @test af.disposed == false
        @test af.listener_handle != 0
        @test parent(af) === af.field

        # Cleanup
        dispose!(af)
    end

    @testset "Refine — constant prolongation (degree 0, D=2)" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = (3.5,)
        af = AdaptiveField(field, mesh)

        refine_cells!(mesh, [1])
        @test n_cells(mesh) == 5  # 1 parent + 4 children for D=2 isotropic
        @test af.field.n == 5

        # Parent (now non-leaf) retains its coefficient
        @test af.field.rho[1][1] == 3.5
        # Each child inherits the parent's value
        for child in 2:5
            @test af.field.rho[child][1] == 3.5
        end

        dispose!(af)
    end

    @testset "Coarsen — degree-0 average" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = (0.0,)
        af = AdaptiveField(field, mesh)

        refine_cells!(mesh, [1])
        # Set the four children to 2, 4, 6, 8.
        af.field.rho[2] = (2.0,)
        af.field.rho[3] = (4.0,)
        af.field.rho[4] = (6.0,)
        af.field.rho[5] = (8.0,)

        coarsen_cells!(mesh, [1])
        @test n_cells(mesh) == 1
        @test af.field.n == 1
        # (2+4+6+8)/4 = 5
        @test af.field.rho[1][1] == 5.0

        dispose!(af)
    end

    @testset "Refine then coarsen round-trip (degree 0, bit-exact)" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = (7.0,)
        af = AdaptiveField(field, mesh)

        refine_cells!(mesh, [1])
        coarsen_cells!(mesh, [1])
        # Each child got 7.0 by constant prolongation; mean of four 7s is 7.
        @test af.field.rho[1][1] === 7.0

        dispose!(af)
    end

    @testset "dispose! unregisters and stops field updates" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = (1.0,)
        af = AdaptiveField(field, mesh)

        n_before = length(mesh._listeners)
        dispose!(af)
        @test length(mesh._listeners) == n_before - 1
        @test af.disposed == true

        # Capture the (frozen) field n before refinement.
        frozen_n = af.field.n
        @test frozen_n == 1

        # Refining now should NOT touch the wrapped field, because the
        # listener has been unregistered.
        refine_cells!(mesh, [1])
        @test n_cells(mesh) == 5
        # af.field is still the original (n=1) PolynomialFieldSet.
        @test af.field.n == frozen_n

        # Calling dispose! again is a no-op.
        dispose!(af)
        @test af.disposed == true
    end

    @testset "Higher-degree coarsening — BernsteinBasis takes L²-projection path" begin
        # BernsteinBasis is now routed through change_basis + the
        # MonomialBasis L²-projection path. The deferred-fallback warning
        # must NOT fire for degree ≥ 1 BernsteinBasis fields.
        mesh = HierarchicalMesh{2}()
        basis = BernsteinBasis{2, 2}()  # degree 2 in 2D
        nc = n_coeffs(basis)
        @test nc > 1

        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        field.rho[1] = ntuple(_ -> 0.0, nc)
        af = AdaptiveField(field, mesh)

        refine_cells!(mesh, [1])
        for i in 2:5
            af.field.rho[i] = ntuple(k -> k == 1 ? 1.0 : Float64(i + k), nc)
        end

        # Capture all log messages and assert none of them contains the
        # deferred-fallback marker. Using `min_level=Logging.Warn` ensures
        # we observe @warn calls.
        msgs = @test_logs min_level=Logging.Warn coarsen_cells!(mesh, [1])
        @test af.field.n == 1
        # The L²-projection path produces non-trivial coefficients (the
        # higher Bernstein coefficients should not all be zero, as they
        # would under the constant-moment-only fallback).
        coeffs_after = collect(af.field.rho[1])
        @test any(!iszero, coeffs_after[2:end])

        dispose!(af)
    end

    @testset "L²-projection coarsening — linear field bit-exact recovery (D=2)" begin
        # A degree-1 MonomialBasis field initialized to a linear function
        # f(x, y) = a + b*x + c*y must round-trip refine→coarsen to itself
        # exactly: linear-in-children L²-projects back to the same linear-in-parent.
        #
        # Setup detail: AdaptiveField uses constant prolongation on refinement,
        # which would corrupt a non-constant field across the refine step. To
        # isolate the *coarsening* projection from the *refinement* prolongation,
        # we re-init the field after refinement so each child holds the exact
        # piecewise-polynomial restriction of f, then coarsen.
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 1}()  # 1, x, y on the unit cube reference frame
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))

        # Pick a generic linear function so coefficients aren't trivially zero.
        a, b, c = 0.7, -1.3, 2.1
        f = x -> a + b * x[1] + c * x[2]
        init_field_from!(field, frame, f)

        # Save root coefficients for later comparison. Single cell, [0, 1]^2,
        # so reference == physical and these are exactly (a, b, c).
        root_coeffs_before = collect(field.rho[1])
        @test length(root_coeffs_before) == 3
        @test isapprox(root_coeffs_before[1], a; atol=1e-12)
        @test isapprox(root_coeffs_before[2], b; atol=1e-12)
        @test isapprox(root_coeffs_before[3], c; atol=1e-12)

        af = AdaptiveField(field, mesh)
        refine_cells!(mesh, [1])  # split root into 4 children
        # Re-initialize children + parent slot with the analytic function to
        # neutralize the (lossy) constant-prolongation step.
        init_field_from!(af.field, frame, f)
        # Coarsen and verify recovery of the original root coefficients.
        coarsen_cells!(mesh, [1])
        @test af.field.n == 1
        coeffs_after = collect(af.field.rho[1])
        @test isapprox(coeffs_after[1], root_coeffs_before[1]; atol=1e-12, rtol=1e-12)
        @test isapprox(coeffs_after[2], root_coeffs_before[2]; atol=1e-12, rtol=1e-12)
        @test isapprox(coeffs_after[3], root_coeffs_before[3]; atol=1e-12, rtol=1e-12)

        dispose!(af)
    end

    @testset "L²-projection coarsening — quadratic moment preservation (D=2)" begin
        # A degree-2 MonomialBasis field. For a polynomial of degree ≤ 2,
        # the L²-projection onto MonomialBasis{2, 2} restricted to children
        # and re-projected onto the parent is the IDENTITY: round-trip
        # recovers the original parent's coefficients to round-off.
        #
        # We re-init the children with the analytic function after refining
        # to neutralize the (lossy) constant-prolongation refinement step
        # — this isolates the coarsening's L²-projection contribution.
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 2}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))

        # Generic quadratic: f(x, y) = 0.4 + 1.1*x - 0.6*y + 0.7*x^2 + 0.3*x*y - 0.8*y^2
        f = x -> 0.4 + 1.1 * x[1] - 0.6 * x[2] +
                  0.7 * x[1]^2 + 0.3 * x[1] * x[2] - 0.8 * x[2]^2
        init_field_from!(field, frame, f)
        root_before = collect(field.rho[1])

        af = AdaptiveField(field, mesh)
        refine_cells!(mesh, [1])
        init_field_from!(af.field, frame, f)
        coarsen_cells!(mesh, [1])
        root_after = collect(af.field.rho[1])
        # Each coefficient bit-equal to round-off.
        for k in 1:length(root_before)
            @test isapprox(root_after[k], root_before[k]; atol=1e-11, rtol=1e-11)
        end

        # Stress with two refinement levels (4 → 13 cells), then re-init
        # all leaves with f and coarsen back. Round-trip should still recover.
        refine_cells!(mesh, [1])
        refine_cells!(mesh, [3])  # refine one of the children further
        init_field_from!(af.field, frame, f)
        coarsen_cells!(mesh, [3])
        coarsen_cells!(mesh, [1])
        root_after2 = collect(af.field.rho[1])
        for k in 1:length(root_before)
            @test isapprox(root_after2[k], root_before[k]; atol=1e-10, rtol=1e-10)
        end

        dispose!(af)
    end

    @testset "L²-projection coarsening — mass conservation under random fuzz" begin
        # Random refine/coarsen sequences must preserve total mass to round-off
        # for a degree-2 field. Total mass = ∫ p dV = Σ_cells (∫_cell p dV)
        # = Σ_cells (cell_volume * c_const), since the constant-moment of a
        # MonomialBasis polynomial on the unit-cube reference frame integrates
        # to c_const * cell_volume.
        rng = MersenneTwister(123)
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 2}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        f = x -> 1.5 + 0.3 * x[1] + 0.2 * x[2] + 0.1 * x[1]^2 - 0.05 * x[2]^2
        init_field_from!(field, frame, f)
        af = AdaptiveField(field, mesh)

        function total_mass(af, frame)
            m = 0.0
            mesh = af.mesh
            for i in 1:n_cells(mesh)
                if is_leaf(mesh.cells[i])
                    lo, hi = cell_physical_box(frame, i)
                    vol = (hi[1] - lo[1]) * (hi[2] - lo[2])
                    # On the unit-cube reference frame, ∫ p dξ = Σ_α c_α / ∏_d (α_d + 1).
                    # In physical x, ∫ p dx = vol * ∫ p dξ.
                    coeffs = collect(af.field.rho[i])
                    multi = moment_multiindices(2, 2)
                    integ = 0.0
                    for k in eachindex(multi)
                        integ += coeffs[k] / ((multi[k][1] + 1) * (multi[k][2] + 1))
                    end
                    m += vol * integ
                end
            end
            return m
        end

        m0 = total_mass(af, frame)
        @test isapprox(m0, 1.5 + 0.3*0.5 + 0.2*0.5 + 0.1/3 - 0.05/3; atol=1e-12)

        # 30 random refine/coarsen events. Track mass drift.
        max_drift = 0.0
        for _ in 1:30
            leaves = enumerate_leaves(af.mesh)
            if rand(rng) < 0.6 && !isempty(leaves)
                # Refine a random leaf
                target = leaves[rand(rng, 1:length(leaves))]
                refine_cells!(af.mesh, [target])
            else
                # Coarsen a random non-leaf parent (one whose children are all leaves)
                candidates = Int[]
                for i in 1:n_cells(af.mesh)
                    cell = af.mesh.cells[i]
                    if !is_leaf(cell)
                        nc = HierarchicalGrids.children_count(cell)
                        all_leaves = true
                        for k in 1:nc
                            ci = i + k  # children are contiguous starting at i+1 in DFS for first-children
                            # but only the FIRST child is at i+1 in DFS layout; later
                            # children are at i+1+subtree_size(...). Use find_children.
                        end
                        kids = HierarchicalGrids.find_children(af.mesh, i)
                        all_leaves = all(is_leaf(af.mesh.cells[k]) for k in kids)
                        if all_leaves
                            push!(candidates, i)
                        end
                    end
                end
                if !isempty(candidates)
                    target = candidates[rand(rng, 1:length(candidates))]
                    coarsen_cells!(af.mesh, [target])
                end
            end
            m = total_mass(af, frame)
            drift = abs(m - m0)
            if drift > max_drift
                max_drift = drift
            end
        end
        # Expect very small drift from accumulated round-off in the per-event
        # mass-matrix solves. Tight bound: ~1e-12 * (number of events).
        @test max_drift < 1e-10

        dispose!(af)
    end

    @testset "L²-projection coarsening — degree comparison" begin
        # Same physical field initialized at degrees 0, 1, 2. After
        # refine→coarsen, the higher-degree fields must reproduce the
        # original parent better (smaller L² error against the analytic
        # projection of the same function).
        function abs_l2_error(af, frame, f, basis_degree)
            err2 = 0.0
            mesh = af.mesh
            # Crude midpoint-rule estimate; sufficient to compare the
            # *relative* fidelity of degree-0 vs degree-1 vs degree-2.
            # Increase resolution if we ever need a tighter bound.
            for i in 1:n_cells(mesh)
                is_leaf(mesh.cells[i]) || continue
                lo, hi = cell_physical_box(frame, i)
                # 4×4 sample grid per cell.
                Nq = 4
                cell_err = 0.0
                cell_vol = (hi[1] - lo[1]) * (hi[2] - lo[2])
                coeffs = collect(af.field.rho[i])
                for ix in 1:Nq, iy in 1:Nq
                    ξ = ((ix - 0.5) / Nq, (iy - 0.5) / Nq)
                    x = (lo[1] + ξ[1] * (hi[1] - lo[1]),
                         lo[2] + ξ[2] * (hi[2] - lo[2]))
                    p_val = evaluate(MonomialBasis{2, basis_degree}(), coeffs, ξ)
                    cell_err += (p_val - f(x))^2
                end
                err2 += cell_err / (Nq * Nq) * cell_vol
            end
            return sqrt(err2)
        end

        # A non-trivial smooth function so degrees 0/1/2 are distinguishable.
        f = x -> sin(x[1]) * cos(x[2]) + 0.3 * x[1]^2 - 0.1 * x[2]^2

        function setup_and_coarsen(degree)
            mesh = HierarchicalMesh{2}()
            basis = MonomialBasis{2, degree}()
            field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
            frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
            af = AdaptiveField(field, mesh)
            refine_cells!(mesh, [1])
            # Re-initialize the children to the analytic function (they got
            # constant prolongation from a single cell, which is too crude
            # for a fair comparison).
            init_field_from!(af.field, frame, f)
            coarsen_cells!(mesh, [1])
            return af, frame
        end

        af0, fr0 = setup_and_coarsen(0)
        af1, fr1 = setup_and_coarsen(1)
        af2, fr2 = setup_and_coarsen(2)

        e0 = abs_l2_error(af0, fr0, f, 0)
        e1 = abs_l2_error(af1, fr1, f, 1)
        e2 = abs_l2_error(af2, fr2, f, 2)

        # Higher-degree should be strictly better. Allow a small slack for
        # the crude quadrature in `abs_l2_error`.
        @test e1 < e0
        @test e2 < e1

        dispose!(af0)
        dispose!(af1)
        dispose!(af2)
    end

    @testset "Degree-0 path bit-exact (regression)" begin
        # The degree-0 fast path was bit-exact before this PR; ensure it
        # remains bit-exact (not just isapprox-equal).
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        af = AdaptiveField(field, mesh)
        # Set known floating-point values that will produce a reproducible
        # arithmetic mean.
        refine_cells!(mesh, [1])
        af.field.rho[2] = (0.1,)
        af.field.rho[3] = (0.2,)
        af.field.rho[4] = (0.3,)
        af.field.rho[5] = (0.4,)
        coarsen_cells!(mesh, [1])
        # Bit-exact: this is the same average computation as before this PR.
        expected = (0.1 + 0.2 + 0.3 + 0.4) / 4.0
        @test af.field.rho[1][1] === expected
        dispose!(af)
    end

    @testset "Base.parent(af) returns the underlying field" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        af = AdaptiveField(field, mesh)
        @test parent(af) === af.field
        # After a refinement the wrapper is rebuilt; parent should track it.
        refine_cells!(mesh, [1])
        @test parent(af) === af.field
        dispose!(af)
    end

    @testset "Multi-field, multiple refinements (D=2)" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(SoA(), basis, 1;
                                            rho=Float64, p=Float64)
        field.rho[1] = (1.0,)
        field.p[1]   = (10.0,)
        af = AdaptiveField(field, mesh)

        refine_cells!(mesh, [1])
        @test af.field.n == 5
        @test all(af.field.rho[i][1] == 1.0 for i in 1:5)
        @test all(af.field.p[i][1]   == 10.0 for i in 1:5)

        # Refine one of the children (cell 3).
        refine_cells!(mesh, [3])
        @test af.field.n == n_cells(mesh)
        # All cells should still have the original values (constant
        # prolongation).
        for i in 1:n_cells(mesh)
            @test af.field.rho[i][1] == 1.0
            @test af.field.p[i][1]   == 10.0
        end

        dispose!(af)
    end

    @testset "AoS layout supported" begin
        mesh = HierarchicalMesh{2}()
        basis = MonomialBasis{2, 0}()
        field = allocate_polynomial_fields(HierarchicalGrids.AoS(), basis, 1;
                                            rho=Float64)
        field.rho[1] = (2.5,)
        af = AdaptiveField(field, mesh)
        refine_cells!(mesh, [1])
        @test af.field.n == 5
        for i in 1:5
            @test af.field.rho[i][1] == 2.5
        end
        coarsen_cells!(mesh, [1])
        @test af.field.n == 1
        @test af.field.rho[1][1] == 2.5
        dispose!(af)
    end

    # ------------------------------------------------------------------
    # BernsteinBasis L²-projection coarsening tests (PR-4)
    # ------------------------------------------------------------------

    @testset "Bernstein L²-projection — linear field bit-exact recovery (D=2)" begin
        # Mirrors the MonomialBasis test. A degree-1 BernsteinBasis field
        # set to a known linear function. After refine→re-init→coarsen,
        # the parent's polynomial recovers the linear function to round-off.
        mesh = HierarchicalMesh{2}()
        basis = BernsteinBasis{2, 1}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))

        a, b, c = 0.7, -1.3, 2.1
        f = x -> a + b * x[1] + c * x[2]
        init_field_from!(field, frame, f)
        root_before = collect(field.rho[1])

        af = AdaptiveField(field, mesh)
        refine_cells!(mesh, [1])
        init_field_from!(af.field, frame, f)
        coarsen_cells!(mesh, [1])
        @test af.field.n == 1
        coeffs_after = collect(af.field.rho[1])
        for k in eachindex(root_before)
            @test isapprox(coeffs_after[k], root_before[k]; atol=1e-11, rtol=1e-11)
        end

        dispose!(af)
    end

    @testset "Bernstein L²-projection — mass conservation under random fuzz (D=2, P=2)" begin
        # Random refine/coarsen sequences must preserve total mass to
        # round-off for a degree-2 BernsteinBasis field. We compute the
        # mass via Bernstein → Monomial conversion (the monomial form
        # admits the closed-form integral on the unit cube reference).
        rng = MersenneTwister(456)
        mesh = HierarchicalMesh{2}()
        basis = BernsteinBasis{2, 2}()
        field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        f = x -> 1.5 + 0.3 * x[1] + 0.2 * x[2] + 0.1 * x[1]^2 - 0.05 * x[2]^2
        init_field_from!(field, frame, f)
        af = AdaptiveField(field, mesh)
        mono_basis_local = MonomialBasis{2, 2}()

        function total_mass_bern(af, frame, mono_basis_local)
            m = 0.0
            mesh = af.mesh
            multi = moment_multiindices(2, 2)
            for i in 1:n_cells(mesh)
                if is_leaf(mesh.cells[i])
                    lo, hi = cell_physical_box(frame, i)
                    vol = (hi[1] - lo[1]) * (hi[2] - lo[2])
                    bern_coeffs = collect(af.field.rho[i])
                    mono_coeffs = HierarchicalGrids.change_basis(
                        mono_basis_local, BernsteinBasis{2, 2}(), bern_coeffs)
                    integ = 0.0
                    for k in eachindex(multi)
                        integ += mono_coeffs[k] /
                                  ((multi[k][1] + 1) * (multi[k][2] + 1))
                    end
                    m += vol * integ
                end
            end
            return m
        end

        m0 = total_mass_bern(af, frame, mono_basis_local)
        max_drift = 0.0
        for _ in 1:30
            leaves = enumerate_leaves(af.mesh)
            if rand(rng) < 0.6 && !isempty(leaves)
                target = leaves[rand(rng, 1:length(leaves))]
                refine_cells!(af.mesh, [target])
            else
                candidates = Int[]
                for i in 1:n_cells(af.mesh)
                    cell = af.mesh.cells[i]
                    if !is_leaf(cell)
                        kids = HierarchicalGrids.find_children(af.mesh, i)
                        all_leaves = all(is_leaf(af.mesh.cells[k]) for k in kids)
                        if all_leaves
                            push!(candidates, i)
                        end
                    end
                end
                if !isempty(candidates)
                    target = candidates[rand(rng, 1:length(candidates))]
                    coarsen_cells!(af.mesh, [target])
                end
            end
            m = total_mass_bern(af, frame, mono_basis_local)
            drift = abs(m - m0)
            if drift > max_drift
                max_drift = drift
            end
        end
        @test max_drift < 1e-10

        dispose!(af)
    end

    @testset "Bernstein L²-projection — degree comparison" begin
        # Same physical field initialized at degrees 0, 1, 2 in BernsteinBasis.
        # After refine→coarsen, higher-degree fields reproduce the original
        # parent better (smaller L² error against the analytic projection of
        # the same function).
        function abs_l2_error_bern(af, frame, f, basis_degree)
            err2 = 0.0
            mesh = af.mesh
            for i in 1:n_cells(mesh)
                is_leaf(mesh.cells[i]) || continue
                lo, hi = cell_physical_box(frame, i)
                Nq = 4
                cell_err = 0.0
                cell_vol = (hi[1] - lo[1]) * (hi[2] - lo[2])
                coeffs = collect(af.field.rho[i])
                for ix in 1:Nq, iy in 1:Nq
                    ξ = ((ix - 0.5) / Nq, (iy - 0.5) / Nq)
                    x = (lo[1] + ξ[1] * (hi[1] - lo[1]),
                         lo[2] + ξ[2] * (hi[2] - lo[2]))
                    p_val = evaluate(BernsteinBasis{2, basis_degree}(), coeffs, ξ)
                    cell_err += (p_val - f(x))^2
                end
                err2 += cell_err / (Nq * Nq) * cell_vol
            end
            return sqrt(err2)
        end

        f = x -> sin(x[1]) * cos(x[2]) + 0.3 * x[1]^2 - 0.1 * x[2]^2

        function setup_and_coarsen_bern(degree)
            mesh = HierarchicalMesh{2}()
            basis = BernsteinBasis{2, degree}()
            field = allocate_polynomial_fields(SoA(), basis, 1; rho=Float64)
            frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
            af = AdaptiveField(field, mesh)
            refine_cells!(mesh, [1])
            init_field_from!(af.field, frame, f)
            coarsen_cells!(mesh, [1])
            return af, frame
        end

        af0, fr0 = setup_and_coarsen_bern(0)
        af1, fr1 = setup_and_coarsen_bern(1)
        af2, fr2 = setup_and_coarsen_bern(2)

        e0 = abs_l2_error_bern(af0, fr0, f, 0)
        e1 = abs_l2_error_bern(af1, fr1, f, 1)
        e2 = abs_l2_error_bern(af2, fr2, f, 2)

        @test e1 < e0
        @test e2 < e1

        dispose!(af0)
        dispose!(af1)
        dispose!(af2)
    end

end
