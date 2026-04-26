using Test
using LagrangianPixelization
using HierarchicalGrids
using HierarchicalGrids: simplex_volume, level_of

@testset "LagrangianPixelization" begin

# -----------------------------------------------------------------------------
# Flow maps
# -----------------------------------------------------------------------------
@testset "Flow maps: identity preserves coordinates" begin
    m = IdentityMap()
    for (x, y, t) in [(0.1, 0.2, 0.0), (0.5, 0.5, 0.5), (0.9, 0.1, 1.0)]
        @test apply_map(m, x, y, t) == (x, y)
    end
end

@testset "Flow maps: rigid rotation preserves distances" begin
    m = RigidRotation(π/2; cx=0.5, cy=0.5)
    # A quarter turn at t=1: (0.5, 0.0) → (1.0, 0.5)
    x1, y1 = apply_map(m, 0.5, 0.0, 1.0)
    @test x1 ≈ 1.0 atol=1e-12
    @test y1 ≈ 0.5 atol=1e-12

    # Distance from center invariant
    for (x, y) in [(0.0, 0.0), (0.3, 0.7), (1.0, 1.0)]
        x_new, y_new = apply_map(m, x, y, 0.7)
        d_old = sqrt((x - 0.5)^2 + (y - 0.5)^2)
        d_new = sqrt((x_new - 0.5)^2 + (y_new - 0.5)^2)
        @test d_old ≈ d_new atol=1e-12
    end
end

@testset "Flow maps: swirl returns to identity at t=0 and t=period" begin
    m = SwirlMap(strength=2.0, period=1.0)
    for (x, y) in [(0.3, 0.4), (0.5, 0.5), (0.7, 0.6)]
        x0, y0 = apply_map(m, x, y, 0.0)
        @test x0 ≈ x atol=1e-12
        @test y0 ≈ y atol=1e-12
        x1, y1 = apply_map(m, x, y, 1.0)
        @test x1 ≈ x atol=1e-12
        @test y1 ≈ y atol=1e-12
    end
end

@testset "Flow maps: swirl preserves boundary (r >= radius)" begin
    m = SwirlMap(strength=2.0, radius=0.45)
    # Far outside swirl
    for (x, y) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        @test apply_map(m, x, y, 0.25) == (x, y)
    end
end

@testset "Flow maps: Taylor-Green is identity at t=0,1" begin
    m = TaylorGreenPulse(amplitude=0.1, k=2.0)
    for (x, y) in [(0.3, 0.4), (0.5, 0.5)]
        @test apply_map(m, x, y, 0.0) == (x, y)
        @test apply_map(m, x, y, 1.0) == (x, y)
    end
end

@testset "Flow maps: CustomMap wraps callable" begin
    m = CustomMap((x, y, t) -> (x + 0.1*t, y - 0.1*t))
    @test apply_map(m, 0.5, 0.5, 1.0) == (0.6, 0.4)
end

@testset "Flow maps: GaussianCompression returns to identity at t=0,period" begin
    m = GaussianCompression(amplitude=0.6, cx=0.35, cy=0.55, sigma=0.18)
    for (x, y) in [(0.3, 0.4), (0.5, 0.5), (0.7, 0.6)]
        x0, y0 = apply_map(m, x, y, 0.0)
        @test x0 ≈ x atol=1e-12
        @test y0 ≈ y atol=1e-12
        x1, y1 = apply_map(m, x, y, 1.0)
        @test x1 ≈ x atol=1e-12
        @test y1 ≈ y atol=1e-12
    end
end

@testset "Flow maps: GaussianCompression compresses toward center at peak" begin
    cx, cy = 0.35, 0.55
    m = GaussianCompression(amplitude=0.6, cx=cx, cy=cy, sigma=0.18)
    # A point near the center should move closer to it at peak (t=0.5).
    x, y = cx + 0.05, cy + 0.05
    r0 = hypot(x - cx, y - cy)
    x_new, y_new = apply_map(m, x, y, 0.5)
    r1 = hypot(x_new - cx, y_new - cy)
    @test r1 < r0          # got closer
    @test r1 > 0           # didn't cross through center
    # A faraway point should barely move (Gaussian has decayed).
    far_x, far_y = cx + 5.0 * 0.18, cy
    fx_new, fy_new = apply_map(m, far_x, far_y, 0.5)
    @test fx_new ≈ far_x atol=1e-3
    @test fy_new ≈ far_y atol=1e-3
end

@testset "Flow maps: GaussianCompression rejects bad parameters" begin
    @test_throws ArgumentError GaussianCompression(amplitude=0.0)
    @test_throws ArgumentError GaussianCompression(amplitude=1.5)
    @test_throws ArgumentError GaussianCompression(sigma=0.0)
    @test_throws ArgumentError GaussianCompression(sigma=-0.1)
end

@testset "Flow maps: ComposedMap chains in order" begin
    # Composing identity with anything gives the same result as the other map.
    m_swirl = SwirlMap(strength=1.0)
    composed1 = ComposedMap(IdentityMap(), m_swirl)
    composed2 = ComposedMap(m_swirl, IdentityMap())
    for (x, y) in [(0.3, 0.4), (0.5, 0.5)]
        a = apply_map(composed1, x, y, 0.25)
        b = apply_map(m_swirl, x, y, 0.25)
        @test a[1] ≈ b[1] atol=1e-14
        @test a[2] ≈ b[2] atol=1e-14
        c = apply_map(composed2, x, y, 0.25)
        @test c[1] ≈ b[1] atol=1e-14
        @test c[2] ≈ b[2] atol=1e-14
    end
end

@testset "Flow maps: ∘ operator gives function-composition order" begin
    # f ∘ g means "apply g first, then f". We verify by chaining a
    # known-asymmetric pair and checking the order.
    f = CustomMap((x, y, t) -> (x + 0.1, y))      # +x by 0.1
    g = CustomMap((x, y, t) -> (2*x, 2*y))        # double both
    # (f ∘ g)(0.5, 0.5) = f(g(0.5, 0.5)) = f(1.0, 1.0) = (1.1, 1.0)
    a = apply_map(f ∘ g, 0.5, 0.5, 0.0)
    @test a[1] ≈ 1.1 atol=1e-14
    @test a[2] ≈ 1.0 atol=1e-14
    # (g ∘ f)(0.5, 0.5) = g(f(0.5, 0.5)) = g(0.6, 0.5) = (1.2, 1.0)
    b = apply_map(g ∘ f, 0.5, 0.5, 0.0)
    @test b[1] ≈ 1.2 atol=1e-14
    @test b[2] ≈ 1.0 atol=1e-14
end

# -----------------------------------------------------------------------------
# Lagrangian mesh construction
# -----------------------------------------------------------------------------
@testset "build_unit_square_lag_mesh: basic shape" begin
    lag = build_unit_square_lag_mesh(4)
    @test n_vertices(lag) == 16
    @test n_simplices(lag) == 18  # 2 * (4-1)^2

    # Total area should equal 1
    total = sum(simplex_volume(lag, s) for s in 1:n_simplices(lag))
    @test total ≈ 1.0 atol=1e-12

    # All triangles should be CCW (positive signed area)
    for s in 1:n_simplices(lag)
        @test simplex_volume(lag, s) > 0
    end
end

@testset "build_unit_square_lag_mesh: rejects bad input" begin
    @test_throws ArgumentError build_unit_square_lag_mesh(1)
    @test_throws ArgumentError build_unit_square_lag_mesh(0)
end

# -----------------------------------------------------------------------------
# advance_lagrangian!
# -----------------------------------------------------------------------------
@testset "advance_lagrangian!: identity map leaves vertices unchanged" begin
    lag = build_unit_square_lag_mesh(8)
    LagrangianPixelization.advance_lagrangian!(lag, IdentityMap(), 0.5)
    for i in 1:n_vertices(lag)
        @test vertex_position(lag, i) == reference_position(lag, i)
    end
end

@testset "advance_lagrangian!: swirl preserves total signed area" begin
    lag = build_unit_square_lag_mesh(16)
    LagrangianPixelization.advance_lagrangian!(lag, SwirlMap(strength=2.0), 0.25)
    total = sum(simplex_volume(lag, s) for s in 1:n_simplices(lag))
    # Even if some triangles flip, the sum stays at 1 because signed
    # areas cancel where overlapping pieces fold over.
    @test total ≈ 1.0 atol=1e-10
end

@testset "advance_lagrangian!: returns to reference at t=period" begin
    lag = build_unit_square_lag_mesh(12)
    saved = [vertex_position(lag, i) for i in 1:n_vertices(lag)]
    LagrangianPixelization.advance_lagrangian!(lag, SwirlMap(strength=2.0, period=1.0), 1.0)
    for i in 1:n_vertices(lag)
        p = vertex_position(lag, i)
        @test p[1] ≈ saved[i][1] atol=1e-12
        @test p[2] ≈ saved[i][2] atol=1e-12
    end
end

# -----------------------------------------------------------------------------
# Pixelization
# -----------------------------------------------------------------------------
@testset "pixelize!: identity map needs no AMR (already meets target)" begin
    # Lag mesh sized so that with depth=4 starting Eulerian, every
    # Lagrangian simplex already has >= target overlapping leaves.
    sim = Simulation(; n_per_axis=8, eul_initial_depth=4,
                       flow_map=IdentityMap(),
                       pixel_params=PixelizationParams(target_eul_per_lag=2,
                                                        max_depth=6))
    eul, frame, ov, n_iters, refined = step_to_time!(sim, 0.0)
    # All cells already meet target → no AMR refinement on first pass
    @test refined[1] == 0
    @test n_iters == 1
end

@testset "pixelize!: AMR makes the count condition hold" begin
    sim = Simulation(; n_per_axis=24, eul_initial_depth=4,
                       flow_map=SwirlMap(strength=2.0),
                       pixel_params=PixelizationParams(target_eul_per_lag=3,
                                                        max_depth=8))
    eul, frame, ov, n_iters, refined = step_to_time!(sim, 0.25)

    # After convergence, most simplices should meet target.
    counts = n_eulerian_per_lagrangian(ov)
    n_under = count(<(3), counts)
    n_total = length(counts)
    # Allow a small fraction below target — the AMR may run out of
    # max_depth for very compressed simplices in the swirl center.
    @test n_under / n_total < 0.05
end

@testset "pixelize!: total overlap volume = 1 to round-off (no inversion)" begin
    sim = Simulation(; n_per_axis=24, eul_initial_depth=5,
                       flow_map=SwirlMap(strength=2.0))   # safe strength
    eul, frame, ov, _, _ = step_to_time!(sim, 0.25)
    @test total_overlap_volume(ov) ≈ 1.0 atol=1e-10
end

@testset "pixelize!: depth distribution is non-uniform under swirl" begin
    sim = Simulation(; n_per_axis=24, eul_initial_depth=5,
                       flow_map=SwirlMap(strength=2.0),
                       pixel_params=PixelizationParams(target_eul_per_lag=4,
                                                        max_depth=8))
    eul, frame, ov, _, _ = step_to_time!(sim, 0.25)
    depths = Int[]
    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) && push!(depths, Int(level_of(eul, ci)))
    end
    # Should have at least two distinct depths (the AMR responds locally)
    @test length(unique(depths)) >= 2
end

# -----------------------------------------------------------------------------
# Density
# -----------------------------------------------------------------------------
@testset "lagrangian_density_factors: identity gives ρ=1" begin
    lag = build_unit_square_lag_mesh(8)
    LagrangianPixelization.advance_lagrangian!(lag, IdentityMap(), 0.5)
    ρ = lagrangian_density_factors(lag)
    for v in ρ
        @test v ≈ 1.0 atol=1e-12
    end
end

@testset "lagrangian_density_factors: compression raises ρ near center" begin
    lag = build_unit_square_lag_mesh(16)
    LagrangianPixelization.advance_lagrangian!(lag,
        GaussianCompression(amplitude=0.6, cx=0.5, cy=0.5, sigma=0.15), 0.5)
    ρ = lagrangian_density_factors(lag)
    # Find which simplex is closest to the center
    s_center = argmin([
        let v = simplex_vertex_positions(lag, s)
            cx = (v[1][1] + v[2][1] + v[3][1]) / 3
            cy = (v[1][2] + v[2][2] + v[3][2]) / 3
            (cx - 0.5)^2 + (cy - 0.5)^2
        end
        for s in 1:n_simplices(lag)
    ])
    s_corner = 1   # always the bottom-left corner triangle
    @test ρ[s_center] > 1.5      # significantly compressed
    @test ρ[s_corner] ≈ 1.0 atol=0.05  # virtually unchanged
end

@testset "eulerian_density: total deposited mass equals Lagrangian reference area" begin
    sim = Simulation(; n_per_axis=24, eul_initial_depth=5,
                       flow_map=ComposedMap(SwirlMap(strength=1.5),
                                            GaussianCompression(amplitude=0.5)),
                       pixel_params=PixelizationParams(target_eul_per_lag=3, max_depth=7))
    eul, frame, ov, _, _ = step_to_time!(sim, 0.5)
    ρ = eulerian_density(ov, sim.lag, frame)
    total_mass = 0.0
    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        lo, hi = cell_physical_box(frame, ci)
        total_mass += ρ[ci] * (hi[1] - lo[1]) * (hi[2] - lo[2])
    end
    # Reference Lagrangian area = 1; mass is conserved by the deposition
    # (every overlap entry deposits exactly its share of its simplex's mass).
    @test total_mass ≈ 1.0 atol=1e-9
end

@testset "eulerian_density: high-density cells are AMR-refined" begin
    flow = ComposedMap(SwirlMap(strength=2.0),
                       GaussianCompression(amplitude=0.6, cx=0.35, cy=0.55, sigma=0.18))
    sim = Simulation(; n_per_axis=24, eul_initial_depth=5, flow_map=flow,
                       pixel_params=PixelizationParams(target_eul_per_lag=4, max_depth=8))
    eul, frame, ov, _, _ = step_to_time!(sim, 0.5)
    ρ = eulerian_density(ov, sim.lag, frame)
    # Among leaves with significant density (ρ > 2), they should all be
    # refined past the base depth. The AMR criterion correlates densities
    # of small Lagrangian simplices with deeper refinement.
    n_total = 0
    n_deep = 0
    for ci in 1:n_cells(eul)
        is_leaf(eul.cells[ci]) || continue
        ρ[ci] > 2.0 || continue
        n_total += 1
        Int(level_of(eul, ci)) > 5 && (n_deep += 1)
    end
    n_total > 0 || error("no high-density leaves; expected some at peak compression")
    @test n_deep / n_total > 0.95
end

# -----------------------------------------------------------------------------
# SVG output
# -----------------------------------------------------------------------------
@testset "write_svg: produces a well-formed SVG" begin
    sim = Simulation(; n_per_axis=8, eul_initial_depth=3,
                       flow_map=SwirlMap(strength=1.5))
    eul, frame, ov, _, _ = step_to_time!(sim, 0.25)
    path = tempname() * ".svg"
    write_svg(path, sim.lag, eul, frame; overlap=ov, title="test")
    @test isfile(path)
    @test filesize(path) > 100
    # Sanity-check the contents
    content = read(path, String)
    @test startswith(content, "<?xml")
    @test occursin("<svg", content)
    @test occursin("</svg>", content)
    @test occursin("<polygon", content)   # at least one Lagrangian triangle
    @test occursin("<rect", content)      # at least one Eulerian leaf
    @test occursin("test", content)        # the title rendered
    rm(path)
end

@testset "write_svg: accepts both palettes" begin
    sim = Simulation(; n_per_axis=8, eul_initial_depth=2,
                       flow_map=IdentityMap())
    eul, frame, ov, _, _ = step_to_time!(sim, 0.0)
    for palette in [:grayscale_warm, :viridis_lite, :rd_pu]
        path = tempname() * ".svg"
        style = SvgStyle(count_color_palette=palette)
        write_svg(path, sim.lag, eul, frame; overlap=ov, style=style)
        @test isfile(path)
        rm(path)
    end
end

# -----------------------------------------------------------------------------
# Integration: run_sequence! produces all frames
# -----------------------------------------------------------------------------
@testset "run_sequence!: writes the requested number of files" begin
    sim = Simulation(; n_per_axis=8, eul_initial_depth=3,
                       flow_map=SwirlMap(strength=1.5))
    out = mktempdir()
    paths = run_sequence!(sim; n_frames=4, output_dir=out, verbose=false)
    @test length(paths) == 4
    for p in paths
        @test isfile(p)
        @test filesize(p) > 100
    end
    rm(out; recursive=true)
end

end # @testset "LagrangianPixelization"
