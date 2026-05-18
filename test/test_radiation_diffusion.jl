# ============================================================================
# Tests for the linear radiation-diffusion wrappers (Tier 3 #7).
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, allocate_phi_rho,
                          MGWorkspace, MGOptions, fill_field!,
                          PERIODIC, allocate_abec_coefs,
                          setup_gray_radiation!, solve_gray_radiation_step!,
                          allocate_multigroup, solve_multigroup_step!
using HierarchicalGrids.Overlap: FrameBoundaries

@testset "Radiation diffusion: gray BE step (constant κ)" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 64
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    fields = allocate_phi_rho(ph)
    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    dt = 0.01
    c_light = 1.0
    κ = 1.0
    coefs = allocate_abec_coefs(ws; A = 1.0, B = dt)
    coefs = setup_gray_radiation!(coefs, ws, x -> κ, dt; c_light = c_light)

    # Initial E_r = sin(2πx) sin(2πy) — an eigenfunction of the operator.
    fill_field!(fields, ph, :phi, x -> sin(2π*x[1]) * sin(2π*x[2]))
    E_prev = allocate_phi_rho(ph)
    fill_field!(E_prev, ph, :phi, x -> sin(2π*x[1]) * sin(2π*x[2]))

    r = solve_gray_radiation_step!(fields, E_prev, coefs, ws;
                                     c_light = c_light, tol = 1e-10, maxiter = 200)
    @test r.converged
    @test r.res_final < 1e-9 * r.res_init || r.res_final < 1e-9
end

@testset "Radiation diffusion: multigroup decoupled" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    dt = 0.01
    c_light = 1.0
    mg = allocate_multigroup(ws, 3; A = 1.0, B = dt)
    for (g, κg) in enumerate((0.5, 1.0, 2.0))
        mg.coefs[g].alpha[1][1] .= 1.0 + dt * c_light * κg
        for d in 1:2
            mg.coefs[g].beta[1][1][d] .= c_light / (3 * κg)
        end
        fill_field!(mg.fields[g], ph, :rho, x -> sin(2π*x[1]) * sin(2π*x[2]))
        fill_field!(mg.fields[g], ph, :phi, x -> 0.0)
    end
    results = solve_multigroup_step!(mg, ws; tol = 1e-10, maxiter = 200)
    for r in results
        @test r.converged
        @test r.res_final / r.res_init < 1e-9
    end
end
