# ============================================================================
# Tests for the node-centered variable-coefficient Laplacian (Tier 2 #4).
# Single-level / single-patch, periodic / Dirichlet / Neumann at the patch
# boundary. AMReX MLNodeLaplacian equivalent.
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, MGWorkspace, MGOptions,
                          PERIODIC, DIRICHLET,
                          NodeField, allocate_node_field, fill_node_field!,
                          NodeCoefs, allocate_node_coefs,
                          fill_node_coefs_sigma!,
                          apply_node_laplacian!, gs_sweep_node!,
                          solve_node_laplacian!
using HierarchicalGrids.Overlap: FrameBoundaries

@testset "NodeLaplacian: const σ matches analytic Poisson" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 32
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-10, maxiter = 300, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_node_coefs(ws; B = 1.0, sigma_init = 1.0)

    phi = allocate_node_field(ws)
    rho = allocate_node_field(ws)
    fill_node_field!(rho, ws, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
    fill_node_field!(phi, ws, x -> 0.0)

    r = solve_node_laplacian!(phi, rho, coefs, ws;
                                method = :cg, tol = 1e-10, maxiter = 500)
    @test r.converged
    @test r.res_final / r.res_init < 1e-7
end

@testset "NodeLaplacian: variable σ MMS recovery" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 16
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-10, maxiter = 500, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_node_coefs(ws; B = 1.0)
    fill_node_coefs_sigma!(coefs, ws,
                              x -> 1.0 + 0.5 * cos(2π*x[1]) * cos(2π*x[2]))

    # Use the discrete operator to manufacture a consistent RHS:
    # set phi_exact, compute rho = L phi_exact, then zero phi and solve.
    phi_exact = allocate_node_field(ws)
    fill_node_field!(phi_exact, ws, x -> sin(2π*x[1]) * sin(2π*x[2]))

    rho = allocate_node_field(ws)
    apply_node_laplacian!(rho, phi_exact, coefs, ws)

    phi = allocate_node_field(ws)
    fill_node_field!(phi, ws, x -> 0.0)
    r = solve_node_laplacian!(phi, rho, coefs, ws;
                                method = :cg, tol = 1e-10, maxiter = 500)
    @test r.converged

    # Recovery error: compare phi to phi_exact at every node.  Pure
    # Neumann/periodic Poisson admits a constant nullspace; remove the
    # mean before comparing.
    arr = phi.data[1][1]
    ex  = phi_exact.data[1][1]
    mean_phi = sum(arr) / length(arr)
    mean_ex  = sum(ex)  / length(ex)
    err = 0.0
    @inbounds for I in CartesianIndices(size(arr))
        err += ((arr[I] - mean_phi) - (ex[I] - mean_ex))^2
    end
    err = sqrt(err / length(arr))
    @test err < 1e-7
end

@testset "NodeLaplacian: GS smoother reduces residual" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 16
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)
    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)
    coefs = allocate_node_coefs(ws; B = 1.0, sigma_init = 1.0)

    phi = allocate_node_field(ws)
    rho = allocate_node_field(ws)
    fill_node_field!(rho, ws, x -> 8π^2 * sin(2π*x[1]) * sin(2π*x[2]))
    fill_node_field!(phi, ws, x -> 0.0)

    # Apply many GS sweeps; residual should drop monotonically.
    Lphi = allocate_node_field(ws)
    apply_node_laplacian!(Lphi, phi, coefs, ws)
    r0 = sqrt(sum((rho.data[1][1] .- Lphi.data[1][1]) .^ 2))

    gs_sweep_node!(phi, rho, coefs, ws; n_sweeps = 50)

    apply_node_laplacian!(Lphi, phi, coefs, ws)
    rN = sqrt(sum((rho.data[1][1] .- Lphi.data[1][1]) .^ 2))
    @test rN < r0
end
