# ============================================================================
# Tests for the MAC (face-centered) velocity projection.
#
# Helmholtz–Hodge decomposition: any vector field u can be written as
#     u = u_div_free + ∇·(β ∇φ)/β
# We construct an analytic divergent u, project it, and verify the result
# is divergence-free (to Krylov tolerance) and matches the analytic
# divergence-free component for a separable test.
# ============================================================================

using Test
using HierarchicalGrids
using HierarchicalGrids: build_uniform_root_hierarchy, allocate_phi_rho,
                          MGWorkspace, MGOptions,
                          PERIODIC, FaceVelocity, allocate_face_velocity,
                          fill_face_velocity!, face_divergence!,
                          face_divergence_l2, mac_project!
using HierarchicalGrids.Overlap: FrameBoundaries
using LinearAlgebra: norm

@testset "MAC projection: 2D periodic, constant β" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 64
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)

    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg, pcg_precond = :jacobi)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    # u = ∇ψ + curl(η)  where curl(η) is divergence-free.
    # Pick ψ = cos(2πx)cos(2πy) ⇒ ∇ψ = (-2π sin(2πx)cos(2πy), -2π cos(2πx)sin(2πy))
    # η_z = sin(2πx)sin(2πy)  ⇒ curl = (∂_y η_z, -∂_x η_z) = 2π(sin(2πx)cos(2πy), -cos(2πx)sin(2πy))
    # We start with u = ∇ψ + curl(η).  After projection the result should
    # equal curl(η) (divergence-free part).
    ψx(x) = -2π * sin(2π*x[1]) * cos(2π*x[2])
    ψy(x) = -2π * cos(2π*x[1]) * sin(2π*x[2])
    curlx(x) = 2π * sin(2π*x[1]) * cos(2π*x[2])
    curly(x) = -2π * cos(2π*x[1]) * sin(2π*x[2])
    ux(x) = ψx(x) + curlx(x)
    uy(x) = ψy(x) + curly(x)

    u = allocate_face_velocity(ws, 1, 1)
    fill_face_velocity!(u, ws, 1, 1, (ux, uy))

    # β = 1 (incompressible, constant density)
    β = ntuple(d -> fill(1.0, size(u.u[d])), Val(2))

    # Pre-projection divergence (large because u contains ∇ψ).
    div_pre = face_divergence_l2(u, ws, 1, 1)
    @test div_pre > 1.0

    r = mac_project!(u, β, ws; tol = 1e-10, maxiter = 200)
    @test r.converged

    # Post-projection divergence should be ~ Krylov tolerance.
    div_post = face_divergence_l2(u, ws, 1, 1)
    @test div_post < 1e-7

    # The projected u should match curl(η).
    u_exact = allocate_face_velocity(ws, 1, 1)
    fill_face_velocity!(u_exact, ws, 1, 1, (curlx, curly))
    err = 0.0
    for d in 1:2
        err += sum((u.u[d] .- u_exact.u[d]) .^ 2)
    end
    err = sqrt(err / (N * N))
    # 2nd-order discretisation error on a 64² grid.
    @test err < 1e-2
end

@testset "MAC projection: 2D periodic, variable β = 1/ρ" begin
    bcs_spec = ((PERIODIC, PERIODIC), (PERIODIC, PERIODIC))
    bcs = FrameBoundaries(bcs_spec)
    N = 64
    ph = build_uniform_root_hierarchy(Val(2), Int(log2(N)), (0.0, 0.0), (1.0, 1.0);
                                       physical_bcs = bcs)

    opts = MGOptions(tol = 1e-10, maxiter = 200, cycle = :pcg, pcg_precond = :jacobi)
    ws = MGWorkspace(ph, bcs_spec; opts = opts)

    # Variable β(x) = 1 + 0.5 cos(2πx) cos(2πy)  (positive).
    β_fun(x) = 1.0 + 0.5 * cos(2π*x[1]) * cos(2π*x[2])
    # Some divergent initial face velocity.
    ux(x) = sin(2π*x[1]) * cos(2π*x[2]) + 0.3 * cos(4π*x[1])
    uy(x) = -cos(2π*x[1]) * sin(2π*x[2]) + 0.3 * sin(4π*x[2])

    u = allocate_face_velocity(ws, 1, 1)
    fill_face_velocity!(u, ws, 1, 1, (ux, uy))

    # Sample β at face centers (same convention as fill_abec_beta!).
    β = ntuple(Val(2)) do d
        arr = similar(u.u[d])
        N_arr = size(arr)
        for I in CartesianIndices(N_arr)
            x_face = ntuple(Val(2)) do j
                if j == d
                    (I[d] - 1) / N
                else
                    (I[j] - 0.5) / N
                end
            end
            arr[I] = β_fun(x_face)
        end
        arr
    end

    div_pre = face_divergence_l2(u, ws, 1, 1)
    r = mac_project!(u, β, ws; tol = 1e-10, maxiter = 200)
    @test r.converged
    div_post = face_divergence_l2(u, ws, 1, 1)
    @test div_post < 1e-7 * div_pre
end
