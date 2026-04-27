using Test
using HierarchicalGrids
using HierarchicalGrids: HierarchicalMesh, refine_cells!, n_cells,
    EulerianFrame, FrameBoundaries, enumerate_leaves,
    PERIODIC, REFLECTING,
    MonomialBasis, n_coeffs, allocate_polynomial_fields, SoA,
    Sequential, OhMyThreadsBackend,
    cell_view, halo_view_multi,
    for_each_cell!, for_each_face!

# ============================================================================
# Determinism under parallelism
#
# A non-trivial 2D upwind advection RHS produces byte-identical output
# regardless of the parallel backend. Floating-point determinism holds
# here because each kernel invocation writes a distinct cell's coefficients
# (no reduction, no shared accumulator), so the result is independent of
# task scheduling.
# ============================================================================

function _build_2d(n_root_refines::Int = 2)
    mesh = HierarchicalMesh{2}()
    refine_cells!(mesh, [1])
    for _ in 2:n_root_refines
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    n = n_cells(mesh)
    basis = MonomialBasis{2, 1}()
    nc = n_coeffs(basis)
    fin = allocate_polynomial_fields(SoA(), basis, n; rho = Float64, u = Float64, v = Float64)
    fout = allocate_polynomial_fields(SoA(), basis, n; rho = Float64)
    for i in 1:n
        fin.rho[i] = ntuple(k -> Float64(sin(0.7 * i + 0.13 * k)), nc)
        fin.u[i]   = ntuple(k -> Float64(0.5 + 0.1 * cos(0.3 * i + k)), nc)
        fin.v[i]   = ntuple(k -> Float64(0.4 + 0.07 * sin(1.7 * i - k)), nc)
        fout.rho[i] = ntuple(_ -> 0.0, nc)
    end
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    return (mesh, frame, fin, fout, basis, nc)
end

@testset "for_each_cell! — byte-equal determinism across all five backends" begin
    (mesh, frame, fin, fout, basis, nc) = _build_2d(3)
    bcs = FrameBoundaries(((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)))

    # 2D upwind RHS in coefficient space (treated component-wise so the
    # result is meaningful as a determinism witness, not a CFD-correct
    # discretization). The kernel reads four neighbors and the central
    # cell, and writes a function of all five.
    function make_run()
        return function (backend)
            # fields_in and fields_out must share names — convention of
            # the CellView double-buffer. Allocate fout with the same set
            # of names; the kernel only writes to :rho but :u and :v
            # remain bound (untouched).
            fout_local = allocate_polynomial_fields(SoA(), basis, n_cells(mesh);
                                                    rho = Float64, u = Float64, v = Float64)
            for i in 1:n_cells(mesh)
                fout_local.rho[i] = ntuple(_ -> 0.0, nc)
                fout_local.u[i]   = ntuple(_ -> 0.0, nc)
                fout_local.v[i]   = ntuple(_ -> 0.0, nc)
            end
            fin_v  = (rho = fin.rho, u = fin.u, v = fin.v)
            fout_v = (rho = fout_local.rho, u = fout_local.u, v = fout_local.v)
            kernel = function (cv, hv, ctx)
                ρ_c = cv[Val(:rho)]
                ρ_e = hv[Val(:rho), (1, 0)]
                ρ_w = hv[Val(:rho), (-1, 0)]
                ρ_n = hv[Val(:rho), (0, 1)]
                ρ_s = hv[Val(:rho), (0, -1)]
                # All four are non-nothing under fully-periodic BCs.
                u = cv[Val(:u)]
                v = cv[Val(:v)]
                cv[Val(:rho)] = ntuple(nc) do k
                    fluxe = u[k] >= 0 ? u[k] * ρ_c[k] : u[k] * ρ_e[k]
                    fluxw = u[k] >= 0 ? u[k] * ρ_w[k] : u[k] * ρ_c[k]
                    fluxn = v[k] >= 0 ? v[k] * ρ_c[k] : v[k] * ρ_n[k]
                    fluxs = v[k] >= 0 ? v[k] * ρ_s[k] : v[k] * ρ_c[k]
                    ρ_c[k] - 0.01 * ((fluxe - fluxw) + (fluxn - fluxs))
                end
                return nothing
            end
            for_each_cell!(kernel, fout_v, fin_v, frame;
                           ghost_depth = 1, bcs = bcs, backend = backend)
            return [collect(fout_local.rho[i]) for i in 1:n_cells(mesh)]
        end
    end

    run = make_run()
    ref = run(Sequential())
    for backend in (OhMyThreadsBackend(:dynamic),
                    OhMyThreadsBackend(:static),
                    OhMyThreadsBackend(:greedy),
                    OhMyThreadsBackend(:serial))
        snap = run(backend)
        @test snap == ref     # byte-equal, NOT ≈
    end
end

@testset "for_each_face! — byte-equal determinism across all five backends" begin
    (mesh, frame, fin, fout, basis, nc) = _build_2d(2)
    fin_v = (rho = fin.rho, u = fin.u, v = fin.v)

    # Each face's flux is written into a thread-local dict keyed by
    # (left_index, right_index). Determinism: we reduce to a sorted vector
    # of (key, value) before comparing.
    function run(backend)
        fluxes = Dict{Tuple{Int, Int}, Float64}()
        lk = ReentrantLock()
        kern = function (cv_l, cv_r, normal, ctx)
            l = cv_l[Val(:rho)][1]
            r = cv_r[Val(:rho)][1]
            u = cv_l[Val(:u)][1]
            f = u >= 0 ? u * l : u * r
            lock(lk) do
                fluxes[(cv_l.index, cv_r.index)] = f
            end
            return nothing
        end
        for_each_face!(kern, (rho = nothing,), fin_v, frame; backend = backend)
        return sort!(collect(fluxes); by = first)
    end

    ref = run(Sequential())
    for backend in (OhMyThreadsBackend(:dynamic),
                    OhMyThreadsBackend(:static),
                    OhMyThreadsBackend(:greedy),
                    OhMyThreadsBackend(:serial))
        snap = run(backend)
        @test snap == ref
    end
end
