using Test
using HierarchicalGrids
using HierarchicalGrids: Sequential, OhMyThreadsBackend
using HierarchicalGrids.Initialization: init_field_from!

# ============================================================================
# Determinism tests: init_field_from! must produce byte-identical
# coefficients across all parallel backends.
#
# Each cell's projection is independent (own RHS vector, own mass matrix
# copy, own per-cell solve), so the result is a function only of the
# per-cell inputs — no cross-cell reductions, no order-dependent
# floating-point sums.
# ============================================================================

const _BACKENDS = [
    ("Sequential",                    Sequential()),
    ("OhMyThreadsBackend(:dynamic)",  OhMyThreadsBackend(:dynamic)),
    ("OhMyThreadsBackend(:static)",   OhMyThreadsBackend(:static)),
    ("OhMyThreadsBackend(:greedy)",   OhMyThreadsBackend(:greedy)),
    ("OhMyThreadsBackend(:serial)",   OhMyThreadsBackend(:serial)),
]

# Helper: extract all coefficients of a one-field PolynomialFieldSet into
# a flat Vector{Float64} so byte-equality is easy to assert with `==`.
function _extract_all_coeffs(field, n)
    nc = n_coeffs_per_element(field)
    out = Vector{Float64}(undef, n * nc)
    for i in 1:n
        c = field.u[i]   # NTuple{nc, Float64}
        for k in 1:nc
            out[(i - 1) * nc + k] = c[k]
        end
    end
    return out
end

# ----------------------------------------------------------------------------
# Eulerian (axis-aligned) overload
# ----------------------------------------------------------------------------

@testset "init_field_from!(EulerianFrame): byte-equal across backends" begin
    # Refined 2D Eulerian mesh with non-uniform refinement so cells differ
    # in size and the parallel work is not trivially uniform.
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    leaves = enumerate_leaves(eul)
    refine_cells!(eul, leaves)
    leaves = enumerate_leaves(eul)
    # Refine half of the leaves once more to break uniformity.
    refine_cells!(eul, leaves[1:div(length(leaves), 2)])
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))

    n = n_cells(eul)
    @assert n > 1

    basis = BernsteinBasis{2, 3}()
    f = x -> sin(2π * x[1]) + cos(3π * x[2])

    # Reference run: Sequential.
    ref_field = allocate_polynomial_fields(SoA(), basis, n; u = Float64)
    init_field_from!(ref_field, frame, f; backend = Sequential())
    ref = _extract_all_coeffs(ref_field, n)

    for (label, backend) in _BACKENDS
        fld = allocate_polynomial_fields(SoA(), basis, n; u = Float64)
        init_field_from!(fld, frame, f; backend = backend)
        coeffs = _extract_all_coeffs(fld, n)
        @test coeffs == ref
    end
end

@testset "init_field_from!(EulerianFrame): default backend matches :dynamic" begin
    eul = HierarchicalMesh{2}()
    refine_cells!(eul, [1])
    refine_cells!(eul, enumerate_leaves(eul))
    frame = EulerianFrame(eul, (0.0, 0.0), (1.0, 1.0))
    n = n_cells(eul)
    basis = BernsteinBasis{2, 2}()
    f = x -> sin(2π * x[1]) + cos(3π * x[2])

    fld_default = allocate_polynomial_fields(SoA(), basis, n; u = Float64)
    init_field_from!(fld_default, frame, f)

    fld_dyn = allocate_polynomial_fields(SoA(), basis, n; u = Float64)
    init_field_from!(fld_dyn, frame, f; backend = OhMyThreadsBackend(:dynamic))

    @test _extract_all_coeffs(fld_default, n) == _extract_all_coeffs(fld_dyn, n)
end

# ----------------------------------------------------------------------------
# Simplicial overload
# ----------------------------------------------------------------------------

@testset "init_field_from!(SimplicialMesh): byte-equal across backends" begin
    # Build a triangulated grid with enough triangles that parallelism engages.
    nside = 6
    h = 1.0 / nside
    positions = NTuple{2, Float64}[]
    for j in 0:nside, i in 0:nside
        push!(positions, (i * h, j * h))
    end
    n_tri = 2 * nside * nside
    sv = Matrix{Int32}(undef, 3, n_tri)
    sn = zeros(Int32, 3, n_tri)
    t = 1
    for j in 0:(nside - 1), i in 0:(nside - 1)
        v00 = j * (nside + 1) + i + 1
        v10 = v00 + 1
        v01 = v00 + (nside + 1)
        v11 = v01 + 1
        sv[1, t] = v00; sv[2, t] = v10; sv[3, t] = v11; t += 1
        sv[1, t] = v00; sv[2, t] = v11; sv[3, t] = v01; t += 1
    end
    mesh = SimplicialMesh{2, Float64}(positions, sv, sn)
    ns = n_simplices(mesh)
    @assert ns == n_tri

    basis = BernsteinBasis{2, 2}()
    f = x -> sin(2π * x[1]) * cos(3π * x[2]) + 0.25 * x[1]

    # Reference: Sequential.
    ref_field = allocate_polynomial_fields(SoA(), basis, ns; u = Float64)
    init_field_from!(ref_field, mesh, f; backend = Sequential())
    ref = _extract_all_coeffs(ref_field, ns)

    for (label, backend) in _BACKENDS
        fld = allocate_polynomial_fields(SoA(), basis, ns; u = Float64)
        init_field_from!(fld, mesh, f; backend = backend)
        coeffs = _extract_all_coeffs(fld, ns)
        @test coeffs == ref
    end
end

@testset "init_field_from!(SimplicialMesh): default backend matches :dynamic" begin
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    sv = Int32[1 4; 2 3; 3 2]
    sn = zeros(Int32, 3, 2)
    mesh = SimplicialMesh{2, Float64}(positions, sv, sn)
    ns = n_simplices(mesh)
    basis = BernsteinBasis{2, 2}()
    f = x -> sin(2π * x[1]) + cos(3π * x[2])

    fld_default = allocate_polynomial_fields(SoA(), basis, ns; u = Float64)
    init_field_from!(fld_default, mesh, f)

    fld_dyn = allocate_polynomial_fields(SoA(), basis, ns; u = Float64)
    init_field_from!(fld_dyn, mesh, f; backend = OhMyThreadsBackend(:dynamic))

    @test _extract_all_coeffs(fld_default, ns) == _extract_all_coeffs(fld_dyn, ns)
end
