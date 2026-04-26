# ============================================================================
# Lagrangian mesh: build a uniform 2D triangulation, advance under flow
# ============================================================================
#
# We tessellate the unit square with `N × N` vertices (so `(N-1)² × 2`
# triangles). Each square cell `(i, j)` is split into two triangles along
# the diagonal `(i,j) → (i+1, j+1)`. Triangle indices are interleaved so
# that the lower-left triangle comes before the upper-right one within
# each cell — kept consistent so the neighbor topology is straightforward
# to derive.
#
# `reference_positions` holds the initial undeformed positions; `positions`
# holds the current deformed positions, advanced via the flow map.

"""
    build_unit_square_lag_mesh(n_per_axis::Integer)
        -> SimplicialMesh{2, Float64}

Build a uniform triangulation of the unit square `[0, 1]²` with
`n_per_axis × n_per_axis` vertices and `2 (n_per_axis - 1)²` right
triangles. Each square cell is split along its (lower-left, upper-right)
diagonal. The resulting mesh has both `positions` and `reference_positions`
initialized to the same uniform grid; advance one of them with
`advance_lagrangian!`.

# Arguments

- `n_per_axis` ≥ 2: number of vertices per side. `n_per_axis = 16` gives
  a 16×16 vertex grid → 450 triangles, a comfortable size for visual
  demos.
"""
function build_unit_square_lag_mesh(n_per_axis::Integer)
    n_per_axis >= 2 ||
        throw(ArgumentError("n_per_axis must be ≥ 2 (got $n_per_axis)"))
    N = Int(n_per_axis)
    nv = N * N
    nt = 2 * (N - 1)^2

    # Vertex positions on a uniform grid
    positions = Vector{NTuple{2, Float64}}(undef, nv)
    for j in 1:N, i in 1:N
        positions[(j-1)*N + i] = ((i - 1) / (N - 1), (j - 1) / (N - 1))
    end

    # Triangles: each (i, j) cell → two triangles
    # Lower-left triangle:  (i, j)   → (i+1, j) → (i, j+1)
    # Upper-right triangle: (i+1, j) → (i+1, j+1) → (i, j+1)
    sv = Matrix{Int32}(undef, 3, nt)
    for j in 1:(N-1), i in 1:(N-1)
        v00 = (j-1)*N + i           # (i,   j)
        v10 = (j-1)*N + (i+1)       # (i+1, j)
        v01 = j*N + i               # (i,   j+1)
        v11 = j*N + (i+1)           # (i+1, j+1)
        cell_idx = (j-1)*(N-1) + i
        # Lower-left triangle (CCW):
        sv[1, 2*cell_idx - 1] = v00
        sv[2, 2*cell_idx - 1] = v10
        sv[3, 2*cell_idx - 1] = v01
        # Upper-right triangle (CCW):
        sv[1, 2*cell_idx]     = v10
        sv[2, 2*cell_idx]     = v11
        sv[3, 2*cell_idx]     = v01
    end

    # Neighbor topology — not strictly needed for compute_overlap (which
    # only consumes vertex positions), so leave as zeros (= all boundary).
    # A future caller that wants distortion-driven refinement of the
    # Lagrangian mesh itself would fill these in.
    sn = zeros(Int32, 3, nt)

    return SimplicialMesh{2, Float64}(positions, sv, sn)
end

"""
    advance_lagrangian!(lag::SimplicialMesh{2, Float64},
                        flow::FlowMap, t::Real)

Advance every Lagrangian vertex under the flow map: each vertex's
**current** position is set to `apply_map(flow, X, Y, t)` where
`(X, Y)` is its **reference** position. This means subsequent calls
don't compound — the map is always applied to the original positions.

This is the natural pattern for a static flow map evaluated at different
times; if you want a true integrator (composing time-dependent velocity
fields), advance reference positions yourself between calls.
"""
function advance_lagrangian!(lag::SimplicialMesh{2, Float64},
                              flow::FlowMap, t::Real)
    nv = n_vertices(lag)
    @inbounds for i in 1:nv
        X = reference_position(lag, i)
        x, y = apply_map(flow, X[1], X[2], t)
        # Clamp to unit square; reference positions are already inside,
        # but a flow map may push outliers slightly out.
        x = clamp(x, 0.0, 1.0)
        y = clamp(y, 0.0, 1.0)
        set_vertex_position!(lag, i, (x, y))
    end
    return lag
end
