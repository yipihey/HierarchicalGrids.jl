"""
    SimplicialMesh{D, T}

A simplicial mesh in `D` spatial dimensions, with vertex coordinates of
type `T`. Topology (which vertices form which simplex, which simplices
are adjacent across each face) is fixed at construction; vertex positions
are mutable and intended to evolve under a Lagrangian flow.

# Reference vs current positions

The mesh holds two parallel position arrays:

- `positions[i]` — the current position of vertex `i` in physical space.
- `reference_positions[i]` — the position at the last regeneration. Used
  for distortion metrics (deformation gradient, volume Jacobian).

This is exactly the data needed for ALE methods, phase-space sheet
projection, and the Lagrangian half of the dfmm framework. See the
`distortion_metric` and `volume_jacobian` queries below.

# Topology arrays (fixed at construction)

- `simplex_vertices[k, s]` — the k-th vertex of simplex `s`, for k = 1..D+1.
  Stored as `Matrix{Int32}` of size `(D+1, n_simplices)`.
- `simplex_neighbors[k, s]` — the simplex sharing the face opposite vertex
  k of simplex s, or 0 if that face is on the boundary. Same shape as
  `simplex_vertices`.

Edge lists, boundary identification, and other derived structure can be
built lazily as needed.

# Construction

```julia
# A single triangle in 2D:
positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
simplex_vertices = reshape(Int32[1, 2, 3], 3, 1)
simplex_neighbors = zeros(Int32, 3, 1)  # all boundaries
mesh = SimplicialMesh{2, Float64}(positions, simplex_vertices, simplex_neighbors)
```

Higher-level constructors that build standard meshes (uniform triangulation
of a rectangle, Delaunay of a point set, etc.) are deferred to the user
or to a future utilities layer.
"""
mutable struct SimplicialMesh{D, T}
    n_vertices::Int
    n_simplices::Int
    simplex_vertices::Matrix{Int32}      # (D+1, n_simplices)
    simplex_neighbors::Matrix{Int32}     # (D+1, n_simplices); 0 means boundary face
    positions::Vector{NTuple{D, T}}      # current positions
    reference_positions::Vector{NTuple{D, T}}  # positions at last regeneration

    # Per-simplex Dirichlet pin flag (PR-D). Length == n_simplices. Default
    # all-false. Advisory: downstream Lagrangian motion should clamp pinned
    # simplices' vertex velocities. The remap and overlap layers do not
    # consume this field.
    pinned::BitVector

    function SimplicialMesh{D, T}(positions::Vector{NTuple{D, T}},
                                   simplex_vertices::Matrix{Int32},
                                   simplex_neighbors::Matrix{Int32};
                                   reference_positions::Union{Nothing, Vector{NTuple{D, T}}}=nothing
                                   ) where {D, T}
        D >= 1 || throw(ArgumentError("dimension D must be ≥ 1"))
        nv = length(positions)
        size(simplex_vertices, 1) == D + 1 ||
            throw(ArgumentError("simplex_vertices must have D+1 = $(D+1) rows"))
        size(simplex_vertices) == size(simplex_neighbors) ||
            throw(ArgumentError("simplex_vertices and simplex_neighbors must have the same shape"))
        ns = size(simplex_vertices, 2)

        # Validate vertex indices
        for s in 1:ns, k in 1:(D+1)
            v = simplex_vertices[k, s]
            (1 <= v <= nv) ||
                throw(ArgumentError("simplex $s vertex $k = $v out of range [1, $nv]"))
        end
        # Validate neighbor indices (0 means boundary, otherwise must be valid)
        for s in 1:ns, k in 1:(D+1)
            n = simplex_neighbors[k, s]
            (n == 0 || 1 <= n <= ns) ||
                throw(ArgumentError("simplex $s neighbor $k = $n out of range"))
        end

        ref_pos = reference_positions === nothing ? copy(positions) : copy(reference_positions)
        length(ref_pos) == nv ||
            throw(ArgumentError("reference_positions length must match positions length"))

        new{D, T}(nv, ns, copy(simplex_vertices), copy(simplex_neighbors),
                  copy(positions), ref_pos, falses(ns))
    end
end

# Convenience constructor: positions as Vector{Vector} (e.g. from a list)
function SimplicialMesh{D, T}(positions, simplex_vertices, simplex_neighbors;
                               reference_positions=nothing) where {D, T}
    pos_tuples = [NTuple{D, T}(p) for p in positions]
    sv = Matrix{Int32}(simplex_vertices)
    sn = Matrix{Int32}(simplex_neighbors)
    rp = reference_positions === nothing ? nothing : [NTuple{D, T}(p) for p in reference_positions]
    return SimplicialMesh{D, T}(pos_tuples, sv, sn; reference_positions=rp)
end

# ============================================================================
# Basic queries
# ============================================================================

"""
    n_vertices(mesh::SimplicialMesh) :: Int

Number of vertices in the mesh.
"""
@inline n_vertices(mesh::SimplicialMesh) = mesh.n_vertices

"""
    n_simplices(mesh::SimplicialMesh) :: Int

Number of simplices in the mesh.
"""
@inline n_simplices(mesh::SimplicialMesh) = mesh.n_simplices

"""
    spatial_dimension(::SimplicialMesh{D, T}) :: Int

The spatial dimension D.
"""
@inline spatial_dimension(::SimplicialMesh{D, T}) where {D, T} = D

"""
    vertex_position(mesh::SimplicialMesh, i) :: NTuple{D, T}

Current position of vertex `i` in physical space.
"""
@inline vertex_position(mesh::SimplicialMesh, i) = mesh.positions[i]

"""
    reference_position(mesh::SimplicialMesh, i) :: NTuple{D, T}

Position of vertex `i` at the last mesh regeneration (the Lagrangian
reference frame).
"""
@inline reference_position(mesh::SimplicialMesh, i) = mesh.reference_positions[i]

"""
    set_vertex_position!(mesh::SimplicialMesh, i, p)

Update vertex `i` to position `p`. Does NOT update reference_positions
(that's done explicitly via `set_reference_to_current!`).
"""
@inline function set_vertex_position!(mesh::SimplicialMesh{D, T}, i::Integer, p) where {D, T}
    mesh.positions[i] = NTuple{D, T}(p)
    return p
end

"""
    set_reference_to_current!(mesh::SimplicialMesh)

Reset reference_positions to the current positions. Use after a mesh
regeneration / resampling operation, when the current state should be
treated as the new "undeformed" configuration.
"""
function set_reference_to_current!(mesh::SimplicialMesh)
    copyto!(mesh.reference_positions, mesh.positions)
    return mesh
end

"""
    simplex_vertex_indices(mesh::SimplicialMesh, s) :: NTuple{D+1, Int}

The vertex indices of simplex `s`.
"""
@inline function simplex_vertex_indices(mesh::SimplicialMesh{D, T}, s::Integer) where {D, T}
    return ntuple(k -> Int(mesh.simplex_vertices[k, s]), Val(D + 1))
end

"""
    simplex_vertex_positions(mesh::SimplicialMesh, s)

Return the D+1 current vertex positions of simplex `s` as a tuple.
"""
@inline function simplex_vertex_positions(mesh::SimplicialMesh{D, T}, s::Integer) where {D, T}
    return ntuple(k -> mesh.positions[mesh.simplex_vertices[k, s]], Val(D + 1))
end

"""
    simplex_reference_positions(mesh::SimplicialMesh, s)

Return the D+1 reference (Lagrangian) vertex positions of simplex `s`.
"""
@inline function simplex_reference_positions(mesh::SimplicialMesh{D, T}, s::Integer) where {D, T}
    return ntuple(k -> mesh.reference_positions[mesh.simplex_vertices[k, s]], Val(D + 1))
end

"""
    simplex_neighbor(mesh::SimplicialMesh, s, k) :: Int

The simplex sharing the face opposite vertex k of simplex s, or 0 if
that face is on the boundary.
"""
@inline function simplex_neighbor(mesh::SimplicialMesh, s::Integer, k::Integer)
    return Int(mesh.simplex_neighbors[k, s])
end

"""
    is_boundary_face(mesh::SimplicialMesh, s, k) :: Bool

Whether the face opposite vertex k of simplex s is a boundary face.
"""
@inline is_boundary_face(mesh::SimplicialMesh, s, k) = simplex_neighbor(mesh, s, k) == 0

# ============================================================================
# Geometric queries (current frame)
# ============================================================================

"""
    simplex_volume(mesh::SimplicialMesh, s) :: T

Signed volume of simplex `s` in the current frame. For D=1 this is the
length; for D=2 the signed area; for D=3 the signed volume.

The sign is positive if the vertex ordering is "right-handed" (the
standard orientation), negative if reversed.
"""
@inline function simplex_volume(mesh::SimplicialMesh{D, T}, s::Integer) where {D, T}
    pts = simplex_vertex_positions(mesh, s)
    return _simplex_signed_volume(pts, Val(D), T)
end

"""
    simplex_reference_volume(mesh::SimplicialMesh, s) :: T

Signed volume of simplex `s` in the reference frame.
"""
@inline function simplex_reference_volume(mesh::SimplicialMesh{D, T}, s::Integer) where {D, T}
    pts = simplex_reference_positions(mesh, s)
    return _simplex_signed_volume(pts, Val(D), T)
end

# 1D: length = x[2] - x[1]
@inline function _simplex_signed_volume(pts, ::Val{1}, ::Type{T}) where T
    return T(pts[2][1] - pts[1][1])
end

# 2D: signed area of triangle with vertices p1, p2, p3
# = 1/2 ((p2.x - p1.x)(p3.y - p1.y) - (p3.x - p1.x)(p2.y - p1.y))
@inline function _simplex_signed_volume(pts, ::Val{2}, ::Type{T}) where T
    p1, p2, p3 = pts
    return T((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p3[1] - p1[1]) * (p2[2] - p1[2])) / T(2)
end

# 3D: signed volume of tetrahedron = 1/6 det([p2-p1; p3-p1; p4-p1])
@inline function _simplex_signed_volume(pts, ::Val{3}, ::Type{T}) where T
    p1, p2, p3, p4 = pts
    a1, a2, a3 = T(p2[1] - p1[1]), T(p2[2] - p1[2]), T(p2[3] - p1[3])
    b1, b2, b3 = T(p3[1] - p1[1]), T(p3[2] - p1[2]), T(p3[3] - p1[3])
    c1, c2, c3 = T(p4[1] - p1[1]), T(p4[2] - p1[2]), T(p4[3] - p1[3])
    det_val = a1*(b2*c3 - b3*c2) - a2*(b1*c3 - b3*c1) + a3*(b1*c2 - b2*c1)
    return det_val / T(6)
end

"""
    deformation_gradient(mesh::SimplicialMesh, s) :: NTuple{D, NTuple{D, T}}

The deformation gradient F = ∂x/∂X mapping the reference simplex to the
current simplex. Returned as a tuple-of-tuples representing a D×D matrix
in row-major order: F[i] is the i-th row, so `F[i][j]` is `∂x_i / ∂X_j`.

For a P1 simplex the deformation gradient is constant within the simplex.
"""
function deformation_gradient(mesh::SimplicialMesh{D, T}, s::Integer) where {D, T}
    cur = simplex_vertex_positions(mesh, s)
    ref = simplex_reference_positions(mesh, s)
    return _deformation_gradient(cur, ref, Val(D), T)
end

# 1D: F = (x2 - x1) / (X2 - X1)
@inline function _deformation_gradient(cur, ref, ::Val{1}, ::Type{T}) where T
    dx = T(cur[2][1] - cur[1][1])
    dX = T(ref[2][1] - ref[1][1])
    return ((dx / dX,),)
end

# 2D: solve F * (P_ref - p_ref_1) = (P_cur - p_cur_1) for the 2x2 matrix F
# Use the 2 edge vectors as columns: dXref = [X2-X1 | X3-X1], dxcur = [x2-x1 | x3-x1]
# F = dxcur * inv(dXref).
@inline function _deformation_gradient(cur, ref, ::Val{2}, ::Type{T}) where T
    # Columns of dXref:
    a11 = T(ref[2][1] - ref[1][1]); a12 = T(ref[3][1] - ref[1][1])
    a21 = T(ref[2][2] - ref[1][2]); a22 = T(ref[3][2] - ref[1][2])
    det_ref = a11*a22 - a12*a21
    inv_det = one(T) / det_ref
    # inv(dXref) = (1/det) * [a22 -a12; -a21 a11]
    inv11 =  a22 * inv_det; inv12 = -a12 * inv_det
    inv21 = -a21 * inv_det; inv22 =  a11 * inv_det
    # Columns of dxcur:
    b11 = T(cur[2][1] - cur[1][1]); b12 = T(cur[3][1] - cur[1][1])
    b21 = T(cur[2][2] - cur[1][2]); b22 = T(cur[3][2] - cur[1][2])
    # F = dxcur * inv(dXref)
    f11 = b11*inv11 + b12*inv21
    f12 = b11*inv12 + b12*inv22
    f21 = b21*inv11 + b22*inv21
    f22 = b21*inv12 + b22*inv22
    return ((f11, f12), (f21, f22))
end

# 3D: same idea, 3x3 inverse and matrix-matrix product
@inline function _deformation_gradient(cur, ref, ::Val{3}, ::Type{T}) where T
    # dXref columns
    a11 = T(ref[2][1] - ref[1][1]); a12 = T(ref[3][1] - ref[1][1]); a13 = T(ref[4][1] - ref[1][1])
    a21 = T(ref[2][2] - ref[1][2]); a22 = T(ref[3][2] - ref[1][2]); a23 = T(ref[4][2] - ref[1][2])
    a31 = T(ref[2][3] - ref[1][3]); a32 = T(ref[3][3] - ref[1][3]); a33 = T(ref[4][3] - ref[1][3])
    det_ref = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
    inv_det = one(T) / det_ref
    # cofactor matrix of dXref, transposed
    c11 =  (a22*a33 - a23*a32) * inv_det
    c12 = -(a12*a33 - a13*a32) * inv_det
    c13 =  (a12*a23 - a13*a22) * inv_det
    c21 = -(a21*a33 - a23*a31) * inv_det
    c22 =  (a11*a33 - a13*a31) * inv_det
    c23 = -(a11*a23 - a13*a21) * inv_det
    c31 =  (a21*a32 - a22*a31) * inv_det
    c32 = -(a11*a32 - a12*a31) * inv_det
    c33 =  (a11*a22 - a12*a21) * inv_det
    # dxcur columns
    b11 = T(cur[2][1] - cur[1][1]); b12 = T(cur[3][1] - cur[1][1]); b13 = T(cur[4][1] - cur[1][1])
    b21 = T(cur[2][2] - cur[1][2]); b22 = T(cur[3][2] - cur[1][2]); b23 = T(cur[4][2] - cur[1][2])
    b31 = T(cur[2][3] - cur[1][3]); b32 = T(cur[3][3] - cur[1][3]); b33 = T(cur[4][3] - cur[1][3])
    # F = dxcur * inv(dXref)
    f11 = b11*c11 + b12*c21 + b13*c31
    f12 = b11*c12 + b12*c22 + b13*c32
    f13 = b11*c13 + b12*c23 + b13*c33
    f21 = b21*c11 + b22*c21 + b23*c31
    f22 = b21*c12 + b22*c22 + b23*c32
    f23 = b21*c13 + b22*c23 + b23*c33
    f31 = b31*c11 + b32*c21 + b33*c31
    f32 = b31*c12 + b32*c22 + b33*c32
    f33 = b31*c13 + b32*c23 + b33*c33
    return ((f11, f12, f13), (f21, f22, f23), (f31, f32, f33))
end

"""
    volume_jacobian(mesh::SimplicialMesh, s) :: T

The Jacobian determinant J = det(F) = current_volume / reference_volume.
A value of 1 means undeformed; > 1 means dilated; < 1 means compressed.
A negative value means the simplex has inverted (shell-crossed); zero
means a degenerate simplex.

This is the dfmm `J = V_Eul / V_Lag` quantity used as the fundamental
additive variable in the Volume Field Theory.
"""
@inline function volume_jacobian(mesh::SimplicialMesh{D, T}, s::Integer) where {D, T}
    return simplex_volume(mesh, s) / simplex_reference_volume(mesh, s)
end

"""
    distortion_metric(mesh::SimplicialMesh, s) :: T

A scalar measure of how badly simplex `s` is distorted from its reference
shape. Currently defined as the Frobenius condition number of the
deformation gradient: ‖F‖_F · ‖F⁻¹‖_F / D, normalized so that an undeformed
simplex returns 1.

Larger values indicate worse distortion. Used as a refinement / regeneration
indicator.
"""
function distortion_metric(mesh::SimplicialMesh{D, T}, s::Integer) where {D, T}
    F = deformation_gradient(mesh, s)
    return _frobenius_condition(F, Val(D), T)
end

# Frobenius norm of a D×D matrix-as-tuples
@inline function _frobenius_norm(F, ::Val{D}, ::Type{T}) where {D, T}
    s = zero(T)
    @inbounds for i in 1:D, j in 1:D
        s += F[i][j] * F[i][j]
    end
    return sqrt(s)
end

# ‖F‖_F · ‖F⁻¹‖_F / D
@inline function _frobenius_condition(F, ::Val{D}, ::Type{T}) where {D, T}
    Finv = _matrix_inverse(F, Val(D), T)
    nF = _frobenius_norm(F, Val(D), T)
    nFi = _frobenius_norm(Finv, Val(D), T)
    return nF * nFi / T(D)
end

@inline function _matrix_inverse(F, ::Val{1}, ::Type{T}) where T
    return ((one(T) / F[1][1],),)
end

@inline function _matrix_inverse(F, ::Val{2}, ::Type{T}) where T
    a11, a12 = F[1][1], F[1][2]
    a21, a22 = F[2][1], F[2][2]
    det = a11*a22 - a12*a21
    inv_det = one(T) / det
    return ((a22*inv_det, -a12*inv_det), (-a21*inv_det, a11*inv_det))
end

@inline function _matrix_inverse(F, ::Val{3}, ::Type{T}) where T
    a11, a12, a13 = F[1][1], F[1][2], F[1][3]
    a21, a22, a23 = F[2][1], F[2][2], F[2][3]
    a31, a32, a33 = F[3][1], F[3][2], F[3][3]
    det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
    inv_det = one(T) / det
    c11 =  (a22*a33 - a23*a32) * inv_det
    c12 = -(a12*a33 - a13*a32) * inv_det
    c13 =  (a12*a23 - a13*a22) * inv_det
    c21 = -(a21*a33 - a23*a31) * inv_det
    c22 =  (a11*a33 - a13*a31) * inv_det
    c23 = -(a11*a23 - a13*a21) * inv_det
    c31 =  (a21*a32 - a22*a31) * inv_det
    c32 = -(a11*a32 - a12*a31) * inv_det
    c33 =  (a11*a22 - a12*a21) * inv_det
    return ((c11, c12, c13), (c21, c22, c23), (c31, c32, c33))
end

"""
    has_inverted_simplex(mesh::SimplicialMesh) :: Bool

Returns true if any simplex has non-positive current volume (relative to
its reference orientation), indicating shell crossing or mesh tangling.
"""
function has_inverted_simplex(mesh::SimplicialMesh)
    for s in 1:mesh.n_simplices
        v_cur = simplex_volume(mesh, s)
        v_ref = simplex_reference_volume(mesh, s)
        # Inverted: current and reference disagree in sign, or current is 0
        if v_cur == 0 || (v_cur > 0) != (v_ref > 0)
            return true
        end
    end
    return false
end

"""
    max_distortion(mesh::SimplicialMesh) :: T

The largest distortion metric over all simplices. Useful as a single
scalar to compare against a regeneration threshold.
"""
function max_distortion(mesh::SimplicialMesh{D, T}) where {D, T}
    m = zero(T)
    for s in 1:mesh.n_simplices
        d = distortion_metric(mesh, s)
        d > m && (m = d)
    end
    return m
end

# ============================================================================
# Edge enumeration
# ============================================================================

"""
    enumerate_edges(mesh::SimplicialMesh) :: Vector{Tuple{Int32, Int32}}

Return a list of unique edges as (v1, v2) pairs with v1 < v2. Each edge
appears once even if shared between multiple simplices.
"""
function enumerate_edges(mesh::SimplicialMesh{D, T}) where {D, T}
    edges = Set{Tuple{Int32, Int32}}()
    @inbounds for s in 1:mesh.n_simplices
        for i in 1:(D+1), j in (i+1):(D+1)
            v1 = mesh.simplex_vertices[i, s]
            v2 = mesh.simplex_vertices[j, s]
            v1, v2 = v1 < v2 ? (v1, v2) : (v2, v1)
            push!(edges, (v1, v2))
        end
    end
    return collect(edges)
end

# ============================================================================
# Show
# ============================================================================

function Base.show(io::IO, mesh::SimplicialMesh{D, T}) where {D, T}
    print(io, "SimplicialMesh{$D, $T}(",
              n_vertices(mesh), " vertices, ", n_simplices(mesh), " simplices)")
end

# ============================================================================
# Periodic wrap-around (PR-D)
# ============================================================================

"""
    periodic!(mesh::SimplicialMesh{D, T}, axes::NTuple{D, Bool},
              bounds::NTuple{D, NTuple{2, T}}) -> mesh

Rewire `simplex_neighbors` so that boundary faces on each periodic axis
are identified with the matching face on the opposite side of the
domain. `axes[d] == true` makes axis `d` periodic; `bounds[d] = (lo, hi)`
gives the period for that axis.

Two boundary faces match if their shared `(D-1)`-coordinates agree
(within an absolute tolerance scaled to the box extent) after wrapping
the periodic axis by its period. The wiring updates *both* affected
entries in `simplex_neighbors`: face `k` of simplex `s` and the
opposite-side face of the matched simplex `s'`.

Faces on non-periodic-axis boundaries are left untouched (still 0).
Faces that already had an interior neighbor are not modified — only
boundary faces (entry == 0) are candidates for wrap-around wiring.

Returns `mesh` (mutated in place).

# Example

```julia
# 1D segment mesh on [0,1] with 4 segments
mesh = SimplicialMesh{1, Float64}(...)
periodic!(mesh, (true,), ((0.0, 1.0),))
# Now segment 1's lo-side neighbor is segment N, etc.
```
"""
function periodic!(mesh::SimplicialMesh{D, T},
                   axes::NTuple{D, Bool},
                   bounds::NTuple{D, NTuple{2, T}}) where {D, T}
    # Bail out fast if no axes are periodic.
    any(axes) || return mesh

    ns = mesh.n_simplices

    # Tolerance: scaled to the largest period. Coordinates are user-supplied
    # floats; we accept a relative ε of ~1e-10 of the max extent.
    max_extent = zero(T)
    for d in 1:D
        ext = bounds[d][2] - bounds[d][1]
        ext > max_extent && (max_extent = ext)
    end
    tol = max(T(1e-10) * max_extent, T(0))

    # For each periodic axis, find boundary faces lying on the lo-side wall
    # and on the hi-side wall, then pair them by matching the OTHER axes'
    # face centroid coordinates.
    @inbounds for axis in 1:D
        axes[axis] || continue
        lo_wall = bounds[axis][1]
        hi_wall = bounds[axis][2]

        # A face on simplex s opposite vertex k consists of the D vertices
        # other than vertex k. Pre-extract the boundary-face entries.
        # `lo_faces[i] = (s, k, centroid_other_axes)` where centroid_other_axes
        # is an NTuple{D, T} (the `axis` slot is set to 0 for matching).
        lo_faces = Tuple{Int, Int, NTuple{D, T}}[]
        hi_faces = Tuple{Int, Int, NTuple{D, T}}[]

        for s in 1:ns, k in 1:(D + 1)
            mesh.simplex_neighbors[k, s] == 0 || continue   # only boundary faces
            face_pts = _face_vertex_positions(mesh, s, k)
            # Face lies on the axis-wall iff all face vertices have the same
            # axis-coordinate (== lo_wall or hi_wall, within tol).
            on_lo = true
            on_hi = true
            for v in face_pts
                if abs(v[axis] - lo_wall) > tol
                    on_lo = false
                end
                if abs(v[axis] - hi_wall) > tol
                    on_hi = false
                end
                (on_lo || on_hi) || break
            end
            (on_lo || on_hi) || continue

            # Centroid in the OTHER axes (axis slot zeroed out).
            cent = _face_centroid_other_axes(face_pts, axis, Val(D), T)
            if on_lo
                push!(lo_faces, (Int(s), Int(k), cent))
            elseif on_hi
                push!(hi_faces, (Int(s), Int(k), cent))
            end
        end

        # Pair lo-face <-> hi-face by closest centroid match. We use a
        # simple nested loop: D-1 dimensional matching, ns is small in
        # typical use; the algorithm is O(L^2) where L is the per-wall face
        # count. If this becomes hot, replace with a quantized hash bucket.
        used_hi = falses(length(hi_faces))
        for (s_lo, k_lo, cent_lo) in lo_faces
            best_j = 0
            best_d2 = T(Inf)
            for j in eachindex(hi_faces)
                used_hi[j] && continue
                cent_hi = hi_faces[j][3]
                d2 = zero(T)
                for d in 1:D
                    d == axis && continue
                    Δ = cent_lo[d] - cent_hi[d]
                    d2 += Δ * Δ
                end
                if d2 < best_d2
                    best_d2 = d2
                    best_j = j
                end
            end
            best_j == 0 && continue
            # Require a match within tol^2 (squared distance).
            best_d2 <= tol * tol + eps(T) || continue
            s_hi, k_hi, _ = hi_faces[best_j]
            mesh.simplex_neighbors[k_lo, s_lo] = Int32(s_hi)
            mesh.simplex_neighbors[k_hi, s_hi] = Int32(s_lo)
            used_hi[best_j] = true
        end
    end

    return mesh
end

# Vertex positions of the face of simplex `s` opposite vertex `k`.
# Returns an NTuple of length D (D vertices form a (D-1)-face).
@inline function _face_vertex_positions(mesh::SimplicialMesh{D, T},
                                         s::Integer, k::Integer) where {D, T}
    # Build a tuple of length D, skipping the k-th vertex.
    return ntuple(Val(D)) do i
        idx = i < k ? i : i + 1
        mesh.positions[mesh.simplex_vertices[idx, s]]
    end
end

# Centroid of the D face-vertices, with the periodic axis slot zeroed
# (so it doesn't contribute to matching).
@inline function _face_centroid_other_axes(face_pts, axis::Integer,
                                            ::Val{D}, ::Type{T}) where {D, T}
    return ntuple(Val(D)) do d
        if d == axis
            zero(T)
        else
            s = zero(T)
            for v in face_pts
                s += v[d]
            end
            s / T(length(face_pts))
        end
    end
end

# ============================================================================
# Dirichlet pinned simplices (PR-D)
# ============================================================================

"""
    pin_boundary_simplices!(mesh::SimplicialMesh, simplex_indices) -> mesh

Mark each simplex listed in `simplex_indices` as Dirichlet-pinned. Pinned
simplices are advisory: downstream Lagrangian motion code reads
`is_pinned` and clamps the vertex velocities of pinned simplices to zero
(or to a prescribed Dirichlet value). The geometric overlap and remap
operators do not consume this flag.

Indices outside `1:n_simplices(mesh)` raise `BoundsError`.
"""
function pin_boundary_simplices!(mesh::SimplicialMesh, simplex_indices)
    ns = mesh.n_simplices
    for i in simplex_indices
        (1 <= i <= ns) || throw(BoundsError(mesh, i))
        mesh.pinned[i] = true
    end
    return mesh
end

"""
    is_pinned(mesh::SimplicialMesh, i::Integer) -> Bool

Whether simplex `i` has been Dirichlet-pinned via
[`pin_boundary_simplices!`](@ref). Default is `false` for newly-constructed
meshes.
"""
@inline is_pinned(mesh::SimplicialMesh, i::Integer) = mesh.pinned[i]

