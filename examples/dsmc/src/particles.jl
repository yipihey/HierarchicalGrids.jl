# Particle representation and cell binning.
#
# Particles are stored as a HierarchicalGrids.FieldSet with whatever layout
# the user picked. Required fields:
#   pos :: NTuple{3, Float64} — position in [0, L)^3 (periodic box)
#   vel :: NTuple{3, Float64} — velocity
# Optional:
#   cell :: UInt32 — current cell index (bin), maintained by sort_into_cells!
#
# Cell binning uses the mesh's leaf cells. For a uniform mesh of side N
# (N cells per axis after refining the root k times with N = 2^k), the
# leaf cells form a regular Cartesian grid and binning is just integer
# division of position by cell size. We use this fast path because the
# mini-app uses a uniform mesh; a general AMR-aware binning would walk
# the tree.

"""
    allocate_particles(layout, n_particles)

Allocate a particle FieldSet with the given layout. Includes pos, vel, and
cell-index fields.
"""
function allocate_particles(layout::HierarchicalGrids.Storage.AbstractLayout, n::Integer)
    return allocate_fields(layout, Int(n);
                           pos = NTuple{3, Float64},
                           vel = NTuple{3, Float64},
                           cell = UInt32)
end

"""
    apply_periodic_bcs!(particles, L)

Wrap particle positions into [0, L) along each axis.
"""
function apply_periodic_bcs!(particles, L::Float64)
    n = n_elements(particles)
    @inbounds for i in 1:n
        p = particles.pos[i]
        # Wrap each component
        x = p[1] - L * floor(p[1] / L)
        y = p[2] - L * floor(p[2] / L)
        z = p[3] - L * floor(p[3] / L)
        particles.pos[i] = (x, y, z)
    end
end

"""
    cell_index_for_position(pos, n_per_axis, L)

Compute the linear cell index for a position in a uniform N×N×N grid in box [0, L)^3.

Returns a 1-based index in `1:n_per_axis^3`. Layout: i + n*j + n*n*k + 1
where (i, j, k) are 0-based axis indices.
"""
@inline function cell_index_for_position(pos::NTuple{3, Float64},
                                         n_per_axis::Int, L::Float64)
    inv_cell = n_per_axis / L
    # Clamp into [0, n_per_axis-1] to handle floating-point edge cases at L
    ix = clamp(floor(Int, pos[1] * inv_cell), 0, n_per_axis - 1)
    iy = clamp(floor(Int, pos[2] * inv_cell), 0, n_per_axis - 1)
    iz = clamp(floor(Int, pos[3] * inv_cell), 0, n_per_axis - 1)
    return UInt32(ix + n_per_axis * iy + n_per_axis * n_per_axis * iz + 1)
end

"""
    sort_into_cells!(particles, cell_lists, n_per_axis, L)

Bin all particles into their containing cells. After this call:

- `particles.cell[i]` holds the cell index for particle i.
- `cell_lists[c]` holds the indices of all particles currently in cell c.

The `cell_lists` argument must be a Vector{Vector{UInt32}} of length
`n_per_axis^3`. We empty each list, then refill — no allocations on hot path
after warmup.
"""
function sort_into_cells!(particles, cell_lists::Vector{Vector{UInt32}},
                          n_per_axis::Int, L::Float64)
    # Empty the cell lists (preserve their capacity for reuse)
    @inbounds for c in eachindex(cell_lists)
        empty!(cell_lists[c])
    end

    # Assign each particle to its cell
    n = n_elements(particles)
    @inbounds for i in 1:n
        c = cell_index_for_position(particles.pos[i], n_per_axis, L)
        particles.cell[i] = c
        push!(cell_lists[c], UInt32(i))
    end

    return cell_lists
end
