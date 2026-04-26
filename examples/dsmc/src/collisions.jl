# NTC (No-Time-Counter) collision algorithm for hard-sphere particles.
#
# Reference: Bird, "Molecular Gas Dynamics and the Direct Simulation of Gas
# Flows" (1994), Chapter 11. Also Bird, "The DSMC Method" (2013).
#
# Algorithm summary (per cell, per step):
#   M = ceil(0.5 * N * (N-1) * F * sigma_T_max * c_max * dt / V)
#   For m = 1..M:
#     pick two distinct particles i, j from cell
#     compute c_rel = |v_i - v_j|
#     accept with probability (sigma_T(c_rel) * c_rel) / (sigma_T_max * c_max)
#     if accepted: do elastic collision
#       update sigma_T_max * c_max if c_rel > current max
#
# For hard spheres, sigma_T = pi * d^2 is constant, so the acceptance
# probability simplifies to c_rel / c_max.
#
# F is the number of real molecules per simulator particle (the "weight").
# For our reduced units we set F = 1 (each particle represents one molecule).

"""
    do_elastic_collision!(particles, i::Integer, j::Integer, rng)

Perform an elastic collision between particles i and j with hard-sphere
kinematics: the relative velocity magnitude is preserved, and the post-
collision relative velocity direction is chosen isotropically in the
center-of-mass frame.

Conserves momentum and kinetic energy of the pair exactly.
"""
@inline function do_elastic_collision!(particles, i::Integer, j::Integer, rng::AbstractRNG)
    vi = particles.vel[i]
    vj = particles.vel[j]

    # Center-of-mass velocity (equal-mass particles)
    vcm = ((vi[1] + vj[1]) * 0.5,
           (vi[2] + vj[2]) * 0.5,
           (vi[3] + vj[3]) * 0.5)

    # Relative velocity magnitude
    dv = (vi[1] - vj[1], vi[2] - vj[2], vi[3] - vj[3])
    c_rel = sqrt(dv[1]^2 + dv[2]^2 + dv[3]^2)

    # Pick isotropic direction for new relative velocity
    # cos(theta) uniform in [-1, 1], phi uniform in [0, 2*pi]
    cos_th = 2.0 * rand(rng) - 1.0
    sin_th = sqrt(max(0.0, 1.0 - cos_th * cos_th))
    phi = 2.0 * pi * rand(rng)
    sin_phi, cos_phi = sincos(phi)

    new_rel = (c_rel * sin_th * cos_phi,
               c_rel * sin_th * sin_phi,
               c_rel * cos_th)

    # New velocities: v = vcm ± new_rel / 2
    particles.vel[i] = (vcm[1] + 0.5 * new_rel[1],
                        vcm[2] + 0.5 * new_rel[2],
                        vcm[3] + 0.5 * new_rel[3])
    particles.vel[j] = (vcm[1] - 0.5 * new_rel[1],
                        vcm[2] - 0.5 * new_rel[2],
                        vcm[3] - 0.5 * new_rel[3])
end

"""
    collide_in_cell!(particles, cell_particle_indices, c_max_state,
                     sigma_T, V_cell, dt, rng)

Run the NTC collision algorithm for one cell.

Arguments:
- `particles`: the particle FieldSet
- `cell_particle_indices`: Vector{UInt32} of particle indices in this cell
- `c_max_state`: a Ref{Float64} holding (and updated with) the running
  estimate of c_rel_max for this cell
- `sigma_T`: total cross-section (constant for hard spheres = π * d^2)
- `V_cell`: volume of the cell
- `dt`: timestep
- `rng`: random number generator

Returns the number of collisions actually performed.

Notes on c_max:
  c_max should track the maximum relative speed observed in the cell. We
  store it per-cell in `c_max_state` so it persists across timesteps. If
  the algorithm sees a c_rel > c_max, it updates c_max — the standard NTC
  procedure. Initial c_max can be estimated from temperature; we use a
  generous starting value.

Notes on F (particle weight):
  F = 1 in this mini-app; each simulator particle is one real molecule.
"""
function collide_in_cell!(particles,
                          cell_particle_indices::Vector{UInt32},
                          c_max_state::Ref{Float64},
                          sigma_T::Float64,
                          V_cell::Float64,
                          dt::Float64,
                          rng::AbstractRNG)
    N = length(cell_particle_indices)
    if N < 2
        return 0
    end

    F = 1.0  # one real molecule per particle
    c_max = c_max_state[]

    # Number of candidate pairs to try
    # M = 0.5 * N * (N-1) * F * sigma_T * c_max * dt / V_cell
    M_real = 0.5 * N * (N - 1) * F * sigma_T * c_max * dt / V_cell
    M = floor(Int, M_real)
    # Stochastic rounding for the fractional part
    if rand(rng) < (M_real - M)
        M += 1
    end

    n_actual_collisions = 0

    @inbounds for _ in 1:M
        # Pick two distinct particles uniformly at random from the cell
        # (rejection sample to avoid i == j)
        i_local = rand(rng, 1:N)
        j_local = rand(rng, 1:N)
        while j_local == i_local
            j_local = rand(rng, 1:N)
        end
        i = cell_particle_indices[i_local]
        j = cell_particle_indices[j_local]

        vi = particles.vel[i]
        vj = particles.vel[j]
        dv = (vi[1] - vj[1], vi[2] - vj[2], vi[3] - vj[3])
        c_rel = sqrt(dv[1]^2 + dv[2]^2 + dv[3]^2)

        # Update c_max if this pair is faster
        if c_rel > c_max
            c_max = c_rel
        end

        # Acceptance: P_accept = sigma_T * c_rel / (sigma_T * c_max) = c_rel / c_max
        # (For hard spheres; for variable hard spheres, sigma_T also depends on c_rel.)
        if rand(rng) * c_max < c_rel
            do_elastic_collision!(particles, i, j, rng)
            n_actual_collisions += 1
        end
    end

    c_max_state[] = c_max
    return n_actual_collisions
end

"""
    collide_all_cells!(particles, cell_lists, c_max_per_cell, sigma_T,
                       V_cell, dt, rng)

Run NTC collisions in every cell. Returns the total number of collisions.
"""
function collide_all_cells!(particles,
                            cell_lists::Vector{Vector{UInt32}},
                            c_max_per_cell::Vector{Float64},
                            sigma_T::Float64,
                            V_cell::Float64,
                            dt::Float64,
                            rng::AbstractRNG)
    total = 0
    @inbounds for c in eachindex(cell_lists)
        c_max_ref = Ref(c_max_per_cell[c])
        total += collide_in_cell!(particles, cell_lists[c], c_max_ref,
                                   sigma_T, V_cell, dt, rng)
        c_max_per_cell[c] = c_max_ref[]
    end
    return total
end
