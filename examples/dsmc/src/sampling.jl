# Per-cell time-averaged macroscopic quantities.
#
# We accumulate sums across many timesteps so that we can compute average
# density n, mean velocity u, and temperature T per cell. This is the
# standard DSMC sampling pattern: instantaneous values are noisy due to
# small particle counts; time-averages converge.
#
# We use a HierarchicalGrids FieldSet for the accumulators — same Storage
# abstraction as for particles. This shows the same machinery serving two
# different purposes (particles, cell quantities) within one app.

"""
    allocate_sampling_fields(layout, n_cells)

Allocate per-cell accumulators for sampling. Fields:

- `count_sum :: Float64` — sum of particle counts over samples
- `vel_sum :: NTuple{3, Float64}` — sum of (sum of velocities in cell) over samples
- `vel2_sum :: NTuple{3, Float64}` — sum of (sum of v[d]^2 in cell) over samples
- `n_samples :: UInt32` — how many samples have been taken

We use Float64 for the velocity sums because they can grow large over many
samples and we want accuracy.
"""
function allocate_sampling_fields(layout::HierarchicalGrids.Storage.AbstractLayout,
                                  n_cells::Integer)
    return allocate_fields(layout, Int(n_cells);
                           count_sum = Float64,
                           vel_sum   = NTuple{3, Float64},
                           vel2_sum  = NTuple{3, Float64},
                           n_samples = UInt32)
end

"""
    reset_sampling!(samples)

Zero out all sampling accumulators. Use after warmup or to start a fresh
averaging window.
"""
function reset_sampling!(samples)
    n = n_elements(samples)
    @inbounds for c in 1:n
        samples.count_sum[c] = 0.0
        samples.vel_sum[c] = (0.0, 0.0, 0.0)
        samples.vel2_sum[c] = (0.0, 0.0, 0.0)
        samples.n_samples[c] = UInt32(0)
    end
end

"""
    sample!(samples, particles, cell_lists)

Add the current state of `particles` to the sampling accumulators, one
contribution per cell. Call this once per timestep (or every K timesteps,
your choice — affects only the convergence rate of the averages).
"""
function sample!(samples, particles, cell_lists::Vector{Vector{UInt32}})
    n_cells = n_elements(samples)
    @assert length(cell_lists) == n_cells

    @inbounds for c in 1:n_cells
        plist = cell_lists[c]
        N = length(plist)

        # Per-cell sums for this sample
        sum_v = (0.0, 0.0, 0.0)
        sum_v2 = (0.0, 0.0, 0.0)
        for k in 1:N
            v = particles.vel[plist[k]]
            sum_v = (sum_v[1] + v[1], sum_v[2] + v[2], sum_v[3] + v[3])
            sum_v2 = (sum_v2[1] + v[1]^2, sum_v2[2] + v[2]^2, sum_v2[3] + v[3]^2)
        end

        # Accumulate into time-sums
        samples.count_sum[c] += Float64(N)
        prev_v = samples.vel_sum[c]
        samples.vel_sum[c] = (prev_v[1] + sum_v[1], prev_v[2] + sum_v[2], prev_v[3] + sum_v[3])
        prev_v2 = samples.vel2_sum[c]
        samples.vel2_sum[c] = (prev_v2[1] + sum_v2[1], prev_v2[2] + sum_v2[2], prev_v2[3] + sum_v2[3])
        samples.n_samples[c] += UInt32(1)
    end
end

"""
    sampled_density(samples, c, V_cell)

Time-averaged number density in cell c (particles per volume).
Returns NaN if no samples taken.
"""
function sampled_density(samples, c::Integer, V_cell::Float64)
    n_s = samples.n_samples[c]
    n_s == 0 && return NaN
    return samples.count_sum[c] / (Float64(n_s) * V_cell)
end

"""
    sampled_velocity(samples, c)

Time-averaged mean velocity in cell c, returned as a 3-tuple. Returns
(NaN, NaN, NaN) if no particles ever sampled.
"""
function sampled_velocity(samples, c::Integer)
    csum = samples.count_sum[c]
    csum == 0.0 && return (NaN, NaN, NaN)
    vs = samples.vel_sum[c]
    return (vs[1] / csum, vs[2] / csum, vs[3] / csum)
end

"""
    sampled_temperature(samples, c)

Time-averaged temperature components in cell c. Returns (T_x, T_y, T_z)
where T_d = ⟨v_d^2⟩ - ⟨v_d⟩^2 (kinetic temperature in mass-1, k_B-1 units;
i.e., k_B T_d / m).

Returns (NaN, NaN, NaN) if not enough samples.
"""
function sampled_temperature(samples, c::Integer)
    csum = samples.count_sum[c]
    csum < 2.0 && return (NaN, NaN, NaN)
    vs = samples.vel_sum[c]
    v2s = samples.vel2_sum[c]
    # mean v^2 - (mean v)^2
    Tx = v2s[1] / csum - (vs[1] / csum)^2
    Ty = v2s[2] / csum - (vs[2] / csum)^2
    Tz = v2s[3] / csum - (vs[3] / csum)^2
    return (Tx, Ty, Tz)
end
