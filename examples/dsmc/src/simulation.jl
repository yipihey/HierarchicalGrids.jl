# Top-level Simulation type assembling mesh + particles + sampling, plus
# time-stepping driver and diagnostic queries.

"""
    SimParams

Physical and numerical parameters for the DSMC simulation.

# Fields
- `L::Float64`             : box side length (cubic box, [0, L)^3)
- `n_per_axis::Int`        : number of cells per axis (uniform mesh)
- `sigma_T::Float64`       : total cross-section (π * d^2 for hard spheres)
- `dt::Float64`            : timestep
- `c_max_init::Float64`    : initial estimate of c_rel_max per cell

# Derived (set by constructor)
- `cell_size::Float64`
- `V_cell::Float64`
"""
struct SimParams
    L::Float64
    n_per_axis::Int
    sigma_T::Float64
    dt::Float64
    c_max_init::Float64
    cell_size::Float64
    V_cell::Float64

    function SimParams(; L::Real, n_per_axis::Integer, sigma_T::Real,
                       dt::Real, c_max_init::Real)
        cell_size = Float64(L) / Int(n_per_axis)
        V_cell = cell_size^3
        return new(Float64(L), Int(n_per_axis), Float64(sigma_T), Float64(dt),
                   Float64(c_max_init), cell_size, V_cell)
    end
end

"""
    Simulation{ParticleLayout, SamplingLayout}

A complete DSMC simulation state. Composes:
- a HierarchicalMesh (currently uniform but the AMR capability is there)
- particles (FieldSet with chosen layout)
- per-cell sampling accumulators (FieldSet with chosen layout)
- per-cell collision state (c_max history)
- physical and numerical parameters

The two layout type parameters are independent: you can pick SoA particles
with AoS sampling, or any other combination, depending on what your kernels
prefer.
"""
mutable struct Simulation{PL, SL}
    params::SimParams
    mesh::HierarchicalMesh{3}
    particles::FieldSet
    sampling::FieldSet

    # Collision state per cell
    cell_lists::Vector{Vector{UInt32}}
    c_max_per_cell::Vector{Float64}

    # Bookkeeping
    step_count::Int
    rng::Random.AbstractRNG

    function Simulation(params::SimParams, n_particles::Integer;
                       particle_layout::HierarchicalGrids.Storage.AbstractLayout = SoA(),
                       sampling_layout::HierarchicalGrids.Storage.AbstractLayout = SoA(),
                       seed::Integer = 42)
        # Build a uniform mesh with n_per_axis cells per axis.
        # For n_per_axis = 2^k, refine the root k times.
        # For other n_per_axis, fall back to manual setup (not implemented here).
        n = params.n_per_axis
        if !ispow2(n)
            error("Currently only power-of-two n_per_axis is supported (got $n). " *
                  "This is a mini-app limitation, not a framework limitation.")
        end
        levels = trailing_zeros(n)
        mesh = HierarchicalMesh{3}()
        for _ in 1:levels
            # Refine all current leaves
            leaf_indices = Int[]
            for i in 1:n_cells(mesh)
                if is_leaf(mesh.cells[i])
                    push!(leaf_indices, i)
                end
            end
            refine_cells!(mesh, leaf_indices)
        end

        n_total_cells = n^3
        @assert count(is_leaf, mesh.cells) == n_total_cells

        # Allocate particles and sampling fields
        particles = allocate_particles(particle_layout, n_particles)
        sampling = allocate_sampling_fields(sampling_layout, n_total_cells)

        # Per-cell collision state
        cell_lists = [UInt32[] for _ in 1:n_total_cells]
        # Pre-allocate some capacity to avoid early reallocations
        avg_per_cell = max(1, n_particles ÷ n_total_cells)
        for cl in cell_lists
            sizehint!(cl, 2 * avg_per_cell)
        end
        c_max_per_cell = fill(params.c_max_init, n_total_cells)

        rng = Random.MersenneTwister(seed)

        # Initialize sampling to zero
        # (FieldSet contents are uninitialized after allocation)
        for c in 1:n_total_cells
            sampling.count_sum[c] = 0.0
            sampling.vel_sum[c] = (0.0, 0.0, 0.0)
            sampling.vel2_sum[c] = (0.0, 0.0, 0.0)
            sampling.n_samples[c] = UInt32(0)
        end

        return new{typeof(particle_layout), typeof(sampling_layout)}(
            params, mesh, particles, sampling,
            cell_lists, c_max_per_cell,
            0, rng
        )
    end
end

"""
    step!(sim::Simulation; do_sample::Bool = true)

Advance the simulation by one timestep. The standard time-splitting:
  1. Move (free-stream)
  2. Sort into cells
  3. Collide
  4. Sample (optional — pass do_sample=false during warmup)
"""
function step!(sim::Simulation; do_sample::Bool = true)
    # 1. Move
    move_particles!(sim.particles, sim.params.dt, sim.params.L)

    # 2. Sort
    sort_into_cells!(sim.particles, sim.cell_lists, sim.params.n_per_axis, sim.params.L)

    # 3. Collide
    n_collisions = collide_all_cells!(sim.particles, sim.cell_lists, sim.c_max_per_cell,
                                       sim.params.sigma_T, sim.params.V_cell,
                                       sim.params.dt, sim.rng)

    # 4. Sample
    if do_sample
        sample!(sim.sampling, sim.particles, sim.cell_lists)
    end

    sim.step_count += 1
    return n_collisions
end

"""
    run_simulation!(sim::Simulation, n_steps::Integer;
                    warmup::Integer = 0,
                    callback = nothing)

Run `n_steps` steps. The first `warmup` steps don't sample (so transient
isn't included in averages). Optional callback `callback(sim, step, n_coll)`
is called after each step with the simulation state.
"""
function run_simulation!(sim::Simulation, n_steps::Integer;
                         warmup::Integer = 0,
                         callback = nothing)
    for s in 1:n_steps
        n_coll = step!(sim; do_sample = (s > warmup))
        if callback !== nothing
            callback(sim, s, n_coll)
        end
    end
end

# ============================================================================
# Diagnostics — exact instantaneous quantities (not time-averaged)
# ============================================================================

"""
    total_particles(sim::Simulation)

Number of particles. Conserved exactly.
"""
total_particles(sim::Simulation) = n_elements(sim.particles)

"""
    total_momentum(sim::Simulation)

Total momentum (sum of all particle velocities, since all particles have
unit mass in our reduced units). Conserved exactly under elastic collisions.
"""
function total_momentum(sim::Simulation)
    n = n_elements(sim.particles)
    px, py, pz = 0.0, 0.0, 0.0
    @inbounds for i in 1:n
        v = sim.particles.vel[i]
        px += v[1]; py += v[2]; pz += v[3]
    end
    return (px, py, pz)
end

"""
    total_kinetic_energy(sim::Simulation)

Total kinetic energy (sum of 0.5 * v^2 over all particles, unit mass).
Conserved under elastic collisions.
"""
function total_kinetic_energy(sim::Simulation)
    n = n_elements(sim.particles)
    e = 0.0
    @inbounds for i in 1:n
        v = sim.particles.vel[i]
        e += 0.5 * (v[1]^2 + v[2]^2 + v[3]^2)
    end
    return e
end

"""
    temperature_per_axis(sim::Simulation)

Instantaneous temperature components T_x, T_y, T_z computed from all
particles. Returns NaN if N < 2.
"""
function temperature_per_axis(sim::Simulation)
    n = n_elements(sim.particles)
    n < 2 && return (NaN, NaN, NaN)

    # Pass 1: mean
    sx, sy, sz = 0.0, 0.0, 0.0
    @inbounds for i in 1:n
        v = sim.particles.vel[i]
        sx += v[1]; sy += v[2]; sz += v[3]
    end
    mx, my, mz = sx / n, sy / n, sz / n

    # Pass 2: variance
    vx, vy, vz = 0.0, 0.0, 0.0
    @inbounds for i in 1:n
        v = sim.particles.vel[i]
        vx += (v[1] - mx)^2
        vy += (v[2] - my)^2
        vz += (v[3] - mz)^2
    end
    return (vx / n, vy / n, vz / n)
end

"""
    mean_velocity(sim::Simulation)

Mean velocity (= total_momentum / total_particles).
"""
function mean_velocity(sim::Simulation)
    n = n_elements(sim.particles)
    p = total_momentum(sim)
    return (p[1] / n, p[2] / n, p[3] / n)
end

# ============================================================================
# Initial conditions
# ============================================================================

"""
    init_two_temperature_relaxation!(sim::Simulation; T_x::Real, T_yz::Real)

Initialize particles for the thermal-relaxation validation problem:

- Positions uniform in the box.
- Velocities sampled from a Maxwell-Boltzmann with anisotropic temperature:
  T_xx = T_x and T_yy = T_zz = T_yz. The system relaxes toward an isotropic
  Maxwellian with T_eq = (T_x + 2*T_yz) / 3.

This is equivalent to a strongly anisotropic distribution that's a standard
DSMC validation problem.
"""
function init_two_temperature_relaxation!(sim::Simulation; T_x::Real, T_yz::Real)
    rng = sim.rng
    L = sim.params.L
    n = n_elements(sim.particles)

    sigma_x = sqrt(Float64(T_x))
    sigma_yz = sqrt(Float64(T_yz))

    @inbounds for i in 1:n
        # Position: uniform in box
        sim.particles.pos[i] = (L * rand(rng), L * rand(rng), L * rand(rng))
        # Velocity: anisotropic Maxwell-Boltzmann
        sim.particles.vel[i] = (sigma_x * randn(rng),
                                 sigma_yz * randn(rng),
                                 sigma_yz * randn(rng))
        sim.particles.cell[i] = UInt32(0)  # will be set by sort
    end

    # Subtract any net drift to give cleaner conservation diagnostics
    px, py, pz = 0.0, 0.0, 0.0
    @inbounds for i in 1:n
        v = sim.particles.vel[i]
        px += v[1]; py += v[2]; pz += v[3]
    end
    mx, my, mz = px / n, py / n, pz / n
    @inbounds for i in 1:n
        v = sim.particles.vel[i]
        sim.particles.vel[i] = (v[1] - mx, v[2] - my, v[3] - mz)
    end

    # Initial sort + reset c_max based on observed velocities
    sort_into_cells!(sim.particles, sim.cell_lists, sim.params.n_per_axis, sim.params.L)

    # Estimate c_max per cell as ~3 * sqrt(2 * T_max), generous to start
    c_init = 4.0 * sqrt(2.0 * max(Float64(T_x), Float64(T_yz)))
    fill!(sim.c_max_per_cell, c_init)

    return sim
end
