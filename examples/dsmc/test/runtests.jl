using Test
using DSMCExample
using HierarchicalGrids
using Random
using Statistics

@testset "DSMC mini-app" begin

    @testset "Particle allocation" begin
        for layout in (SoA(), AoS(), Blocked{8, SoA}())
            particles = DSMCExample.allocate_particles(layout, 100)
            @test n_elements(particles) == 100
            # Write and read
            particles.pos[1] = (1.0, 2.0, 3.0)
            particles.vel[1] = (0.1, 0.2, 0.3)
            particles.cell[1] = UInt32(7)
            @test particles.pos[1] == (1.0, 2.0, 3.0)
            @test particles.vel[1] == (0.1, 0.2, 0.3)
            @test particles.cell[1] == UInt32(7)
        end
    end

    @testset "Cell index computation" begin
        # 4x4x4 grid in [0, 4)^3 — each cell has size 1
        @test DSMCExample.cell_index_for_position((0.5, 0.5, 0.5), 4, 4.0) == UInt32(1)
        @test DSMCExample.cell_index_for_position((1.5, 0.5, 0.5), 4, 4.0) == UInt32(2)
        @test DSMCExample.cell_index_for_position((0.5, 1.5, 0.5), 4, 4.0) == UInt32(5)  # 0 + 4*1 + 0 + 1
        @test DSMCExample.cell_index_for_position((0.5, 0.5, 1.5), 4, 4.0) == UInt32(17) # 0 + 0 + 16 + 1
        # Edge clamping
        @test DSMCExample.cell_index_for_position((4.0 - 1e-15, 0.5, 0.5), 4, 4.0) == UInt32(4)
    end

    @testset "Periodic boundary conditions" begin
        particles = DSMCExample.allocate_particles(SoA(), 5)
        particles.pos[1] = (0.5, 0.5, 0.5)   # inside
        particles.pos[2] = (10.5, 0.5, 0.5)  # past +x boundary
        particles.pos[3] = (-0.5, 0.5, 0.5)  # past -x boundary
        particles.pos[4] = (0.5, 0.5, 20.0)  # past +z, big
        particles.pos[5] = (10.0, 10.0, 10.0) # exactly at boundary
        L = 10.0
        DSMCExample.apply_periodic_bcs!(particles, L)
        @test all(0.0 <= particles.pos[i][d] < L for i in 1:5, d in 1:3)
        @test particles.pos[1] == (0.5, 0.5, 0.5)
        @test all(isapprox(particles.pos[2][d], (0.5, 0.5, 0.5)[d]; atol=1e-12) for d in 1:3)
        @test all(isapprox(particles.pos[3][d], (9.5, 0.5, 0.5)[d]; atol=1e-12) for d in 1:3)
    end

    @testset "Sort into cells" begin
        params = SimParams(L=4.0, n_per_axis=4, sigma_T=1.0, dt=0.1, c_max_init=1.0)
        sim = Simulation(params, 100)
        # Place particles uniformly
        Random.seed!(sim.rng, 42)
        for i in 1:n_elements(sim.particles)
            sim.particles.pos[i] = (params.L * rand(sim.rng),
                                    params.L * rand(sim.rng),
                                    params.L * rand(sim.rng))
        end
        DSMCExample.sort_into_cells!(sim.particles, sim.cell_lists, params.n_per_axis, params.L)

        # All particles should be in some cell
        total = sum(length, sim.cell_lists)
        @test total == 100

        # Each particle's cell should match its position
        for i in 1:n_elements(sim.particles)
            expected = DSMCExample.cell_index_for_position(sim.particles.pos[i],
                                                           params.n_per_axis, params.L)
            @test sim.particles.cell[i] == expected
            @test i in sim.cell_lists[expected]
        end
    end

    @testset "Elastic collision conserves momentum and energy of pair" begin
        particles = DSMCExample.allocate_particles(SoA(), 2)
        particles.vel[1] = (1.0, 0.5, -0.3)
        particles.vel[2] = (-0.4, 0.2, 0.7)

        v1 = particles.vel[1]; v2 = particles.vel[2]
        p_before = (v1[1]+v2[1], v1[2]+v2[2], v1[3]+v2[3])
        e_before = 0.5*(v1[1]^2+v1[2]^2+v1[3]^2 + v2[1]^2+v2[2]^2+v2[3]^2)

        rng = MersenneTwister(42)
        # Run many collisions to confirm conservation each time
        for trial in 1:1000
            DSMCExample.do_elastic_collision!(particles, 1, 2, rng)
            v1 = particles.vel[1]; v2 = particles.vel[2]
            p_after = (v1[1]+v2[1], v1[2]+v2[2], v1[3]+v2[3])
            e_after = 0.5*(v1[1]^2+v1[2]^2+v1[3]^2 + v2[1]^2+v2[2]^2+v2[3]^2)
            @test all(abs(p_after[d] - p_before[d]) < 1e-12 for d in 1:3)
            @test abs(e_after - e_before) < 1e-12
        end
    end

    @testset "Full step conservation" begin
        # Conservation across many steps with collisions
        params = SimParams(L=8.0, n_per_axis=4, sigma_T=1.0, dt=0.05, c_max_init=5.0)
        sim = Simulation(params, 2000; seed=123)
        init_two_temperature_relaxation!(sim; T_x=2.0, T_yz=1.0)

        N0 = total_particles(sim)
        P0 = total_momentum(sim)
        E0 = total_kinetic_energy(sim)

        for s in 1:50
            step!(sim; do_sample=false)
        end

        @test total_particles(sim) == N0
        @test total_kinetic_energy(sim) ≈ E0 atol=1e-9
        # Momentum is initially zero (subtracted off), should stay near zero
        P1 = total_momentum(sim)
        @test all(abs(P1[d] - P0[d]) < 1e-10 for d in 1:3)
    end

    @testset "Thermal relaxation: anisotropy decreases" begin
        params = SimParams(L=8.0, n_per_axis=4, sigma_T=1.0, dt=0.05, c_max_init=5.0)
        sim = Simulation(params, 5000; seed=456)
        init_two_temperature_relaxation!(sim; T_x=4.0, T_yz=1.0)

        T_init = temperature_per_axis(sim)
        spread_init = maximum(T_init) - minimum(T_init)
        @test spread_init > 2.0  # large initial anisotropy

        # Run long enough for relaxation
        for s in 1:300
            step!(sim; do_sample=false)
        end

        T_final = temperature_per_axis(sim)
        spread_final = maximum(T_final) - minimum(T_final)
        # Should have substantially relaxed
        @test spread_final < 0.3 * spread_init

        # Equipartition: total energy preserved, so each component should be ~T_eq = (4 + 1 + 1)/3 = 2
        T_eq = (4.0 + 1.0 + 1.0) / 3
        @test all(abs(T_final[d] - T_eq) < 0.2 for d in 1:3)
    end

    @testset "Layout independence: same physics, different layouts" begin
        # Run two simulations with identical seeds and parameters but different
        # particle layouts. They should produce bit-identical results since the
        # physics kernels are layout-agnostic.
        params = SimParams(L=4.0, n_per_axis=2, sigma_T=1.0, dt=0.1, c_max_init=3.0)

        results = []
        for layout in (SoA(), AoS())
            sim = Simulation(params, 200; particle_layout=layout, seed=789)
            init_two_temperature_relaxation!(sim; T_x=2.0, T_yz=1.0)
            for s in 1:30
                step!(sim; do_sample=false)
            end
            push!(results, (T=temperature_per_axis(sim),
                          E=total_kinetic_energy(sim),
                          P=total_momentum(sim)))
        end

        # Same RNG seed and same kernel order should give identical results
        @test results[1].E ≈ results[2].E rtol=1e-12
        @test all(isapprox(results[1].T[d], results[2].T[d]; rtol=1e-12) for d in 1:3)
        @test all(abs(results[1].P[d] - results[2].P[d]) < 1e-12 for d in 1:3)
    end

    @testset "Sampling accumulators" begin
        params = SimParams(L=4.0, n_per_axis=2, sigma_T=1.0, dt=0.1, c_max_init=3.0)
        sim = Simulation(params, 800; seed=42)
        init_two_temperature_relaxation!(sim; T_x=1.0, T_yz=1.0)  # isotropic, T=1

        # Take many samples
        for s in 1:200
            step!(sim; do_sample=true)
        end

        # Check sampled densities sum correctly (~ total particles / box volume)
        n_total_sampled = sum(c -> sampled_density(sim.sampling, c, params.V_cell) * params.V_cell,
                              1:n_elements(sim.sampling))
        # Each cell's density * V_cell ≈ avg particles in cell; sum ≈ total particles
        @test n_total_sampled ≈ Float64(total_particles(sim)) rtol=0.05

        # Sampled temperature averaged over cells should converge to T_eq = 1
        T_per_cell = [sampled_temperature(sim.sampling, c) for c in 1:n_elements(sim.sampling)]
        T_avg = (Statistics.mean(t[1] for t in T_per_cell),
                 Statistics.mean(t[2] for t in T_per_cell),
                 Statistics.mean(t[3] for t in T_per_cell))
        # With 800 particles in 8 cells = 100/cell × 200 samples = 20K samples,
        # noise is small; expect within ~10%
        @test all(0.85 < T_avg[d] < 1.15 for d in 1:3)
    end

end
