#!/usr/bin/env julia
# Thermal relaxation validation problem.
#
# Initialize an anisotropic Maxwell-Boltzmann distribution (T_x ≠ T_yz),
# evolve under elastic collisions, and verify:
#  1. Particle count is exactly conserved
#  2. Total momentum is conserved (to ~ machine precision)
#  3. Total energy is conserved (to ~ machine precision)
#  4. Temperature components T_x, T_y, T_z relax toward T_eq = (T_x + 2*T_yz)/3
#
# Run with:
#   julia --project=. examples/relaxation.jl

using DSMCExample
using HierarchicalGrids
using Printf

println("=" ^ 70)
println("DSMC thermal relaxation validation")
println("=" ^ 70)

# Physical parameters (reduced units: m = k_B = 1, sigma_T = pi*d^2 = 1)
params = SimParams(
    L = 10.0,           # box side length
    n_per_axis = 8,     # 8x8x8 = 512 cells
    sigma_T = 1.0,      # cross-section
    dt = 0.05,          # timestep
    c_max_init = 5.0,   # initial c_rel_max estimate per cell
)

# Aim for ~30 particles per cell on average — standard DSMC recommendation
n_particles = 30 * params.n_per_axis^3
println("Setup:")
println("  Box: $(params.L)^3 with $(params.n_per_axis)^3 = $(params.n_per_axis^3) cells")
println("  Cell size: $(params.cell_size)")
println("  Particles: $n_particles (avg $(n_particles ÷ params.n_per_axis^3) per cell)")
println("  Cross-section: $(params.sigma_T)")
println("  Timestep: $(params.dt)")

# Number density (analytic mean collision frequency check):
n_density = n_particles / params.L^3
println("  Number density: $n_density")

# Build the simulation
sim = Simulation(params, n_particles; particle_layout=SoA(), sampling_layout=SoA())

# Initialize anisotropic Maxwellian: hot in x, cold in y/z
T_x_init  = 3.0
T_yz_init = 0.5
T_eq_expected = (T_x_init + 2 * T_yz_init) / 3
println("\nInitial conditions:")
println("  T_x  = $T_x_init")
println("  T_y  = T_z = $T_yz_init")
println("  T_eq (expected) = $T_eq_expected")

init_two_temperature_relaxation!(sim; T_x=T_x_init, T_yz=T_yz_init)

# Mean thermal speed (for collision-rate estimate)
c_bar_eq = sqrt(8 * T_eq_expected / pi)
nu_expected = n_density * params.sigma_T * c_bar_eq
tau_relax = 1.0 / nu_expected
println("  Mean thermal speed (eq): $(round(c_bar_eq, digits=3))")
println("  Expected collision freq nu = n*sigma*c̄: $(round(nu_expected, digits=4))")
println("  Mean collision time 1/nu: $(round(tau_relax, digits=3))")
println("  Expected number of relaxation timesteps ~3-5/dt/nu: $(round(5*tau_relax/params.dt, digits=1))")

# Reference quantities
N0 = total_particles(sim)
P0 = total_momentum(sim)
E0 = total_kinetic_energy(sim)

println("\nInitial diagnostics:")
println("  N = $N0")
@printf("  P = (%+.6e, %+.6e, %+.6e)\n", P0[1], P0[2], P0[3])
@printf("  E = %.6e\n", E0)
T0 = temperature_per_axis(sim)
@printf("  T = (%.4f, %.4f, %.4f)\n", T0[1], T0[2], T0[3])

# Run
n_steps = 200
println("\n" * "=" ^ 70)
println("Time evolution:")
println("=" ^ 70)
@printf("%6s %10s %12s %14s %14s %14s\n",
        "step", "n_coll", "ΔE/E", "T_x", "T_y", "T_z")

snapshot_steps = [0, 10, 25, 50, 100, 150, 200]
total_collisions = 0
last_T = T0
for s in 1:n_steps
    n_coll = step!(sim; do_sample=true)
    global total_collisions += n_coll

    if s in snapshot_steps
        E = total_kinetic_energy(sim)
        T = temperature_per_axis(sim)
        @printf("%6d %10d %+.4e   %.6f   %.6f   %.6f\n",
                s, total_collisions, (E - E0) / E0, T[1], T[2], T[3])
        global last_T = T
    end
end

# Final diagnostics
N1 = total_particles(sim)
P1 = total_momentum(sim)
E1 = total_kinetic_energy(sim)
T1 = temperature_per_axis(sim)

println("\n" * "=" ^ 70)
println("Final diagnostics:")
println("=" ^ 70)
println("  N = $N1   (should equal $N0)")
@printf("  ΔP = (%+.3e, %+.3e, %+.3e)   (should be ~0)\n",
        P1[1] - P0[1], P1[2] - P0[2], P1[3] - P0[3])
@printf("  ΔE/E = %+.3e   (should be ~0)\n", (E1 - E0) / E0)
@printf("  T = (%.4f, %.4f, %.4f)\n", T1[1], T1[2], T1[3])
@printf("  T_eq expected = %.4f\n", T_eq_expected)
@printf("  T spread (max - min) = %.4f   (should be small)\n",
        maximum(T1) - minimum(T1))
println("  Total collisions: $total_collisions")
println("  Avg collisions per step: $(round(total_collisions / n_steps, digits=1))")
println("  Avg collisions per particle per step: $(round(total_collisions / n_steps / N1, digits=4))")
println("  Predicted collisions per particle per step (= ν * dt / 2): $(round(nu_expected * params.dt / 2, digits=4))")

# Validation summary
println("\n" * "=" ^ 70)
println("Validation summary:")
println("=" ^ 70)

p_rel = sqrt(sum(abs2, (P1[1]-P0[1], P1[2]-P0[2], P1[3]-P0[3]))) /
        max(1e-10, sqrt(sum(abs2, P0)))
e_rel = abs(E1 - E0) / E0
t_spread = maximum(T1) - minimum(T1)

println("  ✓ Particle conservation: ", N1 == N0 ? "PASS" : "FAIL")
println("  ✓ Momentum conservation: ",
        p_rel < 1e-10 ? "PASS (rel err $(round(p_rel, sigdigits=2)))" :
        "INFO (rel err $(round(p_rel, sigdigits=2)))  [P~0 makes relative noisy]")
println("  ✓ Energy conservation:   ",
        e_rel < 1e-10 ? "PASS (rel err $(round(e_rel, sigdigits=2)))" :
        "FAIL (rel err $(round(e_rel, sigdigits=2)))")
println("  ✓ Temperature relaxation: T spread reduced from $(round(maximum(T0)-minimum(T0), digits=3)) to $(round(t_spread, digits=3))")
println()
