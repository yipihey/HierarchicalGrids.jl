# DSMC mini-app

A minimal Direct Simulation Monte Carlo (DSMC) solver built on HierarchicalGrids.jl.

## What this demonstrates

This example shows how to build a real physics application on top of the framework:

- **Particle storage via Layer 2.5**: particles are stored in a `FieldSet` with a switchable layout (SoA, AoS, Blocked). The same physics kernels work with all three.
- **Cells as collision bins**: the hierarchical mesh's leaf cells serve as DSMC's collision cells. Particles are binned per cell each step.
- **Per-cell time-averaged macroscopics**: density, mean velocity, and temperature are accumulated as cell fields using the same Storage abstraction, separate from the mesh.
- **Conservation diagnostics**: particle count, momentum, and energy should be conserved exactly under elastic collisions; the example verifies this.

## What it doesn't do

This is a mini-app, not a full DSMC code:

- Single species (hard-sphere argon).
- Periodic boundaries only — no walls, no inflow.
- Uniform mesh — no AMR (the hierarchical mesh capability is there, but DSMC uniformly resolved is the simplest validation case).
- No internal energy modes (rotational, vibrational).
- No chemical reactions.
- The collision algorithm is the standard No-Time-Counter (NTC) method.

The point is to show framework usage, not to be a competitive DSMC implementation. The underlying physics is correct, but production DSMC codes (SPARTA, dsmcFoam, MGDS) have many more features.

## Algorithm

Standard time-splitting:
1. **Move**: free-stream particles for dt: `x += v * dt`, applying periodic BCs.
2. **Sort**: bin particles into cells based on their new positions.
3. **Collide**: in each cell, run NTC collisions for time dt.
4. **Sample**: accumulate per-cell macroscopic quantities (n, u, T) for time-averaging.

The NTC scheme picks `M = 0.5 * N * (N-1) * sigma_T_max * c_max * dt / V` candidate pairs per cell, then accepts each with probability `(sigma_T * c_rel) / (sigma_T_max * c_rel_max)`. For hard spheres, sigma_T is constant, so this reduces to acceptance probability `c_rel / c_rel_max`. Accepted pairs collide elastically with isotropic post-collision velocity in the center-of-mass frame.

## Validation

The included test problem is **thermal relaxation**: initialize two velocity populations (e.g., split into two Maxwellians at different temperatures), evolve, and verify:

1. Total energy is conserved (to round-off).
2. Total momentum is conserved (to round-off).
3. Particle count is conserved exactly.
4. Temperature components T_xx, T_yy, T_zz equilibrate over a few mean collision times.

The expected relaxation timescale is τ ~ 1/ν where ν = n * σ * c̄ is the collision frequency. For our default parameters this is a few units of dt.

## Running

```julia
cd examples/dsmc
julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
julia --project=. examples/relaxation.jl
```

This runs the thermal-relaxation problem, prints conservation diagnostics each step, and reports the temperature equilibration.

For tests:

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Files

- `src/DSMCExample.jl` — main module
- `src/particles.jl` — particle types, sorting into cells
- `src/collisions.jl` — NTC collision algorithm, hard-sphere kinematics
- `src/sampling.jl` — per-cell macroscopic accumulators
- `src/move.jl` — particle motion with periodic BCs
- `src/simulation.jl` — top-level Simulation type, time-stepping
- `examples/relaxation.jl` — thermal-relaxation validation problem
- `test/runtests.jl` — unit tests for each component
