# Free-streaming motion: x ← x + v * dt, then apply periodic boundary conditions.

"""
    move_particles!(particles, dt, L)

Free-stream all particles for a step of size dt, then wrap into [0, L)^3.

This is the only place where the spatial coordinate is updated; collisions
modify only velocity, sampling reads but doesn't modify either.
"""
function move_particles!(particles, dt::Float64, L::Float64)
    n = n_elements(particles)
    @inbounds for i in 1:n
        p = particles.pos[i]
        v = particles.vel[i]
        new_pos = (p[1] + v[1] * dt,
                   p[2] + v[2] * dt,
                   p[3] + v[3] * dt)
        # Wrap inline (cheaper than a separate apply_periodic_bcs! pass)
        x = new_pos[1] - L * floor(new_pos[1] / L)
        y = new_pos[2] - L * floor(new_pos[2] / L)
        z = new_pos[3] - L * floor(new_pos[3] / L)
        particles.pos[i] = (x, y, z)
    end
end
