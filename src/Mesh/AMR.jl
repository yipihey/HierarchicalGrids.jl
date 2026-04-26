"""
    AMR

Topology-only AMR driver. Provides [`step_with_amr!`](@ref), a small
interleaver that calls a user `step!` callback and periodically invokes
[`refine_by_indicator!`](@ref) on a hysteresis schedule.

The driver is intentionally minimal — it does not own per-cell state, it
does not compute CFL/dt, and it does not re-implement refinement. User
state is kept consistent across refinement via the PR-B refinement-listener
mechanism: callers register listeners on the mesh that observe each
[`RefinementEvent`](@ref) and update their data accordingly.
"""
module AMR

using ..Mesh: HierarchicalMesh, refine_by_indicator!
using ..Overlap: EulerianFrame

export step_with_amr!

"""
    step_with_amr!(state, frame::EulerianFrame{D,T}, step!, indicator,
                    n_steps::Int;
                    refine_threshold::Real,
                    coarsen_threshold::Real = refine_threshold / 4,
                    hysteresis_steps::Int = 3,
                    max_level::Integer = typemax(Int),
                    isotropic::Bool = true)::Int where {D, T}

Topology-only AMR driver. Calls `step!(state, frame)` `n_steps` times;
every `hysteresis_steps` calls invokes `refine_by_indicator!(frame.mesh,
indicator(frame.mesh); refine_threshold, coarsen_threshold, max_level,
isotropic)`. The PR-B refinement-listener mechanism keeps any per-cell
`state` consistent if `state` registered itself on `frame.mesh`.

Returns the number of physics steps taken (always equal to `n_steps`).

This driver is **topology-only** — CFL/dt is the caller's responsibility.

# Arguments
- `state`: opaque user data (e.g. `PolynomialFieldSet`, dfmm scratch,
  particle list). Not inspected by the driver beyond passing it to `step!`.
- `frame`: `EulerianFrame` whose `mesh` is the target of refinement.
- `step!(state, frame)`: user-supplied callback. Mutates `state` (and
  optionally the frame's mesh) in place.
- `indicator(mesh)`: user-supplied callback. Takes the mesh and returns
  any object that [`refine_by_indicator!`](@ref) accepts — either an
  iterable of length `n_cells(mesh)` or a callable `f(cell_index)`.
- `n_steps`: number of `step!` calls to perform.
- `refine_threshold`: refine any leaf with indicator value above this.
- `coarsen_threshold`: coarsen sibling groups all below this threshold.
  Defaults to `refine_threshold / 4` for hysteresis.
- `hysteresis_steps`: number of `step!` calls between AMR cycles.
- `max_level`: refinement-depth ceiling.
- `isotropic`: if `true` (default), all `D` axes are refined together.

# Termination
The driver does NOT terminate based on time or convergence — it runs
exactly `n_steps` times. Wrap with the user's own outer loop for time
integration.

# Notes
- AMR fires after each `hysteresis_steps`-th `step!` call (i.e. on
  iterations `k` where `k % hysteresis_steps == 0`).
- If `hysteresis_steps <= 0`, no AMR cycles fire (useful for benchmarking
  the no-AMR baseline without changing call sites).
"""
function step_with_amr!(state, frame::EulerianFrame{D, T},
                         step_fn::F, indicator::G,
                         n_steps::Integer;
                         refine_threshold::Real,
                         coarsen_threshold::Real = refine_threshold / 4,
                         hysteresis_steps::Integer = 3,
                         max_level::Integer = typemax(Int),
                         isotropic::Bool = true)::Int where {D, T, F, G}
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative, got $n_steps"))
    for k in 1:Int(n_steps)
        step_fn(state, frame)
        if hysteresis_steps > 0 && (k % Int(hysteresis_steps) == 0)
            ind = indicator(frame.mesh)
            refine_by_indicator!(frame.mesh, ind;
                                  refine_threshold = refine_threshold,
                                  coarsen_threshold = coarsen_threshold,
                                  max_level = max_level,
                                  isotropic = isotropic)
        end
    end
    return Int(n_steps)
end

end # module AMR
