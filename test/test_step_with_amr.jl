using Test
using HierarchicalGrids
using HierarchicalGrids: RefinementEvent, register_refinement_listener!,
    unregister_refinement_listener!

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mutable opaque user state that the driver is supposed to leave alone
# beyond passing it through to `step!`.
mutable struct StepState
    t::Float64
    dt::Float64
    n_step_calls::Int
    n_amr_fires::Int
    center::Float64    # for the moving-Gaussian indicator test
end
StepState(; t=0.0, dt=0.1, center=0.5) = StepState(t, dt, 0, 0, center)

# Build a 1D mesh + frame on the unit interval.
make_1d_frame() = EulerianFrame(HierarchicalMesh{1}(), (0.0,), (1.0,))

# Cell-center physical x-coordinate via the frame.
function cell_center_x(frame, i)
    lo, hi = cell_physical_box(frame, i)
    return 0.5 * (lo[1] + hi[1])
end

# ---------------------------------------------------------------------------
# 1. Smoke test: physics ran 9 times, AMR fired 3 times, state mutated.
# ---------------------------------------------------------------------------
@testset "smoke: step!/AMR cadence on flat indicator" begin
    frame = make_1d_frame()
    state = StepState(dt = 0.25)

    function physics!(s::StepState, _frame)
        s.t += s.dt
        s.n_step_calls += 1
        return s
    end

    # Indicator that fires AMR every time it's invoked: all leaves above
    # threshold. Use length-n_cells(mesh) form for safety.
    function indicator(mesh)
        s = state              # close over outer state
        s.n_amr_fires += 1     # count how many AMR cycles asked us
        return fill(1.0, n_cells(mesh))
    end

    n = step_with_amr!(state, frame, physics!, indicator, 9;
                       refine_threshold = 0.5,
                       hysteresis_steps = 3)

    @test n == 9
    @test state.n_step_calls == 9
    @test state.n_amr_fires == 3      # AMR cycles fire at k = 3, 6, 9
    @test state.t ≈ 9 * 0.25 atol=1e-12
    # Indicator was 1.0 everywhere → mesh should have grown.
    @test n_cells(frame.mesh) > 1
end

# ---------------------------------------------------------------------------
# 2. No-AMR case: zero indicator means nothing ever refines.
# ---------------------------------------------------------------------------
@testset "no refinement when indicator is zero" begin
    frame = make_1d_frame()
    n0 = n_cells(frame.mesh)
    state = StepState()

    physics!(s::StepState, _frame) = (s.n_step_calls += 1; s)
    # All-zero indicator: no leaves above refine_threshold and nothing to
    # coarsen on a single-root mesh.
    indicator(mesh) = fill(0.0, n_cells(mesh))

    step_with_amr!(state, frame, physics!, indicator, 12;
                   refine_threshold = 0.5,
                   coarsen_threshold = 0.1,
                   hysteresis_steps = 3)

    @test state.n_step_calls == 12
    @test n_cells(frame.mesh) == n0          # nothing refined, nothing coarsened
end

# ---------------------------------------------------------------------------
# 3. Tracking refinement: leaf count is denser near a Gaussian peak than
# at the periphery after running the driver for a while.
# ---------------------------------------------------------------------------
@testset "moving-Gaussian indicator concentrates refinement" begin
    frame = make_1d_frame()
    # Pre-refine a couple of times so we have many leaves to work with.
    refine_cells!(frame.mesh, [1])
    # After two passes of root refinement, refine all leaves once more so
    # we start with ~4 leaves of physical width 0.25.
    leaves = [i for i in 1:n_cells(frame.mesh) if is_leaf(frame.mesh.cells[i])]
    refine_cells!(frame.mesh, leaves)

    state = StepState(dt = 0.0, center = 0.5)

    physics!(s::StepState, _frame) = (s.n_step_calls += 1; s)

    # Indicator: Gaussian centered at state.center, narrow width.
    sigma = 0.05
    function indicator(mesh)
        ind = zeros(Float64, n_cells(mesh))
        for i in 1:n_cells(mesh)
            if is_leaf(mesh.cells[i])
                xc = cell_center_x(frame, i)
                ind[i] = exp(-((xc - state.center)^2) / (2 * sigma^2))
            end
        end
        return ind
    end

    step_with_amr!(state, frame, physics!, indicator, 30;
                   refine_threshold = 0.5,
                   coarsen_threshold = 0.05,
                   hysteresis_steps = 1,
                   max_level = 6)

    # Bin leaves by physical x-coordinate. Count leaves in [0.4, 0.6]
    # vs. the periphery (e.g. [0.0, 0.2] or [0.8, 1.0]).
    near_peak = 0
    far_from_peak = 0
    for i in 1:n_cells(frame.mesh)
        if is_leaf(frame.mesh.cells[i])
            xc = cell_center_x(frame, i)
            if 0.4 <= xc <= 0.6
                near_peak += 1
            elseif xc <= 0.2 || xc >= 0.8
                far_from_peak += 1
            end
        end
    end
    @test near_peak > far_from_peak
end

# ---------------------------------------------------------------------------
# 4. Listener integration: a refinement listener counter equals the
# number of AMR cycles that actually changed topology.
# ---------------------------------------------------------------------------
@testset "refinement listener fires once per AMR cycle that mutates" begin
    frame = make_1d_frame()
    state = StepState()

    fire_count = Ref(0)
    handle = register_refinement_listener!(frame.mesh, _evt -> (fire_count[] += 1))

    physics!(s::StepState, _frame) = (s.n_step_calls += 1; s)
    # Always-refine indicator → every AMR cycle mutates, so listener fires.
    indicator(mesh) = fill(1.0, n_cells(mesh))

    step_with_amr!(state, frame, physics!, indicator, 6;
                   refine_threshold = 0.5,
                   hysteresis_steps = 2,
                   max_level = 8)

    # AMR cycles fire at k = 2, 4, 6 → 3 cycles, each mutating → 3 events.
    @test fire_count[] == 3
    unregister_refinement_listener!(frame.mesh, handle)
end

# ---------------------------------------------------------------------------
# 5. max_level cap: even with strong refine indicator, no leaf exceeds
# the configured max_level.
# ---------------------------------------------------------------------------
@testset "max_level=1 prevents deeper refinement" begin
    frame = make_1d_frame()
    state = StepState()

    physics!(s::StepState, _frame) = (s.n_step_calls += 1; s)
    indicator(mesh) = fill(100.0, n_cells(mesh))

    step_with_amr!(state, frame, physics!, indicator, 12;
                   refine_threshold = 0.5,
                   hysteresis_steps = 1,
                   max_level = 1)

    # No cell is at level > 1.
    for i in 1:n_cells(frame.mesh)
        @test level_of(frame.mesh, i) <= 1
    end
end
