# Smoke test for `examples/cfd_patch_amr/`: the worked Berger-Oliger
# patch-AMR demo. We don't pull the example in as a package dep — that
# would force the package's Manifest into the parent. Instead we load
# its source files directly from the example tree and run a single
# short configuration end-to-end. Mass conservation and peak tracking
# are checked at the same thresholds used by the example's own
# `Pkg.test()` suite.

using Test
using HierarchicalGrids

const _CFD_DIR = abspath(joinpath(@__DIR__, "..", "examples", "cfd_patch_amr"))

# Load the example's source into an isolated module so the symbol
# names don't leak.
module _CFDPatchAMRSmoke
    using HierarchicalGrids
    include(joinpath(@__DIR__, "..", "examples", "cfd_patch_amr",
                     "src", "CFDPatchAMR.jl"))
    using .CFDPatchAMR
end

const CFDPatchAMR = _CFDPatchAMRSmoke.CFDPatchAMR

@testset "CFDPatchAMR smoke (worked PR-16 example)" begin
    @test isdir(_CFD_DIR)
    @test isfile(joinpath(_CFD_DIR, "src", "CFDPatchAMR.jl"))
    @test isfile(joinpath(_CFD_DIR, "test", "runtests.jl"))
    @test isfile(joinpath(_CFD_DIR, "Project.toml"))
    @test isfile(joinpath(_CFD_DIR, "README.md"))

    # Short integration run: one tenth of a period, single fine-patch
    # refresh in the middle. Aim is to catch regressions in the AMR
    # pipeline, not to validate accuracy.
    cfg = CFDPatchAMR.PatchAMRConfig(velocity = (0.5, 0.3),
                                       dt = 0.02,
                                       n_steps = 10,
                                       base_refines = 4,
                                       fine_refines = 1,
                                       fine_box_size = 0.2,
                                       refresh_every = 5,
                                       peak_init = (0.3, 0.3),
                                       gauss_sigma = 0.06)
    state, diag = CFDPatchAMR.run!(cfg)

    # Mass conservation at the spec's 1e-8 target (we measure ~1e-15).
    @test abs(diag.mass_drift) < 1e-8

    # Peak should still be inside the fine patch and within ~one coarse
    # cell of the analytic position.
    @test diag.peak_position_error < 0.1
    @test isfinite(diag.peak_position[1])
    @test isfinite(diag.peak_position[2])
end
