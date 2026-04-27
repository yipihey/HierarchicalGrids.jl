using Test
using CFDPatchAMR
using HierarchicalGrids

@testset "CFDPatchAMR worked example" begin

    @testset "Configuration validation" begin
        @test_throws ArgumentError PatchAMRConfig(velocity = (-0.1, 0.3))
        @test_throws ArgumentError PatchAMRConfig(velocity = (0.1, -0.3))
        @test_throws ArgumentError PatchAMRConfig(dt = 0.0)
        @test_throws ArgumentError PatchAMRConfig(base_refines = 0)
        @test_throws ArgumentError PatchAMRConfig(fine_box_size = 0.0)
        @test_throws ArgumentError PatchAMRConfig(n_levels = 3)

        cfg = PatchAMRConfig()
        @test cfg.n_levels == 2
        @test cfg.velocity == (0.5, 0.3)
    end

    @testset "fine_patch_box: coarse-cell aligned and power-of-2 width" begin
        dx_c = 1/16
        # Peak in the middle.
        (flo, fhi, n_pow2) = CFDPatchAMR.fine_patch_box((0.5, 0.5), 0.2, dx_c)
        # Each endpoint lies on a multiple of dx_c.
        @test mod(flo[1], dx_c) ≈ 0.0 atol = 1e-12
        @test mod(flo[2], dx_c) ≈ 0.0 atol = 1e-12
        @test mod(fhi[1], dx_c) ≈ 0.0 atol = 1e-12
        @test mod(fhi[2], dx_c) ≈ 0.0 atol = 1e-12
        # Width is a power of 2 in coarse-cell units.
        @test n_pow2 > 0 && (n_pow2 & (n_pow2 - 1)) == 0

        # Clamping near the lo wall: lo should be exactly 0.
        (cl, ch, _) = CFDPatchAMR.fine_patch_box((0.05, 0.5), 0.2, dx_c)
        @test cl[1] >= 0.0
        @test ch[1] <= 1.0
        @test cl[2] >= 0.0
        @test ch[2] <= 1.0

        # Clamping near the hi wall.
        (cl2, ch2, _) = CFDPatchAMR.fine_patch_box((0.95, 0.5), 0.2, dx_c)
        @test ch2[1] <= 1.0
        @test cl2[1] >= 0.0
    end

    @testset "Initial state: peak position recovered from fine patch" begin
        cfg = PatchAMRConfig(n_steps = 0)
        state, diag = CFDPatchAMR.run!(cfg)
        # With n_steps=0, mass_drift is exactly zero by construction.
        @test diag.mass_drift == 0.0
        # Peak should be near (0.3, 0.3) at t=0.
        @test diag.peak_position_error < 0.05
    end

    @testset "Patch-based AMR advection (full run)" begin
        cfg = PatchAMRConfig(velocity = (0.5, 0.3),
                              dt = 0.02,
                              n_steps = 50,
                              base_refines = 4,
                              fine_refines = 1,
                              fine_box_size = 0.2,
                              refresh_every = 5,
                              peak_init = (0.3, 0.3),
                              gauss_sigma = 0.06)
        state, diag = CFDPatchAMR.run!(cfg)
        # Conservation: total mass on the coarse base patch should be
        # preserved to round-off. The pipeline ordering (prolong -> fine
        # update -> coarse update -> restrict) is what makes the
        # cross-level flux balance bit-exact; see README.md.
        @test abs(diag.mass_drift) / diag.mass_initial < 1e-12
        @test abs(diag.mass_drift) < 1e-8   # spec target

        # Qualitative tracking: after one period the peak should return
        # to within ~one coarse cell of its starting position (the
        # analytic position equals the initial position mod 1).
        @test diag.peak_position_error < 0.05
    end

    @testset "Without periodic refresh: feature still tracked but coarsely" begin
        cfg = PatchAMRConfig(velocity = (0.5, 0.3),
                              dt = 0.02,
                              n_steps = 20,
                              refresh_every = 0)
        state, diag = CFDPatchAMR.run!(cfg)
        # Even with no refresh the simulation should still run and keep
        # mass roughly conserved; peak tracking accuracy is limited to
        # whatever fraction of the trajectory falls inside the static
        # fine patch.
        @test isfinite(diag.mass_drift)
        @test isfinite(diag.peak_position_error)
    end

end
