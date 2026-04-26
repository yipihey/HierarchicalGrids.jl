using Test
using HierarchicalGrids
using HierarchicalGrids.Diagnostics
using Random

# ============================================================================
# WelfordStats — basic correctness
# ============================================================================

@testset "WelfordStats: empty state" begin
    s = WelfordStats()
    @test count_samples(s) == 0
    @test isnan(mean(s))
    @test isnan(variance(s))
    @test isnan(skewness(s))
    @test isnan(kurtosis(s))
end

@testset "WelfordStats: single sample" begin
    s = WelfordStats()
    push_value!(s, 3.5)
    @test count_samples(s) == 1
    @test mean(s) == 3.5
    # Variance with n=1 is undefined for the sample estimator
    @test isnan(variance(s))
    # Population variance from a single sample is 0
    @test variance(s; corrected=false) == 0.0
end

@testset "WelfordStats: matches direct mean/variance for known inputs" begin
    s = WelfordStats()
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    for x in xs
        push_value!(s, x)
    end
    @test count_samples(s) == 5
    @test mean(s) ≈ 3.0
    # sample variance Σ(x-mean)²/(n-1) = (4+1+0+1+4)/4 = 2.5
    @test variance(s) ≈ 2.5
    @test variance(s; corrected=false) ≈ 2.0
    @test std_dev(s) ≈ sqrt(2.5)
end

@testset "WelfordStats: matches Statistics for random inputs" begin
    Random.seed!(42)
    xs = randn(1000)
    s = WelfordStats()
    for x in xs
        push_value!(s, x)
    end
    # Direct calculation
    direct_mean = sum(xs) / length(xs)
    direct_var = sum((x - direct_mean)^2 for x in xs) / (length(xs) - 1)
    @test mean(s) ≈ direct_mean atol=1e-12
    @test variance(s) ≈ direct_var atol=1e-12
end

@testset "WelfordStats: skewness and kurtosis on a Gaussian sample" begin
    Random.seed!(123)
    s = WelfordStats()
    for _ in 1:50_000
        push_value!(s, randn())
    end
    @test mean(s) ≈ 0.0 atol=0.02
    @test variance(s) ≈ 1.0 atol=0.05
    # Gaussian: skewness ≈ 0, kurtosis ≈ 3, excess_kurtosis ≈ 0
    @test abs(skewness(s)) < 0.1
    @test kurtosis(s) ≈ 3.0 atol=0.1
    @test abs(excess_kurtosis(s)) < 0.1
end

@testset "WelfordStats: skewness on a known-skewed sample" begin
    # An exponential distribution has skewness = 2, excess kurtosis = 6
    # Sample size: large enough to get reasonable estimates
    Random.seed!(99)
    s = WelfordStats()
    for _ in 1:50_000
        # Inverse-CDF transform: -log(U) is Exp(1)
        u = rand()
        push_value!(s, -log(u))
    end
    @test mean(s) ≈ 1.0 atol=0.05
    @test variance(s) ≈ 1.0 atol=0.05
    @test skewness(s) ≈ 2.0 atol=0.2
    @test excess_kurtosis(s) ≈ 6.0 atol=1.5  # higher moments more sensitive
end

@testset "WelfordStats: integer inputs convert correctly" begin
    s = WelfordStats(Float64)
    for x in [1, 2, 3, 4, 5]
        push_value!(s, x)  # Int → Float64
    end
    @test mean(s) == 3.0
end

@testset "WelfordStats: type parameterization" begin
    s32 = WelfordStats(Float32)
    @test typeof(mean(s32)) === Float32  # NaN
    push_value!(s32, 1.0f0)
    push_value!(s32, 2.0f0)
    push_value!(s32, 3.0f0)
    @test mean(s32) ≈ 2.0f0
    @test variance(s32) ≈ 1.0f0
end

# ============================================================================
# WelfordStats — merge (parallel reduction)
# ============================================================================

@testset "merge_stats!: empty + non-empty" begin
    a = WelfordStats()
    b = WelfordStats()
    push_value!(b, 1.0); push_value!(b, 2.0); push_value!(b, 3.0)
    merge_stats!(a, b)
    @test count_samples(a) == 3
    @test mean(a) == 2.0
end

@testset "merge_stats!: non-empty + empty" begin
    a = WelfordStats()
    push_value!(a, 1.0); push_value!(a, 2.0)
    b = WelfordStats()
    merge_stats!(a, b)
    @test count_samples(a) == 2
    @test mean(a) == 1.5
end

@testset "merge_stats!: agrees with sequential push" begin
    Random.seed!(7)
    xs = randn(500)
    # Sequential
    seq = WelfordStats()
    for x in xs
        push_value!(seq, x)
    end
    # Split into chunks, merge
    chunks = [WelfordStats() for _ in 1:5]
    for (i, x) in enumerate(xs)
        push_value!(chunks[((i-1) % 5) + 1], x)
    end
    merged = WelfordStats()
    for c in chunks
        merge_stats!(merged, c)
    end
    @test count_samples(merged) == 500
    @test mean(merged) ≈ mean(seq) atol=1e-12
    @test variance(merged) ≈ variance(seq) atol=1e-12
    # Higher moments should also agree (Pébay 2008 formulas)
    @test skewness(merged) ≈ skewness(seq) atol=1e-10
    @test kurtosis(merged) ≈ kurtosis(seq) atol=1e-10
end

@testset "merge_stats!: returns destination for chaining" begin
    a = WelfordStats()
    push_value!(a, 1.0)
    b = WelfordStats()
    push_value!(b, 5.0)
    @test merge_stats!(a, b) === a
end

# ============================================================================
# ExponentialMovingAverage
# ============================================================================

@testset "ExponentialMovingAverage: construction and validation" begin
    # Valid α
    @test ExponentialMovingAverage(0.1) isa ExponentialMovingAverage
    @test ExponentialMovingAverage(1.0) isa ExponentialMovingAverage
    # Invalid α
    @test_throws ArgumentError ExponentialMovingAverage(0.0)
    @test_throws ArgumentError ExponentialMovingAverage(-0.1)
    @test_throws ArgumentError ExponentialMovingAverage(1.1)
end

@testset "ExponentialMovingAverage: uninitialized returns NaN" begin
    ema = ExponentialMovingAverage(0.3)
    @test isnan(value(ema))
end

@testset "ExponentialMovingAverage: first sample initializes" begin
    ema = ExponentialMovingAverage(0.5)
    update!(ema, 7.0)
    @test value(ema) == 7.0
end

@testset "ExponentialMovingAverage: blends subsequent samples" begin
    ema = ExponentialMovingAverage(0.5)
    update!(ema, 0.0)        # value = 0
    update!(ema, 10.0)       # value = 0.5*10 + 0.5*0 = 5
    @test value(ema) == 5.0
    update!(ema, 10.0)       # value = 0.5*10 + 0.5*5 = 7.5
    @test value(ema) == 7.5
end

@testset "ExponentialMovingAverage: converges to constant input" begin
    ema = ExponentialMovingAverage(0.1)
    for _ in 1:1000
        update!(ema, 42.0)
    end
    @test value(ema) ≈ 42.0 atol=1e-6
end

@testset "ExponentialMovingAverage: reset!" begin
    ema = ExponentialMovingAverage(0.3)
    update!(ema, 5.0)
    @test value(ema) == 5.0
    reset!(ema)
    @test isnan(value(ema))
    update!(ema, 9.0)
    @test value(ema) == 9.0  # behaves as fresh after reset
end

@testset "ExponentialMovingAverage: update! returns current value" begin
    ema = ExponentialMovingAverage(0.5)
    @test update!(ema, 4.0) == 4.0
    @test update!(ema, 8.0) == 6.0
end

# ============================================================================
# PerCellStats
# ============================================================================

@testset "PerCellStats: construction" begin
    p = PerCellStats(Float64, 5)
    @test length(p) == 5
    for i in 1:5
        @test count_samples(p[i]) == 0
    end
end

@testset "PerCellStats: independent per-cell accumulation" begin
    p = PerCellStats(Float64, 3)
    push_value!(p, 1, 1.0)
    push_value!(p, 1, 2.0)
    push_value!(p, 1, 3.0)
    push_value!(p, 2, 10.0)
    push_value!(p, 2, 20.0)
    # Cell 3 gets nothing

    @test count_samples(p[1]) == 3
    @test mean(p[1]) == 2.0
    @test count_samples(p[2]) == 2
    @test mean(p[2]) == 15.0
    @test count_samples(p[3]) == 0
    @test isnan(mean(p[3]))
end

@testset "PerCellStats: reset! clears all cells" begin
    p = PerCellStats(Float64, 4)
    for i in 1:4, _ in 1:10
        push_value!(p, i, randn())
    end
    @test all(count_samples(p[i]) == 10 for i in 1:4)
    Diagnostics.reset!(p)
    @test all(count_samples(p[i]) == 0 for i in 1:4)
end

# ============================================================================
# Show methods
# ============================================================================

@testset "show methods" begin
    s = WelfordStats()
    @test sprint(show, s) == "WelfordStats{Float64}(n=0)"
    push_value!(s, 1.0)
    str = sprint(show, s)
    @test occursin("n=1", str)
    @test occursin("mean=1.0", str)
    push_value!(s, 3.0)
    str2 = sprint(show, s)
    @test occursin("var=", str2)

    ema_uninit = ExponentialMovingAverage(0.2)
    @test occursin("uninitialized", sprint(show, ema_uninit))
    update!(ema_uninit, 5.0)
    @test occursin("value=", sprint(show, ema_uninit))
end
