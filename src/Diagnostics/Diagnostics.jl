"""
    Diagnostics

Lightweight running-statistics utilities for AMR/CFD self-consistency
monitoring. Provides Welford-style accumulators for mean/variance/skewness/
kurtosis, exponential moving averages, and per-cell statistic accumulators.

These are general-purpose: any application that wants to track distributional
properties of a residual or signal at runtime — without buffering all
samples — can use them.

# WelfordStats

Single-pass online accumulator using Welford's algorithm and its higher-order
generalizations (Pébay 2008). Numerically stable and merge-friendly.

```julia
s = WelfordStats(Float64)
for x in samples
    push_value!(s, x)
end
println("mean = ", mean(s), ", var = ", variance(s))
println("skew = ", skewness(s), ", kurt = ", kurtosis(s))
```

# ExponentialMovingAverage

Time-series-friendly EMA with configurable smoothing factor.

```julia
ema = ExponentialMovingAverage(0.1)  # α = 0.1
for x in time_series
    update!(ema, x)
end
println("EMA = ", value(ema))
```
"""
module Diagnostics

export WelfordStats, push_value!, mean, variance, std_dev
export skewness, kurtosis, excess_kurtosis, count_samples, merge_stats!
export ExponentialMovingAverage, update!, value, reset!
export PerCellStats
export RemapDiagnostics
export OverlapAuditReport, audit_overlap

# ============================================================================
# WelfordStats — online mean, variance, skewness, kurtosis
# ============================================================================

"""
    WelfordStats{T}()
    WelfordStats(::Type{T}) where T

Online accumulator for mean, variance, skewness, and kurtosis using
Welford's algorithm with the higher-moment extensions of Pébay (2008).
Numerically stable; correct for any number of samples (including 0 and 1).

# Fields (read-only via getters)

- `n` — number of samples accumulated.
- `M1` — running mean.
- `M2` — Σ (x_i - mean)² (used for variance via M2/(n-1) for sample, or M2/n for population).
- `M3, M4` — higher central moment accumulators.
"""
mutable struct WelfordStats{T}
    n::Int
    M1::T
    M2::T
    M3::T
    M4::T
end

WelfordStats{T}() where T = WelfordStats{T}(0, zero(T), zero(T), zero(T), zero(T))
WelfordStats(::Type{T}) where T = WelfordStats{T}()
WelfordStats() = WelfordStats{Float64}()

"""
    push_value!(s::WelfordStats, x)

Add a single sample to the accumulator. Updates mean, variance, skewness,
kurtosis online.
"""
function push_value!(s::WelfordStats{T}, x) where T
    xT = T(x)
    n1 = s.n
    n = n1 + 1
    delta = xT - s.M1
    delta_n = delta / T(n)
    delta_n2 = delta_n * delta_n
    term1 = delta * delta_n * T(n1)

    s.M1 += delta_n
    s.M4 += term1 * delta_n2 * T(n*n - 3*n + 3) +
            T(6) * delta_n2 * s.M2 -
            T(4) * delta_n * s.M3
    s.M3 += term1 * delta_n * T(n - 2) - T(3) * delta_n * s.M2
    s.M2 += term1
    s.n = n
    return s
end

"""
    count_samples(s::WelfordStats) :: Int

Number of samples accumulated.
"""
@inline count_samples(s::WelfordStats) = s.n

"""
    mean(s::WelfordStats) :: T

Sample mean. Returns NaN-equivalent if n = 0.
"""
@inline function mean(s::WelfordStats{T}) where T
    s.n == 0 && return T(NaN)
    return s.M1
end

"""
    variance(s::WelfordStats; corrected=true) :: T

Variance. With `corrected=true` (default), returns the sample variance
M2/(n-1); with `corrected=false`, the population variance M2/n.
"""
function variance(s::WelfordStats{T}; corrected::Bool=true) where T
    if corrected
        s.n < 2 && return T(NaN)
        return s.M2 / T(s.n - 1)
    else
        s.n == 0 && return T(NaN)
        return s.M2 / T(s.n)
    end
end

"""
    std_dev(s::WelfordStats; corrected=true) :: T

Standard deviation, sqrt(variance(...)).
"""
@inline std_dev(s::WelfordStats; corrected::Bool=true) = sqrt(variance(s; corrected=corrected))

"""
    skewness(s::WelfordStats) :: T

Sample skewness (Fisher's g_1) using the standard biased estimator
sqrt(n) * M3 / M2^(3/2). Returns NaN if n < 2 or M2 = 0.
"""
function skewness(s::WelfordStats{T}) where T
    s.n < 2 && return T(NaN)
    s.M2 == 0 && return T(NaN)
    return sqrt(T(s.n)) * s.M3 / s.M2^(T(3)/T(2))
end

"""
    kurtosis(s::WelfordStats) :: T

Raw kurtosis = M4 * n / M2² (= 3 for a Gaussian).
"""
function kurtosis(s::WelfordStats{T}) where T
    s.n < 2 && return T(NaN)
    s.M2 == 0 && return T(NaN)
    return T(s.n) * s.M4 / (s.M2 * s.M2)
end

"""
    excess_kurtosis(s::WelfordStats) :: T

Excess kurtosis = kurtosis - 3. A Gaussian has 0; positive values mean
heavier tails than Gaussian, negative means lighter.
"""
@inline excess_kurtosis(s::WelfordStats) = kurtosis(s) - 3

"""
    merge_stats!(dest::WelfordStats, src::WelfordStats)

Merge `src` into `dest` (parallel reduction). Implements Pébay 2008's
parallel-merge formulas for moments up to order 4. Returns `dest`.
"""
function merge_stats!(a::WelfordStats{T}, b::WelfordStats{T}) where T
    if b.n == 0
        return a
    end
    if a.n == 0
        a.n, a.M1, a.M2, a.M3, a.M4 = b.n, b.M1, b.M2, b.M3, b.M4
        return a
    end
    n_a = T(a.n); n_b = T(b.n)
    n_total = a.n + b.n
    n_total_T = T(n_total)
    delta = b.M1 - a.M1
    delta2 = delta * delta
    delta3 = delta2 * delta
    delta4 = delta2 * delta2

    # New mean
    new_M1 = a.M1 + delta * (n_b / n_total_T)

    # New M2
    new_M2 = a.M2 + b.M2 + delta2 * (n_a * n_b / n_total_T)

    # New M3
    new_M3 = a.M3 + b.M3 +
             delta3 * (n_a * n_b * (n_a - n_b) / (n_total_T * n_total_T)) +
             T(3) * delta * (n_a * b.M2 - n_b * a.M2) / n_total_T

    # New M4
    new_M4 = a.M4 + b.M4 +
             delta4 * (n_a * n_b * (n_a*n_a - n_a*n_b + n_b*n_b)) /
                     (n_total_T * n_total_T * n_total_T) +
             T(6) * delta2 * (n_a*n_a * b.M2 + n_b*n_b * a.M2) /
                            (n_total_T * n_total_T) +
             T(4) * delta * (n_a * b.M3 - n_b * a.M3) / n_total_T

    a.n = n_total
    a.M1 = new_M1
    a.M2 = new_M2
    a.M3 = new_M3
    a.M4 = new_M4
    return a
end

function Base.show(io::IO, s::WelfordStats{T}) where T
    print(io, "WelfordStats{", T, "}(n=", s.n)
    if s.n > 0
        print(io, ", mean=", mean(s))
        if s.n > 1
            print(io, ", var=", variance(s))
        end
    end
    print(io, ")")
end

# ============================================================================
# ExponentialMovingAverage
# ============================================================================

"""
    ExponentialMovingAverage(α::Real; T=Float64)

Exponentially-weighted moving average:

    EMA_t = α * x_t + (1 - α) * EMA_{t-1}

Smaller α gives slower response but more smoothing. The first sample
initializes the EMA; subsequent samples blend.
"""
mutable struct ExponentialMovingAverage{T}
    α::T
    value::T
    initialized::Bool
end

function ExponentialMovingAverage(α::Real; T::Type=Float64)
    0 < α <= 1 || throw(ArgumentError("α must be in (0, 1], got $α"))
    return ExponentialMovingAverage{T}(T(α), zero(T), false)
end

"""
    update!(ema::ExponentialMovingAverage, x) :: T

Add a sample, returning the current EMA value.
"""
function update!(ema::ExponentialMovingAverage{T}, x) where T
    xT = T(x)
    if !ema.initialized
        ema.value = xT
        ema.initialized = true
    else
        ema.value = ema.α * xT + (one(T) - ema.α) * ema.value
    end
    return ema.value
end

"""
    value(ema::ExponentialMovingAverage) :: T

Current EMA value. Returns NaN if no samples have been pushed yet.
"""
@inline function value(ema::ExponentialMovingAverage{T}) where T
    return ema.initialized ? ema.value : T(NaN)
end

"""
    reset!(ema::ExponentialMovingAverage)

Clear the accumulated value.
"""
function reset!(ema::ExponentialMovingAverage{T}) where T
    ema.value = zero(T)
    ema.initialized = false
    return ema
end

function Base.show(io::IO, ema::ExponentialMovingAverage{T}) where T
    print(io, "ExponentialMovingAverage{", T, "}(α=", ema.α, ", ")
    if ema.initialized
        print(io, "value=", ema.value)
    else
        print(io, "uninitialized")
    end
    print(io, ")")
end

# ============================================================================
# PerCellStats — array of per-cell accumulators
# ============================================================================

"""
    PerCellStats{T}(n::Integer)

A vector of `n` `WelfordStats{T}` accumulators, one per cell. Useful for
tracking distributional statistics independently per cell over many time
steps (e.g., variance of a residual within each cell over a calibration
window).
"""
struct PerCellStats{T}
    stats::Vector{WelfordStats{T}}
end

PerCellStats{T}(n::Integer) where T = PerCellStats{T}([WelfordStats{T}() for _ in 1:n])
PerCellStats(::Type{T}, n::Integer) where T = PerCellStats{T}(n)

@inline Base.length(p::PerCellStats) = length(p.stats)
@inline Base.getindex(p::PerCellStats, i::Integer) = p.stats[i]

"""
    push_value!(p::PerCellStats, i::Integer, x)

Add a sample to cell i's accumulator.
"""
@inline function push_value!(p::PerCellStats, i::Integer, x)
    push_value!(p.stats[i], x)
    return p
end

"""
    reset!(p::PerCellStats)

Clear all cells' accumulators.
"""
function reset!(p::PerCellStats{T}) where T
    for i in eachindex(p.stats)
        p.stats[i] = WelfordStats{T}()
    end
    return p
end

# ============================================================================
# RemapDiagnostics — shell-crossing surveillance for polynomial remap kernels
# ============================================================================

"""
    RemapDiagnostics{T<:AbstractFloat}()
    RemapDiagnostics(::Type{T}) where T

Mutable accumulator for per-cell-pair diagnostics tracked during a
polynomial remap pass:

- `liouville_min`, `liouville_max` — per-pair Jacobian (local stretch
  factor) extrema. The proxy used is
  `entry.volume / (source_reference_volume * |det J_src|)` =
  `entry.volume / source_physical_volume`, which is 1 for any
  source-cell pair fully contained in a single target cell under an
  identity remap and < 1 for sub-cell overlaps. The minimum over all
  pairs flags candidate shell-crossings; the maximum surfaces unusually
  large stretch.
- `total_volume_in`, `total_volume_out` — accumulated overlap volumes
  on the source and target sides. Each overlap entry contributes
  `entry.volume` to both (the overlap polytope is symmetric in source
  and target), so the running totals are identical and their equality
  provides a sanity check on the kernel.
- `n_negative_jacobian_cells` — count of overlap entries for which
  the per-pair Jacobian proxy is `≤ 0`. Strictly positive overlap
  volumes are an invariant of `compute_overlap`, so a non-zero count
  signals corruption (e.g. an inverted Lagrangian simplex propagated
  past the builder's checks).

The struct is designed to be allocated once and reused across many
remap passes — call [`reset!`](@ref) between passes — or to be merged
across thread-local copies via `Base.merge!`.
"""
mutable struct RemapDiagnostics{T<:AbstractFloat}
    liouville_min::T
    liouville_max::T
    total_volume_in::T
    total_volume_out::T
    n_negative_jacobian_cells::Int
end

RemapDiagnostics{T}() where {T<:AbstractFloat} =
    RemapDiagnostics{T}(typemax(T), typemin(T), zero(T), zero(T), 0)
RemapDiagnostics(::Type{T}) where {T<:AbstractFloat} = RemapDiagnostics{T}()
RemapDiagnostics() = RemapDiagnostics{Float64}()

"""
    reset!(d::RemapDiagnostics)

Restore `d` to its freshly-constructed state. Returns `d`.
"""
function reset!(d::RemapDiagnostics{T}) where {T}
    d.liouville_min = typemax(T)
    d.liouville_max = typemin(T)
    d.total_volume_in = zero(T)
    d.total_volume_out = zero(T)
    d.n_negative_jacobian_cells = 0
    return d
end

"""
    Base.merge!(a::RemapDiagnostics{T}, b::RemapDiagnostics{T}) -> a

Merge `b` into `a` (parallel reduction). Takes elementwise
`min`/`max`/sum as appropriate. Returns `a`.
"""
function Base.merge!(a::RemapDiagnostics{T}, b::RemapDiagnostics{T}) where {T}
    a.liouville_min = min(a.liouville_min, b.liouville_min)
    a.liouville_max = max(a.liouville_max, b.liouville_max)
    a.total_volume_in += b.total_volume_in
    a.total_volume_out += b.total_volume_out
    a.n_negative_jacobian_cells += b.n_negative_jacobian_cells
    return a
end

function Base.show(io::IO, d::RemapDiagnostics{T}) where {T}
    print(io, "RemapDiagnostics{", T, "}(")
    print(io, "liouville_min=", d.liouville_min,
              ", liouville_max=", d.liouville_max,
              ", total_volume_in=", d.total_volume_in,
              ", total_volume_out=", d.total_volume_out,
              ", n_negative_jacobian_cells=", d.n_negative_jacobian_cells, ")")
end

# ============================================================================
# IntExact audit harness (spans the float and integer-exact backends)
# ============================================================================
#
# Lives in Diagnostics for organizational reasons (a diagnostic harness)
# but reaches across to the Overlap submodule at call time. The module-
# load consistency check is invoked from `__init__` below, which runs
# after the entire HierarchicalGrids package has finished loading, so
# Overlap's symbols are resolvable.
include("exact_audit.jl")

# Run the IntExact consistency check once at package load, mirroring
# `_verify_moment_ordering` in `src/Overlap/r3d_adapter.jl`. Gated by
# the `HG_INTEXACT_VERIFY` environment variable so it can be disabled
# in latency-sensitive contexts; default behavior is "always run".
function __init__()
    _verify_intexact_consistency()
    return nothing
end

end # module Diagnostics
