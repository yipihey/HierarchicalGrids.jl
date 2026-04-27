"""
    KernelContext

Per-task scratch container. Solver authors declare named scratch buffers and
their shapes once at solver-construction time; HG allocates one buffer per
task and threads them through the kernel via property access (`ctx.flux_buf`).

Each declared scratch is backed by a `OhMyThreads.TaskLocalValues.TaskLocalValue`
that lazily constructs a fresh buffer on first access in each task. Property
access resolves to the calling task's instance.

# Construction

```julia
ctx = KernelContext(; scratch = (
    flux_buf = (Float64, (5,)),       # Vector{Float64} of length 5 per task
    jac_buf  = (Float64, (5, 5)),     # Matrix{Float64} 5×5 per task
))
```

The `scratch` NamedTuple maps user-chosen names to `(eltype, dims)` tuples
describing the per-task buffer shape.

# Use inside a kernel

```julia
function my_kernel(cv, hv, ctx)
    flux = ctx.flux_buf       # ::Vector{Float64}, per-task
    jac  = ctx.jac_buf        # ::Matrix{Float64}, per-task
    # fill and use flux/jac
end
```

# Contract

- The kernel MUST NOT hold a reference to a scratch buffer beyond the
  `for_each_cell!` (or similar) call — another task may obtain a reference
  to the same buffer on the next call.
- Different scratch names live in different `TaskLocalValue`s and never
  alias each other within a task.
- Buffers are zero-initialized on first allocation in each task; reuse
  preserves whatever the kernel last wrote (kernels should fill before
  reading).
"""
struct KernelContext{Names, TLVs <: NamedTuple{Names}}
    tlvs::TLVs
end

# Build one TaskLocalValue per (eltype, dims) spec.
function _make_tlv(spec::Tuple)
    T, dims = spec
    A = Array{T, length(dims)}
    return TaskLocalValue{A}(() -> zeros(T, dims...))
end

function KernelContext(; scratch::NamedTuple = NamedTuple())
    Names = keys(scratch)
    tlvs_tuple = ntuple(i -> _make_tlv(scratch[i]), length(scratch))
    tlvs = NamedTuple{Names}(tlvs_tuple)
    return KernelContext{Names, typeof(tlvs)}(tlvs)
end

# Property access resolves the per-task buffer.
@inline function Base.getproperty(ctx::KernelContext, name::Symbol)
    if name === :tlvs
        return getfield(ctx, :tlvs)
    end
    tlv = getproperty(getfield(ctx, :tlvs), name)
    return tlv[]
end

Base.propertynames(::KernelContext{Names}) where {Names} = Names

function Base.show(io::IO, ::KernelContext{Names}) where {Names}
    print(io, "KernelContext(", join(Names, ", "), ")")
end
