"""
    FrameBoundaries{D}

Boundary spec attached to an `EulerianFrame{D, T}`. Travels alongside
the frame rather than being baked into the frame's type, so the
existing two-parameter `EulerianFrame{D, T}` signature is preserved.

# Fields

- `spec::BoundarySpec{D}` — per-axis-half BC kinds.

# Construction

```julia
FrameBoundaries(D::Integer)              # all REFLECTING (defaults)
FrameBoundaries(Val(D))                  # type-stable default
FrameBoundaries(spec::BoundarySpec{D})   # explicit
```

The constructor calls `BoundaryConditions.validate` to enforce that
periodicity is applied symmetrically on each axis.

# Accessors

- `bc(fb, axis, side)` — return the `BCKind` on the given axis-half.
- `is_periodic_axis(fb, axis)` — convenience for the common test.
"""
struct FrameBoundaries{D}
    spec::NTuple{D, NTuple{2, BCKind}}

    function FrameBoundaries{D}(spec::NTuple{D, NTuple{2, BCKind}}) where {D}
        validate(spec)
        return new{D}(spec)
    end
end

# Convenience constructors
FrameBoundaries(spec::NTuple{D, NTuple{2, BCKind}}) where {D} =
    FrameBoundaries{D}(spec)

FrameBoundaries(::Val{D}) where {D} =
    FrameBoundaries{D}(default_bc(Val(D)))

FrameBoundaries(D::Integer) = FrameBoundaries(Val(Int(D)))

"""
    bc(fb::FrameBoundaries, axis::Integer, side::Integer) -> BCKind

Return the boundary kind on `(axis, side)`, where `side == 1` is the
lo side and `side == 2` is the hi side.
"""
@inline bc(fb::FrameBoundaries, axis::Integer, side::Integer) =
    fb.spec[axis][side]

"""
    is_periodic_axis(fb::FrameBoundaries, axis::Integer) -> Bool

Whether `axis` is periodic in this `FrameBoundaries`.
"""
@inline is_periodic_axis(fb::FrameBoundaries{D}, axis::Integer) where {D} =
    is_periodic_axis(fb.spec, axis)

@inline spatial_dimension(::FrameBoundaries{D}) where {D} = D

function Base.show(io::IO, fb::FrameBoundaries{D}) where {D}
    print(io, "FrameBoundaries{", D, "}(")
    for d in 1:D
        d > 1 && print(io, ", ")
        lo, hi = fb.spec[d]
        print(io, lo, "/", hi)
    end
    print(io, ")")
end
