"""
    CompositeMesh{Meshes <: NamedTuple}

A thin container that bundles multiple meshes together under named slots,
giving the user a single object to pass around when an algorithm involves
more than one mesh.

The container is deliberately structural — it does not impose any semantic
relationship between the meshes. For algorithms that need a specific
relationship (e.g., a Lagrangian mesh paired with an Eulerian mesh and
their geometric overlap), see `PairedMesh`.

# Construction

```julia
composite = CompositeMesh((
    main = HierarchicalMesh{2}(...),
    refined_region = HierarchicalMesh{2}(...),
    coarse = HierarchicalMesh{2}(...),
))

# Access:
composite.main       # → HierarchicalMesh
composite.refined_region
```
"""
struct CompositeMesh{Meshes <: NamedTuple}
    meshes::Meshes
end

function Base.getproperty(c::CompositeMesh, name::Symbol)
    if name === :meshes
        return getfield(c, :meshes)
    end
    return getfield(getfield(c, :meshes), name)
end

function Base.propertynames(c::CompositeMesh)
    return (:meshes, propertynames(getfield(c, :meshes))...)
end

function Base.show(io::IO, c::CompositeMesh)
    names = propertynames(c.meshes)
    print(io, "CompositeMesh with submeshes: ", join(names, ", "))
end

# ============================================================================
# PairedMesh — Lagrangian + Eulerian with cached overlap
# ============================================================================

"""
    PairedMesh{LagMesh, EulMesh, OverlapType}

A paired Lagrangian + Eulerian mesh with a slot for a cached geometric
overlap between them. The overlap cache is invalidated whenever either
mesh changes; downstream code calls `ensure_overlap!` before any operation
that requires a current overlap.

The overlap implementation itself (computing the polytope intersections,
filling the cache) is provided by an external function and stored via
`set_overlap_compute_function!`. HierarchicalGrids does not implement the
geometric clipping; that's expected to be provided by an r3d.jl package or
similar.

# Construction

```julia
lag = SimplicialMesh{2, Float64}(...)
eul = HierarchicalMesh{2}(...)
paired = PairedMesh(lag, eul)

# Wire in an overlap implementation (typically once at program startup):
set_overlap_compute_function!(paired) do paired_mesh
    # Fill paired_mesh.overlap_cache[] = ...
    compute_overlap_with_r3d(paired_mesh.lagrangian, paired_mesh.eulerian)
end

# Then any operation that needs the overlap calls:
ensure_overlap!(paired)
ov = overlap_cache(paired)  # returns the cached overlap object
```

# Cache invalidation

These operations invalidate the overlap cache:

- `update_lagrangian_positions!(paired, new_positions)` — updates vertex
  positions of the Lagrangian mesh and invalidates the cache.
- `invalidate_overlap!(paired)` — explicit invalidation. Call after
  modifying the Eulerian mesh (refinement/coarsening) directly.

The user is responsible for calling `invalidate_overlap!` after any
modification not done through the PairedMesh API.
"""
mutable struct PairedMesh{LM, EM, OT}
    lagrangian::LM
    eulerian::EM
    overlap_cache::Base.RefValue{Union{Nothing, OT}}
    overlap_compute_fn::Base.RefValue{Union{Nothing, Function}}
end

function PairedMesh(lagrangian::LM, eulerian::EM;
                    overlap_type::Type{OT}=Any) where {LM, EM, OT}
    cache = Ref{Union{Nothing, OT}}(nothing)
    fn = Ref{Union{Nothing, Function}}(nothing)
    return PairedMesh{LM, EM, OT}(lagrangian, eulerian, cache, fn)
end

"""
    set_overlap_compute_function!(f, paired::PairedMesh)
    set_overlap_compute_function!(paired::PairedMesh, f)

Register the function that computes the geometric overlap between the
Lagrangian and Eulerian meshes. The function should take the PairedMesh
as its argument and return an overlap object (the cache will be populated
with this object on `ensure_overlap!`).

# Example

```julia
set_overlap_compute_function!(paired) do pm
    return MyR3DOverlap(pm.lagrangian, pm.eulerian)
end
```
"""
function set_overlap_compute_function!(f::Function, paired::PairedMesh)
    paired.overlap_compute_fn[] = f
    return paired
end
function set_overlap_compute_function!(paired::PairedMesh, f::Function)
    paired.overlap_compute_fn[] = f
    return paired
end

"""
    overlap_cache(paired::PairedMesh)

Return the currently cached overlap object, or `nothing` if it hasn't
been computed yet.
"""
@inline overlap_cache(paired::PairedMesh) = paired.overlap_cache[]

"""
    invalidate_overlap!(paired::PairedMesh)

Mark the overlap cache as stale. The next call to `ensure_overlap!` will
recompute it.
"""
function invalidate_overlap!(paired::PairedMesh)
    paired.overlap_cache[] = nothing
    return paired
end

"""
    ensure_overlap!(paired::PairedMesh)

If the overlap cache is empty, compute it using the registered function.
Returns the cached overlap object.

Throws an error if no compute function has been registered yet.
"""
function ensure_overlap!(paired::PairedMesh)
    if paired.overlap_cache[] === nothing
        fn = paired.overlap_compute_fn[]
        fn === nothing &&
            throw(ArgumentError("ensure_overlap!: no overlap compute function registered. " *
                                  "Call set_overlap_compute_function!(paired, fn) first " *
                                  "(or use install_r3d_overlap! to wire in the default backend)."))
        paired.overlap_cache[] = fn(paired)
    end
    return paired.overlap_cache[]
end

"""
    update_lagrangian_positions!(paired::PairedMesh, new_positions)

Update the Lagrangian mesh's vertex positions and invalidate the overlap
cache. `new_positions` should be an iterable of D-tuples (or convertible
to such).
"""
function update_lagrangian_positions!(paired::PairedMesh, new_positions)
    lag = paired.lagrangian
    length(new_positions) == lag.n_vertices ||
        throw(ArgumentError("expected $(lag.n_vertices) positions, got $(length(new_positions))"))
    @inbounds for i in eachindex(new_positions)
        set_vertex_position!(lag, i, new_positions[i])
    end
    invalidate_overlap!(paired)
    return paired
end

function Base.show(io::IO, pm::PairedMesh{LM, EM, OT}) where {LM, EM, OT}
    print(io, "PairedMesh{lagrangian=", LM, ", eulerian=", EM, "}")
    cached = pm.overlap_cache[] === nothing ? "no overlap cached" : "overlap cached"
    print(io, " (", cached, ")")
end
