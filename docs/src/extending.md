# Extending HierarchicalGrids

This document is for people building things on top of the framework: new physics, new layouts, new mesh types, new geometric operations.

## The composition principle

The cardinal rule: **add new types, don't extend existing ones.**

If you want a hydrodynamics solver, you don't add hydro fields to `HierarchicalMesh`. You create a `HydroState` struct that holds the hydro fields and references the mesh. The mesh stays a clean structural container; your solver is a separable module that uses the mesh.

Why? Because in 5 years someone else will want to add MHD on the same mesh, and they shouldn't have to touch your hydro code. And in 10 years, someone will want to use the framework for radiative transfer, and they shouldn't have to touch your hydro or your colleague's MHD. The mesh is the shared substrate; everything else is composed on top.

## Adding a physics module

Pattern:

```julia
module MyHydro

using HierarchicalGrids

# Holds your physics state
struct HydroState{L<:AbstractLayout, M<:HierarchicalMesh}
    mesh::M
    fields::FieldSet{L, ...}      # density, momentum, energy, ...
    # any solver-specific persistent state
end

function HydroState(mesh::HierarchicalMesh; layout::AbstractLayout = SoA())
    fields = allocate_fields(layout, n_cells(mesh);
                             density = Float64,
                             momentum = NTuple{3, Float64},
                             energy = Float64)
    return HydroState(mesh, fields)
end

function step!(state::HydroState, dt)
    # ... your timestep
end

# Adapt to refinement: when the mesh gains/loses cells, your fields need to follow
function on_mesh_changed!(state::HydroState, old_to_new::Vector{UInt32})
    # old_to_new: maps old cell indices to new ones (0 = removed)
    # ... migrate your field data
end

end
```

Key points:

- Your `HydroState` *contains* the mesh, but doesn't *modify* it. Refinement decisions are typically made by the solver, but the actual `refine_cells!` call goes through the mesh's API.
- When the mesh changes, your fields need to follow. Register a refinement listener on the mesh and the framework will dispatch a `RefinementEvent` describing the batch (refined parents, new children, removed cells, and an `index_remap` array) — see [Refinement events](refinement_events.md).
- Solver-specific state lives in your struct. Don't pollute the mesh with it.

## Adding a new layout

Covered in detail in `layouts.md`. The procedure:

1. Define your layout type (e.g., `struct AoSoA{W} <: AbstractLayout end`).
2. Implement `_make_storage(::Type{YourLayout}, n, names, types)`.
3. Implement `_get_field(storage, ::Type{YourLayout}, ::Val{name}, i)` and `_set_field!`.
4. Implement `_resize_storage!(storage, ::Type{YourLayout}, new_n)`.

Test by checking that the same kernel produces the same numerical results as SoA.

## Positivity certificates for polynomial fields

A common need for limiter detection in higher-order solvers is
"is this polynomial coefficient column positive everywhere on its
reference cell?" The Bernstein basis answers this cheaply via the
convex-hull property: a sufficient condition is that every Bernstein
coefficient is strictly positive.

Two primitives implement this check:

- `bernstein_positivity_certificate(coeffs, basis; atol = 0)` —
  low-level certificate over a single coefficient vector. Returns
  `(true, nothing)` on success, `(false, multi_index)` on the first
  offending coefficient.
- `is_strictly_positive(field::PolynomialFieldView; atol = 0)` (and
  the `(pfs, name)` convenience overload) — sweeps every cell of a
  Bernstein-basis `PolynomialFieldSet` and returns the first failure
  as `(false, (cell_index, multi_index))`.

```julia
field = allocate_polynomial_fields(SoA(), BernsteinBasis{2, 1}(),
                                    n_cells(mesh); rho = Float64)
init_field_from!(field, frame, x -> 1.0 + 0.1 * x[1])

ok, where = is_strictly_positive(field, :rho)
ok || @warn "non-positive coefficient at cell $(where[1])"
```

A `false` result does NOT imply that the polynomial is actually
non-positive: the Bernstein test is sufficient but not necessary, and
gets sharper under degree elevation. `is_strictly_positive` raises
`ArgumentError` if the field's basis is not a `BernsteinBasis` (the
convex-hull property is what makes the check sound).

## Adding a new geometric operation

The Geometry layer provides exact-integer predicates and volume operations. To add new operations:

```julia
module MyGeometryExtension

using HierarchicalGrids
using HierarchicalGrids.BitPrimitives  # if you need bit ops
using HierarchicalGrids.Geometry        # to compose with existing ops

# Example: cell containment of a point
function cell_contains_point(mesh::HierarchicalMesh, cell_idx::Integer,
                             point::NTuple{D, Integer}) where D
    # Walk down from root, checking each level
    # ...
end

end
```

Key principles:

- Stay in integer arithmetic when possible. Use the bit primitives for low-level ops.
- Work in LCA-relative or cell-local frames. Avoid materializing absolute coordinates unless you have to (and if you do, document why).
- Test with refined meshes, not just root-only ones. Bugs in geometry often show up only after refinement.

## Adding a new mesh type

The `HierarchicalMesh` is one mesh type — cell-tree AMR. Other mesh types (Berger-Oliger patches, Voronoi/moving-mesh, structured grids) are reasonable to add as parallel mesh types.

Pattern:

```julia
module PatchMesh

using HierarchicalGrids
using HierarchicalGrids.BitPrimitives

# Your own mesh type
struct PatchAMRMesh{D, M}
    patches::Vector{Patch{D}}
    # ...
end

# Implement the analogous API
n_cells(mesh::PatchAMRMesh) = sum(n_cells, mesh.patches)
function find_parent(mesh::PatchAMRMesh, i::Integer); ...; end
# ...

end
```

The Storage layer (FieldSet) doesn't care about the mesh type — it's just an array of n elements. So your patch-based fields work with the same `allocate_fields` API.

The Threading layer would need to be taught about your mesh type's iteration patterns. The interface is small: `partition_for_threads(mesh, n_chunks)` returns `Vector{ThreadChunk}`. You'd implement an analog for your mesh type.

## Adding a refinement criterion / solver-mesh coupling

The framework intentionally doesn't impose a refinement-criterion API. Different physics has wildly different criteria, and forcing them through a common interface tends to create awkward fits.

Pattern: your solver computes which cells to refine based on its own state, then calls `refine_cells!` on the mesh:

```julia
function adapt_mesh!(state::HydroState; gradient_threshold = 0.1)
    cells_to_refine = Int[]
    cells_to_coarsen = Int[]

    for i in 1:n_cells(state.mesh)
        is_leaf(state.mesh.cells[i]) || continue

        if compute_refinement_indicator(state, i) > gradient_threshold
            push!(cells_to_refine, i)
        elseif should_coarsen(state, i)
            push!(cells_to_coarsen, i)
        end
    end

    # Refine and coarsen
    if !isempty(cells_to_coarsen)
        # Coarsening must respect parent-of-leaves constraint
        coarsen_cells!(state.mesh, valid_coarsening_parents(cells_to_coarsen))
    end
    if !isempty(cells_to_refine)
        refine_cells!(state.mesh, cells_to_refine)
    end

    # Resize and reinitialize fields
    resize_fields!(state.fields, n_cells(state.mesh))
    interpolate_to_new_cells!(state)
end
```

This gives you full control. The framework provides the mesh operations; the policy is yours.

## Adding I/O

Not currently in scope for the framework. The cell-path mechanism gives you process-independent identities, which is the foundation for portable I/O — you save and load by path rather than by index. A future I/O module would build on this.

For now, the recommended approach for checkpointing is:

```julia
# Save
function save_checkpoint(filename, mesh, fields)
    # Save mesh structure (cells array is enough)
    # Save fields (one array per field)
    # Save canonical reference level and any other metadata
end
```

You'll want to use a real format (HDF5, JLD2, Parquet) for production work; the framework doesn't impose one.

## Style guide for contributors

A few conventions used throughout the framework:

- **Type parameters carry information**: `HierarchicalMesh{D, M}` carries dimension and integer type. Methods dispatch on these. New types should follow the same pattern.

- **`@inline` aggressively for hot paths**: Layer 2.5 storage accessors and Layer 0 bit operations use `@inline` extensively. Without it, the layout abstraction would have indirection costs.

- **Documentation strings on every exported function**: Use the `"""docstring"""` form. Include type signatures for clarity.

- **Tests for every layer**: Each module has a corresponding test file. Tests should cover both happy paths and edge cases (empty meshes, single-cell meshes, deeply refined meshes).

- **Error messages are full sentences**: `error("Cell $i is not a leaf; cannot refine")` not `error("not a leaf")`. Future-you debugging at 2am will thank present-you.

- **Comments explain why, not what**: The code shows what; comments should explain why this particular approach was chosen, what tradeoff is being made, what alternative was considered and rejected.

## Where help is most valuable

If you'd like to contribute:

- **Polytope clipping** (r3d-equivalent) is the biggest missing piece. Integer-exact clipping would unlock Voronoi geometry and moving-mesh hydro.
- **GPU layouts** would extend the layout abstraction to GPU-friendly access patterns.
- **MPI integration** at the Threading layer would scale the framework to distributed-memory machines.
- **Documentation examples**: more example mini-apps showing different use cases (Poisson solver, particle-in-cell, radiative transfer).
- **Performance benchmarking**: realistic comparisons against existing frameworks on representative problems.

The framework is intentionally opinionated about its core. New features are welcome; deviations from the architectural principles need a strong justification.
