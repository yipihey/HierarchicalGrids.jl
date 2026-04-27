# Refinement events and listeners

Mesh refinement is a batch operation that can renumber cells, remove
cells (under coarsening), and create new cells. Any per-cell data
maintained outside the mesh — fields, particle bins, solver scratch —
must follow these structural changes. The framework exposes an
observer-style API so external state can stay synchronized without
the mesh having to know about it.

## RefinementEvent

After every `refine_cells!` or `coarsen_cells!` call, the mesh fires a
single `RefinementEvent` summarizing the batch:

| Field | Meaning |
|-------|---------|
| `refined_parents::Vector{UInt32}`           | OLD indices of cells that became non-leaf parents in this batch. |
| `new_children::Vector{UnitRange{UInt32}}`   | Parallel to `refined_parents`. The NEW-index range of children produced by refining each parent. |
| `coarsened_parents::Vector{UInt32}`         | NEW indices of cells that became leaves again. |
| `removed_old_indices::Vector{UInt32}`       | OLD indices of cells removed by coarsening. |
| `index_remap::Vector{UInt32}`               | Length = old `n_cells`. `index_remap[old_i] == 0` for removed cells; otherwise it is the new index. |

Cell index spaces are NOT preserved across an event — `index_remap` is
how a listener relocates its parallel-array data without doing its own
search.

## Registering a listener

```julia
using HierarchicalGrids

mesh = HierarchicalMesh{2}()
density = Ref(zeros(Float64, n_cells(mesh)))

handle = register_refinement_listener!(mesh) do event::RefinementEvent
    old = density[]
    new = zeros(Float64, n_cells(mesh))
    for old_i in eachindex(event.index_remap)
        ni = event.index_remap[old_i]
        ni == 0 && continue           # cell was coarsened away
        new[ni] = old[old_i]
    end
    # Naive policy: copy parent values to newly-created children.
    for (k, parent_old) in enumerate(event.refined_parents)
        for ci in event.new_children[k]
            new[ci] = old[parent_old]
        end
    end
    density[] = new
end

refine_cells!(mesh, [1])
@assert length(density[]) == n_cells(mesh)

# Later, when the listener is no longer needed:
unregister_refinement_listener!(mesh, handle)
```

The callback signature is `callback(event::RefinementEvent)`. Listeners
fire in registration order and are snapshotted before iteration, so a
listener that registers or unregisters another listener does not see
the in-flight event. Listener exceptions are caught and rethrown after
cache invalidation, leaving the mesh in a consistent state.

## AdaptiveField — auto-tracking polynomial fields

`AdaptiveField` is a thin wrapper that registers a refinement listener on
construction and resizes/permutes the wrapped `PolynomialFieldSet` as the
mesh evolves. The user keeps writing/reading per-cell polynomial
coefficients with the usual storage syntax; index alignment with the
mesh is maintained automatically.

```julia
basis = MonomialBasis{2, 2}()
field = allocate_polynomial_fields(SoA(), basis, n_cells(mesh); rho=Float64)
af = AdaptiveField(field, mesh)
# ... user code: write rho coefficients, refine_cells!, coarsen_cells!, ...
dispose!(af)   # unregister the listener (also runs in the finalizer)
```

### Refinement: constant prolongation

When a parent splits, every child inherits the parent's coefficient
vector verbatim. The (now non-leaf) parent slot also retains its
coefficients. This is exact for degree-0 fields and a reasonable
fallback for higher degrees (no information added or lost — the children's
piecewise reconstruction equals the parent's polynomial only in the
degree-0 case; higher-degree applications usually re-initialize via
their physics step or a re-projection helper after refinement).

### Coarsening: L²-projection (degree ≥ 1, MonomialBasis)

When children are coarsened back into a parent, the parent's coefficients
are computed by **L²-projection** of the children's piecewise polynomial
onto the parent's polynomial space:

```
⟨p_parent, φ_α⟩_P = Σ_k ⟨p_{C_k}, φ_α⟩_{C_k}     for all |α| ≤ P
```

solved as a per-event mass-matrix system

```
M_P · c_parent = Σ_k M_{P, C_k} · c_{C_k}
```

where `M_P` is the parent's reference mass matrix on `[0, 1]^D` and
`M_{P, C_k}` is the cross mass matrix integrated analytically over child
`k`'s sub-box in the parent's reference frame. All moments up to the
basis degree are preserved exactly (modulo round-off): total mass,
centroid×mass, second moments, and so on.

For degree-0 fields the projection collapses to the volume-weighted
average — bit-exact equal to the legacy implementation.

For non-`MonomialBasis` polynomial bases (currently `BernsteinBasis` at
degree ≥ 1, plus any future bases without a direct monomial-coefficient
path), the wrapper falls back to constant-moment-only coarsening with a
single warning. Convert via `change_basis` to the monomial basis to get
the exact L²-projection.

## Where this fits

Hand-written refinement-aware fields, the [`step_with_amr!`](amr_driver.md)
driver, and any user struct with parallel-array indexing all share this
hook. The framework itself uses it to invalidate the cached
`NeighborGraph` (see [neighbors and halos](neighbors_and_halos.md)) on
the first structural change after the graph is built.

## See also

- `refine_cells!`, `coarsen_cells!` — what fires the event.
- `AdaptiveField`, `dispose!` — the auto-tracking polynomial-field wrapper.
- [`step_with_amr!`](amr_driver.md) — a topology-only driver that
  composes with this listener pattern.
- [`face_neighbors`](neighbors_and_halos.md) — derived data that is
  invalidated automatically on each event.
