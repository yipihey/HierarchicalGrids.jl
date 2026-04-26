# Neighbors and halos

Stencil kernels, sparse-Jacobian assembly, and boundary fluxes all need
to ask "which leaf is on the other side of this face?" The neighbor
layer answers that, and the `HaloView` wrapper builds a small
neighborhood-indexing convenience on top of it for use in solver
kernels.

## NeighborGraph

`NeighborGraph{D, M}` is a face-adjacency structure built lazily on top
of a `HierarchicalMesh{D, M}`:

- A representative neighbor per leaf and face, stored as
  `NTuple{2D, UInt32}` in axis-major order
  `(axis 1 lo, axis 1 hi, axis 2 lo, axis 2 hi, …)`.
- A side dictionary holding the full list of fine neighbors when a
  coarse cell shares one face with several finer leaves (only relevant
  for unbalanced meshes).
- A registered refinement listener that invalidates the graph on the
  next structural change.

You rarely need to construct it directly; the public functions below
build and cache it on first use.

## Public functions

### `face_neighbors`

```julia
face_neighbors(mesh, i)::NTuple{2D, UInt32}
```

Per-face representative neighbors of leaf `i`. A face entry of `0`
means the face is on the domain boundary. For a coarse cell against
fine neighbors, the *lowest-indexed* fine leaf is returned as the
representative; use `face_fine_neighbors` to enumerate the rest.

### `face_fine_neighbors`

```julia
face_fine_neighbors(mesh, i, face)::Vector{UInt32}
```

All fine-leaf neighbors on a single face index. For a face with one
neighbor returns a one-element vector; for a domain-boundary face
returns an empty vector.

### `cell_adjacency_sparsity`

```julia
cell_adjacency_sparsity(mesh; depth=1, leaves_only=true) ::
    SparseMatrixCSC{Bool, Int32}
```

Boolean adjacency matrix where `M[i, j]` is `true` iff cells `i` and
`j` are within `depth` face-hops of each other (the result is
symmetric, with the diagonal set). The principal use is supplying a
sparsity pattern to sparse-AD Jacobian tooling.

### `face_neighbors_with_bcs`

```julia
face_neighbors_with_bcs(mesh, i, frame_bcs)::NTuple{2D, UInt32}
```

Periodic-aware variant of `face_neighbors`. Reads
`frame_bcs::FrameBoundaries{D}` (see
[boundary conditions](boundary_conditions.md)), and on every periodic
axis replaces the boundary-side `0` entries with the wrap-around
neighbor. Non-periodic boundary kinds are left as `0` for PDE-level
code to consume.

## A complete example

```julia
using HierarchicalGrids
using SparseArrays

mesh = HierarchicalMesh{2}()
refine_cells!(mesh, [1])              # 4 children
refine_cells!(mesh, [2])              # one child further refined

# Per-leaf face adjacency
for i in 1:n_cells(mesh)
    is_leaf(mesh.cells[i]) || continue
    nbrs = face_neighbors(mesh, i)
    @show i, nbrs                     # (lo_x, hi_x, lo_y, hi_y)
end

# Sparsity pattern, leaves only, 1-hop adjacency
S = cell_adjacency_sparsity(mesh)     # SparseMatrixCSC{Bool, Int32}
```

## HaloView

`HaloView` wraps a `PolynomialFieldView` together with the mesh's
neighbor graph and exposes a stencil-style index. For a depth-1 view,
`hv[i, off]` returns the polynomial-coefficient column of the cell
that is `off` cell-hops away from leaf `i`, walking along the
neighbor graph one axis at a time:

```julia
using HierarchicalGrids

mesh = HierarchicalMesh{2}()
refine_cells!(mesh, [1])
field = allocate_polynomial_fields(SoA(), BernsteinBasis{2, 1}(),
                                    n_cells(mesh); rho = Float64)
init_field_from!(field, frame, x -> 1.0 + 0.1 * x[1])  # see getting_started

hv = halo_view(field.rho, mesh, 1)
center  = hv[2, (0, 0)]               # this cell — a PolynomialView
right   = hv[2, (1, 0)]               # x+1 hop
top     = hv[2, (0, 1)]               # y+1 hop
edge    = hv[2, (-1, 0)]              # `nothing` if off-domain
```

For `D == 1` a scalar offset is also accepted (`hv[i, +1]`). Building
the view is O(1); the underlying graph is built lazily on first
access. Out-of-domain hops return `nothing`; with a `FrameBoundaries`
wired through `face_neighbors_with_bcs`, periodic axes wrap.

The `depth` argument controls the maximum reach (sum of `abs.(off)`).
Passing an offset with greater Manhattan length raises
`ArgumentError`.

## See also

- [`step_with_amr!`](amr_driver.md) — drives refinement; the listener
  in the neighbor graph keeps it valid across cycles.
- [boundary conditions](boundary_conditions.md) — the
  `FrameBoundaries{D}` value that `face_neighbors_with_bcs` consumes.
- `cell_adjacency_sparsity` is the natural input to packages that
  expect a sparse Jacobian template.
