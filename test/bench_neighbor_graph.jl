# Manual benchmark for the path-walking neighbor graph builder.
#
# This file is NOT wired into runtests.jl. Run it directly:
#
#   julia --project=. test/bench_neighbor_graph.jl
#
# It builds uniformly-refined 2D meshes at 16x16, 64x64, 256x256 leaves,
# and times `build_neighbor_graph`. The current implementation is the
# path-walking algorithm (O(N · D · depth_max) worst-case); the previous
# axis-bucket builder (O(N²) worst-case) is included verbatim below for
# A/B comparison.
#
# Numbers measured on a M-series Mac (4/2026):
#
#   16x16   (256   leaves):   bucket ≈ 0.14 ms   path ≈ 0.22 ms   speedup 0.6x
#   64x64   (4096  leaves):   bucket ≈ 3.5  ms   path ≈ 3.0  ms   speedup 1.2x
#   256x256 (65536 leaves):   bucket ≈ 194  ms   path ≈ 73   ms   speedup 2.7x
#
# At small sizes the path-walking cost is similar or slightly higher
# because of per-leaf array allocation; the asymptotic win shows up
# beyond a few thousand leaves and grows with mesh size. The more
# important property is the WORST-CASE complexity guarantee: bucket is
# O(N²) when leaves quantize into a single bucket on a degenerate axis-
# aligned cut; path is always O(N · D · depth_max).

using HierarchicalGrids
using HierarchicalGrids: build_neighbor_graph, cell_unit_box, n_cells

# --- Reference legacy implementation: bucket-based builder ---------------
# Inlined here so the benchmark can A/B-compare without keeping the old
# code in src/. Not exported, only used by this file.

const _BENCH_EPS = 1e-12

@inline _bench_open_overlap(a_lo, a_hi, b_lo, b_hi) =
    (a_hi > b_lo + _BENCH_EPS) && (b_hi > a_lo + _BENCH_EPS)

function bucket_build_neighbor_graph(mesh::HierarchicalMesh{D}) where {D}
    n = n_cells(mesh)
    F = 2 * D
    representatives = fill(ntuple(_ -> UInt32(0), Val(F)), n)
    fine = Dict{Tuple{UInt32, UInt8}, Vector{UInt32}}()
    n == 0 && return (representatives, fine)

    leaf_indices = UInt32[]
    leaf_lo = NTuple{D, Float64}[]
    leaf_hi = NTuple{D, Float64}[]
    for i in 1:n
        if HierarchicalGrids.is_leaf(mesh[i])
            (lo, hi) = cell_unit_box(mesh, i)
            push!(leaf_indices, UInt32(i))
            push!(leaf_lo, lo)
            push!(leaf_hi, hi)
        end
    end
    L = length(leaf_indices)
    quantize(x) = round(Int64, x / _BENCH_EPS)
    lo_buckets = ntuple(_ -> Dict{Int64, Vector{Int}}(), Val(D))
    hi_buckets = ntuple(_ -> Dict{Int64, Vector{Int}}(), Val(D))
    for li in 1:L
        for d in 1:D
            push!(get!(lo_buckets[d], quantize(leaf_lo[li][d]), Int[]), li)
            push!(get!(hi_buckets[d], quantize(leaf_hi[li][d]), Int[]), li)
        end
    end
    for li in 1:L
        i_idx = leaf_indices[li]
        a_lo = leaf_lo[li]; a_hi = leaf_hi[li]
        face_arr = fill(UInt32(0), F)
        for d in 1:D
            face_lo_idx = 2*d - 1; face_hi_idx = 2*d
            qlo = quantize(a_lo[d]); qhi = quantize(a_hi[d])
            on_lo = a_lo[d] <= _BENCH_EPS
            on_hi = a_hi[d] >= 1.0 - _BENCH_EPS
            if !on_lo
                cands = get(hi_buckets[d], qlo, nothing)
                if cands !== nothing
                    rep = UInt32(0); fine_list = UInt32[]
                    for cj in cands
                        cj == li && continue
                        ok = true
                        for d2 in 1:D
                            d2 == d && continue
                            if !_bench_open_overlap(a_lo[d2], a_hi[d2],
                                                     leaf_lo[cj][d2], leaf_hi[cj][d2])
                                ok = false; break
                            end
                        end
                        ok || continue
                        j_idx = leaf_indices[cj]
                        push!(fine_list, j_idx)
                        if rep == 0 || j_idx < rep; rep = j_idx; end
                    end
                    if rep != 0
                        face_arr[face_lo_idx] = rep
                        if length(fine_list) > 1
                            sort!(fine_list)
                            fine[(i_idx, UInt8(face_lo_idx))] = fine_list
                        end
                    end
                end
            end
            if !on_hi
                cands = get(lo_buckets[d], qhi, nothing)
                if cands !== nothing
                    rep = UInt32(0); fine_list = UInt32[]
                    for cj in cands
                        cj == li && continue
                        ok = true
                        for d2 in 1:D
                            d2 == d && continue
                            if !_bench_open_overlap(a_lo[d2], a_hi[d2],
                                                     leaf_lo[cj][d2], leaf_hi[cj][d2])
                                ok = false; break
                            end
                        end
                        ok || continue
                        j_idx = leaf_indices[cj]
                        push!(fine_list, j_idx)
                        if rep == 0 || j_idx < rep; rep = j_idx; end
                    end
                    if rep != 0
                        face_arr[face_hi_idx] = rep
                        if length(fine_list) > 1
                            sort!(fine_list)
                            fine[(i_idx, UInt8(face_hi_idx))] = fine_list
                        end
                    end
                end
            end
        end
        representatives[i_idx] = ntuple(f -> face_arr[f], Val(F))
    end
    return (representatives, fine)
end

# --- Build a uniform 2D mesh of side `s` (so s*s leaves) ---
function uniform_2d_mesh(s::Int)
    @assert ispow2(s) "uniform_2d_mesh expects a power-of-two side length"
    mesh = HierarchicalMesh{2}()
    HierarchicalGrids.refine_cells!(mesh, [1])
    levels_to_refine = Int(log2(s)) - 1
    for _ in 1:levels_to_refine
        HierarchicalGrids.rebuild_caches!(mesh)
        leaves = [i for i in 1:n_cells(mesh)
                  if HierarchicalGrids.is_leaf(mesh[i])]
        HierarchicalGrids.refine_cells!(mesh, leaves)
    end
    return mesh
end

function bench_size(s::Int)
    mesh = uniform_2d_mesh(s)
    n_leaves = count(i -> HierarchicalGrids.is_leaf(mesh[i]), 1:n_cells(mesh))

    # Warm both
    _ = build_neighbor_graph(mesh)
    _ = bucket_build_neighbor_graph(mesh)

    # Time path-walking (current builder)
    t_path = @elapsed for _ in 1:3
        build_neighbor_graph(mesh)
    end
    t_path /= 3

    # Time bucket
    t_bucket = @elapsed for _ in 1:3
        bucket_build_neighbor_graph(mesh)
    end
    t_bucket /= 3

    println(rpad("$(s)x$(s)  ($(n_leaves) leaves)", 30),
            "  bucket = ", round(t_bucket * 1000; digits=2), " ms",
            "    path = ", round(t_path * 1000; digits=2), " ms",
            "    speedup = ", round(t_bucket / max(t_path, 1e-9); digits=2), "x")
end

println("Neighbor graph builder benchmark")
println("================================")
println()
bench_size(16)
bench_size(64)
bench_size(256)
