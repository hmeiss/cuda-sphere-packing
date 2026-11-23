# Technical documentation

This document explains how the packer works, how the code is organized, and how the test mesh is
made. For building and running the tool, see the [main README](../README.md).

- [Method](#method)
  - [Sphere selection](#sphere-selection)
  - [Distance computation](#distance-computation)
- [Memory](#memory)
- [Project structure](#project-structure)
- [Test data](#test-data)
- [References](#references)

## Method

The packer uses a greedy algorithm on a voxel grid. Each voxel stores **the radius of the largest
sphere that can be centered there**. The packer repeatedly takes the largest of these radii,
places a sphere, and lowers the values of the voxels around it.

**1. Grid setup.** A grid of `gridRes³` voxels is placed over a cube around the mesh. The cube is
centered on the mesh bounding box and its side is the longest side of the box, so all voxels are
cubes (`SpherePacker::initializeGrid` in [sphere_packer.cpp](../src/sphere_packer.cpp)).

**2. Distance field.** Each voxel center gets its distance to the nearest triangle. The value is
positive inside the mesh and negative outside. For an inside voxel, this is the radius of the
largest sphere that fits at that point without crossing the surface.

**3. Greedy selection.** Repeatedly take the voxel with the largest value `r`, place a sphere of
radius `r` at its center `c`, and update every voxel `v` near the new sphere:

```
value[v] ← min(value[v], ‖center[v] − c‖ − r)
```

A voxel whose value drops to zero or below has no room left and is ignored from then on. A
voxel's value is never larger than its distance to the surface or to any placed sphere. So **the
spheres never overlap and never leave the mesh**, without any extra checks or cleanup step.

> **About the name "SDF".** The grid is a signed distance field (SDF) only right after step 2.
> Once the first sphere is placed, step 3 replaces values with the free space around placed
> spheres, so the grid no longer describes the mesh. The code and the CLI still use the name
> `sdf` in many places for historical reasons.

### Sphere selection

There are two ways to pick spheres.

**One sphere per round** (`bvh`, `brute`) is the plain greedy loop
(`SpherePacker::packOneSpherePerIteration` in [sphere_packer.cpp](../src/sphere_packer.cpp)).
Each round searches the whole grid for the maximum with `cub::DeviceReduce::ArgMax`, places one
sphere, and updates only the voxels within `2r` of its center. This limit loses nothing: `r` is
the current maximum, so any voxel that needs an update satisfies `d < value[v] + r ≤ 2r`. The
cost is one full-grid search per sphere, which gets slow for large sphere counts.

**Batched** (`multi2`, `cubql`) places about 1000 spheres per full-grid pass
(`packSpheresMulti2SnapshotImpl` in [cuda_functions.cu](../src/cuda_functions.cu)):

1. Collect all voxels with a positive value into a candidate list (`cub::DeviceSelect::Flagged`).
2. Sort the candidates by radius, largest first, with `cub::DeviceRadixSort::SortPairsDescending`.
   The 64-bit sort key holds the radius in the upper 32 bits and a hash of the voxel index in
   the lower 32 bits. The hash breaks ties in a fixed order that doesn't depend on the grid
   layout.
3. Copy the top 1000 to the CPU and accept them in order, skipping any that overlaps a sphere
   already accepted in this batch. Each accepted radius is reduced by a tiny margin,
   `max(8·ε·max(1,r), 10⁻⁵·cellSize)`, to cover floating-point rounding.
4. One kernel updates *all* remaining candidates against the whole batch. Candidates with no room
   left are removed, and the loop repeats.

The result is close to plain greedy but not identical. Within a round, candidates are ranked by
their values at the start of the round. A candidate that is skipped because it overlaps an
accepted sphere would, in plain greedy, only shrink and might still be picked before some of the
spheres accepted later in the round. Once one placement differs, every later round starts from a
different grid, so the two packings drift apart. Ties are also broken in a different order.

The benefit is speed. Plain greedy searches the whole grid once for every sphere. The batched
loop sorts the candidates once per round and places about 1000 spheres from that single sort.
This is what makes hundreds of thousands of spheres practical.

### Distance computation

Step 2 has three implementations, chosen with the `mode` argument:

| Mode | Nearest triangle | Inside/outside test |
|---|---|---|
| `cubql` | cuBQL BVH (SAH), custom traversal | majority vote over 6 rays |
| `bvh`, `multi2` | built-in linear BVH (LBVH) | ray crossing count along one axis, retried with a small offset if the ray hits an edge or vertex |
| `brute` | every voxel against every triangle, `O(V·T)` | ray crossing count against all triangles |

`cubql` ([cubql_sdf.cu](../src/cubql_sdf.cu)) is the fastest and most reliable. Two things make
it fast:

- **Closest point on a triangle.** `closestPtOnTri_fast()` uses Ericson's barycentric method,
  which needs 6 dot products and no cross products, normalization or square roots. It replaces
  cuBQL's built-in `computeOneIntersection`.
- **Custom traversal.** The BVH traversal is written by hand instead of using `forEachLeaf`. Its
  stack stores only node indices (4 bytes per entry, 256 B per thread instead of 512 B), and it
  skips the distance check when popping nodes. This keeps GPU threads in step with each other.
  The BVH is built with `makeLeafThreshold = 1` (one triangle per leaf) to match.

The BVH builder can be swapped. `sahBuilder` is the default. `radixBuilder`, `rebinRadixBuilder`
and `gpuBuilder` are commented out next to it in `initSDF_cuBQL` and take the same arguments.
SAH gives the best trees for large meshes. `radixBuilder` builds faster on small meshes.

## Memory

The grid itself takes `4 · gridRes³` bytes:

| `gridRes` | Voxels | Grid | Peak in batched modes |
|---|---|---|---|
| 128 | 2.1 M | 8 MB | ~65 MB |
| 256 | 16.8 M | 67 MB | ~500 MB |
| 512 | 134 M | 537 MB | ~4 GB |

The batched modes need much more than the grid. The candidate list, sort keys, sorted copies,
flags and CUB's working memory add about 25 bytes per voxel inside the mesh. Building the
candidate list also needs another `5 · gridRes³` bytes for a short time. `bvh` and `brute` only
need the grid.

## Project structure

```
src/                    C++/CUDA sources
  main.cpp              Command line, timing, density, .log file
  mesh.h/.cpp           OBJ loading, bounding box, mesh volume
  sphere_packer.h/.cpp  Grid setup, mode selection, .spheres output
  cuda_functions.h/.cu  Both selection loops, grid updates, CUB calls
  cubql_sdf.h/.cu       cuBQL BVH build, closest point, custom traversal, inside/outside test
  lbvh.h/.cu            Built-in linear BVH: build, distance query, ray counting
  helper_math.h         CUDA float3 operators
  cuda_check.h          CUDA_CHECK error macro
external/
  cuBQL/                Submodule: github.com/NVIDIA/cuBQL
  tiny_obj_loader.h     Single-header OBJ loader
scripts/
  hard_geometry_knot.py Script that creates the test mesh
data/
  hard_geometry.obj     The test mesh, 100k triangles
docs/                   This document and the README images
```

### Main kernels

| Kernel | File | What it does |
|---|---|---|
| `initSDFKernel_cuBQL` | cubql_sdf.cu | Distance to the nearest triangle for each voxel |
| `applySignKernel_cuBQL` | cubql_sdf.cu | Inside/outside vote with 6 rays, makes outside voxels negative |
| `initSDFKernel_BVH` / `applyBVHRaySignKernel` | cuda_functions.cu | The same two steps with the built-in BVH |
| `buildMulti2SortKeysKernel` | cuda_functions.cu | Builds the 64-bit sort keys from radius and index hash |
| `updateMulti2CandidatesAgainstBatchKernel` | cuda_functions.cu | Updates all candidates against an accepted batch |
| `updateSDFKernel_AABB` | cuda_functions.cu | Updates voxels within `2r` of one new sphere |
| `flagPositiveKernel` | cuda_functions.cu | Marks voxels that still have room |

## Test data

`data/hard_geometry.obj` is designed to be hard to pack. It is created by
[scripts/hard_geometry_knot.py](../scripts/hard_geometry_knot.py) as a single closed tube that
follows a (3,2) torus knot. Along the tube, its cross-section changes between:

- round bulges, the thickest parts
- thin rods, about 200 times longer than wide
- flat ribbons with two sharp edges
- four thin fins meeting in the middle
- 5- and 7-pointed stars with sharp tips and narrow inner corners

Where two parts of the knot pass close to each other, the bulges leave only very narrow gaps.
Each feature tests a different weak spot of a voxel grid: parts thinner than one voxel, inner
corners where only tiny spheres fit, and narrow gaps where a wrong inside/outside result would
join two parts of the tube that should stay separate.

The tube is closed by construction, so the mesh is watertight without any repair steps. It has
the shape of a torus (one hole), which is intended. Before saving, the script checks that the mesh
is manifold, has no open edges, has consistent and outward-facing normals, has no degenerate or
self-intersecting triangles, and reports the narrowest gap.

To create the mesh again, you only need NumPy:

```bash
python3 scripts/hard_geometry_knot.py
```

Set `EXPORT_DIR` at the top of the script to save `.obj`, `.ply` or `.stl` files. To load the
mesh into a Blender scene instead, run
`blender --background --python scripts/hard_geometry_knot.py`. The settings that control the
difficulty (`N_ALONG`, `N_CROSS`, `SLOT_GAP`, `CURV_FRAC`) are explained at the top of the file.
A smaller `SLOT_GAP` makes the mesh harder to pack.

## References

1. C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2004. Closest point on a triangle.
2. T. Karras, *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees*, HPG 2012. Linear BVH construction.
3. [cuBQL](https://github.com/NVIDIA/cuBQL), a CUDA library for BVH queries.
4. [CUB](https://nvidia.github.io/cccl/cub/), GPU sort, scan, select and reduce functions.
