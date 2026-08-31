# Triangle Mesh to Sphere Packing (CUDA)

Fill a closed triangle mesh with non-overlapping spheres, on the GPU. Given a watertight `.obj`,
this produces a packing of variable-radius spheres contained in the mesh volume, using a **greedy
voxel-grid** algorithm implemented end-to-end in CUDA. Packing 20,000 spheres into a 100k-triangle
mesh takes under half a second.

Sphere packings of this kind are used as volumetric proxies for collision detection, physically
based simulation, and haptic rendering, where a set of spheres is far cheaper to test than the
original geometry.

<table>
<tr>
  <th width="50%">Input mesh</th>
  <th width="50%">Output sphere packing</th>
</tr>
<tr>
  <td><img src="docs/images/knot.jpg" alt="Input triangle mesh: a knotted tube" width="100%"></td>
  <td><img src="docs/images/knot_packing.jpg" alt="The same shape filled with spheres" width="100%"></td>
</tr>
<tr>
  <td><code>hard_geometry.obj</code>, 100k triangles</td>
  <td>2 million spheres, 90.64% of mesh volume, 65 s</td>
</tr>
</table>

The bundled mesh is deliberately challenging: a knotted tube whose cross-section morphs between
slender rods, knife-edge ribbons, radial fins, and sharp stars, with sub-millimetre slots where
strands pass each other. See [Test data](#test-data).

## Quick start

**Prerequisites:** CUDA Toolkit 12.x with `nvcc` (developed against 12.5), CMake 3.27+, a C++20
compiler, and an NVIDIA GPU.

cuBQL is a git submodule, so `--recursive` is required — a plain `git clone` will not build.

```bash
git clone --recursive git@github.com:hmeiss/TriangleMeshToSpherePackingCUDA.git
cd TriangleMeshToSpherePackingCUDA

cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Pack the bundled stress-test mesh with 20,000 spheres on a 256^3 grid
./build/SpherePacker data/hard_geometry.obj knot.spheres 20000 256 cubql
```

Already cloned without `--recursive`? Run `git submodule update --init --recursive`.

The run above prints its progress and a density summary, and writes two files: `knot.spheres` and
`knot.log`. On an RTX 3080 Ti:

```
cuBQL BVH build: 14.437 ms, 195536 nodes
initSDFKernel_cuBQL: 79.905 ms
applySignKernel_cuBQL: 54.041 ms
Multi2 candidates: 1274711, iterations: 36, selected spheres: 20000

Packing completed: 20000 spheres generated
Total Runtime: 0.473892 seconds
Packing Density (vs Mesh): 80.5277%
```

`cubql` is the recommended mode. See [Usage](#usage) for the other three and what they are for.

### View the result

No install required — drag `knot.spheres` onto the
**[Sphere Packing Viewer](https://hmeiss.github.io/sphere-packing-viewer/)** to inspect it in the
browser. Drop `data/hard_geometry.obj` in as well to see the packing against the original mesh,
and use *Hide Mesh* to switch between the two.


## Method

The algorithm is greedy and voxel-grid based. A uniform grid is laid over the mesh; each voxel
holds **the radius of the largest sphere that may be centered there**. The packing repeatedly
takes the largest such radius, emits a sphere, and shrinks the affected voxels.

**1. Grid setup.** A uniform grid of `gridRes³` voxels is placed over a *cube* centered on the
mesh bounding box, with side equal to the box's longest axis, so voxels stay cubic
([sphere_packer.cpp:21-46](src/sphere_packer.cpp#L21-L46)).

**2. Initialization (SDF).** Each voxel center is assigned its signed distance to the mesh
surface: unsigned distance to the nearest triangle, negated outside the mesh. Positive value =
inside, and equals the largest sphere that fits at that point without crossing the surface.

**3. Greedy selection.** Repeatedly pick the voxel with the largest value `r`, emit a sphere of
radius `r` at its center, then update every voxel `v` that the new sphere constrains:

```
value[v] ← min(value[v], ‖center[v] − c‖ − r)
```

A voxel whose value drops to `≤ 0` is exhausted and dropped. Because a voxel's value is always
clamped to its clearance from every placed sphere, and never exceeds its clearance from the
surface, **non-overlap and containment hold by construction** — no rejection sampling or
post-pass is needed.

> **A note on terminology.** The field is a signed distance field *only immediately after
> step 2*. From the first placement onward the update in step 3 overwrites voxels with their
> clearance to placed spheres, so the grid is a free-radius (clearance) field, not an SDF of the
> mesh. The code and the CLI flag still say `sdf` in many places for historical reasons; the
> paper's *greedy voxel-grid* is the accurate description of the method as a whole.

### Selection loops

Two selection strategies are implemented.

**One sphere per iteration** (`bvh`, `brute`) — the textbook greedy loop
([sphere_packer.cpp:92-132](src/sphere_packer.cpp#L92-L132)). Each iteration runs a
`cub::DeviceReduce::ArgMax` over the whole grid, emits one sphere, and updates only the voxels
within `2r` of the center. That cutoff is exact rather than approximate: the emitted `r` is the
running maximum, so any voxel needing an update satisfies `d < value[v] + r ≤ 2r`. Cost is
`O(K·N)` reductions for `K` spheres over `N` voxels — the reduction dominates and makes large
sphere counts expensive.

**Batched** (`multi2`, `cubql`) — amortizes the reduction over ~1000 spheres
([cuda_functions.cu:832-997](src/cuda_functions.cu#L832-L997)):

1. Compact all positive voxels into a candidate list (`cub::DeviceSelect::Flagged`).
2. Sort candidates by current radius, descending, with `cub::DeviceRadixSort::SortPairsDescending`
   over a 64-bit key: the float bits of the radius in the high word, a hash of the voxel index in
   the low word. The hash makes ties deterministic and free of grid-order bias.
3. Copy the top `1000` to the host and accept them in order, skipping any that overlaps one
   already accepted from this batch. Accepted radii get a tiny safety pad,
   `max(8·ε·max(1,r), 10⁻⁵·cellSize)`, to absorb float rounding.
4. One kernel shrinks *all* remaining candidates against the whole accepted batch; survivors are
   compacted and the loop repeats.

The first element of each batch is the true global maximum, so the batched variant agrees with
strict greedy on that sphere. Later members of a batch are approximations: they are checked for
overlap against earlier acceptances but not re-shrunk mid-batch, so their ordering can differ
from what strict greedy would produce. In exchange, one sort serves ~1000 placements instead of
one reduction per placement, which is what makes six-figure sphere counts tractable.

### Distance backends

Step 2 has three implementations, selected by the same CLI argument:

| Backend | Nearest-triangle query | Inside/outside test |
|---|---|---|
| `cubql` | cuBQL SAH BVH, custom shrinking-radius traversal | cuBQL 6-ray majority vote |
| `bvh` | in-repo linear BVH (LBVH), Morton-code built | single-axis ray parity count, with a deterministic epsilon retry on degenerate hits |
| `brute` | every voxel against every triangle, `O(V·T)` | ray parity against all triangles |

`cubql` ([cubql_sdf.cu](src/cubql_sdf.cu)) is the fastest and most robust. Two details matter for
its performance:

- **Closest-point routine.** `closestPtOnTri_fast()` uses Ericson's barycentric region test — 6
  dot products, no cross products, no `normalize`/`sqrt` — instead of cuBQL's built-in
  `computeOneIntersection`.
- **Custom traversal.** The traversal is written inline rather than via `forEachLeaf`, with an
  index-only stack (4 bytes/entry, 256 B/thread instead of 512 B) and no per-pop distance filter,
  which reduces SIMT divergence. The builder is configured with `makeLeafThreshold = 1` to match.

The builder is swappable — `sahBuilder` is the default; `radixBuilder`, `rebinRadixBuilder`, and
`gpuBuilder` are commented alternatives at [cubql_sdf.cu:205-218](src/cubql_sdf.cu#L205-L218)
taking identical arguments. SAH wins on large meshes, where Morton-code collisions degrade a
radix tree's spatial coherence; `radixBuilder` builds faster on small meshes.

## Build details

The [Quick start](#quick-start) covers the normal case. A few extras:

- **Other GPUs.** `CUDA_ARCHITECTURES` is `native`, so the build targets the machine it compiles
  on. To build portable binaries, replace it in [CMakeLists.txt:30](CMakeLists.txt#L30):
  ```cmake
  set_property(TARGET SpherePacker PROPERTY CUDA_ARCHITECTURES 70 75 80 86 89)
  ```
- **Windows.** With the Visual Studio generator the executable lands in
  `build/Release/SpherePacker.exe`. Configure with
  `cmake -B build -S . -G "Visual Studio 17 2022"`.
- **Host compiler.** If `nvcc` rejects your default host compiler, point CMake at a supported one
  with `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`.

## Usage

```
SpherePacker <input.obj> <output.spheres> [targetCount] [gridRes] [mode]
```

| # | Argument | Description | Default |
|---|---|---|---|
| 1 | `input.obj` | Input Wavefront OBJ. Must be closed and consistently oriented. | required |
| 2 | `output.spheres` | Output sphere file. A `.log` file is written alongside it. | required |
| 3 | `targetCount` | Number of spheres to place. | `100` |
| 4 | `gridRes` | Grid resolution; the grid is `gridRes³` voxels. | `256` |
| 5 | `mode` | `cubql`, `multi2`, `bvh`, or `brute`. | `bvh` |

An unrecognized `mode` prints a warning and falls back to `bvh`.

### Modes

| Mode | Initialization | Selection | Notes |
|---|---|---|---|
| `cubql` | cuBQL SAH BVH | batched | **Recommended.** Fastest, most robust signing. |
| `multi2` | in-repo LBVH | batched | Same selection loop, older initialization. |
| `bvh` | in-repo LBVH | one per iteration | Reference greedy; slow at high sphere counts. |
| `brute` | brute force | one per iteration | Reference/validation only. `O(voxels × triangles)`. |

### Examples

```bash
# Recommended: a large packing of the bundled stress-test mesh
# (100k spheres, 85.4% density, 1.56 s on an RTX 3080 Ti — needs ~4 GB VRAM)
./build/SpherePacker data/hard_geometry.obj knot.spheres 100000 512 cubql

# Strict greedy reference, for comparing against the batched result
./build/SpherePacker data/hard_geometry.obj ref.spheres 1000 256 bvh

# Defaults: 100 spheres, 256^3 grid, bvh mode
./build/SpherePacker data/hard_geometry.obj out.spheres
```

`.spheres` and `.log` output is gitignored, so runs inside the repo leave the tree clean.

## Output

### `.spheres`

Space-delimited, one sphere per line, written at `float` round-trip precision. No header.

```
x y z r
```

This is exactly the format read by the
[Sphere Packing Viewer](https://hmeiss.github.io/sphere-packing-viewer/), which also accepts the
source `.obj` so you can view a packing against the mesh it came from. The viewer ignores lines
beginning with `#`, so you can annotate a result file by hand without breaking it.

## Performance and memory

Grid memory is `4 · gridRes³` bytes for the field itself:

| `gridRes` | Voxels | Field | Peak, batched modes |
|---|---|---|---|
| 128 | 2.1 M | 8 MB | ~65 MB |
| 256 | 16.8 M | 67 MB | ~500 MB |
| 512 | 134 M | 537 MB | ~4 GB |

The batched modes need substantially more than the field alone: the candidate list, its sort keys
and sorted copies, the keep-flags, and CUB's temporary storage add roughly 25 bytes per *positive*
voxel, on top of a transient `5·gridRes³` bytes during the initial compaction. `bvh` and `brute`
need only the field.

Measured on an RTX 3080 Ti (12 GB) against the bundled 100k-triangle stress-test mesh, `cubql`
mode, initialization included:

| Spheres | `gridRes` | Density vs. mesh | Total runtime |
|---|---|---|---|
| 20,000 | 256 | 80.5% | 0.47 s |
| 100,000 | 512 | 85.4% | 1.56 s |

Resolution controls the packing's fineness: the smallest representable sphere is on the order of
one voxel, and sphere centers are snapped to voxel centers. Raise `gridRes` for finer packings,
and expect the batched modes to be the only practical choice above ~10⁴ spheres.

## Limitations

- **Watertight meshes only.** Both inside/outside tests are ray-parity based. Open, self-
  intersecting, or inconsistently oriented meshes will produce wrong signs and therefore wrong
  packings. There is no validation pass — bad input fails silently.
- **Centers are snapped to voxel centers.** No sub-voxel refinement of center or radius is
  performed, so accuracy is bounded by `gridRes`.
- **No minimum-radius stopping criterion.** The one-per-iteration loop runs exactly
  `targetCount` iterations regardless of whether the grid is exhausted; if it runs out of
  positive voxels it will emit zero-radius spheres. The batched loop does stop when no candidates
  remain.
- **Uniform grid.** Memory grows as `gridRes³`, which bounds resolution for meshes with high
  aspect ratios or large empty regions. A sparse or hierarchical grid would lift this.
- **32-bit voxel indexing.** Voxel counts must fit in `int`, capping `gridRes` at 1290.
- **OBJ only**, via tinyobjloader.

## Project structure

```
src/                    C++/CUDA sources
  main.cpp              CLI, timing, density reporting, .log writer
  mesh.h/.cpp           OBJ loading, bounds, divergence-theorem volume
  sphere_packer.h/.cpp  Grid setup, mode dispatch, .spheres export
  cuda_functions.h/.cu  Both selection loops, grid updates, CUB reductions
  cubql_sdf.h/.cu       cuBQL BVH build, fast closest-point, custom traversal, signing
  lbvh.h/.cu            In-repo linear BVH: build, distance query, ray counting
  helper_math.h         CUDA float3 operators
  cuda_check.h          CUDA_CHECK error macro
external/
  cuBQL/                Submodule: github.com/NVIDIA/cuBQL
  tiny_obj_loader.h     Header-only OBJ loader
scripts/
  hard_geometry_knot.py Generator for the stress-test mesh (see Test data)
data/
  hard_geometry.obj     The generated stress-test mesh, 100k triangles
docs/images/            README figures
```

### Key kernels

| Kernel | File | Role |
|---|---|---|
| `initSDFKernel_cuBQL` | cubql_sdf.cu | Unsigned distance per voxel via custom cuBQL traversal |
| `applySignKernel_cuBQL` | cubql_sdf.cu | 6-ray majority vote, negates outside voxels |
| `initSDFKernel_BVH` / `applyBVHRaySignKernel` | cuda_functions.cu | Same two passes on the in-repo LBVH |
| `buildMulti2SortKeysKernel` | cuda_functions.cu | Packs radius + index hash into 64-bit sort keys |
| `updateMulti2CandidatesAgainstBatchKernel` | cuda_functions.cu | Shrinks all candidates against an accepted batch |
| `updateSDFKernel_AABB` | cuda_functions.cu | Shrinks voxels within `2r` of one placed sphere |
| `flagPositiveKernel` | cuda_functions.cu | Marks live candidates for compaction |

## Test data

`data/hard_geometry.obj` is a deliberately adversarial input, generated by
[scripts/hard_geometry_knot.py](scripts/hard_geometry_knot.py). It is a single closed tube swept
along a (3,2) torus knot whose cross-section blends between:

- round bulges — the largest local feature size
- slender rods — aspect ratio around 200:1
- thin ribbons — shells with two knife edges
- four thin radial fins — plates meeting at re-entrant corners
- 5- and 7-pointed stars — sharp convex tips and sharp concave valleys

and where two strands of the knot pass closest, bulges narrow the crossings into sub-millimetre
slots. Every one of these features targets a different weakness of a voxel-grid method: thin
features that fall below one voxel, concave corners where the largest inscribed sphere is tiny,
and near-contacts where a wrong inside/outside sign bridges two strands that should stay separate.

Because it is a swept closed tube, the mesh is watertight *by construction* — no booleans,
remeshing, or hole filling. Topologically it is a torus (genus 1), which is intended, not a
defect. The script verifies this before exporting: manifoldness, boundary edges, winding
consistency, degenerate triangles, outward normals, non-self-crossing cross-sections, an exact
triangle-triangle self-intersection test, and the narrowest slot width.

Regenerating it needs only NumPy:

```bash
python3 scripts/hard_geometry_knot.py
```

Set `EXPORT_DIR` near the top of the script to write `.obj`/`.ply`/`.stl`. Run it inside Blender
(`blender --background --python scripts/hard_geometry_knot.py`) to get the mesh as a scene object
instead. The parameters controlling difficulty — `N_ALONG`, `N_CROSS`, `SLOT_GAP`, `CURV_FRAC` —
are documented at the top of the file; lowering `SLOT_GAP` makes the packing problem harder.

<!--## Citation

```bibtex
TODO
```-->

## License

This software is released under the MIT License. See [LICENSE](LICENSE) for details.

This project bundles [cuBQL](https://github.com/NVIDIA/cuBQL) (Apache License 2.0) as a submodule
and [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) (MIT License) as a vendored
header.

## References

1. C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2004 — closest point on triangle.
2. T. Karras, *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees*, HPG 2012 — LBVH construction.
3. [cuBQL](https://github.com/NVIDIA/cuBQL) — CUDA bounding-volume query library.
4. [CUB](https://nvidia.github.io/cccl/cub/) — device-wide sort, scan, select, and reduce primitives.
