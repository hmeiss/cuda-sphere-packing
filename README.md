# CUDA Sphere Packing of Triangle Meshes

A CUDA implementation of greedy voxel-grid sphere packing for triangle meshes. It fills a
watertight `.obj` mesh with spheres of different sizes that don't overlap and stay inside the
mesh. Packing 20,000 spheres into a mesh with 100k triangles takes under half a second.

Sphere packings like this are used to approximate a mesh's volume in collision detection,
physics simulation and haptic rendering, because testing spheres is much cheaper than testing
the original triangles.

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
  <td>2 million spheres, 90.64% density, 65 s, gridRes=1024</td>
</tr>
</table>

The included test mesh is built to be hard to pack. It is a knotted tube whose shape changes
along its length between thin rods, flat ribbons with sharp edges, fins and star shapes, with
very narrow gaps where the tube passes close to itself. See [Test data](docs/README.md#test-data).

## Quick start

**Requirements:** an NVIDIA GPU with CUDA Toolkit 12.x (tested with 12.5), CMake 3.27 or newer,
and a C++20 compiler.

Clone with `--recursive`, because the cuBQL library is included as a git submodule:

```bash
git clone --recursive https://github.com/hmeiss/cuda-sphere-packing.git
cd cuda-sphere-packing

cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Pack the test mesh with 20,000 spheres on a 256^3 grid
./build/SpherePacker data/hard_geometry.obj knot.spheres 20000 256 cubql
```

If you already cloned without `--recursive`, run `git submodule update --init --recursive`.

This prints progress and a summary, and writes `knot.spheres` and `knot.log`. Example output on
an RTX 3080 Ti:

```
cuBQL BVH build: 14.437 ms, 195536 nodes
initSDFKernel_cuBQL: 79.905 ms
applySignKernel_cuBQL: 54.041 ms
Multi2 candidates: 1274711, iterations: 36, selected spheres: 20000

Packing completed: 20000 spheres generated
Total Runtime: 0.473892 seconds
Packing Density (vs Mesh): 80.5277%
```

`cubql` is the fastest mode. The other modes are described under [Usage](#usage).

### View the result

Open the **[Sphere Packing Viewer](https://hmeiss.github.io/sphere-packing-viewer/)** in your
browser and drag `knot.spheres` onto it. Nothing needs to be installed. To compare with the
original mesh, also drop in `data/hard_geometry.obj` and use *Hide Mesh* to switch between them.

<details>
<summary><b>Build details</b>: other GPUs, Windows, compiler</summary>

- **Other GPUs.** By default the build only targets the GPU in the machine you compile on
  (`CUDA_ARCHITECTURES native` in [CMakeLists.txt](CMakeLists.txt)). To build a binary that also
  runs on other GPUs, list their architectures instead:
  ```cmake
  set_property(TARGET SpherePacker PROPERTY CUDA_ARCHITECTURES 70 75 80 86 89)
  ```
- **Windows.** Configure with `cmake -B build -S . -G "Visual Studio 17 2022"`. The program is
  then built as `build/Release/SpherePacker.exe`.
- **Compiler.** If `nvcc` doesn't accept your default C++ compiler, choose a supported one with
  `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`.

</details>

## Method

A uniform grid is placed over the mesh. Each voxel starts with its distance to the mesh surface,
which is the radius of the largest sphere that fits at that point. The packer then repeats two
steps: take the voxel with the largest value `r` and place a sphere of radius `r` at its center
`c`, then lower the value of every voxel near the new sphere:

```
value[v] ← min(value[v], ‖center[v] − c‖ − r)
```

A voxel's value is never larger than its distance to the surface or to any placed sphere. So the
spheres never overlap and never leave the mesh.

Placing one sphere at a time needs a search over the whole grid for every sphere. The batched
modes (`cubql` and `multi2`) instead sort all remaining voxels by value once per round and accept
the top ~1000, skipping any that overlap each other. This is much faster for large sphere
counts. The result is close to strict greedy but not identical. Details are in
[Sphere selection](docs/README.md#sphere-selection).

More detail on the algorithm, memory use and code layout is in [docs/README.md](docs/README.md).

## Usage

```
SpherePacker <input.obj> <output.spheres> [targetCount] [gridRes] [mode]
```

| # | Argument | Description | Default |
|---|---|---|---|
| 1 | `input.obj` | Input mesh. Must be closed, with consistently oriented faces. | required |
| 2 | `output.spheres` | Output file. A `.log` file with the same name is also written. | required |
| 3 | `targetCount` | Number of spheres to place. | `100` |
| 4 | `gridRes` | Grid resolution. The grid has `gridRes³` voxels. | `256` |
| 5 | `mode` | `cubql`, `multi2`, `bvh` or `brute`. | `bvh` |

An unknown `mode` prints a warning and uses `bvh`.

### Modes

| Mode | Distance computation | Spheres per round | Notes |
|---|---|---|---|
| `cubql` | cuBQL BVH | ~1000 | **Recommended.** Fastest and most reliable. |
| `multi2` | built-in BVH | ~1000 | Like `cubql`, with the older built-in BVH. |
| `bvh` | built-in BVH | 1 | Exact greedy order. Slow for many spheres. |
| `brute` | every triangle | 1 | Very slow. Only for checking results. |

### Examples

```bash
# 100k spheres: 85.4% density in 1.56 s on an RTX 3080 Ti, needs ~4 GB GPU memory
./build/SpherePacker data/hard_geometry.obj knot.spheres 100000 512 cubql

# Exact greedy order, to compare with the batched result
./build/SpherePacker data/hard_geometry.obj ref.spheres 1000 256 bvh

# Defaults: 100 spheres, 256^3 grid, bvh mode
./build/SpherePacker data/hard_geometry.obj out.spheres
```

`.spheres` and `.log` files are listed in `.gitignore`, so results saved inside the repo don't
show up in `git status`.

### Output format

Each line of a `.spheres` file is one sphere: center `x y z` and radius `r`, separated by spaces.
There is no header line.

```
x y z r
```

The [Sphere Packing Viewer](https://hmeiss.github.io/sphere-packing-viewer/) reads this format
directly. It skips lines starting with `#`, so you can add comments to a file by hand.

## Performance

Measured on an RTX 3080 Ti (12 GB) with the test mesh in `cubql` mode, including setup:

| Spheres | `gridRes` | Density | Total time | Peak GPU memory |
|---|---|---|---|---|
| 20,000 | 256 | 80.5% | 0.47 s | ~500 MB |
| 100,000 | 512 | 85.4% | 1.56 s | ~4 GB |

`gridRes` sets how fine the packing can get. The smallest sphere is about one voxel wide, and
sphere centers always sit on voxel centers. For more than about 10,000 spheres, use `cubql` or
`multi2`. Memory grows with `gridRes³`. See [Memory](docs/README.md#memory) for details.

## Limitations

- **Closed meshes only.** The inside/outside test counts ray crossings. Meshes with holes,
  self-intersections or flipped faces give wrong results, and the tool does not warn about it.
- **Sphere centers sit on voxel centers.** Accuracy depends on `gridRes`.
- **No early stop in `bvh` and `brute`.** They always place exactly `targetCount` spheres. Once
  the grid is full, the remaining spheres have radius zero. `cubql` and `multi2` stop when no
  space is left.
- **Uniform grid.** Memory grows with `gridRes³`, including empty parts of the bounding box. This
  limits the resolution for long, thin meshes.
- **`gridRes` is limited to 1290**, because voxel indices are 32-bit.
- **Only OBJ files** are supported.

<!--## Citation

```bibtex
TODO
```-->

## License

This software is released under the MIT License. See [LICENSE](LICENSE) for details.

This project includes [cuBQL](https://github.com/NVIDIA/cuBQL) (Apache License 2.0) as a git
submodule and [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) (MIT License) as a
single header file.
