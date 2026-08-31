// cuBQL builder implementation must be instantiated in exactly one translation unit
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1

#include "cuBQL/bvh.h"
// closestPointOnAnyTriangle.h transitively pulls in shrinkingRadiusQuery.h and CUBQL_INF
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"
#include "cuBQL/queries/triangleData/pointInsideOutside.h"
// rebinRadixBuilder is declared in builder/cuda.h but its impl is NOT pulled in by
// the default macro include list — include it explicitly to instantiate it here.
#include "cuBQL/builder/cuda/rebinMortonBuilder.h"

#include "cubql_sdf.h"
#include "cuda_check.h"
#include "helper_math.h"

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

// Efficient closest-point-on-triangle using Ericson barycentric method.
// Uses 6 dot products, no cross products, no normalize/sqrt — 3-4x fewer ops
// than cuBQL's built-in computeOneIntersection (which uses 4 cross products + normalize).
__forceinline__ __device__ float3 closestPtOnTri_fast(
    float3 p, float3 a, float3 b, float3 c)
{
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.f && d2 <= 0.f) return a;
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.f && d4 <= d3) return b;
    float vc = d1*d4 - d3*d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }
    float3 cp_c = p - c;
    float d5 = dot(ab, cp_c), d6 = dot(ac, cp_c);
    if (d6 >= 0.f && d5 <= d6) return c;
    float vb = d5*d2 - d1*d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }
    float va = d3*d6 - d5*d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }
    float denom = 1.f / (va + vb + vc);
    float v = vb * denom, w = vc * denom;
    return a + v * ab + w * ac;
}

// Convert project Triangle (float3 v0,v1,v2) to cuBQL Triangle and compute its AABB.
__global__ void convertAndBoxKernel(
    cuBQL::Triangle* dst_tris,
    cuBQL::box3f*    dst_boxes,
    const Triangle*  src,
    int              n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Triangle& t = src[i];
    cuBQL::Triangle tri = {
        {t.v0.x, t.v0.y, t.v0.z},
        {t.v1.x, t.v1.y, t.v1.z},
        {t.v2.x, t.v2.y, t.v2.z}
    };
    dst_tris[i]  = tri;
    dst_boxes[i] = tri.bounds();
}

// Point-to-AABB squared distance using cuBQL types.
__forceinline__ __device__
float sqDistPtBox(cuBQL::vec3f q, cuBQL::box3f b) {
    cuBQL::vec3f pr = clamp(q, b.lower, b.upper);
    cuBQL::vec3f d  = pr - q;
    return dot(d, d);
}

// Custom inline traversal on cuBQL BVH layout.
// Differences vs forEachLeaf:
//   - Index-only stack (4 bytes/entry vs 8): 256 bytes per thread instead of 512
//   - No per-pop distance filter loop (simpler pop → less SIMT divergence)
//   - No lambda indirection overhead
__global__ void initSDFKernel_cuBQL(
    SDFGrid              grid,
    cuBQL::bvh3f         bvh,
    const cuBQL::Triangle* tris)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) return;

    float3 p  = grid.getVoxelCenter(x, y, z);
    cuBQL::vec3f qp = {p.x, p.y, p.z};

    uint32_t nodeStack[64];
    uint32_t *sp = nodeStack;
    *sp++ = (uint32_t)-1;  // sentinel: (int)(-1) < 0 → terminates loop

    float sqrDist = 1e18f;
    uint32_t cur  = 0;

    do {
        const cuBQL::bvh3f::Node& node = bvh.nodes[cur];
        const uint32_t cnt = node.admin.count;

        if (cnt > 0) {
            // Leaf — test every primitive (makeLeafThreshold=1 → cnt is almost always 1)
            const uint32_t base = node.admin.offset;
            for (uint32_t i = 0; i < cnt; i++) {
                const cuBQL::Triangle& t = tris[bvh.primIDs[base + i]];
                const float3 a = {t.a.x, t.a.y, t.a.z};
                const float3 b = {t.b.x, t.b.y, t.b.z};
                const float3 c = {t.c.x, t.c.y, t.c.z};
                float3 cp = closestPtOnTri_fast(p, a, b, c);
                float3 d  = p - cp;
                float d2  = dot(d, d);
                if (d2 < sqrDist) sqrDist = d2;
            }
            cur = *--sp;
        } else {
            // Inner node — test both children, near-first
            const uint32_t c0 = node.admin.offset;
            const uint32_t c1 = c0 + 1;
            float d0 = sqDistPtBox(qp, bvh.nodes[c0].bounds);
            float d1 = sqDistPtBox(qp, bvh.nodes[c1].bounds);
            bool h0 = (d0 < sqrDist);
            bool h1 = (d1 < sqrDist);
            if (h0 && h1) {
                if (d0 <= d1) { *sp++ = c1; cur = c0; }
                else           { *sp++ = c0; cur = c1; }
            } else if (h0) {
                cur = c0;
            } else if (h1) {
                cur = c1;
            } else {
                cur = *--sp;
            }
        }
    } while ((int)cur >= 0);

    grid.setDistance(x, y, z, sqrtf(sqrDist));
}

// For each voxel: determine inside/outside using cuBQL 6-ray majority vote,
// then negate the stored distance for outside voxels.
__global__ void applySignKernel_cuBQL(
    SDFGrid              grid,
    cuBQL::bvh3f         bvh,
    const cuBQL::Triangle* tris)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N   = grid.dimX * grid.dimY * grid.dimZ;
    if (idx >= N) return;

    int z = idx / (grid.dimX * grid.dimY);
    int y = (idx / grid.dimX) % grid.dimY;
    int x = idx % grid.dimX;
    float3 p = grid.getVoxelCenter(x, y, z);

    bool inside = cuBQL::triangles::pointIsInsideSurface(
        bvh,
        [tris](uint32_t i) { return tris[i]; },
        {p.x, p.y, p.z});

    if (!inside)
        grid.data[idx] = -fabsf(grid.data[idx]);
}

void initSDF_cuBQL(SDFGrid& grid, const std::vector<Triangle>& triangles)
{
    int n = (int)triangles.size();
    if (n == 0) return;

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    // Upload project triangles to device
    Triangle* d_src;
    CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(d_src, triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));

    // Allocate cuBQL triangle array and per-triangle AABB array
    cuBQL::Triangle* d_tris;
    cuBQL::box3f*    d_boxes;
    CUDA_CHECK(cudaMalloc(&d_tris,  n * sizeof(cuBQL::Triangle)));
    CUDA_CHECK(cudaMalloc(&d_boxes, n * sizeof(cuBQL::box3f)));

    convertAndBoxKernel<<<(n + 255) / 256, 256>>>(d_tris, d_boxes, d_src, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_src));

    // Build cuBQL BVH from per-triangle bounding boxes.
    // SAH (surface-area heuristic) is the default: it builds the highest-quality tree and
    // gives the best query performance, especially on large meshes where Morton-code
    // collisions degrade the radix tree's spatial coherence. makeLeafThreshold=1 forces
    // single-prim leaves to match the custom traversal's assumption.
    //
    // To swap builders, comment out sahBuilder and uncomment one alternative (same args):
    //   radixBuilder       - fast Morton LBVH; great on small meshes, slower on large ones
    //   rebinRadixBuilder  - radix with rebinning (needs rebinMortonBuilder.h, included above)
    //   gpuBuilder         - adaptive spatial median (note: cuBQL:: , not cuBQL::cuda::)
    std::cout << "Building cuBQL BVH over " << n << " triangles..." << std::endl;
    cuBQL::BuildConfig config;
    config.makeLeafThreshold = 1;
    cuBQL::bvh3f bvh;

    CUDA_CHECK(cudaEventRecord(evStart));
    cuBQL::cuda::sahBuilder(bvh, d_boxes, (uint32_t)n, config);
    // cuBQL::cuda::radixBuilder(bvh, d_boxes, (uint32_t)n, config);
    // cuBQL::cuda::rebinRadixBuilder(bvh, d_boxes, (uint32_t)n, config);
    // cuBQL::gpuBuilder(bvh, d_boxes, (uint32_t)n, config);
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float buildMs = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&buildMs, evStart, evStop));
    printf("cuBQL BVH build: %.3f ms, %u nodes\n", buildMs, bvh.numNodes);
    CUDA_CHECK(cudaFree(d_boxes));

    // Distance pass — 3D grid launch matching the SDF grid dimensions
    dim3 block(8, 8, 2);
    dim3 gridDim(
        (grid.dimX + block.x - 1) / block.x,
        (grid.dimY + block.y - 1) / block.y,
        (grid.dimZ + block.z - 1) / block.z);

    CUDA_CHECK(cudaEventRecord(evStart));
    initSDFKernel_cuBQL<<<gridDim, block>>>(grid, bvh, d_tris);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    printf("initSDFKernel_cuBQL: %.3f ms\n", ms);

    // Sign pass — linear 1D launch over all voxels
    int N          = grid.dimX * grid.dimY * grid.dimZ;
    int signBlocks = (N + 255) / 256;

    CUDA_CHECK(cudaEventRecord(evStart));
    applySignKernel_cuBQL<<<signBlocks, 256>>>(grid, bvh, d_tris);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    printf("applySignKernel_cuBQL: %.3f ms\n", ms);

    // Cleanup
    cuBQL::cuda::free(bvh);
    CUDA_CHECK(cudaFree(d_tris));
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
}
