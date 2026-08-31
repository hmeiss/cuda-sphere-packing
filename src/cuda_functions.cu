#include "cuda_functions.h"
#include "lbvh.h"
#include "helper_math.h"
#include "cuda_check.h"
#include <limits>
#include <bit>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <chrono>

#include <cub/cub.cuh>

__device__ __forceinline__ float3 closestPointOnTriangle(const float3& p,
                                                         const float3& a,
                                                         const float3& b,
                                                         const float3& c)
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;

    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float3 bc = c - b;
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * bc;
    }

    // Inside face region
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + v * ab + w * ac;
}

__device__ float distanceFromTriangle(float3 p, float3 v0, float3 v1, float3 v2)
{
    float3 ab = v1 - v0;
    float3 ac = v2 - v0;

    float3 n = cross(ab, ac);
    float nlen = length(n);
    if (nlen < 1e-12f) {
        // Degenerate triangle: ignore it by returning huge distance
        return 1e9f;
    }
    float3 cp = closestPointOnTriangle(p, v0, v1, v2);
    return length(p - cp);
}


// Kernel to initialize the SDF grid
__global__ void initSDFKernel(SDFGrid grid, const Triangle* d_triangles, int numTriangles) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) return;

    float3 voxelCenter = grid.getVoxelCenter(x, y, z);
    float minDist = 1e9f;  // Large positive value instead of std::numeric_limits
    
    for (int i = 0; i < numTriangles; ++i) {
        float dist = distanceFromTriangle(voxelCenter, d_triangles[i].v0, d_triangles[i].v1, d_triangles[i].v2);
        if (dist < minDist) {
            minDist = dist;
        }
    }
    grid.setDistance(x, y, z, minDist);
}

// Kernel to apply odd-even sign via brute-force ray casting against all triangles
__global__ void applyBruteRaySignKernel(SDFGrid grid, const Triangle* d_triangles, int numTriangles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = grid.dimX * grid.dimY * grid.dimZ;
    if (idx >= N) return;

    int z = idx / (grid.dimX * grid.dimY);
    int y = (idx / grid.dimX) % grid.dimY;
    int x = idx % grid.dimX;

    float3 p = grid.getVoxelCenter(x, y, z);
    float3 origin = p;

    int hits = 0;
    bool degenerate = false;
    for (int i = 0; i < numTriangles; ++i) {
        int res = ray_intersect_triangle_x_classify(origin, d_triangles[i]);
        if (res == 1) {
            ++hits;
        } else if (res == 2) {
            degenerate = true;
        }
    }

    if (degenerate) {
        // Deterministic epsilon shift and retry once.
        float3 origin_shifted = make_float3(p.x, p.y + 1e-6f * 37.0f, p.z + 1e-6f * 53.0f);
        hits = 0;
        for (int i = 0; i < numTriangles; ++i) {
            int res = ray_intersect_triangle_x_classify(origin_shifted, d_triangles[i]);
            if (res == 1) {
                ++hits;
            }
        }
    }

    // Even => outside (negative)
    if ((hits & 1 ) == 0) {
        grid.data[idx] = -grid.data[idx];
    }
}

__global__ void applyBVHRaySignKernel(SDFGrid grid, BVH bvh)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = grid.dimX * grid.dimY * grid.dimZ;
    if (idx >= N) return;

    int z = idx / (grid.dimX * grid.dimY);
    int y = (idx / grid.dimX) % grid.dimY;
    int x = idx % grid.dimX;

    float3 p = grid.getVoxelCenter(x, y, z);

    RayCountResult r = bvh_count_ray_hits_x(bvh, p);

    if (r.degenerate) {
        float3 origin_shifted = make_float3(p.x, p.y + 1e-6f * 37.0f, p.z + 1e-6f * 53.0f);
        // float3 origin_shifted = make_float3(
        //     p.x,
        //     nextafterf(p.y, p.y + 1.0f),   // +1 ulp-ish toward +inf
        //     nextafterf(p.z, p.z - 1.0f)    // toward -inf (different direction helps)
        // );
        r = bvh_count_ray_hits_x(bvh, origin_shifted);
    }

    if (r.degenerate) { printf("MISSED DEGNERATE CASE r.hits & 1 = %d\n", r.hits & 1); }

    // Even => outside (negative)
    if ((r.hits & 1 ) == 0) {
        grid.data[idx] = -grid.data[idx];
    }
}

// BVH-accelerated kernel
__global__ void initSDFKernel_BVH(SDFGrid grid, BVH bvh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) return;

    float3 voxel_center = grid.getVoxelCenter(x, y, z);
    
    // Use BVH to find closest point on mesh
    DistanceHit dh = traverse_for_distance(bvh, voxel_center, 1e9f);
    
    // Store unsigned distance; sign will be applied in a scanline pass.
    float unsigned_distance = dh.hit ? dh.distance : 1e9f;
    grid.setDistance(x, y, z, unsigned_distance);
}

void initSDF_BVH(SDFGrid& grid, const BVH& bvh) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // BVH traversal uses more registers/local memory; keep block small
    dim3 blockSize(8, 8, 2);
    dim3 gridSize(
        (grid.dimX + blockSize.x - 1) / blockSize.x,
        (grid.dimY + blockSize.y - 1) / blockSize.y,
        (grid.dimZ + blockSize.z - 1) / blockSize.z
    );
    CUDA_CHECK(cudaEventRecord(start));
    initSDFKernel_BVH<<<gridSize, blockSize>>>(grid, bvh);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("initSDFKernel_BVH: %.3f ms\n", ms);

    int N = grid.dimX * grid.dimY * grid.dimZ;
    int block = 256;
    int blocks = (N + block - 1) / block;

    CUDA_CHECK(cudaEventRecord(start));
    applyBVHRaySignKernel<<<blocks, block>>>(grid, bvh);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("applyBVHRaySignKernel: %.3f ms\n", ms);
}

void initSDF_Brute(SDFGrid& grid, const Triangle* d_triangles, int numTriangles) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 blockSize(8, 8, 2);
    dim3 gridSize(
        (grid.dimX + blockSize.x - 1) / blockSize.x,
        (grid.dimY + blockSize.y - 1) / blockSize.y,
        (grid.dimZ + blockSize.z - 1) / blockSize.z
    );

    CUDA_CHECK(cudaEventRecord(start));
    initSDFKernel<<<gridSize, blockSize>>>(grid, d_triangles, numTriangles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("initSDFKernel_Unsigned: %.3f ms\n", ms);

    int N = grid.dimX * grid.dimY * grid.dimZ;
    int block = 256;
    int blocks = (N + block - 1) / block;

    CUDA_CHECK(cudaEventRecord(start));
    applyBruteRaySignKernel<<<blocks, block>>>(grid, d_triangles, numTriangles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("applyBruteRaySignKernel: %.3f ms\n", ms);
}

// Kernel to update the SDF grid after placing a new sphere
__global__ void updateSDFKernel(SDFGrid grid, Sphere newSphere) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) return;

    float3 voxelCenter = grid.getVoxelCenter(x, y, z);
    float currentDist = grid.getDistance(x, y, z);

    float distToSphereCenter = length(voxelCenter - newSphere.center);
    float newDist = distToSphereCenter - newSphere.radius;

    // Only update if the new sphere is closer and inside the current SDF
    if (newDist < currentDist) {
        grid.setDistance(x, y, z, newDist);
    }
}

void updateSDF(SDFGrid& grid, const Sphere& newSphere) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 blockSize(16, 16, 4);
    dim3 gridSize(
        (grid.dimX + blockSize.x - 1) / blockSize.x,
        (grid.dimY + blockSize.y - 1) / blockSize.y,
        (grid.dimZ + blockSize.z - 1) / blockSize.z
    );

    CUDA_CHECK(cudaEventRecord(start));
    updateSDFKernel<<<gridSize, blockSize>>>(grid, newSphere);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("updateSDFKernel: %.3f ms\n", ms);
}



__global__ void updateSDFKernel_AABB(
    SDFGrid grid,
    float3 center,
    float  radius,
    int x0, int y0, int z0,
    int sx, int sy, int sz
) {
    int lx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int ly = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int lz = (int)(blockIdx.z * blockDim.z + threadIdx.z);

    if (lx >= sx || ly >= sy || lz >= sz) return;

    int x = x0 + lx;
    int y = y0 + ly;
    int z = z0 + lz;

    // Compute voxel center quickly (no function calls)
    float px = grid.bounds.min.x + (x + 0.5f) * grid.cellSize.x;
    float py = grid.bounds.min.y + (y + 0.5f) * grid.cellSize.y;
    float pz = grid.bounds.min.z + (z + 0.5f) * grid.cellSize.z;

    float dx = px - center.x;
    float dy = py - center.y;
    float dz = pz - center.z;

    float dist2 = dx*dx + dy*dy + dz*dz;

    float updateR2 = 2*radius * 2*radius;
    if (dist2 > updateR2) return;

    int idx = x + y * grid.dimX + z * grid.dimX * grid.dimY;
    float cur = grid.data[idx];

    // compute signed distance to this sphere (negative inside)
    float newDist = sqrtf(dist2) - radius;

    // union update: inside becomes negative, outside only decreases if closer
    if (newDist < cur) grid.data[idx] = newDist;
}

// Convert world coordinate range to voxel indices whose centers lie within [wmin, wmax]
static inline int idxMinForCenter(float wmin, float gridMin, float cell) {
    // center = gridMin + (i + 0.5)*cell  >= wmin
    // i >= (wmin - gridMin)/cell - 0.5
    return (int)ceilf((wmin - gridMin) / cell - 0.5f);
}
static inline int idxMaxForCenter(float wmax, float gridMin, float cell) {
    // center <= wmax
    // i <= (wmax - gridMin)/cell - 0.5
    return (int)floorf((wmax - gridMin) / cell - 0.5f);
}

void updateSDF_SphereAABB(SDFGrid& grid, const Sphere& s) {
    const float R = 2 * s.radius;

    float3 wmin = make_float3(s.center.x - R, s.center.y - R, s.center.z - R);
    float3 wmax = make_float3(s.center.x + R, s.center.y + R, s.center.z + R);

    int x0 = idxMinForCenter(wmin.x, grid.bounds.min.x, grid.cellSize.x);
    int y0 = idxMinForCenter(wmin.y, grid.bounds.min.y, grid.cellSize.y);
    int z0 = idxMinForCenter(wmin.z, grid.bounds.min.z, grid.cellSize.z);

    int x1 = idxMaxForCenter(wmax.x, grid.bounds.min.x, grid.cellSize.x);
    int y1 = idxMaxForCenter(wmax.y, grid.bounds.min.y, grid.cellSize.y);
    int z1 = idxMaxForCenter(wmax.z, grid.bounds.min.z, grid.cellSize.z);

    // Clamp to grid
    x0 = max(0, min(x0, grid.dimX - 1));
    y0 = max(0, min(y0, grid.dimY - 1));
    z0 = max(0, min(z0, grid.dimZ - 1));

    x1 = max(0, min(x1, grid.dimX - 1));
    y1 = max(0, min(y1, grid.dimY - 1));
    z1 = max(0, min(z1, grid.dimZ - 1));

    if (x1 < x0 || y1 < y0 || z1 < z0) return; // sphere AABB doesn't touch grid

    int sx = x1 - x0 + 1;
    int sy = y1 - y0 + 1;
    int sz = z1 - z0 + 1;

    // Good memory-coalescing on X, reasonable occupancy
    dim3 block(32, 4, 2); // 256 threads
    dim3 gridDim((sx + block.x - 1) / block.x,
                 (sy + block.y - 1) / block.y,
                 (sz + block.z - 1) / block.z);

    updateSDFKernel_AABB<<<gridDim, block>>>(grid, s.center, s.radius, x0, y0, z0, sx, sy, sz);
    CUDA_CHECK(cudaGetLastError());
}



struct PosFiniteTransform {
    const float* data;
    __device__ float operator()(int i) const {
        float d = data[i];
        return (d > 0.0f) ? d : __int_as_float(0xff800000 ); // -INF
    }
};

void findMaxDistanceVoxel(const SDFGrid& grid, float& maxDist, float3& maxDistPos)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // Use 64-bit for N to avoid overflow when dims are large
    const int64_t N64 = (int64_t)grid.dimX * grid.dimY * grid.dimZ;
    if (N64 <= 0) { maxDist = 0.0f; maxDistPos = make_float3(0,0,0); return; }

    // CUB ArgMax supports different offset types; choose 64-bit if needed
    using OffsetT = int; // change to int64_t if N64 can exceed INT_MAX
    const OffsetT N = (OffsetT)N64;

    using PairT = cub::KeyValuePair<OffsetT, float>;

    // Counting iterator for indices [0..N)
    cub::CountingInputIterator<OffsetT> counting(0);
    PosFiniteTransform op{ grid.data };

    // Transform iterator that maps index -> value (or -INF if invalid)
    using ItT = cub::TransformInputIterator<float, PosFiniteTransform, decltype(counting)>;
    ItT values(counting, op);

    // Output pair: (argmax_index, max_value)
    PairT* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(PairT));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    cub::DeviceReduce::ArgMax(d_temp, temp_bytes, values, d_out, N);
    cudaMalloc(&d_temp, temp_bytes);
    CUDA_CHECK(cudaEventRecord(start));
    cub::DeviceReduce::ArgMax(d_temp, temp_bytes, values, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    // printf("findMaxPackedKernel: %.3f ms\n", ms);

    PairT h;
    cudaMemcpy(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost);

    maxDist = h.value;

    if (maxDist > 0.0f) {
        const OffsetT idx = h.key;
        int z = (int)(idx / (grid.dimX * grid.dimY));
        int y = (int)((idx / grid.dimX) % grid.dimY);
        int x = (int)(idx % grid.dimX);
        maxDistPos = grid.getVoxelCenter(x, y, z);

    } else {
        maxDist = 0.0f;
        maxDistPos = make_float3(0,0,0);
    }

    cudaFree(d_temp);
    cudaFree(d_out);
}

namespace {

constexpr int kBinMin = -32;
constexpr int kBinMax = 32;
constexpr int kBinShift = -kBinMin;
constexpr int kBinCount = (kBinMax - kBinMin + 1);
constexpr int kMulti2BatchSize = 1000;
constexpr float kPriorityEps = 1e-7f;

__host__ __device__ __forceinline__ uint32_t hash32_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ void decodeVoxelIndex(int idx, int dimX, int dimY, int& x, int& y, int& z) {
    z = idx / (dimX * dimY);
    y = (idx / dimX) % dimY;
    x = idx % dimX;
}

__device__ __forceinline__ float3 voxelCenterFromLinear(int idx, const SDFGrid& grid) {
    int x, y, z;
    decodeVoxelIndex(idx, grid.dimX, grid.dimY, x, y, z);
    return make_float3(grid.bounds.min.x + (x + 0.5f) * grid.cellSize.x,
                       grid.bounds.min.y + (y + 0.5f) * grid.cellSize.y,
                       grid.bounds.min.z + (z + 0.5f) * grid.cellSize.z);
}

__host__ __forceinline__ float3 voxelCenterFromLinearHost(int idx, const SDFGrid& grid) {
    const int z = idx / (grid.dimX * grid.dimY);
    const int y = (idx / grid.dimX) % grid.dimY;
    const int x = idx % grid.dimX;
    return make_float3(grid.bounds.min.x + (x + 0.5f) * grid.cellSize.x,
                       grid.bounds.min.y + (y + 0.5f) * grid.cellSize.y,
                       grid.bounds.min.z + (z + 0.5f) * grid.cellSize.z);
}

__device__ __forceinline__ bool overlapsExact(int voxA, float rA, int voxB, float rB, const SDFGrid& grid) {
    const float3 cA = voxelCenterFromLinear(voxA, grid);
    const float3 cB = voxelCenterFromLinear(voxB, grid);
    const float3 d = cA - cB;
    const float dist2 = d.x * d.x + d.y * d.y + d.z * d.z;
    const float rr = rA + rB;
    return dist2 < (rr * rr);
}

__host__ __forceinline__ bool overlapsExactHost(const Sphere& a, const Sphere& b) {
    const float3 d = a.center - b.center;
    const float dist2 = d.x * d.x + d.y * d.y + d.z * d.z;
    const float rr = a.radius + b.radius;
    return dist2 < (rr * rr);
}

__device__ __forceinline__ bool priorityGreater(int voxA, float rA, int voxB, float rB) {
    if (rA > rB + kPriorityEps) return true;
    if (rB > rA + kPriorityEps) return false;
    const uint32_t hA = hash32_u32((uint32_t)voxA);
    const uint32_t hB = hash32_u32((uint32_t)voxB);
    if (hA != hB) return hA > hB;
    return voxA > voxB;
}

__host__ __forceinline__ float radiusFromMulti2SortKey(unsigned long long key) {
    const uint32_t bits = (uint32_t)(key >> 32);
    return std::bit_cast<float>(bits);
}

__host__ __forceinline__ float multi2RadiusSafety(float radius, const SDFGrid& grid) {
    const float ulpPad = 8.0f * std::numeric_limits<float>::epsilon() * fmaxf(1.0f, radius);
    const float voxelPad = 1e-5f * grid.cellSize.x;
    return fmaxf(ulpPad, voxelPad);
}

__device__ __forceinline__ int lowerBoundInt(const int* arr, int n, int key) {
    int l = 0;
    int r = n;
    while (l < r) {
        int m = l + ((r - l) >> 1);
        if (arr[m] < key) l = m + 1;
        else r = m;
    }
    return l;
}

__device__ __forceinline__ int upperBoundInt(const int* arr, int n, int key) {
    int l = 0;
    int r = n;
    while (l < r) {
        int m = l + ((r - l) >> 1);
        if (arr[m] <= key) l = m + 1;
        else r = m;
    }
    return l;
}

__global__ void flagPositiveKernel(const float* sdf, int N, uint8_t* flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    flags[idx] = (sdf[idx] > 0.0f) ? 1 : 0;
}

__global__ void flagLocalMax26Kernel(SDFGrid grid, uint8_t* flags) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) return;

    const int idx = x + y * grid.dimX + z * grid.dimX * grid.dimY;
    const float v = grid.data[idx];
    if (v <= 0.0f) {
        flags[idx] = 0;
        return;
    }

    bool isMax = true;
    for (int dz = -1; dz <= 1 && isMax; ++dz) {
        for (int dy = -1; dy <= 1 && isMax; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                const int nx = x + dx;
                const int ny = y + dy;
                const int nz = z + dz;
                if (nx < 0 || ny < 0 || nz < 0 ||
                    nx >= grid.dimX || ny >= grid.dimY || nz >= grid.dimZ) {
                    continue;
                }
                const int nidx = nx + ny * grid.dimX + nz * grid.dimX * grid.dimY;
                const float nv = grid.data[nidx];
                if (nv > v + kPriorityEps || (fabsf(nv - v) <= kPriorityEps && nidx < idx)) {
                    isMax = false;
                    break;
                }
            }
        }
    }

    flags[idx] = isMax ? 1 : 0;
}


__global__ void buildMulti2SortKeysKernel(SDFGrid grid, const int* voxels, int count, unsigned long long* sortKeys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const int vox = voxels[i];
    const float r = grid.data[vox];
    if (r <= 0.0f) {
        sortKeys[i] = 0ULL;
        return;
    }

    const unsigned long long radiusKey = (unsigned long long)__float_as_uint(r);
    const unsigned long long hashKey = (unsigned long long)hash32_u32((uint32_t)vox);
    sortKeys[i] = (radiusKey << 32) | hashKey;
}

__global__ void updateMulti2CandidatesAgainstBatchKernel(
    SDFGrid grid,
    const int* activeVoxels,
    int activeCount,
    const Sphere* placedSpheres,
    int placedCount,
    uint8_t* keepFlags
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= activeCount) return;

    const int vox = activeVoxels[i];
    float r = grid.data[vox];
    if (r <= 0.0f) {
        keepFlags[i] = 0;
        return;
    }

    const float3 c = voxelCenterFromLinear(vox, grid);
    for (int s = 0; s < placedCount; ++s) {
        const Sphere placed = placedSpheres[s];
        const float3 d = c - placed.center;
        const float dist2 = d.x * d.x + d.y * d.y + d.z * d.z;
        const float limit = r + placed.radius;
        if (dist2 < limit * limit) {
            const float newR = sqrtf(dist2) - placed.radius;
            if (newR < r) {
                r = newR;
            }
        }
    }

    grid.data[vox] = r;
    keepFlags[i] = (r > 0.0f) ? 1 : 0;
}

bool isCudaOOM(cudaError_t err) {
    return err == cudaErrorMemoryAllocation;
}

bool tryCudaMalloc(void** ptr, size_t bytes, bool& sawOOM) {
    *ptr = nullptr;
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err == cudaSuccess) return true;
    if (isCudaOOM(err)) sawOOM = true;
    return false;
}

bool runDeviceSelectFlagged(
    cub::CountingInputIterator<int> counting,
    const uint8_t* d_flags,
    int* d_out,
    int* d_numSelected,
    int N,
    bool& sawOOM
) {
    void* d_temp = nullptr;
    size_t tempBytes = 0;
    cub::DeviceSelect::Flagged(d_temp, tempBytes, counting, d_flags, d_out, d_numSelected, N);
    if (!tryCudaMalloc(&d_temp, tempBytes, sawOOM)) return false;
    cub::DeviceSelect::Flagged(d_temp, tempBytes, counting, d_flags, d_out, d_numSelected, N);
    cudaError_t err = cudaGetLastError();
    cudaFree(d_temp);
    return err == cudaSuccess;
}

bool runDeviceSelectFlagged(
    const int* d_in,
    const uint8_t* d_flags,
    int* d_out,
    int* d_numSelected,
    int N,
    bool& sawOOM
) {
    void* d_temp = nullptr;
    size_t tempBytes = 0;
    cub::DeviceSelect::Flagged(d_temp, tempBytes, d_in, d_flags, d_out, d_numSelected, N);
    if (!tryCudaMalloc(&d_temp, tempBytes, sawOOM)) return false;
    cub::DeviceSelect::Flagged(d_temp, tempBytes, d_in, d_flags, d_out, d_numSelected, N);
    cudaError_t err = cudaGetLastError();
    cudaFree(d_temp);
    return err == cudaSuccess;
}

bool runDeviceSortPairsDescending(
    const unsigned long long* d_keysIn,
    unsigned long long* d_keysOut,
    const int* d_valsIn,
    int* d_valsOut,
    int N,
    bool& sawOOM
) {
    void* d_temp = nullptr;
    size_t tempBytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp, tempBytes, d_keysIn, d_keysOut, d_valsIn, d_valsOut, N);
    if (!tryCudaMalloc(&d_temp, tempBytes, sawOOM)) return false;
    cub::DeviceRadixSort::SortPairsDescending(d_temp, tempBytes, d_keysIn, d_keysOut, d_valsIn, d_valsOut, N);
    cudaError_t err = cudaGetLastError();
    cudaFree(d_temp);
    return err == cudaSuccess;
}

bool buildCandidateVoxelList(
    const SDFGrid& grid,
    bool localMaxOnly,
    int*& d_voxelsOut,
    int& candidateCountOut,
    bool& sawOOM
) {
    d_voxelsOut = nullptr;
    candidateCountOut = 0;

    const int64_t N64 = (int64_t)grid.dimX * grid.dimY * grid.dimZ;
    if (N64 <= 0 || N64 > INT_MAX) {
        return false;
    }
    const int N = (int)N64;

    uint8_t* d_flags = nullptr;
    int* d_voxelsBig = nullptr;
    int* d_numSelected = nullptr;
    cub::CountingInputIterator<int> counting(0);
    bool ok = true;

    if (!tryCudaMalloc((void**)&d_flags, (size_t)N * sizeof(uint8_t), sawOOM)) {
        ok = false;
        goto cleanup;
    }

    if (localMaxOnly) {
        dim3 block(8, 8, 2);
        dim3 gridDim((grid.dimX + block.x - 1) / block.x,
                     (grid.dimY + block.y - 1) / block.y,
                     (grid.dimZ + block.z - 1) / block.z);
        flagLocalMax26Kernel<<<gridDim, block>>>(grid, d_flags);
    } else {
        int block = 256;
        int gridDim = (N + block - 1) / block;
        flagPositiveKernel<<<gridDim, block>>>(grid.data, N, d_flags);
    }

    if (cudaGetLastError() != cudaSuccess) {
        ok = false;
        goto cleanup;
    }

    if (!tryCudaMalloc((void**)&d_voxelsBig, (size_t)N * sizeof(int), sawOOM)) {
        ok = false;
        goto cleanup;
    }
    if (!tryCudaMalloc((void**)&d_numSelected, sizeof(int), sawOOM)) {
        ok = false;
        goto cleanup;
    }

    if (!runDeviceSelectFlagged(counting, d_flags, d_voxelsBig, d_numSelected, N, sawOOM)) {
        ok = false;
        goto cleanup;
    }

    if (cudaMemcpy(&candidateCountOut, d_numSelected, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        ok = false;
        goto cleanup;
    }

    if (candidateCountOut <= 0) {
        d_voxelsOut = nullptr;
        candidateCountOut = 0;
        ok = true;
        goto cleanup;
    }

    {
        int* d_voxelsTight = nullptr;
        if (tryCudaMalloc((void**)&d_voxelsTight, (size_t)candidateCountOut * sizeof(int), sawOOM)) {
            if (cudaMemcpy(d_voxelsTight, d_voxelsBig, (size_t)candidateCountOut * sizeof(int), cudaMemcpyDeviceToDevice) == cudaSuccess) {
                d_voxelsOut = d_voxelsTight;
            } else {
                cudaFree(d_voxelsTight);
                d_voxelsOut = d_voxelsBig;
                d_voxelsBig = nullptr;
            }
        } else {
            d_voxelsOut = d_voxelsBig;
            d_voxelsBig = nullptr;
        }
    }

cleanup:
    if (d_flags) cudaFree(d_flags);
    if (d_numSelected) cudaFree(d_numSelected);
    if (d_voxelsBig) cudaFree(d_voxelsBig);
    return ok;
}

bool packSpheresMulti2SnapshotImpl(
    const SDFGrid& grid,
    int targetSphereCount,
    std::vector<Sphere>& outSpheres,
    bool& sawOOM,
    std::vector<PackingProgressSample>* progressSamples,
    double elapsedOffsetSeconds
) {
    outSpheres.clear();
    sawOOM = false;
    const auto packStart = std::chrono::steady_clock::now();

    if (targetSphereCount <= 0) {
        return true;
    }

    int* d_candidateVoxels = nullptr;
    int candidateCount = 0;
    if (!buildCandidateVoxelList(grid, false, d_candidateVoxels, candidateCount, sawOOM)) {
        if (d_candidateVoxels) cudaFree(d_candidateVoxels);
        return false;
    }
    if (candidateCount <= 0) {
        return true;
    }

    bool ok = true;
    int* d_activeVoxels = nullptr;
    int* d_sortedVoxels = nullptr;
    int* d_activeNext = nullptr;
    unsigned long long* d_sortKeys = nullptr;
    unsigned long long* d_sortKeysSorted = nullptr;
    uint8_t* d_keepFlags = nullptr;
    int* d_activeNextCount = nullptr;
    Sphere* d_batchSpheres = nullptr;
    std::vector<int> windowVoxelsHost;
    std::vector<unsigned long long> windowKeysHost;
    std::vector<Sphere> acceptedBatchHost;
    int activeCount = candidateCount;
    int iterations = 0;

    d_activeVoxels = d_candidateVoxels;
    d_candidateVoxels = nullptr;
    if (!tryCudaMalloc((void**)&d_sortedVoxels, (size_t)candidateCount * sizeof(int), sawOOM)) { ok = false; goto cleanup; }
    if (!tryCudaMalloc((void**)&d_activeNext, (size_t)candidateCount * sizeof(int), sawOOM)) { ok = false; goto cleanup; }
    if (!tryCudaMalloc((void**)&d_sortKeys, (size_t)candidateCount * sizeof(unsigned long long), sawOOM)) { ok = false; goto cleanup; }
    if (!tryCudaMalloc((void**)&d_sortKeysSorted, (size_t)candidateCount * sizeof(unsigned long long), sawOOM)) { ok = false; goto cleanup; }
    if (!tryCudaMalloc((void**)&d_keepFlags, (size_t)candidateCount * sizeof(uint8_t), sawOOM)) { ok = false; goto cleanup; }
    if (!tryCudaMalloc((void**)&d_activeNextCount, sizeof(int), sawOOM)) { ok = false; goto cleanup; }
    if (!tryCudaMalloc((void**)&d_batchSpheres, (size_t)kMulti2BatchSize * sizeof(Sphere), sawOOM)) { ok = false; goto cleanup; }

    outSpheres.reserve((size_t)targetSphereCount);
    windowVoxelsHost.reserve((size_t)kMulti2BatchSize);
    windowKeysHost.reserve((size_t)kMulti2BatchSize);
    acceptedBatchHost.reserve((size_t)kMulti2BatchSize);

    while (activeCount > 0 && (int)outSpheres.size() < targetSphereCount) {
        ++iterations;

        {
            const int block = 256;
            const int gridDim = (activeCount + block - 1) / block;
            buildMulti2SortKeysKernel<<<gridDim, block>>>(grid, d_activeVoxels, activeCount, d_sortKeys);
            if (cudaGetLastError() != cudaSuccess) { ok = false; goto cleanup; }
        }

        if (!runDeviceSortPairsDescending(
                d_sortKeys, d_sortKeysSorted, d_activeVoxels, d_sortedVoxels, activeCount, sawOOM)) {
            ok = false;
            goto cleanup;
        }

        const int windowCount = min(kMulti2BatchSize, activeCount);
        windowVoxelsHost.resize((size_t)windowCount);
        windowKeysHost.resize((size_t)windowCount);
        if (cudaMemcpy(windowVoxelsHost.data(), d_sortedVoxels, (size_t)windowCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(windowKeysHost.data(), d_sortKeysSorted, (size_t)windowCount * sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
            ok = false;
            goto cleanup;
        }

        acceptedBatchHost.clear();
        for (int i = 0; i < windowCount && (int)outSpheres.size() + (int)acceptedBatchHost.size() < targetSphereCount; ++i) {
            const float radius = radiusFromMulti2SortKey(windowKeysHost[(size_t)i]);
            if (radius <= 0.0f) {
                continue;
            }

            Sphere candidate;
            candidate.center = voxelCenterFromLinearHost(windowVoxelsHost[(size_t)i], grid);
            candidate.radius = fmaxf(0.0f, radius - multi2RadiusSafety(radius, grid));
            if (candidate.radius <= 0.0f) {
                continue;
            }

            bool overlapsAccepted = false;
            for (const Sphere& accepted : acceptedBatchHost) {
                if (overlapsExactHost(candidate, accepted)) {
                    overlapsAccepted = true;
                    break;
                }
            }

            if (!overlapsAccepted) {
                acceptedBatchHost.push_back(candidate);
            }
        }

        if (acceptedBatchHost.empty()) {
            goto cleanup;
        }

        if (cudaMemcpy(d_batchSpheres, acceptedBatchHost.data(), acceptedBatchHost.size() * sizeof(Sphere), cudaMemcpyHostToDevice) != cudaSuccess) {
            ok = false;
            goto cleanup;
        }

        {
            const int block = 256;
            const int gridDim = (activeCount + block - 1) / block;
            updateMulti2CandidatesAgainstBatchKernel<<<gridDim, block>>>(
                grid,
                d_sortedVoxels,
                activeCount,
                d_batchSpheres,
                (int)acceptedBatchHost.size(),
                d_keepFlags);
            if (cudaGetLastError() != cudaSuccess) { ok = false; goto cleanup; }
        }

        if (!runDeviceSelectFlagged(
                d_sortedVoxels, d_keepFlags, d_activeNext, d_activeNextCount, activeCount, sawOOM)) {
            ok = false;
            goto cleanup;
        }
        if (cudaMemcpy(&activeCount, d_activeNextCount, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            ok = false;
            goto cleanup;
        }
        std::swap(d_activeVoxels, d_activeNext);

        outSpheres.insert(outSpheres.end(), acceptedBatchHost.begin(), acceptedBatchHost.end());
        if (progressSamples != nullptr) {
            const double elapsedSeconds =
                elapsedOffsetSeconds +
                std::chrono::duration<double>(std::chrono::steady_clock::now() - packStart).count();
            progressSamples->push_back({(int)outSpheres.size(), elapsedSeconds});
        }
    }

    std::cout << "Multi2 candidates: " << candidateCount
              << ", iterations: " << iterations
              << ", selected spheres: " << outSpheres.size() << std::endl;

cleanup:
    if (d_candidateVoxels) cudaFree(d_candidateVoxels);
    if (d_activeVoxels) cudaFree(d_activeVoxels);
    if (d_sortedVoxels) cudaFree(d_sortedVoxels);
    if (d_activeNext) cudaFree(d_activeNext);
    if (d_sortKeys) cudaFree(d_sortKeys);
    if (d_sortKeysSorted) cudaFree(d_sortKeysSorted);
    if (d_keepFlags) cudaFree(d_keepFlags);
    if (d_activeNextCount) cudaFree(d_activeNextCount);
    if (d_batchSpheres) cudaFree(d_batchSpheres);
    return ok;
}

} // namespace


bool packSpheresMulti2Snapshot(
    const SDFGrid& grid,
    int targetSphereCount,
    std::vector<Sphere>& outSpheres,
    std::vector<PackingProgressSample>* progressSamples,
    double elapsedOffsetSeconds
) {
    bool sawOOM = false;
    const bool ok = packSpheresMulti2SnapshotImpl(grid, targetSphereCount, outSpheres, sawOOM, progressSamples, elapsedOffsetSeconds);
    if (!ok && sawOOM) {
        std::cerr << "multi2 mode failed: GPU memory limits exceeded while processing all positive SDF voxels" << std::endl;
    }
    return ok;
}
