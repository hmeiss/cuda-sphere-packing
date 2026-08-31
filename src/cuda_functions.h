#pragma once

#include <cuda_runtime.h>
#include "mesh.h"
#include "helper_math.h" // For float3 operations
#include "lbvh.h"
#include <vector>

// SDF grid parameters
struct SDFGrid {
    float* data;         // Device memory for SDF values
    int dimX, dimY, dimZ;
    BoundingBox bounds;  // Bounding box of the grid volume
    float3 cellSize;     // Size of each voxel

    __host__ __device__ float3 getVoxelCenter(int x, int y, int z) const {
        return make_float3(bounds.min.x + (x + 0.5f) * cellSize.x,
                           bounds.min.y + (y + 0.5f) * cellSize.y,
                           bounds.min.z + (z + 0.5f) * cellSize.z);
    }

    __host__ __device__ int getIndex(int x, int y, int z) const {
        if (x < 0 || x >= dimX ||
            y < 0 || y >= dimY ||
            z < 0 || z >= dimZ) {
            return -1; // Out of bounds
        }
        return x + y * dimX + z * dimX * dimY;
    }

    __host__ __device__ float getDistance(int x, int y, int z) const {
        int idx = getIndex(x, y, z);
        if (idx != -1) {
            return data[idx];
        }
        return -1e9; // Arbitrarily large negative for out of bounds
    }

    __host__ __device__ void setDistance(int x, int y, int z, float dist) {
        int idx = getIndex(x, y, z);
        if (idx != -1) {
            data[idx] = dist;
        }
    }
};

// Sphere structure for output
struct Sphere {
    float3 center;
    float radius;
};

struct PackingProgressSample {
    int sphereCount;
    double elapsedSeconds;
};

// CUDA function declarations
void initSDF_BVH(SDFGrid& grid, const BVH& bvh);
void initSDF_Brute(SDFGrid& grid, const Triangle* d_triangles, int numTriangles);
void updateSDF(SDFGrid& grid, const Sphere& newSphere);
void findMaxDistanceVoxel(const SDFGrid& grid, float& maxDist, float3& maxDistPos);
void updateSDF_SphereAABB(SDFGrid& grid, const Sphere& newSphere);
bool packSpheresMulti2Snapshot(const SDFGrid& grid, int targetSphereCount, std::vector<Sphere>& outSpheres, std::vector<PackingProgressSample>* progressSamples = nullptr, double elapsedOffsetSeconds = 0.0);
