#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>

struct Triangle {
    float3 v0, v1, v2;
};

struct BoundingBox {
    float3 min;
    float3 max;

    __host__ __device__ float3 size() const {
        return make_float3(max.x - min.x, max.y - min.y, max.z - min.z);
    }

    __host__ __device__ float3 center() const {
        return make_float3((min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f, (min.z + max.z) * 0.5f);
    }
};

class Mesh {
public:
    Mesh();
    ~Mesh();

    bool loadFromObj(const std::string& filename);
    
    const std::vector<Triangle>& getTriangles() const { return triangles; }
    const BoundingBox& getBounds() const { return bounds; }
    
    // Compute volume enclosed by the mesh using divergence theorem
    // Assumes mesh is closed and properly oriented
    double computeVolume() const;

private:
    std::vector<Triangle> triangles;
    BoundingBox bounds;

    void computeBounds();
};
