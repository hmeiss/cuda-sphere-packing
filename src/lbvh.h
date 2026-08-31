#pragma once

#include <cuda_runtime.h>
#include "mesh.h"

struct Hit {
    float3 hitpoint;
    /// Whether a hit is recorded.
    int hit;
};

struct BVHNode {
    /// If nullptr, then this node is a leaf.
    BVHNode *left;
    BVHNode *right;
    BVHNode *parent;
    /** Bounding box. */
    float3 min;
    float3 max;

    /// If leaf, holds the id of the object.
    unsigned int object_id;
    /// If internal node, holds whether the node has been visited once
    /// while setting bounding boxes. The first thread (child) sets
    /// it equal to its own bounding box and continues up the tree.
    /// The second thread (child) sets it to their union and terminates.
    unsigned int visited;

    __forceinline__ __device__ bool is_leaf() const
    { return left == nullptr; }
};

struct BVH {
    /// Leaf nodes, one for every photon data element.
    BVHNode *leaf_nodes;
    /// Internal nodes, amount equal to #leaf nodes - 1.
    BVHNode *internal_nodes;

    // triangle data on device
    Triangle* d_triangles;
};

int build(BVH *bvh, std::vector<Triangle>& triangles, float3 sceneBoundMin, float3 sceneBoundMax);

void clean(BVH *bvh);

// Distance query structures
struct DistanceHit {
    float distance;      // Unsigned distance to closest point on triangle
    int object_id;       // Triangle ID
    float3 closest_point; // Closest point on the triangle
    bool hit;
};

struct RayHit {
    float t;             // Ray parameter (distance along ray)
    int object_id;       // Triangle ID
    float3 hitpoint;     // Hit point in 3D
    bool hit;
};

// Device functions for BVH traversal (device-only, called from kernels)
// These are implemented in lbvh.cu
__device__ DistanceHit traverse_for_distance(BVH bvh, float3 point, float max_dist);


__device__ int ray_intersect_triangle_x_classify(
        const float3& origin, const Triangle& tri);

struct RayCountResult {
    int hits;
    bool degenerate;
};
__device__ RayCountResult bvh_count_ray_hits_x(
    const BVH& bvh,
    const float3& origin);

