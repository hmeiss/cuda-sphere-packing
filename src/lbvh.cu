
#include "lbvh.h"
#include "cuda_check.h"
#include "helper_math.h"
#include <cub/device/device_radix_sort.cuh>
#include "cuda_check.h"

/// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__forceinline__ __device__ unsigned int expand_bits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/// Calculates a 30-bit Morton code for the given 3D point located
/// within the unit cube [0,1].
__forceinline__ __device__ unsigned int morton_3d(float x, float y, float z)
{
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expand_bits((unsigned int) x);
    unsigned int yy = expand_bits((unsigned int) y);
    unsigned int zz = expand_bits((unsigned int) z);
    return xx * 4 + yy * 2 + zz;
}

__global__ void assign_morton(
        BVH bvh, unsigned int *d_morton, unsigned int *d_ids,
        unsigned int object_count, float3 sceneBoundMin, float3 sceneBoundMax)
{
    unsigned int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    unsigned int thread_id = block_id * (blockDim.x * blockDim.y * blockDim.z)
           + (threadIdx.z * (blockDim.x * blockDim.y))
           + (threadIdx.y * blockDim.x) + threadIdx.x;;
    if (thread_id >= object_count) return;

    // obtain center of triangle
    Triangle tri = bvh.d_triangles[thread_id];
    float3 pos = (1.f / 3.f) * (tri.v0 + tri.v1 + tri.v2);

    // normalize position
    float x = (pos.x - sceneBoundMin.x) / (sceneBoundMax.x - sceneBoundMin.x);
    float y = (pos.y - sceneBoundMin.y) / (sceneBoundMax.y - sceneBoundMin.y);
    float z = (pos.z - sceneBoundMin.z) / (sceneBoundMax.z - sceneBoundMin.z);
    // clamp to deal with numeric issues
    x = fminf(fmaxf(x, 0.f), 1.f);
    y = fminf(fmaxf(y, 0.f), 1.f);
    z = fminf(fmaxf(z, 0.f), 1.f);

    // obtain and set morton code based on normalized position
    d_morton[thread_id] = morton_3d(x, y, z);
    d_ids[thread_id] = thread_id;
}

// todo this kernel is pretty small, can it be combined with another?
__global__ void leaf_nodes(
        unsigned int *sorted_object_ids, unsigned int num_objects, BVH bvh)
{
    unsigned int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    unsigned int thread_id = block_id * (blockDim.x * blockDim.y * blockDim.z)
           + (threadIdx.z * (blockDim.x * blockDim.y))
           + (threadIdx.y * blockDim.x) + threadIdx.x;;
    if (thread_id >= num_objects) return;

    // no need to set parent to nullptr, each child will have a parent
    bvh.leaf_nodes[thread_id].object_id = sorted_object_ids[thread_id];
    // needed to recognize that this node is a leaf
    bvh.leaf_nodes[thread_id].left = nullptr;
    bvh.leaf_nodes[thread_id].right = nullptr;
    bvh.leaf_nodes[thread_id].parent = nullptr;   // optional, overwritten later for non-root leaves
    bvh.leaf_nodes[thread_id].visited = 0;        // optional if present

    // need to set for internal node parent to nullptr, for testing later
    // there is one less internal node than leaf node, test for that
    if (thread_id >= num_objects - 1) return;
    bvh.internal_nodes[thread_id].parent = nullptr;
}

__forceinline__ __device__ int delta(
        int a, int b, unsigned int n, unsigned int *c, unsigned int ka)
{
    // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
    if (b < 0 || b > n - 1) return -1;
    unsigned int kb = c[b];
    if (ka == kb) {
        // if keys are equal, use id as fallback
        // (+32 because they have the same morton code)
        return 32 + __clz((unsigned int) a ^ (unsigned int) b);
    }
    // clz = count leading zeros
    return __clz(ka ^ kb);
}

__forceinline__ __device__ int2 determine_range(
        unsigned int *sorted_morton_codes, unsigned int n, int i)
{
    unsigned int *c = sorted_morton_codes;
    unsigned int ki = c[i]; // key of i

    // determine direction of the range (+1 or -1)
    const int delta_l = delta(i, i - 1, n, c, ki);
    const int delta_r = delta(i, i + 1, n, c, ki);

    int d; // direction
    int delta_min; // min of delta_r and delta_l
    if (delta_r < delta_l) {
        d = -1;
        delta_min = delta_r;
    } else {
        d = 1;
        delta_min = delta_l;
    }
    // int d = (delta_r - delta_l > 0) ? 1 : -1;
    // int delta_min = delta(i, i - d, n, c, ki);  // note i - d


    // compute upper bound of the length of the range
    unsigned int l_max = 2;
    while (delta(i, i + l_max * d, n, c, ki) > delta_min) {
        l_max <<= 1;
    }

    // find other end using binary search
    unsigned int l = 0;
    for (unsigned int t = l_max >> 1; t > 0; t >>= 1) {
        if (delta(i, i + (l + t) * d, n, c, ki) > delta_min) {
            l += t;
        }
    }
    const int j = i + l * d;

    // ensure i <= j
    return i < j ? make_int2(i, j) : make_int2(j, i);
}

__forceinline__ __device__ int find_split(
        unsigned int *sorted_morton_codes, int first, int last, unsigned int n)
{
    const unsigned int first_code = sorted_morton_codes[first];

    // calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic

    const int common_prefix =
            delta(first, last, n, sorted_morton_codes, first_code);

    // use binary search to find where the next bit differs
    // specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one

    int split = first; // initial guess
    int step = last - first;

    do {
        step = (step + 1) >> 1; // exponential decrease
        const int new_split = split + step; // proposed new position

        if (new_split < last) {
            const int split_prefix = delta(
                    first, new_split, n, sorted_morton_codes, first_code);
            if (split_prefix > common_prefix) {
                split = new_split; // accept proposal
            }
        }
    } while (step > 1);

    return split;
}

__global__ void internal_nodes(
        unsigned int *sorted_morton_codes, unsigned int *sorted_object_ids,
        unsigned int num_objects, BVHNode *d_leaf_nodes, BVHNode *d_internal_nodes)
{
    unsigned int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    unsigned int thread_id = block_id * (blockDim.x * blockDim.y * blockDim.z)
           + (threadIdx.z * (blockDim.x * blockDim.y))
           + (threadIdx.y * blockDim.x) + threadIdx.x;;
    // notice the -1, we want i in range [0, num_objects - 2]
    if (thread_id >= num_objects - 1) return;

    // find out which range of objects the node corresponds to
    const int2 range = determine_range(
            sorted_morton_codes, num_objects, thread_id);

    // determine where to split the range
    const int split = find_split(
            sorted_morton_codes, range.x, range.y, num_objects);

    // select child a
    BVHNode *left;
    if (split == range.x) {
        left = &d_leaf_nodes[split];
    } else {
        left = &d_internal_nodes[split];
    }

    // select child b
    BVHNode *right;
    if (split + 1 == range.y) {
        right = &d_leaf_nodes[split + 1];
    } else {
        right = &d_internal_nodes[split + 1];
    }

    // record parent-child relationships
    d_internal_nodes[thread_id].left = left;
    d_internal_nodes[thread_id].right = right;
    d_internal_nodes[thread_id].visited = 0;
    left->parent = &d_internal_nodes[thread_id];
    right->parent = &d_internal_nodes[thread_id];
}

__global__ void set_aabb(
        unsigned int num_objects, BVHNode *d_leaf_nodes,
        BVHNode *d_internal_nodes, BVH bvh)
{
    unsigned int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    unsigned int thread_id = block_id * (blockDim.x * blockDim.y * blockDim.z)
           + (threadIdx.z * (blockDim.x * blockDim.y))
           + (threadIdx.y * blockDim.x) + threadIdx.x;;
    if (thread_id >= num_objects) return;

    const unsigned int object_id = d_leaf_nodes[thread_id].object_id;

    Triangle tri = bvh.d_triangles[object_id];
    float3 u = tri.v0;
    float3 v = tri.v1;
    float3 w = tri.v2;

    // IMPORTANT: make leaf min/max visible before parent reads it
    __threadfence();

    // set bounding box of leaf node
    d_leaf_nodes[thread_id].min = fminf(u, fminf(v, w));
    d_leaf_nodes[thread_id].max = fmaxf(u, fmaxf(v, w));

    // recursively set tree bounding boxes
    // {current_node} is always an internal node (since it is parent of another)
    BVHNode *current_node = d_leaf_nodes[thread_id].parent;
    while (true) {
        // we have reached the parent of the root node: terminate
        if (current_node == nullptr) break;

        // we have reached an inner node: check whether the node was visited
        unsigned int visited = atomicAdd(&current_node->visited, 1);

        // this is the first thread entering: terminate
        if (visited == 0) break;

        // second arrival: children should be ready, but fence before reading
        __threadfence();
        // this is the second thread entering, we know that our sibling has
        // reached the current node and terminated,
        // and hence the sibling bounding box is correct

        // set running bounding box to be the union of bounding boxes
        current_node->min = fminf(
                current_node->left->min, current_node->right->min);
        current_node->max = fmaxf(
                current_node->left->max, current_node->right->max);

        // make this internal node's AABB visible before its parent consumes it
        __threadfence();
        // continue traversal
        current_node = current_node->parent;
    }
}

void radix_sort(
        int num_items,
        unsigned int *d_keys_in, unsigned int *d_keys_out,
        unsigned int *d_values_in, unsigned int *d_values_out)
{
    // determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // run sorting operation
    // https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
    cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // free temporary storage
    CUDA_CHECK(cudaFree(d_temp_storage));
}

int build(BVH *bvh, std::vector<Triangle>& triangles, float3 sceneBoundMin, float3 sceneBoundMax)
{
    uint triangle_count = triangles.size();
    // must have at least two triangles. we cannot build a bvh for zero
    // triangles, and a bvh of one triangle has no internal nodes
    // which requires special handling which we forgo
    if (triangle_count <= 1) {
        fprintf(stderr, "too few triangles in scene: %d", triangle_count);
        return -1;
    }

    // events for measuring elapsed time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    uint BLOCK_SIZE = 1024;
    dim3 block_count(((triangle_count + BLOCK_SIZE - 1) / BLOCK_SIZE), 1, 1);
    uint block_size(BLOCK_SIZE);

    // copy scene to device
    CUDA_CHECK(cudaMalloc(
            &bvh->d_triangles, sizeof(Triangle) * triangles.size()));
    CUDA_CHECK(cudaMemcpy(
            bvh->d_triangles, triangles.data(),
            sizeof(Triangle) * triangles.size(), cudaMemcpyHostToDevice));

    // allocate array for morton and ids in dimension of triangles
    unsigned int *d_morton = nullptr;
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_morton),
            sizeof(unsigned int) * triangle_count));
    unsigned int *d_ids = nullptr;
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_ids),
            sizeof(unsigned int) * triangle_count));

    assign_morton<<<block_count, block_size>>>(
            *bvh, d_morton, d_ids, triangle_count, sceneBoundMin, sceneBoundMax);
    CUDA_SYNC_CHECK();

    // sort the key, value pairs according to morton codes
    unsigned int *d_morton_sorted = nullptr;
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_morton_sorted),
            sizeof(unsigned int) * triangle_count));
    unsigned int *d_ids_sorted = nullptr;
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_ids_sorted),
            sizeof(unsigned int) * triangle_count));
    // sort is stable (https://groups.google.com/g/cub-users/c/1iXn3sVMEuA)
    radix_sort(triangle_count, d_morton, d_morton_sorted, d_ids, d_ids_sorted);
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_ids)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_morton)));

    // allocate BVH (n leaf nodes, n - 1 internal nodes)
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&bvh->leaf_nodes),
            sizeof(BVHNode) * triangle_count));
    CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&bvh->internal_nodes),
            sizeof(BVHNode) * (triangle_count - 1)));
    // construct leaf nodes
    leaf_nodes<<<block_count, block_size>>>(d_ids_sorted, triangle_count, *bvh);

    CUDA_SYNC_CHECK();

    // construct internal nodes
    internal_nodes<<<block_count, block_size>>>(
            d_morton_sorted, d_ids_sorted, triangle_count,
            bvh->leaf_nodes, bvh->internal_nodes);

    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_ids_sorted)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_morton_sorted)));

    // calculate bounding boxes by walking the hierarchy toward the root
    set_aabb<<<block_count, block_size>>>(
            triangle_count, bvh->leaf_nodes, bvh->internal_nodes, *bvh);

    // print elapsed time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("building took %fs\n", milliseconds * 1e-3f);

    CUDA_SYNC_CHECK();

    return 0;
}

void clean(BVH *bvh)
{
    // free bvh
    CUDA_CHECK(cudaFree(bvh->leaf_nodes));
    CUDA_CHECK(cudaFree(bvh->internal_nodes));

    // free device scene
    CUDA_CHECK(cudaFree(bvh->d_triangles));
}

// Ray (+X) vs triangle intersection with robust classification:
// 0 = miss, 1 = proper crossing, 2 = degenerate (near edge/vertex)
__device__ int ray_intersect_triangle_x_classify(
        const float3& origin, const Triangle& tri)
{
    constexpr float det_eps = 1e-7f;
    constexpr float edge_eps = 1e-7f;
    const float3 dir = make_float3(1.0f, 0.0f, 0.0f);
    float3 v0 = tri.v0;
    float3 v1 = tri.v1;
    float3 v2 = tri.v2;

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross(dir, e2);
    float det = dot(e1, pvec);
    if (fabsf(det) < det_eps) return 0;

    float invDet = 1.0f / det;
    float3 tvec = origin - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < -edge_eps || u > 1.0f + edge_eps) return 0;

    float3 qvec = cross(tvec, e1);
    float v = dot(dir, qvec) * invDet;
    if (v < -edge_eps || (u + v) > 1.0f + edge_eps) return 0;

    float t = dot(e2, qvec) * invDet;
    if (t <= det_eps) return 0;

    float w = 1.0f - u - v;
    bool near_edge = (u < edge_eps) || (v < edge_eps) || (w < edge_eps);
    return near_edge ? 2 : 1;
}

// Traversal of BVH

// efficient slab test without divisions
__device__ __forceinline__ bool ray_x_hits_aabb(
    const float3& o, const float3& bmin, const float3& bmax)
{
    constexpr float eps_yz = 1e-5f;
    // Ray: o + t*(1,0,0), t>=0
    // Must overlap in y/z slabs, and the box must extend to the right of origin.x.
    if (o.x > bmax.x) return false;

    if (o.y < bmin.y - eps_yz || o.y > bmax.y + eps_yz) return false;
    if (o.z < bmin.z - eps_yz || o.z > bmax.z + eps_yz) return false;

    return true;
}



__device__ RayCountResult bvh_count_ray_hits_x(
    const BVH& bvh, const float3& origin)
{
    // Make stack bigger if needed. 64 is often enough for LBVH but 128/256 is safer.
    constexpr int STACK_CAP = 64;
    BVHNode* stack[STACK_CAP];
    BVHNode** sp = stack;
    *sp++ = nullptr;

    int hits = 0;
    bool deg = false;

    // Root: in your current build root is internal_nodes[0] (as tested)
    BVHNode* curr = bvh.internal_nodes;

    do {
        BVHNode* L = curr->left;
        BVHNode* R = curr->right;

        bool hitL = ray_x_hits_aabb(origin, L->min, L->max);
        bool hitR = ray_x_hits_aabb(origin, R->min, R->max);
        // hitL = hitR = true; // for debug purposes
        // Process leaves
        if (hitL && L->is_leaf()) {
            const Triangle tri = bvh.d_triangles[L->object_id];
            int res = ray_intersect_triangle_x_classify(origin, tri);
            if (res == 1) ++hits;
            else if (res == 2) deg = true;
        }
        if (hitR && R->is_leaf()) {
            const Triangle tri = bvh.d_triangles[R->object_id];
            int res = ray_intersect_triangle_x_classify(origin, tri);
            if (res == 1) ++hits;
            else if (res == 2) deg = true;
        }

        // Traverse internal children
        bool travL = hitL && !L->is_leaf();
        bool travR = hitR && !R->is_leaf();

        if (!travL && !travR) {
            curr = *--sp;
        } else {
            // Choose one to descend; push the other
            // (For +X ray you *could* prefer smaller min.x first, but not required for correctness.)
            curr = travL ? L : R;
            if (travL && travR) {
                if ((sp - stack) >= STACK_CAP) {
                    // Should be rare; if it happens, fall back conservatively:
                    // mark degenerate so we retry with shifted origin (more stable than silent corruption)
                    deg = true;
                    curr = nullptr;
                } else {
                    *sp++ = travL ? R : L;
                }
            }
        }
    } while (curr != nullptr);

    return {hits, deg};
}





// ============================================================================
// Distance Query Functions - Optimized with BVH
// ============================================================================

/// Finds the closest point on a triangle to a given point.
__forceinline__ __device__ float3 closest_point_on_triangle(
        const float3& p,
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

__device__ __forceinline__ float dist2_point_aabb(float3 p, float3 mn, float3 mx) {
    // branchless-ish clamping
    float dx = fmaxf(fmaxf(mn.x - p.x, 0.0f), p.x - mx.x);
    float dy = fmaxf(fmaxf(mn.y - p.y, 0.0f), p.y - mx.y);
    float dz = fmaxf(fmaxf(mn.z - p.z, 0.0f), p.z - mx.z);
    return dx*dx + dy*dy + dz*dz;
}

/// Device function to compute distance to a triangle using BVH traversal
__device__ DistanceHit traverse_for_distance(
        BVH bvh, float3 point, float max_dist)
{
    // Stack for BVH traversal
    BVHNode *stack[64];
    BVHNode **stack_ptr = stack;
    *stack_ptr++ = NULL;

    float closest_distance2 = max_dist;
    int closest_object_id = -1;
    float3 closest_point = point;

    // Start traversal from root
    BVHNode *curr = bvh.internal_nodes;
    do {
        BVHNode *child_l = curr->left;
        BVHNode *child_r = curr->right;

        // Simple AABB sphere intersection: check if point is within expanded bounding box
        // Expand box by current closest distance for early termination
        float3 box_min_l = child_l->min;
        float3 box_max_l = child_l->max;
        float3 box_min_r = child_r->min;
        float3 box_max_r = child_r->max;

        // Check if point is within AABB (expanded by search radius)
        float dist2_to_box_l = dist2_point_aabb(point, box_min_l, box_max_l);
        float dist2_to_box_r = dist2_point_aabb(point, box_min_r, box_max_r);

        bool hit_l = (dist2_to_box_l < closest_distance2);
        bool hit_r = (dist2_to_box_r < closest_distance2);
        // hit_l = hit_r = true; // for debug purposes

        // Query leaf nodes
        if (hit_l && child_l->is_leaf()) {
            Triangle tri = bvh.d_triangles[child_l->object_id];
            float3 v0 = tri.v0;
            float3 v1 = tri.v1;
            float3 v2 = tri.v2;

            float3 cp = closest_point_on_triangle(point, v0, v1, v2);
            float3 d  = make_float3(point.x - cp.x, point.y - cp.y, point.z - cp.z);
            float dist2 = d.x*d.x + d.y*d.y + d.z*d.z;

            if (dist2 < closest_distance2) {
                closest_distance2 = dist2;
                closest_object_id = child_l->object_id;
                closest_point = cp;
            }
        }

        if (hit_r && child_r->is_leaf()) {
            Triangle tri = bvh.d_triangles[child_r->object_id];
            float3 v0 = tri.v0;
            float3 v1 = tri.v1;
            float3 v2 = tri.v2;

            float3 cp = closest_point_on_triangle(point, v0, v1, v2);
            float3 d  = make_float3(point.x - cp.x, point.y - cp.y, point.z - cp.z);
            float dist2 = d.x*d.x + d.y*d.y + d.z*d.z;

            if (dist2 < closest_distance2) {
                closest_distance2 = dist2;
                closest_object_id = child_r->object_id;
                closest_point = cp;
            }
        }

        // Traverse internal nodes
        bool traverse_l = (hit_l && !child_l->is_leaf());
        bool traverse_r = (hit_r && !child_r->is_leaf());


        // if (!traverse_l && !traverse_r) {
        //     curr = *--stack_ptr;
        // } else {
        //     curr = traverse_l ? child_l : child_r;
        //     if (traverse_l && traverse_r) {
        //         *stack_ptr++ = child_r;
        //     }
        // }
        if (traverse_l && traverse_r) {
            // Near-first
            if (dist2_to_box_l < dist2_to_box_r) { *stack_ptr++ = child_r; curr = child_l; }
            else           { *stack_ptr++ = child_l; curr = child_r; }
        } else if (traverse_l) {
            curr = child_l;
        } else if (traverse_r) {
            curr = child_r;
        } else {
            curr = *--stack_ptr;
        }
    } while (curr != NULL);

    DistanceHit dh;
    dh.distance = sqrtf(closest_distance2);
    dh.object_id = closest_object_id;
    dh.closest_point = closest_point;
    dh.hit = (closest_object_id >= 0);
    return dh;
}



