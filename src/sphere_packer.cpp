#include "sphere_packer.h"
#include "lbvh.h"
#include "cubql_sdf.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <chrono>
#include <cuda_runtime.h>

SpherePacker::SpherePacker(int gridResolution, SDFInitMode mode)
    : gridResolution(gridResolution), sdfMode(mode), lastInitializationSeconds(0.0) {
    grid.data = nullptr;
    grid.dimX = grid.dimY = grid.dimZ = 0;
}

SpherePacker::~SpherePacker() {
    cleanupGrid();
}

bool SpherePacker::initializeGrid(const Mesh& mesh) {
    const BoundingBox& meshBounds = mesh.getBounds();
    float3 size = meshBounds.size();
    float maxDim = fmax(fmax(size.x, size.y), size.z);
    
    // Create a uniform grid with specified resolution
    grid.dimX = grid.dimY = grid.dimZ = gridResolution;
    // Keep cubic voxels but ensure grid bounds match the voxel extent.
    grid.cellSize = make_float3(maxDim / gridResolution,
                                maxDim / gridResolution,
                                maxDim / gridResolution);
    float3 center = meshBounds.center();
    float half = 0.5f * maxDim;
    grid.bounds.min = make_float3(center.x - half, center.y - half, center.z - half);
    grid.bounds.max = make_float3(center.x + half, center.y + half, center.z + half);

    // Allocate device memory
    size_t totalVoxels = (size_t)grid.dimX * grid.dimY * grid.dimZ;
    if (cudaMalloc(&grid.data, totalVoxels * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for SDF grid" << std::endl;
        return false;
    }

    // Initialize grid to large negative values (outside the mesh)
    cudaMemset(grid.data, 0xFF800000, totalVoxels * sizeof(float)); // Pattern for -1e9

    // Build BVH for efficient mesh traversal
    std::vector<Triangle> triangles = mesh.getTriangles();

    if (sdfMode == SDFInitMode::BVH || sdfMode == SDFInitMode::Multi2) {
        BVH bvh = {};
        std::cout << "Building Linear BVH with " << triangles.size() << " triangles..." << std::endl;
        if (build(&bvh, triangles, meshBounds.min, meshBounds.max) != 0) {
            std::cerr << "Failed to build BVH" << std::endl;
            return false;
        }

        // Initialize SDF using BVH-accelerated computation
        std::cout << "Computing SDF using BVH-accelerated distance..." << std::endl;
        initSDF_BVH(grid, bvh);

        // Clean up BVH
        cudaDeviceSynchronize();
        clean(&bvh);
    } else if (sdfMode == SDFInitMode::BruteForce) {
        std::cout << "Computing SDF using brute-force..." << std::endl;
        Triangle* d_triangles = nullptr;
        size_t triBytes = triangles.size() * sizeof(Triangle);
        if (cudaMalloc(&d_triangles, triBytes) != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory for triangles" << std::endl;
            return false;
        }
        cudaMemcpy(d_triangles, triangles.data(), triBytes, cudaMemcpyHostToDevice);
        initSDF_Brute(grid, d_triangles, static_cast<int>(triangles.size()));
        cudaFree(d_triangles);
    } else if (sdfMode == SDFInitMode::cuBQL) {
        std::cout << "Computing SDF using cuBQL BVH..." << std::endl;
        initSDF_cuBQL(grid, triangles);
    }

    return true;
}

bool SpherePacker::cleanupGrid() {
    if (grid.data != nullptr) {
        cudaFree(grid.data);
        grid.data = nullptr;
    }
    return true;
}

bool SpherePacker::packOneSpherePerIteration(const Mesh& mesh, int targetSphereCount, std::vector<Sphere>& resultSpheres) {
    resultSpheres.reserve(targetSphereCount);
    resultSpheres.clear();
    progressSamples.clear();
    lastInitializationSeconds = 0.0;

    const auto packStart = std::chrono::steady_clock::now();
    if (!initializeGrid(mesh)) {
        return false;
    }
    lastInitializationSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - packStart).count();
    
    for (int iter = 0; iter < targetSphereCount; ++iter) {
        float maxDist = 0;
        float3 maxDistPos = make_float3(0, 0, 0);

        // Find the voxel with the maximum distance value
        // findMaxDistanceVoxel(grid, maxDist, maxDistPos);
        findMaxDistanceVoxel(grid, maxDist, maxDistPos);
        // Create a sphere at this location
        Sphere newSphere;
        newSphere.center = maxDistPos;
        newSphere.radius = maxDist;
        resultSpheres.push_back(newSphere);

        // std::cout << "Sphere " << (iter + 1) << ": center=(" << newSphere.center.x << ", " 
        //           << newSphere.center.y << ", " << newSphere.center.z << "), radius=" 
        //           << newSphere.radius << std::endl;

        // Update SDF to exclude this sphere
        updateSDF_SphereAABB(grid, newSphere);
        progressSamples.push_back({
            static_cast<int>(resultSpheres.size()),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - packStart).count()
        });
    }

    cleanupGrid();
    std::cout << "Packing complete: " << resultSpheres.size() << " spheres placed" << std::endl;
    return true;
}


bool SpherePacker::packMulti2Spheres(const Mesh& mesh, int targetSphereCount, std::vector<Sphere>& resultSpheres)
{
    resultSpheres.reserve(targetSphereCount);
    resultSpheres.clear();
    progressSamples.clear();
    lastInitializationSeconds = 0.0;

    const auto packStart = std::chrono::steady_clock::now();
    if (!initializeGrid(mesh)) {
        return false;
    }
    lastInitializationSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - packStart).count();

    const bool ok = packSpheresMulti2Snapshot(grid, targetSphereCount, resultSpheres, &progressSamples, lastInitializationSeconds);

    cleanupGrid();

    if (!ok) {
        std::cerr << "Multi2 sphere packing failed" << std::endl;
        return false;
    }

    std::cout << "Packing complete (multi2): " << resultSpheres.size() << " spheres placed" << std::endl;
    return true;
}

double SpherePacker::getLastInitializationSeconds() const {
    return lastInitializationSeconds;
}

const std::vector<PackingProgressSample>& SpherePacker::getProgressSamples() const {
    return progressSamples;
}

bool SpherePacker::exportToFile(const std::vector<Sphere>& spheres, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return false;
    }

    // Format: x y z r
    outfile << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto& sphere : spheres) {
        outfile << sphere.center.x << " " 
                << sphere.center.y << " " 
                << sphere.center.z << " " 
                << sphere.radius << "\n";
    }

    outfile.close();
    std::cout << "Exported " << spheres.size() << " spheres to " << filename << std::endl;
    return true;
}
