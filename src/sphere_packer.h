#pragma once

#include "cuda_functions.h"
#include "mesh.h"
#include <vector>
#include <string>

enum class SDFInitMode {
    BVH,
    BruteForce,
    Multi2,
    cuBQL,
};

class SpherePacker {
public:
    SpherePacker(int gridResolution = 256, SDFInitMode mode = SDFInitMode::BVH);
    ~SpherePacker();

    bool packOneSpherePerIteration(const Mesh& mesh, int targetSphereCount, std::vector<Sphere>& resultSpheres);
    bool packMulti2Spheres(const Mesh& mesh, int targetSphereCount, std::vector<Sphere>& resultSpheres);
    bool exportToFile(const std::vector<Sphere>& spheres, const std::string& filename);
    double getLastInitializationSeconds() const;
    const std::vector<PackingProgressSample>& getProgressSamples() const;

private:
    int gridResolution;
    SDFGrid grid;
    SDFInitMode sdfMode;
    double lastInitializationSeconds;
    std::vector<PackingProgressSample> progressSamples;

    bool initializeGrid(const Mesh& mesh);
    bool cleanupGrid();
};
