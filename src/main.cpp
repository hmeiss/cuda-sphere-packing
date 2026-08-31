#include "mesh.h"
#include "sphere_packer.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>
#include <cctype>
#include <algorithm>
#include <fstream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " <input.obj> <output.spheres> [targetCount=100] [gridRes=256] [sdfMode=bvh|brute|multi|multi2|cubql]" << std::endl;
}

std::string modeToString(SDFInitMode mode) {
    switch (mode) {
        case SDFInitMode::BVH: return "bvh";
        case SDFInitMode::BruteForce: return "brute";
        case SDFInitMode::Multi2: return "multi2";
        case SDFInitMode::cuBQL:  return "cubql";
    }
    return "unknown";
}

std::string makeLogFilename(const std::string& outputSpheres) {
    const size_t sep = outputSpheres.find_last_of("\\/");
    const size_t dot = outputSpheres.find_last_of('.');
    if (dot != std::string::npos && (sep == std::string::npos || dot > sep)) {
        return outputSpheres.substr(0, dot) + ".log";
    }
    return outputSpheres + ".log";
}

std::vector<PackingProgressSample> selectProgressSamples(
    const std::vector<PackingProgressSample>& progressSamples,
    int maxSamples
) {
    std::vector<PackingProgressSample> selected;
    if (progressSamples.empty() || maxSamples <= 0) {
        return selected;
    }
    if ((int)progressSamples.size() <= maxSamples) {
        return progressSamples;
    }

    selected.reserve((size_t)maxSamples);
    size_t lastIndex = 0;
    for (int i = 0; i < maxSamples; ++i) {
        const double t = (maxSamples == 1) ? 1.0 : (double)i / (double)(maxSamples - 1);
        size_t index = (size_t)std::llround(t * (double)(progressSamples.size() - 1));
        if (!selected.empty() && index <= lastIndex) {
            index = lastIndex + 1;
        }
        if (index >= progressSamples.size()) {
            index = progressSamples.size() - 1;
        }
        selected.push_back(progressSamples[index]);
        lastIndex = index;
    }
    return selected;
}

bool writeProgressLog(
    const std::string& logFilename,
    const std::string& inputObj,
    const std::string& outputSpheres,
    int targetSphereCount,
    int gridResolution,
    SDFInitMode sdfMode,
    size_t triangleCount,
    const BoundingBox& bounds,
    double meshVolume,
    double boundingBoxVolume,
    double initializationSeconds,
    double totalRuntimeSeconds,
    const std::vector<Sphere>& spheres,
    const std::vector<PackingProgressSample>& progressSamples
) {
    std::ofstream log(logFilename);
    if (!log.is_open()) {
        std::cerr << "Failed to open progress log file: " << logFilename << std::endl;
        return false;
    }

    log << "INFO       : SpherePacker progress log\n";
    log << "INFO       : Input OBJ: " << inputObj << "\n";
    log << "INFO       : Output spheres: " << outputSpheres << "\n";
    log << "INFO       : Target sphere count: " << targetSphereCount << "\n";
    log << "INFO       : Grid resolution: " << gridResolution << "\n";
    log << "INFO       : SDF mode: " << modeToString(sdfMode) << "\n";
    log << "INFO       : Triangle count: " << triangleCount << "\n";
    log << "INFO       : Bounds min: " << bounds.min.x << " " << bounds.min.y << " " << bounds.min.z << "\n";
    log << "INFO       : Bounds max: " << bounds.max.x << " " << bounds.max.y << " " << bounds.max.z << "\n";
    log << "INFO       : Mesh volume: " << std::setprecision(9) << meshVolume << "\n";
    log << "INFO       : Bounding box volume: " << std::setprecision(9) << boundingBoxVolume << "\n";
    log << "INFO       : Initialization time [s]: " << std::fixed << std::setprecision(6) << initializationSeconds << "\n";
    log << "INFO       : Total runtime [s]: " << std::fixed << std::setprecision(6) << totalRuntimeSeconds << "\n";

    std::vector<PackingProgressSample> logSamples = selectProgressSamples(progressSamples, 500);
    if (logSamples.empty() && !spheres.empty()) {
        logSamples.push_back({(int)spheres.size(), totalRuntimeSeconds});
    }
    log << "INFO       : Progress data points: " << logSamples.size() << "\n";

    std::vector<double> cumulativeVolumes(spheres.size() + 1, 0.0);
    for (size_t i = 0; i < spheres.size(); ++i) {
        cumulativeVolumes[i + 1] = cumulativeVolumes[i] + (4.0 / 3.0) * M_PI * std::pow(spheres[i].radius, 3.0);
    }

    for (const PackingProgressSample& sample : logSamples) {
        const int count = std::max(0, std::min(sample.sphereCount, (int)spheres.size()));
        if (count <= 0) {
            continue;
        }
        const double densityPercent = (meshVolume > 0.0)
            ? (cumulativeVolumes[(size_t)count] / meshVolume) * 100.0
            : 0.0;
        log << "INFO       : Estimated object packing density with " << count
            << " spheres: " << std::defaultfloat << std::setprecision(6) << densityPercent
            << "% Total time [s]: " << std::fixed << std::setprecision(6) << sample.elapsedSeconds << "\n";
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputObj = argv[1];
    std::string outputSpheres = argv[2];
    int targetSphereCount = 100;
    int gridResolution = 256;
    SDFInitMode sdfMode = SDFInitMode::BVH;

    if (argc > 3) {
        targetSphereCount = std::stoi(argv[3]);
    }
    if (argc > 4) {
        gridResolution = std::stoi(argv[4]);
    }
    if (argc > 5) {
        std::string mode = argv[5];
        for (auto& c : mode) c = static_cast<char>(std::tolower(c));
        if (mode == "brute" || mode == "bruteforce") {
            sdfMode = SDFInitMode::BruteForce;
        } else if (mode == "bvh") {
            sdfMode = SDFInitMode::BVH;
        } else if (mode == "multi2") {
            sdfMode = SDFInitMode::Multi2;
        } else if (mode == "cubql") {
            sdfMode = SDFInitMode::cuBQL;
        } else {
            std::cerr << "Unknown sdfMode '" << mode << "', defaulting to BVH" << std::endl;
        }
    }

    // Load mesh
    Mesh mesh;
    if (!mesh.loadFromObj(inputObj)) {
        std::cerr << "Failed to load mesh from: " << inputObj << std::endl;
        return 1;
    }

    const BoundingBox& bounds = mesh.getBounds();
    std::cout << "Mesh loaded successfully" << std::endl;
    std::cout << "  Triangles: " << mesh.getTriangles().size() << std::endl;
    std::cout << "  Bounds: [" << bounds.min.x << ", " << bounds.min.y << ", " << bounds.min.z << "] -> "
              << "[" << bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z << "]" << std::endl;
    std::cout << std::endl;

    // Record start time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Perform sphere packing
    SpherePacker packer(gridResolution, sdfMode);
    std::vector<Sphere> spheres;

    std::cout << "Starting sphere packing..." << std::endl;
    if (sdfMode != SDFInitMode::Multi2
        && sdfMode != SDFInitMode::cuBQL
        && !packer.packOneSpherePerIteration(mesh, targetSphereCount, spheres)) {
        std::cerr << "Failed to pack mesh" << std::endl;
        return 1;
    }
    else if ((sdfMode == SDFInitMode::Multi2 || sdfMode == SDFInitMode::cuBQL)
        && !packer.packMulti2Spheres(mesh, targetSphereCount, spheres)) {
        std::cerr << "Failed to pack mesh with multi2/cubql method" << std::endl;
        return 1;
    }

    // Record end time
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;

    std::cout << std::endl;
    std::cout << "Packing completed: " << spheres.size() << " spheres generated" << std::endl;
    std::cout << "Total Runtime: " << duration.count() << " seconds" << std::endl;
    std::cout << std::endl;

    // Calculate packing density
    double totalSphereVolume = 0.0;
    for (const auto& s : spheres) {
        totalSphereVolume += (4.0/3.0) * M_PI * std::pow(s.radius, 3.0);
    }

    float3 meshSize = bounds.size();
    double meshVolume = mesh.computeVolume();
    double boundingBoxVolume = static_cast<double>(meshSize.x * meshSize.y * meshSize.z);
    
    double packingDensity = (meshVolume > 0) ? (totalSphereVolume / meshVolume) : 0.0;
    double boundingBoxDensity = (boundingBoxVolume > 0) ? (totalSphereVolume / boundingBoxVolume) : 0.0;

    std::cout << "Total Sphere Volume: " << totalSphereVolume << std::endl;
    std::cout << "Mesh Volume: " << meshVolume << std::endl;
    std::cout << "Bounding Box Volume: " << boundingBoxVolume << std::endl;
    std::cout << "Packing Density (vs Mesh): " << packingDensity * 100.0 << "%" << std::endl;
    std::cout << "Packing Density (vs Bounding Box): " << boundingBoxDensity * 100.0 << "%" << std::endl;
    std::cout << std::endl;

    // Export to file
    std::cout << "Exporting spheres to " << outputSpheres << "..." << std::endl;
    if (!packer.exportToFile(spheres, outputSpheres)) {
        std::cerr << "Failed to export spheres" << std::endl;
        return 1;
    }

    const std::string progressLog = makeLogFilename(outputSpheres);
    std::cout << "Exporting progress log to " << progressLog << "..." << std::endl;
    if (!writeProgressLog(
            progressLog,
            inputObj,
            outputSpheres,
            targetSphereCount,
            gridResolution,
            sdfMode,
            mesh.getTriangles().size(),
            bounds,
            meshVolume,
            boundingBoxVolume,
            packer.getLastInitializationSeconds(),
            duration.count(),
            spheres,
            packer.getProgressSamples())) {
        std::cerr << "Failed to export progress log" << std::endl;
        return 1;
    }

    std::cout << "Success!" << std::endl;
    return 0;
}
