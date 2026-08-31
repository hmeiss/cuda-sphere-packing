#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "mesh.h"
#include <iostream>
#include <algorithm>
#include <limits>

Mesh::Mesh() {
    bounds.min = make_float3(0, 0, 0);
    bounds.max = make_float3(0, 0, 0);
}

Mesh::~Mesh() {}

bool Mesh::loadFromObj(const std::string& filename) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str())) {
        std::cerr << "Failed to load OBJ: " << err << std::endl;
        return false;
    }

    triangles.clear();
    for (const auto& shape : shapes) {
        for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++) {
            Triangle tri;
            tinyobj::index_t idx0 = shape.mesh.indices[3 * f + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[3 * f + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[3 * f + 2];

            tri.v0 = make_float3(attrib.vertices[3 * idx0.vertex_index + 0],
                                 attrib.vertices[3 * idx0.vertex_index + 1],
                                 attrib.vertices[3 * idx0.vertex_index + 2]);
            tri.v1 = make_float3(attrib.vertices[3 * idx1.vertex_index + 0],
                                 attrib.vertices[3 * idx1.vertex_index + 1],
                                 attrib.vertices[3 * idx1.vertex_index + 2]);
            tri.v2 = make_float3(attrib.vertices[3 * idx2.vertex_index + 0],
                                 attrib.vertices[3 * idx2.vertex_index + 1],
                                 attrib.vertices[3 * idx2.vertex_index + 2]);
            triangles.push_back(tri);
        }
    }

    computeBounds();
    return true;
}

void Mesh::computeBounds() {
    if (triangles.empty()) return;

    float3 minVal = make_float3(std::numeric_limits<float>::max(), 
                                std::numeric_limits<float>::max(), 
                                std::numeric_limits<float>::max());
    float3 maxVal = make_float3(-std::numeric_limits<float>::max(), 
                                -std::numeric_limits<float>::max(), 
                                -std::numeric_limits<float>::max());

    for (const auto& tri : triangles) {
        auto update = [&](float3 v) {
            minVal.x = std::min(minVal.x, v.x);
            minVal.y = std::min(minVal.y, v.y);
            minVal.z = std::min(minVal.z, v.z);
            maxVal.x = std::max(maxVal.x, v.x);
            maxVal.y = std::max(maxVal.y, v.y);
            maxVal.z = std::max(maxVal.z, v.z);
        };
        update(tri.v0);
        update(tri.v1);
        update(tri.v2);
    }
    bounds.min = minVal;
    bounds.max = maxVal;
}

double Mesh::computeVolume() const {
    if (triangles.empty()) return 0.0;

    // Use divergence theorem: Volume = (1/6) * Σ((v0 × v1) · v2) for each triangle
    // where × is cross product and · is dot product
    double volume = 0.0;

    for (const auto& tri : triangles) {
        // Cross product: v0 × v1
        float3 cross;
        cross.x = tri.v0.y * tri.v1.z - tri.v0.z * tri.v1.y;
        cross.y = tri.v0.z * tri.v1.x - tri.v0.x * tri.v1.z;
        cross.z = tri.v0.x * tri.v1.y - tri.v0.y * tri.v1.x;

        // Dot product: cross · v2
        float dotProduct = cross.x * tri.v2.x + cross.y * tri.v2.y + cross.z * tri.v2.z;

        volume += static_cast<double>(dotProduct);
    }

    // Divide by 6 (keep sign to detect orientation)
    volume /= 6.0;
    return volume;
}

