#pragma once

#include "cuda_functions.h"
#include "mesh.h"
#include <vector>

// Builds a cuBQL BVH over the triangle mesh, computes unsigned distances and
// determines inside/outside sign using a 6-ray majority vote, then frees the BVH.
void initSDF_cuBQL(SDFGrid& grid, const std::vector<Triangle>& triangles);
