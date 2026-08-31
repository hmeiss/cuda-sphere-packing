cmake_minimum_required(VERSION 3.27)

project(TriangleMeshToSpherePacking LANGUAGES CXX CUDA)

# (Optional but handy)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prefer target-based standard settings (instead of global CMAKE_CXX_STANDARD)
# but either style is fine. This is the modern target approach:
add_executable(SpherePacker
  # Prefer listing sources explicitly (recommended)
  src/cuda_functions.cu
  src/lbvh.cu
  src/cubql_sdf.cu
  src/sphere_packer.cpp
  src/mesh.cpp
  src/main.cpp
)

target_compile_features(SpherePacker PRIVATE cxx_std_20 cuda_std_20)

# Include directories (target-based)
target_include_directories(SpherePacker PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/external
  ${CMAKE_CURRENT_SOURCE_DIR}/external/cuBQL
)

# CUDA architectures
set_property(TARGET SpherePacker PROPERTY CUDA_ARCHITECTURES native)
# set_property(TARGET SpherePacker PROPERTY CUDA_ARCHITECTURES 70 75 80 86)

# Enable separable compilation
set_property(TARGET SpherePacker PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# If still device-link issues on Windows with separable compilation, try:
#set_property(TARGET SpherePacker PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Use the modern toolkit package + imported targets (instead of FindCUDA / CUDA_LIBRARIES)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(SpherePacker PRIVATE CUDA::cudart)

# Thrust is header-only and ships with the CUDA toolkit, so you usually do NOT need
# a separate find_package(Thrust). If your environment provides Thrust::Thrust, you can do:
# find_package(Thrust CONFIG QUIET)
# if(TARGET Thrust::Thrust)
#   target_link_libraries(SpherePacker PRIVATE Thrust::Thrust)
# endif()

# CUDA-specific flags
target_compile_options(SpherePacker PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-extended-lambda>
)

# Warnings
if(MSVC)
  target_compile_options(SpherePacker PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(SpherePacker PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()
