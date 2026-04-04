cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# CUDA 12.9.1 installed alongside JetPack 6 default 12.6.
# Configure with: cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
#                       -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 ..
project(deusridet LANGUAGES CXX CUDA)

# ============================================================================
# Build standards
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA architecture: SM87 (Ampere / Jetson AGX Orin)
# JIT compilation not recommended on Tegra — always compile for specific SM
# See: CUDA for Tegra Application Note
# ============================================================================
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_87,code=sm_87 --expt-relaxed-constexpr -O3 --use_fast_math -Xptxas -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Include directories
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

# ============================================================================
# Core engine static library (machina — inference engine)
# ============================================================================
file(GLOB_RECURSE MACHINA_SOURCES
    "src/machina/*.cpp"
    "src/machina/*.cu"
)

add_library(machina STATIC ${MACHINA_SOURCES})
target_link_libraries(machina PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)

# ============================================================================
# Common utilities static library
# ============================================================================
file(GLOB_RECURSE COMMUNIS_SOURCES
    "src/communis/*.cpp"
)

add_library(communis STATIC ${COMMUNIS_SOURCES})
target_link_libraries(communis PRIVATE CUDA::cudart)

# ============================================================================
# Main executable
# ============================================================================
add_executable(deusridet
    src/main.cpp
    src/commands.cpp
)
target_link_libraries(deusridet PRIVATE machina communis stdc++fs rt dl)

# Embed RPATH (not RUNPATH) so the binary finds CUDA 12.9 libs.
# DT_RPATH takes precedence over ldconfig cache, unlike DT_RUNPATH.
set_target_properties(deusridet PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)
