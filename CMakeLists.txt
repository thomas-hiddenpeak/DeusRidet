cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# CUDA 12.9.1 installed alongside JetPack 6 default 12.6.
# Configure with: cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
#                       -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 ..
project(deusridet LANGUAGES C CXX CUDA)

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
    /usr/include/aarch64-linux-gnu  # TensorRT headers
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
target_link_libraries(communis PRIVATE CUDA::cudart nvinfer nvonnxparser)

# ============================================================================
# Network layer static library (nexus — WebSocket + HTTP server)
# ============================================================================
file(GLOB_RECURSE NEXUS_SOURCES
    "src/nexus/*.cpp"
)

add_library(nexus STATIC ${NEXUS_SOURCES})
target_link_libraries(nexus PRIVATE communis)

# ============================================================================
# TEN VAD static library (third_party — Agora TEN Framework VAD)
# ============================================================================
file(GLOB TEN_VAD_CC_SOURCES "third_party/ten_vad/src/*.cc")
file(GLOB TEN_VAD_C_SOURCES  "third_party/ten_vad/src/*.c")
add_library(ten_vad STATIC ${TEN_VAD_CC_SOURCES} ${TEN_VAD_C_SOURCES})
target_include_directories(ten_vad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ten_vad/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ten_vad/src
)
target_link_libraries(ten_vad PRIVATE onnxruntime m)
# Suppress warnings in third-party code.
target_compile_options(ten_vad PRIVATE -w)

# ============================================================================
# Senses library (sensus — audio pipeline, Mel, VAD)
# ============================================================================
file(GLOB_RECURSE SENSUS_SOURCES
    "src/sensus/*.cpp"
    "src/sensus/*.cu"
)

add_library(sensus STATIC ${SENSUS_SOURCES})
target_link_libraries(sensus PRIVATE communis machina CUDA::cudart CUDA::cublas CUDA::cufft onnxruntime ten_vad)

# ============================================================================
# Speaker identification static library (orator — CAM++ speaker encoder)
# ============================================================================
file(GLOB_RECURSE ORATOR_SOURCES
    "src/orator/*.cpp"
    "src/orator/*.cu"
)

add_library(orator STATIC ${ORATOR_SOURCES})
target_link_libraries(orator PRIVATE communis CUDA::cudart CUDA::cublas cudnn onnxruntime)

# ============================================================================
# Memory management static library (memoria — KV cache, block pool, SSD swap)
# ============================================================================
file(GLOB_RECURSE MEMORIA_SOURCES
    "src/memoria/*.cpp"
    "src/memoria/*.cu"
)

add_library(memoria STATIC ${MEMORIA_SOURCES})
target_link_libraries(memoria PRIVATE machina communis CUDA::cudart)

# ============================================================================
# Consciousness stream static library (conscientia — continuous Prefill engine)
# ============================================================================
file(GLOB_RECURSE CONSCIENTIA_SOURCES
    "src/conscientia/*.cpp"
)

add_library(conscientia STATIC ${CONSCIENTIA_SOURCES})
target_link_libraries(conscientia PRIVATE machina memoria communis CUDA::cudart)

# ============================================================================
# Main executable
# ============================================================================
add_executable(deusridet
    src/main.cpp
    src/commands.cpp
)
target_link_libraries(deusridet PRIVATE machina memoria conscientia communis nexus sensus orator stdc++fs rt dl)

# Embed RPATH (not RUNPATH) so the binary finds CUDA 12.9 libs.
# DT_RPATH takes precedence over ldconfig cache, unlike DT_RUNPATH.
set_target_properties(deusridet PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)
