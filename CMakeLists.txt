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
# timeline_logger.cpp depends on sensus — built in main exe, not here.
list(FILTER COMMUNIS_SOURCES EXCLUDE REGEX "timeline_logger\\.cpp$")

add_library(communis STATIC ${COMMUNIS_SOURCES})
target_link_libraries(communis PRIVATE CUDA::cudart)

# ============================================================================
# Network layer static library (nexus — WebSocket + HTTP server)
# ============================================================================
file(GLOB_RECURSE NEXUS_SOURCES
    "src/nexus/*.cpp"
)

add_library(nexus STATIC ${NEXUS_SOURCES})
target_link_libraries(nexus PRIVATE communis)

# ============================================================================
# Senses library (sensus — audio pipeline, Mel, VAD)
# ============================================================================
file(GLOB_RECURSE SENSUS_SOURCES
    "src/sensus/*.cpp"
    "src/sensus/*.cu"
)

add_library(sensus STATIC ${SENSUS_SOURCES})
target_link_libraries(sensus PRIVATE communis machina CUDA::cudart CUDA::cublas CUDA::cufft cudnn)

# ============================================================================
# Speaker identification static library (orator — CAM++ speaker encoder)
# ============================================================================
file(GLOB_RECURSE ORATOR_SOURCES
    "src/orator/*.cpp"
    "src/orator/*.cu"
)

add_library(orator STATIC ${ORATOR_SOURCES})
target_link_libraries(orator PRIVATE communis CUDA::cudart CUDA::cublas cudnn)

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
    src/communis/timeline_logger.cpp
)
target_link_libraries(deusridet PRIVATE machina memoria conscientia communis nexus sensus orator stdc++fs rt dl)

# Embed RPATH (not RUNPATH) so the binary finds CUDA 12.9 libs.
# DT_RPATH takes precedence over ldconfig cache, unlike DT_RUNPATH.
set_target_properties(deusridet PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

# ============================================================================
# Test: FRCRN GPU standalone test
# ============================================================================
add_executable(test_frcrn_gpu tests/test_frcrn_gpu.cpp)
target_include_directories(test_frcrn_gpu PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_frcrn_gpu PRIVATE sensus machina communis CUDA::cudart CUDA::cublas CUDA::cufft cudnn)
set_target_properties(test_frcrn_gpu PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

# Test: MossFormer2 native CUDA numerical validation
# ============================================================================
add_executable(test_mossformer2 tests/test_mossformer2.cpp)
target_include_directories(test_mossformer2 PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_mossformer2 PRIVATE sensus machina communis CUDA::cudart CUDA::cublas)
set_target_properties(test_mossformer2 PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

# Test: P1 Overlap Detector + P2 Speech Separator

# Benchmark: MossFormer2 native CUDA performance
add_executable(bench_mossformer2 tools/bench_mossformer2.cpp)
target_include_directories(bench_mossformer2 PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(bench_mossformer2 PRIVATE sensus machina communis CUDA::cudart CUDA::cublas)
set_target_properties(bench_mossformer2 PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)
# ============================================================================
add_executable(test_overlap_separator tests/test_overlap_separator.cpp)
target_include_directories(test_overlap_separator PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_overlap_separator PRIVATE sensus communis CUDA::cudart)

# Tool: Overlap → Separation → Speaker ID pipeline test
# ============================================================================
add_executable(test_overlap_spkid tools/test_overlap_spkid.cpp)
target_include_directories(test_overlap_spkid PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_overlap_spkid PRIVATE sensus orator communis CUDA::cudart CUDA::cublas CUDA::cufft cudnn)

# Test: PyannoteSeg3 native CUDA validation
add_executable(test_pyannote_seg3 tests/test_pyannote_seg3.cpp)
target_include_directories(test_pyannote_seg3 PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_pyannote_seg3 PRIVATE sensus machina communis CUDA::cudart CUDA::cublas cudnn)

add_executable(test_silero_vad tests/test_silero_vad.cpp)
target_include_directories(test_silero_vad PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_silero_vad PRIVATE sensus machina communis CUDA::cudart CUDA::cublas cudnn)

add_executable(test_fsmn_vad tests/test_fsmn_vad.cpp)
target_include_directories(test_fsmn_vad PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_fsmn_vad PRIVATE sensus machina communis CUDA::cudart CUDA::cublas cudnn)
