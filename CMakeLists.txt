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
    src/actus/actus.cpp
    src/actus/load_model.cpp
    src/actus/load_weights.cpp
    src/actus/awaken.cpp
    src/actus/awaken_consciousness.cpp
    src/actus/awaken_router.cpp
    src/actus/awaken_router_consciousness.cpp
    src/actus/awaken_hello.cpp
    src/communis/timeline_logger.cpp
    src/communis/tempus.h
    src/sensus/auditus/auditus_facade.h
    src/sensus/auditus/auditus_facade.cpp
    src/sensus/auditus/auditus_facade_broadcasts.cpp
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
# Developer instruments (Step 10a — formerly cmd_bench_* under src/actus/)
# ============================================================================
add_executable(bench_gptq tools/bench_gptq.cpp)
target_include_directories(bench_gptq PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(bench_gptq PRIVATE machina communis CUDA::cudart CUDA::cublas)
set_target_properties(bench_gptq PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(bench_gptq_v2 tools/bench_gptq_v2.cpp)
target_include_directories(bench_gptq_v2 PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(bench_gptq_v2 PRIVATE machina communis CUDA::cudart CUDA::cublas)
set_target_properties(bench_gptq_v2 PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(bench_prefill tools/bench_prefill.cpp)
target_include_directories(bench_prefill PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(bench_prefill PRIVATE machina communis CUDA::cudart CUDA::cublas)
set_target_properties(bench_prefill PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(profile_forward tools/profile_forward.cpp)
target_include_directories(profile_forward PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(profile_forward PRIVATE machina communis CUDA::cudart CUDA::cublas)
set_target_properties(profile_forward PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(profile_prefill tools/profile_prefill.cpp)
target_include_directories(profile_prefill PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(profile_prefill PRIVATE machina communis CUDA::cudart CUDA::cublas)
set_target_properties(profile_prefill PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)
# ============================================================================
add_executable(test_overlap_separator tests/test_overlap_separator.cpp)
target_include_directories(test_overlap_separator PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_overlap_separator PRIVATE sensus communis CUDA::cudart)

# Test: Overlap → Separation → Speaker ID pipeline
# ============================================================================
add_executable(test_overlap_spkid tests/test_overlap_spkid.cpp)
target_include_directories(test_overlap_spkid PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_overlap_spkid PRIVATE sensus orator communis CUDA::cudart CUDA::cublas CUDA::cufft cudnn)

# Test: PyannoteSeg3 native CUDA validation
add_executable(test_pyannote_seg3 tests/test_pyannote_seg3.cpp)
target_include_directories(test_pyannote_seg3 PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_pyannote_seg3 PRIVATE sensus machina communis CUDA::cudart CUDA::cublas cudnn)

add_executable(test_silero_vad tests/test_silero_vad.cpp)
target_include_directories(test_silero_vad PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_silero_vad PRIVATE sensus machina communis CUDA::cudart CUDA::cublas cudnn)

# Tool: dump voiced segments of an audio file via SileroVad (Step 2a GT refinement)
add_executable(silero_vad_segments tools/silero_vad_segments.cpp)
target_include_directories(silero_vad_segments PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(silero_vad_segments PRIVATE sensus machina communis CUDA::cudart CUDA::cublas cudnn)

# Tool: offline FSMN VAD → voiced regions (CAM++'s official companion, Step 2a²)
add_executable(fsmn_vad_segments tools/fsmn_vad_segments.cpp)
target_include_directories(fsmn_vad_segments PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(fsmn_vad_segments PRIVATE sensus machina communis CUDA::cudart CUDA::cublas cudnn)

# Tool: extract CAM++ embeddings for every GT segment x window-strategy (Step 2b)
add_executable(cam_extract_embeddings tools/cam_extract_embeddings.cpp)
target_include_directories(cam_extract_embeddings PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(cam_extract_embeddings PRIVATE sensus machina communis orator CUDA::cudart CUDA::cublas cudnn)

# Tool: extract WavLM+ECAPA embeddings in parallel layout (Step 2c)
add_executable(wl_extract_embeddings tools/wl_extract_embeddings.cpp)
target_include_directories(wl_extract_embeddings PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(wl_extract_embeddings PRIVATE orator sensus machina communis CUDA::cudart CUDA::cublas cudnn)

add_executable(test_fsmn_vad tests/test_fsmn_vad.cpp)
target_include_directories(test_fsmn_vad PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_fsmn_vad PRIVATE sensus machina communis CUDA::cudart CUDA::cublas cudnn)

# ============================================================================
# Integration tests (Step 10d — formerly cmd_test_* under src/actus/)
# Each one was an Actus verb only because the Step 5 split landed every
# CLI subcommand under src/actus/. They are component-level integration
# checks against the Machina engine, so they belong here.
# ============================================================================
add_executable(test_tokenizer tests/integration/test_tokenizer.cpp)
target_include_directories(test_tokenizer PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_tokenizer PRIVATE machina communis CUDA::cudart)
set_target_properties(test_tokenizer PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(test_weights tests/integration/test_weights.cpp)
target_include_directories(test_weights PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_weights PRIVATE machina communis CUDA::cudart CUDA::cublas)
set_target_properties(test_weights PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(test_gptq tests/integration/test_gptq.cpp)
target_include_directories(test_gptq PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_gptq PRIVATE machina communis CUDA::cudart CUDA::cublas)
set_target_properties(test_gptq PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(test_forward tests/integration/test_forward.cpp)
target_include_directories(test_forward PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_forward PRIVATE machina communis CUDA::cudart CUDA::cublas)
set_target_properties(test_forward PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(test_sample tests/integration/test_sample.cpp)
target_include_directories(test_sample PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_sample PRIVATE machina communis CUDA::cudart CUDA::cublas)
set_target_properties(test_sample PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(test_wavlm_cnn tests/integration/test_wavlm_cnn.cpp)
target_include_directories(test_wavlm_cnn PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_wavlm_cnn PRIVATE orator machina communis CUDA::cudart CUDA::cublas cudnn)
set_target_properties(test_wavlm_cnn PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)
