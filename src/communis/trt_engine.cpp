// trt_engine.cpp — TensorRT ONNX inference implementation.
//
// Build pipeline: ONNX → NvOnnxParser → TRT INetworkDefinition → IBuilderConfig
// → serialize engine → cache to disk.
//
// Runtime: deserialize engine → create execution context → enqueueV3.
//
// Requires: TensorRT 10.7+, libnvonnxparser, CUDA runtime.

#include "trt_engine.h"
#include "log.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <fstream>
#include <cstring>

namespace deusridet {

// ============================================================================
// TRT Logger adapter
// ============================================================================

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                LOG_ERROR("TRT", "%s", msg);
                break;
            case Severity::kWARNING:
                LOG_WARN("TRT", "%s", msg);
                break;
            case Severity::kINFO:
                // TRT is very verbose at INFO — demote to debug.
                break;
            default:
                break;
        }
    }
};

static TrtLogger& get_trt_logger() {
    static TrtLogger logger;
    return logger;
}

// ============================================================================
// Helpers
// ============================================================================

static size_t dtype_size(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 0;
    }
}

static size_t dims_volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; i++) {
        if (d.d[i] > 0) v *= d.d[i];
    }
    return v;
}

static const char* dtype_name(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return "FP32";
        case nvinfer1::DataType::kHALF:  return "FP16";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kINT64: return "INT64";
        case nvinfer1::DataType::kINT8:  return "INT8";
        case nvinfer1::DataType::kBOOL:  return "BOOL";
        default: return "?";
    }
}

// ============================================================================
// TrtEngine implementation
// ============================================================================

TrtEngine::TrtEngine() = default;

TrtEngine::~TrtEngine() {
    if (context_) { delete context_; context_ = nullptr; }
    if (engine_)  { delete engine_;  engine_  = nullptr; }
    if (runtime_) { delete runtime_; runtime_ = nullptr; }
}

bool TrtEngine::init(const std::string& onnx_path,
                     const std::string& cache_path,
                     const std::vector<TrtDimRange>& dim_ranges) {
    // Try loading cached engine first.
    if (!cache_path.empty() && load_cached(cache_path)) {
        LOG_INFO("TRT", "Loaded cached engine: %s", cache_path.c_str());
    } else {
        // Build from ONNX.
        if (!build_engine(onnx_path, dim_ranges)) {
            return false;
        }
        if (!cache_path.empty()) {
            save_cache(cache_path);
        }
    }

    // Create execution context.
    context_ = engine_->createExecutionContext();
    if (!context_) {
        LOG_ERROR("TRT", "Failed to create execution context");
        return false;
    }

    enumerate_tensors();
    return true;
}

bool TrtEngine::build_engine(const std::string& onnx_path,
                             const std::vector<TrtDimRange>& dim_ranges) {
    auto& logger = get_trt_logger();

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    if (!builder) {
        LOG_ERROR("TRT", "createInferBuilder failed");
        return false;
    }

    // Explicit batch mode.
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0));
    if (!network) {
        LOG_ERROR("TRT", "createNetworkV2 failed");
        return false;
    }

    // Parse ONNX.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        LOG_ERROR("TRT", "Failed to parse ONNX: %s", onnx_path.c_str());
        for (int i = 0; i < parser->getNbErrors(); i++) {
            LOG_ERROR("TRT", "  %s", parser->getError(i)->desc());
        }
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               64 * 1024 * 1024);  // 64 MB workspace

    // Set optimization profile for dynamic shapes.
    if (!dim_ranges.empty()) {
        auto* profile = builder->createOptimizationProfile();

        for (auto& dr : dim_ranges) {
            // Get current network input dims.
            int idx = -1;
            for (int i = 0; i < network->getNbInputs(); i++) {
                if (dr.tensor_name == network->getInput(i)->getName()) {
                    idx = i;
                    break;
                }
            }
            if (idx < 0) {
                LOG_WARN("TRT", "Dynamic dim tensor '%s' not found in network",
                         dr.tensor_name.c_str());
                continue;
            }

            nvinfer1::Dims net_dims = network->getInput(idx)->getDimensions();
            nvinfer1::Dims min_d = net_dims, opt_d = net_dims, max_d = net_dims;
            min_d.d[dr.dim_index] = dr.min_val;
            opt_d.d[dr.dim_index] = dr.opt_val;
            max_d.d[dr.dim_index] = dr.max_val;

            profile->setDimensions(dr.tensor_name.c_str(),
                                   nvinfer1::OptProfileSelector::kMIN, min_d);
            profile->setDimensions(dr.tensor_name.c_str(),
                                   nvinfer1::OptProfileSelector::kOPT, opt_d);
            profile->setDimensions(dr.tensor_name.c_str(),
                                   nvinfer1::OptProfileSelector::kMAX, max_d);
        }
        config->addOptimizationProfile(profile);
    }

    // Build serialized engine.
    LOG_INFO("TRT", "Building engine from %s (this may take a moment)...",
             onnx_path.c_str());
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized || serialized->size() == 0) {
        LOG_ERROR("TRT", "buildSerializedNetwork failed");
        return false;
    }

    // Deserialize.
    runtime_ = nvinfer1::createInferRuntime(logger);
    engine_ = runtime_->deserializeCudaEngine(serialized->data(),
                                              serialized->size());
    if (!engine_) {
        LOG_ERROR("TRT", "deserializeCudaEngine failed");
        return false;
    }

    LOG_INFO("TRT", "Engine built: %zu bytes", serialized->size());
    return true;
}

bool TrtEngine::load_cached(const std::string& cache_path) {
    std::ifstream f(cache_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;

    size_t size = f.tellg();
    if (size == 0) return false;
    f.seekg(0);

    std::vector<char> data(size);
    f.read(data.data(), size);
    if (!f.good()) return false;

    auto& logger = get_trt_logger();
    runtime_ = nvinfer1::createInferRuntime(logger);
    engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
    return engine_ != nullptr;
}

bool TrtEngine::save_cache(const std::string& cache_path) {
    if (!engine_) return false;

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        engine_->serialize());
    if (!serialized) return false;

    std::ofstream f(cache_path, std::ios::binary);
    f.write(static_cast<const char*>(serialized->data()), serialized->size());
    LOG_INFO("TRT", "Cached engine: %s (%zu bytes)",
             cache_path.c_str(), serialized->size());
    return f.good();
}

void TrtEngine::enumerate_tensors() {
    if (!engine_) return;

    num_io_ = engine_->getNbIOTensors();
    tensor_map_.clear();

    for (int i = 0; i < num_io_; i++) {
        const char* name = engine_->getIOTensorName(i);
        TrtTensorInfo info;
        info.name = name;
        info.dtype = engine_->getTensorDataType(name);
        info.is_input = (engine_->getTensorIOMode(name) ==
                         nvinfer1::TensorIOMode::kINPUT);
        info.dims = engine_->getTensorShape(name);
        info.byte_size = dims_volume(info.dims) * dtype_size(info.dtype);
        tensor_map_[name] = info;

        LOG_INFO("TRT", "  %s '%s' %s",
                 info.is_input ? "IN " : "OUT",
                 name, dtype_name(info.dtype));
    }
}

bool TrtEngine::set_input_shape(const char* name,
                                const std::vector<int>& shape) {
    if (!context_) return false;

    nvinfer1::Dims dims;
    dims.nbDims = (int)shape.size();
    for (int i = 0; i < dims.nbDims; i++) {
        dims.d[i] = shape[i];
    }
    bool ok = context_->setInputShape(name, dims);

    // Update cached tensor info.
    auto it = tensor_map_.find(name);
    if (it != tensor_map_.end()) {
        it->second.dims = dims;
        it->second.byte_size = dims_volume(dims) *
                               dtype_size(it->second.dtype);
    }
    return ok;
}

bool TrtEngine::set_tensor_address(const char* name, void* d_ptr) {
    if (!context_) return false;
    return context_->setTensorAddress(name, d_ptr);
}

bool TrtEngine::execute(cudaStream_t stream) {
    if (!context_) return false;
    return context_->enqueueV3(stream);
}

const TrtTensorInfo* TrtEngine::tensor_info(const char* name) const {
    auto it = tensor_map_.find(name);
    return (it != tensor_map_.end()) ? &it->second : nullptr;
}

int TrtEngine::num_io_tensors() const { return num_io_; }

const char* TrtEngine::io_tensor_name(int index) const {
    if (!engine_ || index < 0 || index >= num_io_) return nullptr;
    return engine_->getIOTensorName(index);
}

} // namespace deusridet
