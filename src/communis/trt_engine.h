// trt_engine.h — TensorRT ONNX inference wrapper for Jetson.
//
// Loads an ONNX model, builds a TRT engine (cached to disk), and runs
// inference entirely on GPU. Supports dynamic shapes via optimization
// profiles.
//
// Replaces ONNX Runtime (CPU-only on Jetson) for all model inference.
// TensorRT 10.7 + ONNX parser are pre-installed in JetPack 6.

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace deusridet {

// Dynamic shape range for one dimension of one tensor.
struct TrtDimRange {
    std::string tensor_name;
    int dim_index;          // which dimension (0-based)
    int min_val;
    int opt_val;
    int max_val;
};

// Description of a tensor binding (input or output).
struct TrtTensorInfo {
    std::string name;
    nvinfer1::Dims dims;    // static or profile-resolved dims
    nvinfer1::DataType dtype;
    bool is_input;
    size_t byte_size;       // total bytes at current shape
};

// TensorRT engine wrapper.
// Usage:
//   TrtEngine engine;
//   engine.init("model.onnx", "model.trt", {dim_ranges...});
//   engine.set_input_shape("speech", {1, 10, 400});
//   engine.copy_input("speech", d_data, bytes);
//   engine.execute(stream);
//   engine.copy_output("logits", d_out, bytes);
class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Build or load cached engine. Returns false on failure.
    // onnx_path: path to .onnx model
    // cache_path: path to save/load serialized .trt engine
    // dim_ranges: dynamic shape specifications (empty if all static)
    bool init(const std::string& onnx_path,
              const std::string& cache_path,
              const std::vector<TrtDimRange>& dim_ranges = {});

    // Set runtime shape for a dynamic input tensor.
    bool set_input_shape(const char* name, const std::vector<int>& shape);

    // Set GPU pointer for a tensor (input or output).
    bool set_tensor_address(const char* name, void* d_ptr);

    // Execute inference on the given CUDA stream.
    bool execute(cudaStream_t stream = nullptr);

    // Query tensor info.
    const TrtTensorInfo* tensor_info(const char* name) const;
    int num_io_tensors() const;
    const char* io_tensor_name(int index) const;

    bool initialized() const { return context_ != nullptr; }

private:
    bool build_engine(const std::string& onnx_path,
                      const std::vector<TrtDimRange>& dim_ranges);
    bool load_cached(const std::string& cache_path);
    bool save_cache(const std::string& cache_path);
    void enumerate_tensors();

    nvinfer1::IRuntime* runtime_  = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    std::unordered_map<std::string, TrtTensorInfo> tensor_map_;
    int num_io_ = 0;
};

} // namespace deusridet
