/**
 * @file model_weights_loader.cpp
 * @philosophical_role Weight incarnation — turns on-disk bytes into a living body. Every tensor name is a coordinate in the model's anatomy; dispatch_tensor is the midwife that puts each bone in place.
 * @serves load_model_weights — shard streaming + pool allocation + per-tensor dispatch.
 */
// model_weights_loader.cpp — tensor-name parsing, device dispatch, and the
// public load_model_weights() that streams multi-shard safetensors into
// pool-allocated device memory. Peer TU of model.cpp under R1 500-line cap.

#include "model.h"
#include "layer.h"
#include "marlin.h"
#include "fp16_gemm.h"
#include "safetensors.h"
#include "allocator.h"
#include "convert.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <filesystem>

namespace deusridet {

// ============================================================================
// Tensor name classification and parsing
// ============================================================================

static const char* LAYER_PREFIX = "model.language_model.layers.";
static const size_t LAYER_PREFIX_LEN = 28;  // strlen("model.language_model.layers.")

// Parse layer index from "model.language_model.layers.{i}...."
static int parse_layer_idx(const char* name) {
    return atoi(name + LAYER_PREFIX_LEN);
}

// Extract suffix after "model.language_model.layers.{i}."
static const char* layer_suffix(const char* name) {
    const char* p = name + LAYER_PREFIX_LEN;
    while (*p >= '0' && *p <= '9') p++;
    if (*p == '.') p++;
    return p;
}

// Check if a tensor name is part of the model we load
static bool is_model_tensor(const std::string& name) {
    if (name == "model.language_model.embed_tokens.weight") return true;
    if (name == "lm_head.weight") return true;
    if (name == "model.language_model.norm.weight") return true;

    if (name.compare(0, LAYER_PREFIX_LEN, LAYER_PREFIX) != 0) return false;

    const char* suffix = layer_suffix(name.c_str());

    // Layer norms
    if (strcmp(suffix, "input_layernorm.weight") == 0) return true;
    if (strcmp(suffix, "post_attention_layernorm.weight") == 0) return true;
    // Full Attention
    if (strcmp(suffix, "self_attn.q_proj.weight") == 0) return true;
    if (strcmp(suffix, "self_attn.k_proj.weight") == 0) return true;
    if (strcmp(suffix, "self_attn.v_proj.weight") == 0) return true;
    if (strcmp(suffix, "self_attn.o_proj.weight") == 0) return true;
    if (strcmp(suffix, "self_attn.q_norm.weight") == 0) return true;
    if (strcmp(suffix, "self_attn.k_norm.weight") == 0) return true;
    // DeltaNet
    if (strcmp(suffix, "linear_attn.in_proj_qkv.weight") == 0) return true;
    if (strcmp(suffix, "linear_attn.in_proj_z.weight") == 0) return true;
    if (strcmp(suffix, "linear_attn.in_proj_a.weight") == 0) return true;
    if (strcmp(suffix, "linear_attn.in_proj_b.weight") == 0) return true;
    if (strcmp(suffix, "linear_attn.out_proj.weight") == 0) return true;
    if (strcmp(suffix, "linear_attn.conv1d.weight") == 0) return true;
    if (strcmp(suffix, "linear_attn.A_log") == 0) return true;
    if (strcmp(suffix, "linear_attn.dt_bias") == 0) return true;
    if (strcmp(suffix, "linear_attn.norm.weight") == 0) return true;
    // MLP GPTQ
    if (strcmp(suffix, "mlp.gate_proj.qweight") == 0) return true;
    if (strcmp(suffix, "mlp.gate_proj.scales") == 0) return true;
    if (strcmp(suffix, "mlp.up_proj.qweight") == 0) return true;
    if (strcmp(suffix, "mlp.up_proj.scales") == 0) return true;
    if (strcmp(suffix, "mlp.down_proj.qweight") == 0) return true;
    if (strcmp(suffix, "mlp.down_proj.scales") == 0) return true;
    // MLP FP16/BF16 (unquantized models)
    if (strcmp(suffix, "mlp.gate_proj.weight") == 0) return true;
    if (strcmp(suffix, "mlp.up_proj.weight") == 0) return true;
    if (strcmp(suffix, "mlp.down_proj.weight") == 0) return true;

    return false;
}

// ============================================================================
// Dispatch: assign device pointer to the correct ModelWeights field
// ============================================================================

// Set Linear fields from a weight tensor (BF16/FP16 → FP16)
static void assign_linear(Linear& linear, void* d_ptr, const Tensor& tensor) {
    linear.weight = static_cast<__half*>(d_ptr);
    linear.out_features = (int)tensor.shape()[0];
    linear.in_features  = (int)tensor.shape()[1];
}

// Load FP16/BF16 weight, then quantize to INT8 per-channel.
// d_ptr holds the temporary FP16 data (from copy_tensor_to_device).
static void assign_int8_linear(Int8Linear& int8, void* d_ptr, const Tensor& tensor,
                               cudaStream_t stream) {
    int N = (int)tensor.shape()[0];
    int K = (int)tensor.shape()[1];
    quantize_fp16_to_int8(static_cast<const __half*>(d_ptr), int8, N, K, stream);
}

// Save FP16 pointer from pool (no extra alloc — data lives in pool_blocks)
static void assign_fp16_linear(Linear& lin, void* d_ptr, const Tensor& tensor) {
    lin.weight = static_cast<__half*>(d_ptr);
    lin.out_features = (int)tensor.shape()[0];
    lin.in_features  = (int)tensor.shape()[1];
}

// Set GPTQ qweight fields
static void assign_gptq_qweight(GptqWeight& gptq, void* d_ptr, const Tensor& tensor) {
    gptq.qweight = static_cast<const uint32_t*>(d_ptr);
    gptq.K = (int)tensor.shape()[0] * 8;  // unpack INT4
    gptq.N = (int)tensor.shape()[1];
}

// Set GPTQ scales fields
static void assign_gptq_scales(GptqWeight& gptq, void* d_ptr) {
    gptq.scales = static_cast<const __half*>(d_ptr);
}

// Copy tensor data to device.  All data types are raw-copied first.
// BF16→FP16 and +1 (RMSNorm) transforms run as GPU kernels after the copy.
static void copy_tensor_to_device(void* d_ptr, const Tensor& tensor,
                                  bool add_one, cudaStream_t stream) {
    DataType dt = tensor.dtype();

    // Always raw memcpy first — uniform throughput for all dtypes
    cudaMemcpyAsync(d_ptr, tensor.data(), tensor.nbytes(),
                    cudaMemcpyHostToDevice, stream);

    // Post-copy GPU transforms
    if (dt == DataType::BF16) {
        bf16_to_fp16_gpu(d_ptr, tensor.numel(), add_one, stream);
    } else if (dt == DataType::FP16 && add_one) {
        fp16_add_one_gpu(static_cast<__half*>(d_ptr), tensor.numel(), stream);
    }
}

// Dispatch a tensor: assign pointer to ModelWeights field + copy data
static void dispatch_tensor(const std::string& name, Tensor& tensor,
                            void* d_ptr, ModelWeights& weights,
                            cudaStream_t stream) {
    // Global tensors
    if (name == "model.language_model.embed_tokens.weight") {
        weights.embed_tokens = static_cast<__half*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        return;
    }
    if (name == "lm_head.weight") {
        weights.lm_head = static_cast<__half*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        // Quantize lm_head to INT8 for decode GEMV (halves weight data: 2.54GB → 1.27GB)
        quantize_fp16_to_int8(weights.lm_head, weights.lm_head_int8,
                              ModelConfig::VOCAB_SIZE, ModelConfig::HIDDEN_SIZE, stream);
        return;
    }
    if (name == "model.language_model.norm.weight") {
        weights.final_norm = static_cast<__half*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, true, stream);  // +1
        return;
    }

    // Layer tensors
    int li = parse_layer_idx(name.c_str());
    const char* suffix = layer_suffix(name.c_str());
    auto& lw = weights.layers[li];

    // Layer norms (BF16 → FP16, +1)
    if (strcmp(suffix, "input_layernorm.weight") == 0) {
        lw.input_layernorm = static_cast<__half*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, true, stream);
    }
    else if (strcmp(suffix, "post_attention_layernorm.weight") == 0) {
        lw.post_attn_layernorm = static_cast<__half*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, true, stream);
    }
    // Full Attention
    else if (strcmp(suffix, "self_attn.q_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.full_attn.fp16_q, d_ptr, tensor);
    }
    else if (strcmp(suffix, "self_attn.k_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.full_attn.fp16_k, d_ptr, tensor);
    }
    else if (strcmp(suffix, "self_attn.v_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.full_attn.fp16_v, d_ptr, tensor);
    }
    else if (strcmp(suffix, "self_attn.o_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.full_attn.fp16_o, d_ptr, tensor);
    }
    else if (strcmp(suffix, "self_attn.q_norm.weight") == 0) {
        lw.full_attn.q_norm = static_cast<__half*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, true, stream);  // +1
    }
    else if (strcmp(suffix, "self_attn.k_norm.weight") == 0) {
        lw.full_attn.k_norm = static_cast<__half*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, true, stream);  // +1
    }
    // DeltaNet
    else if (strcmp(suffix, "linear_attn.in_proj_qkv.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.delta_net.fp16_qkv, d_ptr, tensor);
    }
    else if (strcmp(suffix, "linear_attn.in_proj_z.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.delta_net.fp16_z, d_ptr, tensor);
    }
    else if (strcmp(suffix, "linear_attn.in_proj_a.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.delta_net.fp16_a, d_ptr, tensor);
    }
    else if (strcmp(suffix, "linear_attn.in_proj_b.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.delta_net.fp16_b, d_ptr, tensor);
    }
    else if (strcmp(suffix, "linear_attn.out_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.delta_net.fp16_out, d_ptr, tensor);
    }
    else if (strcmp(suffix, "linear_attn.conv1d.weight") == 0) {
        lw.delta_net.conv1d_weight = static_cast<__half*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
    }
    else if (strcmp(suffix, "linear_attn.A_log") == 0) {
        lw.delta_net.A_log = static_cast<float*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
    }
    else if (strcmp(suffix, "linear_attn.dt_bias") == 0) {
        lw.delta_net.dt_bias = static_cast<__half*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
    }
    else if (strcmp(suffix, "linear_attn.norm.weight") == 0) {
        lw.delta_net.norm_weight = static_cast<float*>(d_ptr);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
    }
    // MLP GPTQ
    else if (strcmp(suffix, "mlp.gate_proj.qweight") == 0) {
        assign_gptq_qweight(lw.mlp.gate_proj, d_ptr, tensor);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
    }
    else if (strcmp(suffix, "mlp.gate_proj.scales") == 0) {
        assign_gptq_scales(lw.mlp.gate_proj, d_ptr);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
    }
    else if (strcmp(suffix, "mlp.up_proj.qweight") == 0) {
        assign_gptq_qweight(lw.mlp.up_proj, d_ptr, tensor);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
    }
    else if (strcmp(suffix, "mlp.up_proj.scales") == 0) {
        assign_gptq_scales(lw.mlp.up_proj, d_ptr);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
    }
    else if (strcmp(suffix, "mlp.down_proj.qweight") == 0) {
        assign_gptq_qweight(lw.mlp.down_proj, d_ptr, tensor);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
    }
    else if (strcmp(suffix, "mlp.down_proj.scales") == 0) {
        assign_gptq_scales(lw.mlp.down_proj, d_ptr);
        copy_tensor_to_device(d_ptr, tensor, false, stream);
    }
    // MLP FP16/BF16 (unquantized models)
    else if (strcmp(suffix, "mlp.gate_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.mlp.fp16_gate_proj, d_ptr, tensor);
    }
    else if (strcmp(suffix, "mlp.up_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.mlp.fp16_up_proj, d_ptr, tensor);
    }
    else if (strcmp(suffix, "mlp.down_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_fp16_linear(lw.mlp.fp16_down_proj, d_ptr, tensor);
    }
}

// ============================================================================
// Public API: load_model_weights (stream_load + pool allocation)
// ============================================================================

bool load_model_weights(const std::string& model_dir, ModelWeights& weights,
                        bool repack_marlin_flag) {
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    using MC = ModelConfig;

    // Initialize model config from config.json
    if (!MC::initialized()) {
        if (!MC::init(model_dir)) {
            LOG_ERROR("Model", "Failed to initialize ModelConfig from %s", model_dir.c_str());
            return false;
        }
    }

    LOG_INFO("Model", "Loading Qwen3.5 weights from %s", model_dir.c_str());
    LOG_INFO("Model", "  Strategy: shard-streaming + pool allocation + async copy");

    // Resize layers vector
    weights.layers.resize(MC::NUM_LAYERS);

    // Pre-init is_full_attention flags
    for (int i = 0; i < MC::NUM_LAYERS; i++)
        weights.layers[i].is_full_attention = MC::is_full_attention(i);

    // Auto-detect GPTQ vs BF16 MLP from config.json (quantization_config presence)
    {
        std::string config_path = model_dir + "/config.json";
        std::ifstream qcheck(config_path);
        if (qcheck) {
            std::string content((std::istreambuf_iterator<char>(qcheck)),
                                 std::istreambuf_iterator<char>());
            if (content.find("\"quant_method\"") != std::string::npos &&
                content.find("\"gptq\"") != std::string::npos) {
                MC::MLP_IS_GPTQ = true;
                LOG_INFO("Model", "  MLP format: GPTQ-Int4 (detected quantization_config)");
            } else {
                MC::MLP_IS_GPTQ = false;
                LOG_INFO("Model", "  MLP format: FP16/BF16 Linear (no quantization_config)");
            }
        }
    }

    // Dedicated stream for async H2D copies
    cudaStream_t copy_stream;
    cudaStreamCreate(&copy_stream);

    bool ok = true;
    int total_tensors = 0;
    int total_cudamallocs = 0;

    SafetensorsLoader::stream_load(model_dir,
        [&](size_t shard_idx, SafetensorsFile& shard) {
            if (!ok) return;
            auto shard_t0 = Clock::now();
            auto all_names = shard.tensor_names();

            // Filter to model tensors only (skip MTP, visual, etc.)
            std::vector<std::string> names;
            names.reserve(all_names.size());
            for (const auto& n : all_names)
                if (is_model_tensor(n)) names.push_back(n);

            if (names.empty()) return;

            // First pass: compute total device bytes for this shard (256-byte aligned)
            size_t shard_total = 0;
            for (const auto& n : names) {
                auto tensor = shard.get_tensor(n);
                size_t aligned = (tensor->nbytes() + 255) & ~(size_t)255;
                shard_total += aligned;
            }

            // Single cudaMalloc for the entire shard
            void* pool_base = nullptr;
            cudaError_t err = cudaMalloc(&pool_base, shard_total);
            if (err != cudaSuccess) {
                LOG_ERROR("Model", "cudaMalloc failed for shard %zu (%.1f MB): %s",
                          shard_idx, shard_total / 1048576.0, cudaGetErrorString(err));
                ok = false;
                return;
            }
            weights.pool_blocks.push_back(pool_base);
            total_cudamallocs++;

            // Second pass: sub-allocate, dispatch, and async-copy each tensor
            size_t offset = 0;
            for (const auto& n : names) {
                auto tensor = shard.get_tensor(n);
                size_t nbytes = tensor->nbytes();
                void* d_ptr = static_cast<char*>(pool_base) + offset;

                dispatch_tensor(n, *tensor, d_ptr, weights, copy_stream);

                size_t aligned = (nbytes + 255) & ~(size_t)255;
                offset += aligned;
                total_tensors++;
            }

            weights.total_bytes += shard_total;

            // Wait for all copies from this shard before mmap is released.
            // When callback returns, ~SafetensorsFile triggers munmap + FADV_DONTNEED,
            // reclaiming physical pages for future cudaMalloc use on Tegra.
            cudaStreamSynchronize(copy_stream);

            double shard_sec = std::chrono::duration<double>(Clock::now() - shard_t0).count();
            LOG_INFO("Model", "  Shard %2zu: %3zu tensors, %7.1f MB, 1 cudaMalloc, %.1fs (%.0f MB/s)",
                     shard_idx, names.size(), shard_total / 1048576.0,
                     shard_sec, (shard_total / 1048576.0) / shard_sec);
        });

    cudaStreamDestroy(copy_stream);

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    if (ok) {
        LOG_INFO("Model", "All weights loaded: %.2f GB in %d cudaMalloc calls, %.1fs (%.0f MB/s)",
                 weights.total_bytes / 1073741824.0, total_cudamallocs,
                 elapsed, (weights.total_bytes / 1048576.0) / elapsed);

        // Repack GPTQ weights to Marlin tile format (in-place, GPTQ only)
        if (repack_marlin_flag && MC::MLP_IS_GPTQ)
            repack_all_marlin(weights);
    } else {
        LOG_ERROR("Model", "Weight loading FAILED");
    }

    return ok;
}

} // namespace deusridet
