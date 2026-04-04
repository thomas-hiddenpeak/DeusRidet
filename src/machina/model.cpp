// model.cpp — Qwen3.5-27B weight loading
//
// Loads weights from multi-shard safetensors into device memory.
// BF16 → FP16 conversion at load time. RMSNorm weights precomputed as (1+w).
// Pool allocation: one cudaMalloc per shard, sub-allocate tensors from pool.
// Shard streaming: each shard's mmap is released immediately after copy,
// reclaiming physical pages for GPU use (critical on Tegra unified memory).

#include "model.h"
#include "layer.h"
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

namespace deusridet {

// BF16→FP16 conversion is done on-device via convert.cu kernels.
// Raw BF16 bytes are copied to GPU first, then converted in-place.
// This avoids the scalar CPU loop bottleneck on ARM.

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
        assign_int8_linear(lw.full_attn.q_proj, d_ptr, tensor, stream);
    }
    else if (strcmp(suffix, "self_attn.k_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_int8_linear(lw.full_attn.k_proj, d_ptr, tensor, stream);
    }
    else if (strcmp(suffix, "self_attn.v_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_int8_linear(lw.full_attn.v_proj, d_ptr, tensor, stream);
    }
    else if (strcmp(suffix, "self_attn.o_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_int8_linear(lw.full_attn.o_proj, d_ptr, tensor, stream);
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
        assign_int8_linear(lw.delta_net.in_proj_qkv, d_ptr, tensor, stream);
    }
    else if (strcmp(suffix, "linear_attn.in_proj_z.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_int8_linear(lw.delta_net.in_proj_z, d_ptr, tensor, stream);
    }
    else if (strcmp(suffix, "linear_attn.in_proj_a.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_int8_linear(lw.delta_net.in_proj_a, d_ptr, tensor, stream);
    }
    else if (strcmp(suffix, "linear_attn.in_proj_b.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_int8_linear(lw.delta_net.in_proj_b, d_ptr, tensor, stream);
    }
    else if (strcmp(suffix, "linear_attn.out_proj.weight") == 0) {
        copy_tensor_to_device(d_ptr, tensor, false, stream);
        assign_int8_linear(lw.delta_net.out_proj, d_ptr, tensor, stream);
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
}

// ============================================================================
// Public API: load_model_weights (stream_load + pool allocation)
// ============================================================================

bool load_model_weights(const std::string& model_dir, ModelWeights& weights) {
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    using MC = ModelConfig;

    LOG_INFO("Model", "Loading Qwen3.5-27B weights from %s", model_dir.c_str());
    LOG_INFO("Model", "  Strategy: shard-streaming + pool allocation + async copy");

    // Pre-init is_full_attention flags
    for (int i = 0; i < MC::NUM_LAYERS; i++)
        weights.layers[i].is_full_attention = MC::is_full_attention(i);

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
    } else {
        LOG_ERROR("Model", "Weight loading FAILED");
    }

    return ok;
}

// ============================================================================
// free_model_weights — release pool blocks
// ============================================================================

void free_model_weights(ModelWeights& w) {
    // All tensor pointers are sub-allocated from pool blocks.
    // Only free the pool blocks themselves.
    for (void* block : w.pool_blocks) {
        cudaFree(block);
    }
    w.pool_blocks.clear();

    // Free separately-allocated INT8 lm_head (not in pool)
    if (w.lm_head_int8.weight) { cudaFree(w.lm_head_int8.weight); }
    if (w.lm_head_int8.scales) { cudaFree(w.lm_head_int8.scales); }
    w.lm_head_int8 = Int8Linear{};

    // Zero out all pointers (they pointed into freed pool blocks)
    w.embed_tokens = nullptr;
    w.final_norm   = nullptr;
    w.lm_head      = nullptr;
    for (int i = 0; i < ModelConfig::NUM_LAYERS; i++) {
        auto& lw = w.layers[i];
        lw.input_layernorm = nullptr;
        lw.post_attn_layernorm = nullptr;
        lw.full_attn = FullAttentionWeights{};
        lw.delta_net = DeltaNetWeights{};
        lw.mlp = MLPWeights{};
    }
    w.total_bytes = 0;
}

// ============================================================================
// InferenceState allocation
// ============================================================================

bool InferenceState::allocate(int max_seq) {
    using MC = ModelConfig;
    max_seq_len = max_seq;
    size_t total = 0;

    auto alloc_fp16 = [&](size_t numel) -> __half* {
        size_t bytes = numel * sizeof(__half);
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        cudaMemset(p, 0, bytes);
        total += bytes;
        return static_cast<__half*>(p);
    };

    auto alloc_f32 = [&](size_t numel) -> float* {
        size_t bytes = numel * sizeof(float);
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        cudaMemset(p, 0, bytes);
        total += bytes;
        return static_cast<float*>(p);
    };

    size_t S = (size_t)max_seq;
    size_t H = MC::HIDDEN_SIZE;

    hidden   = alloc_fp16(S * H);
    residual = alloc_fp16(S * H);
    norm_out = alloc_fp16(S * H);
    attn_out = alloc_fp16(S * MC::ATTN_OUT_DIM);
    q_buf    = alloc_fp16(S * MC::Q_PROJ_DIM);
    kv_buf   = alloc_fp16(S * MC::KV_PROJ_DIM);
    mlp_gate = alloc_fp16(S * MC::INTERMEDIATE_SIZE);
    mlp_up   = alloc_fp16(S * MC::INTERMEDIATE_SIZE);
    mlp_down = alloc_fp16(S * H);
    dn_qkv   = alloc_fp16(S * MC::LIN_CONV_DIM);
    dn_z     = alloc_fp16(S * MC::LIN_VALUE_DIM);
    dn_a     = alloc_fp16(S * MC::LIN_NUM_V_HEADS);
    dn_b     = alloc_fp16(S * MC::LIN_NUM_V_HEADS);
    dn_g     = alloc_f32(MC::LIN_NUM_V_HEADS);
    dn_beta  = alloc_f32(MC::LIN_NUM_V_HEADS);
    logits   = alloc_fp16(MC::VOCAB_SIZE);
    probs    = alloc_f32(MC::VOCAB_SIZE);  // Sampling workspace (FP32)

    // Full Attention scratch: scores and FP16 conversion buffer
    attn_scores = alloc_f32((size_t)MC::NUM_ATTN_HEADS * S);
    scores_h16  = alloc_fp16((size_t)MC::NUM_KV_GROUPS * S);

    // Token IDs on device
    {
        size_t bytes = S * sizeof(int);
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return false;
        cudaMemset(p, 0, bytes);
        token_ids = static_cast<int*>(p);
        total += bytes;
    }

    // Sample output (single int for GPU argmax)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(int)) != cudaSuccess) return false;
        sample_out = static_cast<int*>(p);
        total += sizeof(int);
    }

    // Device pos for graph-capturable kernels
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(int)) != cudaSuccess) return false;
        d_pos = static_cast<int*>(p);
        total += sizeof(int);
    }

    // Pinned host staging for CUDA Graph
    cudaHostAlloc(&h_pos_pinned, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_token_pinned, sizeof(int), cudaHostAllocDefault);

    // Compute stream for graph capture (cannot capture on default stream 0)
    cudaStreamCreate(&compute_stream);

    // Auxiliary stream for concurrent MLP gate+up projections (prefill only)
    cudaStreamCreate(&aux_stream);
    cudaEventCreateWithFlags(&aux_fork_event, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&aux_join_event, cudaEventDisableTiming);

    // DeltaNet recurrent states: one [128, 128] per head per layer (F32)
    // Count linear attention layers
    num_dn_layers = 0;
    for (int i = 0; i < MC::NUM_LAYERS; i++) {
        if (!MC::is_full_attention(i)) num_dn_layers++;
    }

    size_t state_size = (size_t)MC::LIN_K_HEAD_DIM * MC::LIN_V_HEAD_DIM;  // 128*128
    size_t heads = MC::LIN_NUM_V_HEADS;  // 48

    dn_states = new float*[num_dn_layers];
    for (int i = 0; i < num_dn_layers; i++) {
        dn_states[i] = alloc_f32(heads * state_size);
        if (!dn_states[i]) return false;
    }

    // Conv states: [10240, 3] per linear attention layer (FP16)
    size_t conv_state_size = (size_t)MC::LIN_CONV_DIM * (MC::CONV_KERNEL - 1);
    conv_states = new __half*[num_dn_layers];
    for (int i = 0; i < num_dn_layers; i++) {
        conv_states[i] = alloc_fp16(conv_state_size);
        if (!conv_states[i]) return false;
    }

    if (!hidden || !residual || !norm_out || !logits) {
        LOG_ERROR("Model", "InferenceState allocation failed");
        return false;
    }

    LOG_INFO("Model", "InferenceState: %.1f MB for max_seq=%d (%d DN layers)",
             total / 1048576.0, max_seq, num_dn_layers);
    return true;
}

void InferenceState::free() {
    auto safe_free = [](void* p) { if (p) cudaFree(p); };
    safe_free(hidden);   safe_free(residual);  safe_free(norm_out);
    safe_free(attn_out); safe_free(q_buf);     safe_free(kv_buf);
    safe_free(mlp_gate); safe_free(mlp_up);    safe_free(mlp_down);
    safe_free(dn_qkv);  safe_free(dn_z);      safe_free(dn_a);
    safe_free(dn_b);    safe_free(dn_g);       safe_free(dn_beta);
    safe_free(token_ids);  safe_free(sample_out);  safe_free(logits);  safe_free(probs);
    safe_free(attn_scores); safe_free(scores_h16);
    safe_free(d_pos);
    if (h_pos_pinned) { cudaFreeHost(h_pos_pinned); h_pos_pinned = nullptr; }
    if (h_token_pinned) { cudaFreeHost(h_token_pinned); h_token_pinned = nullptr; }
    if (cuda_graph_exec) { cudaGraphExecDestroy(cuda_graph_exec); cuda_graph_exec = nullptr; }
    if (cuda_graph) { cudaGraphDestroy(cuda_graph); cuda_graph = nullptr; }
    if (compute_stream) { cudaStreamDestroy(compute_stream); compute_stream = nullptr; }
    if (aux_stream) { cudaStreamDestroy(aux_stream); aux_stream = nullptr; }
    if (aux_fork_event) { cudaEventDestroy(aux_fork_event); aux_fork_event = nullptr; }
    if (aux_join_event) { cudaEventDestroy(aux_join_event); aux_join_event = nullptr; }
    graph_captured = false;

    if (dn_states) {
        for (int i = 0; i < num_dn_layers; i++) safe_free(dn_states[i]);
        delete[] dn_states;
        dn_states = nullptr;
    }
    if (conv_states) {
        for (int i = 0; i < num_dn_layers; i++) safe_free(conv_states[i]);
        delete[] conv_states;
        conv_states = nullptr;
    }
    *this = InferenceState{};
}

} // namespace deusridet
