// model.cpp — Qwen3.5-27B weight loading
//
// Loads weights from multi-shard safetensors into device memory.
// BF16 → FP16 conversion at load time. RMSNorm weights precomputed as (1+w).
// Pool allocation: one cudaMalloc per shard, sub-allocate tensors from pool.

#include "model.h"
#include "safetensors.h"
#include "allocator.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <unordered_map>

namespace deusridet {

// ============================================================================
// BF16 → FP16 conversion helpers
// ============================================================================

static inline float bf16_to_float(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Convert BF16 buffer to FP16 in-place on host, then copy to device.
// Also optionally adds 1.0 to each element (for RMSNorm precompute).
static void load_bf16_to_fp16(const void* src, __half* dst_device,
                              size_t numel, bool add_one = false) {
    std::vector<__half> host(numel);
    const uint16_t* bf16 = static_cast<const uint16_t*>(src);
    for (size_t i = 0; i < numel; i++) {
        float val = bf16_to_float(bf16[i]);
        if (add_one) val += 1.0f;
        host[i] = __float2half(val);
    }
    cudaMemcpy(dst_device, host.data(), numel * sizeof(__half),
               cudaMemcpyHostToDevice);
}

// Copy F32 buffer directly to device F32.
static void load_f32(const void* src, float* dst_device, size_t numel) {
    cudaMemcpy(dst_device, src, numel * sizeof(float), cudaMemcpyHostToDevice);
}

// Copy F16 buffer directly to device.
static void load_f16(const void* src, __half* dst_device, size_t numel) {
    cudaMemcpy(dst_device, src, numel * sizeof(__half), cudaMemcpyHostToDevice);
}

// Copy I32 buffer directly to device.
static void load_i32(const void* src, uint32_t* dst_device, size_t numel) {
    cudaMemcpy(dst_device, src, numel * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

// ============================================================================
// Device memory allocation with tracking
// ============================================================================

struct DevicePool {
    size_t total = 0;

    __half* alloc_fp16(size_t numel) {
        size_t bytes = numel * sizeof(__half);
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            LOG_ERROR("Model", "cudaMalloc failed (%zu bytes): %s",
                      bytes, cudaGetErrorString(err));
            return nullptr;
        }
        total += bytes;
        return static_cast<__half*>(ptr);
    }

    float* alloc_f32(size_t numel) {
        size_t bytes = numel * sizeof(float);
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            LOG_ERROR("Model", "cudaMalloc failed (%zu bytes): %s",
                      bytes, cudaGetErrorString(err));
            return nullptr;
        }
        total += bytes;
        return static_cast<float*>(ptr);
    }

    uint32_t* alloc_i32(size_t numel) {
        size_t bytes = numel * sizeof(uint32_t);
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            LOG_ERROR("Model", "cudaMalloc failed (%zu bytes): %s",
                      bytes, cudaGetErrorString(err));
            return nullptr;
        }
        total += bytes;
        return static_cast<uint32_t*>(ptr);
    }
};

// ============================================================================
// Tensor name helpers
// ============================================================================

static std::string layer_prefix(int i) {
    return "model.language_model.layers." + std::to_string(i);
}

// ============================================================================
// Load a single Linear (BF16 → FP16)
// ============================================================================

static bool load_linear(SafetensorsLoader& loader, DevicePool& pool,
                        const std::string& name, Linear& linear) {
    std::string wname = name + ".weight";
    if (!loader.has_tensor(wname)) {
        LOG_ERROR("Model", "Missing tensor: %s", wname.c_str());
        return false;
    }
    auto tensor = loader.get_tensor(wname);
    auto& shape = tensor->shape();
    linear.out_features = (int)shape[0];
    linear.in_features  = (int)shape[1];
    size_t numel = (size_t)linear.out_features * linear.in_features;

    linear.weight = pool.alloc_fp16(numel);
    if (!linear.weight) return false;

    if (tensor->dtype() == DataType::BF16) {
        load_bf16_to_fp16(tensor->data(), linear.weight, numel);
    } else if (tensor->dtype() == DataType::FP16) {
        load_f16(tensor->data(), linear.weight, numel);
    } else {
        LOG_ERROR("Model", "Unexpected dtype for %s", wname.c_str());
        return false;
    }
    return true;
}

// ============================================================================
// Load GPTQ weight (qweight I32 + scales F16)
// ============================================================================

static bool load_gptq(SafetensorsLoader& loader, DevicePool& pool,
                      const std::string& name, GptqWeight& gptq) {
    std::string qw_name = name + ".qweight";
    std::string sc_name = name + ".scales";

    if (!loader.has_tensor(qw_name) || !loader.has_tensor(sc_name)) {
        LOG_ERROR("Model", "Missing GPTQ tensors for %s", name.c_str());
        return false;
    }

    auto qw = loader.get_tensor(qw_name);
    auto sc = loader.get_tensor(sc_name);

    int packed_K = (int)qw->shape()[0];
    int N = (int)qw->shape()[1];
    gptq.K = packed_K * 8;  // unpack INT4
    gptq.N = N;

    size_t qw_numel = (size_t)packed_K * N;
    size_t sc_numel = (size_t)sc->shape()[0] * sc->shape()[1];

    gptq.qweight = pool.alloc_i32(qw_numel);
    gptq.scales  = pool.alloc_fp16(sc_numel);
    if (!gptq.qweight || !gptq.scales) return false;

    load_i32(qw->data(), const_cast<uint32_t*>(gptq.qweight), qw_numel);
    load_f16(sc->data(), const_cast<__half*>(gptq.scales), sc_numel);

    return true;
}

// ============================================================================
// Load RMSNorm weight (BF16 → FP16, with optional +1 precompute)
// ============================================================================

static bool load_norm(SafetensorsLoader& loader, DevicePool& pool,
                      const std::string& name, __half*& dst, bool add_one) {
    if (!loader.has_tensor(name)) {
        LOG_ERROR("Model", "Missing tensor: %s", name.c_str());
        return false;
    }
    auto tensor = loader.get_tensor(name);
    size_t numel = tensor->numel();
    dst = pool.alloc_fp16(numel);
    if (!dst) return false;

    if (tensor->dtype() == DataType::BF16) {
        load_bf16_to_fp16(tensor->data(), dst, numel, add_one);
    } else if (tensor->dtype() == DataType::FP16) {
        // F16: convert through float to apply +1 if needed
        if (add_one) {
            std::vector<__half> host(numel);
            const __half* src = static_cast<const __half*>(tensor->data());
            for (size_t i = 0; i < numel; i++) {
                host[i] = __float2half(__half2float(src[i]) + 1.0f);
            }
            cudaMemcpy(dst, host.data(), numel * sizeof(__half),
                       cudaMemcpyHostToDevice);
        } else {
            load_f16(tensor->data(), dst, numel);
        }
    } else {
        LOG_ERROR("Model", "Unexpected dtype for %s", name.c_str());
        return false;
    }
    return true;
}

// ============================================================================
// Load F32 small tensors (A_log, gated norm weight)
// ============================================================================

static bool load_f32_tensor(SafetensorsLoader& loader, DevicePool& pool,
                            const std::string& name, float*& dst) {
    if (!loader.has_tensor(name)) {
        LOG_ERROR("Model", "Missing tensor: %s", name.c_str());
        return false;
    }
    auto tensor = loader.get_tensor(name);
    size_t numel = tensor->numel();
    dst = pool.alloc_f32(numel);
    if (!dst) return false;

    if (tensor->dtype() == DataType::FP32) {
        load_f32(tensor->data(), dst, numel);
    } else {
        LOG_ERROR("Model", "Expected F32 for %s", name.c_str());
        return false;
    }
    return true;
}

// ============================================================================
// Load BF16 small tensors (dt_bias, conv1d_weight) → FP16
// ============================================================================

static bool load_small_bf16(SafetensorsLoader& loader, DevicePool& pool,
                            const std::string& name, __half*& dst) {
    if (!loader.has_tensor(name)) {
        LOG_ERROR("Model", "Missing tensor: %s", name.c_str());
        return false;
    }
    auto tensor = loader.get_tensor(name);
    size_t numel = tensor->numel();
    dst = pool.alloc_fp16(numel);
    if (!dst) return false;

    if (tensor->dtype() == DataType::BF16) {
        load_bf16_to_fp16(tensor->data(), dst, numel);
    } else if (tensor->dtype() == DataType::FP16) {
        load_f16(tensor->data(), dst, numel);
    } else {
        LOG_ERROR("Model", "Unexpected dtype for %s", name.c_str());
        return false;
    }
    return true;
}

// ============================================================================
// Public API: load_model_weights
// ============================================================================

bool load_model_weights(const std::string& model_dir, ModelWeights& weights) {
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    LOG_INFO("Model", "Loading Qwen3.5-27B weights from %s", model_dir.c_str());

    SafetensorsLoader loader(model_dir);
    DevicePool pool;
    bool ok = true;
    using MC = ModelConfig;

    // --- Embedding + LM head (BF16, large) ---
    {
        size_t embed_numel = (size_t)MC::VOCAB_SIZE * MC::HIDDEN_SIZE;
        weights.embed_tokens = pool.alloc_fp16(embed_numel);
        weights.lm_head      = pool.alloc_fp16(embed_numel);
        if (!weights.embed_tokens || !weights.lm_head) return false;

        auto et = loader.get_tensor("model.language_model.embed_tokens.weight");
        load_bf16_to_fp16(et->data(), weights.embed_tokens, embed_numel);
        LOG_INFO("Model", "  embed_tokens: [%d, %d] → FP16 (%.1f MB)",
                 MC::VOCAB_SIZE, MC::HIDDEN_SIZE,
                 embed_numel * 2 / 1048576.0);

        auto lh = loader.get_tensor("lm_head.weight");
        load_bf16_to_fp16(lh->data(), weights.lm_head, embed_numel);
        LOG_INFO("Model", "  lm_head: [%d, %d] → FP16 (%.1f MB)",
                 MC::VOCAB_SIZE, MC::HIDDEN_SIZE,
                 embed_numel * 2 / 1048576.0);
    }

    // --- Final norm ---
    ok = ok && load_norm(loader, pool,
                         "model.language_model.norm.weight",
                         weights.final_norm, /*add_one=*/true);

    // --- Per-layer weights ---
    for (int i = 0; i < MC::NUM_LAYERS && ok; i++) {
        std::string lp = layer_prefix(i);
        LayerWeights& lw = weights.layers[i];
        lw.is_full_attention = MC::is_full_attention(i);

        // Layer norms (both layer types have these)
        ok = ok && load_norm(loader, pool, lp + ".input_layernorm.weight",
                             lw.input_layernorm, true);
        ok = ok && load_norm(loader, pool, lp + ".post_attention_layernorm.weight",
                             lw.post_attn_layernorm, true);

        // Attention weights
        if (lw.is_full_attention) {
            auto& fa = lw.full_attn;
            ok = ok && load_linear(loader, pool, lp + ".self_attn.q_proj", fa.q_proj);
            ok = ok && load_linear(loader, pool, lp + ".self_attn.k_proj", fa.k_proj);
            ok = ok && load_linear(loader, pool, lp + ".self_attn.v_proj", fa.v_proj);
            ok = ok && load_linear(loader, pool, lp + ".self_attn.o_proj", fa.o_proj);
            ok = ok && load_norm(loader, pool, lp + ".self_attn.q_norm.weight",
                                 fa.q_norm, true);
            ok = ok && load_norm(loader, pool, lp + ".self_attn.k_norm.weight",
                                 fa.k_norm, true);
        } else {
            auto& dn = lw.delta_net;
            ok = ok && load_linear(loader, pool, lp + ".linear_attn.in_proj_qkv", dn.in_proj_qkv);
            ok = ok && load_linear(loader, pool, lp + ".linear_attn.in_proj_z", dn.in_proj_z);
            ok = ok && load_linear(loader, pool, lp + ".linear_attn.in_proj_a", dn.in_proj_a);
            ok = ok && load_linear(loader, pool, lp + ".linear_attn.in_proj_b", dn.in_proj_b);
            ok = ok && load_linear(loader, pool, lp + ".linear_attn.out_proj", dn.out_proj);
            ok = ok && load_small_bf16(loader, pool,
                                       lp + ".linear_attn.conv1d.weight",
                                       dn.conv1d_weight);
            ok = ok && load_f32_tensor(loader, pool,
                                       lp + ".linear_attn.A_log", dn.A_log);
            ok = ok && load_small_bf16(loader, pool,
                                       lp + ".linear_attn.dt_bias", dn.dt_bias);
            ok = ok && load_f32_tensor(loader, pool,
                                       lp + ".linear_attn.norm.weight",
                                       dn.norm_weight);
        }

        // MLP (GPTQ quantized, all layers)
        ok = ok && load_gptq(loader, pool, lp + ".mlp.gate_proj", lw.mlp.gate_proj);
        ok = ok && load_gptq(loader, pool, lp + ".mlp.up_proj", lw.mlp.up_proj);
        ok = ok && load_gptq(loader, pool, lp + ".mlp.down_proj", lw.mlp.down_proj);

        if (ok && (i % 8 == 7 || i == MC::NUM_LAYERS - 1)) {
            LOG_INFO("Model", "  Layers 0..%d loaded (%.1f GB device)",
                     i, pool.total / 1073741824.0);
        }
    }

    weights.total_bytes = pool.total;
    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    if (ok) {
        LOG_INFO("Model", "All weights loaded: %.2f GB in %.1fs",
                 pool.total / 1073741824.0, elapsed);
    } else {
        LOG_ERROR("Model", "Weight loading FAILED at %.2f GB",
                  pool.total / 1073741824.0);
    }

    return ok;
}

// ============================================================================
// free_model_weights
// ============================================================================

static void free_linear(Linear& l) {
    if (l.weight) { cudaFree(l.weight); l.weight = nullptr; }
}

static void free_gptq(GptqWeight& g) {
    if (g.qweight) { cudaFree(const_cast<uint32_t*>(g.qweight)); g.qweight = nullptr; }
    if (g.scales)  { cudaFree(const_cast<__half*>(g.scales));     g.scales = nullptr; }
}

void free_model_weights(ModelWeights& w) {
    if (w.embed_tokens) { cudaFree(w.embed_tokens); w.embed_tokens = nullptr; }
    if (w.final_norm)   { cudaFree(w.final_norm);   w.final_norm = nullptr; }
    if (w.lm_head)      { cudaFree(w.lm_head);      w.lm_head = nullptr; }

    for (int i = 0; i < ModelConfig::NUM_LAYERS; i++) {
        auto& lw = w.layers[i];
        if (lw.input_layernorm)    { cudaFree(lw.input_layernorm);    lw.input_layernorm = nullptr; }
        if (lw.post_attn_layernorm) { cudaFree(lw.post_attn_layernorm); lw.post_attn_layernorm = nullptr; }

        if (lw.is_full_attention) {
            free_linear(lw.full_attn.q_proj);
            free_linear(lw.full_attn.k_proj);
            free_linear(lw.full_attn.v_proj);
            free_linear(lw.full_attn.o_proj);
            if (lw.full_attn.q_norm) { cudaFree(lw.full_attn.q_norm); }
            if (lw.full_attn.k_norm) { cudaFree(lw.full_attn.k_norm); }
        } else {
            free_linear(lw.delta_net.in_proj_qkv);
            free_linear(lw.delta_net.in_proj_z);
            free_linear(lw.delta_net.in_proj_a);
            free_linear(lw.delta_net.in_proj_b);
            free_linear(lw.delta_net.out_proj);
            if (lw.delta_net.conv1d_weight) { cudaFree(lw.delta_net.conv1d_weight); }
            if (lw.delta_net.A_log)         { cudaFree(lw.delta_net.A_log); }
            if (lw.delta_net.dt_bias)       { cudaFree(lw.delta_net.dt_bias); }
            if (lw.delta_net.norm_weight)   { cudaFree(lw.delta_net.norm_weight); }
        }

        free_gptq(lw.mlp.gate_proj);
        free_gptq(lw.mlp.up_proj);
        free_gptq(lw.mlp.down_proj);
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
    logits   = alloc_fp16(MC::VOCAB_SIZE);

    // Token IDs on device
    {
        size_t bytes = S * sizeof(int);
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return false;
        cudaMemset(p, 0, bytes);
        token_ids = static_cast<int*>(p);
        total += bytes;
    }

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
    safe_free(dn_b);    safe_free(token_ids);  safe_free(logits);

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
