/**
 * @file model.cpp
 * @philosophical_role The model — weights + shape + tokenizer + forward dispatch, as one addressable object. 'Model' is the loaded body of the entity; destroying it is dying.
 * @serves Actus::load_model, Actus::awaken, Machina forward.
 */
// model.cpp — Qwen3.5 weight loading (multi-model support)
//
// Loads weights from multi-shard safetensors into device memory.
// Supports both GPTQ-Int4 (e.g. 27B) and BF16/FP16 (e.g. 9B) MLP weights.
// BF16 → FP16 conversion at load time. RMSNorm weights precomputed as (1+w).
// Pool allocation: one cudaMalloc per shard, sub-allocate tensors from pool.
// Shard streaming: each shard's mmap is released immediately after copy,
// reclaiming physical pages for GPU use (critical on Tegra unified memory).
//
// ModelConfig is initialized from config.json before weight loading.
// GPTQ vs BF16 MLP format is auto-detected from tensor names.

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
// ModelConfig static member definitions
// ============================================================================

int   ModelConfig::NUM_LAYERS         = 0;
int   ModelConfig::HIDDEN_SIZE        = 0;
int   ModelConfig::INTERMEDIATE_SIZE  = 0;
int   ModelConfig::VOCAB_SIZE         = 0;
float ModelConfig::RMS_EPS            = 1e-6f;

int   ModelConfig::NUM_ATTN_HEADS     = 0;
int   ModelConfig::NUM_KV_HEADS       = 0;
int   ModelConfig::HEAD_DIM           = 0;
int   ModelConfig::NUM_KV_GROUPS      = 0;
int   ModelConfig::Q_PROJ_DIM         = 0;
int   ModelConfig::KV_PROJ_DIM        = 0;
int   ModelConfig::ATTN_OUT_DIM       = 0;
int   ModelConfig::FULL_ATTN_INTERVAL = 0;

float ModelConfig::ROPE_THETA         = 0.0f;
float ModelConfig::PARTIAL_ROTARY     = 0.0f;
int   ModelConfig::ROTARY_DIM         = 0;

int   ModelConfig::LIN_NUM_K_HEADS    = 0;
int   ModelConfig::LIN_NUM_V_HEADS    = 0;
int   ModelConfig::LIN_K_HEAD_DIM     = 0;
int   ModelConfig::LIN_V_HEAD_DIM     = 0;
int   ModelConfig::LIN_KEY_DIM        = 0;
int   ModelConfig::LIN_VALUE_DIM      = 0;
int   ModelConfig::LIN_CONV_DIM       = 0;
int   ModelConfig::CONV_KERNEL        = 0;

int   ModelConfig::LIN_QKV_AB_DIM     = 0;
int   ModelConfig::FA_KV_DIM          = 0;

bool  ModelConfig::MLP_IS_GPTQ        = true;
int   ModelConfig::NUM_FA_LAYERS      = 0;

static bool s_config_initialized = false;

bool ModelConfig::initialized() { return s_config_initialized; }

// ============================================================================
// Minimal JSON value parser for config.json
// ============================================================================

namespace {

// Extract a numeric or string value for a key from JSON text.
// Handles nested objects by searching for the key in text_config section.
static std::string find_json_value(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(":", pos + needle.size());
    if (pos == std::string::npos) return "";
    size_t start = json.find_first_not_of(" \t\n\r", pos + 1);
    if (start == std::string::npos) return "";
    if (json[start] == '"') {
        size_t end = json.find('"', start + 1);
        return json.substr(start + 1, end - start - 1);
    }
    // Numeric/bool value
    size_t end = json.find_first_of(",}\n\r", start);
    std::string val = json.substr(start, end - start);
    // Trim whitespace
    while (!val.empty() && (val.back() == ' ' || val.back() == '\t'))
        val.pop_back();
    return val;
}

static int json_int(const std::string& json, const std::string& key, int def = 0) {
    std::string v = find_json_value(json, key);
    if (v.empty()) return def;
    return std::stoi(v);
}

static float json_float(const std::string& json, const std::string& key, float def = 0.0f) {
    std::string v = find_json_value(json, key);
    if (v.empty()) return def;
    return std::stof(v);
}

} // anonymous namespace

// ============================================================================
// ModelConfig::init — read config.json and compute derived constants
// ============================================================================

bool ModelConfig::init(const std::string& model_dir) {
    std::string config_path = model_dir + "/config.json";
    std::ifstream ifs(config_path);
    if (!ifs) {
        LOG_ERROR("Model", "Cannot open %s", config_path.c_str());
        return false;
    }
    std::string json((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());

    // Find text_config section for nested models
    std::string text_json = json;
    size_t tc_pos = json.find("\"text_config\"");
    if (tc_pos != std::string::npos) {
        size_t brace = json.find("{", tc_pos);
        if (brace != std::string::npos) {
            int depth = 1;
            size_t end = brace + 1;
            while (end < json.size() && depth > 0) {
                if (json[end] == '{') depth++;
                else if (json[end] == '}') depth--;
                end++;
            }
            text_json = json.substr(brace, end - brace);
        }
    }

    // Read primary config values
    NUM_LAYERS         = json_int(text_json, "num_hidden_layers");
    HIDDEN_SIZE        = json_int(text_json, "hidden_size");
    INTERMEDIATE_SIZE  = json_int(text_json, "intermediate_size");
    VOCAB_SIZE         = json_int(text_json, "vocab_size", 248320);
    RMS_EPS            = json_float(text_json, "rms_norm_eps", 1e-6f);

    NUM_ATTN_HEADS     = json_int(text_json, "num_attention_heads");
    NUM_KV_HEADS       = json_int(text_json, "num_key_value_heads");
    HEAD_DIM           = json_int(text_json, "head_dim", 256);
    FULL_ATTN_INTERVAL = json_int(text_json, "full_attention_interval", 4);

    PARTIAL_ROTARY     = json_float(text_json, "partial_rotary_factor", 0.25f);

    // RoPE theta: may be in rope_scaling sub-object or top-level
    std::string rope_section = text_json;
    size_t rs_pos = text_json.find("\"rope_scaling\"");
    if (rs_pos != std::string::npos) {
        size_t rb = text_json.find("{", rs_pos);
        if (rb != std::string::npos) {
            size_t re = text_json.find("}", rb);
            rope_section = text_json.substr(rb, re - rb + 1);
        }
    }
    ROPE_THETA = json_float(rope_section, "rope_theta", 10000000.0f);

    // DeltaNet SSM parameters
    LIN_NUM_K_HEADS   = json_int(text_json, "linear_num_key_heads", 16);
    LIN_NUM_V_HEADS   = json_int(text_json, "linear_num_value_heads");
    LIN_K_HEAD_DIM    = json_int(text_json, "linear_key_head_dim", 128);
    LIN_V_HEAD_DIM    = json_int(text_json, "linear_value_head_dim", 128);
    CONV_KERNEL       = json_int(text_json, "linear_conv_kernel_dim", 4);

    // Compute derived constants
    NUM_KV_GROUPS  = NUM_ATTN_HEADS / NUM_KV_HEADS;
    Q_PROJ_DIM     = NUM_ATTN_HEADS * HEAD_DIM * 2;  // *2 for output gate
    KV_PROJ_DIM    = NUM_KV_HEADS * HEAD_DIM;
    ATTN_OUT_DIM   = NUM_ATTN_HEADS * HEAD_DIM;
    ROTARY_DIM     = (int)(HEAD_DIM * PARTIAL_ROTARY);

    LIN_KEY_DIM    = LIN_NUM_K_HEADS * LIN_K_HEAD_DIM;
    LIN_VALUE_DIM  = LIN_NUM_V_HEADS * LIN_V_HEAD_DIM;
    LIN_CONV_DIM   = LIN_KEY_DIM * 2 + LIN_VALUE_DIM;

    // Merged DN projection: qkv+a+b, padded to 256-element boundary
    int raw_qkv_ab = LIN_CONV_DIM + LIN_NUM_V_HEADS + LIN_NUM_V_HEADS;
    LIN_QKV_AB_DIM = ((raw_qkv_ab + 255) / 256) * 256;

    FA_KV_DIM      = KV_PROJ_DIM + KV_PROJ_DIM;

    // Count FA layers
    NUM_FA_LAYERS = 0;
    for (int i = 0; i < NUM_LAYERS; i++)
        if (is_full_attention(i)) NUM_FA_LAYERS++;

    // MLP_IS_GPTQ will be set later by load_model_weights after tensor name scan
    MLP_IS_GPTQ = true;  // default, overridden during loading

    s_config_initialized = true;

    LOG_INFO("Model", "Config: %d layers (%d FA + %d DN), hidden=%d, inter=%d, heads=%d/%d",
             NUM_LAYERS, NUM_FA_LAYERS, NUM_LAYERS - NUM_FA_LAYERS,
             HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_ATTN_HEADS, NUM_KV_HEADS);
    LOG_INFO("Model", "  DeltaNet: k_heads=%d, v_heads=%d, conv_dim=%d, qkv_ab=%d",
             LIN_NUM_K_HEADS, LIN_NUM_V_HEADS, LIN_CONV_DIM, LIN_QKV_AB_DIM);

    return true;
}

// BF16→FP16 conversion is done on-device via convert.cu kernels.
// Raw BF16 bytes are copied to GPU first, then converted in-place.
// This avoids the scalar CPU loop bottleneck on ARM.

} // namespace deusridet
