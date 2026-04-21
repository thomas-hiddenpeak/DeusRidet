/**
 * @file src/machina/forward_decode_main.cu
 * @philosophical_role
 *   Peer TU of forward.cu under R1 800-line hard cap — single-token decode forward (with CUDA Graph + sampled variant).
 * @serves
 *   Conscientia + Actus diagnostic verbs.
 */
#include "forward.h"
#include "layer.h"
#include "gptq.h"
#include "gptq_gemm_v2.h"
#include "marlin.h"
#include "fp16_gemm.h"
#include "../communis/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

namespace deusridet {

// ============================================================================
// Complete single-token forward pass
//
// embed → for each layer: norm → attn/deltanet → residual → norm → MLP → residual
// → final_norm → lm_head → greedy sample
//
// Convention: after attn/deltanet, output is in state.norm_out[0..hidden-1]
//             after MLP, output is in state.mlp_down[0..hidden-1]
// ============================================================================

// ============================================================================
// Graph-capturable forward body
//
// Contains ALL GPU operations for a single decode token, with NO host sync.
// Reads token_id and pos from pinned staging buffers (set by caller before
// graph launch). This function is called once during graph capture, then
// the captured graph is replayed for all subsequent tokens.
//
// Prerequisite: h_token_pinned and h_pos_pinned are set by the caller.
// ============================================================================

static void forward_body(const ModelWeights& model,
                         InferenceState& state,
                         __half* kv_cache,
                         int max_kv_len,
                         cudaStream_t stream) {
    using MC = ModelConfig;

    // H2D from pinned staging (graph replays read current values from pinned memory)
    cudaMemcpyAsync(state.token_ids, state.h_token_pinned, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(state.d_pos, state.h_pos_pinned, sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Embedding lookup → hidden[0, 0..5119]
    embedding_lookup(model.embed_tokens, state.token_ids,
                     state.hidden, 1, MC::HIDDEN_SIZE, stream);

    // Copy to residual for first layer
    cudaMemcpyAsync(state.residual, state.hidden,
                    MC::HIDDEN_SIZE * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream);

    int dn_layer_idx = 0;  // Counter for DeltaNet layers (0..47)

    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];

        // Pre-attention RMSNorm
        rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                 1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

        // Attention / DeltaNet
        if (lw.is_full_attention) {
            full_attention_forward(state.norm_out, lw.full_attn,
                                   kv_cache, layer, 0 /*pos read from d_pos*/, max_kv_len,
                                   state, stream);
        } else {
            deltanet_forward(state.norm_out, lw.delta_net,
                             dn_layer_idx, state, stream);
            dn_layer_idx++;
        }

        // Fused: residual += attn_output; norm_out = RMSNorm(residual)
        residual_rms_norm(state.residual, state.norm_out,
                          lw.post_attn_layernorm, state.norm_out,
                          1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

        // MLP with fused residual add: residual += down_proj(silu(gate)*up)
        mlp_forward(state.norm_out, lw.mlp, state.residual, state, stream);
    }

    // Final RMSNorm
    rms_norm(state.residual, model.final_norm, state.hidden,
             1, MC::HIDDEN_SIZE, MC::RMS_EPS, stream);

    // LM head: hidden[5120] → logits[248320] (INT8 quantized — halves weight read)
    int8_linear_forward(state.hidden, model.lm_head_int8, state.logits, 1, stream);

    // GPU argmax (no sync — result extracted by caller after graph launch)
    argmax_async(state.logits, MC::VOCAB_SIZE, state.sample_out, stream);
}

// ============================================================================
// Single-token forward pass with CUDA Graph acceleration
//
// First call: captures the entire forward body into a CUDA Graph.
// Subsequent calls: replay the graph (eliminates ~1400 kernel launch overhead).
//
// Per-token-changing parameters (token_id, pos) are passed via pinned host
// staging → H2D memcpy (captured in graph; reads current pinned values at replay).
// ============================================================================

int forward_one_token(const ModelWeights& model,
                      InferenceState& state,
                      __half* kv_cache,
                      int token_id, int pos, int max_kv_len,
                      cudaStream_t stream) {
    // Use non-default compute stream for graph capture/replay
    // (cudaStreamBeginCapture requires non-default stream)
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // Write per-token values to pinned staging buffers
    *state.h_token_pinned = token_id;
    *state.h_pos_pinned = pos;

    if (!state.graph_captured) {
        // First invocation: capture CUDA Graph
        cudaError_t err;

        err = cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            LOG_ERROR("Machina", "Graph BeginCapture failed: %s", cudaGetErrorString(err));
        }

        forward_body(model, state, kv_cache, max_kv_len, s);

        err = cudaStreamEndCapture(s, &state.cuda_graph);
        if (err != cudaSuccess) {
            LOG_ERROR("Machina", "Graph EndCapture failed: %s", cudaGetErrorString(err));
        }

        // Check graph node count
        size_t num_nodes = 0;
        cudaGraphGetNodes(state.cuda_graph, nullptr, &num_nodes);
        LOG_INFO("Machina", "CUDA Graph captured: %zu nodes", num_nodes);

        err = cudaGraphInstantiate(&state.cuda_graph_exec, state.cuda_graph, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Machina", "Graph Instantiate failed: %s", cudaGetErrorString(err));
        }

        state.graph_captured = true;
    }

    // Replay captured graph (pinned staging contains current token_id & pos)
    cudaError_t err = cudaGraphLaunch(state.cuda_graph_exec, s);
    if (err != cudaSuccess) {
        LOG_ERROR("Machina", "Graph Launch failed: %s", cudaGetErrorString(err));
    }

    // Extract result (outside graph — single D2H + sync)
    int result;
    cudaMemcpyAsync(&result, state.sample_out, sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    return result;
}

// ============================================================================
// Single-token forward pass with configurable sampling (no CUDA Graph)
//
// Runs the full forward body then applies top-k/top-p sampling.
// Intended for generation with non-greedy sampling strategies.
// ============================================================================

int forward_one_token_sampled(const ModelWeights& model,
                              InferenceState& state,
                              __half* kv_cache,
                              int token_id, int pos, int max_kv_len,
                              const SamplingParams& params,
                              cudaStream_t stream) {
    using MC = ModelConfig;
    cudaStream_t s = state.compute_stream ? state.compute_stream : stream;

    // Copy token_id and pos to device
    cudaMemcpyAsync(state.token_ids, &token_id, sizeof(int),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(state.d_pos, &pos, sizeof(int),
                    cudaMemcpyHostToDevice, s);

    // Embedding lookup → hidden
    embedding_lookup(model.embed_tokens, state.token_ids,
                     state.hidden, 1, MC::HIDDEN_SIZE, s);

    // Copy to residual
    cudaMemcpyAsync(state.residual, state.hidden,
                    MC::HIDDEN_SIZE * sizeof(__half),
                    cudaMemcpyDeviceToDevice, s);

    int dn_layer_idx = 0;
    for (int layer = 0; layer < MC::NUM_LAYERS; layer++) {
        const LayerWeights& lw = model.layers[layer];

        rms_norm(state.residual, lw.input_layernorm, state.norm_out,
                 1, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        if (lw.is_full_attention) {
            full_attention_forward(state.norm_out, lw.full_attn,
                                   kv_cache, layer, pos, max_kv_len,
                                   state, s);
        } else {
            deltanet_forward(state.norm_out, lw.delta_net,
                             dn_layer_idx, state, s);
            dn_layer_idx++;
        }

        residual_rms_norm(state.residual, state.norm_out,
                          lw.post_attn_layernorm, state.norm_out,
                          1, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

        mlp_forward(state.norm_out, lw.mlp, state.residual, state, s);
    }

    rms_norm(state.residual, model.final_norm, state.hidden,
             1, MC::HIDDEN_SIZE, MC::RMS_EPS, s);

    int8_linear_forward(state.hidden, model.lm_head_int8, state.logits, 1, s);

    // Top-k/top-p sampling
    unsigned long long seed = params.seed ? params.seed : ++state.rng_counter;
    sample_top_k_top_p(state.logits, state.probs, MC::VOCAB_SIZE,
                        params, seed, state.sample_out, s);

    int result;
    cudaMemcpyAsync(&result, state.sample_out, sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    return result;
}


} // namespace deusridet
