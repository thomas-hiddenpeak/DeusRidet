/**
 * @file src/orator/wavlm_ecapa_extract.cu
 * @philosophical_role
 *   Peer TU of wavlm_ecapa_encoder.cu under R1 800-line hard cap — Full extract pipeline (extract / extract_gpu).
 * @serves
 *   Orator speaker embedding extraction.
 */
#include "wavlm_ecapa_encoder.h"
#include "wavlm_ecapa_kernels.cuh"
#include "../communis/log.h"
#include "../machina/safetensors.h"

#include <cmath>
#include <cstdio>
#include <cassert>

namespace deusridet {

std::vector<float> WavLMEcapaEncoder::extract(const float* pcm, int n_samples) {
    if (!initialized_) return {};
    if (!ensure_scratch(n_samples)) {
        LOG_WARN("WavLMEcapa", "scratch allocation failed for %d samples — skipping extraction", n_samples);
        cudaGetLastError();  // clear sticky CUDA error
        return {};
    }

    // Grow pre-allocated PCM buffer if needed.
    size_t need = (size_t)n_samples;
    if (need > pcm_buf_size_) {
        if (d_pcm_buf_) cudaFree(d_pcm_buf_);
        pcm_buf_size_ = std::max(need, (size_t)(16000 * 10));  // min 10s @ 16kHz
        if (cudaMalloc(&d_pcm_buf_, pcm_buf_size_ * sizeof(float)) != cudaSuccess) {
            LOG_WARN("WavLMEcapa", "PCM buffer alloc failed (%zu bytes)", pcm_buf_size_ * sizeof(float));
            d_pcm_buf_ = nullptr;
            pcm_buf_size_ = 0;
            cudaGetLastError();
            return {};
        }
    }
    cudaMemcpyAsync(d_pcm_buf_, pcm, n_samples * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    return extract_gpu(d_pcm_buf_, n_samples);
}

std::vector<float> WavLMEcapaEncoder::extract_gpu(const float* d_pcm, int n_samples) {
    if (!initialized_) return {};
    if (!ensure_scratch(n_samples)) {
        LOG_WARN("WavLMEcapa", "scratch allocation failed (gpu path) for %d samples — skipping", n_samples);
        cudaGetLastError();
        return {};
    }

    // Timing events for latency breakdown.
    cudaEvent_t ev_start = nullptr, ev_cnn = nullptr, ev_enc = nullptr, ev_done = nullptr;
    if (cudaEventCreate(&ev_start) != cudaSuccess ||
        cudaEventCreate(&ev_cnn)   != cudaSuccess ||
        cudaEventCreate(&ev_enc)   != cudaSuccess ||
        cudaEventCreate(&ev_done)  != cudaSuccess) {
        LOG_WARN("WavLMEcapa", "cudaEventCreate failed — skipping extraction");
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_cnn)   cudaEventDestroy(ev_cnn);
        if (ev_enc)   cudaEventDestroy(ev_enc);
        if (ev_done)  cudaEventDestroy(ev_done);
        cudaGetLastError();
        return {};
    }
    cudaEventRecord(ev_start, stream_);

    int D = WavLMConfig::embed_dim;
    int num_hs = WavLMConfig::num_hidden_states;  // 25

    // ── 1. WavLM CNN + Projection + PosConv + Encoder ──
    int T;
    test_cnn(d_pcm, n_samples, T);
    cudaEventRecord(ev_cnn, stream_);
    test_projection(scratch_a_, T, T);
    test_pos_conv(scratch_a_, T, T);
    test_encoder(scratch_a_, T, T);  // populates d_hidden_states_
    cudaEventRecord(ev_enc, stream_);

    // ── 2. Featurizer: softmax weighted sum of 25 hidden states ──
    float* d_nw = scratch_b_;  // [25] — normalized weights
    softmax_1d_kernel<<<1, 1, 0, stream_>>>(
        w("frontend.featurizer.weights").ptr, d_nw, num_hs);

    float* d_feat = scratch_a_;  // [T, D] — featurizer output
    weighted_sum_kernel<<<div_ceil(T * D, BLOCK), BLOCK, 0, stream_>>>(
        d_hidden_states_, d_nw, d_feat, num_hs, T, D);

    // ── 3. UtteranceMVN: subtract mean per feature dim ──
    float* d_mvn = scratch_b_;  // [T, D]
    utterance_mvn_kernel<<<div_ceil(D, BLOCK), BLOCK, 0, stream_>>>(
        d_feat, d_mvn, T, D);

    // ── 4. ECAPA-TDNN ──
    // Need channel-first layout: [1024, T] from [T, 1024]
    // d_mvn is [T, D] row-major. Transpose to [D, T] = [1024, T]
    float* d_ecapa = scratch_a_;  // [1024, T] channel-first
    transpose_2d_kernel<<<div_ceil(T * D, BLOCK), BLOCK, 0, stream_>>>(
        d_mvn, d_ecapa, T, D);

    // Initial conv + ReLU + BN: [1024, T] → [1024, T]
    float* d_xe = scratch_c_;  // [1024, T]
    forward_conv1d(d_ecapa, d_xe, 1024, 1024, T, 5, 2, 1,
                   w("encoder.conv.weight").ptr, w("encoder.conv.bias").ptr,
                   w("encoder.conv.weight").fp16);
    relu_kernel<<<div_ceil(1024 * T, BLOCK), BLOCK, 0, stream_>>>(d_xe, 1024 * T);
    forward_batch_norm_1d(d_xe, d_xe, 1024, T, "encoder.bn");

    // Three ECAPA blocks with different dilations
    // We need to keep x_e, x1, x2, x3 for the skip connections
    // x_e is in scratch_c_ [1024, T]
    // For each block, the input must be in a buffer that won't be overwritten
    // Block input: scratch_a_ = working buffer, scratch_b_/scratch_c_ = intermediates

    // Save x_e to a safe location
    float* d_xe_saved = d_hidden_states_;  // Reuse hidden states buffer temporarily (25 * T * D is big enough)
    cudaMemcpyAsync(d_xe_saved, d_xe, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // layer1: input = x_e, output = x1
    cudaMemcpyAsync(scratch_a_, d_xe, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    forward_ecapa_block(scratch_a_, 1024, T, 2, "encoder.layer1");
    // scratch_a_ now has x1; save it
    float* d_x1 = d_xe_saved + 1024 * T;
    cudaMemcpyAsync(d_x1, scratch_a_, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // layer2: input = x_e + x1
    // scratch_a_ = x_e + x1
    cudaMemcpyAsync(scratch_a_, d_xe_saved, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    vector_add_kernel<<<div_ceil(1024 * T, BLOCK), BLOCK, 0, stream_>>>(
        scratch_a_, d_x1, 1024 * T);
    forward_ecapa_block(scratch_a_, 1024, T, 3, "encoder.layer2");
    float* d_x2 = d_x1 + 1024 * T;
    cudaMemcpyAsync(d_x2, scratch_a_, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // layer3: input = x_e + x1 + x2
    cudaMemcpyAsync(scratch_a_, d_xe_saved, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    vector_add_kernel<<<div_ceil(1024 * T, BLOCK), BLOCK, 0, stream_>>>(
        scratch_a_, d_x1, 1024 * T);
    vector_add_kernel<<<div_ceil(1024 * T, BLOCK), BLOCK, 0, stream_>>>(
        scratch_a_, d_x2, 1024 * T);
    forward_ecapa_block(scratch_a_, 1024, T, 4, "encoder.layer3");
    float* d_x3 = d_x2 + 1024 * T;
    cudaMemcpyAsync(d_x3, scratch_a_, 1024 * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // Concatenate x1, x2, x3 → [3072, T]
    // They're already contiguous in memory: d_x1, d_x2, d_x3
    float* d_cat = d_x1;  // [3072, T] starting at d_x1

    // layer4: Conv1d(3072→1536, K=1) + ReLU
    float* d_x4 = scratch_a_;  // [1536, T]
    forward_conv1d(d_cat, d_x4, 3072, 1536, T, 1, 0, 1,
                   w("encoder.layer4.weight").ptr, w("encoder.layer4.bias").ptr,
                   w("encoder.layer4.weight").fp16);
    relu_kernel<<<div_ceil(1536 * T, BLOCK), BLOCK, 0, stream_>>>(d_x4, 1536 * T);

    // ── 5. Channel Attention Stat Pooling ──
    // global_x = cat(x4, mean(x4).expand, std(x4).expand) → [4608, T]
    int C_pool = 1536;
    float* d_global = scratch_b_;  // [4608, T]
    pool_global_stats_kernel<<<C_pool, std::min(T, 256), 0, stream_>>>(
        d_x4, d_global, C_pool, T);

    // Attention: Conv1d(4608→128, K=1) → ReLU → BN → Conv1d(128→1536, K=1) → Softmax
    float* d_attn_w = scratch_c_;  // [128, T]
    forward_conv1d(d_global, d_attn_w, 4608, 128, T, 1, 0, 1,
                   w("pooling.attention.0.weight").ptr, w("pooling.attention.0.bias").ptr,
                   w("pooling.attention.0.weight").fp16);
    relu_kernel<<<div_ceil(128 * T, BLOCK), BLOCK, 0, stream_>>>(d_attn_w, 128 * T);
    forward_batch_norm_1d(d_attn_w, d_attn_w, 128, T, "pooling.attention.2");

    float* d_attn_out = d_attn_w + 128 * T;  // [1536, T]
    forward_conv1d(d_attn_w, d_attn_out, 128, 1536, T, 1, 0, 1,
                   w("pooling.attention.3.weight").ptr, w("pooling.attention.3.bias").ptr,
                   w("pooling.attention.3.weight").fp16);

    // Softmax along T dimension (dim=2 in [1, 1536, T])
    // Each channel's T values get softmaxed independently
    int sm_threads_pool = ((std::min(T, 256) + 31) / 32) * 32;
    softmax_kernel<<<C_pool, sm_threads_pool, 0, stream_>>>(
        d_attn_out, C_pool, T);

    // Weighted stats: mu, sg
    float* d_mu = scratch_c_;         // [1536]
    float* d_sg = d_mu + C_pool;      // [1536]
    weighted_stats_kernel<<<div_ceil(C_pool, BLOCK), BLOCK, 0, stream_>>>(
        d_x4, d_attn_out, d_mu, d_sg, C_pool, T);

    // cat(mu, sg) → [3072]
    float* d_pool_out = d_mu;  // mu and sg are contiguous → [3072]

    // ── 6. Projector: BN(3072) → Linear(3072→192) ──
    // BN on [3072, 1] (treating as [C=3072, T=1])
    float* d_bn = scratch_b_;  // [3072]
    batch_norm_1d_kernel<<<div_ceil(3072, BLOCK), BLOCK, 0, stream_>>>(
        d_pool_out, d_bn,
        w("projector.bn.weight").ptr, w("projector.bn.bias").ptr,
        w("projector.bn.running_mean").ptr, w("projector.bn.running_var").ptr,
        3072, 1, 1e-5f);

    // Linear(3072→192): d_bn [1, 3072] → d_fc [1, 192]
    float* d_fc = d_bn + 3072;  // [192]
    forward_linear(d_bn, d_fc, 1, 3072, 192,
                   w("projector.fc.weight").ptr, w("projector.fc.bias").ptr,
                   w("projector.fc.weight").fp16);

    // L2 normalize
    l2_normalize_kernel<<<1, 192, 0, stream_>>>(d_fc, 192);

    cudaEventRecord(ev_done, stream_);
    cudaStreamSynchronize(stream_);

    // Compute latency breakdown.
    cudaEventElapsedTime(&last_lat_cnn_ms_,     ev_start, ev_cnn);
    cudaEventElapsedTime(&last_lat_encoder_ms_, ev_cnn,   ev_enc);
    cudaEventElapsedTime(&last_lat_ecapa_ms_,   ev_enc,   ev_done);
    cudaEventElapsedTime(&last_lat_total_ms_,   ev_start, ev_done);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_cnn);
    cudaEventDestroy(ev_enc);
    cudaEventDestroy(ev_done);

    // Copy to host
    std::vector<float> result(192);
    cudaMemcpy(result.data(), d_fc, 192 * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}


} // namespace deusridet
