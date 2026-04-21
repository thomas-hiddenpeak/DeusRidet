/**
 * @file src/orator/wavlm_ecapa_ecapa.cu
 * @philosophical_role
 *   Peer TU of wavlm_ecapa_encoder.cu under R1 800-line hard cap — ECAPA-TDNN host methods (conv1d / SE-block / ECAPA block) + featurizer headers.
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


// ============================================================================
// Featurizer: softmax weighted sum of hidden states
// ============================================================================

// Softmax over a small 1D vector (25 elements) — single thread block

// Weighted sum: output[t*D+d] = sum_l norm_weights[l] * hidden_states[l*T*D + t*D + d]

// ============================================================================
// UtteranceMVN: mean subtraction over time
// ============================================================================

// Compute mean per feature dim across T frames, then subtract
// Input/output: [T, D] row-major

// ============================================================================
// ECAPA-TDNN kernels
// ============================================================================

// BatchNorm1d eval mode: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
// Input: [C, T] (channel-first, 1D) — one element per thread

// ReLU in-place

// Conv1d: [C_in, T] → [C_out, T_out], with kernel K, stride 1, padding P, dilation D
// Using im2col + cuBLAS GEMM
// For K=1: direct GEMM (y = W @ x), weight shape [C_out, C_in, 1] → [C_out, C_in]
// For K>1: im2col then GEMM

// im2col for 1D conv: extract [C_in * K, T_out] column matrix from [C_in, T] input

// Sigmoid kernel

// Element-wise multiply: y[i] *= x[i] (for SE scaling with broadcast)
// x is [C, 1], y is [C, T] — broadcast multiply

// Adaptive avg pool over last dimension: [C, T] → [C, 1]

// ============================================================================
// Pooling kernels
// ============================================================================

// Compute global_x = cat(x, mean(x).expand, std(x).expand) along channel dim
// x: [C, T], output: [3*C, T]

// Weighted stats: mu[c] = sum_t(x[c,t] * w[c,t]), sg[c] = sqrt(sum_t(x^2*w) - mu^2)

// L2 normalize: x[i] /= sqrt(sum(x[i]^2))

// ============================================================================
// ECAPA-TDNN forward helpers
// ============================================================================

// Conv1d forward: input [C_in, T] → output [C_out, T_out]
// Weight: [C_out, C_in, K], Bias: [C_out]
void WavLMEcapaEncoder::forward_conv1d(const float* d_in, float* d_out,
                                         int C_in, int C_out, int T, int K,
                                         int pad, int dilation,
                                         const float* d_weight, const float* d_bias,
                                         const __half* d_weight_fp16) {
    int T_out = T;
    float alpha = 1.0f, beta = 0.0f;

    if (K == 1) {
        if (d_weight_fp16) {
            int in_count = C_in * T;
            int conv_blocks = (in_count / 2 + 255) / 256;
            f32_to_f16_wlecapa<<<conv_blocks, 256, 0, stream_>>>(d_in, d_gemm_a_, in_count);

            cublasGemmEx(cublas_,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         T, C_out, C_in,
                         &alpha,
                         d_gemm_a_, CUDA_R_16F, T,
                         d_weight_fp16, CUDA_R_16F, C_in,
                         &beta,
                         d_out, CUDA_R_32F, T,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                         T, C_out, C_in,
                         &alpha, d_in, T, d_weight, C_in,
                         &beta, d_out, T);
        }
    } else {
        int col_size = C_in * K * T_out;
        float* d_cols = d_im2col_;

        im2col_1d_kernel<<<div_ceil(col_size, BLOCK), BLOCK, 0, stream_>>>(
            d_in, d_cols, C_in, T, K, pad, dilation, T_out);

        if (d_weight_fp16) {
            int conv_blocks = (col_size / 2 + 255) / 256;
            f32_to_f16_wlecapa<<<conv_blocks, 256, 0, stream_>>>(d_cols, d_gemm_a_, col_size);

            cublasGemmEx(cublas_,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         T_out, C_out, C_in * K,
                         &alpha,
                         d_gemm_a_, CUDA_R_16F, T_out,
                         d_weight_fp16, CUDA_R_16F, C_in * K,
                         &beta,
                         d_out, CUDA_R_32F, T_out,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                         T_out, C_out, C_in * K,
                         &alpha, d_cols, T_out, d_weight, C_in * K,
                         &beta, d_out, T_out);
        }
    }

    if (d_bias) {
        bias_add_channel_kernel<<<div_ceil(C_out * T_out, BLOCK), BLOCK, 0, stream_>>>(
            d_out, d_bias, C_out, T_out);
    }
}

// cuDNN-accelerated Conv1d (as Conv2d with H=1).
// Handles groups, stride, padding, dilation. Used for CNN extractor + pos conv.
void WavLMEcapaEncoder::forward_conv1d_cudnn(
        const float* d_in, float* d_out,
        int C_in, int C_out, int T, int K,
        int stride, int pad, int groups, int dilation,
        const float* d_weight, const float* d_bias) {

    int T_out = (T + 2 * pad - dilation * (K - 1) - 1) / stride + 1;

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, C_in, 1, T);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, C_out, 1, T_out);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               C_out, C_in / groups, 1, K);
    cudnnSetConvolution2dDescriptor(conv_desc,
        0, pad,       // padH, padW
        1, stride,    // strideH, strideW
        1, dilation,  // dilationH, dilationW
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(conv_desc, groups);
    cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);

    // Find best algorithm
    int returned = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn_, in_desc, filt_desc,
        conv_desc, out_desc, 1, &returned, &algo_perf);
    auto algo = algo_perf.algo;

    // Ensure workspace
    size_t ws_need = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_, in_desc, filt_desc,
        conv_desc, out_desc, algo, &ws_need);
    if (ws_need > cudnn_ws_size_) {
        if (d_cudnn_ws_) cudaFree(d_cudnn_ws_);
        cudaMalloc(&d_cudnn_ws_, ws_need);
        cudnn_ws_size_ = ws_need;
    }

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn_, &alpha,
        in_desc, d_in, filt_desc, d_weight,
        conv_desc, algo, d_cudnn_ws_, ws_need,
        &beta, out_desc, d_out);

    if (d_bias) {
        cudnnTensorDescriptor_t bias_desc;
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, C_out, 1, 1);
        float ab = 1.0f, bb = 1.0f;
        cudnnAddTensor(cudnn_, &ab, bias_desc, d_bias, &bb, out_desc, d_out);
        cudnnDestroyTensorDescriptor(bias_desc);
    }

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}

// BatchNorm1d eval-mode forward
void WavLMEcapaEncoder::forward_batch_norm_1d(const float* d_in, float* d_out,
                                                int C, int T, const std::string& prefix) {
    auto& bn_w = w(prefix + ".weight");
    auto& bn_b = w(prefix + ".bias");
    auto& bn_rm = w(prefix + ".running_mean");
    auto& bn_rv = w(prefix + ".running_var");
    batch_norm_1d_kernel<<<div_ceil(C * T, BLOCK), BLOCK, 0, stream_>>>(
        d_in, d_out, bn_w.ptr, bn_b.ptr, bn_rm.ptr, bn_rv.ptr, C, T, 1e-5f);
}

// ECAPA SE block: AdaptiveAvgPool → Conv1d(C→bottleneck, K=1) → ReLU → BN → Conv1d(bottleneck→C, K=1) → Sigmoid
// Modifies d_x in-place: d_x *= sigmoid(fc2(bn(relu(fc1(avg_pool(d_x))))))
void WavLMEcapaEncoder::forward_se_block(float* d_x, int C, int T,
                                           const std::string& prefix) {
    int bottleneck = 128;
    // Reuse scratch_b_ tail for SE intermediates
    // Need [C] for pool, [bottleneck] for fc1, [bottleneck] for bn, [C] for fc2
    float* d_pool = scratch_b_ + scratch_max_T_ * WavLMConfig::embed_dim;  // [C]
    float* d_se1 = d_pool + C;          // [bottleneck]
    float* d_se2 = d_se1 + bottleneck;  // [C]

    // AdaptiveAvgPool1d: [C, T] → [C, 1]
    adaptive_avg_pool_1d_kernel<<<div_ceil(C, BLOCK), BLOCK, 0, stream_>>>(
        d_x, d_pool, C, T);

    // Conv1d(C→128, K=1) = linear: d_se1 = W @ d_pool + bias
    // W: [128, C, 1], pool is [C, 1], output [128, 1]
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                 1, bottleneck, C,
                 &alpha,
                 d_pool, 1,
                 w(prefix + ".1.weight").ptr, C,
                 &beta,
                 d_se1, 1);
    // Add bias: [bottleneck, 1] channel-first
    bias_add_channel_kernel<<<div_ceil(bottleneck, BLOCK), BLOCK, 0, stream_>>>(
        d_se1, w(prefix + ".1.bias").ptr, bottleneck, 1);

    // ReLU
    relu_kernel<<<div_ceil(bottleneck, BLOCK), BLOCK, 0, stream_>>>(d_se1, bottleneck);

    // BN(128)
    // For [128, 1] — just apply element-wise
    batch_norm_1d_kernel<<<div_ceil(bottleneck, BLOCK), BLOCK, 0, stream_>>>(
        d_se1, d_se1, w(prefix + ".3.weight").ptr, w(prefix + ".3.bias").ptr,
        w(prefix + ".3.running_mean").ptr, w(prefix + ".3.running_var").ptr,
        bottleneck, 1, 1e-5f);

    // Conv1d(128→C, K=1) = linear
    cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                 1, C, bottleneck,
                 &alpha,
                 d_se1, 1,
                 w(prefix + ".4.weight").ptr, bottleneck,
                 &beta,
                 d_se2, 1);
    bias_add_channel_kernel<<<div_ceil(C, BLOCK), BLOCK, 0, stream_>>>(
        d_se2, w(prefix + ".4.bias").ptr, C, 1);

    // Sigmoid
    sigmoid_kernel<<<div_ceil(C, BLOCK), BLOCK, 0, stream_>>>(d_se2, C);

    // Multiply: d_x[c, t] *= d_se2[c]
    broadcast_mul_kernel<<<div_ceil(C * T, BLOCK), BLOCK, 0, stream_>>>(d_x, d_se2, C, T);
}

// ECAPA Res2Net block
void WavLMEcapaEncoder::forward_ecapa_block(float* d_x, int C, int T,
                                              int dilation, const std::string& prefix) {
    int width = 128;  // C / model_scale = 1024 / 8
    int nums = 7;     // model_scale - 1

    // d_x is [C, T] channel-first
    // Need temp buffers for conv1 output, res2net splits, conv3 output
    // Use scratch_b_ for intermediate (scratch_a_ has the residual input)
    float* d_conv1 = scratch_b_;  // [C, T]

    // conv1: Conv1d(C→C, K=1) + ReLU + BN
    forward_conv1d(d_x, d_conv1, C, C, T, 1, 0, 1,
                   w(prefix + ".conv1.weight").ptr, w(prefix + ".conv1.bias").ptr,
                   w(prefix + ".conv1.weight").fp16);
    relu_kernel<<<div_ceil(C * T, BLOCK), BLOCK, 0, stream_>>>(d_conv1, C * T);
    forward_batch_norm_1d(d_conv1, d_conv1, C, T, prefix + ".bn1");

    // Res2Net: split d_conv1 [C=1024, T] into 8 groups of [128, T]
    // Process groups: sp starts as group[0], then for i=1..6: sp = conv(sp + group[i])
    // Accumulate outputs: out = cat(processed_groups)
    float* d_out = scratch_c_;  // [C, T] for the concatenated output
    // d_sp buffer for the cascading intermediate
    float* d_sp = d_out + C * T;  // [width, T]

    for (int i = 0; i < nums; i++) {
        float* group_i = d_conv1 + i * width * T;  // pointer into d_conv1
        if (i == 0) {
            // sp = spx[0] — just copy
            cudaMemcpyAsync(d_sp, group_i, width * T * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream_);
        } else {
            // sp = sp + spx[i]
            vector_add_kernel<<<div_ceil(width * T, BLOCK), BLOCK, 0, stream_>>>(
                d_sp, group_i, width * T);
        }

        // Dilated conv: Conv1d(128→128, K=3, pad=dilation, dil=dilation) + ReLU + BN
        float* d_conv_out = d_out + i * width * T;  // write directly to output slot
        forward_conv1d(d_sp, d_conv_out, width, width, T, 3, dilation, dilation,
                       w(prefix + ".convs." + std::to_string(i) + ".weight").ptr,
                       w(prefix + ".convs." + std::to_string(i) + ".bias").ptr,
                       w(prefix + ".convs." + std::to_string(i) + ".weight").fp16);
        relu_kernel<<<div_ceil(width * T, BLOCK), BLOCK, 0, stream_>>>(d_conv_out, width * T);
        forward_batch_norm_1d(d_conv_out, d_conv_out, width, T,
                              prefix + ".bns." + std::to_string(i));

        // Update sp for next iteration (conv output is the new sp)
        cudaMemcpyAsync(d_sp, d_conv_out, width * T * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }

    // Append last group (spx[nums] = group[7]) — passes through unchanged
    float* group_last = d_conv1 + nums * width * T;
    cudaMemcpyAsync(d_out + nums * width * T, group_last, width * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // conv3: Conv1d(C→C, K=1) + ReLU + BN
    float* d_conv3 = scratch_b_;  // reuse
    forward_conv1d(d_out, d_conv3, C, C, T, 1, 0, 1,
                   w(prefix + ".conv3.weight").ptr, w(prefix + ".conv3.bias").ptr,
                   w(prefix + ".conv3.weight").fp16);
    relu_kernel<<<div_ceil(C * T, BLOCK), BLOCK, 0, stream_>>>(d_conv3, C * T);
    forward_batch_norm_1d(d_conv3, d_conv3, C, T, prefix + ".bn3");

    // SE block
    forward_se_block(d_conv3, C, T, prefix + ".se.se");

    // Residual add: d_conv3 += d_x (original input)
    vector_add_kernel<<<div_ceil(C * T, BLOCK), BLOCK, 0, stream_>>>(d_conv3, d_x, C * T);

    // Copy result back to d_x
    cudaMemcpyAsync(d_x, d_conv3, C * T * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
}

// ============================================================================
// Full extract pipeline
// ============================================================================


} // namespace deusridet
