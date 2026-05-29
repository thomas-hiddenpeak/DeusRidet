/**
 * @file diarizen_resnet34_forward.cu
 * @philosophical_role P2a forward pass: fbank -> ResNet34 stem + 4 stages of
 *   BasicBlocks -> TSTP weighted statistics pooling -> 256-d projection. Convs
 *   are cuDNN, the projection is cuBLAS, norms/pool are the header kernels.
 * @serves DiarizenResnet34Embedder.
 */
#include "diarizen_resnet34_embedder.h"

#include <vector>

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>

#include "../communis/log.h"
#include "diarizen_resnet34_kernels.cuh"

namespace deusridet {
namespace orator {

namespace {
constexpr const char* kFLog = "DiariZenResNet34";
constexpr int kBlk = 256;
inline int grd(int n) { return (n + kBlk - 1) / kBlk; }
}  // namespace

// cuDNN conv2d, NCHW fp32, cross-correlation, no bias. N=1.
void DiarizenResnet34Embedder::conv2d_(const float* d_in, float* d_out, int C_in,
                                       int C_out, int H, int W, int K, int stride,
                                       int pad, const float* d_w, int& H_out,
                                       int& W_out) {
    H_out = (H + 2 * pad - K) / stride + 1;
    W_out = (W + 2 * pad - K) / stride + 1;

    cudnnTensorDescriptor_t in_d, out_d;
    cudnnFilterDescriptor_t filt_d;
    cudnnConvolutionDescriptor_t conv_d;
    cudnnCreateTensorDescriptor(&in_d);
    cudnnCreateTensorDescriptor(&out_d);
    cudnnCreateFilterDescriptor(&filt_d);
    cudnnCreateConvolutionDescriptor(&conv_d);

    cudnnSetTensor4dDescriptor(in_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_in,
                               H, W);
    cudnnSetTensor4dDescriptor(out_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                               C_out, H_out, W_out);
    cudnnSetFilter4dDescriptor(filt_d, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out,
                               C_in, K, K);
    cudnnSetConvolution2dDescriptor(conv_d, pad, pad, stride, stride, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    int ret = 0;
    cudnnConvolutionFwdAlgoPerf_t perf;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn_, in_d, filt_d, conv_d, out_d, 1,
                                           &ret, &perf);
    cudnnConvolutionFwdAlgo_t algo = perf.algo;
    size_t need = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_, in_d, filt_d, conv_d, out_d,
                                            algo, &need);
    if (need > ws_bytes_) {
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_, in_d, filt_d, conv_d,
                                                out_d, algo, &need);
    }
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn_, &alpha, in_d, d_in, filt_d, d_w, conv_d, algo,
                            d_ws_, ws_bytes_, &beta, out_d, d_out);

    cudnnDestroyTensorDescriptor(in_d);
    cudnnDestroyTensorDescriptor(out_d);
    cudnnDestroyFilterDescriptor(filt_d);
    cudnnDestroyConvolutionDescriptor(conv_d);
}

// out = relu(bn2(conv2(relu(bn1(conv1(in))))) + shortcut(in)).
// Scratch: d_bufC_ (conv1 out), d_bufD_ (shortcut out).
void DiarizenResnet34Embedder::run_block_(const Resnet34Block& blk, float* d_in,
                                          int H, int W, float* d_out, int& H_out,
                                          int& W_out, float* /*unused*/) {
    int h1, w1;
    conv2d_(d_in, d_bufC_, blk.in_planes, blk.planes, H, W, 3, blk.stride, 1,
            blk.conv1_w, h1, w1);
    r34_bn_relu<<<grd(blk.planes * h1 * w1), kBlk>>>(d_bufC_, blk.bn1_scale,
                                                     blk.bn1_bias, blk.planes,
                                                     h1 * w1, 1);
    int h2, w2;
    conv2d_(d_bufC_, d_out, blk.planes, blk.planes, h1, w1, 3, 1, 1, blk.conv2_w,
            h2, w2);
    r34_bn_relu<<<grd(blk.planes * h2 * w2), kBlk>>>(d_out, blk.bn2_scale,
                                                     blk.bn2_bias, blk.planes,
                                                     h2 * w2, 0);
    const float* d_sc;
    if (blk.has_shortcut) {
        int hs, ws;
        conv2d_(d_in, d_bufD_, blk.in_planes, blk.planes, H, W, 1, blk.stride, 0,
                blk.sc_w, hs, ws);
        r34_bn_relu<<<grd(blk.planes * hs * ws), kBlk>>>(d_bufD_, blk.sc_scale,
                                                         blk.sc_bias, blk.planes,
                                                         hs * ws, 0);
        d_sc = d_bufD_;
    } else {
        d_sc = d_in;  // identity (in_planes == planes, stride 1)
    }
    int n = blk.planes * h2 * w2;
    r34_add_relu<<<grd(n), kBlk>>>(d_out, d_sc, n);
    H_out = h2;
    W_out = w2;
}

bool DiarizenResnet34Embedder::debug_fbank(const float* wave, int n_samples,
                                           std::vector<float>& out,
                                           int& out_frames) {
    if (!loaded_) return false;
    int T = compute_fbank_(wave, n_samples, d_fbank_);
    if (T <= 0) return false;
    out.resize((size_t)T * DiarizenResnet34Arch::kNumMel);
    cudaMemcpy(out.data(), d_fbank_, out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    out_frames = T;
    return true;
}

bool DiarizenResnet34Embedder::embed(const float* wave, int n_samples,
                                     const float* mask, int num_frames,
                                     float* out_embed) {
    using A = DiarizenResnet34Arch;
    if (!loaded_) return false;

    int T = compute_fbank_(wave, n_samples, d_fbank_);
    if (T <= 0) return false;

    // Lay fbank [T,80] out as conv input [1,1,80,T] (freq=H, time=W).
    r34_transpose_TM_to_MT<<<grd(T * A::kNumMel), kBlk>>>(d_fbank_, d_bufA_, T,
                                                          A::kNumMel);

    // Stem conv1 (3x3, s1, p1) -> [32, 80, T], bn1 + relu.
    int H, W;
    conv2d_(d_bufA_, d_bufB_, 1, A::kBaseWidth, A::kNumMel, T, 3, 1, 1, conv1_w_,
            H, W);
    r34_bn_relu<<<grd(A::kBaseWidth * H * W), kBlk>>>(d_bufB_, bn1_scale_,
                                                      bn1_bias_, A::kBaseWidth,
                                                      H * W, 1);
    float* cur = d_bufB_;
    float* alt = d_bufA_;
    for (const auto& blk : blocks_) {
        int H2, W2;
        run_block_(blk, cur, H, W, alt, H2, W2, nullptr);
        std::swap(cur, alt);
        H = H2;
        W = W2;
    }
    // cur is layer4 output [256, 10, Tp] == [2560, Tp].
    const int rows = A::kStatsDim;  // 2560
    const int Tp = W;               // pooling time axis

    // Interpolate the speaker mask (nearest) to Tp; store weights in d_bufD_.
    float* d_w = d_bufD_;
    if (mask) {
        // Upload mask to the head of d_bufC_ scratch, then resample.
        cudaMemcpy(d_bufC_, mask, num_frames * sizeof(float),
                   cudaMemcpyHostToDevice);
        r34_interp_nearest<<<grd(Tp), kBlk>>>(d_bufC_, num_frames, d_w, Tp);
    } else {
        std::vector<float> ones(Tp, 1.0f);
        cudaMemcpy(d_w, ones.data(), Tp * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Weighted statistics pooling -> d_pool_ [5120] (mean|std).
    r34_stats_pool<<<grd(rows), kBlk>>>(cur, d_w, d_pool_, rows, Tp);

    // seg_1 projection: out[256] = W[256,5120] @ pool[5120] + bias.
    float* d_out = d_bufA_;  // reuse
    cudaMemcpy(d_out, seg1_b_, A::kEmbedDim * sizeof(float),
               cudaMemcpyDeviceToDevice);
    float one = 1.0f;
    cublasSgemv(blas_, CUBLAS_OP_T, A::kPoolDim, A::kEmbedDim, &one, seg1_w_,
                A::kPoolDim, d_pool_, 1, &one, d_out, 1);
    cudaMemcpy(out_embed, d_out, A::kEmbedDim * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return true;
}

}  // namespace orator
}  // namespace deusridet
