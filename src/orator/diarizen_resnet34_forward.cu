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
// Descriptors + chosen algo are cached per (C_in,C_out,H,W,K,stride,pad): all
// ResNet34 conv shapes are fixed once the chunk length is fixed, so every
// (chunk, speaker) embed call replays the same ~36 configs. The cache removes
// the per-conv descriptor create/destroy and algo heuristic; the kernel
// selected is identical, so the convolution output is bit-for-bit unchanged.
void DiarizenResnet34Embedder::conv2d_(const float* d_in, float* d_out, int C_in,
                                       int C_out, int H, int W, int K, int stride,
                                       int pad, const float* d_w, int& H_out,
                                       int& W_out) {
    const std::uint64_t key =
        (std::uint64_t)(C_in & 0x1FF) | ((std::uint64_t)(C_out & 0x1FF) << 9) |
        ((std::uint64_t)(H & 0x7FF) << 18) | ((std::uint64_t)(W & 0x7FF) << 29) |
        ((std::uint64_t)(K & 0x3) << 40) |
        ((std::uint64_t)(stride & 0x3) << 42) |
        ((std::uint64_t)(pad & 0x3) << 44);

    auto it = conv_cache_.find(key);
    if (it == conv_cache_.end()) {
        ConvPlan p;
        p.H_out = (H + 2 * pad - K) / stride + 1;
        p.W_out = (W + 2 * pad - K) / stride + 1;
        cudnnCreateTensorDescriptor(&p.in_d);
        cudnnCreateTensorDescriptor(&p.out_d);
        cudnnCreateFilterDescriptor(&p.filt_d);
        cudnnCreateConvolutionDescriptor(&p.conv_d);
        cudnnSetTensor4dDescriptor(p.in_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                                   C_in, H, W);
        cudnnSetTensor4dDescriptor(p.out_d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, C_out, p.H_out, p.W_out);
        cudnnSetFilter4dDescriptor(p.filt_d, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                   C_out, C_in, K, K);
        cudnnSetConvolution2dDescriptor(p.conv_d, pad, pad, stride, stride, 1, 1,
                                        CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT);

        int ret = 0;
        cudnnConvolutionFwdAlgoPerf_t perf;
        cudnnGetConvolutionForwardAlgorithm_v7(cudnn_, p.in_d, p.filt_d, p.conv_d,
                                               p.out_d, 1, &ret, &perf);
        p.algo = perf.algo;
        size_t need = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_, p.in_d, p.filt_d,
                                                p.conv_d, p.out_d, p.algo, &need);
        if (need > ws_bytes_) {
            p.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        }
        it = conv_cache_.emplace(key, p).first;
    }
    const ConvPlan& p = it->second;
    H_out = p.H_out;
    W_out = p.W_out;

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn_, &alpha, p.in_d, d_in, p.filt_d, d_w, p.conv_d,
                            p.algo, d_ws_, ws_bytes_, &beta, p.out_d, d_out);
}

// out = relu(bn2(conv2(relu(bn1(conv1(in))))) + shortcut(in)).
// Scratch: d_bufC_ (conv1 out), d_bufD_ (shortcut out).
void DiarizenResnet34Embedder::run_block_(const Resnet34Block& blk, float* d_in,
                                          int H, int W, float* d_out, int& H_out,
                                          int& W_out, float* /*unused*/) {
    int h1, w1;
    conv2d_(d_in, d_bufC_, blk.in_planes, blk.planes, H, W, 3, blk.stride, 1,
            blk.conv1_w, h1, w1);
    r34_bn_relu<<<grd(blk.planes * h1 * w1), kBlk, 0, stream_>>>(
        d_bufC_, blk.bn1_scale, blk.bn1_bias, blk.planes, h1 * w1, 1);
    int h2, w2;
    conv2d_(d_bufC_, d_out, blk.planes, blk.planes, h1, w1, 3, 1, 1, blk.conv2_w,
            h2, w2);
    r34_bn_relu<<<grd(blk.planes * h2 * w2), kBlk, 0, stream_>>>(
        d_out, blk.bn2_scale, blk.bn2_bias, blk.planes, h2 * w2, 0);
    const float* d_sc;
    if (blk.has_shortcut) {
        int hs, ws;
        conv2d_(d_in, d_bufD_, blk.in_planes, blk.planes, H, W, 1, blk.stride, 0,
                blk.sc_w, hs, ws);
        r34_bn_relu<<<grd(blk.planes * hs * ws), kBlk, 0, stream_>>>(
            d_bufD_, blk.sc_scale, blk.sc_bias, blk.planes, hs * ws, 0);
        d_sc = d_bufD_;
    } else {
        d_sc = d_in;  // identity (in_planes == planes, stride 1)
    }
    int n = blk.planes * h2 * w2;
    r34_add_relu<<<grd(n), kBlk, 0, stream_>>>(d_out, d_sc, n);
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
    cudaMemcpyAsync(out.data(), d_fbank_, out.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    out_frames = T;
    return true;
}

bool DiarizenResnet34Embedder::embed(const float* wave, int n_samples,
                                     const float* mask, int num_frames,
                                     float* out_embed) {
    if (!embed_backbone(wave, n_samples)) return false;
    return embed_pool(mask, num_frames, out_embed);
}

// Backbone: fbank -> stem conv -> ResNet34 blocks. Depends only on the
// waveform; result [kStatsDim, Tp] is stored in d_backbone_ for reuse across
// every speaker that shares this chunk.
bool DiarizenResnet34Embedder::embed_backbone(const float* wave, int n_samples) {
    using A = DiarizenResnet34Arch;
    if (!loaded_) return false;

    int T = compute_fbank_(wave, n_samples, d_fbank_);
    if (T <= 0) return false;

    // Lay fbank [T,80] out as conv input [1,1,80,T] (freq=H, time=W).
    r34_transpose_TM_to_MT<<<grd(T * A::kNumMel), kBlk, 0, stream_>>>(
        d_fbank_, d_bufA_, T, A::kNumMel);

    // Stem conv1 (3x3, s1, p1) -> [32, 80, T], bn1 + relu.
    int H, W;
    conv2d_(d_bufA_, d_bufB_, 1, A::kBaseWidth, A::kNumMel, T, 3, 1, 1, conv1_w_,
            H, W);
    r34_bn_relu<<<grd(A::kBaseWidth * H * W), kBlk, 0, stream_>>>(
        d_bufB_, bn1_scale_, bn1_bias_, A::kBaseWidth, H * W, 1);
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
    backbone_Tp_ = Tp;
    cudaMemcpyAsync(d_backbone_, cur, (size_t)rows * Tp * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    return true;
}

// Pooling head: weighted statistics pooling of the stored backbone under the
// speaker mask, then the seg_1 projection -> 256-d embedding.
bool DiarizenResnet34Embedder::embed_pool(const float* mask, int num_frames,
                                          float* out_embed) {
    using A = DiarizenResnet34Arch;
    if (!loaded_ || backbone_Tp_ <= 0) return false;
    const int rows = A::kStatsDim;  // 2560
    const int Tp = backbone_Tp_;
    float* cur = d_backbone_;

    // Interpolate the speaker mask (nearest) to Tp; store weights in d_bufD_.
    float* d_w = d_bufD_;
    if (mask) {
        // Upload mask to the head of d_bufC_ scratch, then resample.
        cudaMemcpyAsync(d_bufC_, mask, num_frames * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        r34_interp_nearest<<<grd(Tp), kBlk, 0, stream_>>>(d_bufC_, num_frames,
                                                          d_w, Tp);
    } else {
        std::vector<float> ones(Tp, 1.0f);
        cudaMemcpyAsync(d_w, ones.data(), Tp * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        // ones[] is a function-local staging buffer; the async H2D above must
        // finish before it is reclaimed.
        cudaStreamSynchronize(stream_);
    }

    // Weighted statistics pooling -> d_pool_ [5120] (mean|std).
    r34_stats_pool<<<grd(rows), kBlk, 0, stream_>>>(cur, d_w, d_pool_, rows, Tp);

    // seg_1 projection: out[256] = W[256,5120] @ pool[5120] + bias.
    float* d_out = d_bufA_;  // reuse
    cudaMemcpyAsync(d_out, seg1_b_, A::kEmbedDim * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    float one = 1.0f;
    cublasSgemv(blas_, CUBLAS_OP_T, A::kPoolDim, A::kEmbedDim, &one, seg1_w_,
                A::kPoolDim, d_pool_, 1, &one, d_out, 1);
    cudaMemcpyAsync(out_embed, d_out, A::kEmbedDim * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return true;
}

}  // namespace orator
}  // namespace deusridet
