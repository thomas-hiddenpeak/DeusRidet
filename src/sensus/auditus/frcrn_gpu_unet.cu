/**
 * @file src/sensus/auditus/frcrn_gpu_unet.cu
 * @philosophical_role
 *   FRCRN UNet forward + top-level enhance wrapper — peer TU of frcrn_gpu.cu.
 *   Split under R1 800-line hard cap.
 * @serves
 *   Auditus denoise stage. Complements the per-op implementations in
 *   frcrn_gpu.cu (STFT/iSTFT/conv2d/complex ops/FSMN).
 */
// frcrn_gpu_unet.cu — contains:
//   - FrcrnGpu::forward_se_layer
//   - FrcrnGpu::forward_unet
//   - FrcrnGpu::enhance
//   - FrcrnGpu::enhance_host

#include "frcrn_gpu.h"
#include "../../communis/log.h"
#include "../../machina/safetensors.h"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace deusridet {

using namespace frcrn;
using namespace frcrn_kernels;

// TU-local duplicate of the CHECK macros from frcrn_gpu.cu.
#define CUDNN_CHECK(call) do {                                              \
    cudnnStatus_t st = (call);                                              \
    if (st != CUDNN_STATUS_SUCCESS) {                                       \
        LOG_ERROR("FRCRN", "cuDNN error at %s:%d: %s",                     \
                  __FILE__, __LINE__, cudnnGetErrorString(st));             \
        return;                                                             \
    }                                                                       \
} while(0)

#define CUDA_CHECK(call) do {                                               \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        LOG_ERROR("FRCRN", "CUDA error at %s:%d: %s",                      \
                  __FILE__, __LINE__, cudaGetErrorString(err));             \
    }                                                                       \
} while(0)

// ============================================================================
// SE Layer (complex squeeze-excite)
// ============================================================================

void FrcrnGpu::forward_se_layer(
    float* re, float* im, int C, int H, int W,
    const std::string& prefix)
{
    // SELayer:
    //   x_r = avg_pool(x[...,0]).view(b,c)  → [C]
    //   x_i = avg_pool(x[...,1]).view(b,c)  → [C]
    //   y_r = fc_r(x_r) - fc_i(x_i)         → [C]
    //   y_i = fc_r(x_i) + fc_i(x_r)         → [C]
    //   x = x * cat(y_r, y_i)               → channel-wise scale

    int HW = H * W;
    int bottleneck = kSeBotNeck;  // 16

    // Reuse scratch buffers carefully (we know FSMN isn't running here)
    float* pool_re = d_scratch_c_;                        // [C]
    float* pool_im = pool_re + C;                         // [C]
    float* fc_out1 = pool_im + C;                         // [bottleneck]
    float* fc_out2 = fc_out1 + bottleneck;                // [bottleneck]
    float* fc_out3 = fc_out2 + bottleneck;                // [C]
    float* fc_out4 = fc_out3 + C;                         // [C]
    float* scale_re = fc_out4 + C;                        // [C]
    float* scale_im = scale_re + C;                       // [C]

    // Global average pool
    launch_se_avg_pool(re, pool_re, C, HW, stream_);
    launch_se_avg_pool(im, pool_im, C, HW, stream_);

    // fc_r: Linear(C→16, ReLU) → Linear(16→C, Sigmoid)
    // fc_i: same architecture
    auto fc_r_w1 = w(prefix + ".fc_r.0.weight");  // [16, C]
    auto fc_r_b1 = w(prefix + ".fc_r.0.bias");    // [16]
    auto fc_r_w2 = w(prefix + ".fc_r.2.weight");  // [C, 16]
    auto fc_r_b2 = w(prefix + ".fc_r.2.bias");    // [C]
    auto fc_i_w1 = w(prefix + ".fc_i.0.weight");  // [16, C]
    auto fc_i_b1 = w(prefix + ".fc_i.0.bias");    // [16]
    auto fc_i_w2 = w(prefix + ".fc_i.2.weight");  // [C, 16]
    auto fc_i_b2 = w(prefix + ".fc_i.2.bias");    // [C]

    float alpha = 1.0f, beta_zero = 0.0f;

    // fc_r(pool_re):
    //   step1: [16, C] × [C, 1] → [16, 1] + bias → ReLU
    cublasSgemv(cublas_, CUBLAS_OP_T, C, bottleneck,
                &alpha, fc_r_w1.ptr, C, pool_re, 1,
                &beta_zero, fc_out1, 1);
    launch_bias_add(fc_out1, fc_r_b1.ptr, bottleneck, 1, stream_);
    launch_relu(fc_out1, bottleneck, stream_);

    //   step2: [C, 16] × [16, 1] → [C, 1] + bias → Sigmoid
    cublasSgemv(cublas_, CUBLAS_OP_T, bottleneck, C,
                &alpha, fc_r_w2.ptr, bottleneck, fc_out1, 1,
                &beta_zero, fc_out3, 1);  // fc_r(pool_re) → fc_out3 [C]
    launch_bias_add(fc_out3, fc_r_b2.ptr, C, 1, stream_);
    launch_sigmoid(fc_out3, C, stream_);

    // fc_i(pool_im):
    cublasSgemv(cublas_, CUBLAS_OP_T, C, bottleneck,
                &alpha, fc_i_w1.ptr, C, pool_im, 1,
                &beta_zero, fc_out2, 1);
    launch_bias_add(fc_out2, fc_i_b1.ptr, bottleneck, 1, stream_);
    launch_relu(fc_out2, bottleneck, stream_);

    cublasSgemv(cublas_, CUBLAS_OP_T, bottleneck, C,
                &alpha, fc_i_w2.ptr, bottleneck, fc_out2, 1,
                &beta_zero, fc_out4, 1);  // fc_i(pool_im) → fc_out4 [C]
    launch_bias_add(fc_out4, fc_i_b2.ptr, C, 1, stream_);
    launch_sigmoid(fc_out4, C, stream_);

    // fc_r(pool_im) → need another call
    float* fc_r_pool_im = scale_re;  // reuse
    cublasSgemv(cublas_, CUBLAS_OP_T, C, bottleneck,
                &alpha, fc_r_w1.ptr, C, pool_im, 1,
                &beta_zero, fc_out1, 1);
    launch_bias_add(fc_out1, fc_r_b1.ptr, bottleneck, 1, stream_);
    launch_relu(fc_out1, bottleneck, stream_);
    cublasSgemv(cublas_, CUBLAS_OP_T, bottleneck, C,
                &alpha, fc_r_w2.ptr, bottleneck, fc_out1, 1,
                &beta_zero, fc_r_pool_im, 1);
    launch_bias_add(fc_r_pool_im, fc_r_b2.ptr, C, 1, stream_);
    launch_sigmoid(fc_r_pool_im, C, stream_);

    // fc_i(pool_re) → need another call
    float* fc_i_pool_re = scale_im;  // reuse
    cublasSgemv(cublas_, CUBLAS_OP_T, C, bottleneck,
                &alpha, fc_i_w1.ptr, C, pool_re, 1,
                &beta_zero, fc_out2, 1);
    launch_bias_add(fc_out2, fc_i_b1.ptr, bottleneck, 1, stream_);
    launch_relu(fc_out2, bottleneck, stream_);
    cublasSgemv(cublas_, CUBLAS_OP_T, bottleneck, C,
                &alpha, fc_i_w2.ptr, bottleneck, fc_out2, 1,
                &beta_zero, fc_i_pool_re, 1);
    launch_bias_add(fc_i_pool_re, fc_i_b2.ptr, C, 1, stream_);
    launch_sigmoid(fc_i_pool_re, C, stream_);

    // Complex SE scale:
    //   y_r = fc_r(x_r) - fc_i(x_i)  =  fc_out3 - fc_out4
    //   y_i = fc_r(x_i) + fc_i(x_r)  =  fc_r_pool_im + fc_i_pool_re
    // Rewrite scale buffers:
    //   scale_re[c] = fc_out3[c] - fc_out4[c]
    //   scale_im[c] = fc_r_pool_im[c] + fc_i_pool_re[c]
    // But fc_r_pool_im = scale_re and fc_i_pool_re = scale_im (aliased!)
    // Need temp copies. Use fc_out1/fc_out2 as temp.
    CUDA_CHECK(cudaMemcpyAsync(fc_out1, fc_r_pool_im, C * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(fc_out2, fc_i_pool_re, C * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));

    // scale_re = fc_out3 - fc_out4 (these are independent, no alias)
    launch_complex_combine(fc_out3, fc_out4, fc_out1, fc_out2,
                           scale_re, scale_im, C, stream_);

    // Apply: x_re *= scale_re, x_im *= scale_im (channel-wise)
    // Actually the SE output multiplies the full complex tensor:
    // out[c,h,w,0] = x[c,h,w,0] * y_r[c] - x[c,h,w,1] * y_i[c]  (wait, no)
    // Looking at the Python: return x * y where y has shape [B,C,1,1,2]
    // This is element-wise * in the last dim. With [..,0] and [..,1]:
    //   x[...,0] * y[...,0] and x[...,1] * y[...,1]
    // Wait no, that's just regular element-wise multiplication, not complex multiplication!
    // The SE applies y as a real-valued channel scale to the complex tensor:
    //   output_re = input_re * y_re
    //   output_im = input_im * y_im
    // This is NOT complex multiplication. It's separate real scaling.

    launch_se_scale(re, scale_re, C, HW, stream_);
    launch_se_scale(im, scale_im, C, HW, stream_);
}

// ============================================================================
// Full UNet forward pass
// ============================================================================

void FrcrnGpu::forward_unet(
    const float* re_in, const float* im_in,
    float* re_out, float* im_out,
    int H, int T,
    const std::string& prefix)
{
    // Track dimensions through encoder
    int cur_C = 1;
    int cur_H = H;   // kFreqBins = 321
    int cur_W = T;

    // Store input as skip connection 0
    int n0 = cur_C * cur_H * cur_W;
    CUDA_CHECK(cudaMemcpyAsync(enc_skip_[0].re, re_in, n0 * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(enc_skip_[0].im, im_in, n0 * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));
    enc_skip_[0].H = cur_H;
    enc_skip_[0].W = cur_W;

    // Current data pointers (start with input)
    // We use enc_skip_[0] as initial working buffer
    float* cur_re = enc_skip_[0].re;
    float* cur_im = enc_skip_[0].im;

    // (Dimensions are tracked via enc_skip_[].H/W)

    // ---- Encoder ----
    for (int i = 0; i < kNumStages; i++) {
        // FSMN_enc (skip for i=0 per Python code: "if i > 0: x = fsmn_enc[i](x)")
        if (i > 0) {
            std::string fsmn_name = prefix + ".fsmn_enc" + std::to_string(i);
            forward_complex_fsmn_l1(cur_re, cur_im, cur_C, cur_H, cur_W, fsmn_name);
        }

        // Encoder: ComplexConv2d + ComplexBN + LeakyReLU
        int Ho, Wo;
        std::string enc_name = prefix + ".encoder" + std::to_string(i);

        // Output goes to next skip buffer
        forward_complex_conv2d(cur_re, cur_im,
                               enc_skip_[i + 1].re, enc_skip_[i + 1].im,
                               kEncInCh[i], kEncOutCh[i], cur_H, cur_W,
                               kEncKernH[i], kEncKernW[i],
                               kStrideH, kStrideW, kPadH, kPadW,
                               enc_name + ".conv", Ho, Wo);

        // BN + LeakyReLU in-place
        forward_complex_bn_relu(enc_skip_[i + 1].re, enc_skip_[i + 1].im,
                                kEncOutCh[i], Ho, Wo,
                                enc_name + ".bn");

        cur_C = kEncOutCh[i];
        cur_H = Ho;
        cur_W = Wo;

        // SE layer: creates the skip connection value (xs_se[i+1])
        // We store SE output separately for skip connections
        enc_skip_[i + 1].H = cur_H;
        enc_skip_[i + 1].W = cur_W;

        // Apply SE and store result as skip value
        // First, copy to SE result buffers, then apply SE in-place
        // The Python code: xs_se.append(self.se_layers_enc[i](x))
        // SE doesn't modify x directly — it returns a new tensor.
        // But since we stored the encoder output in enc_skip_[i+1],
        // we need to apply SE to a copy for the skip, while keeping
        // the unmodified output as the input to the next encoder.
        //
        // Actually re-reading the Python:
        //   xs.append(x)           ← raw input (before encoder) stored
        //   x = encoder(x)         ← x is now encoder output
        //   xs_se.append(se(x))    ← SE applied to encoder output, stored for skip
        //
        // Then encoder output x continues to next iteration unchanged.
        // So SE is only used for skip connections, not on the main path.
        //
        // We store enc_skip_[i+1] as the SE(encoder_output).
        // The main path continues from the raw encoder output.
        // But we already wrote encoder output to enc_skip_[i+1]...
        //
        // Solution: we need separate buffers. The encoder output is the main
        // path, and we create SE(encoder_output) for skip connections.
        // The simplest approach: apply SE in-place to enc_skip_[i+1]
        // (making it the SE output for skip), and copy the raw encoder
        // output to a temporary buffer before SE to continue the main path.

        // Copy raw encoder output to main path buffer before SE modifies skip
        int enc_out_n = cur_C * cur_H * cur_W;
        // Use d_scratch_a_ / d_scratch_b_ as ping-pong main path buffers
        float* next_re = (i % 2 == 0) ? d_scratch_a_ : d_scratch_b_;
        float* next_im = next_re + enc_out_n;

        CUDA_CHECK(cudaMemcpyAsync(next_re, enc_skip_[i + 1].re,
                                   enc_out_n * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(next_im, enc_skip_[i + 1].im,
                                   enc_out_n * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_));

        // Apply SE in-place on enc_skip_[i+1] (this becomes the skip connection value)
        std::string se_name = prefix + ".se_layer_enc" + std::to_string(i);
        forward_se_layer(enc_skip_[i + 1].re, enc_skip_[i + 1].im,
                         cur_C, cur_H, cur_W, se_name);

        cur_re = next_re;
        cur_im = next_im;


    }

    // ---- Bottleneck FSMN ----
    forward_complex_fsmn(cur_re, cur_im, cur_C, cur_H, cur_W,
                         prefix + ".fsmn");


    // ---- Decoder ----
    for (int i = 0; i < kNumStages; i++) {
        // Decoder: ComplexConvTranspose2d + ComplexBN + LeakyReLU
        int Ho, Wo;
        std::string dec_name = prefix + ".decoder" + std::to_string(i);

        // Output to temporary buffer
        float* dec_re = (i % 2 == 0) ? d_scratch_b_ : d_scratch_a_;
        float* dec_im = dec_re + kDecOutCh[i] * (cur_H * 2 + 10) * (cur_W + 2);

        forward_complex_tconv2d(cur_re, cur_im,
                                dec_re, dec_im,
                                kDecInCh[i], kDecOutCh[i], cur_H, cur_W,
                                kDecKernH[i], kDecKernW[i],
                                kStrideH, kStrideW, kPadH, kPadW,
                                dec_name + ".transconv", Ho, Wo);

        // BN + LeakyReLU
        forward_complex_bn_relu(dec_re, dec_im,
                                kDecOutCh[i], Ho, Wo,
                                dec_name + ".bn");

        cur_C = kDecOutCh[i];
        cur_H = Ho;
        cur_W = Wo;

        // FSMN dec (skip for last decoder: "if i < model_length - 1")
        if (i < kNumStages - 1) {
            std::string fsmn_name = prefix + ".fsmn_dec" + std::to_string(i);
            forward_complex_fsmn_l1(dec_re, dec_im, cur_C, cur_H, cur_W, fsmn_name);
        }

        // Last decoder: break before SE and concat
        if (i == kNumStages - 1) {
            cur_re = dec_re;
            cur_im = dec_im;
            break;
        }

        // SE layer on decoder output (skip for second-to-last: "if i < model_length - 2")
        if (i < kNumStages - 2) {
            std::string se_name = prefix + ".se_layer_dec" + std::to_string(i);
            forward_se_layer(dec_re, dec_im, cur_C, cur_H, cur_W, se_name);
        }

        // Skip connection: concat(decoder_output, enc_skip[model_length - 1 - i])
        int skip_idx = kNumStages - 1 - i;
        int skip_C = (skip_idx == 0) ? 1 : kChannels;  // skip_idx 0 is input (1 channel)
        int cur_HW = cur_H * cur_W;

        // Use a fresh buffer for concatenated result
        float* cat_re = (i % 2 == 0) ? d_scratch_a_ : d_scratch_b_;
        int total_C = cur_C + skip_C;
        float* cat_im = cat_re + total_C * cur_HW;

        // Concatenate along channel dim
        launch_concat_channels(dec_re, enc_skip_[skip_idx].re,
                               cat_re, cur_C, skip_C, cur_HW, stream_);
        launch_concat_channels(dec_im, enc_skip_[skip_idx].im,
                               cat_im, cur_C, skip_C, cur_HW, stream_);

        cur_re = cat_re;
        cur_im = cat_im;
        cur_C = total_C;
    }

    // ---- Output linear: ComplexConv2d(dec_channels[-1]=1, 1, kernel=1) ----
    // Wait, looking at the Python: linear = ComplexConv2d(dec_channels[-1], 1, 1)
    // dec_channels[-1] = 1, so it's ComplexConv2d(1, 1, 1).
    // It's a 1×1 conv from 1 channel to 1 channel — essentially a learned scalar per freq bin.
    int Ho, Wo;
    forward_complex_conv2d(cur_re, cur_im,
                           re_out, im_out,
                           cur_C, 1, cur_H, cur_W,
                           1, 1, 1, 1, 0, 0,
                           prefix + ".linear", Ho, Wo);
}

// ============================================================================
// Main enhance function
// ============================================================================

int FrcrnGpu::enhance(const float* d_pcm_in, float* d_pcm_out, int n_samples) {
    if (!initialized_) return 0;
    if (n_samples <= kWinLen) return 0;

    auto t0 = std::chrono::steady_clock::now();

    // Edge padding: kWinLen/2 zeros at start and end (center-style STFT)
    int edge_pad = kWinLen / 2;  // 320
    int total = n_samples + 2 * edge_pad;

    // Pad to hop alignment
    int padded = total;
    if (padded % kHop != 0) {
        padded = ((padded / kHop) + 1) * kHop;
    }
    if (padded < kWinLen) padded = kWinLen;

    // Build padded buffer: [zeros | input | zeros]
    // Handle potential aliasing: d_pcm_in might point into d_pcm_staging_
    // Use device-to-device memmove-safe approach: copy input first, then zero edges
    if (d_pcm_in != d_pcm_staging_ + edge_pad) {
        // If source != dest, can copy directly then zero edges
        CUDA_CHECK(cudaMemcpyAsync(d_pcm_staging_ + edge_pad, d_pcm_in,
                                   n_samples * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_));
    }
    // If d_pcm_in == d_pcm_staging_ (from enhance_host), we need memmove.
    // Since edge_pad > 0 and source starts before dest, this is a forward
    // overlapping copy — use the OLA buffer as a temporary.
    else if (d_pcm_in == d_pcm_staging_) {
        CUDA_CHECK(cudaMemcpyAsync(d_ola_buf_, d_pcm_in,
                                   n_samples * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_pcm_staging_ + edge_pad, d_ola_buf_,
                                   n_samples * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_));
    }
    // Zero the edges
    CUDA_CHECK(cudaMemsetAsync(d_pcm_staging_, 0, edge_pad * sizeof(float), stream_));
    int trail_start = edge_pad + n_samples;
    if (trail_start < padded) {
        CUDA_CHECK(cudaMemsetAsync(d_pcm_staging_ + trail_start, 0,
                                   (padded - trail_start) * sizeof(float), stream_));
    }

    const float* pcm = d_pcm_staging_;

    // 1. STFT
    int n_frames = 0;
    forward_stft(pcm, padded, n_frames);

    // spec_re, spec_im now contain [kFreqBins, n_frames]
    int spec_n = kFreqBins * n_frames;

    // 2. Reshape: [kFreqBins, n_frames] → [1, kFreqBins, n_frames] (already in this layout)
    // UNet input is [1, kFreqBins, n_frames] where C=1, H=kFreqBins, W=n_frames

    // 3. UNet1: spec → mask1 (output to d_mask1_re/im, then we need unet1_raw for UNet2)
    // Store UNet1 raw output before tanh
    float* unet1_re = d_mask1_re_;
    float* unet1_im = d_mask1_im_;

    forward_unet(d_spec_re_, d_spec_im_, unet1_re, unet1_im,
                 kFreqBins, n_frames, "unet");


    // 4. mask1 = tanh(unet1_out)
    // But we also need the raw unet1 output as input to unet2!
    // Save raw unet1 output first.
    float* unet1_raw_re = d_mask2_re_;  // temporarily use mask2 buffers
    float* unet1_raw_im = d_mask2_im_;
    CUDA_CHECK(cudaMemcpyAsync(unet1_raw_re, unet1_re, spec_n * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(unet1_raw_im, unet1_im, spec_n * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));

    launch_tanh(unet1_re, spec_n, stream_);
    launch_tanh(unet1_im, spec_n, stream_);
    // d_mask1_re/im now contains mask1

    // 5. UNet2: unet1_raw → mask2_raw
    float* unet2_re = d_mask2_re_;
    float* unet2_im = d_mask2_im_;

    forward_unet(unet1_raw_re, unet1_raw_im, unet2_re, unet2_im,
                 kFreqBins, n_frames, "unet2");


    // 6. mask2 = mask1 + tanh(unet2_out)
    launch_tanh(unet2_re, spec_n, stream_);
    launch_tanh(unet2_im, spec_n, stream_);

    launch_add_inplace(unet2_re, unet1_re, spec_n, stream_);
    launch_add_inplace(unet2_im, unet1_im, spec_n, stream_);

    // d_mask2_re/im now contains final mask

    // 7. Apply complex mask: est_spec = spec * mask2
    launch_complex_mask(d_spec_re_, d_spec_im_,
                        d_mask2_re_, d_mask2_im_,
                        d_spec_re_, d_spec_im_,  // in-place overwrite
                        spec_n, stream_);


    // 8. iSTFT → padded output into staging buffer
    forward_istft(n_frames, d_pcm_staging_, padded);

    // 9. Trim edge padding: copy [edge_pad, edge_pad + n_samples) to output
    CUDA_CHECK(cudaMemcpyAsync(d_pcm_out, d_pcm_staging_ + edge_pad,
                               n_samples * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream_));

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto t1 = std::chrono::steady_clock::now();
    last_lat_ms_ = std::chrono::duration<float, std::milli>(t1 - t0).count();

    return n_samples;
}

// ============================================================================
// Host-memory convenience wrapper
// ============================================================================

int FrcrnGpu::enhance_host(const float* pcm_in, float* pcm_out, int n_samples) {
    if (!initialized_) return 0;
    if (n_samples > max_samples_) return 0;

    CUDA_CHECK(cudaMemcpyAsync(d_pcm_staging_, pcm_in,
                               n_samples * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));

    int out_len = enhance(d_pcm_staging_, d_pcm_staging_, n_samples);

    CUDA_CHECK(cudaMemcpyAsync(pcm_out, d_pcm_staging_,
                               out_len * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    return out_len;
}


}  // namespace deusridet
