/**
 * @file diarizen_resnet34_embedder.cu
 * @philosophical_role P2a of the DiariZen native CUDA port: the WeSpeaker
 *   ResNet34-LM x-vector embedder. A 16 kHz chunk plus a per-frame speaker
 *   activity mask becomes a 256-d embedding. The Kaldi-Hamming fbank reuses
 *   the existing GPU PoveyFbank kernel; every convolution is cuDNN, the final
 *   projection is cuBLAS, and BatchNorm is folded into per-channel scale/bias
 *   at load. The CPU only sequences the blocks and folds the norms.
 * @serves DiarizenResnet34Embedder. Bit-checked against the pyannote
 *   WeSpeakerResNet34 reference embeddings (P2a bit-eq harness).
 */
#include "diarizen_resnet34_embedder.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../communis/log.h"
#include "../machina/safetensors.h"
#include "../machina/tensor.h"
#include "../sensus/auditus/povey_fbank_gpu.h"
#include "diarizen_resnet34_kernels.cuh"

namespace deusridet {
namespace orator {

namespace {
constexpr const char* kRLog = "DiariZenResNet34";
constexpr int kBlock = 256;
inline int grid(int n) { return (n + kBlock - 1) / kBlock; }

// Portable IEEE fp16 -> fp32 (host).
inline float half_to_float_host(std::uint16_t h) {
    std::uint32_t sign = (std::uint32_t)(h & 0x8000) << 16;
    std::uint32_t exp = (h & 0x7C00) >> 10;
    std::uint32_t mant = (h & 0x03FF);
    std::uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}
}  // namespace

DiarizenResnet34Embedder::DiarizenResnet34Embedder() = default;

DiarizenResnet34Embedder::~DiarizenResnet34Embedder() {
    for (float* p : owned_)
        if (p) cudaFree(p);
    if (d_window_) cudaFree(d_window_);
    if (d_mel_fb_) cudaFree(d_mel_fb_);
    if (d_ws_) cudaFree(d_ws_);
    if (d_bufA_) cudaFree(d_bufA_);
    if (d_bufB_) cudaFree(d_bufB_);
    if (d_bufC_) cudaFree(d_bufC_);
    if (d_bufD_) cudaFree(d_bufD_);
    if (d_pool_) cudaFree(d_pool_);
    if (d_wave_) cudaFree(d_wave_);
    if (d_fbank_) cudaFree(d_fbank_);
    if (d_backbone_) cudaFree(d_backbone_);
    for (auto& kv : conv_cache_) {
        ConvPlan& p = kv.second;
        if (p.in_d) cudnnDestroyTensorDescriptor(p.in_d);
        if (p.out_d) cudnnDestroyTensorDescriptor(p.out_d);
        if (p.filt_d) cudnnDestroyFilterDescriptor(p.filt_d);
        if (p.conv_d) cudnnDestroyConvolutionDescriptor(p.conv_d);
    }
    if (cudnn_) cudnnDestroy(cudnn_);
    if (blas_) cublasDestroy(blas_);
}

// --------------------------------------------------------------------------
// safetensors loader (fp16 -> host fp32)
// --------------------------------------------------------------------------
bool DiarizenResnet34Embedder::load_safetensors_(
    const std::string& path,
    std::unordered_map<std::string, HostTensor>& tensors) {
    SafetensorsFile sf(path);
    auto names = sf.tensor_names();
    if (names.empty()) {
        LOG_ERROR(kRLog, "no tensors in %s", path.c_str());
        return false;
    }
    for (const auto& n : names) {
        auto t = sf.get_tensor(n);
        if (!t) {
            LOG_ERROR(kRLog, "missing tensor %s", n.c_str());
            return false;
        }
        if (t->dtype() != DataType::FP16) {
            LOG_ERROR(kRLog, "%s not fp16", n.c_str());
            return false;
        }
        HostTensor ht;
        const auto& shp = t->shape();
        ht.shape.assign(shp.begin(), shp.end());
        const std::size_t numel = t->numel();
        ht.data.resize(numel);
        const auto* src = static_cast<const std::uint16_t*>(t->data());
        for (std::size_t i = 0; i < numel; ++i)
            ht.data[i] = half_to_float_host(src[i]);
        tensors.emplace(n, std::move(ht));
    }
    return true;
}

float* DiarizenResnet34Embedder::upload_(const std::vector<float>& v) {
    float* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(float));
    cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    owned_.push_back(d);
    return d;
}

void DiarizenResnet34Embedder::fold_bn_(
    const std::unordered_map<std::string, HostTensor>& t,
    const std::string& prefix, float** scale_out, float** bias_out) {
    const auto& g = t.at(prefix + ".weight").data;
    const auto& b = t.at(prefix + ".bias").data;
    const auto& m = t.at(prefix + ".running_mean").data;
    const auto& v = t.at(prefix + ".running_var").data;
    const int C = (int)g.size();
    std::vector<float> scale(C), bias(C);
    for (int c = 0; c < C; ++c) {
        float s = g[c] / std::sqrt(v[c] + DiarizenResnet34Arch::kBnEps);
        scale[c] = s;
        bias[c] = b[c] - m[c] * s;
    }
    *scale_out = upload_(scale);
    *bias_out = upload_(bias);
}

void DiarizenResnet34Embedder::build_block_(
    const std::unordered_map<std::string, HostTensor>& t,
    const std::string& prefix, int in_planes, int planes, int stride,
    Resnet34Block& blk) {
    blk.in_planes = in_planes;
    blk.planes = planes;
    blk.stride = stride;
    blk.conv1_w = upload_(t.at(prefix + ".conv1.weight").data);
    fold_bn_(t, prefix + ".bn1", &blk.bn1_scale, &blk.bn1_bias);
    blk.conv2_w = upload_(t.at(prefix + ".conv2.weight").data);
    fold_bn_(t, prefix + ".bn2", &blk.bn2_scale, &blk.bn2_bias);
    blk.has_shortcut = (stride != 1 || in_planes != planes);
    if (blk.has_shortcut) {
        blk.sc_w = upload_(t.at(prefix + ".shortcut.0.weight").data);
        fold_bn_(t, prefix + ".shortcut.1", &blk.sc_scale, &blk.sc_bias);
    }
}

// --------------------------------------------------------------------------
// fbank frontend (Kaldi-compatible Hamming, 80 mel, CMN) — reuses the GPU
// PoveyFbank kernel.
// --------------------------------------------------------------------------
void DiarizenResnet34Embedder::build_frontend_() {
    using A = DiarizenResnet34Arch;
    const int M = A::kNumMel, FL = A::kFrameLen, NF = A::kNFft;
    const int freq = NF / 2 + 1;
    cudaMalloc(&d_window_, FL * sizeof(float));
    cudaMalloc(&d_mel_fb_, M * freq * sizeof(float));

    // Hamming window: 0.54 - 0.46*cos(2*pi*n/(N-1)).
    std::vector<float> win(FL);
    for (int i = 0; i < FL; ++i)
        win[i] = 0.54f - 0.46f * std::cos(2.0f * (float)M_PI * i / (FL - 1));
    cudaMemcpy(d_window_, win.data(), FL * sizeof(float),
               cudaMemcpyHostToDevice);

    // Kaldi triangular mel filterbank (low=20 Hz, high=nyquist), matches
    // torchaudio.compliance.kaldi.get_mel_banks.
    auto hz2mel = [](float hz) { return 1127.0f * std::log(1.0f + hz / 700.0f); };
    float mel_low = hz2mel(20.0f);
    float mel_high = hz2mel((float)A::kSampleRate / 2.0f);
    float mel_delta = (mel_high - mel_low) / (M + 1);
    float bin_w = (float)A::kSampleRate / NF;
    std::vector<float> fb(M * freq, 0.0f);
    for (int m = 0; m < M; ++m) {
        float lm = mel_low + m * mel_delta;
        float cm = mel_low + (m + 1) * mel_delta;
        float rm = mel_low + (m + 2) * mel_delta;
        for (int k = 0; k < freq; ++k) {
            float mk = hz2mel(k * bin_w);
            float up = (cm > lm) ? (mk - lm) / (cm - lm) : 0.0f;
            float dn = (rm > cm) ? (rm - mk) / (rm - cm) : 0.0f;
            float w = up < dn ? up : dn;
            fb[m * freq + k] = w > 0.0f ? w : 0.0f;
        }
    }
    cudaMemcpy(d_mel_fb_, fb.data(), M * freq * sizeof(float),
               cudaMemcpyHostToDevice);
}

// Compute the CMN fbank [T, 80] into d_fbank_TM (device). Returns T.
int DiarizenResnet34Embedder::compute_fbank_(const float* wave, int n_samples,
                                             float* d_fbank_TM) {
    using A = DiarizenResnet34Arch;
    const int FL = A::kFrameLen, NF = A::kNFft, HOP = A::kFrameShift,
              M = A::kNumMel;
    if (n_samples < FL) return 0;
    const int T = (n_samples - FL) / HOP + 1;
    // Upload waveform scaled to int16 range (kaldi.fbank input = wave * 2^15).
    if ((size_t)n_samples > wave_cap_) {
        if (d_wave_) cudaFree(d_wave_);
        cudaMalloc(&d_wave_, n_samples * sizeof(float));
        wave_cap_ = n_samples;
    }
    std::vector<float> scaled(n_samples);
    for (int i = 0; i < n_samples; ++i) scaled[i] = wave[i] * 32768.0f;
    cudaMemcpyAsync(d_wave_, scaled.data(), n_samples * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
    // The scaled[] host staging buffer is reclaimed at function exit, so the
    // async H2D must complete before then: synchronise the stream here. (On
    // the default stream this is equivalent to the old blocking copy.)
    cudaStreamSynchronize(stream_);
    // Kaldi log floor = FLT_EPSILON.
    launch_povey_fbank(d_wave_, d_window_, d_mel_fb_, d_fbank_TM, T,
                       /*pcm_offset=*/0, FL, NF, HOP, M, 1.1920929e-07f,
                       stream_);
    // CMN: subtract per-mel time mean.
    r34_cmn<<<grid(M), kBlock, 0, stream_>>>(d_fbank_TM, T, M);
    return T;
}

bool DiarizenResnet34Embedder::load(const std::string& path) {
    using A = DiarizenResnet34Arch;
    std::unordered_map<std::string, HostTensor> t;
    if (!load_safetensors_(path, t)) return false;

    cudnnCreate(&cudnn_);
    cublasCreate(&blas_);

    conv1_w_ = upload_(t.at("resnet.conv1.weight").data);
    fold_bn_(t, "resnet.bn1", &bn1_scale_, &bn1_bias_);

    int in_planes = A::kBaseWidth;
    for (int s = 0; s < A::kNumStages; ++s) {
        int planes = A::kPlanes[s];
        for (int b = 0; b < A::kBlocks[s]; ++b) {
            char pfx[64];
            std::snprintf(pfx, sizeof(pfx), "resnet.layer%d.%d", s + 1, b);
            int stride = (b == 0 && s > 0) ? 2 : 1;
            Resnet34Block blk;
            build_block_(t, pfx, in_planes, planes, stride, blk);
            blocks_.push_back(blk);
            in_planes = planes;
        }
    }
    seg1_w_ = upload_(t.at("resnet.seg_1.weight").data);
    seg1_b_ = upload_(t.at("resnet.seg_1.bias").data);

    build_frontend_();

    // Workspace + activation buffers (sized for the 16 s chunk).
    const int MAXT = 1600;
    buf_floats_ = (size_t)A::kBaseWidth * A::kNumMel * MAXT;
    cudaMalloc(&d_bufA_, buf_floats_ * sizeof(float));
    cudaMalloc(&d_bufB_, buf_floats_ * sizeof(float));
    cudaMalloc(&d_bufC_, buf_floats_ * sizeof(float));
    cudaMalloc(&d_bufD_, buf_floats_ * sizeof(float));
    cudaMalloc(&d_pool_, A::kPoolDim * sizeof(float));
    fbank_cap_floats_ = (size_t)MAXT * A::kNumMel;
    cudaMalloc(&d_fbank_, fbank_cap_floats_ * sizeof(float));
    // Backbone features [kStatsDim, Tp<=MAXT]; same capacity as the activation
    // buffers so any valid Tp fits.
    backbone_cap_floats_ = buf_floats_;
    cudaMalloc(&d_backbone_, backbone_cap_floats_ * sizeof(float));
    ws_bytes_ = 64 * 1024 * 1024;
    cudaMalloc(&d_ws_, ws_bytes_);

    loaded_ = true;
    LOG_INFO(kRLog, "loaded ResNet34-LM: %zu blocks, %zu param buffers",
             blocks_.size(), owned_.size());
    return true;
}

void DiarizenResnet34Embedder::set_stream(cudaStream_t s) {
    stream_ = s;
    if (cudnn_) cudnnSetStream(cudnn_, s);
    if (blas_) cublasSetStream(blas_, s);
}

}  // namespace orator
}  // namespace deusridet
