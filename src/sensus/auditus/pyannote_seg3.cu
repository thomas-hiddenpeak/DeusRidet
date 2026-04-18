// pyannote_seg3.cu — Native CUDA inference for pyannote/segmentation-3.0.
//
// SincNet + 4x BiLSTM + Linear head.
// Uses cuDNN for BiLSTM, cuBLAS for linear, custom kernels for the rest.

#include "pyannote_seg3.h"
#include "../../communis/log.h"
#include "../../machina/safetensors.h"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace deusridet {

// ============================================================
// CUDA Kernels
// ============================================================

// Instance Normalization: y = (x - mean) / sqrt(var + eps) * w + b
// Input: (C, L), one norm per channel
static __global__ void instance_norm_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     const float* __restrict__ weight,
                                     const float* __restrict__ bias,
                                     int C, int L, float eps) {
    int c = blockIdx.x;
    if (c >= C) return;

    const float* row = input + c * L;
    float* out_row = output + c * L;
    float w = weight ? weight[c] : 1.0f;
    float b = bias ? bias[c] : 0.0f;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        sum += row[i];
    }
    // Intra-warp reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float s_sum[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    if (lane == 0) s_sum[wid] = sum;
    __syncthreads();
    // Inter-warp reduce — all threads in warp 0 must call __shfl_down_sync
    if (wid == 0) {
        int nwarps = blockDim.x / warpSize;
        sum = (lane < nwarps) ? s_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) s_sum[0] = sum;
    }
    __syncthreads();
    float mean = s_sum[0] / (float)L;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        float d = row[i] - mean;
        var_sum += d * d;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) s_sum[wid] = var_sum;
    __syncthreads();
    if (wid == 0) {
        int nwarps = blockDim.x / warpSize;
        var_sum = (lane < nwarps) ? s_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        if (lane == 0) s_sum[0] = var_sum;
    }
    __syncthreads();
    float inv_std = rsqrtf(s_sum[0] / (float)L + eps);

    // Normalize
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        out_row[i] = (row[i] - mean) * inv_std * w + b;
    }
}

// Conv1d: (C_in, L_in) * (C_out, C_in, K) + bias → (C_out, L_out)
// L_out = (L_in - K) / stride + 1
static __global__ void conv1d_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int C_in, int L_in, int C_out, int K, int stride,
                              int L_out) {
    int co = blockIdx.x;
    int t  = blockIdx.y * blockDim.x + threadIdx.x;
    if (co >= C_out || t >= L_out) return;

    float sum = bias ? bias[co] : 0.0f;
    int t_in = t * stride;
    const float* w = weight + co * C_in * K;
    for (int ci = 0; ci < C_in; ci++) {
        const float* x = input + ci * L_in + t_in;
        const float* ww = w + ci * K;
        for (int k = 0; k < K; k++) {
            sum += x[k] * ww[k];
        }
    }
    output[co * L_out + t] = sum;
}

// Abs kernel: y = |x|
static __global__ void abs_kernel(float* __restrict__ data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = fabsf(data[idx]);
}

// MaxPool1d: (C, L_in) → (C, L_out), kernel=3, stride=3
static __global__ void maxpool1d_k3_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int C, int L_in, int L_out) {
    int c = blockIdx.x;
    int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= C || t >= L_out) return;

    int base = c * L_in + t * 3;
    float a = input[base];
    float b = (t * 3 + 1 < L_in) ? input[base + 1] : -1e30f;
    float c_ = (t * 3 + 2 < L_in) ? input[base + 2] : -1e30f;
    output[c * L_out + t] = fmaxf(a, fmaxf(b, c_));
}

// LeakyReLU: y = x > 0 ? x : slope * x
static __global__ void leaky_relu_kernel(float* __restrict__ data, int N, float slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = data[idx];
        data[idx] = v > 0.0f ? v : v * slope;
    }
}

// Transpose (C, L) → (L, C)
static __global__ void transpose_2d_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        output[c * rows + r] = input[r * cols + c];
    }
}

// Add bias to each row: out[t, c] += bias[c], for (T, C)
static __global__ void add_bias_kernel(float* __restrict__ data,
                                const float* __restrict__ bias,
                                int T, int C) {
    int t = blockIdx.x;
    int c = threadIdx.x;
    if (t < T && c < C) {
        data[t * C + c] += bias[c];
    }
}

// LogSoftmax over last dim: (T, C) → (T, C)
static __global__ void log_softmax_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int T, int C) {
    int t = blockIdx.x;
    if (t >= T) return;

    const float* row = input + t * C;
    float* out_row = output + t * C;

    // Find max
    float max_val = row[0];
    for (int c = 1; c < C; c++) {
        max_val = fmaxf(max_val, row[c]);
    }

    // Sum exp
    float sum_exp = 0.0f;
    for (int c = 0; c < C; c++) {
        sum_exp += __expf(row[c] - max_val);
    }

    float log_sum = __logf(sum_exp) + max_val;
    for (int c = 0; c < C; c++) {
        out_row[c] = row[c] - log_sum;
    }
}

// ============================================================
// Constructor / Destructor
// ============================================================

PyannoteSeg3::PyannoteSeg3() = default;

PyannoteSeg3::~PyannoteSeg3() {
    // Free weight buffers
    auto free_gpu = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    free_gpu(d_wav_norm_w_); free_gpu(d_wav_norm_b_);
    free_gpu(d_sinc_filters_);
    free_gpu(d_norm0_w_); free_gpu(d_norm0_b_);
    free_gpu(d_conv1_w_); free_gpu(d_conv1_b_);
    free_gpu(d_norm1_w_); free_gpu(d_norm1_b_);
    free_gpu(d_conv2_w_); free_gpu(d_conv2_b_);
    free_gpu(d_norm2_w_); free_gpu(d_norm2_b_);
    free_gpu(d_linear0_w_); free_gpu(d_linear0_b_);
    free_gpu(d_linear1_w_); free_gpu(d_linear1_b_);
    free_gpu(d_classifier_w_); free_gpu(d_classifier_b_);
    for (int i = 0; i < kLstmLayers; i++) {
        free_gpu(d_lstm_Wih_[i]);
        free_gpu(d_lstm_Whh_[i]);
        free_gpu(d_lstm_bias_[i]);
        free_gpu(d_lstm_weights_[i]);
    }

    // Free scratch buffers
    free_gpu(d_sinc_out_); free_gpu(d_pool0_);
    free_gpu(d_norm_tmp_);
    free_gpu(d_conv1_out_); free_gpu(d_pool1_);
    free_gpu(d_conv2_out_); free_gpu(d_pool2_);
    free_gpu(d_lstm_in_); free_gpu(d_lstm_out_);
    free_gpu(d_linear_buf_); free_gpu(d_precomp_gates_);
    free_gpu(d_hx_); free_gpu(d_cx_);
    if (d_lstm_workspace_) { cudaFree(d_lstm_workspace_); d_lstm_workspace_ = nullptr; }

    // Destroy cuDNN descriptors
    if (rnn_desc_) cudnnDestroyRNNDescriptor(rnn_desc_);
    if (x_desc_)   cudnnDestroyRNNDataDescriptor(x_desc_);
    if (y_desc_)   cudnnDestroyRNNDataDescriptor(y_desc_);
    if (h_desc_)   cudnnDestroyTensorDescriptor(h_desc_);
    if (c_desc_)   cudnnDestroyTensorDescriptor(c_desc_);
    if (w_desc_)   cudnnDestroyTensorDescriptor(w_desc_);

    if (cudnn_)  cudnnDestroy(cudnn_);
    if (cublas_) cublasDestroy(cublas_);
    if (own_stream_ && stream_) cudaStreamDestroy(stream_);
}

// ============================================================
// Weight Loading
// ============================================================

bool PyannoteSeg3::load_weights(const std::string& model_path) {
    SafetensorsFile loader(model_path);

    auto upload = [&](const char* name, float** d_ptr, size_t expected_bytes) -> bool {
        auto t = loader.get_tensor(name);
        if (!t) {
            LOG_ERROR("Seg3", "Missing tensor: %s", name);
            return false;
        }
        if (t->nbytes() != expected_bytes) {
            LOG_ERROR("Seg3", "Tensor %s: expected %zu bytes, got %zu",
                      name, expected_bytes, t->nbytes());
            return false;
        }
        cudaMalloc(d_ptr, expected_bytes);
        cudaMemcpy(*d_ptr, t->data(), expected_bytes, cudaMemcpyHostToDevice);
        return true;
    };

    // SincNet weights
    if (!upload("sincnet.wav_norm1d.weight", &d_wav_norm_w_, 1 * 4)) return false;
    if (!upload("sincnet.wav_norm1d.bias", &d_wav_norm_b_, 1 * 4)) return false;
    if (!upload("/sincnet/conv1d.0/Concat_2_output_0", &d_sinc_filters_,
                kSincFilters * 1 * kSincKernel * 4)) return false;
    if (!upload("sincnet.norm1d.0.weight", &d_norm0_w_, kSincFilters * 4)) return false;
    if (!upload("sincnet.norm1d.0.bias", &d_norm0_b_, kSincFilters * 4)) return false;
    if (!upload("sincnet.conv1d.1.weight", &d_conv1_w_, kConv1Filters * kSincFilters * kConv1Kernel * 4)) return false;
    if (!upload("sincnet.conv1d.1.bias", &d_conv1_b_, kConv1Filters * 4)) return false;
    if (!upload("sincnet.norm1d.1.weight", &d_norm1_w_, kConv1Filters * 4)) return false;
    if (!upload("sincnet.norm1d.1.bias", &d_norm1_b_, kConv1Filters * 4)) return false;
    if (!upload("sincnet.conv1d.2.weight", &d_conv2_w_, kConv2Filters * kConv1Filters * kConv2Kernel * 4)) return false;
    if (!upload("sincnet.conv1d.2.bias", &d_conv2_b_, kConv2Filters * 4)) return false;
    if (!upload("sincnet.norm1d.2.weight", &d_norm2_w_, kConv2Filters * 4)) return false;
    if (!upload("sincnet.norm1d.2.bias", &d_norm2_b_, kConv2Filters * 4)) return false;

    // LSTM weights — ONNX format: W_ih=(2, 4*hidden, input), W_hh=(2, 4*hidden, hidden), bias=(2, 4*hidden*2)
    // But ONNX stores bias as (2, 8*hidden) = input_bias + hidden_bias concatenated
    // cuDNN format is different, so we store raw and pack later
    struct LstmInfo {
        const char* w_name;   // W_ih
        const char* r_name;   // W_hh
        const char* b_name;   // bias
        int input_size;
    };
    LstmInfo lstm_info[kLstmLayers] = {
        {"onnx::LSTM_784", "onnx::LSTM_785", "onnx::LSTM_783", 60},
        {"onnx::LSTM_827", "onnx::LSTM_828", "onnx::LSTM_826", 256},
        {"onnx::LSTM_870", "onnx::LSTM_871", "onnx::LSTM_869", 256},
        {"onnx::LSTM_913", "onnx::LSTM_914", "onnx::LSTM_912", 256},
    };

    for (int i = 0; i < kLstmLayers; i++) {
        int in_sz = lstm_info[i].input_size;
        size_t wih_bytes = 2 * 4 * kLstmHidden * in_sz * 4;
        size_t whh_bytes = 2 * 4 * kLstmHidden * kLstmHidden * 4;
        size_t bias_bytes = 2 * 4 * kLstmHidden * 4;  // ONNX: (2, 4*hidden) but actually (2, 8*hidden) split

        // ONNX bias shape is (2, 8*hidden) where first 4*hidden is input bias, next 4*hidden is hidden bias
        // But checking the reference: shape=(2, 1024) where 1024 = 8*128
        // So total bias = 2 * 8 * 128 = 2048 floats = 8192 bytes
        size_t bias_full_bytes = 2 * 8 * kLstmHidden * 4;

        auto wih_t = loader.get_tensor(lstm_info[i].w_name);
        auto whh_t = loader.get_tensor(lstm_info[i].r_name);
        auto bias_t = loader.get_tensor(lstm_info[i].b_name);
        if (!wih_t || !whh_t || !bias_t) {
            LOG_ERROR("Seg3", "Missing LSTM layer %d weights", i);
            return false;
        }

        cudaMalloc(&d_lstm_Wih_[i], wih_t->nbytes());
        cudaMemcpy(d_lstm_Wih_[i], wih_t->data(), wih_t->nbytes(), cudaMemcpyHostToDevice);

        cudaMalloc(&d_lstm_Whh_[i], whh_t->nbytes());
        cudaMemcpy(d_lstm_Whh_[i], whh_t->data(), whh_t->nbytes(), cudaMemcpyHostToDevice);

        cudaMalloc(&d_lstm_bias_[i], bias_t->nbytes());
        cudaMemcpy(d_lstm_bias_[i], bias_t->data(), bias_t->nbytes(), cudaMemcpyHostToDevice);
    }

    // Linear head
    if (!upload("onnx::MatMul_915", &d_linear0_w_, 256 * 128 * 4)) return false;
    if (!upload("linear.0.bias", &d_linear0_b_, 128 * 4)) return false;
    if (!upload("onnx::MatMul_916", &d_linear1_w_, 128 * 128 * 4)) return false;
    if (!upload("linear.1.bias", &d_linear1_b_, 128 * 4)) return false;
    if (!upload("onnx::MatMul_917", &d_classifier_w_, 128 * 7 * 4)) return false;
    if (!upload("classifier.bias", &d_classifier_b_, 7 * 4)) return false;

    LOG_INFO("Seg3", "Loaded weights (%.1f KB) from %s",
             0.0f, model_path.c_str());
    return true;
}

// ============================================================
// cuDNN LSTM Setup
// ============================================================

bool PyannoteSeg3::setup_cudnn_lstm() {
    // We'll run LSTM layers one at a time with cuDNN since each layer
    // has different input size (layer 0: 60, layers 1-3: 256).
    // For simplicity, use cuBLAS-based manual LSTM to avoid cuDNN weight
    // format complexity. cuDNN RNN API requires specific weight packing
    // that differs from ONNX format.
    //
    // Actually, let's implement LSTM manually with cuBLAS GEMM.
    // For seq_len=589, hidden=128, bidirectional — this is small enough
    // that manual implementation is straightforward and avoids cuDNN
    // weight format conversion headaches.

    return true;
}

// ============================================================
// Init
// ============================================================

bool PyannoteSeg3::init(const std::string& model_path, cudaStream_t stream) {
    if (stream) {
        stream_ = stream;
        own_stream_ = false;
    } else {
        cudaStreamCreate(&stream_);
        own_stream_ = true;
    }

    cublasCreate(&cublas_);
    cublasSetStream(cublas_, stream_);

    if (!load_weights(model_path)) return false;

    // Allocate scratch buffers
    cudaMalloc(&d_sinc_out_,   kSincFilters * kSincOutLen * sizeof(float));
    cudaMalloc(&d_pool0_,      kSincFilters * kPool0OutLen * sizeof(float));
    cudaMalloc(&d_norm_tmp_,   kSincFilters * kPool0OutLen * sizeof(float));
    cudaMalloc(&d_conv1_out_,  kConv1Filters * kConv1OutLen * sizeof(float));
    cudaMalloc(&d_pool1_,      kConv1Filters * kPool1OutLen * sizeof(float));
    cudaMalloc(&d_conv2_out_,  kConv2Filters * kConv2OutLen * sizeof(float));
    cudaMalloc(&d_pool2_,      kConv2Filters * kPool2OutLen * sizeof(float));
    cudaMalloc(&d_lstm_in_,    kSeqLen * kLstmBiDir * sizeof(float));  // max(60, 256) → 256
    cudaMalloc(&d_lstm_out_,   kSeqLen * kLstmBiDir * sizeof(float));
    cudaMalloc(&d_linear_buf_, kSeqLen * kLstmBiDir * sizeof(float));
    cudaMalloc(&d_precomp_gates_, kSeqLen * 4 * kLstmHidden * sizeof(float)); // (589, 512)
    cudaMalloc(&d_hx_,         2 * kLstmHidden * sizeof(float));
    cudaMalloc(&d_cx_,         2 * kLstmHidden * sizeof(float));

    // Zero h/c states
    cudaMemset(d_hx_, 0, 2 * kLstmHidden * sizeof(float));
    cudaMemset(d_cx_, 0, 2 * kLstmHidden * sizeof(float));

    initialized_ = true;
    LOG_INFO("Seg3", "Native CUDA ready: %d frames, %d classes",
             kNumFrames, kNumClasses);
    return true;
}

// ============================================================
// LSTM Forward (optimized: batched GEMM + fused per-step kernel)
// ============================================================

// Fused kernel: loads Whh from global, computes Whh @ h_prev in shared memory,
// adds to precomputed gates, applies LSTM gate activations, updates h and c.
// One block per timestep direction — blockDim.x = hidden (128).
// This avoids per-timestep cuBLAS calls entirely.
static __global__ void lstm_fused_step_kernel(
    const float* __restrict__ precomputed_gates,  // (T, 4*H) — Wih@x + biases
    const float* __restrict__ Whh,                // (4*H, H)
    float* __restrict__ h,                        // (H,) current hidden state
    float* __restrict__ c,                        // (H,) current cell state
    float* __restrict__ output,                   // (T, 2*H) output buffer
    int T, int hidden, int dir, int t_idx) {

    int idx = threadIdx.x;  // 0..hidden-1
    if (idx >= hidden) return;

    extern __shared__ float smem[];
    float* s_h = smem;  // (hidden,)

    // Load h_prev into shared memory for Whh matmul
    s_h[idx] = h[idx];
    __syncthreads();

    // Compute Whh @ h for all 4 gates for this idx
    // Whh is (4*H, H), row-major: Whh[gate*H + idx, :] dot s_h[:]
    int gates_4 = 4 * hidden;
    float gate_vals[4];
    for (int g = 0; g < 4; g++) {
        float sum = 0.0f;
        const float* w_row = Whh + (g * hidden + idx) * hidden;
        for (int j = 0; j < hidden; j++) {
            sum += w_row[j] * s_h[j];
        }
        // Add precomputed gates (Wih @ x + bias_ih + bias_hh)
        gate_vals[g] = precomputed_gates[t_idx * gates_4 + g * hidden + idx] + sum;
    }

    // ONNX LSTM gate order: i, o, f, g
    float sig_i = 1.0f / (1.0f + __expf(-gate_vals[0]));
    float sig_o = 1.0f / (1.0f + __expf(-gate_vals[1]));
    float sig_f = 1.0f / (1.0f + __expf(-gate_vals[2]));
    float tanh_g = tanhf(gate_vals[3]);

    float c_new = sig_f * c[idx] + sig_i * tanh_g;
    float h_new = sig_o * tanhf(c_new);

    c[idx] = c_new;
    h[idx] = h_new;

    // Write to output: output[t_idx, dir*hidden + idx]
    output[t_idx * 2 * hidden + dir * hidden + idx] = h_new;
}

// Add bias kernel for precomputed gates: gates[t, :] += bias_ih[:] + bias_hh[:]
static __global__ void add_dual_bias_kernel(float* __restrict__ gates,
                                     const float* __restrict__ bias_ih,
                                     const float* __restrict__ bias_hh,
                                     int T, int gates_sz) {
    int t = blockIdx.x;
    int g = threadIdx.x;
    if (t < T && g < gates_sz) {
        gates[t * gates_sz + g] += bias_ih[g] + bias_hh[g];
    }
}

// Run one BiLSTM layer: (T, input_size) → (T, 2*hidden)
// Optimized: batch Wih GEMM for all timesteps, then fused per-step kernel
static void run_bilstm_layer(cublasHandle_t cublas, cudaStream_t stream,
                             const float* d_input, float* d_output,
                             const float* d_Wih, const float* d_Whh,
                             const float* d_bias,
                             float* d_precomp_gates,  // scratch: (T, 4*hidden)
                             float* d_h, float* d_c,
                             int T, int input_size, int hidden) {
    const float one = 1.0f, zero = 0.0f;
    int gates_sz = 4 * hidden;

    for (int dir = 0; dir < 2; dir++) {
        const float* Wih = d_Wih + dir * gates_sz * input_size;
        const float* Whh = d_Whh + dir * gates_sz * hidden;
        const float* bias_ih = d_bias + dir * 2 * gates_sz;
        const float* bias_hh = d_bias + dir * 2 * gates_sz + gates_sz;

        float* h = d_h + dir * hidden;
        float* c = d_c + dir * hidden;

        // Zero initial states
        cudaMemsetAsync(h, 0, hidden * sizeof(float), stream);
        cudaMemsetAsync(c, 0, hidden * sizeof(float), stream);

        // Step 1: Batch GEMM — precomp_gates(T, 4H) = input(T, in) @ Wih^T(in, 4H)
        // cuBLAS row-major trick: C(M,N) = A(M,K) @ B(K,N)
        // → cublasSgemm(N, N, N, M, K, alpha, B, N, A, K, beta, C, N)
        // Here: M=T, K=input_size, N=4*hidden
        // Wih is stored as (4H, input_size) row-major. We need input @ Wih^T.
        // Wih^T is (input_size, 4H). So B = Wih^T with N=4H.
        // cublasSgemm(N, N, 4H, T, in, 1, Wih^T, 4H, input, in, 0, gates, 4H)
        // But Wih^T in row-major = Wih in col-major. So we use Wih directly with OP_T:
        // cublasSgemm(OP_T, OP_N, 4H, T, in, 1, Wih, in, input, in, 0, gates, 4H)
        // Wait — let me think in cuBLAS col-major terms:
        // We want C_rm(T, 4H) = A_rm(T, in) @ Wih_rm(in, 4H)
        // But Wih is stored as (4H, in) row-major, i.e. Wih_rm(4H, in).
        // So we need A_rm(T, in) @ Wih_rm^T... no, we need (4H, in)^T = (in, 4H)
        // cuBLAS sees everything as col-major:
        //   C_cm = Wih_cm^T @ A_cm  where C_cm is (4H, T), Wih_cm is (in, 4H), A_cm is (in, T)
        // But Wih stored as (4H, in) row-major = (in, 4H) col-major. So Wih_cm is already (in, 4H).
        // A(T, in) row-major = (in, T) col-major.
        // C(T, 4H) row-major = (4H, T) col-major.
        // cublasSgemm(OP_T, OP_N, 4H, T, in, 1, Wih_cm(in,4H), in, A_cm(in,T), in, 0, C_cm(4H,T), 4H)
        cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    gates_sz, T, input_size,
                    &one,
                    Wih, input_size,       // Wih col-major: (in, 4H)
                    d_input, input_size,   // A col-major: (in, T)
                    &zero,
                    d_precomp_gates, gates_sz);  // C col-major: (4H, T) = (T, 4H) row-major

        // Step 2: Add biases to all timesteps
        add_dual_bias_kernel<<<T, gates_sz, 0, stream>>>(
            d_precomp_gates, bias_ih, bias_hh, T, gates_sz);

        // Step 3: Sequential per-timestep with fused kernel
        int smem_bytes = hidden * sizeof(float);
        for (int step = 0; step < T; step++) {
            int t = (dir == 0) ? step : (T - 1 - step);
            lstm_fused_step_kernel<<<1, hidden, smem_bytes, stream>>>(
                d_precomp_gates, Whh,
                h, c, d_output,
                T, hidden, dir, t);
        }
    }
}

void PyannoteSeg3::forward_sincnet(const float* d_pcm, int n_samples) {
    // Step 1: Instance norm on raw PCM (C=1, L=160000)
    // Use d_norm_tmp_ for normalized PCM (big enough: 80*5325 > 160000)
    instance_norm_kernel<<<1, 256, 0, stream_>>>(
        d_pcm, d_norm_tmp_,
        d_wav_norm_w_, d_wav_norm_b_,
        1, n_samples, 1e-5f);

    // Step 2: SincConv — Conv1d(1, 80, k=251, s=10)
    // Input: d_norm_tmp_ (1, 160000), Output: d_sinc_out_ (80, 15975)
    {
        int L_out = kSincOutLen;
        dim3 grid(kSincFilters, (L_out + 255) / 256);
        conv1d_kernel<<<grid, 256, 0, stream_>>>(
            d_norm_tmp_,  // normalized PCM (1, 160000)
            d_sinc_filters_, nullptr,  // no bias for SincNet
            d_sinc_out_,  // conv output (80, 15975)
            1, n_samples, kSincFilters, kSincKernel, kSincStride, L_out);
    }

    // Step 3: Abs
    {
        int N = kSincFilters * kSincOutLen;
        abs_kernel<<<(N + 255) / 256, 256, 0, stream_>>>(d_sinc_out_, N);
    }

    // Step 4: MaxPool(3) → (80, 5325)
    {
        dim3 grid(kSincFilters, (kPool0OutLen + 255) / 256);
        maxpool1d_k3_kernel<<<grid, 256, 0, stream_>>>(
            d_sinc_out_, d_pool0_, kSincFilters, kSincOutLen, kPool0OutLen);
    }

    // Step 5: InstanceNorm → LeakyReLU (80 channels, L=5325)
    instance_norm_kernel<<<kSincFilters, 256, 0, stream_>>>(
        d_pool0_, d_pool0_, d_norm0_w_, d_norm0_b_,
        kSincFilters, kPool0OutLen, 1e-5f);
    {
        int N = kSincFilters * kPool0OutLen;
        leaky_relu_kernel<<<(N + 255) / 256, 256, 0, stream_>>>(
            d_pool0_, N, kLeakySlope);
    }

    // Step 6: Conv1d(80→60, k=5, s=1) → (60, 5321)
    {
        int L_out = kConv1OutLen;
        dim3 grid(kConv1Filters, (L_out + 255) / 256);
        conv1d_kernel<<<grid, 256, 0, stream_>>>(
            d_pool0_, d_conv1_w_, d_conv1_b_, d_conv1_out_,
            kSincFilters, kPool0OutLen, kConv1Filters, kConv1Kernel, 1, L_out);
    }

    // Step 7: MaxPool(3) → (60, 1773)
    {
        dim3 grid(kConv1Filters, (kPool1OutLen + 255) / 256);
        maxpool1d_k3_kernel<<<grid, 256, 0, stream_>>>(
            d_conv1_out_, d_pool1_, kConv1Filters, kConv1OutLen, kPool1OutLen);
    }

    // Step 8: InstanceNorm → LeakyReLU (60 channels, L=1773)
    instance_norm_kernel<<<kConv1Filters, 256, 0, stream_>>>(
        d_pool1_, d_pool1_, d_norm1_w_, d_norm1_b_,
        kConv1Filters, kPool1OutLen, 1e-5f);
    {
        int N = kConv1Filters * kPool1OutLen;
        leaky_relu_kernel<<<(N + 255) / 256, 256, 0, stream_>>>(
            d_pool1_, N, kLeakySlope);
    }

    // Step 9: Conv1d(60→60, k=5, s=1) → (60, 1769)
    {
        int L_out = kConv2OutLen;
        dim3 grid(kConv2Filters, (L_out + 255) / 256);
        conv1d_kernel<<<grid, 256, 0, stream_>>>(
            d_pool1_, d_conv2_w_, d_conv2_b_, d_conv2_out_,
            kConv1Filters, kPool1OutLen, kConv2Filters, kConv2Kernel, 1, L_out);
    }

    // Step 10: MaxPool(3) → (60, 589)
    {
        dim3 grid(kConv2Filters, (kPool2OutLen + 255) / 256);
        maxpool1d_k3_kernel<<<grid, 256, 0, stream_>>>(
            d_conv2_out_, d_pool2_, kConv2Filters, kConv2OutLen, kPool2OutLen);
    }

    // Step 11: InstanceNorm → LeakyReLU (60 channels, L=589)
    instance_norm_kernel<<<kConv2Filters, 256, 0, stream_>>>(
        d_pool2_, d_pool2_, d_norm2_w_, d_norm2_b_,
        kConv2Filters, kPool2OutLen, 1e-5f);
    {
        int N = kConv2Filters * kPool2OutLen;
        leaky_relu_kernel<<<(N + 255) / 256, 256, 0, stream_>>>(
            d_pool2_, N, kLeakySlope);
    }

    // Step 12: Transpose (60, 589) → (589, 60) into d_lstm_in_
    {
        dim3 block(16, 16);
        dim3 grid_t((kPool2OutLen + 15) / 16, (kConv2Filters + 15) / 16);
        transpose_2d_kernel<<<grid_t, block, 0, stream_>>>(
            d_pool2_, d_lstm_in_, kConv2Filters, kPool2OutLen);
    }
}

void PyannoteSeg3::forward_lstm() {
    float* d_h = d_hx_;  // (2*hidden)
    float* d_c = d_cx_;  // (2*hidden)

    int input_sizes[kLstmLayers] = {60, 256, 256, 256};

    for (int layer = 0; layer < kLstmLayers; layer++) {
        float* in;
        float* out;

        // Alternate buffers to avoid aliasing
        if (layer == 0) {
            in = d_lstm_in_;
            out = d_lstm_out_;
        } else if (layer % 2 == 1) {
            in = d_lstm_out_;
            out = d_lstm_in_;
        } else {
            in = d_lstm_in_;
            out = d_lstm_out_;
        }

        run_bilstm_layer(cublas_, stream_,
                         in, out,
                         d_lstm_Wih_[layer], d_lstm_Whh_[layer],
                         d_lstm_bias_[layer],
                         d_precomp_gates_, d_h, d_c,
                         kSeqLen, input_sizes[layer], kLstmHidden);
    }

    // After 4 layers (kLstmLayers=4, even):
    // Layer 0: in=lstm_in, out=lstm_out
    // Layer 1: in=lstm_out, out=lstm_in
    // Layer 2: in=lstm_in, out=lstm_out
    // Layer 3: in=lstm_out, out=lstm_in
    // Final output is in d_lstm_in_
    // Copy to d_lstm_out_ for consistency with linear head
    if (kLstmLayers % 2 == 0) {
        // Even layers: last write was to d_lstm_out_ (layer 2) then d_lstm_in_ (layer 3)
        // Layer 3 (odd) writes to d_lstm_in_
        cudaMemcpyAsync(d_lstm_out_, d_lstm_in_,
                        kSeqLen * kLstmBiDir * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_);
    }
    // If odd layers, last write is already in d_lstm_out_ (no copy needed... but let's keep it simple)
}

void PyannoteSeg3::forward_linear_head(float* d_output) {
    const float one = 1.0f, zero = 0.0f;

    // LSTM output is in d_lstm_out_: (589, 256)
    // Linear 0: (589, 256) @ (256, 128) → (589, 128) + bias
    // ONNX has MatMul_915 as (256, 128), so: out = input @ weight
    // cuBLAS: C = alpha * A * B + beta * C
    // A = d_lstm_out_ (589, 256), B = d_linear0_w_ (256, 128)
    // C = d_linear_buf_ (589, 128)
    // cublasSgemm: C(m,n) = A(m,k) * B(k,n)
    // Column-major: cublasSgemm(N, N, n, m, k, ..., B, n, A, k, ..., C, n)
    // Actually for row-major matrices with cuBLAS:
    // C_row(M,N) = A_row(M,K) @ B_row(K,N)
    // In column-major terms: C_col = B_col^T @ A_col^T ??? No.
    // Simpler: cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)
    // where A is (M,K) row-major, B is (K,N) row-major, C is (M,N) row-major

    // Linear 0: d_lstm_out_(589,256) @ d_linear0_w_(256,128) → d_linear_buf_(589,128)
    cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                kLinear0Out, kSeqLen, kLstmBiDir,
                &one,
                d_linear0_w_, kLinear0Out,    // B(256,128) col-major = (128,256) row-major... 
                d_lstm_out_, kLstmBiDir,      // A(589,256)
                &zero,
                d_linear_buf_, kLinear0Out);   // C(589,128)

    // Add bias
    add_bias_kernel<<<kSeqLen, kLinear0Out, 0, stream_>>>(
        d_linear_buf_, d_linear0_b_, kSeqLen, kLinear0Out);

    // LeakyReLU
    leaky_relu_kernel<<<(kSeqLen * kLinear0Out + 255) / 256, 256, 0, stream_>>>(
        d_linear_buf_, kSeqLen * kLinear0Out, kLeakySlope);

    // Linear 1: d_linear_buf_(589,128) @ d_linear1_w_(128,128) → d_lstm_out_(589,128) [reuse]
    cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                kLinear1Out, kSeqLen, kLinear0Out,
                &one,
                d_linear1_w_, kLinear1Out,
                d_linear_buf_, kLinear0Out,
                &zero,
                d_lstm_out_, kLinear1Out);

    add_bias_kernel<<<kSeqLen, kLinear1Out, 0, stream_>>>(
        d_lstm_out_, d_linear1_b_, kSeqLen, kLinear1Out);

    leaky_relu_kernel<<<(kSeqLen * kLinear1Out + 255) / 256, 256, 0, stream_>>>(
        d_lstm_out_, kSeqLen * kLinear1Out, kLeakySlope);

    // Classifier: d_lstm_out_(589,128) @ d_classifier_w_(128,7) → d_output(589,7)
    cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                kNumClasses, kSeqLen, kLinear1Out,
                &one,
                d_classifier_w_, kNumClasses,
                d_lstm_out_, kLinear1Out,
                &zero,
                d_output, kNumClasses);

    add_bias_kernel<<<kSeqLen, kNumClasses, 0, stream_>>>(
        d_output, d_classifier_b_, kSeqLen, kNumClasses);

    // LogSoftmax
    log_softmax_kernel<<<kSeqLen, 1, 0, stream_>>>(
        d_output, d_output, kSeqLen, kNumClasses);
}

// ============================================================
// Forward
// ============================================================

int PyannoteSeg3::forward(const float* d_pcm, float* d_output, int n_samples) {
    if (!initialized_) return 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream_);

    // SincNet encoder
    forward_sincnet(d_pcm, n_samples);

    // 4x BiLSTM
    forward_lstm();

    // Linear head + LogSoftmax
    forward_linear_head(d_output);

    cudaEventRecord(stop, stream_);
    cudaStreamSynchronize(stream_);
    cudaEventElapsedTime(&last_latency_ms_, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return kNumFrames;
}

} // namespace deusridet
