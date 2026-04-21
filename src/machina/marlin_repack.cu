/**
 * @file src/machina/marlin_repack.cu
 * @philosophical_role
 *   Weight-repacking companion to marlin.cu — converts GPTQ on-disk format to
 *   Marlin's in-GPU tile layout, and uploads the permutation tables needed by
 *   the main kernel.
 * @serves
 *   Machina GPTQ weight load. Split from marlin.cu under R1 800-line hard cap.
 */
// marlin_repack.cu — peer TU of marlin.cu
//
// Contains Section 5 (Weight repacking: GPTQ → Marlin format) and the
// upload_marlin_perm_tables helper. The fast-path Marlin kernel and the
// marlin_gemm{,_add} launchers remain in marlin.cu.

#include "marlin.h"
#include "model.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace deusridet {

// TU-local duplicate of marlin.cu's ceildiv (static, trivially inline-able).
static constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

// ============================================================================

// Precompute the Marlin weight permutation table (1024 entries).
// Matches the _get_perms() function from Python marlin/__init__.py
static std::vector<int> compute_marlin_perm() {
    std::vector<int> perm;
    perm.reserve(1024);
    for (int i = 0; i < 32; i++) {
        std::vector<int> perm1;
        int col = i / 4;
        for (int block = 0; block < 2; block++) {
            int rows[] = {2 * (i % 4), 2 * (i % 4) + 1,
                          2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1};
            for (int r = 0; r < 4; r++)
                perm1.push_back(16 * rows[r] + col + 8 * block);
        }
        for (int j = 0; j < 4; j++)
            for (auto p : perm1)
                perm.push_back(p + 256 * j);
    }
    // Apply interleave: [0,2,4,6,1,3,5,7]
    int interleave[] = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<int> result(1024);
    for (int i = 0; i < 1024 / 8; i++) {
        for (int j = 0; j < 8; j++)
            result[i * 8 + j] = perm[i * 8 + interleave[j]];
    }
    return result;
}

// Precompute scale permutation (64 entries for grouped quantization).
static std::vector<int> compute_scale_perm() {
    std::vector<int> perm;
    perm.reserve(64);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            perm.push_back(i + 8 * j);
    return perm;
}

// GPU kernel: repack GPTQ qweight to Marlin B format.
// Each thread produces one output uint32 (8 packed INT4 values).
__global__ void repack_qweight_kernel(
    const uint32_t* __restrict__ in,   // GPTQ qweight [K/8, N] (copy in temp)
    uint32_t* __restrict__ out,        // Marlin B [K/16, 2*N]
    const int* __restrict__ perm,      // [1024] Marlin permutation
    int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (K / 16) * (2 * N);
    if (idx >= total) return;

    int tr = idx / (2 * N);     // tile row [0, K/16)
    int pc = idx % (2 * N);     // packed column [0, 2*N)

    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col_perm_idx = 8 * pc + i;
        int block = col_perm_idx / 1024;
        int within = col_perm_idx % 1024;
        int col_tiled = block * 1024 + perm[within];

        int n_tile = col_tiled / 256;
        int k_within = (col_tiled % 256) / 16;
        int n_within = col_tiled % 16;

        int k = tr * 16 + k_within;
        int n = n_tile * 16 + n_within;

        // Read INT4 from GPTQ packed format
        int k_packed = k / 8;
        int k_bit = k % 8;
        uint32_t packed = in[k_packed * N + n];
        int val = (packed >> (k_bit * 4)) & 0xF;

        result |= ((uint32_t)val << (i * 4));
    }
    out[idx] = result;
}

// GPU kernel: permute scale columns within 64-element blocks.
__global__ void repack_scales_kernel(
    const __half* __restrict__ in,      // original scales [K/128, N] (copy in temp)
    __half* __restrict__ out,           // permuted scales [K/128, N]
    const int* __restrict__ scale_perm, // [64] scale permutation
    int num_groups, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_groups * N;
    if (idx >= total) return;

    int g = idx / N;
    int n = idx % N;
    int block = n / 64;
    int within = n % 64;
    int src_n = block * 64 + scale_perm[within];

    out[g * N + n] = in[g * N + src_n];
}

void repack_gptq_to_marlin(
    uint32_t* qweight, __half* scales,
    int K, int N,
    int* d_perm, int* d_scale_perm,
    void* temp_buf, size_t temp_buf_bytes,
    cudaStream_t stream)
{
    // --- Repack qweight ---
    int qw_size = (K / 8) * N;    // original size in uint32
    int B_size  = (K / 16) * 2 * N; // Marlin size in uint32 (same total bytes)
    size_t qw_bytes = (size_t)qw_size * sizeof(uint32_t);

    // Use pre-allocated temp buffer (caller ensures it's big enough)
    uint32_t* temp_qw = static_cast<uint32_t*>(temp_buf);
    cudaMemcpyAsync(temp_qw, qweight, qw_bytes,
                    cudaMemcpyDeviceToDevice, stream);

    int threads_per_block = 256;
    int blocks = ceildiv(B_size, threads_per_block);
    repack_qweight_kernel<<<blocks, threads_per_block, 0, stream>>>(
        temp_qw, qweight, d_perm, K, N);

    // --- Repack scales ---
    int num_groups = K / 128;
    int s_size = num_groups * N;
    size_t s_bytes = (size_t)s_size * sizeof(__half);

    // Reuse same temp buffer for scales (smaller than qweight)
    __half* temp_s = reinterpret_cast<__half*>(temp_buf);
    cudaMemcpyAsync(temp_s, scales, s_bytes,
                    cudaMemcpyDeviceToDevice, stream);

    int s_blocks = ceildiv(s_size, threads_per_block);
    repack_scales_kernel<<<s_blocks, threads_per_block, 0, stream>>>(
        temp_s, scales, d_scale_perm, num_groups, N);
}

// Upload Marlin permutation tables to device memory.
void upload_marlin_perm_tables(int** d_perm, int** d_scale_perm) {
    static std::vector<int> h_perm = compute_marlin_perm();
    static std::vector<int> h_scale_perm = compute_scale_perm();
    cudaMalloc(d_perm, 1024 * sizeof(int));
    cudaMemcpy(*d_perm, h_perm.data(), 1024 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(d_scale_perm, 64 * sizeof(int));
    cudaMemcpy(*d_scale_perm, h_scale_perm.data(), 64 * sizeof(int), cudaMemcpyHostToDevice);
}

// Repack all MLP weights in the model. Returns workspace bytes allocated.
size_t repack_all_marlin(ModelWeights& weights, cudaStream_t stream) {
    using MC = ModelConfig;
    printf("[marlin] Repacking GPTQ weights to Marlin format...\n");

    auto t0 = std::chrono::steady_clock::now();

    // Upload permutation tables to device (once)
    static std::vector<int> h_perm = compute_marlin_perm();
    static std::vector<int> h_scale_perm = compute_scale_perm();
    int* d_perm = nullptr;
    int* d_scale_perm = nullptr;
    cudaMalloc(&d_perm, 1024 * sizeof(int));
    cudaMemcpy(d_perm, h_perm.data(), 1024 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_scale_perm, 64 * sizeof(int));
    cudaMemcpy(d_scale_perm, h_scale_perm.data(), 64 * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate single temp buffer for largest tensor: max(gate/up, down) qweight
    // gate/up: (5120/8)*17408 = 11141120 uint32 = 42.5 MB
    // down:    (17408/8)*5120 = 11141120 uint32 = 42.5 MB (same!)
    size_t max_qw_bytes = (size_t)(MC::HIDDEN_SIZE / 8) * MC::INTERMEDIATE_SIZE * sizeof(uint32_t);
    void* temp_buf = nullptr;
    cudaMalloc(&temp_buf, max_qw_bytes);

    for (int li = 0; li < MC::NUM_LAYERS; li++) {
        auto& mlp = weights.layers[li].mlp;

        repack_gptq_to_marlin(
            const_cast<uint32_t*>(mlp.gate_proj.qweight),
            const_cast<__half*>(mlp.gate_proj.scales),
            mlp.gate_proj.K, mlp.gate_proj.N,
            d_perm, d_scale_perm, temp_buf, max_qw_bytes, stream);

        repack_gptq_to_marlin(
            const_cast<uint32_t*>(mlp.up_proj.qweight),
            const_cast<__half*>(mlp.up_proj.scales),
            mlp.up_proj.K, mlp.up_proj.N,
            d_perm, d_scale_perm, temp_buf, max_qw_bytes, stream);

        repack_gptq_to_marlin(
            const_cast<uint32_t*>(mlp.down_proj.qweight),
            const_cast<__half*>(mlp.down_proj.scales),
            mlp.down_proj.K, mlp.down_proj.N,
            d_perm, d_scale_perm, temp_buf, max_qw_bytes, stream);
    }

    cudaStreamSynchronize(stream);

    // Free temp resources (only used during repacking)
    cudaFree(temp_buf);
    cudaFree(d_perm);
    cudaFree(d_scale_perm);

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[marlin] Repacked %d GPTQ weights in %.1f ms (temp buf %.1f MB, freed)\n",
           MC::NUM_LAYERS * 3, ms, max_qw_bytes / 1048576.0);

    int max_N = MC::INTERMEDIATE_SIZE;
    size_t ws_bytes = marlin_workspace_size(max_N);
    return ws_bytes;
}


} // namespace deusridet
