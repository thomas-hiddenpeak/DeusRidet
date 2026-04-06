// commands.cpp — CLI command implementations
//
// All test, benchmark, and diagnostic commands live here.
// Each function is self-contained: load → run → cleanup → return.

#include "commands.h"
#include "communis/config.h"
#include "communis/log.h"
#include "communis/tegra.h"
#include "machina/gptq.h"
#include "machina/gptq_gemm_v2.h"
#include "machina/model.h"
#include "machina/forward.h"
#include "machina/allocator.h"
#include "machina/safetensors.h"
#include "machina/tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include <signal.h>
#include "nexus/ws_server.h"
#include "sensus/auditus/audio_pipeline.h"

namespace deusridet {

static const char* VERSION    = "0.1.0";
static const char* BUILD_DATE = __DATE__;

volatile sig_atomic_t g_shutdown_requested = 0;

// ============================================================================
// version / usage
// ============================================================================

void print_version() {
    printf("DeusRidet v%s (%s)\n", VERSION, BUILD_DATE);
    printf("  \"When humans think, God laughs; when AI thinks, humans stop laughing.\"\n\n");

    int driver_ver = 0, runtime_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    cudaRuntimeGetVersion(&runtime_ver);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("  Device:    %s\n", prop.name);
    printf("  SM:        %d.%d (%d SMs)\n", prop.major, prop.minor, prop.multiProcessorCount);
    printf("  Memory:    %.1f GB\n", prop.totalGlobalMem / 1073741824.0);
    printf("  CUDA:      Driver %d.%d, Runtime %d.%d\n",
           driver_ver / 1000, (driver_ver % 100) / 10,
           runtime_ver / 1000, (runtime_ver % 100) / 10);
    printf("  License:   GPLv3\n");
}

void print_usage() {
    printf("\n  DeusRidet v%s — Continuous Consciousness Engine\n\n", VERSION);
    printf("  Usage:\n");
    printf("    deusridet <command> [options]\n\n");
    printf("  Commands:\n");
    printf("    test-tokenizer <text>   Encode/decode round-trip test\n");
    printf("    test-weights            Load weights and print tensor summary\n");
    printf("    test-gptq               GPTQ kernel correctness test with model weights\n");
    printf("    bench-gptq              GPTQ GEMV/GEMM benchmark (Marlin)\n");
    printf("    bench-gptq-v2           GPTQ v2 kernel benchmark (vs Marlin)\n");
    printf("    load-model              Load all weights to device, hold for inspection\n");
    printf("    load-weights            Structured weight load (model.h) with validation\n");
    printf("    test-forward            Single-token forward pass test\n");
    printf("    test-sample             Sampling test (greedy + top-k/p)\n");
    printf("    profile-forward         Profile single-token forward pass timing\n");
    printf("    profile-prefill         Profile prefill pass (Marlin MLP) at various M\n");
    printf("    profile-prefill-gptq-v2 (legacy alias for profile-prefill)\n");
    printf("    bench-prefill           Benchmark Marlin vs cuBLAS FP16 projections\n");
    printf("    test-ws                 Start WebSocket server + serve WebUI\n");
    printf("    version                 Print version and hardware info\n\n");
    printf("  Options:\n");
    printf("    --config <file>         Configuration file (default: configs/machina.conf)\n");
    printf("    --model-dir <path>      Override LLM model directory\n\n");
}

// ============================================================================
// test-tokenizer
// ============================================================================

int cmd_test_tokenizer(const std::string& model_dir, const std::string& text) {
    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        LOG_ERROR("Main", "Failed to load tokenizer from %s", model_dir.c_str());
        return 1;
    }

    printf("[Tokenizer] vocab_size = %d\n", tokenizer.vocab_size());
    printf("[Tokenizer] eos_id = %d, im_start_id = %d, im_end_id = %d\n",
           tokenizer.eos_token_id(), tokenizer.im_start_id(), tokenizer.im_end_id());

    auto ids = tokenizer.encode(text);
    printf("\n[Encode] \"%s\"\n  → %zu tokens: [", text.c_str(), ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        printf("%d%s", ids[i], i + 1 < ids.size() ? ", " : "");
    }
    printf("]\n");

    std::string decoded = tokenizer.decode(ids);
    printf("\n[Decode] → \"%s\"\n", decoded.c_str());

    bool match = (decoded == text);
    printf("\n[Round-trip] %s\n", match ? "PASS ✓" : "MISMATCH ✗");

    // Chat template test
    std::vector<std::pair<std::string, std::string>> messages = {
        {"system", "You are a helpful assistant."},
        {"user",   text}
    };
    auto chat_ids = tokenizer.apply_chat_template(messages);
    printf("\n[ChatML] %zu tokens\n", chat_ids.size());

    return match ? 0 : 1;
}

// ============================================================================
// test-weights
// ============================================================================

int cmd_test_weights(const std::string& model_dir) {
    LOG_INFO("Main", "Loading weights from %s ...", model_dir.c_str());

    SafetensorsLoader loader(model_dir);
    auto names = loader.tensor_names();

    printf("\n[Weights] %zu tensors across %zu shards\n\n", names.size(), loader.shard_count());

    // Sort names for readable output
    std::sort(names.begin(), names.end());

    size_t total_bytes = 0;
    int shown = 0;
    for (const auto& name : names) {
        auto tensor = loader.get_tensor(name);
        size_t nb = tensor->nbytes();
        total_bytes += nb;

        // Print first 20 and last 5 for brevity
        if (shown < 20 || names.size() - shown <= 5) {
            printf("  %-60s  %6s  [", name.c_str(),
                   dtype_name(tensor->dtype()));
            for (size_t i = 0; i < tensor->shape().size(); i++) {
                printf("%lld%s", (long long)tensor->shape()[i],
                       i + 1 < tensor->shape().size() ? ", " : "");
            }
            printf("]  %.2f MB\n", nb / 1048576.0);
        } else if (shown == 20) {
            printf("  ... (%zu more tensors) ...\n", names.size() - 25);
        }
        shown++;
    }

    printf("\n[Total] %.2f GB across %zu tensors\n", total_bytes / 1073741824.0, names.size());
    return 0;
}

// ============================================================================
// test-gptq: Correctness test using actual model weights
// ============================================================================

int cmd_test_gptq(const std::string& model_dir) {
    LOG_INFO("Main", "Loading model weights for GPTQ correctness test...");

    SafetensorsLoader loader(model_dir);

    // Test with layer 0 MLP gate_proj (a quantized layer)
    const char* layer_name = "model.language_model.layers.0.mlp.gate_proj";
    std::string qw_name = std::string(layer_name) + ".qweight";
    std::string sc_name = std::string(layer_name) + ".scales";

    if (!loader.has_tensor(qw_name)) {
        LOG_ERROR("Main", "Tensor not found: %s", qw_name.c_str());
        return 1;
    }

    auto qw_tensor = loader.get_tensor(qw_name);
    auto sc_tensor = loader.get_tensor(sc_name);

    int packed_K = (int)qw_tensor->shape()[0];
    int N        = (int)qw_tensor->shape()[1];
    int K        = packed_K * 8;
    int num_groups = (int)sc_tensor->shape()[0];

    printf("[GPTQ Test] Layer: %s\n", layer_name);
    printf("  qweight: [%d, %d] (K=%d, N=%d)\n", packed_K, N, K, N);
    printf("  scales:  [%d, %d]\n", num_groups, N);
    printf("  group_size: %d, bits: 4, sym: true\n\n", K / num_groups);

    // Allocate device memory for x and y
    static DeviceAllocator dev_alloc;
    size_t x_bytes = (size_t)K * sizeof(__half);
    size_t y_bytes = (size_t)N * sizeof(__half);

    __half* d_x = (__half*)dev_alloc.allocate(x_bytes);
    __half* d_y = (__half*)dev_alloc.allocate(y_bytes);

    // Fill x with small values on host, copy to device
    __half* h_x = (__half*)malloc(x_bytes);
    srand(42);
    for (int i = 0; i < K; i++) {
        h_x[i] = __float2half(((float)(rand() % 1000) - 500.0f) / 5000.0f);
    }
    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);

    // Copy GPTQ weights to device memory.
    // On Tegra, mmap'd files can't be registered via cudaHostRegister with
    // PROT_READ-only mappings, so we copy to device memory which also avoids
    // coherency overhead for frequently-read weight data.
    size_t qw_bytes = qw_tensor->nbytes();
    size_t sc_bytes = sc_tensor->nbytes();

    printf("[GPTQ Test] Copying weights to device: qweight %.1f MB, scales %.1f MB\n",
           qw_bytes / 1048576.0, sc_bytes / 1048576.0);

    uint32_t* d_qw;
    __half*   d_sc;

    d_qw = (uint32_t*)dev_alloc.allocate(qw_bytes);
    d_sc = (__half*)dev_alloc.allocate(sc_bytes);
    cudaMemcpy(d_qw, qw_tensor->data(), qw_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, sc_tensor->data(), sc_bytes, cudaMemcpyHostToDevice);

    // Keep host copies for CPU reference check before releasing mmap
    uint32_t* h_qw = (uint32_t*)malloc(qw_bytes);
    __half*   h_sc = (__half*)malloc(sc_bytes);
    memcpy(h_qw, qw_tensor->data(), qw_bytes);
    memcpy(h_sc, sc_tensor->data(), sc_bytes);

    // Release mmap — weights are now in device memory, free physical pages
    // for GPU use. On Tegra unified memory this is critical to avoid
    // double-occupancy (mmap pages + cudaMalloc pages).
    qw_tensor.reset();
    sc_tensor.reset();
    loader.for_each_shard([&](size_t idx, SafetensorsFile&) {
        loader.release_shard(idx);
    });

    printf("[GPTQ Test] Device memory allocated: %.1f MB\n",
           DeviceAllocator::total_allocated() / 1048576.0);

    GptqWeight weight;
    weight.qweight = d_qw;
    weight.scales  = d_sc;
    weight.K       = K;
    weight.N       = N;

    // Run GEMV
    printf("[GPTQ Test] Running GEMV (M=1, K=%d, N=%d)...\n", K, N);
    gptq_gemv(d_x, weight, d_y);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("Main", "CUDA error: %s", cudaGetErrorString(err));
        dev_alloc.deallocate(d_x); dev_alloc.deallocate(d_y);
        dev_alloc.deallocate(d_qw); dev_alloc.deallocate(d_sc);
        free(h_x);
        return 1;
    }

    // Copy result back
    __half* h_y = (__half*)malloc(y_bytes);
    cudaMemcpy(h_y, d_y, y_bytes, cudaMemcpyDeviceToHost);

    // CPU reference (partial — first 256 columns for speed)
    int check_N = (N < 256) ? N : 256;
    printf("[GPTQ Test] CPU reference check (first %d columns)...\n", check_N);

    float max_err = 0.0f;
    float max_abs_err = 0.0f;
    int max_err_col = 0;
    for (int n = 0; n < check_N; n++) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            int pk = k / 8;
            int ki = k % 8;
            uint32_t packed = h_qw[pk * N + n];
            int q_val = (packed >> (ki * 4)) & 0xF;
            int group = k / 128;
            float s = __half2float(h_sc[group * N + n]);
            float w = s * (float)(q_val - 8);
            sum += (double)__half2float(h_x[k]) * (double)w;
        }
        float gpu_val = __half2float(h_y[n]);
        float ref_val = (float)sum;
        float abs_err = fabsf(gpu_val - ref_val);
        float rel = (fabsf(ref_val) > 1e-6f) ? abs_err / fabsf(ref_val) : abs_err;
        if (rel > max_err) {
            max_err = rel;
            max_err_col = n;
        }
        max_abs_err = fmaxf(max_abs_err, abs_err);
    }

    printf("\n[GPTQ Test] Max relative error: %.6f (column %d)\n", max_err, max_err_col);
    printf("[GPTQ Test] Max absolute error: %.6f\n", max_abs_err);
    // FP16 accumulation across K=5120 elements introduces ~1-3% relative error
    // in worst case. This is expected for half-precision arithmetic.
    bool pass = (max_err < 0.05f);
    printf("[GPTQ Test] %s\n", pass ? "PASS ✓" : "FAIL ✗");

    // Print a few output values
    printf("\n[GPTQ Test] First 8 output values:\n  ");
    for (int i = 0; i < 8 && i < N; i++) {
        printf("%.4f ", __half2float(h_y[i]));
    }
    printf("\n");

    // Also test GEMM with M=4
    int M_test = 4;
    size_t xm_bytes = (size_t)M_test * K * sizeof(__half);
    size_t ym_bytes = (size_t)M_test * N * sizeof(__half);
    __half* d_xm;
    __half* d_ym;
    d_xm = (__half*)dev_alloc.allocate(xm_bytes);
    d_ym = (__half*)dev_alloc.allocate(ym_bytes);

    __half* h_xm = (__half*)malloc(xm_bytes);
    for (int i = 0; i < M_test * K; i++) {
        h_xm[i] = __float2half(((float)(rand() % 1000) - 500.0f) / 5000.0f);
    }
    cudaMemcpy(d_xm, h_xm, xm_bytes, cudaMemcpyHostToDevice);

    printf("\n[GPTQ Test] Running GEMM (M=%d, K=%d, N=%d)...\n", M_test, K, N);
    gptq_gemm(d_xm, weight, d_ym, M_test);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("Main", "CUDA error in GEMM: %s", cudaGetErrorString(err));
        pass = false;
    } else {
        __half* h_ym = (__half*)malloc(ym_bytes);
        cudaMemcpy(h_ym, d_ym, ym_bytes, cudaMemcpyDeviceToHost);

        // Check first row against CPU reference
        float max_err_gemm = 0.0f;
        for (int n = 0; n < check_N; n++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                int pk = k / 8;
                int ki = k % 8;
                uint32_t packed = h_qw[pk * N + n];
                int q_val = (packed >> (ki * 4)) & 0xF;
                int group = k / 128;
                float s = __half2float(h_sc[group * N + n]);
                float w = s * (float)(q_val - 8);
                sum += (double)__half2float(h_xm[k]) * (double)w;
            }
            float gpu_val = __half2float(h_ym[n]);
            float ref_val = (float)sum;
            float err2 = fabsf(gpu_val - ref_val);
            float rel = (fabsf(ref_val) > 1e-6f) ? err2 / fabsf(ref_val) : err2;
            max_err_gemm = fmaxf(max_err_gemm, rel);
        }

        printf("[GPTQ Test] GEMM max relative error (row 0): %.6f\n", max_err_gemm);
        bool gemm_pass = (max_err_gemm < 0.05f);
        printf("[GPTQ Test] GEMM %s\n", gemm_pass ? "PASS ✓" : "FAIL ✗");
        pass = pass && gemm_pass;

        free(h_ym);
    }

    dev_alloc.deallocate(d_x); dev_alloc.deallocate(d_y);
    dev_alloc.deallocate(d_xm); dev_alloc.deallocate(d_ym);
    dev_alloc.deallocate(d_qw); dev_alloc.deallocate(d_sc);
    free(h_x); free(h_y); free(h_xm);
    free(h_qw); free(h_sc);

    return pass ? 0 : 1;
}

// ============================================================================
// load-model: Load all weights to device memory and hold for inspection
// ============================================================================

int cmd_load_model(const std::string& model_dir) {
    using Clock = std::chrono::steady_clock;

    // Drop page caches for clean baseline measurement
    drop_page_caches();

    // Force CUDA context init
    cudaFree(0);
    size_t avail_baseline = read_memavail_kb() / 1024;
    printf("[Load Model] Baseline (after drop_caches + CUDA init): MemAvail=%zuMB\n", avail_baseline);

    printf("[Load Model] Loading from %s\n", model_dir.c_str());

    auto t0 = Clock::now();

    ModelWeights weights;
    bool ok = load_model_weights(model_dir, weights);
    if (!ok) {
        fprintf(stderr, "[Load Model] Weight loading failed\n");
        return 1;
    }

    double total_sec = std::chrono::duration<double>(Clock::now() - t0).count();

    printf("\n[Load Model] Done: %.2f GB in %d pool blocks, %.1fs (%.0f MB/s)\n",
           weights.total_bytes / 1073741824.0,
           (int)weights.pool_blocks.size(),
           total_sec,
           (weights.total_bytes / 1048576.0) / total_sec);

    // Memory breakdown: clean comparison against drop_caches baseline
    {
        size_t vm_rss_kb = 0, rss_anon_kb = 0, rss_file_kb = 0;
        FILE* st = fopen("/proc/self/status", "r");
        if (st) {
            char line[256];
            while (fgets(line, sizeof(line), st)) {
                if (strncmp(line, "VmRSS:", 6) == 0) sscanf(line+6, " %zu", &vm_rss_kb);
                else if (strncmp(line, "RssAnon:", 8) == 0) sscanf(line+8, " %zu", &rss_anon_kb);
                else if (strncmp(line, "RssFile:", 8) == 0) sscanf(line+8, " %zu", &rss_file_kb);
            }
            fclose(st);
        }
        size_t cuda_free = 0, cuda_total = 0;
        cudaMemGetInfo(&cuda_free, &cuda_total);
        size_t avail_now = read_memavail_kb() / 1024;
        size_t consumed = (avail_baseline > avail_now) ? (avail_baseline - avail_now) : 0;

        printf("\n[Memory Breakdown]\n");
        printf("  Pool blocks (cudaMalloc): %8d\n", (int)weights.pool_blocks.size());
        printf("  cudaMemGetInfo free:      %8.1f MB / %.1f MB\n",
               cuda_free / 1048576.0, cuda_total / 1048576.0);
        printf("  Process VmRSS:            %8.1f MB\n", vm_rss_kb / 1024.0);
        printf("    RssAnon  (heap+GPU):    %8.1f MB\n", rss_anon_kb / 1024.0);
        printf("    RssFile  (file-backed): %8.1f MB\n", rss_file_kb / 1024.0);
        printf("  MemAvail baseline:        %8zu MB  (after drop_caches + CUDA init)\n", avail_baseline);
        printf("  MemAvail now:             %8zu MB\n", avail_now);
        printf("  System RAM consumed:      %8zu MB  (baseline - now)\n", consumed);
        printf("  Weight data loaded:       %8.1f MB\n", weights.total_bytes / 1048576.0);
        printf("  Overhead (consumed-load): %8ld MB\n",
               (long)consumed - (long)(weights.total_bytes / 1048576));
    }

    printf("\n[Load Model] Weights held in %d pool blocks. Press Enter to release...\n",
           (int)weights.pool_blocks.size());
    fflush(stdout);
    getchar();

    free_model_weights(weights);
    drop_page_caches();

    size_t avail_after = read_memavail_kb() / 1024;
    printf("[Load Model] Released. MemAvail=%zuMB (recovered ~%ldMB vs baseline)\n",
           avail_after, (long)avail_after - (long)avail_baseline);

    return 0;
}

// ============================================================================
// load-weights: Load all model weights into device memory (structured)
// ============================================================================

int cmd_load_weights(const std::string& model_dir) {
    // Drop page caches for clean baseline
    drop_page_caches();
    cudaFree(0);

    ModelWeights weights;
    bool ok = load_model_weights(model_dir, weights);

    if (ok) {
        printf("\n[load-weights] Summary:\n");
        printf("  Total device memory: %.2f GB\n", weights.total_bytes / 1073741824.0);
        printf("  Full attention layers:");
        for (int i = 0; i < ModelConfig::NUM_LAYERS; i++) {
            if (weights.layers[i].is_full_attention) printf(" %d", i);
        }
        printf("\n");

        // Spot-check: print first layer MLP dimensions
        auto& mlp0 = weights.layers[0].mlp;
        printf("  Layer 0 MLP: gate[K=%d,N=%d] up[K=%d,N=%d] down[K=%d,N=%d]\n",
               mlp0.gate_proj.K, mlp0.gate_proj.N,
               mlp0.up_proj.K, mlp0.up_proj.N,
               mlp0.down_proj.K, mlp0.down_proj.N);

        // Spot-check: layer 0 DeltaNet dims
        auto& dn0 = weights.layers[0].delta_net;
        printf("  Layer 0 DeltaNet: qkv[%d→%d] z[%d→%d] out[%d→%d]\n",
               dn0.fp16_qkv.in_features, dn0.fp16_qkv.out_features,
               dn0.fp16_z.in_features, dn0.fp16_z.out_features,
               dn0.fp16_out.in_features, dn0.fp16_out.out_features);

        // Spot-check: layer 3 Full Attention dims
        auto& fa3 = weights.layers[3].full_attn;
        printf("  Layer 3 FullAttn: q[%d→%d] k[%d→%d] v[%d→%d] o[%d→%d]\n",
               fa3.fp16_q.in_features, fa3.fp16_q.out_features,
               fa3.fp16_k.in_features, fa3.fp16_k.out_features,
               fa3.fp16_v.in_features, fa3.fp16_v.out_features,
               fa3.fp16_o.in_features, fa3.fp16_o.out_features);
    }

    printf("\n[load-weights] Press Enter to release...\n");
    fflush(stdout);
    getchar();

    free_model_weights(weights);

    // Reclaim CMA pages
    drop_page_caches();

    printf("[load-weights] Released.\n");
    return ok ? 0 : 1;
}

// ============================================================================
// test-forward: Load model, run single-token forward pass, verify output
// ============================================================================

int cmd_test_forward(const std::string& model_dir) {
    using MC = ModelConfig;

    printf("[test-forward] Loading model weights...\n");

    // Drop caches
    drop_page_caches();
    cudaFree(0);

    // Load tokenizer
    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        fprintf(stderr, "[test-forward] Tokenizer load failed\n");
        return 1;
    }

    // Load weights
    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[test-forward] Weight load failed\n");
        return 1;
    }
    merge_projection_weights(weights);
    printf("[test-forward] Weights loaded: %.2f GB\n",
           weights.total_bytes / 1073741824.0);

    // Allocate inference state
    int max_seq = 128;  // Enough for chat template + generation
    InferenceState state;
    if (!state.allocate(max_seq)) {
        fprintf(stderr, "[test-forward] State allocation failed\n");
        free_model_weights(weights);
        return 1;
    }

    // Allocate KV cache for full attention layers
    // Layout: [num_full_attn_layers * 2 * num_kv_heads * max_kv_len * head_dim]
    int num_full_attn = 0;
    for (int i = 0; i < MC::NUM_LAYERS; i++)
        if (MC::is_full_attention(i)) num_full_attn++;

    int max_kv_len = max_seq;
    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    size_t kv_bytes = (size_t)MC::NUM_LAYERS * 2 * kv_plane * sizeof(__half);
    __half* kv_cache = nullptr;
    cudaMalloc(&kv_cache, kv_bytes);
    cudaMemset(kv_cache, 0, kv_bytes);
    printf("[test-forward] KV cache: %.1f MB (max_kv=%d, %d full-attn layers)\n",
           kv_bytes / 1048576.0, max_kv_len, num_full_attn);

    // Encode prompt using ChatML template for proper model behavior
    std::string prompt = "Hello";
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}
    };
    auto tokens = tokenizer.apply_chat_template(messages);
    printf("[test-forward] Prompt: \"%s\" → %zu tokens:", prompt.c_str(), tokens.size());
    for (int t : tokens) printf(" %d", t);
    printf("\n");

    if (tokens.empty()) {
        fprintf(stderr, "[test-forward] Tokenizer returned empty tokens\n");
        cudaFree(kv_cache);
        state.free();
        free_model_weights(weights);
        return 1;
    }

    // Generate tokens
    int max_gen = 16;
    printf("[test-forward] Generating %d tokens...\n", max_gen);

    std::vector<int> generated;

    // Process prompt tokens one by one (prefill)
    int pos = 0;
    int next_token = tokens[0];

    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    // Batched prefill: process all prompt tokens in one pass (GEMM instead of 11× GEMV)
    next_token = forward_prefill(weights, state, kv_cache,
                                 tokens.data(), (int)tokens.size(),
                                 pos, max_kv_len);
    pos += (int)tokens.size();
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    generated.push_back(next_token);
    printf("[test-forward] Prefill: %zu tokens in %.1f ms (%.1f ms/token)\n",
           tokens.size(), prefill_ms, prefill_ms / tokens.size());
    printf("[test-forward] First generated token: %d = \"%s\"\n",
           next_token, tokenizer.decode(next_token).c_str());

    // Generate remaining tokens (decode)
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    for (int g = 1; g < max_gen; g++) {
        if (g_shutdown_requested) break;
        if (pos >= max_kv_len - 1) break;  // Safety limit
        next_token = forward_one_token(weights, state, kv_cache,
                                       next_token, pos, max_kv_len);
        pos++;
        generated.push_back(next_token);

        // Check for EOS (token 151643 or 151645)
        if (next_token == 151643 || next_token == 151645) break;
    }
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    int decode_count = (int)generated.size() - 1;  // exclude first token (from prefill)

    // Print results
    double total_ms = prefill_ms + decode_ms;
    printf("\n[test-forward] Decode: %d tokens in %.1f ms (%.1f ms/token)\n",
           decode_count, decode_ms, decode_count > 0 ? decode_ms / decode_count : 0.0);
    printf("[test-forward] Total: %.1f ms  (prefill %.1f + decode %.1f)\n",
           total_ms, prefill_ms, decode_ms);
    printf("[test-forward] Output: ");
    for (int t : generated) {
        printf("%s", tokenizer.decode(t).c_str());
    }
    printf("\n");
    printf("[test-forward] Token IDs:");
    for (int t : generated) printf(" %d", t);
    printf("\n");

    // Cleanup
    cudaFree(kv_cache);
    state.free();
    free_model_weights(weights);
    drop_page_caches();

    printf("[test-forward] Done.\n");
    return 0;
}

// ============================================================================
// test-sample: Test generation with top-k/top-p sampling
// ============================================================================

int cmd_test_sample(const std::string& model_dir) {
    using MC = ModelConfig;

    printf("[test-sample] Loading model weights...\n");
    drop_page_caches();
    cudaFree(0);

    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        fprintf(stderr, "[test-sample] Tokenizer load failed\n");
        return 1;
    }

    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[test-sample] Weight load failed\n");
        return 1;
    }
    merge_projection_weights(weights);

    InferenceState state;
    int max_kv_len = 128;
    if (!state.allocate(max_kv_len)) {
        fprintf(stderr, "[test-sample] State allocation failed\n");
        return 1;
    }

    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    size_t kv_bytes = (size_t)MC::NUM_LAYERS * 2 * kv_plane * sizeof(__half);
    __half* kv_cache = nullptr;
    cudaMalloc(&kv_cache, kv_bytes);
    cudaMemset(kv_cache, 0, kv_bytes);

    std::string prompt = "Hello";
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}
    };
    auto tokens = tokenizer.apply_chat_template(messages);
    printf("[test-sample] Prompt: \"%s\" → %zu tokens\n", prompt.c_str(), tokens.size());

    if (tokens.empty()) {
        cudaFree(kv_cache);
        state.free();
        free_model_weights(weights);
        return 1;
    }

    SamplingParams params;
    params.temperature = 0.7f;
    params.top_k = 50;
    params.top_p = 0.9f;

    int max_gen = 32;
    printf("[test-sample] Sampling: temp=%.1f, top_k=%d, top_p=%.2f, max_gen=%d\n",
           params.temperature, params.top_k, params.top_p, max_gen);

    std::vector<int> generated;
    int pos = 0;
    int next_token = tokens[0];

    // Prefill (batched)
    auto t_start = std::chrono::high_resolution_clock::now();
    next_token = forward_prefill(weights, state, kv_cache,
                                 tokens.data(), (int)tokens.size(),
                                 pos, max_kv_len);
    pos += (int)tokens.size();
    generated.push_back(next_token);

    // Decode
    for (int g = 1; g < max_gen; g++) {
        if (g_shutdown_requested) break;
        if (pos >= max_kv_len - 1) break;
        next_token = forward_one_token_sampled(weights, state, kv_cache,
                                                next_token, pos, max_kv_len, params);
        pos++;
        generated.push_back(next_token);
        if (next_token == 151643 || next_token == 151645) break;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    printf("[test-sample] Generated %zu tokens in %.1f ms (%.1f ms/token)\n",
           generated.size(), total_ms, total_ms / generated.size());
    printf("[test-sample] Output: ");
    for (int t : generated) printf("%s", tokenizer.decode(t).c_str());
    printf("\n");
    printf("[test-sample] Token IDs:");
    for (int t : generated) printf(" %d", t);
    printf("\n");

    cudaFree(kv_cache);
    state.free();
    free_model_weights(weights);
    drop_page_caches();

    printf("[test-sample] Done.\n");
    return 0;
}

// ============================================================================
// profile-forward: Per-component timing breakdown of one forward pass
// ============================================================================

int cmd_profile_forward(const std::string& model_dir) {
    using MC = ModelConfig;
    printf("[profile-forward] Loading model...\n");

    drop_page_caches();
    cudaFree(0);

    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        fprintf(stderr, "[profile-forward] Tokenizer load failed\n");
        return 1;
    }

    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[profile-forward] Weight load failed\n");
        return 1;
    }

    int max_seq = 32;
    InferenceState state;
    if (!state.allocate(max_seq)) {
        fprintf(stderr, "[profile-forward] State alloc failed\n");
        free_model_weights(weights);
        return 1;
    }

    int max_kv_len = max_seq;
    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    size_t kv_bytes = (size_t)MC::NUM_LAYERS * 2 * kv_plane * sizeof(__half);
    __half* kv_cache = nullptr;
    cudaMalloc(&kv_cache, kv_bytes);
    cudaMemset(kv_cache, 0, kv_bytes);

    // Use a simple token for profiling
    int token_id = 9419;  // "Hello"
    printf("[profile-forward] Profiling single-token decode...\n");

    profile_forward(weights, state, kv_cache, token_id, 0, max_kv_len);

    cudaFree(kv_cache);
    state.free();
    free_model_weights(weights);
    drop_page_caches();
    return 0;
}

// ============================================================================
// bench-gptq: GPTQ GEMV/GEMM benchmark with synthetic data
// ============================================================================

int cmd_bench_gptq() {
    printf("[GPTQ Benchmark] — SM87 Jetson AGX Orin\n");
    printf("  GPTQ: bits=4, group_size=128, sym=true\n\n");

    // Dimensions from Qwen3.5-27B model
    struct BenchCase {
        const char* name;
        int K, N, M;
        int warmup, iters;
    };

    BenchCase cases[] = {
        // Decode (M=1)
        {"gate_proj GEMV  (5120→17408)",          5120, 17408,   1, 10, 50},
        {"down_proj GEMV  (17408→5120)",          17408,  5120,   1, 10, 50},
        // Prefill (various M)
        {"gate_proj GEMM M=32  (5120→17408)",     5120, 17408,  32,  5, 20},
        {"gate_proj GEMM M=128 (5120→17408)",     5120, 17408, 128,  3, 10},
        {"gate_proj GEMM M=512 (5120→17408)",     5120, 17408, 512,  2,  5},
        {"down_proj GEMM M=128 (17408→5120)",    17408,  5120, 128,  3, 10},
    };

    int num_cases = sizeof(cases) / sizeof(cases[0]);
    bool all_correct = true;

    printf("  %-45s %10s %10s %10s %8s\n",
           "Case", "Time(us)", "BW(GB/s)", "TFLOPS", "Correct");
    printf("  %s\n", std::string(90, '-').c_str());

    for (int c = 0; c < num_cases; c++) {
        auto& bc = cases[c];
        auto r = gptq_benchmark(bc.K, bc.N, bc.M, bc.warmup, bc.iters);

        if (bc.M == 1) {
            printf("  %-45s %10.1f %10.1f %10s %8s\n",
                   bc.name, r.gemv_us, r.gemv_gbps, "—",
                   r.correct ? "✓" : "✗");
        } else {
            printf("  %-45s %10.1f %10s %10.3f %8s\n",
                   bc.name, r.gemm_us, "—", r.gemm_tflops,
                   r.correct ? "✓" : "✗");
        }

        if (!r.correct) all_correct = false;
    }

    printf("\n  %s\n", all_correct ? "All correctness checks PASSED ✓" :
                                      "Some correctness checks FAILED ✗");
    return all_correct ? 0 : 1;
}

// ============================================================================
// bench-gptq-v2: Marlin-format custom GPTQ v2 kernel benchmark
// ============================================================================

int cmd_bench_gptq_v2() {
    bench_gptq_v2_kernels();
    return 0;
}

// ============================================================================
// profile-prefill: Load model, run profiled prefill pass with event timestamps
// ============================================================================

int cmd_profile_prefill(const std::string& model_dir) {
    using MC = ModelConfig;

    printf("[profile-prefill] Loading model weights...\n");
    drop_page_caches();
    cudaFree(0);

    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        fprintf(stderr, "[profile-prefill] Tokenizer load failed\n");
        return 1;
    }

    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[profile-prefill] Weight load failed\n");
        return 1;
    }
    merge_projection_weights(weights);

    // M values to test — small to large
    const int M_vals[] = {11, 32, 64, 128, 256, 512};  // consciousness frame sizes
    const int N_M = sizeof(M_vals) / sizeof(M_vals[0]);
    int max_M = M_vals[N_M - 1];

    InferenceState state;
    int max_kv_len = max_M;
    if (!state.allocate(max_kv_len)) {
        fprintf(stderr, "[profile-prefill] State allocation failed\n");
        free_model_weights(weights);
        return 1;
    }

    size_t kv_plane = (size_t)MC::NUM_KV_HEADS * max_kv_len * MC::HEAD_DIM;
    size_t kv_bytes = (size_t)MC::NUM_LAYERS * 2 * kv_plane * sizeof(__half);
    __half* kv_cache = nullptr;
    cudaMalloc(&kv_cache, kv_bytes);

    // Build token arrays: use chat-templated "Hello" then pad with token 198
    // (newline) to reach target M. Token values don't affect kernel timing.
    std::string prompt = "Hello";
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}
    };
    auto base_tokens = tokenizer.apply_chat_template(messages);
    printf("[profile-prefill] Base prompt: \"%s\" → %zu tokens\n",
           prompt.c_str(), base_tokens.size());

    for (int mi = 0; mi < N_M; mi++) {
        int M = M_vals[mi];

        // Build token vector of exactly M tokens
        std::vector<int> tokens(M);
        for (int i = 0; i < M; i++)
            tokens[i] = (i < (int)base_tokens.size()) ? base_tokens[i] : 198;

        // Reset all state for clean measurement
        cudaMemset(kv_cache, 0, kv_bytes);
        for (int i = 0; i < state.num_dn_layers; i++) {
            size_t state_bytes = (size_t)MC::LIN_NUM_V_HEADS * MC::LIN_K_HEAD_DIM
                               * MC::LIN_V_HEAD_DIM * sizeof(float);
            cudaMemset(state.dn_states[i], 0, state_bytes);
            size_t conv_bytes = (size_t)MC::LIN_CONV_DIM * (MC::CONV_KERNEL - 1) * sizeof(__half);
            cudaMemset(state.conv_states[i], 0, conv_bytes);
        }

        // Warmup
        forward_prefill(weights, state, kv_cache,
                        tokens.data(), M,
                        0, max_kv_len);

        // Reset again for profiled run
        cudaMemset(kv_cache, 0, kv_bytes);
        for (int i = 0; i < state.num_dn_layers; i++) {
            size_t state_bytes = (size_t)MC::LIN_NUM_V_HEADS * MC::LIN_K_HEAD_DIM
                               * MC::LIN_V_HEAD_DIM * sizeof(float);
            cudaMemset(state.dn_states[i], 0, state_bytes);
            size_t conv_bytes = (size_t)MC::LIN_CONV_DIM * (MC::CONV_KERNEL - 1) * sizeof(__half);
            cudaMemset(state.conv_states[i], 0, conv_bytes);
        }

        // Profiled run
        profile_forward_prefill(weights, state, kv_cache,
                                tokens.data(), M,
                                0, max_kv_len);

        // Sub-layer profiling
        for (int i = 0; i < state.num_dn_layers; i++) {
            size_t state_bytes = (size_t)MC::LIN_NUM_V_HEADS * MC::LIN_K_HEAD_DIM
                               * MC::LIN_V_HEAD_DIM * sizeof(float);
            cudaMemset(state.dn_states[i], 0, state_bytes);
            size_t conv_bytes = (size_t)MC::LIN_CONV_DIM * (MC::CONV_KERNEL - 1) * sizeof(__half);
            cudaMemset(state.conv_states[i], 0, conv_bytes);
        }
        profile_sublayer_prefill(weights, state, kv_cache,
                                 M, 0, max_kv_len);
    }

    cudaFree(kv_cache);
    state.free();
    free_model_weights(weights);
    drop_page_caches();

    printf("[profile-prefill] Done.\n");
    return 0;
}

// ============================================================================
// bench-prefill: Benchmark Marlin INT4 vs cuBLAS FP16 projections at various M
// ============================================================================

int cmd_bench_prefill(const std::string& model_dir) {
    using MC = ModelConfig;

    printf("[bench-prefill] Loading model weights...\n");
    drop_page_caches();
    cudaFree(0);

    ModelWeights weights;
    if (!load_model_weights(model_dir, weights)) {
        fprintf(stderr, "[bench-prefill] Weight load failed\n");
        return 1;
    }
    merge_projection_weights(weights);

    // Allocate state with large max_seq to support up to M=2048
    InferenceState state;
    int max_seq = 2048;
    if (!state.allocate(max_seq)) {
        fprintf(stderr, "[bench-prefill] State allocation failed (max_seq=%d)\n", max_seq);
        free_model_weights(weights);
        return 1;
    }

    printf("[bench-prefill] State allocated (max_seq=%d)\n", max_seq);

    bench_prefill_projections(weights, state);

    state.free();
    free_model_weights(weights);
    drop_page_caches();

    printf("[bench-prefill] Done.\n");
    return 0;
}

// ============================================================================
// test-ws: WebSocket server + audio pipeline (Ring → Mel → VAD) with WebUI.
// ============================================================================

int cmd_test_ws(const std::string& webui_dir) {
    printf("[test-ws] Starting WebSocket + Audio Pipeline...\n");
    printf("[test-ws] WebUI dir: %s\n", webui_dir.c_str());

    WsServer server;
    WsServerConfig ws_cfg;
    ws_cfg.port = 8080;
    ws_cfg.static_dir = webui_dir;

    // Audio pipeline.
    AudioPipeline audio;
    AudioPipelineConfig audio_cfg;
    // defaults: n_fft=400, hop=160, n_mels=128, sr=16000

    // Configure Silero VAD model path.
    audio_cfg.silero.model_path = std::string(getenv("HOME") ? getenv("HOME") : "/home/rm01")
                                  + "/models/dev/vad/silero_vad.onnx";

    // Configure FSMN VAD model paths.
    std::string home = getenv("HOME") ? getenv("HOME") : "/home/rm01";
    audio_cfg.fsmn.model_path = home + "/models/dev/vad/fsmn/model_quant.onnx";
    audio_cfg.fsmn.cmvn_path  = home + "/models/dev/vad/fsmn/am.mvn";

    // Configure TEN VAD model path.
    audio_cfg.ten.model_path = home + "/models/dev/vad/ten/ten-vad.onnx";

    // Track WS-level stats.
    std::atomic<uint64_t> total_frames{0};
    std::atomic<uint64_t> total_bytes{0};
    std::atomic<bool> loopback{false};
    std::atomic<int> active_ws_fd{-1};

    // Audio pipeline callbacks.
    audio.set_on_vad([&](const VadResult& vr, int frame_idx) {
        int fd = active_ws_fd.load(std::memory_order_relaxed);
        if (fd < 0) return;
        char json[256];
        snprintf(json, sizeof(json),
            R"({"type":"vad","speech":%s,"event":"%s","frame":%d,"energy":%.2f})",
            vr.is_speech ? "true" : "false",
            vr.segment_start ? "start" : (vr.segment_end ? "end" : "none"),
            frame_idx, vr.energy);
        server.send_text(fd, json);
        if (vr.segment_start)
            printf("[test-ws] VAD: speech START at frame %d (energy=%.2f)\n",
                   frame_idx, vr.energy);
        if (vr.segment_end)
            printf("[test-ws] VAD: speech END at frame %d\n", frame_idx);
    });

    audio.set_on_stats([&](const AudioPipelineStats& st) {
        int fd = active_ws_fd.load(std::memory_order_relaxed);
        if (fd < 0) return;
        char json[1024];
        snprintf(json, sizeof(json),
            R"({"type":"pipeline_stats","pcm_samples":%lu,"mel_frames":%lu,)"
            R"("speech_frames":%lu,"rms":%.4f,"energy":%.2f,"is_speech":%s,)"
            R"("threshold":%.2f,"noise_floor":%.2f,"gain":%.1f,)"
            R"("silero_prob":%.3f,"silero_speech":%s,"silero_threshold":%.2f,"silero_enabled":%s,)"
            R"("fsmn_prob":%.3f,"fsmn_speech":%s,"fsmn_threshold":%.2f,"fsmn_enabled":%s,)"
            R"("ten_prob":%.3f,"ten_speech":%s,"ten_threshold":%.2f,"ten_enabled":%s})",
            (unsigned long)st.pcm_samples_in,
            (unsigned long)st.mel_frames,
            (unsigned long)st.speech_frames,
            st.last_rms, st.last_energy,
            st.is_speech ? "true" : "false",
            audio.vad_threshold(), audio.vad_noise_floor(),
            audio.gain(),
            st.silero_prob, st.silero_speech ? "true" : "false",
            audio.silero_threshold(),
            audio.silero_enabled() ? "true" : "false",
            st.fsmn_prob, st.fsmn_speech ? "true" : "false",
            audio.fsmn_threshold(),
            audio.fsmn_enabled() ? "true" : "false",
            st.ten_prob, st.ten_speech ? "true" : "false",
            audio.ten_threshold(),
            audio.ten_enabled() ? "true" : "false");
        server.send_text(fd, json);
    });

    server.set_on_connect([&](int fd) {
        active_ws_fd.store(fd, std::memory_order_relaxed);
        printf("[test-ws] WS client connected  (fd=%d)\n", fd);
    });

    server.set_on_disconnect([&](int fd) {
        if (active_ws_fd.load() == fd)
            active_ws_fd.store(-1, std::memory_order_relaxed);
        printf("[test-ws] WS client disconnected (fd=%d)\n", fd);
    });

    server.set_on_text([&](int fd, const std::string& msg) {
        if (msg == "loopback:on") {
            loopback.store(true, std::memory_order_relaxed);
            server.send_text(fd, R"({"type":"loopback","enabled":true})");
            printf("[test-ws] Loopback ON (fd=%d)\n", fd);
        } else if (msg == "loopback:off") {
            loopback.store(false, std::memory_order_relaxed);
            server.send_text(fd, R"({"type":"loopback","enabled":false})");
            printf("[test-ws] Loopback OFF (fd=%d)\n", fd);
        } else if (msg.rfind("vad_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 14, nullptr);
            audio.set_vad_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"vad_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] VAD threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg.rfind("gain:", 0) == 0) {
            float g = std::strtof(msg.c_str() + 5, nullptr);
            if (g < 0.1f) g = 0.1f;
            if (g > 20.0f) g = 20.0f;
            audio.set_gain(g);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"gain","value":%.1f})", g);
            server.send_text(fd, json);
            printf("[test-ws] Gain = %.1fx (fd=%d)\n", g, fd);
        } else if (msg.rfind("silero_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 17, nullptr);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            audio.set_silero_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"silero_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] Silero threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg.rfind("fsmn_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 15, nullptr);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            audio.set_fsmn_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"fsmn_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] FSMN threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg.rfind("ten_threshold:", 0) == 0) {
            float t = std::strtof(msg.c_str() + 14, nullptr);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            audio.set_ten_threshold(t);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"ten_threshold","value":%.2f})", t);
            server.send_text(fd, json);
            printf("[test-ws] TEN threshold = %.2f (fd=%d)\n", t, fd);
        } else if (msg == "silero_enable:on" || msg == "silero_enable:off") {
            bool on = msg.back() == 'n';
            audio.set_silero_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"silero_enable","enabled":%s})", on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] Silero %s (fd=%d)\n", on ? "ON" : "OFF", fd);
        } else if (msg == "fsmn_enable:on" || msg == "fsmn_enable:off") {
            bool on = msg.back() == 'n';
            audio.set_fsmn_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"fsmn_enable","enabled":%s})", on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] FSMN %s (fd=%d)\n", on ? "ON" : "OFF", fd);
        } else if (msg == "ten_enable:on" || msg == "ten_enable:off") {
            bool on = msg.back() == 'n';
            audio.set_ten_enabled(on);
            char json[128];
            snprintf(json, sizeof(json),
                R"({"type":"ten_enable","enabled":%s})", on ? "true" : "false");
            server.send_text(fd, json);
            printf("[test-ws] TEN %s (fd=%d)\n", on ? "ON" : "OFF", fd);
        } else {
            printf("[test-ws] Text from fd=%d: %s\n", fd, msg.c_str());
        }
    });

    server.set_on_binary([&](int fd, const uint8_t* data, size_t len) {
        uint64_t f = total_frames.fetch_add(1, std::memory_order_relaxed) + 1;
        total_bytes.fetch_add(len, std::memory_order_relaxed);

        // Push PCM to audio pipeline.
        const int16_t* samples = reinterpret_cast<const int16_t*>(data);
        int n_samples = len / sizeof(int16_t);
        audio.push_pcm(samples, n_samples);

        // Quick RMS/peak for WS-level feedback (every 10 frames).
        if (f % 10 == 0) {
            double sum_sq = 0;
            int16_t peak_abs = 0;
            for (int i = 0; i < n_samples; i++) {
                int16_t s = samples[i];
                sum_sq += (double)s * s;
                int16_t a = s < 0 ? (int16_t)(-s) : s;
                if (a > peak_abs) peak_abs = a;
            }
            float rms  = n_samples > 0 ? sqrtf((float)(sum_sq / n_samples)) / 32768.0f : 0;
            float peak = peak_abs / 32768.0f;
            char json[256];
            snprintf(json, sizeof(json),
                R"({"type":"audio_stats","frames":%lu,"rms":%.4f,"peak":%.4f})",
                (unsigned long)f, rms, peak);
            server.send_text(fd, json);
        }

        // Loopback.
        if (loopback.load(std::memory_order_relaxed)) {
            server.send_binary(fd, data, len);
        }

        if (f % 500 == 0) {
            auto& st = audio.stats();
            printf("[test-ws] PCM: %lu frames | Mel: %lu | Speech: %lu | Energy: %.2f\n",
                   (unsigned long)f, (unsigned long)st.mel_frames,
                   (unsigned long)st.speech_frames, st.last_energy);
        }
    });

    // Start audio pipeline.
    if (!audio.start(audio_cfg)) {
        fprintf(stderr, "[test-ws] Failed to start audio pipeline\n");
        return 1;
    }

    // Start WS server.
    if (!server.start(ws_cfg)) {
        fprintf(stderr, "[test-ws] Failed to start WS server\n");
        audio.stop();
        return 1;
    }

    printf("[test-ws] Server running on http://localhost:%d\n", ws_cfg.port);
    printf("[test-ws] Audio pipeline: Mel(n_fft=%d hop=%d mels=%d) + VAD\n",
           audio_cfg.mel.n_fft, audio_cfg.mel.hop_length, audio_cfg.mel.n_mels);
    printf("[test-ws] Press Ctrl+C to stop...\n");

    // Block until SIGINT/SIGTERM.
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    sigprocmask(SIG_BLOCK, &mask, nullptr);
    int sig = 0;
    sigwait(&mask, &sig);
    printf("\n[test-ws] Caught signal %d, shutting down...\n", sig);

    audio.stop();
    server.stop();
    printf("[test-ws] Total: %lu WS frames, %.1f KB\n",
           total_frames.load(), total_bytes.load() / 1024.0);
    return 0;
}

} // namespace deusridet
