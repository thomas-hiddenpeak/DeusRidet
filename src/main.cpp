// main.cpp — DeusRidet entry point
//
// Boot sequence: parse config → load weights → initialize tokenizer
//                → validate → report status
//
// Commands:
//   test-tokenizer   — Encode/decode round-trip test
//   test-weights     — Load and print weight tensor summary
//   version          — Print version and hardware info
//   (future)         — boot consciousness stream
//
// DeusRidet — consciousness should not be locked behind closed doors.
// Licensed under GPLv3.

#include "communis/config.h"
#include "communis/log.h"
#include "machina/gptq.h"
#include "machina/safetensors.h"
#include "machina/tokenizer.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#include <cuda_runtime.h>

static const char* VERSION    = "0.1.0";
static const char* BUILD_DATE = __DATE__;

// ============================================================================
// Crash handler
// ============================================================================

static void crash_handler(int sig) {
    const char* msg = "\n[CRASH] Signal: ";
    ssize_t r_ __attribute__((unused));
    r_ = write(STDERR_FILENO, msg, strlen(msg));
    char num[16];
    int len = snprintf(num, sizeof(num), "%d\n", sig);
    r_ = write(STDERR_FILENO, num, len);

    void* frames[32];
    int n = backtrace(frames, 32);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);

    signal(sig, SIG_DFL);
    raise(sig);
}

// ============================================================================
// Commands
// ============================================================================

static void print_version() {
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

static void print_usage() {
    printf("\n  DeusRidet v%s — Continuous Consciousness Engine\n\n", VERSION);
    printf("  Usage:\n");
    printf("    deusridet <command> [options]\n\n");
    printf("  Commands:\n");
    printf("    test-tokenizer <text>   Encode/decode round-trip test\n");
    printf("    test-weights            Load weights and print tensor summary\n");
    printf("    test-gptq               GPTQ kernel correctness test with model weights\n");
    printf("    bench-gptq              GPTQ GEMV/GEMM benchmark\n");
    printf("    version                 Print version and hardware info\n\n");
    printf("  Options:\n");
    printf("    --config <file>         Configuration file (default: configs/machina.conf)\n");
    printf("    --model-dir <path>      Override LLM model directory\n\n");
}

static int cmd_test_tokenizer(const std::string& model_dir, const std::string& text) {
    deusridet::Tokenizer tokenizer;
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

static int cmd_test_weights(const std::string& model_dir) {
    LOG_INFO("Main", "Loading weights from %s ...", model_dir.c_str());

    deusridet::SafetensorsLoader loader(model_dir);
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
                   deusridet::dtype_name(tensor->dtype()));
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
static int cmd_test_gptq(const std::string& model_dir) {
    LOG_INFO("Main", "Loading model weights for GPTQ correctness test...");

    deusridet::SafetensorsLoader loader(model_dir);

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
    size_t x_bytes = (size_t)K * sizeof(__half);
    size_t y_bytes = (size_t)N * sizeof(__half);

    __half* d_x;
    __half* d_y;
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_y, y_bytes);

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
    cudaError_t cerr;

    cerr = cudaMalloc(&d_qw, qw_bytes);
    if (cerr != cudaSuccess) {
        LOG_ERROR("Main", "cudaMalloc qweight failed: %s", cudaGetErrorString(cerr));
        return 1;
    }
    cerr = cudaMalloc(&d_sc, sc_bytes);
    if (cerr != cudaSuccess) {
        LOG_ERROR("Main", "cudaMalloc scales failed: %s", cudaGetErrorString(cerr));
        cudaFree(d_qw);
        return 1;
    }
    cudaMemcpy(d_qw, qw_tensor->data(), qw_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, sc_tensor->data(), sc_bytes, cudaMemcpyHostToDevice);

    deusridet::GptqWeight weight;
    weight.qweight = d_qw;
    weight.scales  = d_sc;
    weight.K       = K;
    weight.N       = N;

    // Run GEMV
    printf("[GPTQ Test] Running GEMV (M=1, K=%d, N=%d)...\n", K, N);
    deusridet::gptq_gemv(d_x, weight, d_y);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("Main", "CUDA error: %s", cudaGetErrorString(err));
        cudaFree(d_x); cudaFree(d_y); free(h_x);
        return 1;
    }

    // Copy result back
    __half* h_y = (__half*)malloc(y_bytes);
    cudaMemcpy(h_y, d_y, y_bytes, cudaMemcpyDeviceToHost);

    // CPU reference (partial — first 256 columns for speed)
    int check_N = (N < 256) ? N : 256;
    printf("[GPTQ Test] CPU reference check (first %d columns)...\n", check_N);

    const uint32_t* h_qw = (const uint32_t*)qw_tensor->data();
    const __half*   h_sc = (const __half*)sc_tensor->data();

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
    cudaMalloc(&d_xm, xm_bytes);
    cudaMalloc(&d_ym, ym_bytes);

    __half* h_xm = (__half*)malloc(xm_bytes);
    for (int i = 0; i < M_test * K; i++) {
        h_xm[i] = __float2half(((float)(rand() % 1000) - 500.0f) / 5000.0f);
    }
    cudaMemcpy(d_xm, h_xm, xm_bytes, cudaMemcpyHostToDevice);

    printf("\n[GPTQ Test] Running GEMM (M=%d, K=%d, N=%d)...\n", M_test, K, N);
    deusridet::gptq_gemm(d_xm, weight, d_ym, M_test);
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

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_xm); cudaFree(d_ym);
    cudaFree(d_qw); cudaFree(d_sc);
    free(h_x); free(h_y); free(h_xm);

    return pass ? 0 : 1;
}

// ============================================================================
// bench-gptq: GPTQ GEMV/GEMM benchmark with synthetic data
// ============================================================================
static int cmd_bench_gptq() {
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
        auto r = deusridet::gptq_benchmark(bc.K, bc.N, bc.M, bc.warmup, bc.iters);

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
// Main
// ============================================================================

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    signal(SIGSEGV, crash_handler);
    signal(SIGBUS,  crash_handler);
    signal(SIGFPE,  crash_handler);
    signal(SIGABRT, crash_handler);

    if (argc < 2) {
        print_usage();
        return 0;
    }

    std::string cmd = argv[1];

    // Parse global options
    std::string config_path = "configs/machina.conf";
    std::string model_dir_override;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir_override = argv[++i];
        }
    }

    // Load config
    deusridet::Config cfg;
    cfg.load(config_path);

    std::string model_dir = model_dir_override.empty()
        ? cfg.get_string("llm_model_dir")
        : model_dir_override;

    // Dispatch
    int rc = 0;
    if (cmd == "version" || cmd == "--version" || cmd == "-v") {
        print_version();
    }
    else if (cmd == "test-tokenizer") {
        std::string text = "Hello, world! 你好世界";
        // Use remaining args as text
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--config" || arg == "--model-dir") { i++; continue; }
            text = arg;
            break;
        }
        rc = cmd_test_tokenizer(model_dir, text);
    }
    else if (cmd == "test-weights") {
        rc = cmd_test_weights(model_dir);
    }
    else if (cmd == "test-gptq") {
        rc = cmd_test_gptq(model_dir);
    }
    else if (cmd == "bench-gptq") {
        rc = cmd_bench_gptq();
    }
    else if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        print_usage();
    }
    else {
        fprintf(stderr, "Unknown command: %s\nRun 'deusridet --help' for usage.\n", cmd.c_str());
        rc = 1;
    }

    cudaDeviceReset();
    return rc;
}
