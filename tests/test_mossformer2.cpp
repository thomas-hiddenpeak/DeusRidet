// test_mossformer2.cpp — Numerical validation for native CUDA MossFormer2.
//
// Compares forward pass output against PyTorch reference stored in raw files.
// Reference: /tmp/mf2_ref_*.raw (generated from mossformer2_reference.npz)
//
// Usage:  ./build/test_mossformer2

#include "src/sensus/auditus/mossformer2.h"
#include "src/communis/log.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>

using namespace deusridet;

static std::vector<float> load_raw(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return {}; }
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0);
    std::vector<float> v(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

struct CompareResult {
    float max_abs_err;
    float mean_abs_err;
    float cos_sim;
    int   nan_count;
    int   inf_count;
};

static CompareResult compare(const float* a, const float* b, int n) {
    CompareResult r{};
    double sum_abs = 0, dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        float va = a[i], vb = b[i];
        if (std::isnan(va) || std::isnan(vb)) { r.nan_count++; continue; }
        if (std::isinf(va) || std::isinf(vb)) { r.inf_count++; continue; }
        float d = fabsf(va - vb);
        if (d > r.max_abs_err) r.max_abs_err = d;
        sum_abs += d;
        dot += (double)va * vb;
        na += (double)va * va;
        nb += (double)vb * vb;
    }
    r.mean_abs_err = (float)(sum_abs / n);
    double denom = sqrt(na) * sqrt(nb);
    r.cos_sim = denom > 0 ? (float)(dot / denom) : 0.f;
    return r;
}

static void print_result(const char* name, const CompareResult& r, float tol) {
    const char* status = (r.nan_count == 0 && r.inf_count == 0 &&
                          r.max_abs_err < tol) ? "PASS" : "FAIL";
    printf("  %-25s %s  max_err=%.6f  mean_err=%.6f  cos=%.8f",
           name, status, r.max_abs_err, r.mean_abs_err, r.cos_sim);
    if (r.nan_count) printf("  NaN=%d", r.nan_count);
    if (r.inf_count) printf("  Inf=%d", r.inf_count);
    printf("\n");
}

int main() {
    printf("=== MossFormer2 Numerical Validation ===\n\n");

    // Load reference
    auto ref_in  = load_raw("/tmp/mf2_ref_input.raw");
    auto ref_o0  = load_raw("/tmp/mf2_ref_output0.raw");
    auto ref_o1  = load_raw("/tmp/mf2_ref_output1.raw");
    if (ref_in.empty() || ref_o0.empty() || ref_o1.empty()) {
        fprintf(stderr, "Missing reference files in /tmp/mf2_ref_*.raw\n");
        return 1;
    }
    int n_samples = (int)ref_in.size();
    printf("Reference: input=%d samples, output0=%d, output1=%d\n\n",
           n_samples, (int)ref_o0.size(), (int)ref_o1.size());

    // Init model
    std::string model_path = std::string(getenv("HOME") ? getenv("HOME") : "/home/rm01")
                             + "/models/dev/vad/mossformer2_ss_16k.safetensors";

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    MossFormer2 model;
    printf("Loading model from %s ...\n", model_path.c_str());
    auto t0 = std::chrono::steady_clock::now();
    bool ok = model.init(model_path, n_samples, stream);
    auto t1 = std::chrono::steady_clock::now();
    float init_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (!ok) {
        fprintf(stderr, "MossFormer2::init() failed!\n");
        return 1;
    }
    printf("Init OK: %.1f ms\n\n", init_ms);

    // Allocate GPU I/O
    float *d_in, *d_out0, *d_out1;
    cudaMalloc(&d_in,   n_samples * sizeof(float));
    cudaMalloc(&d_out0, n_samples * sizeof(float));
    cudaMalloc(&d_out1, n_samples * sizeof(float));
    cudaMemcpy(d_in, ref_in.data(), n_samples * sizeof(float),
               cudaMemcpyHostToDevice);

    int L = (n_samples - 16) / 8 + 1;  // encoder output length
    printf("L = %d\n\n", L);

    // ---- Stage 1: Encoder only ----
    printf("--- Stage 1: Encoder ---\n");
    model.dbg_forward_encoder(d_in, n_samples, L);
    cudaStreamSynchronize(stream);

    auto ref_enc = load_raw("/tmp/mf2_ref_enc.raw");
    if (!ref_enc.empty()) {
        std::vector<float> enc_out(512 * L);
        cudaMemcpy(enc_out.data(), model.dbg_enc_out(), 512 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(enc_out.data(), ref_enc.data(), 512 * L);
        print_result("enc [512,L]", r, 1e-4f);
        // Print first few values
        printf("  CUDA enc[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", enc_out[i]);
        printf("\n  Ref  enc[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_enc[i]);
        printf("\n");
    }

    // ---- Stage 1b: MaskNet stage-by-stage ----
    printf("\n--- Stage 1b: MaskNet stage-by-stage ---\n");

    // Stage 1: CLNorm → d_work_c_ [512,L]
    model.dbg_forward_masknet_stage(L, 1);
    cudaStreamSynchronize(stream);
    auto ref_norm = load_raw("/tmp/mf2_ref_norm.raw");
    if (!ref_norm.empty()) {
        std::vector<float> buf(512 * L);
        cudaMemcpy(buf.data(), model.dbg_work_c(), 512 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_norm.data(), 512 * L);
        print_result("CLNorm [512,L]", r, 1e-3f);
        printf("  CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n  Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_norm[i]);
        printf("\n");
    }

    // Stage 2: Conv1d_enc → d_work_a_ [512,L]
    model.dbg_forward_masknet_stage(L, 2);
    cudaStreamSynchronize(stream);
    auto ref_conv_enc = load_raw("/tmp/mf2_ref_conv_enc.raw");
    if (!ref_conv_enc.empty()) {
        std::vector<float> buf(512 * L);
        cudaMemcpy(buf.data(), model.dbg_work_a(), 512 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_conv_enc.data(), 512 * L);
        print_result("Conv1d_enc [512,L]", r, 1e-3f);
        printf("  CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n  Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_conv_enc[i]);
        printf("\n");
    }

    // Stage 3: Transpose+SinuEmb → d_x_ [L,512]
    // PyTorch ref pos_enc is [512,L] channel-first, need to compare transposed
    model.dbg_forward_masknet_stage(L, 3);
    cudaStreamSynchronize(stream);
    auto ref_pos_enc = load_raw("/tmp/mf2_ref_pos_enc.raw");
    if (!ref_pos_enc.empty()) {
        // d_x_ is [L,512] row-major. Ref is [512,L] col-first.
        // Element [c,t] in ref = ref_pos_enc[c*L + t]
        // Element [t,c] in d_x_ = d_x_[t*512 + c]
        // They should match: d_x_[t*512+c] == ref[c*L+t]
        std::vector<float> buf(512 * L);
        cudaMemcpy(buf.data(), model.dbg_x(), 512 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        // Transpose ref to [L,512] for comparison
        std::vector<float> ref_T(512 * L);
        for (int t = 0; t < (int)L; t++)
            for (int c = 0; c < 512; c++)
                ref_T[t * 512 + c] = ref_pos_enc[c * L + t];
        auto r = compare(buf.data(), ref_T.data(), 512 * L);
        print_result("pos_enc (transposed) [L,512]", r, 1e-3f);
        printf("  CUDA d_x_[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n  Ref  [0:5]:     ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_T[i]);
        printf("\n");
    }

    // Layer 0 detailed: run from stage 4 (skip saved), then step-by-step
    // First: re-run to stage 4 to set up clean state
    model.dbg_forward_masknet_stage(L, 4);
    cudaStreamSynchronize(stream);

    // Substep 1: token shift → d_x_ [L,512]
    model.dbg_forward_flash_substep(0, L, 1);
    cudaStreamSynchronize(stream);
    auto ref_shifted = load_raw("/tmp/mf2_flash_shifted.raw");
    if (!ref_shifted.empty()) {
        std::vector<float> buf(512 * L);
        cudaMemcpy(buf.data(), model.dbg_x(), 512 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_shifted.data(), 512 * L);
        print_result("  FLASH.shift [L,512]", r, 0.01f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_shifted[i]);
        printf("\n");
    }

    // Substep 2: to_hidden → d_hidden_ [L,2048]
    model.dbg_forward_masknet_stage(L, 4);  // reset
    model.dbg_forward_flash_substep(0, L, 2);
    cudaStreamSynchronize(stream);
    auto ref_hidden = load_raw("/tmp/mf2_flash_hidden.raw");
    if (!ref_hidden.empty()) {
        std::vector<float> buf(2048 * L);
        cudaMemcpy(buf.data(), model.dbg_hidden(), 2048 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_hidden.data(), 2048 * L);
        print_result("  FLASH.hidden [L,2048]", r, 0.1f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_hidden[i]);
        printf("\n");
    }

    // Substep 3: to_qk → d_qk_ [L,128]
    model.dbg_forward_masknet_stage(L, 4);  // reset
    model.dbg_forward_flash_substep(0, L, 3);
    cudaStreamSynchronize(stream);
    auto ref_qk = load_raw("/tmp/mf2_flash_qk.raw");
    if (!ref_qk.empty()) {
        std::vector<float> buf(128 * L);
        cudaMemcpy(buf.data(), model.dbg_qk(), 128 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_qk.data(), 128 * L);
        print_result("  FLASH.qk [L,128]", r, 0.01f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_qk[i]);
        printf("\n");
    }

    // Substep 5: attention → d_work_b_ [Lpad,1024] (att_v, att_u)
    model.dbg_forward_masknet_stage(L, 4);  // reset
    model.dbg_forward_flash_substep(0, L, 5);
    cudaStreamSynchronize(stream);
    auto ref_att_v = load_raw("/tmp/mf2_flash_att_v.raw");
    auto ref_att_u = load_raw("/tmp/mf2_flash_att_u.raw");
    if (!ref_att_v.empty()) {
        std::vector<float> buf(1024 * L);
        cudaMemcpy(buf.data(), model.dbg_work_b(), 1024 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_att_v.data(), 1024 * L);
        print_result("  FLASH.att_v [L,1024]", r, 0.1f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_att_v[i]);
        printf("\n");
    }
    int Lpad = ((L + 255) / 256) * 256;
    if (!ref_att_u.empty()) {
        std::vector<float> buf(1024 * L);
        cudaMemcpy(buf.data(), model.dbg_work_b() + Lpad * 1024,
                   1024 * L * sizeof(float), cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_att_u.data(), 1024 * L);
        print_result("  FLASH.att_u [L,1024]", r, 0.1f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_att_u[i]);
        printf("\n");
    }

    // Substep 6: gate → d_hidden_ [L,1024]
    model.dbg_forward_masknet_stage(L, 4);  // reset
    model.dbg_forward_flash_substep(0, L, 6);
    cudaStreamSynchronize(stream);
    auto ref_gate = load_raw("/tmp/mf2_flash_gate_out.raw");
    if (!ref_gate.empty()) {
        std::vector<float> buf(1024 * L);
        cudaMemcpy(buf.data(), model.dbg_hidden(), 1024 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_gate.data(), 1024 * L);
        print_result("  FLASH.gate [L,1024]", r, 0.1f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_gate[i]);
        printf("\n");
    }

    // Substep 7: to_out + residual → d_x_ [L,512] (full FLASH output)
    model.dbg_forward_masknet_stage(L, 4);  // reset
    model.dbg_forward_flash_substep(0, L, 7);
    cudaStreamSynchronize(stream);
    auto ref_flash0 = load_raw("/tmp/mf2_flash_final.raw");
    if (!ref_flash0.empty()) {
        std::vector<float> buf(512 * L);
        cudaMemcpy(buf.data(), model.dbg_x(), 512 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_flash0.data(), 512 * L);
        print_result("  FLASH.final [L,512]", r, 0.01f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_flash0[i]);
        printf("\n");
    }

    // Run FSMN block 0 substeps (on top of FLASH layer 0 output)
    // Reset to FLASH output
    model.dbg_forward_masknet_stage(L, 4);
    model.dbg_forward_flash_substep(0, L, 7);
    cudaStreamSynchronize(stream);

    // FSMN substep 1: conv1+prelu → d_fsmn_a_ [256,L]
    model.dbg_forward_fsmn_substep(0, L, 1);
    cudaStreamSynchronize(stream);
    auto ref_fsmn_conv1 = load_raw("/tmp/mf2_fsmn_conv1.raw");
    if (!ref_fsmn_conv1.empty()) {
        std::vector<float> buf(256 * L);
        cudaMemcpy(buf.data(), model.dbg_fsmn_a(), 256 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_fsmn_conv1.data(), 256 * L);
        print_result("  FSMN.conv1 [256,L]", r, 0.1f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_fsmn_conv1[i]);
        printf("\n");
    }

    // FSMN substep 2: norm1 → d_fsmn_b_ [256,L]
    model.dbg_forward_masknet_stage(L, 4);
    model.dbg_forward_flash_substep(0, L, 7);
    model.dbg_forward_fsmn_substep(0, L, 2);
    cudaStreamSynchronize(stream);
    auto ref_fsmn_norm1 = load_raw("/tmp/mf2_fsmn_norm1.raw");
    if (!ref_fsmn_norm1.empty()) {
        std::vector<float> buf(256 * L);
        cudaMemcpy(buf.data(), model.dbg_fsmn_b(), 256 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_fsmn_norm1.data(), 256 * L);
        print_result("  FSMN.norm1 [256,L]", r, 0.1f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_fsmn_norm1[i]);
        printf("\n");
    }

    // FSMN substep 3: to_u → d_fsmn_b_ [L,256]
    model.dbg_forward_masknet_stage(L, 4);
    model.dbg_forward_flash_substep(0, L, 7);
    model.dbg_forward_fsmn_substep(0, L, 3);
    cudaStreamSynchronize(stream);
    auto ref_fsmn_to_u = load_raw("/tmp/mf2_fsmn_to_u.raw");
    if (!ref_fsmn_to_u.empty()) {
        std::vector<float> buf(256 * L);
        cudaMemcpy(buf.data(), model.dbg_fsmn_b(), 256 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_fsmn_to_u.data(), 256 * L);
        print_result("  FSMN.to_u [L,256]", r, 0.1f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_fsmn_to_u[i]);
        printf("\n");
    }

    // FSMN substep 4: to_v → d_hidden_ [L,256]
    model.dbg_forward_masknet_stage(L, 4);
    model.dbg_forward_flash_substep(0, L, 7);
    model.dbg_forward_fsmn_substep(0, L, 4);
    cudaStreamSynchronize(stream);
    auto ref_fsmn_to_v = load_raw("/tmp/mf2_fsmn_to_v.raw");
    if (!ref_fsmn_to_v.empty()) {
        std::vector<float> buf(256 * L);
        cudaMemcpy(buf.data(), model.dbg_hidden(), 256 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_fsmn_to_v.data(), 256 * L);
        print_result("  FSMN.to_v [L,256]", r, 0.1f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_fsmn_to_v[i]);
        printf("\n");
    }

    // FSMN substep 5: uni_fsmn → d_fsmn_b_ [L,256]
    // Also check intermediates from forward_uni_fsmn stored in d_work_a_
    model.dbg_forward_masknet_stage(L, 4);
    model.dbg_forward_flash_substep(0, L, 7);
    model.dbg_forward_fsmn_substep(0, L, 5);
    cudaStreamSynchronize(stream);

    // Check uni_relu: d_work_a_[0..n256-1] = [L,256]
    auto ref_uni_relu = load_raw("/tmp/mf2_fsmn_uni_relu.raw");
    if (!ref_uni_relu.empty()) {
        int n256 = 256 * L;
        std::vector<float> buf(n256);
        cudaMemcpy(buf.data(), model.dbg_work_a(), n256 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_uni_relu.data(), n256);
        print_result("  FSMN.uni_relu [L,256]", r, 0.1f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_uni_relu[i]);
        printf("\n");
    }

    // Check uni_proj: d_work_a_[n256..2*n256-1] = [L,256]
    auto ref_uni_proj = load_raw("/tmp/mf2_fsmn_uni_proj.raw");
    if (!ref_uni_proj.empty()) {
        int n256 = 256 * L;
        std::vector<float> buf(n256);
        cudaMemcpy(buf.data(), model.dbg_work_a() + n256, n256 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_uni_proj.data(), n256);
        print_result("  FSMN.uni_proj [L,256]", r, 0.1f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_uni_proj[i]);
        printf("\n");
    }

    // Check uni_ddc: d_work_a_[3*n256..4*n256-1] = [256,L] (channel-first)
    // Ref is [L,256] row-major, need to transpose CUDA for comparison
    auto ref_uni_ddc = load_raw("/tmp/mf2_fsmn_uni_ddc.raw");
    if (!ref_uni_ddc.empty()) {
        int n256 = 256 * L;
        std::vector<float> buf(n256);
        cudaMemcpy(buf.data(), model.dbg_work_a() + 3 * n256, n256 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        // Transpose [256,L] → [L,256]
        std::vector<float> buf_T(n256);
        for (int t = 0; t < (int)L; t++)
            for (int c = 0; c < 256; c++)
                buf_T[t * 256 + c] = buf[c * L + t];
        auto r = compare(buf_T.data(), ref_uni_ddc.data(), n256);
        print_result("  FSMN.uni_ddc [L,256]", r, 0.5f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf_T[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_uni_ddc[i]);
        printf("\n");
    }

    auto ref_fsmn_uni = load_raw("/tmp/mf2_fsmn_uni_out.raw");
    if (!ref_fsmn_uni.empty()) {
        std::vector<float> buf(256 * L);
        cudaMemcpy(buf.data(), model.dbg_fsmn_b(), 256 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_fsmn_uni.data(), 256 * L);
        print_result("  FSMN.uni_out [L,256]", r, 0.5f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_fsmn_uni[i]);
        printf("\n");
    }

    // FSMN substep 6: gate → d_fsmn_b_ [L,256]
    model.dbg_forward_masknet_stage(L, 4);
    model.dbg_forward_flash_substep(0, L, 7);
    model.dbg_forward_fsmn_substep(0, L, 6);
    cudaStreamSynchronize(stream);
    auto ref_fsmn_gate = load_raw("/tmp/mf2_fsmn_gate_out.raw");
    if (!ref_fsmn_gate.empty()) {
        std::vector<float> buf(256 * L);
        cudaMemcpy(buf.data(), model.dbg_fsmn_b(), 256 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_fsmn_gate.data(), 256 * L);
        print_result("  FSMN.gate [L,256]", r, 0.5f);
        printf("    CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n    Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_fsmn_gate[i]);
        printf("\n");
    }

    // FSMN substep 8: full (conv2+residual) → d_x_ [L,512]
    model.dbg_forward_masknet_stage(L, 4);
    model.dbg_forward_flash_substep(0, L, 7);
    model.dbg_forward_fsmn_substep(0, L, 8);
    cudaStreamSynchronize(stream);
    auto ref_layer0 = load_raw("/tmp/mf2_fsmn_final.raw");
    if (!ref_layer0.empty()) {
        std::vector<float> buf(512 * L);
        cudaMemcpy(buf.data(), model.dbg_x(), 512 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_layer0.data(), 512 * L);
        print_result("Layer 0 (FLASH+FSMN) [L,512]", r, 0.5f);
        printf("  CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n  Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_layer0[i]);
        printf("\n");
    }

    // Stage 5+24: After all 24 layers
    model.dbg_forward_masknet_stage(L, 29);  // 5 + 24 layers
    cudaStreamSynchronize(stream);
    {
        std::vector<float> buf(512 * L);
        cudaMemcpy(buf.data(), model.dbg_x(), 512 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        float mn = *std::min_element(buf.begin(), buf.end());
        float mx = *std::max_element(buf.begin(), buf.end());
        bool has_nan = false;
        for (auto v : buf) if (std::isnan(v)) { has_nan = true; break; }
        printf("After all 24 layers (pre-LN): range [%.4f, %.4f]%s\n", mn, mx,
               has_nan ? " ** HAS NaN **" : "");
        printf("  d_x_[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n");
    }

    // After LayerNorm + transpose + IntraNorm + skip add + PReLU (stage 33)
    model.dbg_forward_masknet_stage(L, 33);
    cudaStreamSynchronize(stream);
    auto ref_prelu = load_raw("/tmp/mf2_ref_prelu.raw");
    if (!ref_prelu.empty()) {
        std::vector<float> buf(512 * L);
        cudaMemcpy(buf.data(), model.dbg_work_a(), 512 * L * sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto r = compare(buf.data(), ref_prelu.data(), 512 * L);
        print_result("PReLU [512,L]", r, 0.5f);
        printf("  CUDA[0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", buf[i]);
        printf("\n  Ref [0:5]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", ref_prelu[i]);
        printf("\n");
    }

    // Full masknet for outputs comparison
    printf("\n--- Stage 2: Full forward ---\n");
    // Re-copy input (encoder modified scratch)
    cudaMemcpy(d_in, ref_in.data(), n_samples * sizeof(float),
               cudaMemcpyHostToDevice);

    // Run forward
    printf("Running forward (n=%d) ...\n", n_samples);
    t0 = std::chrono::steady_clock::now();
    ok = model.forward(d_in, d_out0, d_out1, n_samples);
    t1 = std::chrono::steady_clock::now();
    float fwd_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (!ok) {
        fprintf(stderr, "MossFormer2::forward() failed!\n");
        return 1;
    }
    printf("Forward OK: %.2f ms (%.4fx RT for 2s audio)\n\n", fwd_ms, fwd_ms / 2000.f);

    // Copy results back
    std::vector<float> out0(n_samples), out1(n_samples);
    cudaMemcpy(out0.data(), d_out0, n_samples * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out1.data(), d_out1, n_samples * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compare
    printf("--- Final output comparison ---\n");
    float tol = 0.01f;  // absolute tolerance
    auto r0 = compare(out0.data(), ref_o0.data(), n_samples);
    auto r1 = compare(out1.data(), ref_o1.data(), n_samples);
    print_result("output0 (speaker 1)", r0, tol);
    print_result("output1 (speaker 2)", r1, tol);

    // Output stats
    printf("\n--- Output stats ---\n");
    float min0 = 1e9, max0 = -1e9, min1 = 1e9, max1 = -1e9;
    for (int i = 0; i < n_samples; i++) {
        min0 = std::min(min0, out0[i]); max0 = std::max(max0, out0[i]);
        min1 = std::min(min1, out1[i]); max1 = std::max(max1, out1[i]);
    }
    printf("  CUDA output0: min=%.6f max=%.6f\n", min0, max0);
    printf("  Ref  output0: min=%.6f max=%.6f\n", ref_o0[0], *std::max_element(ref_o0.begin(), ref_o0.end()));
    printf("  CUDA output1: min=%.6f max=%.6f\n", min1, max1);
    printf("  Ref  output1: min=%.6f max=%.6f\n", ref_o1[0], *std::max_element(ref_o1.begin(), ref_o1.end()));

    // Run 2 more iterations for timing
    printf("\n--- Timing (3 more runs) ---\n");
    for (int r = 0; r < 3; r++) {
        t0 = std::chrono::steady_clock::now();
        model.forward(d_in, d_out0, d_out1, n_samples);
        t1 = std::chrono::steady_clock::now();
        printf("  Run %d: %.2f ms\n", r + 1,
               std::chrono::duration<float, std::milli>(t1 - t0).count());
    }

    cudaFree(d_in); cudaFree(d_out0); cudaFree(d_out1);

    bool pass = (r0.nan_count == 0 && r1.nan_count == 0 &&
                 r0.inf_count == 0 && r1.inf_count == 0 &&
                 r0.cos_sim > 0.99f && r1.cos_sim > 0.99f);
    printf("\n=== %s ===\n", pass ? "VALIDATION PASSED" : "VALIDATION FAILED");
    return pass ? 0 : 1;
}
