---
applyTo: "**/*.{cu,cuh}"
---

# CUDA — Tegra / SM87 Engineering Rules

## Target Hardware

| Spec | Primary (Orin) | Future (Thor) |
|------|---------------|---------------|
| Platform | Jetson AGX Orin 64 GB | Jetson AGX Thor 128 GB |
| GPU Arch | SM87 (Ampere) | SM110a (Blackwell) |
| Memory BW | ~192 GB/s | ~273 GB/s |
| CUDA Build | `-gencode arch=compute_87,code=sm_87` | `compute_110,code=sm_110a` |
| Supported Quant | INT4 (GPTQ), FP16, INT8 | + FP4, FP8 |

## CUDA for Tegra — Critical Reference

**Mandatory reading before writing any CUDA kernel**:
[CUDA for Tegra Application Note](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/)

The AGX Orin is an integrated GPU (iGPU) with **unified memory** — CPU and
iGPU share the same physical DRAM. This fundamentally changes memory
programming compared to discrete GPUs (dGPU):

- **No** separate VIDMEM — `cudaMalloc` allocates from system DRAM.
- **I/O coherency** (one-way): iGPU can read latest CPU cache lines; GPU
  cache management still required (handled by CUDA driver for
  managed/interop memory).
- **Pinned memory is CPU-cached** on SM87 (compute capability ≥ 7.2, I/O
  coherent) — different from dGPU where pinned = uncached.
- **Unified memory** cached on both CPU and iGPU with coherency overhead at
  kernel launch/sync. Prefer `cudaStreamAttachMemAsync()` prefetch hints.
- **Device memory** preferred for GPU-only buffers (intermediate activations,
  KV Cache blocks) — avoids coherency overhead entirely.
- **`cudaMemGetInfo`** underestimates allocatable memory on Tegra (does not
  account for swap). Use `/proc/meminfo` + `NvMapMemUsed` for accurate estimate.
- **JIT compilation not recommended** — always compile for specific SM target
  to avoid determinism issues.
- **No P2P**, no `nvGRAPH`, no UVM between CUDA and DLA on Orin.
- **Synchronization**: `cudaDeviceScheduleSpin` for low-latency;
  `cudaDeviceBlockingSync` for power savings.

Every CUDA kernel and memory allocation must be evaluated against these
Tegra-specific constraints. When in doubt, consult the application note —
do not assume dGPU behavior applies.

## Performance Optimization Principle

**Do not use "hardware limitations" as an excuse to skip optimizations.**
Every CUDA kernel must pursue maximum memory bandwidth utilization. Target:
**at least 60% of theoretical DRAM bandwidth** (≥ 115 GB/s on Orin).
Kernels below 40% must be investigated and optimized before new features.

### Mandatory practices

- **Vectorized memory access**: All elementwise and GEMV kernels must use
  `float4` (128-bit) loads/stores. Scalar `__half` loads are prohibited in
  performance-critical paths.
- **Shared memory for broadcast data**: Any input vector read by multiple
  threads (e.g. x in GEMV) must be loaded to SMEM, not read redundantly
  from global/L1.
- **Register caching**: Two-pass kernels (e.g. RMSNorm) must cache
  intermediate values in registers between passes — never re-read from
  global memory.
- **Kernel fusion**: Fuse adjacent elementwise operations into GEMV output
  writes or norm kernels. Every standalone elementwise kernel on a small
  buffer (< 64 KB) is a red flag — launch overhead dominates.
- **Scale/constant hoisting**: Values constant within a loop iteration
  group must be loaded once and reused (e.g. GPTQ scales per group of 16 rows).
- **Loop unrolling for memory pipelining**: Inner loops with global memory
  loads must be unrolled (≥ 4-way) to keep multiple loads in-flight.
- **Fast math intrinsics**: Use `__expf`, `__logf`, `exp2f` instead of
  `expf`, `logf` where precision permits (SiLU, softmax, etc.).
- **Reduce kernel launches**: Prefer fused kernels over sequential launches.
  One kernel doing 3 operations > 3 kernels doing 1 each.

## Quantization Kernel Requirements

| Format | Decode (B=1~few) | Prefill (B≥17) | Status |
|--------|-------------------|-----------------|--------|
| GPTQ-Int4 | Dequant + GEMV (group_size=128, symmetric) | Dequant + GEMM | **P0** |
| FP16 (BF16) | cuBLAS GEMV | CUTLASS/cuBLAS GEMM | **P1** |
| INT8 | Dequant + GEMV | Dequant + GEMM | **P2** |

The GPTQ kernel is entirely new work — neither reference project supports it.

## Kernel Hygiene

- One kernel per `.cu` file where practical. Kernel + its launcher may coexist.
- Use NVTX markers for all significant GPU operations.
- Name kernels `<module>_<op>_kernel` — never generic names.
- File-size hard limit: 800 lines. If a kernel family exceeds this, split
  by operation (e.g. `gptq_gemv.cu` + `gptq_gemm.cu`).

## Reference Projects

| Project | Role |
|---------|------|
| [qwen35-thor](https://github.com/thomas-hiddenpeak/qwen35-thor) | C++/CUDA inference engine reference (SM110a Blackwell) |
| qwen35-orin (`~/qwen35-orin`) | C++/CUDA engine + ASR/TTS plugins (SM87 Orin) |

These are references only — **do not copy code verbatim**. Adapt ideas to
fit the consciousness-centric design. Attribution: see `docs.instructions.md`.
