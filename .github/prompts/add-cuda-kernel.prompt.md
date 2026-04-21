---
mode: agent
description: Author a new CUDA kernel with Tegra-aware performance discipline.
---

# Add CUDA Kernel

Before writing the kernel, answer all of:

1. **Philosophical role**: Which subsystem does this serve? What principle
   from `docs/en/architecture/` does it implement? Write the
   `@philosophical_role` block first.
2. **Arithmetic intensity**: FLOPs per byte. Memory-bound or compute-bound?
3. **Memory layout**: Where are inputs/outputs? Device / pinned / unified /
   mmap? Alignment?
4. **Target utilization**: What fraction of 192 GB/s (Orin) is achievable?
   If below 40%, explain why — if no good reason, redesign.

## Mandatory checklist (per `cuda.instructions.md`)

- [ ] `float4` or equivalent 128-bit vectorized loads/stores
- [ ] Broadcast data (GEMV x, norms, scales) staged in SMEM
- [ ] Register caching between passes (no re-read from global)
- [ ] Inner loop unrolled ≥ 4-way
- [ ] Fast math intrinsics where precision permits
- [ ] Fused with adjacent elementwise ops where possible
- [ ] NVTX marker around the kernel launch
- [ ] One kernel per `.cu` file (or kernel + launcher)
- [ ] File will not exceed 800 lines
- [ ] Compiled with `-gencode arch=compute_87,code=sm_87`

## Tegra-specific checks

- [ ] No `cudaMemcpy(HostToDevice)` for pinned CPU-cached buffers (Orin
      pinned = cached, different from dGPU)
- [ ] Unified memory only where CPU and GPU both touch the buffer; else
      device memory
- [ ] `cudaMemGetInfo` not used for budget decisions (underestimates on Tegra)

## After writing

1. Correctness test in `tests/` vs a CPU reference (fp32 truth).
2. Bandwidth benchmark in `tools/` reporting `achieved / 192 GB/s`.
3. DEVLOG entry with achieved bandwidth and any optimization iterations.
