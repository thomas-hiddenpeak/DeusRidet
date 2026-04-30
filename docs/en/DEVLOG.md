# DeusRidet Development Log

Daily entries (newest first). Each link opens that day's full notes.

To add a new entry, create `devlog/YYYY-MM-DD.md` and prepend it here.

## DEVLOG

- [2026-04-30](devlog/2026-04-30.md) — Step 18n-18y: Auditus separated-source identity fusion selector, runtime fusion shadow ASR/speaker evidence, arbitrium + ledger/canary telemetry, deterministic/strong-pending enrollment, and Mel rolling compaction; r2 replay restored 4/4 births, 0 stable wrong, 0 authority violations
- [2026-04-29](devlog/2026-04-29.md) — Step 18l-18m: upstream ClearerVoice + 3D-Speaker + FunASR smoke and top30 Auditus upstream-vs-C++ probe; production py313 CUDA torch preserved
- [2026-04-28](devlog/2026-04-28.md) — Step 18g-18k: Auditus separated-source ASR + speaker-ID validation against GT timeline; raw_o16000 kept as production primary and failure-attributed by identity monitor
- [2026-04-27](devlog/2026-04-27.md) — Step 18: Auditus homogeneity audit, ASR split shadow metadata, real-clip MossFormer2 separation, FRCRN A/B, and fixed-window separator adaptation
- [2026-04-26](devlog/2026-04-26.md) — Step 17b-A: RetroFullRing Live Amend (speaker_amend frames; 3× 1x replays, decided_macro 0.924 / 0.930 / 0.949; no whole-speaker collapse)
- [2026-04-24](devlog/2026-04-24.md) — Step 11 Closed: Function-level Decomposition (A3 spectral_cluster, A2 SpeakerTracker::check, A1 AudioPipeline::process_loop 1574 → 353); Step 13: R1 residual clean-up (spectral_cluster_stages 636 → 3×≤235; audio_pipeline.h 511 → 489) — R1 fully clean
- [2026-04-23](devlog/2026-04-23.md) — Step 10 Complete: Actus Charter Restoration (5 atomic commits; `cmd_` prefix dropped, bench/profile → tools/, engine probes → tests/integration/, `test-ws` → `awaken`)
- [2026-04-22](devlog/2026-04-22.md) — Step 8a: Conscientia Facade Extraction (cmd_test_ws.cpp 458 → 392 lines; JSON helpers promoted to Communis)
- [2026-04-21](devlog/2026-04-21.md) — Step 7 Complete: Auditus Facade + Actus WS Routing (cmd_test_ws.cpp 1543 → 458) ; Step 9 Complete: CUDA/audio R1 Split Campaign (20 atomic commits, 12 oversized files resolved) ; Step 11 Opened: Function-level Decomposition
- [2026-04-20](devlog/2026-04-20.md) — S1–S4 Cycle: Semantic Metric + S3 FRCRN-Bypass + Seg3 Threshold Sweep ; Speaker Identification: A/B Testing & Direct Evaluation
- [2026-04-19](devlog/2026-04-19.md) — Overlap Detection Re-enabled: Test6 & Test7 Results ; v24d/v24e Speaker ID: Discovery Phase + Extensive Parameter Search ; GPTQ GEMM Optimization Round: 4 Experiments, 1 Win
- [2026-04-18](devlog/2026-04-18.md) — MossFormer2 Native CUDA Rewrite: Architecture Analysis & Prep ; Speaker Identification: 90%+ Accuracy Achieved ; Prefill Optimization Analysis: Near Hardware Limits (88 ms, M=11)
- [2026-04-17](devlog/2026-04-17.md) — Decode Fusion + INT4 Marlin Attention: 113→89 ms/tok (21% Speedup)
- [2026-04-16](devlog/2026-04-16.md) — Marlin Optimization: SMEM Right-Sizing + Fused Add + Tile Config
- [2026-04-15](devlog/2026-04-15.md) — Audio Enhancement P1 + P2: Overlap Detection & Speech Separation
- [2026-04-07](devlog/2026-04-07.md) — Phase 3.8 (Experiment): PTX MMA Replacing WMMA (Reverted)
- [2026-04-06](devlog/2026-04-06.md) — Phase 3.6: Sub-layer Profiling, Kernel Fusions, Inline Dequant ; Phase 3.7: Merged Projection Weights
- [2026-04-05](devlog/2026-04-05.md) — Phase 2.3: Bandwidth Optimization — SMEM, Vectorization, Kernel Fusion ; Phase 3.3: DeltaNet Kernel Fusion + Register-Cached State ; Phase 3.4: GPTQ + INT8 Kernel Deep Optimization ; Phase 3.5: SM87 Structural Analysis & INT8 Register Prefetch ; FP16 GEMM: ldmatrix.x2.trans + Weight Repacking + Bandwidth Ceiling Discovery
- [2026-04-04](devlog/2026-04-04.md) — Phase 2.1: Decode Speed Optimization ; Phase 2.2: CUDA Graph + Kernel Optimizations ; Phase 2.4: INT8 Quantization for FP16 Projections ; Phase 2.5: GPTQ Scale Deferral ; Phase 3.1: Tensor Core WMMA GEMM ; Phase 3.2: SMEM Bank Conflict Elimination
- [2026-04-03](devlog/2026-04-03.md) — Phase 3.0: Batched Prefill
- [2026-04-02](devlog/2026-04-02.md) — Phase 1: GPTQ-Int4 Kernels ; Phase 0 Complete: Foundation ; Development Plan ; Phase 2.6: Kernel Fusion + INT8 lm_head
- [2025-11-20](devlog/2025-11-20.md) — Overlap Detection 3-Stage Optimization: No Net Gain
- [2025-07-26](devlog/2025-07-26.md) — Speaker Identification Optimization: v14–v20b Experiments
- [2025-07-02](devlog/2025-07-02.md) — FP16 GEMM Kernel Optimization: cuBLAS Parity Achieved

