# DeusRidet Development Log

## 2026-04-02 — Phase 0 Complete: Foundation

### Result

Phase 0 implemented and validated. All milestone criteria met.

**Components built:**
- CMake scaffold: SM87 targeting, CUTLASS submodule, `machina` + `communis` static libs
- `SafetensorsLoader`: zero-copy mmap, multi-shard support via `model.safetensors.index.json`
- `Tokenizer`: GPT2 byte-level BPE, 248070 vocab, 247504 merges, 26 special tokens
- `Allocator`: `DeviceAllocator` (cudaMalloc), `UnifiedAllocator` (cudaMallocManaged),
  `MmapAllocator` (adaptive mmap with populate/willneed strategy)
- `Tensor`: lightweight descriptor with owning/non-owning modes
- `Config`: key=value parser (`machina.conf`) with typed accessors
- `Log`: structured stderr logging with timestamps

**Validation on AGX Orin 64GB:**
- `test-tokenizer "Hello, world! 你好世界"` → 7 tokens, round-trip PASS
- `test-tokenizer "人类一思考，上帝就发笑；AI一思考，人类就不笑了。"` → round-trip PASS
- `test-weights` → 1775 tensors, 28.16 GB across 11 shards, all loaded
- `version` → Orin SM8.7, 16 SMs, 61.4 GB, CUDA 12.6

**Files (19 new, ~10800 lines):**
`CMakeLists.txt`, `LICENSE`, `.gitignore`, `.gitmodules`, `configs/machina.conf`,
`docs/ACKNOWLEDGMENTS.md`, `src/main.cpp`, `src/communis/{config.h,config.cpp,log.h}`,
`src/machina/{allocator.h,allocator.cpp,tensor.h,safetensors.h,safetensors.cpp,
tokenizer.h,tokenizer.cpp}`, `third_party/{cutlass,stb}`

**Next:** Phase 1 — GPTQ-Int4 dequant kernels (GEMV for decode, GEMM for prefill)

---

## 2026-04-02 — Development Plan

### Context

Environment verified: CUDA 12.6, CMake 4.1.0, GCC 11.4.0, Jetson AGX Orin 64 GB
(ARM v8l, SM87). Reference project qwen35-orin provides a complete inference
engine (safetensors loader, BPE tokenizer, DeltaNet SSM, GQA attention, paged KV
cache with SSD swap, ViT vision, ASR, TTS, speaker ID, VAD, WS/HTTP server) but
lacks GPTQ-Int4 kernels and GPU sampling.

LLM config (Qwen3.5-27B-GPTQ-Int4): vocab 248320, hidden 5120, 64 layers (48
linear_attention + 16 full_attention, every 4th), intermediate 17408, 24 attn
heads / 4 KV heads, head_dim 256, SSM: 16 key heads / 48 value heads / conv
kernel 4 / key/value_head_dim 128, MTP 1 layer, rope_theta 10M, partial_rotary
0.25, mRoPE interleaved [11,11,10]. GPTQ bits=4, group_size=128, sym=true,
desc_act=false.

### Strategy

Critical path: LLM must produce correct tokens before consciousness can exist.
Build bottom-up: kernels → single forward pass → continuous loop → perception →
expression. Reference code is adapted, never copied — every kernel validated
against the Tegra unified memory model.

---

### Phase 0 — Foundation (Build System + Weight Loading + Tokenizer)

Goal: CMake project compiles, loads weights, tokenizes text, allocates GPU memory.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 0.1 | CMake scaffold | — | Builds empty `deusridet` binary, SM87, CUTLASS submodule linked | `cmake --build` succeeds |
| 0.2 | SafetensorsLoader | 11 shard files | Tensor map: name → {dtype, shape, mmap ptr} | Print tensor names + shapes, compare with `model.safetensors.index.json` |
| 0.3 | GPU memory pool (Allocator) | — | Arena allocator on device memory | Alloc/free round-trip, fragmentation test |
| 0.4 | BPE Tokenizer | `vocab.json` + `merges.txt` | `encode(text) → token_ids`, `decode(ids) → text` | Round-trip: `decode(encode(s)) == s` for 100 test strings, special tokens correct |
| 0.5 | Config parser | `machina.conf` | Struct with model paths, quant params, memory budget | Load + print, no crash |

Milestone: `./deusridet --test-tokenizer "Hello world"` prints token IDs and
decoded text.

---

### Phase 1 — GPTQ-Int4 Kernel (The Hardest Piece)

Goal: Correct GPTQ dequant + matmul for both Decode (GEMV) and Prefill (GEMM).

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 1.1 | GPTQ weight unpacker | Packed INT4 tensor + scales + zeros | Per-group dequant to FP16 buffer | Bit-exact match against Python GPTQ reference dequant |
| 1.2 | GPTQ GEMV kernel (Decode) | x[1, K] × W_q[K, N] (INT4) | y[1, N] (FP16) | Max abs error < 1e-2 vs FP16 reference for 100 random inputs |
| 1.3 | GPTQ GEMM kernel (Prefill) | X[M, K] × W_q[K, N] (INT4), M=32..512 | Y[M, N] (FP16) | Same tolerance, M=32,64,128,256,512 |
| 1.4 | Benchmark GEMV/GEMM | Various M,K,N | tok/s, TFLOPS, bandwidth utilization | Must beat cuBLAS FP16 on memory bandwidth (4× less data moved) |
| 1.5 | Integration: replace linear layers | Model forward scaffold | GPTQ path for quantized layers, FP16 path for non-quantized | Single MLP block output matches Python reference |

Key dimensions: K=5120→N=17408 (gate/up proj), K=17408→N=5120 (down proj),
K=5120→N=5120 (QKV proj). Group size 128, symmetric quantization.

Milestone: `./deusridet --bench-gptq` shows correct outputs + throughput numbers.

---

### Phase 2 — Single-Layer Forward Pass

Goal: One full Transformer layer (SSM or Full Attention) produces correct output.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 2.1 | RMSNorm kernel | x[B, L, 5120] | normalized x | Max error < 1e-5 vs Python |
| 2.2 | RoPE kernel (mRoPE) | Q/K tensors, position ids | Rotated Q/K with partial_rotary=0.25, interleaved [11,11,10] | Compare first 4 positions vs Python transformers |
| 2.3 | SiLU + elementwise gate | gate[B,L,17408], up[B,L,17408] | gate_output[B,L,17408] | Exact match |
| 2.4 | DeltaNet SSM forward | input, recurrent state, conv state | output, updated states | Compare vs HuggingFace DeltaNet reference for 1 layer |
| 2.5 | GQA Full Attention (Prefill) | Q[B,L,24,256], K[B,L,4,256], V | Attention output, KV stored to paged blocks | Compare vs `F.scaled_dot_product_attention` |
| 2.6 | Paged Attention (Decode) | Q[B,1,24,256], block table | Decoded output from paged KV | Compare for cached sequence |
| 2.7 | Full layer assembly | Token embeddings for 1 layer | Layer output | End-to-end layer output vs Python, both SSM and FA types |

Milestone: `./deusridet --test-layer 0` and `--test-layer 3` (SSM vs FA) match
Python reference outputs.

---

### Phase 3 — Full Model Forward + Sampling + Text Generation

Goal: Model generates coherent text from a prompt.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 3.1 | Embed + 64-layer forward (Prefill) | Token IDs [1, L] | Hidden states [1, L, 5120] → logits [1, L, 248320] | Last-token logits top-5 match Python top-5 |
| 3.2 | KV Cache Manager | — | Block pool alloc/free, block table per sequence | Stress test: alloc→fill→evict→realloc |
| 3.3 | Decode loop | Prefilled state + new token | Next-token logits | Autoregressive: 10 tokens match Python greedy |
| 3.4 | GPU sampling (top-k, top-p, temperature) | Logits [1, V] | Sampled token ID | Statistical distribution test (1000 samples, chi-squared) |
| 3.5 | MTP speculative decoding | Draft layer + verify | Accepted tokens | Acceptance rate > 50% on standard prompts |
| 3.6 | Chat template + stop tokens | ChatML formatted input | Correct generation with EOS handling | Multi-turn conversation produces sensible output |
| 3.7 | End-to-end text generation | "What is consciousness?" | Coherent multi-sentence response | Human evaluation: makes sense |

Milestone: `./deusridet --chat` interactive text generation works.

---

### Phase 4 — Consciousness Stream (The Core Innovation)

Goal: Continuous prefill loop with multi-track decode.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 4.1 | Consciousness frame abstraction | Merged inputs | Frame struct: tokens + metadata | Unit test: merge multiple input sources |
| 4.2 | Pulsed Prefill loop | Frames from input queue | Continuous KV state growth | Runs 1000 frames without OOM or crash |
| 4.3 | SSM state continuity | Consecutive frames | Recurrent state carries across frames | State divergence test: continuous vs reset |
| 4.4 | Multi-track Decode branches | Shared prefill prefix | Thinking, Speech, Action, Daydream outputs | All 4 branches produce tokens independently |
| 4.5 | Time-division multiplexer | Branch priorities | Scheduled execution on single GPU stream | Priority preemption works under load |
| 4.6 | Arbiter (branch merge) | Multiple branch outputs | Final decision: speak/act/think/dream | Decision reflects highest-priority actionable output |
| 4.7 | Wakefulness monitor | Input density metrics | Wakefulness level [0.0–1.0] | Level rises on input, decays on idle |
| 4.8 | P/D time-budget scheduler | Wakefulness level | Prefill/Decode GPU time split | Smooth ratio adjustment observable in metrics |

Milestone: System runs continuously for 10 minutes, processing injected text,
branching into thinking + speech outputs, with observable wakefulness transitions.

---

### Phase 5 — Cache Manager (Continuous Eviction)

Goal: KV Cache grows indefinitely via SSD overflow + importance-based eviction.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 5.1 | Block pool + block tracker | — | GPU block alloc with location tracking | 14 GB pool, alloc/free/query |
| 5.2 | KV Swapper (GPU ↔ SSD) | Full blocks | Swap-out writes to SSD, swap-in restores | Bit-exact round-trip |
| 5.3 | SSM/Conv state snapshots | Recurrent states | `.ssm` / `.conv` files alongside KV blocks | State restore after snapshot produces same output |
| 5.4 | Importance scorer | Attention weights per block | Cumulative importance scores | Low-attention blocks ranked lowest |
| 5.5 | Continuous eviction loop | Budget pressure | Least-important blocks swapped to SSD | System sustains 256K+ tokens without OOM |
| 5.6 | Eviction-triggered consolidation hook | Block about to be evicted | Event to SomniumConsolidator (stub) | Hook fires, metadata captured |
| 5.7 | Prefix cache (hash-based) | Repeated prefixes | Cache hit skips re-computation | Hit rate > 0 on repeated prompts |

Milestone: Consciousness stream runs for 1 hour, KV cache overflows to SSD and
recovers gracefully, no memory leak.

---

### Phase 6 — Multimodal Perception (ASR + Vision + Text)

Goal: System hears, sees, and reads — all feeding the consciousness stream.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 6.1 | ASR engine (adapt from reference) | Audio PCM | Text tokens | WER < 10% on test set |
| 6.2 | Mel-spectrogram GPU kernel | PCM samples | 128-bin mel features | Match librosa reference |
| 6.3 | VAD integration | Continuous audio stream | Speech segments | Correct segmentation on test audio |
| 6.4 | Speaker encoder (CAM++) | Audio segment | Speaker embedding | Known-speaker identification accuracy > 90% |
| 6.5 | Vision encoder (ViT) | Image / video frame | Vision tokens | Match Python ViT output for test image |
| 6.6 | Adaptive frame sampler | Camera stream | Sampled frames | 1-2 fps idle, burst on scene change |
| 6.7 | Text input channel | WebSocket text messages | Tokens in Prefill queue | Round-trip: send text → appears in consciousness |
| 6.8 | Input merger | ASR + Vision + Text + internal | Unified consciousness frame | All modalities appear in frame |

Milestone: Speak to the system, it hears (ASR → text in consciousness), show it
an image, it sees (ViT → vision tokens). Text input works via WebSocket.

---

### Phase 7 — Voice Output (TTS)

Goal: System speaks with consistent persona voice.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 7.1 | TTS model forward (adapt from reference) | Text | Codec tokens | Match reference output |
| 7.2 | Codec tokenizer decoder | Codec tokens | PCM audio (24 kHz) | Intelligible speech |
| 7.3 | Streaming TTS pipeline | Speech decode branch output | Streaming PCM via WebSocket | First-packet < 200ms |
| 7.4 | Persona voice config | persona.conf voice settings | Consistent speaker identity | Same voice across all outputs |
| 7.5 | Download TTS Tokenizer model | — | `~/models/dev/tts/Qwen/Qwen3-TTS-Tokenizer-12Hz` | Model files present |

Milestone: System speaks responses aloud through WebSocket audio stream.

---

### Phase 8 — Long-Term Memory (Memoria Longa)

Goal: Episodic recall + semantic knowledge graph, consolidated during dreams.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 8.1 | HNSW index (from scratch, C++) | Vectors dim=5120 | Approximate nearest neighbors | Recall@10 > 95% on 10K vectors |
| 8.2 | Episodic store (SSD-backed) | {embedding, text, timestamp, ...} | Insert, search, delete | 100K records, search < 5ms |
| 8.3 | CSR semantic graph | Entities + relations | Insert node/edge, traverse | 10K nodes, 2-hop traversal < 1ms |
| 8.4 | Graph traversal (bounded) | Seed nodes, hop limit, time budget | Top-K paths | Respects hop/time constraints |
| 8.5 | Hybrid retriever | Current context embedding | Episodic + graph results merged | Relevant memories surface |
| 8.6 | Prefill injection | Retrieved memories | Context prepended to next frame | Memories influence generation |

Milestone: Ask "what did I say earlier?" — system retrieves from episodic store.

---

### Phase 9 — Dreaming & Consolidation (Somnium)

Goal: Low-wakefulness states trigger memory consolidation and creative exploration.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 9.1 | Wakefulness-driven dream trigger | Wakefulness < threshold | Fork 4–16 daydream decode branches | Branches diverge from consciousness |
| 9.2 | Episodic compression | Similar episodic records | Merged/pruned records | Store size decreases after consolidation |
| 9.3 | Graph maintenance | Decayed edges, co-activation | Pruned weak edges, strengthened active | Graph stats change measurably |
| 9.4 | Eviction backlog processing | Deferred eviction events | Consolidated summaries in episodic store | No backlog leak |
| 9.5 | Re-embedding (oldest records) | Old embeddings + current model | Refreshed embeddings | HNSW search quality stable over time |

Milestone: Leave system idle for 30 minutes — it dreams, consolidates memory,
wakes up with organized knowledge.

---

### Phase 10 — Persona & Expression (Persona + Arbiter)

Goal: Entity has a rich inner world and autonomously shapes its outer expression.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 10.1 | Inner world state tracker | All decode branch outputs | Internal state representation | State reflects recent thought/dream content |
| 10.2 | Outer expression shaper | Inner state + context | Tone, style, content decisions | Expression adapts to context (formal/casual) |
| 10.3 | Persona config loader | `persona.conf` | Personality traits, voice style | Config changes reflected in behavior |
| 10.4 | Arbiter integration | Branches + persona | Final action/speech output | Coherent responses aligned with persona |

---

### Phase 11 — Tool Use (Instrumenta)

Goal: Entity can discover, invoke, and create tools.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 11.1 | MCP client (JSON-RPC over stdio/SSE) | Tool server URL | Available tools list | Connects to test MCP server |
| 11.2 | Function calling parser | LLM output with tool calls | Structured invocation | Correct parameter extraction |
| 11.3 | Async tool executor | Tool call | Result merged into future Prefill frame | Consciousness continues during tool execution |
| 11.4 | Tool registry | Multiple tool sources | Unified tool catalogue | Discovery across MCP + local skills |

---

### Phase 12 — WebUI & Nexus (External Interface)

Goal: Full observable dashboard, realtime audio/video, consciousness visualization.

| # | Task | Input | Output | Validation |
|---|------|-------|--------|------------|
| 12.1 | WebSocket server | — | Audio/state/control channels | Bidirectional communication works |
| 12.2 | HTTP REST endpoints | — | /health, /api/state, /api/memory, /api/config | All endpoints respond correctly |
| 12.3 | WebUI consciousness panel | State stream JSON | Real-time consciousness visualization | Updates at frame rate |
| 12.4 | Decode branch viewer | Branch state stream | All 4 branches visible with outputs | Branches update in real-time |
| 12.5 | Audio upstream/downstream | Mic PCM / TTS PCM | Bidirectional audio streaming | < 500ms round-trip latency |
| 12.6 | Video upstream | Camera frame | Frame visible in WebUI | < 1 fps display |
| 12.7 | Performance dashboard | perf_stats | GPU util, memory, KV occupancy, eviction rate | All metrics accurate |

---

### Phase Summary

| Phase | Name | Depends On | Estimated Complexity |
|-------|------|------------|---------------------|
| 0 | Foundation | — | Low |
| 1 | GPTQ-Int4 Kernel | Phase 0 | **High** (new kernel development) |
| 2 | Single Layer Forward | Phase 0, 1 | Medium |
| 3 | Full Model + Generation | Phase 2 | Medium |
| 4 | Consciousness Stream | Phase 3 | **High** (core innovation) |
| 5 | Cache Manager | Phase 3 | Medium-High |
| 6 | Multimodal Perception | Phase 4 | Medium (adapt from reference) |
| 7 | Voice Output (TTS) | Phase 4 | Medium (adapt from reference) |
| 8 | Long-Term Memory | Phase 4, 5 | **High** (HNSW + graph from scratch) |
| 9 | Dreaming & Consolidation | Phase 4, 5, 8 | High |
| 10 | Persona & Expression | Phase 4 | Medium |
| 11 | Tool Use | Phase 4 | Medium |
| 12 | WebUI & Nexus | Phase 4 | Medium |

Phases 6–12 are partially parallelizable after Phase 4 is stable.

### Decision

Begin with Phase 0. First commit target: CMake scaffold + SafetensorsLoader +
Tokenizer + Allocator.
