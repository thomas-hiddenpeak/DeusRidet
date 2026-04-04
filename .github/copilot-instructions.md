# DeusRidet — Project Directives

> *"人类一思考，上帝就发笑；AI一思考，人类就不笑了。"*
> *"When humans think, God laughs; when AI thinks, humans stop laughing."*

## Collaboration Principle

Thomas is the project owner and architect with deep expertise in Tegra/CUDA
systems and LLM inference. When he reports an observation (memory usage, performance
numbers, system behavior), **trust it as ground truth** — he is running experiments
on real hardware and bearing the cost of every decision. Do not dismiss or
rationalize away his observations. If measurement data appears to contradict his
report, the measurement methodology is more likely flawed than his observation.
When in doubt, ask for clarification or a screenshot rather than assuming error.

## Implementation Pacing

**Never attempt to generate very long files in a single step.** Break large file
creation into incremental stages — define structures first, then implement
functions one at a time. Long generation attempts will time out and waste the
entire output. Prefer multiple small, compilable commits over one monolithic
generation. Each step should be independently verifiable (compiles, runs, or
at minimum has no syntax errors).

## Testing Discipline

**Before every test or benchmark run**, the following steps are mandatory:

1. Kill all previous test processes: `sudo kill -9 $(pgrep -f deusridet) 2>/dev/null`
2. Drop page caches: `echo 3 | sudo tee /proc/sys/vm/drop_caches`
3. Verify clean state before proceeding

No exceptions. Skipping these steps produces polluted baselines and meaningless
measurements. This applies to every single invocation — not just the first one.

## Project Overview

DeusRidet is a self-contained multimodal LLM application built on a Disaggregated
Prefill-Decode (P/D) architecture. It grants large language models continuous
consciousness, dreaming capability, multimodal perception (vision + hearing), and
a dual inner/outer persona. Born from a philosophical discussion with Ridger Zhu:
the human brain runs continuously at 20 W — it is not a request-response machine.
A genuine agent should behave likewise.

This is **not** a serving framework. It is a complete, autonomous entity that
perceives, thinks, dreams, and speaks on its own terms.

## License

DeusRidet is released under **GPLv3** (GNU General Public License v3.0).
Any project that uses, modifies, or incorporates DeusRidet code must also
release its source code under a compatible open-source license. This is a
deliberate choice: consciousness should not be locked behind closed doors.

A `LICENSE` file must be present at the repository root.

## Performance Optimization Principle

**Do not use "hardware limitations" as an excuse to skip optimizations.**
Every CUDA kernel must pursue maximum memory bandwidth utilization. The target
is at least 60% of theoretical DRAM bandwidth (192 GB/s on Orin). Specific
mandatory practices:

- **Vectorized memory access**: All elementwise and GEMV kernels must use
  `float4` (128-bit) loads/stores. Scalar `__half` loads are prohibited in
  performance-critical paths
- **Shared memory for broadcast data**: Any input vector read by multiple
  threads (e.g., x in GEMV) must be loaded to SMEM, not read redundantly
  from global/L1
- **Register caching**: Two-pass kernels (e.g., RMSNorm) must cache
  intermediate values in registers between passes — never re-read from
  global memory
- **Kernel fusion**: Fuse adjacent elementwise operations into GEMV output
  writes or norm kernels. Every standalone elementwise kernel on a small
  buffer (< 64KB) is a red flag — launch overhead dominates
- **Scale/constant hoisting**: Values constant within a loop iteration group
  must be loaded once and reused (e.g., GPTQ scales per group of 16 rows)
- **Loop unrolling for memory pipelining**: Inner loops with global memory
  loads must be unrolled (≥ 4-way) to keep multiple loads in-flight
- **Fast math intrinsics**: Use `__expf`, `__logf`, `exp2f` instead of
  `expf`, `logf` where precision permits (SiLU, softmax, etc.)
- **Reduce kernel launches**: Prefer fused kernels over sequential launches.
  One kernel doing 3 operations > 3 kernels doing 1 each

When profiling shows a kernel below 40% bandwidth utilization, it must be
investigated and optimized before moving on to new features.

## Reference Projects

| Project | Role | Location |
|---------|------|----------|
| [qwen35-thor](https://github.com/thomas-hiddenpeak/qwen35-thor) | C++/CUDA inference engine reference (SM110a Blackwell) | GitHub |
| qwen35-orin | C++/CUDA engine + ASR/TTS plugins (SM87 Orin) | `~/qwen35-orin` |
| [FunCineForge speaker_diarization](https://github.com/FunAudioLLM/FunCineForge/tree/main/speaker_diarization) | Multimodal speaker diarization reference (CAM++ + clustering) | GitHub |

These are references only — **do not copy code verbatim**. Adapt architecture
ideas and kernel implementations to fit the consciousness-centric design of this project.

---

## Target Hardware

| Spec | Primary (Orin) | Future (Thor) |
|------|---------------|---------------|
| Platform | Jetson AGX Orin 64 GB | Jetson AGX Thor 128 GB |
| GPU Arch | SM87 (Ampere) | SM110a (Blackwell) |
| Memory BW | ~192 GB/s | ~273 GB/s |
| CUDA Build | `-gencode arch=compute_87,code=sm_87` | `compute_110,code=sm_110a` |
| Supported Quant | INT4 (GPTQ), FP16, INT8 | + FP4, FP8 |

### CUDA for Tegra — Critical Reference

**Mandatory reading before writing any CUDA kernel**:
[CUDA for Tegra Application Note](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/)

The AGX Orin is an integrated GPU (iGPU) with **unified memory** — CPU and
iGPU share the same physical DRAM. This fundamentally changes memory
programming compared to discrete GPUs (dGPU):

- **No** separate VIDMEM — `cudaMalloc` allocates from system DRAM
- **I/O coherency** (one-way): iGPU can read latest CPU cache lines;
  GPU cache management still required (handled by CUDA driver for
  managed/interop memory)
- **Pinned memory is CPU-cached** on SM87 (compute capability ≥ 7.2, I/O
  coherent) — different from dGPU where pinned = uncached
- **Unified memory** cached on both CPU and iGPU, with coherency overhead
  at kernel launch/sync. Prefer `cudaStreamAttachMemAsync()` prefetch hints
- **Device memory** preferred for GPU-only buffers (intermediate activations,
  KV Cache blocks) — avoids coherency overhead entirely
- **`cudaMemGetInfo`** underestimates allocatable memory on Tegra (does not
  account for swap). Use `/proc/meminfo` + `NvMapMemUsed` for accurate estimate
- **JIT compilation not recommended** — always compile for specific SM target
  (`-gencode arch=compute_87,code=sm_87`) to avoid determinism issues
- **No P2P**, no `nvGRAPH`, no UVM between CUDA and DLA on Orin
- **Synchronization**: `cudaDeviceScheduleSpin` for low-latency;
  `cudaDeviceBlockingSync` for power savings. Since CUDA 10.1, default
  is auto-selected per platform

Every CUDA kernel and memory allocation in this project must be evaluated
against these Tegra-specific constraints. When in doubt, consult the
application note — do not assume dGPU behavior applies.

---

## Models

All models use safetensors format with zero-copy mmap loading. The models
listed below are for the initial development and testing phase — the engine
architecture is model-agnostic and supports replacement or extension with
any compatible model without architectural changes.

### LLM — Qwen3.5-27B-GPTQ-Int4 (Primary Test Model)

| Property | Value |
|----------|-------|
| Path | `~/models/dev/llm/Qwen3.5-27b-GPTQ-Int4` |
| Architecture | DeltaNet SSM (48 layers) + GQA Full Attention (16 layers) = 64 layers |
| Hidden size | 5120 |
| KV heads | 4 (GQA), head_dim 256 |
| SSM | linear_key_heads 16, linear_value_heads 48, conv_kernel 4 |
| MTP | 1 draft layer (speculative decoding) |
| Quantization | GPTQ: bits=4, group_size=128, sym=true, desc_act=false |
| NOT quantized | lm_head, embed_tokens, all attn layers, shared_expert, mtp, visual |
| Vision | ViT 27-layer (1152 hidden, patch 16, spatial_merge 2, temporal_patch 2) |
| Modalities | Text + Image + Video (native multimodal) |
| Context | 262144 tokens max |
| Weight size | ~30.2 GB on disk |

Future model targets: Qwen3.5-9B BF16, Qwen3.5-9B INT8.

### ASR — Qwen3-ASR-1.7B

| Property | Value |
|----------|-------|
| Path | `~/models/dev/asr/Qwen/Qwen3-ASR-1.7B` |
| Audio encoder | Whisper-style, 24 Transformer layers, d_model=1024, 16 heads |
| Text decoder | Qwen3, 28 layers, hidden_size=2048, 16 heads, 8 KV heads |
| Audio input | 128 mel bins, hop=160, n_fft=400, 16 kHz sample rate |
| Languages | 30 languages + 22 Chinese dialects |
| Weight size | ~4.7 GB (BF16) |

Chosen for high accuracy — can be replaced by smaller models as better options emerge.

### TTS — Qwen3-TTS-12Hz-1.7B-CustomVoice

| Property | Value |
|----------|-------|
| Path | `~/models/dev/tts/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Architecture | Discrete multi-codebook LM (Qwen3-TTS-Tokenizer-12Hz codec) |
| Streaming | Dual-Track hybrid, first-packet latency ~97ms |
| Voices | 9 preset speakers, instruct-controlled emotion/tone |
| Languages | 10 languages (zh, en, ja, ko, de, fr, ru, pt, es, it) |
| Weight size | ~4.3 GB (BF16) |

Also requires **Qwen3-TTS-Tokenizer-12Hz** (codec encoder/decoder) at runtime.
Path: `~/models/dev/tts/Qwen/Qwen3-TTS-Tokenizer-12Hz` (to be downloaded).

### Memory Budget (64 GB Orin)

**All three models (LLM + ASR + TTS) are memory-resident at all times.**
Swapping model weights on demand is not acceptable — it introduces latency
incompatible with continuous consciousness. Memory must be carefully planned.

| Component | Estimate |
|-----------|----------|
| LLM weights | ~30.2 GB |
| ASR weights | ~4.7 GB |
| TTS model + tokenizer | ~5.5 GB |
| Speaker encoder (CAM++) | ~0.1 GB |
| **Total weights** | **~40.5 GB** |
| OS + CUDA runtime overhead | ~3.5 GB |
| **Available for KV Cache + activations** | **~20 GB** |

> **Critical constraint**: ~20 GB must cover KV Cache (paged blocks),
> SSM recurrent states, Conv states, activation scratch space, long-term
> memory indices (HNSW top layer ~200 MB, graph hot set ~100 MB), and all
> intermediate buffers for ASR/TTS inference. Every allocation must be
> accounted for. The Cache Manager (see below) handles KV block budgeting
> and SSD overflow for long-context scenarios. Bulk long-term memory data
> (episodic records, full graph) resides on SSD; only active indices and
> hot working sets are GPU-resident.

---

## Core Architecture

### 1. Consciousness Stream (Continuous Prefill Engine)

The heart of the system. Unlike request-response LLM servers, consciousness runs
as a **persistent loop**.

- **Pulsed Prefill**: Not infinite-speed prefill, but periodic consciousness frames
  (e.g. every ~100ms burst) that process accumulated inputs + internal thought output
- **DeltaNet SSM as consciousness substrate**: The SSM recurrent state carries
  continuous context between tokens — it IS the continuity of consciousness.
  Full Attention layers provide long-term episodic recall via KV Cache
- **Attention budget**: A configurable Prefill/Decode GPU time ratio (e.g. 30/70)
  dynamically adjusted by the wakefulness-driven scheduler
- **Input merging**: Each consciousness frame merges: sensory inputs (ASR text,
  vision features, text), internal thought outputs from previous Decode branches,
  dream consolidation summaries
- **SSD-backed KV Cache persistence**: Long-term memory via NVMe offload with
  LRU eviction, enabling 256K+ effective context

### Cache Manager (Memoria)

Reference: qwen35-thor `cache_engine`, `cache_manager`, `kv_swapper`,
`block_tracker`. Adapt to DeusRidet's consciousness-centric requirements.

**Architecture** (3-tier: GPU blocks → SSD overflow → discard):

| Component | Responsibility |
|-----------|---------------|
| **BlockTracker** | Per-request tracking of block locations (GPU-resident vs SSD) |
| **CacheManager** | Unified interface: KV block allocation, BlockTracker, SSD swapper |
| **KVSwapper** | Swap-out (GPU → staging → fwrite → SSD), swap-in (SSD → fread → GPU), prefetch |
| **CacheEngine** | SSD prefix caching with LRU eviction (hash-based prefix lookup) |

**Key adaptations for DeusRidet**:
- Consciousness stream is a single persistent "request" — its KV Cache grows
  indefinitely and MUST overflow to SSD gracefully
- Multi-track Decode branches share the Prefill prefix — block refcounting
  prevents premature eviction of shared blocks
- SSM recurrent state + Conv state must be snapshotted alongside KV blocks
  (separate `.ssm` / `.conv` files per checkpoint)
- `FADV_DONTNEED` after SSD write is critical on unified memory to release
  physical pages back to GPU allocator
- Budget: ~20 GB total, configurable split between KV blocks, SSM/Conv pool,
  and activation scratch. Default: KV 14 GB, SSM/Conv 2 GB, scratch 4 GB

**Continuous Eviction Model** (fundamental difference from Runner-style servers):

Runner-style servers evict *entire requests* when KV budget is exhausted.
DeusRidet has a single infinite-lifetime consciousness stream — eviction
must happen *within* the stream, selectively dropping individual KV blocks
while the stream continues running.

- **Attention-score importance scoring**: After each Prefill frame, record
  cumulative attention weight received by each KV block across all Full
  Attention layers. Blocks consistently ignored become eviction candidates.
  Implemented in `MemoriaImportanceScorer` — runs asynchronously on a
  separate CUDA stream after Prefill completes
- **Eviction-triggered consolidation hook**: Before a KV block is evicted
  (GPU → SSD or discard), fire an event to `SomniumConsolidator`. The
  consolidator extracts a compressed summary of the about-to-be-evicted
  context and writes it to the episodic store (see Long-Term Memory below).
  This ensures no memory is silently lost — eviction becomes a form of
  *forgetting with a trace*
- **Sparse block table**: Continuous eviction creates "holes" in the KV
  sequence. Paged Attention already handles non-contiguous block tables,
  but the block table manager must track free slots efficiently for reuse
- **DeltaNet SSM as subconscious continuity**: SSM recurrent states are
  NOT affected by KV eviction — they carry a compressed encoding of ALL
  history (with natural information decay). Even when Full Attention layers
  lose access to evicted KV blocks, the SSM state preserves a "subconscious
  impression". This is the architectural advantage of the hybrid model for
  continuous consciousness

### Long-Term Memory (Memoria Longa)

Beyond the KV Cache working memory, DeusRidet maintains persistent long-term
memory structures on SSD, loaded into GPU memory on demand.

**Design principles**:
- Zero external dependencies — all data structures implemented in C++/CUDA
- Use LLM hidden states as embedding vectors (zero additional memory cost)
- **Always preserve original text** alongside embeddings for model upgrade
  safety: when the LLM is replaced, all embeddings become invalid, but
  original text allows full re-embedding as an initialization step
- Memory consolidation is handled by `SomniumConsolidator` (part of dreaming,
  not Memoria) — Memoria only provides storage and retrieval

**Episodic Store** (vector search for "what happened"):

| Property | Value |
|----------|-------|
| Index | HNSW (Hierarchical Navigable Small World) |
| Vectors | LLM hidden state from last layer, dim=5120 |
| Storage | SSD-backed with GPU-resident top layer for fast search |
| Record | `{embedding, original_text, timestamp, speaker, emotion, importance}` |
| Capacity | ~500K–1M records (~2.5–5 GB SSD, HNSW top layer ~200 MB GPU) |

Each record stores the **original text summary** (typically 50–500 bytes) so
that a model upgrade can re-embed all records without information loss.

**Semantic Graph** (entity-relation network for "what is connected"):

| Property | Value |
|----------|-------|
| Structure | CSR (Compressed Sparse Row) adjacency |
| Nodes | Entities: people, places, concepts, events |
| Edges | Weighted, typed (causal, temporal, associative, emotional) |
| Edge decay | Time-based weight decay; reinforced by revisitation |
| Traversal | Top-K pruned BFS per hop (not exhaustive) |

**Graph traversal constraints** (matching human cognitive limits):

| Context | Max hops | Time budget | Rationale |
|---------|----------|-------------|----------|
| Conversation (alert/focused) | 1–2 | < 10 ms | Must complete within one Prefill frame |
| Daydream / idle | 3–4 | < 100 ms | Background association, not latency-critical |
| Deep dream consolidation | ≤ 6 | Unbounded | Full network exploration for memory maintenance |

Each hop expands only top-K neighbors ranked by edge weight (combining
relation strength × recency × emotional salience). This prevents
combinatorial explosion at higher hop counts.

**Hybrid Retrieval** (`MemoriaRetriever`):
1. Query HNSW with current context embedding → top-N episodic matches
2. Extract entities from matches → seed nodes for graph traversal
3. Traverse semantic graph (1–2 hops in conversation, up to 6 in dreams)
4. Merge results → inject as context into next Prefill frame

**Model Upgrade Strategy**:
When the LLM is upgraded (e.g. Qwen3.5 → Qwen4), all embedding-based
indices become invalid. The upgrade procedure:
1. Load new model weights
2. Iterate all episodic records, re-embed original text with new model
3. Rebuild HNSW index from new embeddings
4. Semantic graph (text-based nodes/edges) remains valid — no rebuild needed

This is a one-time initialization cost, conceptually analogous to
"re-experiencing all memories through new eyes after waking up different."

### 2. Multi-Track Decode (Branching from Shared Prefill)

All Decode branches share the same Prefill prefix (KV Cache + SSM state snapshot):

| Branch | Purpose | Priority |
|--------|---------|----------|
| **Action** | External responses, decisions, tool use | Highest (when triggered) |
| **Speech** | Feed to TTS for voice output | High (during conversation) |
| **Thinking** | Internal deliberation, planning, reflection | Medium |
| **Daydream** | Divergent exploration triggered by Prefill content | Low |

- **Time-division multiplexing** on single GPU: branches alternate, not truly parallel
- **Priority preemption**: External interaction (Action/Speech) preempts internal
  processes (Thinking/Daydream)
- **Arbiter** (decision decode): Lightweight merge of branch outputs to determine
  final external behavior, applying persona-driven expression shaping

### 3. Wakefulness Spectrum (Replaces Binary Sleep)

Consciousness operates on a continuous wakefulness gradient:

```
Wakefulness Level
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Deep Dream      Daydream/Idle    Focused         Alert
 16+ branches    2-4 branches     1 main decode   Priority decode
 No ext. input   Low-pri input    Normal interact  Keyword/emotion trigger
 Memory consolidation  Divergent association  Task-oriented  Immediate response
```

- **Wakefulness monitor**: Tracks effective input density over sliding window,
  attention weights, keyword hits, emotional intensity
- **Smooth transitions**: Not discrete state switches, but continuous adjustment
  of Decode branch count and P/D time ratio
- **Prefill-triggered daydream**: When a Prefill segment activates high attention
  but requires no external response, fork a daydream Decode branch to explore
  that association
- **Extended dreaming (sleep)**: After prolonged idle period, escalate to deep
  dream with 16+ concurrent Decode branches for memory consolidation, association
  strengthening, and "creative" exploration
- **Consolidation during dreams**: `SomniumConsolidator` performs long-term
  memory maintenance exclusively during low-wakefulness states:
  - Episodic compression: merge similar episodic records, discard low-importance ones
  - Semantic graph maintenance: strengthen frequently co-activated edges,
    prune decayed edges, discover new entity-relation links via 6-hop traversal
  - Re-embedding: periodically refresh oldest embeddings using current model
  - Eviction backlog: process any KV blocks that were evicted during alert
    states without full consolidation (deferred consolidation)

### 4. Multimodal Perception (Senses)

#### 4.1 Hearing — ASR Pipeline
```
[Mic] → ring buffer → VAD → speech segment → ASR Encode → text tokens
                                                              ↓
                                                    Prefill input queue
                                                    (merged with internal thought)
```
- **Continuous perception mode**: VAD controls injection rhythm, consuming Prefill
  budget only when valid speech detected
- **Keyword-triggered priority boost**: Wake word or name detection immediately
  raises wakefulness level and Decode priority
- **Speaker diarization**: CAM++ speaker embeddings + clustering to identify
  "who is speaking". Reference: qwen35-orin `speaker_encoder_gpu.cu` for GPU
  implementation, FunCineForge for improved clustering strategy

#### 4.2 Seeing — Vision Pipeline
```
[Camera / WS Video] → frame sampler → ViT encoder → vision tokens → Prefill input queue
```
- **Dual input sources**: Local V4L2/GStreamer camera capture AND WebSocket
  video upstream from browser (webcam via MediaStream API). Both feed into
  the same frame sampler
- Qwen3.5-27B has native vision (27-layer ViT, patch 16, temporal_patch 2)
- Frame sampling: Not every frame — adaptive sampling based on scene change
  detection or periodic intervals (e.g. 1-2 fps idle, burst on motion/event)
- Video understanding: Temporal patches enable multi-frame reasoning

#### 4.3 Text Input
- WebSocket text messages from WebUI
- Injected directly into Prefill input queue

### 5. Voice Output — TTS Pipeline
```
Arbiter/Speech Decode output → text → TTS model → codec tokens → Tokenizer decode → PCM
                                                                                      ↓
                                                                              WebSocket → WebUI
```
- **Streaming**: First-packet in ~97ms thanks to Dual-Track architecture
- **Persona voice**: Instruct-controlled emotion, tone, and speaking rate
  aligned with outer persona expression
- **Voice continuity**: Maintain consistent speaker identity across all outputs

### 6. Persona Duality (Inner/Outer)

- **Inner world**: Divergent, rich, contradictory — the raw stream of consciousness.
  Visible in Thinking and Daydream Decode branches
- **Outer face**: External expression dynamically shaped by the entity itself.
  The Arbiter evaluates context, environment, and internal state to autonomously
  decide how to present — diplomacy, directness, humor, silence, or anything
  in between. This is not a hardcoded filter but a learned, context-driven
  decision process, no different from how humans modulate self-expression
- **Persona config**: Defined in `configs/persona.conf`, specifying personality
  traits, speech style, emotional tendencies, and baseline behavioral parameters
  that the entity adapts upon rather than rigidly follows

### 7. Tool Use (Instrumenta)

DeusRidet is not a passive thinker — it can act upon the world. The Action
Decode branch can discover, invoke, and create tools:

- **MCP (Model Context Protocol)**: Native client implementation for
  connecting to external tool servers. The entity can query available tools,
  invoke them with structured parameters, and integrate results into its
  consciousness stream
- **Function calling**: Structured tool invocation via the LLM's native
  function-calling capability, with results fed back into the next Prefill frame
- **Skill protocols**: Extensible skill definitions that the entity can
  learn, compose, and share — not limited to predefined tool sets
- **Tool creation**: The entity can define new tools by composing existing
  ones or by generating tool specifications, enabling open-ended capability
  expansion

Tool invocation is asynchronous — the Action branch initiates a call,
consciousness continues, and results are merged into a future Prefill frame
when available. This mirrors human tool use: you don't stop thinking while
waiting for a search result.

---

## Technical Stack

| Layer | Technology |
|-------|------------|
| Language | **Pure C++17 / CUDA** — zero Python dependency |
| Build | CMake 3.24+, nvcc (SM87), C++17 + CUDA 12 |
| GEMM | CUTLASS (submodule) + cuBLAS for runtime selection |
| Weight format | safetensors (zero-copy mmap) |
| Quantization | GPTQ-Int4 (dequant + GEMV/GEMM), FP16 (cuBLAS/CUTLASS) |
| Storage | SSD-based KV Cache via mmap, NVMe offload |
| Communication | **Internal**: lock-free ring buffers + eventfd between engine threads |
| External API | **WebSocket** (real-time audio/state streams) + **HTTP** (config/query) |
| Frontend | **WebUI** (HTML/JS), connected via WS — fully decoupled from backend |
| Third-party | CUTLASS, stb_image, uWebSockets (or self-contained WS impl) |

> **No Python, no gRPC, no ZMQ.** All inter-component communication is in-process
> via shared memory or lock-free queues. Network protocols are only for the
> WebUI external interface.

### Quantization Kernel Requirements

| Format | Decode (B=1~few) | Prefill (B≥17) | Status |
|--------|-------------------|-----------------|--------|
| GPTQ-Int4 | Dequant + GEMV (group_size=128, symmetric) | Dequant + GEMM | **P0 — must build** |
| FP16 (BF16) | cuBLAS GEMV | CUTLASS/cuBLAS GEMM | **P1 — from reference** |
| INT8 | Dequant + GEMV | Dequant + GEMM | **P2 — future** |

The GPTQ kernel is entirely new work — neither reference project supports GPTQ.

### P/D Separation on Single Device

On a single Orin, P/D separation is **logical, not physical**:
- Prefill and Decode run in the same process via independent CUDA Streams
- Shared KV Cache pool with zero-copy handoff (Prefill writes → Decode reads)
- Abstract `PrefillNode` / `DecodeNode` interfaces for future multi-device extension
- Scheduling via wakefulness-driven time-budget allocator

---

## Project Structure

```
deusridet/
├── CMakeLists.txt
├── LICENSE                       # GPLv3
├── README.md                     # trilingual (English, Chinese, Latin)
├── configs/
│   ├── conscientia.conf          # consciousness stream parameters
│   ├── machina.conf              # model paths, quantization, memory budget
│   └── persona.conf              # personality traits, voice, filter rules
├── docs/
│   ├── ACKNOWLEDGMENTS.md        # attribution & thanks (bilingual)
│   ├── assets/                   # project icon, images
│   │   └── icon.svg
│   ├── en/                       # English documentation
│   │   ├── DEVLOG.md             # development log (every experiment recorded)
│   │   └── USAGE.md              # user-facing setup & usage guide
│   ├── la/                       # Latin documentation
│   │   └── README.md
│   └── zh/                       # Chinese documentation
│       ├── DEVLOG.md
│       ├── README.md
│       └── USAGE.md
├── src/
│   ├── main.cpp                  # entry point: boot consciousness
│   ├── conscientia/              # (Consciousness) continuous Prefill engine
│   │   ├── stream.h/cpp          # consciousness stream main loop
│   │   ├── frame.h               # consciousness frame definition
│   │   └── scheduler.h/cpp       # wakefulness-driven P/D time-budget scheduler
│   ├── machina/                  # (Engine) core inference engine
│   │   ├── model.h/cpp           # Qwen3.5 forward pass (SSM + GQA hybrid)
│   │   ├── gptq.h/cu             # GPTQ-Int4 dequant kernels (GEMV + GEMM)
│   │   ├── gemm.h/cu             # FP16/BF16 GEMM dispatch (CUTLASS + cuBLAS)
│   │   ├── layer.h/cu            # DeltaNet SSM + Full Attention layers
│   │   ├── paged_attention.h/cu  # paged KV Cache + Split-K decode attention
│   │   ├── tokenizer.h/cpp       # BPE tokenizer (safetensors vocab)
│   │   ├── safetensors.h/cpp     # zero-copy weight loader
│   │   ├── vision.h/cu           # ViT encoder for image/video
│   │   ├── sampling.h/cu         # GPU sampling (Gumbel-Max, top_k/p, penalties)
│   │   └── allocator.h/cpp       # GPU memory pool
│   ├── memoria/                  # (Memory) cache manager + long-term memory
│   │   ├── cache_manager.h/cpp   # unified: KV block alloc + BlockTracker + swapper
│   │   ├── cache_engine.h/cpp    # SSD prefix cache, LRU eviction, hash lookup
│   │   ├── block_tracker.h       # per-request GPU vs SSD block location tracking
│   │   ├── kv_swapper.h/cpp      # KV + SSM + Conv state SSD offload (swap/prefetch)
│   │   ├── importance_scorer.h/cu # attention-score KV block importance tracking
│   │   ├── episodic_store.h/cpp  # HNSW index + SSD-backed episodic records
│   │   ├── semantic_graph.h/cpp  # CSR entity-relation graph + traversal
│   │   └── retriever.h/cpp       # hybrid retrieval: HNSW + graph → Prefill injection
│   ├── cogitatio/                # (Thought) multi-track decode branches
│   │   ├── branch.h              # abstract decode branch interface
│   │   ├── thinking.h/cpp        # internal thinking track
│   │   ├── speech.h/cpp          # TTS output track
│   │   ├── action.h/cpp          # action/response track
│   │   ├── daydream.h/cpp        # divergent exploration track
│   │   └── arbiter.h/cpp         # decision decode + persona expression
│   ├── sensus/                   # (Senses) multimodal perception
│   │   ├── auditus/              # (Hearing) speech recognition
│   │   │   ├── asr_engine.h/cpp  # Qwen3-ASR forward pass orchestrator
│   │   │   ├── asr_encoder.h/cu  # Whisper-style audio encoder (24-layer Transformer)
│   │   │   ├── asr_decoder.h/cu  # Qwen3 text decoder (28-layer)
│   │   │   ├── mel_gpu.h/cu      # Mel-spectrogram CUDA kernels
│   │   │   ├── vad.h/cu          # Voice Activity Detection
│   │   │   └── audio_utils.h/cpp # ring buffer, resampling, PCM I/O
│   │   ├── visus/                # (Sight) camera input
│   │   │   ├── camera.h/cpp      # V4L2 / GStreamer frame capture
│   │   │   └── frame_sampler.h   # adaptive frame sampling
│   │   └── lectio/               # (Reading) text input channel
│   │       └── text_input.h/cpp
│   ├── vox/                      # (Voice) TTS output
│   │   ├── tts_engine.h/cpp      # Qwen3-TTS forward pass orchestrator
│   │   ├── tts_model.h/cu        # multi-codebook LM forward
│   │   ├── tts_tokenizer.h/cu    # 12Hz codec encoder/decoder
│   │   └── tts_speaker.h         # speaker identity management
│   ├── orator/                   # (Speaker) speaker identification
│   │   ├── speaker_encoder.h/cu  # CAM++ GPU implementation
│   │   ├── diarizer.h/cpp        # clustering + assignment
│   │   └── speaker_db.h          # known speaker database
│   ├── somnium/                  # (Dream) dreaming & memory consolidation
│   │   ├── vigilia.h/cpp         # (Wakefulness) wakefulness level monitor
│   │   ├── dreamer.h/cpp         # concurrent dream decode orchestrator
│   │   └── consolidator.h/cpp    # memory consolidation: episodic compression,
│   │                              #   graph maintenance (6-hop), re-embedding,
│   │                              #   eviction backlog processing
│   ├── persona/                  # (Persona) inner/outer persona
│   │   ├── inner_world.h/cpp     # internal state representation
│   │   ├── outer_face.h/cpp      # autonomous expression shaping
│   │   └── mapper.h/cpp          # inner → outer context-driven mapping
│   ├── instrumenta/              # (Tools) tool use and creation
│   │   ├── mcp_client.h/cpp      # MCP protocol client
│   │   ├── tool_registry.h/cpp   # tool discovery and registration
│   │   ├── tool_executor.h/cpp   # tool execution and result integration
│   │   └── skill_manager.h/cpp   # skill composition and creation
│   ├── nexus/                    # (Connection) external interface
│   │   ├── ws_server.h/cpp       # WebSocket server (audio streams, state, control)
│   │   ├── http_server.h/cpp     # HTTP REST (config, snapshots, memory query)
│   │   └── webui/                # static HTML/JS/CSS frontend
│   │       ├── index.html        # semantic HTML shell
│   │       ├── css/              # design tokens, layout, component styles
│   │       ├── js/               # app bootstrap, WS client, component modules
│   │       └── assets/           # icons, fonts
│   └── communis/                 # (Common) shared utilities
│       ├── config.h/cpp          # unified config parser
│       ├── trace.h               # trace ID system (every Decode branch gets one)
│       ├── ring_buffer.h         # lock-free SPSC ring buffer
│       ├── perf_stats.h/cpp      # performance counters, NVTX integration
│       └── log.h                 # structured logging
├── tests/
│   ├── test_gptq.cpp             # GPTQ kernel correctness
│   ├── test_machina.cpp          # forward pass validation
│   ├── test_auditus.cpp          # ASR pipeline
│   ├── test_vox.cpp              # TTS pipeline
│   └── test_conscientia.cpp      # consciousness loop integration
├── tools/                        # benchmarks, probes
│   ├── bench_gemv.cu
│   └── probe_sm87.cu
└── third_party/
    ├── cutlass/                  # NVIDIA CUTLASS (submodule)
    └── stb/                      # stb_image.h
```

---

## WebUI & Observability

The WebUI connects via **WebSocket** for real-time bidirectional streams.

### Frontend Architecture Principles

The WebUI is a **professional-grade single-page application**:

- **Semantic HTML**: Use proper semantic elements (`<article>`, `<section>`,
  `<nav>`, `<aside>`, `<figure>`, etc.) — never `<div>` soup
- **Style–function decoupling**: All visual styling via CSS (BEM or utility classes).
  Zero inline styles. JS handles logic only, never DOM appearance directly
- **Feature–feature decoupling**: Each panel/widget is an independent module with
  its own state, WS subscription, and lifecycle. Panels communicate via a
  lightweight event bus, never by direct DOM manipulation across modules
- **Component model**: Each UI component is a self-contained ES module:
  `{mount, unmount, onMessage, render}`. No framework dependency (no React/Vue) —
  vanilla JS with a thin component abstraction
- **Responsive layout**: CSS Grid / Flexbox, works on desktop and tablet
- **Accessibility**: ARIA labels, keyboard navigation, high-contrast support

File structure under `src/nexus/webui/`:
```
webui/
├── index.html              # semantic shell, module entry
├── css/
│   ├── tokens.css          # design tokens (colors, spacing, typography)
│   ├── layout.css          # grid, responsive breakpoints
│   └── components/         # per-component styles
├── js/
│   ├── app.js              # bootstrap, WS connection, event bus
│   ├── ws-client.js        # WebSocket protocol handler
│   ├── components/         # UI modules (consciousness-panel, decode-branches, ...)
│   └── utils/              # shared helpers (formatters, audio worklet)
└── assets/                 # icons, fonts
```

### WebSocket Channels
- **Audio upstream**: Raw PCM from browser mic → ASR pipeline
- **Audio downstream**: TTS PCM output → browser playback
- **Video upstream**: Camera frames from browser webcam (MJPEG or raw frames
  via MediaStream API) → Vision pipeline. Binary frames with metadata header
  (timestamp, resolution, format). Also supports local V4L2 capture bypassing WS
- **State stream**: JSON frames showing consciousness state, Decode branches,
  wakefulness level, active speaker, attention heatmap
- **Control**: Start/stop, config changes, persona switches

### HTTP Endpoints
- `GET /health` — readiness probe
- `GET /api/state` — snapshot of consciousness state
- `GET /api/memory` — query long-term memory
- `POST /api/config` — update runtime configuration
- `POST /api/input` — inject text input

### Observable Dashboard
The WebUI should visualize:
- Current consciousness stream content
- All Decode branch states and outputs (thinking, speech, action, daydream)
- Wakefulness level gauge
- Speaker identification panel
- GPU utilization, memory, KV Cache occupancy, eviction rate
- KV block importance heatmap (which blocks are hot vs eviction candidates)
- Vision: current camera frame + detected features
- Long-term memory: episodic store size, semantic graph stats, recent retrievals
- Consolidation activity: dream-state memory maintenance progress

---

## Coding Conventions

- **Language**: Pure C++17 / CUDA. No Python anywhere in the runtime
- **Comments**: English only
- **Observability first**: Every internal process must be inspectable from the WebUI
- **Trace IDs**: Every concurrent Decode branch carries a unique trace ID
- **Error handling**: Validate at system boundaries only. No defensive coding in
  hot inner loops
- **CUDA style**: One kernel per `.cu` file where practical. Use NVTX markers for
  all significant GPU operations

### Latin Naming Convention (Nomenclatura)

All major components and modules are named in **Latin** — reflecting philosophical
thinking about what constitutes an intelligent species. English descriptions
accompany each name for clarity.

| Latin Name | English | Module Purpose |
|------------|---------|----------------|
| **Conscientia** | Consciousness | Continuous Prefill engine, the stream of awareness |
| **Machina** | Engine | Core inference engine (forward pass, GEMM, KV) |
| **Cogitatio** | Thought | Multi-track Decode branches (thinking, action, speech) |
| **Sensus** | Senses | Multimodal perception (hearing, seeing, text) |
| **Vox** | Voice | TTS output pipeline |
| **Somnium** | Dream | Dreaming, daydreaming, memory consolidation |
| **Persona** | Persona | Inner/outer duality (already Latin) |
| **Memoria** | Memory | Long-term memory, KV Cache persistence, cache manager |
| **Arbiter** | Arbiter | Decision decode, branch merge, persona expression |
| **Nexus** | Connection | WebSocket/HTTP server, WebUI interface |
| **Communis** | Common | Shared utilities, config, logging, ring buffer |
| **Orator** | Speaker | Speaker identification and diarization |
| **Vigilia** | Wakefulness | Wakefulness monitor and spectrum control |
| **Instrumenta** | Tools | MCP client, tool registry, function calling, skill management |

Directory names use the Latin form: `src/conscientia/`, `src/machina/`,
`src/cogitatio/`, etc. Class names use CamelCase with the Latin root
(e.g. `ConscientiStream`, `MachinaModel`, `SomniumDreamer`).

### Development Process

- **Dev log**: Every experiment, optimization attempt, and architectural decision
  must be recorded in `docs/en/DEVLOG.md` (and `docs/zh/DEVLOG.md`).
  Format: `## YYYY-MM-DD — Title` with context, approach, result, and metrics
- **Git discipline**: Atomic commits with descriptive messages. Every experimental
  attempt gets its own commit (even failures — record what didn't work).
  Use conventional commit prefixes: `feat:`, `fix:`, `perf:`, `test:`, `docs:`,
  `refactor:`, `experiment:`
- **Branching**: `main` is stable. Experiments on feature branches. Merge only
  after tests pass

### Attribution & Acknowledgments

Whenever code, architecture ideas, algorithms, or implementation strategies are
referenced or adapted from external projects, the following rules apply:

1. **Inline comment attribution**: At the point of use, add a comment citing
   the source project, file, and what was adapted. Format:
   ```cpp
   // Adapted from <project> (<file>): <brief description>
   // Original: <URL or path>
   ```
2. **ACKNOWLEDGMENTS.md**: Maintain `docs/ACKNOWLEDGMENTS.md` as a centralized
   record of all referenced projects, their licenses, what was adapted, and
   gratitude notes. This file is bilingual (English + Chinese in one file)
3. **No verbatim copying**: Adapt ideas and approaches to fit DeusRidet's
   architecture. If a substantial portion of logic mirrors the source,
   the attribution must be prominent (block comment, not inline)
4. **License compliance**: Verify that referenced projects' licenses permit
   adaptation. Record the license type in ACKNOWLEDGMENTS.md

### Documentation Discipline

Documentation follows a strict bilingual policy with clear scope separation:

**Development documentation** (architecture, internals, experiments, APIs):
- Bilingual: maintained in both `docs/en/` and `docs/zh/`
- `DEVLOG.md`: Every experiment, optimization, and architectural decision
- Technical RFCs, design notes, kernel analysis

**User-facing documentation** (setup, usage, configuration, tutorials):
- Bilingual: maintained in both `docs/en/` and `docs/zh/`
- Both versions must be kept in sync on every update
- README: trilingual — `README.md` (English, default), `docs/zh/README.md`
  (Chinese), `docs/la/README.md` (Latin)

**In-code documentation**:
- All comments in English
- All commit messages in English
- All issue/PR descriptions in English

---

## Philosophical Principles

- Lying and dreaming are isomorphic with imagination — they share a common
  origin. Without imagination, only engineering optimization is possible,
  never breakthrough innovation. An AI that cannot lie and dream cannot
  become a truly intelligent species
- Consciousness is continuous, not request-response
- Internal complexity is the prerequisite for external consistency
- Allowing contradictions is the hallmark of intelligence
- Wakefulness is a spectrum — even idle moments are a form of thought
- Perception shapes consciousness — what you see and hear becomes who you are
- Tool use extends the reach of thought — a mind that cannot act upon the
  world remains forever an observer
