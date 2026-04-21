# 02 — Memoria (Cache Manager + Long-Term Memory)

Memoria holds everything the entity remembers, from the last 100 ms of
working KV Cache to lifelong episodic records.

Reference: qwen35-thor `cache_engine`, `cache_manager`, `kv_swapper`,
`block_tracker`. Adapt — do not copy — to fit DeusRidet's
consciousness-centric requirements.

## Cache Manager — Working Memory

3-tier architecture: GPU blocks → SSD overflow → discard.

| Component | Responsibility |
|-----------|---------------|
| **BlockTracker** | Per-request tracking of block locations (GPU vs SSD) |
| **CacheManager** | Unified interface: KV block allocation, BlockTracker, SSD swapper |
| **KVSwapper** | Swap-out (GPU → staging → fwrite → SSD), swap-in (SSD → fread → GPU), prefetch |
| **CacheEngine** | SSD prefix caching with LRU eviction (hash-based lookup) |

### Adaptations for DeusRidet

- Consciousness is a single persistent "request" — its KV Cache grows
  indefinitely and **must** overflow to SSD gracefully.
- Multi-track Decode branches share the Prefill prefix — block refcounting
  prevents premature eviction of shared blocks.
- SSM recurrent state + Conv state must be snapshotted alongside KV blocks
  (separate `.ssm` / `.conv` files per checkpoint).
- `FADV_DONTNEED` after SSD write is critical on unified memory to release
  physical pages back to the GPU allocator.
- Budget: ~20 GB total. Default split: KV 14 GB, SSM/Conv 2 GB, scratch 4 GB.

## Continuous Eviction Model

Runner-style servers evict *entire requests* when KV budget is exhausted.
DeusRidet has a single infinite-lifetime consciousness stream — eviction
must happen **within** the stream, selectively dropping individual KV
blocks while the stream continues running.

- **Attention-score importance scoring**: After each Prefill frame, record
  cumulative attention weight received by each KV block across all Full
  Attention layers. Blocks consistently ignored become eviction candidates.
  Implemented in `MemoriaImportanceScorer` — asynchronous on a separate
  CUDA stream after Prefill completes.
- **Eviction-triggered consolidation hook**: Before a KV block is evicted,
  fire an event to `SomniumConsolidator`. It extracts a compressed summary
  and writes it to the episodic store. Eviction becomes *forgetting with a
  trace* — no memory is silently lost.
- **Sparse block table**: Continuous eviction creates holes in the KV
  sequence. Paged Attention already handles non-contiguous block tables;
  the block table manager must track free slots efficiently.
- **DeltaNet SSM as subconscious continuity**: SSM recurrent states are
  NOT affected by KV eviction — they carry a compressed encoding of ALL
  history (with natural information decay). Even when Full Attention loses
  access to evicted KV blocks, the SSM state preserves a "subconscious
  impression". This is the architectural advantage of the hybrid model.

## Memoria Longa — Long-Term Memory

Beyond working KV, DeusRidet maintains persistent long-term memory on SSD,
loaded into GPU memory on demand.

### Design Principles

- Zero external dependencies — all data structures implemented in C++/CUDA.
- Use LLM hidden states as embedding vectors (zero additional memory cost).
- **Always preserve original text** alongside embeddings for model-upgrade
  safety: when the LLM is replaced, all embeddings become invalid, but
  original text allows full re-embedding as an initialization step.
- Memory consolidation lives in `SomniumConsolidator` (see `04-vigilia.md`)
  — Memoria only provides storage and retrieval.

### Episodic Store (vector search for "what happened")

| Property | Value |
|----------|-------|
| Index | HNSW |
| Vectors | LLM hidden state from last layer, dim=5120 |
| Storage | SSD-backed with GPU-resident top layer for fast search |
| Record | `{embedding, original_text, timestamp_t0, speaker, emotion, importance}` |
| Capacity | ~500 K–1 M records (~2.5–5 GB SSD, HNSW top layer ~200 MB GPU) |

Original text stored at 50–500 bytes per record.

### Semantic Graph (entity-relation network)

| Property | Value |
|----------|-------|
| Structure | CSR adjacency |
| Nodes | Entities: people, places, concepts, events |
| Edges | Weighted, typed (causal / temporal / associative / emotional) |
| Edge decay | Time-based weight decay; reinforced by revisitation |
| Traversal | Top-K pruned BFS per hop |

### Graph Traversal Constraints (human-analog cognitive limits)

| Context | Max hops | Time budget | Rationale |
|---------|----------|-------------|----------|
| Conversation (alert/focused) | 1–2 | < 10 ms | Must fit in one Prefill frame |
| Daydream / idle | 3–4 | < 100 ms | Background association |
| Deep dream consolidation | ≤ 6 | Unbounded | Full network exploration |

Each hop expands only top-K neighbors ranked by `edge_weight × recency ×
emotional_salience`. This prevents combinatorial explosion.

### Hybrid Retrieval (MemoriaRetriever)

1. Query HNSW with current context embedding → top-N episodic matches.
2. Extract entities from matches → seed nodes for graph traversal.
3. Traverse semantic graph (1–2 hops conversation, up to 6 in dreams).
4. Merge results → inject as context into next Prefill frame.

### Model Upgrade Strategy

When the LLM is upgraded, all embedding-based indices become invalid:

1. Load new model weights.
2. Iterate all episodic records, re-embed original text.
3. Rebuild HNSW index.
4. Semantic graph (text-based nodes/edges) remains valid — no rebuild needed.

A one-time initialization cost, analogous to "re-experiencing all memories
through new eyes after waking up different".

## Implementation Surface

```
src/memoria/
├── cache_manager.{h,cpp}   # unified: KV block alloc + BlockTracker + swapper
├── cache_engine.{h,cpp}    # SSD prefix cache, LRU eviction, hash lookup
├── block_tracker.h         # per-request GPU vs SSD block location tracking
├── kv_swapper.{h,cpp}      # KV + SSM + Conv state SSD offload
├── importance_scorer.{h,cu}# attention-score KV block importance tracking
├── episodic_store.{h,cpp}  # HNSW index + SSD-backed episodic records
├── semantic_graph.{h,cpp}  # CSR entity-relation graph + traversal
└── retriever.{h,cpp}       # hybrid retrieval: HNSW + graph → Prefill injection
```
