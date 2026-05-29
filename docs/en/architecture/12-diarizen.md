# 12 — DiariZen Reclusterer (Session-Boundary Speaker Re-Attribution)

> *Latin* (placeholder): **Orator Recapitulator** — "the speaker who
> reconsiders". Lives in [`src/orator/`](../../../src/orator/) as
> a session-boundary reclusterer; not a new top-level subsystem.

## Why This Exists

The 2026-05-29 encoder verification sweep (see
[`tools/verification_2026/PLAN.md`](../../../tools/verification_2026/PLAN.md))
established a hard datum:

```
accuracy(tests/test.mp3, speaker-id 4-way):
  baseline (live, dual encoder)              31.0 %
  DiariZen-WavLM-large-s80-md-v2 (offline)   93.5 %   (Δ = +62.5 pp)
```

That `+62.5 pp` is the largest unexploited gain in the project. The
constitutional rule (`philosophy.instructions.md` §"Accuracy Is the
Sole Metric") obliges us to land it — but only via a live `awaken` run,
and only without destroying the current streaming behaviour.

## The Algorithmic Stack to Port

DiariZen-v2 is a three-stage pipeline. Each stage maps to a C++/CUDA
component that must be authored or extended:

| Stage | Algorithm | Existing C++/CUDA | New work |
|-------|-----------|-------------------|----------|
| **S — Segmentation** | WavLM-Large encoder (25 hidden states, **structured-pruned** by BUT-FIT — per-layer dims vary, e.g. conv layer 1 is 153 ch, not 512), 16 s chunks, 0.1 s stride | `src/orator/wavlm_ecapa_encoder.*` exports vanilla 512-ch WavLM-Large 24-layer; cannot load pruned weights directly. New loader must drive layer dims from the checkpoint's tensor shapes | medium |
| **C — Conformer EEND head** | 4-layer Conformer (d=256, ffn=1024, 4 heads), MHA + conv module + FFN macaron + classifier head **outputting 16-d powerset logits** (= C(4,0)+C(4,1)+C(4,2)+C(4,3)+C(4,4) = 1+4+6+4+1, for up-to-4 concurrent speakers), median filtering | none | **large** |
| **E — Embedding** | WeSpeaker ResNet34-LM, 256-d cosine | none (existing WavLM-ECAPA is 192-d, incompatible) | medium |
| **K — Clustering** | VBx (variational Bayes HMM) with PLDA prior, `xvec_transform.npz` + `plda.npz` | `spectral_cluster_gpu.cu` exists but is a different algorithm | medium |
| **O — Orchestration** | Pyannote `SpeakerDiarization.apply` (16-s chunk loop, embedding-per-chunk, AHC seed, VBx finalisation, label stitching) | `auditus` is per-frame streaming; structure does not match | medium |

Hard-numbers anchor: the offline run on a 3615 s recording took 740 s
wall-clock on the Orin GPU (RTF 0.20); the bulk of that is stage S.

## Integration Mode — Hybrid (Reclusterer)

Three integration shapes were considered. The chosen one is **C —
Hybrid**:

1. **Streaming layer is untouched.** Live `awaken` keeps emitting
   `speaker_event` from the current WavLM-ECAPA + DualDb path. Latency
   stays at today's value; the 31.0 % live baseline is the worst-case
   fallback the system can never drop below.
2. **DiariZen runs as a session-boundary reclusterer.** When Vigilia
   detects a session boundary (idle → active transition, sleep,
   long-silence threshold, or explicit user request from Nexus), the
   recently-captured PCM ring is fed through the DiariZen stack, top-4
   cluster ids are mapped to existing live speaker ids by overlap, and
   a stream of `speaker_amend` events (already implemented in Step
   17b-A, see [`/memories/auditus-tuning.md`](../../../) note "17b-A
   PASSED") rewrites the transcript's `speaker_id` field retroactively.
3. **No default flip until live accuracy is measured.** Per the
   constitutional rule, the reclusterer is shipped with
   `DEUSRIDET_DIARIZEN_RECLUSTER=0` by default; flip to `1` only after
   a live `awaken` run produces the
   `accuracy(tests/test.mp3, speaker-id 4-way): 31.0% → X%` line.

### Why this shape

- **Anti-entropy.** It adds capability without removing or rewriting
  any working code. The live streaming SpeakerVectorStore remains the
  source of truth for immediate decisions; DiariZen produces a strictly
  better posterior given the full recording.
- **Philosophical fit.** "Continuity over request-response" — the
  streaming brain still runs at 20 W; the reclusterer is the
  consolidator that fires at idle, the same way Somnium fires at sleep.
- **Risk containment.** If the C++/CUDA DiariZen port disagrees with
  the Python reference, the symptom is a wrong `speaker_amend` event,
  which is observable in the WebUI and revertible to the streaming
  identity by toggling one env var.
- **Reuses Step 17b-A infrastructure.** The `speaker_amend` envelope
  type, the broadcast wiring, and the WebUI consumer already exist and
  are validated; this work only changes the *source* of amend events
  from RetroFullRing (peek_best one-shot) to DiariZen (whole-session
  re-segmentation + re-clustering).

## File Layout (planned)

All new code lives under `src/orator/` — the algorithm is speaker
attribution. No new top-level subsystem is created.

```
src/orator/
├── diarizen_pipeline.{h,cpp}          # facade: PCM → speaker_amend list
├── diarizen_conformer_head.{h,cu}     # 4-layer Conformer EEND head
├── diarizen_segmentation.{h,cu}       # 16-s chunk loop, median filter
├── diarizen_resnet34_embed.{h,cu}     # WeSpeaker ResNet34-LM 256-d
├── diarizen_vbx_cluster.{h,cu}        # VBx VB-HMM + PLDA scoring
└── diarizen_weights.{h,cpp}           # safetensors loader for all four parts
```

Plus a one-shot Python tool (acceptable because it runs once at install
time and produces deterministic safetensors output):

```
tools/convert_diarizen_to_safetensors.py
  → ~/models/dev/diarizen_v2/
       wavlm_pruned.safetensors          (127 MB FP16 — BUT-FIT pruned)
       conformer_head.safetensors        ( 12 MB FP16)
       wespeaker_resnet34.safetensors    ( 13 MB FP16)
       xvec_transform.npz                (134 KB — LDA matrices, verbatim)
       plda.npz                          (134 KB — PLDA priors, verbatim)
       shapes.json                       (per-tensor shape index)
```

The `.npz` priors are kept verbatim because they're tiny and the C++
VBx kernel will unpack them via a small custom reader; safetensors is
overkill here.

## Memory Budget Impact

All five DiariZen weight artefacts together: **~152 MB FP16**, all
**lazy-loaded** when the first reclusterer trigger fires, **paged out**
after the session ends (no permanent residency). The combined footprint
is a fraction of what the always-resident LLM consumes; this is **not**
a budget concern at this stage.

- Peak transient: ~152 MB weights + ~600 MB activation scratch +
  ~200 MB embedding cache per session ≤ **~1.0 GB transient**.
- Compute path: stage S (WavLM-large) dominates at ~85 % of wall-clock,
  stages C/E/K each ≤ 5 %.
- Streams: reclusterer runs on a dedicated low-priority CUDA stream so
  it cannot starve Conscientia prefill / decode.

Update `11-machina.md` Machina memory budget table when the loader
lands, marking these as `env-gated, lazy, session-scoped`.

## Phased Plan (independently-verifiable steps)

Per [`workflow.instructions.md`](../../../.github/instructions/workflow.instructions.md)
no step exceeds the soft size cap; each ends with a green build.

> **2026-05-29 update.** The native CUDA port (P1–P3 below) is
> *deferred*. We shipped the equivalent capability via an IPC fast-path
> calling the existing Python DiariZen-v2 stack out-of-process. See
> `docs/{en,zh}/devlog/2026-05-29.md`. The table is preserved for the
> future native port; the *Hybrid IPC* row at the bottom records what
> actually ships today.

| Phase | Deliverable | Verify | Status |
|-------|-------------|--------|--------|
| **P0** | This RFC (en/zh) + `00-overview.md` TOC update + weight conversion script (Python, runs in existing `py310_diarizen` env) | RFC reads, script produces 4 safetensors files | **done 2026-05-29** |
| **P1a** | WavLM-Large 25-hidden tap + s80-md safetensors loader extension to `wavlm_ecapa_encoder` | `test_wavlm_s80md` bit-equality (cosine ≥ 0.999) vs Python reference on one 16 s chunk | deferred (replaced by IPC fast-path) |
| **P1b** | `diarizen_conformer_head.cu` forward path, weight loader, median filter | dry-run on a fixed input tensor matches Python `model.head(x)` to ≤ 1e-3 abs | deferred |
| **P1c** | `diarizen_segmentation.cu` orchestrator (16 s × 0.1 s sliding, stitch) on top of P1a + P1b | end-to-end segmentation logits vs Python reference cosine ≥ 0.99 on `tests/test.mp3` first 60 s | deferred |
| **P2a** | `diarizen_resnet34_embed.cu` + safetensors loader | embedding cosine ≥ 0.999 vs Python on 10 reference clips | deferred |
| **P2b** | `diarizen_vbx_cluster.cu` (NumPy → CUDA port of `VBx.py`) | label sequence bit-equality vs Python on a fixed embedding sequence | deferred |
| **P3a** | `diarizen_pipeline.cpp` facade wiring stages S→C→E→K | offline run on `tests/test.mp3` reproduces 93.5 % ± 0.5 pp via `tools/verification_2026/offline_score.py` | **done via IPC** (`e96255b`) |
| **P3b** | `awaken` integration: session-boundary trigger + `speaker_amend` broadcast, gated by `DEUSRIDET_DIARIZEN_RECLUSTER=1` | live `awaken` run captured by `tools/replay_to_transcript.py` produces `accuracy(tests/test.mp3, speaker-id 4-way): 31.0% → X%` | **done via IPC** (`b0e3a8f` + `0cc9d0d`) |
| **P3c** | Default flip to `=1` *if and only if* P3b accuracy ≥ 80 % live | constitutional accuracy line in commit message | pending LLM-loaded retest |
| **Hybrid IPC P0** | `DiarizenFacade` C++/Python line-JSON bridge using `tools/diarizen_worker.py` | round-trip diarize call returns 1658-seg list on `tests/test.mp3` | **done 2026-05-29** (`e96255b`) |
| **Hybrid IPC P1** | `AudioPipeline` session capture tap + WS `diarizen_finalize` | `accuracy(tests/test.mp3, diarization): — → 93.6%` via `tools/diarizen_live_score.py` | **done 2026-05-29** (`b0e3a8f`) |
| **Hybrid IPC P2** | `TranscriptHoldback` + `DiarizenPeriodicWorker`; WS `diarizen_trigger` / `diarizen_finalize`; LLM-facing `speaker_id` rewrite before injection | `accuracy(tests/test.mp3, diarization): 93.5% → 93.6%` no-regression run | **done 2026-05-29** (`0cc9d0d`) |
| **Hybrid IPC P2-verify** | LLM loaded (`DEUSRIDET_TEST_WS_ENABLE_LLM=1`) end-to-end re-run | accuracy stays ≥ 93.5% with holdback active | pending (Qwen3.5-9B path missing on box) |

A failed phase does not block reporting — per `workflow.instructions.md`
git discipline, every attempted phase commits its work even if reverted.

## Risks Identified at Planning Time

1. **VBx is dataset-tied.** `xvec_transform.npz` and `plda.npz` are
   fit jointly with the ResNet34-LM 256-d embeddings. Substituting any
   other encoder (e.g. ReDimNet) requires re-fitting both priors — out
   of scope for this work, see `tools/verification_2026/PLAN.md`
   "Deferred candidates".
2. **WavLM-Large weight variant.** `wavlm_large_s80_md` is BUT-FIT's
   self-distilled variant, not vanilla Microsoft WavLM. The 25-hidden
   tap order and the `selected_channel=0` flag in their config must be
   reproduced exactly; P1a is gated by a bit-equality check against
   Python.
3. **VBx is CPU-friendly.** The VB-HMM iteration is O(K²T) with small
   K (≤ 4 final speakers) and T ≈ 36 000 (3600 s × 10 fps). This is
   genuinely on the edge of "tiny enough for CPU"; cf.
   `philosophy.instructions.md` §"Compute Belongs on the GPU". Decision:
   first cut on CPU (deterministic, traceable, ≤ 5 % of wall-clock per
   the Python reference); promote to GPU only if profiling shows it
   matters.
4. **Conformer convolution module.** Depth-wise 1-D conv with kernel 15
   and GLU activation — small but unfamiliar shape. P1b includes a
   per-block bit-equality check, not just end-to-end.
5. **Median filter at 0.1 s stride.** Pyannote uses `scipy.signal`
   median filter; equivalent CUDA kernel is trivial but the window
   size must match the Python default (`9` frames at 0.1 s = 0.9 s).

## What This Does Not Do

- Does **not** replace `speaker_db` or `speaker_vector_store`. Those
  remain the live streaming source of truth.
- Does **not** remove `wavlm_ecapa_encoder`. It is the streaming
  encoder; DiariZen S-stage runs in parallel (different weight set,
  different output tap, different lifetime).
- Does **not** affect ASR. The `speaker_amend` envelope only mutates
  the `speaker_id` field of already-finalised transcript entries.
- Does **not** change Conscientia, Memoria, or Vox.

## References

- Pipeline source: <https://github.com/BUTSpeechFIT/DiariZen>
- WavLM-large-s80-md(-v2) weights: `BUT-FIT/diarizen-wavlm-large-s80-md-v2` on HuggingFace
- WeSpeaker ResNet34-LM weights: `pyannote/wespeaker-voxceleb-resnet34-LM` on HuggingFace
- Offline accuracy result: `tools/verification_2026/PLAN.md` row "#7-v2"
- GPU driver used during selection: `tools/verification_2026/diar_diarizen_gpu.py`
- Live-evidence constitutional rule:
  [`philosophy.instructions.md`](../../../.github/instructions/philosophy.instructions.md)
