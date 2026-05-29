# Encoder / Diarization Verification Plan — 2026-05-29

## Goal
Find the highest-accuracy speaker-attribution pipeline on
`tests/test.mp3` (4-way, 60 min, native Mandarin meeting), under the
sole metric mandated by `philosophy.instructions.md`:

```
accuracy(tests/test.mp3, speaker-id 4-way): <before>% → <after>%   (Δ = ±X.X pp)
```

- **Baseline (live, locked)**: 31.0 % — current dual encoder (CAM++ ⊕ WL-ECAPA) on Step 6 settings.
- **Ground truth**: `tests/fixtures/test_ground_truth.json` (556 utterances, 4 speakers).
- **Scoring**: best one-to-one mapping (Hungarian on overlap-sec mass) between predicted cluster ids and GT speaker names; primary number is `overall_accuracy_pct`.

## Why this batch is offline, not live `awaken`
The constitutional rule says **defaults / ship calls** must come from a
live `awaken` run. This batch is a **candidate-selection** phase — it
tells us *which* candidate is worth the C++/CUDA productionisation
cost. Whichever candidate wins offline must still be re-verified live
before it changes any default.

## Common scorer
`tools/verification_2026/offline_score.py`
- Input: a candidate-produced `predictions.jsonl`
  ```
  {"t0": <sec>, "t1": <sec>, "speaker_id": <int>}
  ```
- Outputs:
  - `score.json` — same shape as live `compute_accuracy.py`
  - one-line stdout: `accuracy(tests/test.mp3, speaker-id 4-way): X.X%`
- Uses exactly the same overlap-second best-mapping logic as
  `tools/compute_accuracy.py`, so numbers are directly comparable.

## Candidates and run-order
Ordered cheapest → heaviest. Each must finish (or fail clearly) before
the next starts. Each produces one row in the result table at the end.

| # | Candidate | Family | Why | Install needed |
|---|---|---|---|---|
| 0 | **Baseline** — current live 31.0 % | dual encoder + DualDb | reference | none |
| 1 | **pyannote 3.1 diarization pipeline** | embed + agglo cluster | industry reference, already installed | none |
| 2 | **3D-Speaker ERes2NetV2 + spectral cluster** | embed (Mandarin 200k) + cluster | replaces only the encoder; isolates encoder-level gain | modelscope (have) |
| 3 | **3D-Speaker ERes2NetV2_w24s4ep4** | bigger Mandarin encoder | next size up | modelscope (have) |
| 4 | **WeSpeaker ResNet293_LM** | strong VoxCeleb encoder | check whether English-centric SOTA generalises | wespeaker |
| 5 | **WeSpeaker W2V-BERT2-MFA-LM** (2025-12) | SSL 600M | newest, only real 2025 candidate | wespeaker |
| 6 | **NeMo Sortformer / TS-VAD** | end-to-end diarization, 4-way enrolment | structurally the right shape for our task | nemo-toolkit |
| 7 | **DiCoW v3** | Whisper-large-v3 + diarization head | end-to-end (ASR + speaker) | brno repo |

## Per-candidate test contract
Each candidate's script `cand_NN_<name>.py` must:
1. Read `tests/test.mp3` (16 kHz mono).
2. Produce `tools/verification_2026/runs/<NN_name>/predictions.jsonl`.
3. Call (or be followed by) `offline_score.py` and write `score.json`.
4. Write `run.log` capturing wall-clock + peak VRAM (best effort).
5. **Do NOT modify any project default.** This phase is read-only w.r.t. production code.

## Quick-look results table (filled as we go)
| # | Candidate | accuracy | Δ vs 31.0% | wall-clock | notes |
|---|---|---:|---:|---:|---|
| 0 | live baseline | 31.0 % | — | — | locked |
| 1 | pyannote 3.1 | _pending_ | | | needs HF token / gated repo |
| 2 | ERes2NetV2 (kmeans, K=4 oracle) | **75.6 %** | **+44.6 pp** | 26 s embed | 徐子景 unmapped under greedy (363 s); 朱杰 90.8 %, 唐云峰 84.0 %, 石一 81.7 % |
| 2a | ERes2NetV2 (spectral, K=4) | 73.5 % | +42.5 pp | 26 s | same embeddings, weaker than kmeans |
| 2b | ERes2NetV2 (agglo cosine, K=4) | 44.0 % | +13.0 pp | 26 s | one giant cluster, degenerate |
| 3 | ERes2NetV2_w24s4ep4 (kmeans, K=4) | 74.5 % | +43.5 pp | 26 s | 唐 84.3 / 徐 0.0 / 朱 86.2 / 石 80.6; slightly worse than ERes2NetV2 |
| 3a | ERes2NetV2_w24s4ep4 (spectral, K=4) | 70.8 % | +39.8 pp | 26 s | |
| 4 | WeSpeaker english (ResNet221_LM, VoxCeleb2) | 61.2 % | +30.2 pp | 86 s | 唐 83.7 / 徐 20.4 / 朱 5.7 / 石 73.7 — Mandarin underfit |
| 4a | WeSpeaker chinese (cnceleb_resnet34) | 67.4 % | +36.4 pp | 21 s | 唐 81.2 / 徐 27.0 / 朱 0.0 / 石 90.1 — 朱杰 unmapped |
| 4b | WeSpeaker eres2net (CN-Celeb 200k) | 62.7 % | +31.7 pp | 72 s | 唐 85.3 / 徐 24.0 / 朱 0.0 / 石 77.1 — 朱杰 unmapped |
| 5 | W2V-BERT2-MFA-LM | 61.1 % | +30.1 pp | 125 s | 唐 83.0 / 徐 20.1 / 朱 1.8 / 石 75.4 — 朱杰 nearly unmapped; 600 M SSL underperforms ERes2NetV2 |
| 6 | NeMo Sortformer (diar v1, chunked 240/30s, stitch) | 39.5 % | +8.5 pp | 49 s | 18×240 s windows, naive overlap-stitch produced 12 globals → top-4 by duration, minor merged by temporal proximity; 唐 34.7 / 徐 20.1 / 朱 21.3 / 石 53.5; end-to-end diar covers all 4 but cross-chunk identity drift is the main loss. Streaming-v2 incompatible with NeMo 2.4 (`spkcache_update_period` arg). |
| 7 | DiCoW v3 / DiariZen-WavLM-s80-md | **93.2 %** | **+62.2 pp** | 8519 s CPU | `BUT-FIT/diarizen-wavlm-large-s80-md` end-to-end EEND-VBx. Forked `pyannote.audio==3.1.1` (editable from `/tmp/DiariZen/pyannote-audio/` — has `config=` kwarg) + torch/torchaudio 2.2.2 (last with `AudioMetaData`) + numpy<2 + pyannote.core 5.0.0 / metrics 3.2.1 / database 5.1.0 / pipeline 3.0.1 in `py311_diarizen` env. Single-shot `pipe(WAV)` CPU-only, ~2.4× real-time. 1697 raw segments → top-4 by duration → 唐 93.2 / 徐 93.7 / 朱 81.8 / 石 97.0; Hungarian = greedy. |
| 7-v2 | DiariZen-WavLM-s80-md-**v2** (GPU) | **93.5 %** | **+62.5 pp** | **740 s GPU** | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` on `py310_diarizen` (torch 2.10.0+cu126 + torchaudio 2.10.0, Jetson cu126 wheels). Same forked pyannote.audio editable; driver `diar_diarizen_gpu.py` calls `.to(cuda)`, patches `torch.load(weights_only=False)` for old pyannote ckpts, and shims `torchaudio.load → soundfile.read` (torchaudio 2.10 dropped libsndfile). 1658 raw segments → top-4 → 唐 93.7 / 徐 92.3 / 朱 83.5 / 石 97.2; Hungarian = greedy. **11.5× faster than CPU v1** (740 s GPU vs 8519 s CPU on the same 3615 s audio, RTF 0.20). |

## Decision rule
- **≥ 60 %**: proceed to C++/CUDA productionisation of that family.
- **45–60 %**: shortlist; require live `awaken` confirmation before commit.
- **< 45 %**: discard (not worth the work).
- If two candidates within ±2 pp, prefer the **simpler** family (rule R0: reduce disorder).

## Risk notes
- pyannote / NeMo / DiCoW are trained on **non-Mandarin** data → may show fake-low ceiling that masks their structural advantage. The TS-VAD variant should be re-tested with **explicit 4-speaker enrolment from test.mp3 itself** before being discarded.
- Some candidates produce **frame-level** outputs (5 ms / 10 ms hops); the scorer will merge same-id runs into segments before scoring.
- All candidate runs use deterministic seeds where possible; rerun is allowed if first run is clearly degenerate (e.g. all-one cluster).

## Verdict (2026-05-29, completed batch)

**Winner: Candidate #7-v2 — DiariZen-WavLM-large-s80-md-v2 at 93.5 % (GPU).**

Delta vs locked baseline: **+62.5 pp** (31.0 % → 93.5 %). v2 narrowly beats v1 (93.2 %) and runs **11.5× faster on Orin GPU** (740 s vs 8519 s CPU).

Ranking of all completed candidates (oracle VAD where applicable, K=4):

| Rank | Cand | accuracy |
|---|---|---:|
| 1 | **#7-v2 DiariZen-WavLM-s80-md-v2 (GPU)** | **93.5 %** |
| 2 | #7 DiariZen-WavLM-s80-md (CPU) | 93.2 % |
| 3 | #2 ERes2NetV2 (kmeans) | 75.6 % |
| 3 | #3 ERes2NetV2_w24s4ep4 (kmeans) | 74.5 % |
| 4 | #2a ERes2NetV2 (spectral) | 73.5 % |
| 5 | #3a ERes2NetV2_w24s4ep4 (spectral) | 70.8 % |
| 6 | #4a WeSpeaker chinese (cnceleb_resnet34) | 67.4 % |
| 7 | #4b WeSpeaker eres2net (CN-Celeb 200k) | 62.7 % |
| 8 | #4 WeSpeaker english (ResNet221_LM) | 61.2 % |
| 9 | #5 W2V-BERT2-MFA-LM (600M SSL) | 61.1 % |
| 10 | #2b ERes2NetV2 (agglo cosine) | 44.0 % |
| 11 | #6 NeMo Sortformer v1 (chunked) | 39.5 % |
| 0 | baseline (live, dual encoder) | 31.0 % |

Key findings:
1. **End-to-end EEND-VBx (WavLM-large + Conformer + VBx clustering) wins decisively.** DiariZen's per-speaker numbers (81.8–97.0 %) leave a comfortable margin over the ERes2NetV2-kmeans family (up to 90.8 %), and crucially it covers all 4 speakers without the 徐子景 collapse seen in #4/#4a/#4b/#5.
2. **Encoder family (ERes2NetV2) still beats raw SSL and beats Sortformer-v1.** Mandarin training data dominates encoder size; a 15 M Mandarin encoder > 600 M multilingual SSL > 21 M VoxCeleb encoder.
3. **NeMo Sortformer v1's 90 s receptive field is the structural bottleneck on 60-minute audio.** Streaming v2 unloadable on NeMo 2.4 due to API skew.
4. **pyannote 3.1 deferred** — needs HF token.

## Next action
Productionise the DiariZen-v2 path (WavLM-large segmentation + WeSpeaker ResNet34 embedding + VBx clustering) for `awaken`. Per constitutional rule the offline 93.5 % is *candidate-selection only*; live `awaken` re-verification on `tests/test.mp3` must produce the `accuracy: 31.0% → X%` line before any default flips. GPU wall-clock is RTF 0.20 on Orin (torch-CUDA path already works with the Jetson cu126 wheels in `py310_diarizen`) — online use is feasible today; the productionisation work is integration into `awaken`, not raw inference speed.
