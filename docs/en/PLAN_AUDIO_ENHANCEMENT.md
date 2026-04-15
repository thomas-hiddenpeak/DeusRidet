# Audio Pipeline Enhancement Plan — Comprehensive Evaluation

> Date: 2025-07-22
> Status: DRAFT — Pending Thomas's review
> Context: v24d baseline at 60-65% median speaker diarization accuracy
> Test material: 4 similar Chinese male speakers, 60:15 min audio

## Executive Summary

Three-phase enhancement targeting the root causes of speaker misidentification:
**signal quality → overlap awareness → source separation**. Each phase is
independently testable and buildable, with clear go/no-go gates.

```
Current Pipeline:
  [Mic PCM] → gain → RMS → Silero VAD → speech accumulation → speaker embedding → tracking
                                                                      ↑
                                                              (noise degrades embedding)
                                                              (overlap confuses single-emb)

Enhanced Pipeline:
  [Mic PCM] → gain → RMS → [P0: FRCRN denoise] → Silero VAD → speech accumulation
                                                                     ↓
                                    [P1: pyannote OSD] ← ← ← ← fbank/emb features
                                         ↓ overlap?
                                    Yes → [P2: MossFormer2-SS] → per-speaker PCM → embedding × 2
                                    No  → single-speaker embedding (as today)
```

---

## Phase 0 (P0): FRCRN Speech Enhancement (Denoising)

### Objective

Clean the PCM signal before ALL downstream processing (VAD, speaker embedding,
ASR). This improves everything simultaneously — cleaner signal means better
VAD decisions, sharper speaker embeddings, and more accurate ASR transcription.

### Model: FRCRN_SE_16K

| Property | Value |
|----------|-------|
| Source | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) |
| Architecture | DCCRN = ConvSTFT → dual-UNet (complex mask estimation) → ConviSTFT |
| Weight size | ~161 MB (FP32), ~80 MB (FP16) |
| License | Apache-2.0 |
| Input | Raw PCM float32, 16kHz mono |
| Output | Enhanced PCM float32, same length |
| STFT params | win_len=640 (40ms), win_inc=320 (20ms hop), fft_len=640, hanning |
| Mask type | Complex Ideal Ratio Mask (cIRM), tanh activation |
| Key ops | ConvSTFT, Conv2d, ConvTranspose2d, BatchNorm — all ONNX-friendly |
| Latency | ~20ms per 320-sample hop (single frame); batch processing amortizes |

### ONNX Export Strategy

FRCRN uses standard PyTorch ops (Conv2d, BatchNorm, STFT via learned
convolutions). The ConvSTFT/ConviSTFT are already implemented as Conv1d
with fixed weights — no torchaudio dependency, pure convolution-based STFT.

```python
# Export script (run on dev machine with PyTorch):
import torch
from clearvoice.models.frcrn_se.frcrn import DCCRN

model = DCCRN(complex=True, model_complexity=45, model_depth=14,
              log_amp=False, padding_mode="zeros",
              win_len=640, win_inc=320, fft_len=640, win_type='hanning')
# Load pretrained weights
state_dict = torch.load("FRCRN_SE_16K/model.pt", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Dynamic batch + time axis
dummy = torch.randn(1, 48000)  # 3 seconds
torch.onnx.export(model, dummy, "frcrn_se_16k.onnx",
                  input_names=["audio"],
                  output_names=["enhanced"],
                  dynamic_axes={"audio": {0: "batch", 1: "time"},
                               "enhanced": {0: "batch", 1: "time"}},
                  opset_version=17)
```

**Fallback if ONNX export has issues**: The ConvSTFT uses `F.conv1d` with
fixed (non-learnable) DFT basis as weight — this is ONNX-safe. The UNet is
standard Conv2d/ConvTranspose2d. No known blockers, but verify on first attempt.

### Runtime Options

| Option | Pros | Cons |
|--------|------|------|
| ONNX Runtime CPU EP | Consistent with existing pattern (Silero, FSMN) | Slower, ~30-50ms per chunk |
| ONNX Runtime CUDA EP | Faster (~5-10ms), still ONNX API | Requires CUDA EP build |
| TensorRT (via communis/trt_engine) | Fastest (~2-5ms), FP16 auto-optimization | Need to verify all ops supported |

**Recommendation**: Start with **ONNX Runtime CPU EP** for correctness validation.
If latency exceeds 40ms per chunk, move to TRT. The existing `TrtEngine` class
in `src/communis/trt_engine.h` handles ONNX→TRT build, serialization, caching —
the infrastructure is already in place.

### C++ Class Design

```cpp
// src/sensus/auditus/frcrn_enhancer.h
#pragma once
#include <string>
#include <vector>

namespace deusridet {

struct FrcrnConfig {
    std::string model_path;          // path to frcrn_se_16k.onnx
    int sample_rate    = 16000;
    int chunk_samples  = 48000;      // 3s default processing chunk
    int overlap_samples = 6400;      // 400ms overlap for smooth stitching
    bool enabled       = true;
};

class FrcrnEnhancer {
public:
    FrcrnEnhancer();
    ~FrcrnEnhancer();

    bool init(const FrcrnConfig& cfg);

    // Enhance a chunk of PCM. Input/output are float32 [-1, 1].
    // Returns enhanced PCM of same length.
    // Thread-safe: uses internal session, no shared state.
    std::vector<float> enhance(const float* pcm, int n_samples);

    // In-place enhancement for int16 PCM (converts internally).
    void enhance_inplace(int16_t* pcm, int n_samples);

    bool initialized() const { return initialized_; }

private:
    FrcrnConfig cfg_;
    bool initialized_ = false;
    void* env_     = nullptr;  // Ort::Env*
    void* session_ = nullptr;  // Ort::Session*
};

} // namespace deusridet
```

### Pipeline Integration Point

In `audio_pipeline.cpp`, between RMS computation and Silero VAD (line ~349→351):

```cpp
// EXISTING: RMS computation
stats_.last_rms = n_samples > 0 ? sqrtf((float)(sum_sq / n_samples)) : 0;

// NEW: FRCRN speech enhancement (P0)
if (frcrn_.initialized() && enable_frcrn_.load(std::memory_order_relaxed)) {
    frcrn_.enhance_inplace(pcm_buf.data(), n_samples);
    // Stats: record enhancement latency
    stats_.frcrn_active = true;
}

// EXISTING: Run Silero VAD on (now enhanced) PCM
if (silero_.initialized() && silero_window > 0 && ...
```

**Critical**: Enhancement happens BEFORE all downstream processing. Silero VAD,
FSMN VAD, TEN VAD, speaker embedding extraction, and ASR all receive clean PCM.

### New Member Variables in AudioPipeline

```cpp
FrcrnEnhancer frcrn_;
std::atomic<bool> enable_frcrn_{true};
```

### Chunk Processing Strategy

FRCRN processes variable-length audio (dynamic axis). For streaming:

1. Accumulate PCM in a buffer (e.g., 48000 samples = 3s)
2. Process with FRCRN → get enhanced PCM
3. Use overlap-add with Hanning window for smooth stitching between chunks
4. Feed enhanced PCM to downstream VAD/speaker pipeline

For minimum latency, process in smaller chunks (e.g., 16000 samples = 1s)
at the cost of slightly more frequent ONNX calls.

**Alternative**: Process per-VAD-segment (i.e., only enhance speech segments
detected by Silero). This saves compute on silence but requires Silero to
run on unenhanced audio first. Worth testing both approaches.

### Testing Protocol

1. **A/B comparison**: Run v24d pipeline with/without FRCRN on test.mp3
2. **Metric**: Speaker accuracy via `tools/eval_speaker_accuracy.py`
3. **Ablation**: Check if FRCRN alone (without P1/P2) improves accuracy
4. **Latency**: Measure per-chunk enhancement time, overall pipeline latency impact
5. **Audio quality**: Listen to enhanced output, check for artifacts

### Go/No-Go Gate

- **Go**: Any measurable improvement in speaker accuracy (>2% median increase)
  OR audible noise reduction without artifacts
- **No-go**: Latency > 100ms per chunk (pipeline becomes unusable) OR
  audio artifacts that degrade ASR quality

---

## Phase 1 (P1): Learned Overlap Detection

### Objective

Replace the heuristic overlap detection (`jitter > 0.15f && low_sim_count_ >= 1`)
with a learned model that directly classifies audio frames as single-speaker
vs multi-speaker. The current heuristic has high false-positive rate and misses
many real overlaps.

### Model Options

#### Option A: pyannote/segmentation-3.0 (Recommended)

| Property | Value |
|----------|-------|
| Source | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |
| Architecture | PyanNet (SincNet + LSTM + Linear) |
| License | MIT |
| Input | 10s mono audio @ 16kHz → (1, 1, 160000) |
| Output | (num_frames, 7) powerset encoding |
| Classes | non-speech, spk1, spk2, spk3, spk1+2, spk1+3, spk2+3 |
| Max speakers/chunk | 3 |
| Max speakers/frame | 2 |
| Weight size | ~5 MB |
| Frame resolution | ~16ms per frame (10s → ~625 frames) |
| Training data | AISHELL, AliMeeting, AMI, AVA-AVD, DIHARD, Ego4D, MSDWild, REPERE, VoxConverse |
| Key advantage | **Single model does VAD + OSD + speaker segmentation** |

This model is specifically designed for overlapped speech detection.
The powerset encoding elegantly handles the combinatorics of who is speaking:
- Frame classified as `spk1+2` → overlap detected
- Frame classified as `spk1` → single speaker
- Frame classified as `non-speech` → silence

**Important**: The model processes 10s chunks. For streaming, use a sliding
window with 50% overlap (process every 5s, use the center 5s of each output).

#### Option B: 3D-Speaker's Overlap Detector

The 3D-Speaker diarization pipeline uses pyannote/segmentation-3.0 as its OD
backend, so it's the same model. Their contribution is in the pipeline integration
(which we'll adapt), not a different OD model.

#### Option C: Custom OD on existing features

Train a small classifier on existing features (Silero prob, FSMN prob, F0 jitter,
embedding instability) to detect overlap. Lower accuracy ceiling but zero
additional model weight.

**Decision**: Start with **Option A** (pyannote segmentation-3.0). If ONNX export
is problematic, fall back to **Option C** as interim.

### ONNX Export Strategy

pyannote models use SincNet (Conv1d with learned sinc filter) + LSTM + Linear.
All standard PyTorch ops, ONNX-friendly. Need to verify `torch.sinc` handling.

```python
# Export script:
from pyannote.audio import Model
import torch

model = Model.from_pretrained("pyannote/segmentation-3.0")
model.eval()

# Fixed 10s input
dummy = torch.randn(1, 1, 160000)
torch.onnx.export(model, dummy, "pyannote_seg3.onnx",
                  input_names=["waveform"],
                  output_names=["segmentation"],
                  opset_version=17)
```

**Note**: The model uses SincNet which has `torch.sinc()` — verify this Op is
supported in ONNX opset 17. If not, pre-compute the sinc filters and replace
the SincNet layer with standard Conv1d before export.

### Runtime: Powerset → Multi-label → Overlap Decision

The model outputs 7-class powerset encoding. To get overlap:

```cpp
// Convert powerset → multi-label (done offline or in C++)
// Classes: [non-speech, spk1, spk2, spk3, spk1+2, spk1+3, spk2+3]
//
// For each frame:
//   argmax over 7 classes
//   if argmax ∈ {4, 5, 6} → overlap
//   if argmax ∈ {1, 2, 3} → single speaker
//   if argmax == 0 → non-speech

// Or use multi-label conversion:
//   spk1_active = P(spk1) + P(spk1+2) + P(spk1+3) > threshold
//   spk2_active = P(spk2) + P(spk1+2) + P(spk2+3) > threshold
//   spk3_active = P(spk3) + P(spk1+3) + P(spk2+3) > threshold
//   overlap = count(active) >= 2
```

### C++ Class Design

```cpp
// src/sensus/auditus/overlap_detector.h
#pragma once
#include <string>
#include <vector>

namespace deusridet {

struct OverlapDetectorConfig {
    std::string model_path;         // path to pyannote_seg3.onnx
    float overlap_threshold = 0.5f; // softmax threshold for overlap classes
    int chunk_samples = 160000;     // 10s @ 16kHz (model's native window)
    int hop_samples   = 80000;      // 5s hop for streaming (50% overlap)
    bool enabled      = true;
};

struct OverlapResult {
    bool is_overlap;          // any frame in analysis window has overlap
    float overlap_ratio;      // fraction of frames with overlap in window
    int overlap_start_frame;  // first overlapping frame index
    int overlap_end_frame;    // last overlapping frame index
    // Per-frame results for fine-grained analysis
    std::vector<bool> frame_overlap;    // per-frame overlap flag
    std::vector<int>  frame_num_spk;    // per-frame speaker count (0, 1, 2, 3)
};

class OverlapDetector {
public:
    OverlapDetector();
    ~OverlapDetector();

    bool init(const OverlapDetectorConfig& cfg);

    // Process 10s of PCM, return overlap analysis.
    OverlapResult detect(const float* pcm, int n_samples);

    // Streaming mode: accumulate internally, return result when ready.
    // Returns true when a new result is available.
    bool feed(const float* pcm, int n_samples, OverlapResult& result);

    bool initialized() const { return initialized_; }

private:
    OverlapDetectorConfig cfg_;
    bool initialized_ = false;
    void* env_     = nullptr;
    void* session_ = nullptr;

    // Streaming buffer
    std::vector<float> stream_buf_;
    int stream_pos_ = 0;

    // Powerset → multi-label conversion
    void powerset_to_overlap(const float* output, int num_frames,
                             OverlapResult& result);
};

} // namespace deusridet
```

### Pipeline Integration

The overlap detector runs **in parallel** with speaker embedding extraction,
not sequentially. It replaces the heuristic at line ~2100:

```cpp
// BEFORE (heuristic):
bool overlap_suspected = (jitter > 0.15f && low_sim_count_ >= 1);

// AFTER (learned model):
bool overlap_suspected = false;
if (overlap_det_.initialized() && enable_overlap_det_.load(std::memory_order_relaxed)) {
    // Feed current speech segment to overlap detector
    OverlapResult odr;
    if (overlap_det_.feed(enhanced_pcm, n_samples, odr)) {
        overlap_suspected = odr.is_overlap;
        stats_.overlap_ratio = odr.overlap_ratio;
    }
} else {
    // Fallback to heuristic
    overlap_suspected = (jitter > 0.15f && low_sim_count_ >= 1);
}

if (overlap_suspected) {
    state_ = TrackerState::OVERLAP;
    // P2: trigger speech separation if available
    if (separator_.initialized() && enable_separator_.load(std::memory_order_relaxed)) {
        trigger_separation(speech_pcm_buf_.data(), speech_pcm_buf_.size());
    }
}
```

### Additional Benefit: Better VAD

pyannote segmentation-3.0 also provides frame-level VAD (non-speech class).
This could complement Silero VAD for a more robust speech detection, especially
in noisy conditions. Consider using it as a fourth VAD source.

### Testing Protocol

1. **OD accuracy**: Manually annotate overlap regions in test.mp3, measure
   precision/recall of learned OD vs heuristic
2. **False positive rate**: Count how often OD fires incorrectly on clean
   single-speaker segments
3. **Integration test**: Check that TrackerState::OVERLAP is set correctly
   and no regression in non-overlap tracking
4. **Latency**: 10s chunk processing time (should be <50ms for ~5MB model)

### Go/No-Go Gate

- **Go**: OD precision > 70% AND false positive rate < 10%
- **No-go**: ONNX export impossible (SincNet issues) → fall back to Option C
  (classifier on existing features)

---

## Phase 2 (P2): MossFormer2 Speech Separation

### Objective

When P1 detects overlap, separate the mixed audio into individual speaker
streams, then extract embeddings from each separated stream independently.
This is the key capability gap — the current pipeline cannot handle overlapping
speakers at all.

### Model: MossFormer2_SS_16K

| Property | Value |
|----------|-------|
| Source | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) |
| Architecture | Encoder (Conv1d stride=8) → MossFormer2 blocks × 24 → Decoder (ConvTranspose1d) |
| Weight size | ~670 MB (FP32), ~335 MB (FP16) |
| License | Apache-2.0 |
| Input | Mixed PCM float32, 16kHz mono |
| Output | 2 separated PCM streams, same length |
| num_spks | 2 (fixed — produces exactly 2 output streams) |
| encoder_kernel_size | 16 |
| encoder_embedding_dim | 512 |
| mossformer_sequence_dim | 512 |
| num_mossformer_layer | 24 |
| Segment length | 2s for one-pass decode; longer audio → segmented processing |
| Key ops | Conv1d, Transformer attention, FSMN, LayerNorm |

### Memory Impact

This is the biggest concern. Adding 670 MB (FP32) or 335 MB (FP16) to the
running pipeline is significant on a 64 GB Orin with ~20 GB free for
non-weight usage.

**Mitigation strategies** (in preference order):
1. **FP16 weights**: Halves to 335 MB — minimal accuracy loss for speech separation
2. **Lazy loading**: Only load the model when overlap is detected for the
   first time; keep resident after that. Most conversations don't have frequent
   overlap, so this saves memory during calm periods.
3. **INT8 quantization**: If ClearerVoice provides or we generate INT8 model,
   further halves to ~167 MB. Need to verify separation quality.
4. **On-demand swap**: Load from SSD when needed, unload after 60s idle.
   SSD read at ~2 GB/s means 335 MB loads in ~170ms — acceptable if overlap
   detection gives enough lead time.

**Recommendation**: Start with FP16 lazy loading (option 1+2). Only 335 MB
budget when separation is actively needed.

### ONNX Export Strategy

MossFormer2 uses Conv1d encoder → Transformer-style attention blocks → Conv1d
decoder. The FSMN components may have some non-standard ops.

```python
# Export script:
from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
import torch

# Load model
model = MossFormer2_SS_16K(args)  # args from config
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

# Fixed-length input for export (dynamic shapes later)
dummy = torch.randn(1, 32000)  # 2s @ 16kHz
torch.onnx.export(model, dummy, "mossformer2_ss_16k.onnx",
                  input_names=["mixture"],
                  output_names=["source1", "source2"],
                  dynamic_axes={"mixture": {1: "time"},
                               "source1": {1: "time"},
                               "source2": {1: "time"}},
                  opset_version=17)
```

**Risk**: The model uses 24 MossFormer2 blocks with complex internal structure
(attention + FSMN + gating). Verify ONNX export completeness. If export fails:
- Try `torch.jit.trace` → `torch.onnx.export` from traced model
- Consider exporting encoder, MossFormer2 blocks, and decoder separately
- Last resort: implement Conv1d + attention forward pass directly in C++/CUDA

### C++ Class Design

```cpp
// src/sensus/auditus/speech_separator.h
#pragma once
#include <string>
#include <vector>

namespace deusridet {

struct SpeechSeparatorConfig {
    std::string model_path;          // path to mossformer2_ss_16k.onnx
    int sample_rate     = 16000;
    int max_chunk       = 32000;     // 2s processing chunk (model native)
    int overlap_samples = 3200;      // 200ms overlap for stitching
    bool fp16           = true;      // use FP16 inference
    bool lazy_load      = true;      // load on first use
};

struct SeparationResult {
    std::vector<float> source1;     // separated speaker 1 PCM
    std::vector<float> source2;     // separated speaker 2 PCM
    float energy1;                  // RMS energy of source 1
    float energy2;                  // RMS energy of source 2
    bool valid;                     // separation succeeded
};

class SpeechSeparator {
public:
    SpeechSeparator();
    ~SpeechSeparator();

    bool init(const SpeechSeparatorConfig& cfg);

    // Separate a mixed audio chunk into 2 speaker streams.
    // Input PCM should be float32 [-1, 1].
    // For audio > 2s, internally segments and stitches.
    SeparationResult separate(const float* pcm, int n_samples);

    bool initialized() const { return initialized_; }
    bool loaded() const { return loaded_; }

    // Lazy loading: call when first overlap detected.
    bool ensure_loaded();
    // Manual unload to free memory.
    void unload();

private:
    SpeechSeparatorConfig cfg_;
    bool initialized_ = false;
    bool loaded_ = false;
    void* env_     = nullptr;
    void* session_ = nullptr;

    // Segmented processing for long audio
    SeparationResult separate_chunk(const float* pcm, int n_samples);
    SeparationResult separate_long(const float* pcm, int n_samples);
};

} // namespace deusridet
```

### Pipeline Integration

When overlap is detected (by P1), the pipeline:

1. Takes the current `speech_pcm_buf_` (accumulated PCM for the speech segment)
2. Feeds it through MossFormer2_SS → gets `source1` and `source2`
3. Extracts speaker embedding from each source independently
4. Matches each embedding against the speaker database
5. Updates tracking state accordingly

```cpp
void AudioPipeline::trigger_separation(const int16_t* pcm, int n_samples) {
    // Convert to float
    std::vector<float> fpcm(n_samples);
    for (int i = 0; i < n_samples; i++)
        fpcm[i] = pcm[i] / 32768.0f;

    // Ensure model is loaded (lazy)
    if (!separator_.ensure_loaded()) return;

    // Separate
    auto result = separator_.separate(fpcm.data(), n_samples);
    if (!result.valid) return;

    // Extract speaker embedding from each separated source
    auto emb1 = extract_embedding(result.source1);
    auto emb2 = extract_embedding(result.source2);

    // Match against speaker database
    auto match1 = score_best(emb1, 0);
    auto match2 = score_best(emb2, 0);

    LOG_INFO("Separator", "Overlap resolved: source1→spk%d (%.3f), source2→spk%d (%.3f)",
             match1.speaker_id, match1.score, match2.speaker_id, match2.score);

    // Update timeline with both speakers
    // ...
}
```

### Speaker-Source Assignment Problem

MossFormer2 outputs source1 and source2, but the assignment to _which_ speaker
is arbitrary (permutation problem). Solution:

1. Extract embeddings from both sources
2. Compare both embeddings against the current `ref_emb_` (known speaker)
3. The source with higher similarity to `ref_emb_` is the current speaker
4. The other source is the overlapping speaker
5. Match the overlapping speaker against the full speaker database

This is the standard permutation-invariant approach for speaker separation.

### Testing Protocol

1. **Separation quality**: Create test mixtures from known single-speaker
   segments, verify SI-SDR improvement after separation
2. **End-to-end**: Feed overlapping segments through full pipeline,
   check if both speakers are correctly identified
3. **Latency**: Measure separation time for 1s, 2s, 5s segments
4. **Memory**: Monitor GPU memory with `nvidia-smi` during separation
5. **Lazy loading**: Verify model loads within 500ms on first overlap,
   stays resident for subsequent calls

### Go/No-Go Gate

- **Go**: Both speakers correctly identified >60% of the time on synthetic
  overlap test data AND separation latency < 500ms for 2s chunks
- **No-go**: ONNX export fails (MossFormer2 blocks too complex) OR
  memory exceeds available budget → defer to P3 (simpler separator)

---

## Implementation Order & Dependencies

```
P0 (FRCRN)         P1 (OD)              P2 (MossFormer2-SS)
│                   │                    │
├─ ONNX export      ├─ ONNX export       ├─ ONNX export
├─ C++ wrapper      ├─ C++ wrapper       ├─ C++ wrapper
├─ Pipeline insert  ├─ Pipeline insert   ├─ Pipeline insert
├─ Test alone       ├─ Test alone        ├─ Test with P1
│                   │                    │
▼                   ▼                    ▼
Gate 0              Gate 1               Gate 2
```

- **P0 is independent** — can start immediately. Benefits all downstream tasks.
- **P1 is independent** — can start in parallel with P0.
- **P2 depends on P1** — separation is only triggered when overlap is detected.
  P2 code can be developed independently but integration testing requires P1.

### Phased Rollout

1. **v25a**: P0 only — FRCRN denoising. Measure baseline improvement.
2. **v25b**: P0 + P1 — add overlap detection. Measure OD accuracy.
3. **v25c**: P0 + P1 + P2 — full pipeline with separation. Measure
   end-to-end speaker accuracy improvement.

Each version gets its own commit, benchmark run, and DEVLOG entry.

---

## Model Preparation Steps (Prerequisite)

All model ONNX exports must be done on a machine with PyTorch + ClearerVoice
+ pyannote.audio installed. The resulting `.onnx` files are then deployed to
the Orin.

### Step 1: Download models

```bash
# FRCRN
pip install clearvoice
# Download checkpoint via ClearerVoice API or manually from ModelScope

# pyannote segmentation-3.0
pip install pyannote.audio
# Requires HuggingFace auth token — agree to model terms first

# MossFormer2_SS
# Same as FRCRN — via ClearerVoice
```

### Step 2: Export to ONNX

Run the export scripts listed in each phase section above.

### Step 3: Validate ONNX

```bash
# Verify ONNX model loads correctly
python -c "
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('frcrn_se_16k.onnx')
inp = np.random.randn(1, 48000).astype(np.float32)
out = sess.run(None, {'audio': inp})
print('Output shape:', out[0].shape)  # should be (1, 48000)
"
```

### Step 4: Deploy to Orin

```bash
scp frcrn_se_16k.onnx ~/models/dev/audio/frcrn/
scp pyannote_seg3.onnx ~/models/dev/audio/pyannote/
scp mossformer2_ss_16k.onnx ~/models/dev/audio/mossformer2/
```

### Step 5: (Optional) TensorRT optimization

```bash
# Convert ONNX → TRT engine on Orin for maximum performance
# Use existing TrtEngine infrastructure
```

---

## CMake Integration

```cmake
# In src/sensus/CMakeLists.txt — add new source files:
target_sources(sensus PRIVATE
    auditus/frcrn_enhancer.cpp      # P0
    auditus/overlap_detector.cpp    # P1
    auditus/speech_separator.cpp    # P2
)

# All use ONNX Runtime — already linked to sensus target
```

---

## Configuration (machina.conf additions)

```ini
[audio.enhancement]
frcrn_model_path   = ~/models/dev/audio/frcrn/frcrn_se_16k.onnx
frcrn_enabled      = true
frcrn_chunk_ms     = 3000    # 3s processing chunk

[audio.overlap]
od_model_path      = ~/models/dev/audio/pyannote/pyannote_seg3.onnx
od_enabled         = true
od_threshold       = 0.5

[audio.separation]
separator_model_path = ~/models/dev/audio/mossformer2/mossformer2_ss_16k.onnx
separator_enabled    = true
separator_fp16       = true
separator_lazy_load  = true
```

---

## WebUI Observability Additions

New panel elements for the enhanced pipeline:

| Metric | Location | Update Rate |
|--------|----------|-------------|
| FRCRN active (on/off) | Audio Pipeline panel | Per chunk |
| FRCRN latency (ms) | Audio Pipeline panel | Per chunk |
| Overlap detected (on/off) | Speaker Tracker panel | Per OD window |
| Overlap ratio (%) | Speaker Tracker panel | Per OD window |
| Separation active (on/off) | Speaker Tracker panel | Per overlap event |
| Separated sources (2x waveform) | New: Separator panel | Per separation |
| Source→Speaker mapping | Speaker Tracker panel | Per separation |

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| FRCRN ONNX export fails | P0 blocked | Low | ConvSTFT is pure Conv1d, UNet is standard |
| pyannote SincNet ONNX issue | P1 blocked | Medium | Pre-compute sinc filters, replace with Conv1d |
| MossFormer2 ONNX too complex | P2 blocked | Medium | Try traced model, or separate sub-model export |
| MossFormer2 memory too high | P2 impractical | Low | FP16 + lazy load caps at 335MB |
| FRCRN degrades embeddings | P0 harmful | Low | A/B test; can disable per-config |
| OD false positives trigger bad separations | P1+P2 harmful | Medium | Tune OD threshold; require N consecutive overlap frames |
| Separation produces artifacts | P2 harmful | Medium | Energy check: if one source is <5% energy, skip it |
| Latency budget exceeded | Pipeline unusable | Low | Start CPU EP, upgrade to TRT if needed |

---

## Success Criteria

| Metric | Current (v24d) | Target (v25c) | Method |
|--------|---------------|--------------|--------|
| Median speaker accuracy | 60-65% | 75-80% | eval_speaker_accuracy.py |
| Overlap handling | None (TrackerState::OVERLAP unused effectively) | Both speakers identified >60% | Manual overlap segment test |
| VAD false negative rate | Unknown | Measurably lower | Silero+FRCRN vs Silero alone |
| ASR quality on noisy segments | Baseline | Improved with FRCRN | Manual comparison |

---

## Open Questions for Thomas

1. **FRCRN chunk size**: Should we process the full pipeline chunk (whatever
   size the ring buffer delivers, ~30ms) or accumulate to 1-3s before
   enhancement? Shorter = lower latency, longer = better enhancement quality.

2. **OD granularity**: pyannote processes 10s windows with ~625 frames (16ms
   resolution). For our streaming pipeline, is 5s hop (50% overlap) acceptable,
   or do we need tighter integration with the existing VAD windowing?

3. **Separation trigger**: When OD detects overlap, should we separate:
   - (a) The entire current speech segment (speech_pcm_buf_), or
   - (b) Only the overlap region (OD tells us exactly which frames), or
   - (c) The overlap region + 1s padding on each side for context?

4. **Model preparation**: Do you want to run the ONNX export yourself (you have
   the hardware + PyTorch), or should I write a complete export script package
   that you just run?

5. **Priority**: Start P0 first (denoising alone), or start P0+P1 in parallel?
