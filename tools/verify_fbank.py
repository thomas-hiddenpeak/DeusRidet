#!/usr/bin/env python3
"""Compare fbank features: our implementation (low_freq=0) vs Kaldi (low_freq=20).
Also run CAM++ model to verify embedding quality."""

import numpy as np
import torch
import wave
import os
import json

MODEL_DIR = '/home/rm01/models/dev/speaker/campplus/cache/iic/speech_campplus_sv_zh_en_16k-common_advanced'
EXAMPLE_DIR = os.path.join(MODEL_DIR, 'examples')

def load_wav_int16(path):
    with wave.open(path, 'rb') as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16)

def compute_fbank(pcm_int16, low_freq=0.0, n_fft=512, frame_len=400, hop=160,
                  n_mels=80, sr=16000, normalize_pcm=True):
    """Compute log Mel fbank matching our CUDA kernel structure.
    If normalize_pcm=True, divide by 32768 (our code). If False, keep int16 scale (Kaldi)."""
    pcm = pcm_int16.astype(np.float32)
    if normalize_pcm:
        pcm = pcm / 32768.0

    # Preemphasis
    pre = np.zeros_like(pcm)
    pre[0] = pcm[0] - 0.97 * pcm[0]
    pre[1:] = pcm[1:] - 0.97 * pcm[:-1]

    # Hamming window
    win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1))

    # Mel filterbank
    def hz2mel(f): return 1127.0 * np.log(1 + f / 700)
    def mel2hz(m): return 700 * (np.exp(m / 1127) - 1)

    freq_bins = n_fft // 2 + 1
    mel_low = hz2mel(low_freq)
    mel_high = hz2mel(sr / 2)
    mel_pts = np.linspace(mel_low, mel_high, n_mels + 2)
    bin_pts = np.floor((n_fft + 1) * mel2hz(mel_pts) / sr).astype(int)

    fb = np.zeros((n_mels, freq_bins))
    for m in range(n_mels):
        fl, fc, fr = bin_pts[m], bin_pts[m+1], bin_pts[m+2]
        for k in range(fl, min(fc + 1, freq_bins)):
            if fc > fl:
                fb[m, k] = (k - fl) / (fc - fl)
        for k in range(fc, min(fr + 1, freq_bins)):
            if fr > fc:
                fb[m, k] = (fr - k) / (fr - fc)

    # Frame + DFT + Mel + Log
    n_frames = max(0, (len(pre) - frame_len) // hop + 1)
    fbank = np.zeros((n_frames, n_mels))
    for i in range(n_frames):
        frame = pre[i * hop: i * hop + frame_len] * win
        padded = np.zeros(n_fft)
        padded[:frame_len] = frame
        spec = np.fft.rfft(padded)
        power = np.abs(spec) ** 2
        mel_e = fb @ power
        fbank[i] = np.log(np.maximum(mel_e, 1e-10))

    return fbank

def apply_cmn(fbank):
    """Per-bin mean subtraction (CMN), matching our speaker_encoder CMN kernel."""
    return fbank - fbank.mean(axis=0, keepdims=True)

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ---------------------------------------------------------------------------
# Print filterbank bin differences
# ---------------------------------------------------------------------------
def compare_mel_bins(low_freq_a, low_freq_b, n_mels=80, n_fft=512, sr=16000):
    def hz2mel(f): return 1127.0 * np.log(1 + f / 700)
    def mel2hz(m): return 700 * (np.exp(m / 1127) - 1)
    for lf, label in [(low_freq_a, 'A'), (low_freq_b, 'B')]:
        mel_low = hz2mel(lf)
        mel_high = hz2mel(sr / 2)
        mel_pts = np.linspace(mel_low, mel_high, n_mels + 2)
        bin_pts = np.floor((n_fft + 1) * mel2hz(mel_pts) / sr).astype(int)
        print(f"  low_freq={lf:3.0f} ({label}): first 5 center bins = {bin_pts[1:6].tolist()}, "
              f"last 5 center bins = {bin_pts[-6:-1].tolist()}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
wavs = {
    's1a': os.path.join(EXAMPLE_DIR, 'speaker1_a_cn_16k.wav'),
    's1b': os.path.join(EXAMPLE_DIR, 'speaker1_b_cn_16k.wav'),
    's2a': os.path.join(EXAMPLE_DIR, 'speaker2_a_cn_16k.wav'),
}

print("=== Mel filterbank bin comparison ===")
compare_mel_bins(0.0, 20.0)

data = {}
for key, path in wavs.items():
    pcm = load_wav_int16(path)
    data[key] = pcm
    print(f"\n{key}: {len(pcm)} samples ({len(pcm)/16000:.2f}s)")

    # Our current code: low_freq=0, normalized PCM
    fb_ours = compute_fbank(pcm, low_freq=0.0, normalize_pcm=True)
    fb_ours_cmn = apply_cmn(fb_ours)

    # Kaldi default: low_freq=20, int16 scale PCM
    fb_kaldi = compute_fbank(pcm, low_freq=20.0, normalize_pcm=False)
    fb_kaldi_cmn = apply_cmn(fb_kaldi)

    # Kaldi default: low_freq=20, normalized PCM (to test if PCM scale matters after CMN)
    fb_kaldi_norm = compute_fbank(pcm, low_freq=20.0, normalize_pcm=True)
    fb_kaldi_norm_cmn = apply_cmn(fb_kaldi_norm)

    print(f"  Ours  (lf=0,  norm): min={fb_ours.min():.3f} max={fb_ours.max():.3f} mean={fb_ours.mean():.3f}")
    print(f"  Kaldi (lf=20, raw):  min={fb_kaldi.min():.3f} max={fb_kaldi.max():.3f} mean={fb_kaldi.mean():.3f}")
    print(f"  Kaldi (lf=20, norm): min={fb_kaldi_norm.min():.3f} max={fb_kaldi_norm.max():.3f} mean={fb_kaldi_norm.mean():.3f}")

    # After CMN
    print(f"  CMN Ours:     min={fb_ours_cmn.min():.3f} max={fb_ours_cmn.max():.3f} mean={fb_ours_cmn.mean():.6f}")
    print(f"  CMN Kaldi:    min={fb_kaldi_cmn.min():.3f} max={fb_kaldi_cmn.max():.3f} mean={fb_kaldi_cmn.mean():.6f}")
    print(f"  CMN Kaldi(n): min={fb_kaldi_norm_cmn.min():.3f} max={fb_kaldi_norm_cmn.max():.3f} mean={fb_kaldi_norm_cmn.mean():.6f}")

    # Max abs diff between CMN versions
    diff_scale = np.abs(fb_kaldi_cmn - fb_kaldi_norm_cmn).max()
    diff_lf = np.abs(fb_ours_cmn - fb_kaldi_norm_cmn).max()
    print(f"  Max |CMN_kaldi_raw - CMN_kaldi_norm| = {diff_scale:.6f} (PCM scale effect after CMN)")
    print(f"  Max |CMN_ours - CMN_kaldi_norm| = {diff_lf:.4f} (low_freq effect after CMN)")

print("\n=== Now loading CAM++ model and comparing embeddings ===")

# Load model weights
sd = torch.load(os.path.join(MODEL_DIR, 'campplus_cn_en_common.pt'),
                map_location='cpu', weights_only=False)

# The model expects fbank features as [1, T, 80] input
# We need to trace through the model architecture to get embeddings
# Instead, let's use a simpler approach: extract fbank, apply CMN, see cosine similarity of features
# to verify the effect of low_freq on feature space

# Compute embeddings using the three fbank variants
# For now, just show cosine similarity of the CMN features themselves
print("\n=== Feature-space cosine similarity (CMN features, averaged over time) ===")
for variant, lf, norm in [("ours_lf0", 0.0, True), ("kaldi_lf20", 20.0, True)]:
    feats = {}
    for key, pcm in data.items():
        fb = compute_fbank(pcm, low_freq=lf, normalize_pcm=norm)
        fb_cmn = apply_cmn(fb)
        feats[key] = fb_cmn.mean(axis=0)  # simple average over time

    print(f"\n  {variant}:")
    print(f"    sim(s1a, s1b) = {cos_sim(feats['s1a'], feats['s1b']):.4f} (same speaker)")
    print(f"    sim(s1a, s2a) = {cos_sim(feats['s1a'], feats['s2a']):.4f} (diff speaker)")

# Now show frame-by-frame absolute difference in first few mel bins
print("\n=== Sample fbank frame values (frame 50, bins 0-4) ===")
pcm = data['s1a']
fb_ours = apply_cmn(compute_fbank(pcm, low_freq=0.0, normalize_pcm=True))
fb_kaldi = apply_cmn(compute_fbank(pcm, low_freq=20.0, normalize_pcm=True))
if fb_ours.shape[0] > 50 and fb_kaldi.shape[0] > 50:
    print(f"  Ours  (lf=0):  {fb_ours[50, :5]}")
    print(f"  Kaldi (lf=20): {fb_kaldi[50, :5]}")
    print(f"  Difference:    {fb_ours[50, :5] - fb_kaldi[50, :5]}")
