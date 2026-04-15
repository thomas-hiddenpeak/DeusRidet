#!/usr/bin/env python3
"""Test FRCRN CUDA implementation against PyTorch reference.

Generates a test signal (white noise + sine), runs it through PyTorch FRCRN,
saves raw PCM for the C++ test to consume, and writes the reference output
for comparison.

Usage:
    python3 tools/test_frcrn_cuda.py

Outputs:
    /tmp/frcrn_test_input.raw    — float32 PCM (16kHz, mono)
    /tmp/frcrn_test_ref.raw      — float32 PCM reference output
    /tmp/frcrn_test_spec.raw     — float32 complex spec [2, 321, T] (re, im)
    /tmp/frcrn_test_mask1.raw    — float32 mask1 [2, 321, T]
    /tmp/frcrn_test_mask2.raw    — float32 mask2 [2, 321, T]
"""

import sys
import os
import numpy as np
import torch

# Add FRCRN model path
FRCRN_DIR = os.path.expanduser("~/models/dev/vad/FRCRN")
sys.path.insert(0, FRCRN_DIR)

from frcrn import FRCRN

def generate_test_signal(duration_s=1.0, sr=16000):
    """Generate a test signal: sine wave + white noise."""
    t = np.arange(int(duration_s * sr)) / sr
    # 440Hz sine + noise
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    signal = signal.astype(np.float32)
    # Normalize to [-1, 1]
    signal = signal / max(abs(signal.max()), abs(signal.min()))
    return signal

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate test input
    pcm = generate_test_signal(duration_s=1.0, sr=16000)
    n_samples = len(pcm)
    print(f"Test signal: {n_samples} samples ({n_samples/16000:.3f}s)")

    # Pad to hop alignment
    hop = 320
    if n_samples % hop != 0:
        padded = ((n_samples // hop) + 1) * hop
    else:
        padded = n_samples
    pcm_padded = np.zeros(padded, dtype=np.float32)
    pcm_padded[:n_samples] = pcm

    # Save input
    pcm_padded.tofile("/tmp/frcrn_test_input.raw")
    print(f"Input saved: /tmp/frcrn_test_input.raw ({padded} samples, {padded*4} bytes)")

    # Load PyTorch model
    model = FRCRN(
        complex=True,
        model_complexity=45,
        model_depth=14,
        log_amp=False,
        padding_mode="zeros",
        win_len=640,
        win_inc=320,
        fft_len=640,
        win_type="hann",
    )
    ckpt_path = os.path.join(FRCRN_DIR, "checkpoints", "model.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded PyTorch checkpoint: {ckpt_path}")
    else:
        # Try safetensors path
        st_path = os.path.expanduser("~/models/dev/vad/frcrn_weights/model.safetensors")
        print(f"No PyTorch checkpoint at {ckpt_path}")
        print("Using whatever weights the model initializes with (random)")
        print("This test only verifies shape and pipeline correctness, not numerical match")

    model.eval()

    # Run inference
    input_tensor = torch.from_numpy(pcm_padded).unsqueeze(0)  # [1, T]
    print(f"Input tensor shape: {input_tensor.shape}")

    with torch.no_grad():
        # Get intermediate outputs for debugging
        # Manual forward pass to capture intermediates
        specs = model.stft(input_tensor)
        print(f"STFT output shape: {specs.shape}")  # [B, 2, F, T]

        # Save STFT spec for comparison
        spec_np = specs.squeeze(0).numpy()  # [2, F, T]
        spec_np.tofile("/tmp/frcrn_test_spec.raw")
        print(f"Spec saved: /tmp/frcrn_test_spec.raw shape={spec_np.shape}")

        # Full model forward
        _, enhanced = model(input_tensor)
        enhanced_np = enhanced.squeeze(0).numpy()  # [T]
        print(f"Output shape: {enhanced_np.shape}")

    # Save reference output (trim to original length)
    output = enhanced_np[:n_samples] if len(enhanced_np) >= n_samples else enhanced_np
    output_padded = np.zeros(padded, dtype=np.float32)
    output_padded[:len(output)] = output
    output_padded.tofile("/tmp/frcrn_test_ref.raw")
    print(f"Reference saved: /tmp/frcrn_test_ref.raw ({len(output_padded)} samples)")

    # Stats
    print(f"\nInput  stats: min={pcm_padded.min():.4f} max={pcm_padded.max():.4f} rms={np.sqrt(np.mean(pcm_padded**2)):.4f}")
    print(f"Output stats: min={output_padded.min():.4f} max={output_padded.max():.4f} rms={np.sqrt(np.mean(output_padded**2)):.4f}")

    # SNR improvement estimate
    noise_est_in = np.sqrt(np.mean((pcm_padded - output_padded)**2))
    print(f"RMS difference (in vs out): {noise_est_in:.6f}")

if __name__ == "__main__":
    main()
