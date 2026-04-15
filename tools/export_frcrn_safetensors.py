#!/usr/bin/env python3
"""Export FRCRN weights to safetensors for custom CUDA inference engine.

Exports ALL parameters AND buffers (running_mean, running_var) needed for
eval-mode inference. Also computes and stores the STFT Hann window and
iSTFT synthesis window for use with cuFFT (replacing the Conv1d DFT basis).

Usage:
    python3 tools/export_frcrn_safetensors.py [--output ~/models/dev/vad/frcrn_weights]

Output directory will contain a single model.safetensors file.
"""

import argparse
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Export FRCRN to safetensors")
    parser.add_argument("--output", type=str,
                        default=os.path.expanduser("~/models/dev/vad/frcrn_weights"),
                        help="Output directory")
    parser.add_argument("--cache-dir", type=str, default="/tmp/frcrn_cache",
                        help="ModelScope download cache directory")
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file

    # Download model from ModelScope.
    print("[1/3] Loading FRCRN model...")
    from modelscope.hub.snapshot_download import snapshot_download
    model_dir = snapshot_download('iic/speech_frcrn_ans_cirm_16k',
                                  cache_dir=args.cache_dir)

    from modelscope.models.audio.ans.frcrn import FRCRN
    model = FRCRN(
        complex=True, model_complexity=45, model_depth=14,
        log_amp=False, padding_mode="zeros",
        win_len=640, win_inc=320, fft_len=640, win_type="hann",
    )

    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Loaded weights from: {weights_path}")

    # Collect all tensors: parameters + buffers (running stats).
    print("[2/3] Collecting tensors...")
    tensors = {}

    # Parameters (learned weights)
    for name, param in model.named_parameters():
        tensors[name] = param.detach().contiguous().float()

    # Buffers (running_mean, running_var, num_batches_tracked, etc.)
    for name, buf in model.named_buffers():
        if 'num_batches_tracked' in name:
            continue  # Not needed for inference
        tensors[name] = buf.detach().contiguous().float()

    # Compute and store the Hann analysis window (sqrt(hann)) for STFT.
    # This matches what ConvSTFT uses: get_window('hann', win_len, fftbins=True)**0.5
    from scipy.signal import get_window
    win_len = 640
    hann_sqrt = get_window('hann', win_len, fftbins=True) ** 0.5
    tensors['stft_window'] = torch.from_numpy(hann_sqrt.astype(np.float32))

    # Compute the iSTFT synthesis window normalization.
    # In PyTorch ConviSTFT, the output is divided by the window-squared OLA sum.
    # We precompute the window for cuFFT iSTFT.
    tensors['istft_window'] = torch.from_numpy(hann_sqrt.astype(np.float32))

    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"  Collected {len(tensors)} tensors ({total_bytes / 1024 / 1024:.1f} MB)")

    # Print summary of tensor groups
    groups = {}
    for name in tensors:
        prefix = name.split('.')[0]
        if prefix == 'unet' or prefix == 'unet2':
            sub = name.split('.')[1] if len(name.split('.')) > 1 else ''
            key = f"{prefix}.{sub}"
        else:
            key = prefix
        groups[key] = groups.get(key, 0) + 1
    for key in sorted(groups):
        print(f"    {key}: {groups[key]} tensors")

    # Save to safetensors.
    print("[3/3] Saving to safetensors...")
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "model.safetensors")
    save_file(tensors, output_path)
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")
    print(f"\nDone! {len(tensors)} tensors exported.")

if __name__ == "__main__":
    main()
