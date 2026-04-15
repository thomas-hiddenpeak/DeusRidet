#!/usr/bin/env python3
"""Export FRCRN speech enhancement model to ONNX.

Downloads the FRCRN_SE_16K model from ModelScope and exports to ONNX format
with dynamic time axis for variable-length audio input.

Usage:
    python3 tools/export_frcrn_onnx.py [--output ~/models/dev/vad/frcrn_se_16k.onnx]

The model takes raw PCM float32 [1, T] and returns enhanced PCM float32 [1, T].
T must be a multiple of win_inc (320 samples = 20ms).

Reference: ModelScope iic/speech_frcrn_ans_cirm_16k (Apache-2.0)
Architecture: DCCRN = ConvSTFT → dual-UNet (cIRM) → ConviSTFT
"""

import argparse
import os
import sys

import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description="Export FRCRN to ONNX")
    parser.add_argument("--output", type=str,
                        default=os.path.expanduser("~/models/dev/vad/frcrn_se_16k.onnx"),
                        help="Output ONNX file path")
    parser.add_argument("--cache-dir", type=str, default="/tmp/frcrn_cache",
                        help="ModelScope download cache directory")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX output matches PyTorch")
    args = parser.parse_args()

    # Download model from ModelScope.
    print("[1/4] Downloading FRCRN model from ModelScope...")
    from modelscope.hub.snapshot_download import snapshot_download
    model_dir = snapshot_download('iic/speech_frcrn_ans_cirm_16k',
                                  cache_dir=args.cache_dir)
    print(f"  Model dir: {model_dir}")

    # Import FRCRN architecture from modelscope.
    print("[2/4] Loading FRCRN model...")
    from modelscope.models.audio.ans.frcrn import FRCRN

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

    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Loaded weights: {weights_path} ({os.path.getsize(weights_path)/1024/1024:.1f} MB)")

    # Create an inference-only wrapper that returns only the enhanced waveform.
    class FRCRNInference(nn.Module):
        """Thin wrapper: input [1, T] → output [1, T] (enhanced PCM)."""
        def __init__(self, frcrn):
            super().__init__()
            self.frcrn = frcrn

        def forward(self, audio):
            # FRCRN.forward returns [spec_l1, wav_l1, mask_l1, spec_l2, wav_l2, mask_l2]
            # We want wav_l2 (index 4) — the second-pass refined output.
            out_list = self.frcrn(audio)
            enhanced = out_list[4]  # [B, T']
            # Trim or pad to match input length.
            T_in = audio.shape[1]
            T_out = enhanced.shape[1]
            if T_out >= T_in:
                enhanced = enhanced[:, :T_in]
            else:
                enhanced = torch.nn.functional.pad(enhanced, (0, T_in - T_out))
            return enhanced

    wrapper = FRCRNInference(model)
    wrapper.eval()

    # Export to ONNX.
    print("[3/4] Exporting to ONNX...")
    # Use 1 second of audio as dummy input (16000 samples).
    dummy = torch.randn(1, 16000)

    # Test forward pass first.
    with torch.no_grad():
        test_out = wrapper(dummy)
    print(f"  Test forward: input {dummy.shape} → output {test_out.shape}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["audio"],
        output_names=["enhanced"],
        dynamic_axes={
            "audio": {1: "time"},
            "enhanced": {1: "time"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    file_size = os.path.getsize(args.output) / 1024 / 1024
    print(f"  Exported: {args.output} ({file_size:.1f} MB)")

    # Verify.
    if args.verify:
        print("[4/4] Verifying ONNX output...")
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        print(f"  Input:  {inp.name} {inp.shape} {inp.type}")
        print(f"  Output: {out.name} {out.shape} {out.type}")

        # Compare with PyTorch output.
        test_audio = np.random.randn(1, 32000).astype(np.float32)
        ort_out = sess.run(None, {"audio": test_audio})[0]

        with torch.no_grad():
            pt_out = wrapper(torch.from_numpy(test_audio)).numpy()

        diff = np.abs(ort_out - pt_out).max()
        print(f"  Max diff (ONNX vs PyTorch): {diff:.6f}")
        if diff < 0.001:
            print("  ✓ Verification passed")
        else:
            print(f"  ✗ Large difference detected: {diff}")
    else:
        print("[4/4] Skipping verification (use --verify to enable)")

    print("\nDone!")


if __name__ == "__main__":
    main()
