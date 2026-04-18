#!/usr/bin/env python3
"""Export pyannote/segmentation-3.0 to ONNX for overlap detection (P1).

Model: PyanNet (SincNet + LSTM + Linear)
Input:  (1, 1, 160000) — 10s mono @ 16kHz
Output: (1, num_frames, 7) — powerset encoding
Classes: [non-speech, spk1, spk2, spk3, spk1+2, spk1+3, spk2+3]
"""

import os
import sys
import torch
import numpy as np


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export pyannote/segmentation-3.0 to ONNX")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace auth token (hf_...). "
                             "Required for gated model. Can also set HF_TOKEN env var.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX path (default: ~/models/dev/vad/pyannote_seg3.onnx)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN", None)
    output_dir = os.path.expanduser("~/models/dev/vad")
    output_path = args.output or os.path.join(output_dir, "pyannote_seg3.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not token:
        print("ERROR: pyannote/segmentation-3.0 is a gated model.")
        print("  1. Visit https://hf.co/pyannote/segmentation-3.0 and accept terms")
        print("  2. Create token at https://hf.co/settings/tokens")
        print("  3. Run: python3 tools/export_pyannote_onnx.py --token hf_YOUR_TOKEN")
        print("  Or: export HF_TOKEN=hf_YOUR_TOKEN")
        sys.exit(1)

    print("Loading pyannote/segmentation-3.0 ...")
    from pyannote.audio import Model
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
    model.eval()

    # Print model structure
    print(f"Model type: {type(model).__name__}")
    print(f"Specifications: {model.specifications}")
    
    # Analyze model layers
    for name, module in model.named_modules():
        if name and '.' not in name:
            print(f"  {name}: {type(module).__name__}")

    # Fixed 10s input
    dummy = torch.randn(1, 1, 160000)
    
    # Test forward pass first
    with torch.no_grad():
        out = model(dummy)
    print(f"Forward pass OK — output shape: {out.shape}")
    print(f"Output example (first frame): {out[0, 0, :].numpy()}")

    # Check for SincNet and handle potential issues
    print("\nExporting to ONNX ...")
    try:
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["waveform"],
            output_names=["segmentation"],
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"ONNX export OK: {output_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("\nTrying with torch.jit.trace ...")
        traced = torch.jit.trace(model, dummy)
        torch.onnx.export(
            traced,
            dummy,
            output_path,
            input_names=["waveform"],
            output_names=["segmentation"],
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"ONNX export (traced) OK: {output_path}")

    # Validate
    print("\nValidating ONNX model ...")
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    inp = np.random.randn(1, 1, 160000).astype(np.float32)
    ort_out = sess.run(None, {"waveform": inp})
    print(f"ONNX output shape: {ort_out[0].shape}")
    print(f"ONNX output range: [{ort_out[0].min():.4f}, {ort_out[0].max():.4f}]")

    # Verify consistency with PyTorch output
    pt_out = model(torch.tensor(inp)).detach().numpy()
    diff = np.abs(ort_out[0] - pt_out).max()
    print(f"Max diff PyTorch vs ONNX: {diff:.6f}")
    
    fsize = os.path.getsize(output_path)
    print(f"\nModel file: {output_path} ({fsize / 1e6:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
