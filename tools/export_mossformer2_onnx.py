#!/usr/bin/env python3
"""Export MossFormer2_SS_16K to ONNX for speech separation (P2).

Model: Conv1d encoder → 24× MossFormer2 blocks → ConvTranspose1d decoder
Input:  (1, variable_length) — mixed PCM float32, 16kHz mono
Output: (1, num_spks, variable_length) — 2 separated speaker streams
"""

import os
import sys
import torch
import numpy as np


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export MossFormer2_SS_16K to ONNX")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX path (default: ~/models/dev/vad/mossformer2_ss_16k.onnx)")
    parser.add_argument("--fp16", action="store_true",
                        help="Export with FP16 weights (halves model size)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (auto-detected if not specified)")
    args = parser.parse_args()

    output_dir = os.path.expanduser("~/models/dev/vad")
    output_path = args.output or os.path.join(output_dir, "mossformer2_ss_16k.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading MossFormer2_SS_16K ...")
    from types import SimpleNamespace
    from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K

    # Model hyperparameters from ClearerVoice config.
    model_args = SimpleNamespace(
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
        encoder_kernel_size=16,
        num_spks=2,
    )
    model = MossFormer2_SS_16K(model_args)

    # Find and load checkpoint.
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        # Check known locations
        candidates = [
            os.path.expanduser("~/DeusRidet/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt"),
            "checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                checkpoint_path = cand
                break

    if not checkpoint_path:
        print("ERROR: Could not find MossFormer2_SS_16K checkpoint.")
        print("  Run: python3 -c \"from huggingface_hub import snapshot_download; "
              "snapshot_download('alibabasglab/MossFormer2_SS_16K', "
              "local_dir='checkpoints/MossFormer2_SS_16K')\"")
        print("  Or: python3 tools/export_mossformer2_onnx.py --checkpoint /path/to/checkpoint.pt")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # ClearVoice wraps weights under 'model' key.
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model type: {type(model).__name__}")

    # Test forward pass with 2s audio (model native segment)
    dummy = torch.randn(1, 32000)  # 2s @ 16kHz
    with torch.no_grad():
        out = model(dummy)
    if isinstance(out, (list, tuple)):
        print(f"Forward pass OK — {len(out)} outputs:")
        for i, o in enumerate(out):
            if isinstance(o, torch.Tensor):
                print(f"  output[{i}]: shape={o.shape}, dtype={o.dtype}")
    elif isinstance(out, torch.Tensor):
        print(f"Forward pass OK — output shape: {out.shape}")

    # Export to ONNX
    print(f"\nExporting to ONNX: {output_path}")
    try:
        # Determine output names based on forward pass
        if isinstance(out, torch.Tensor) and out.dim() == 3:
            # Shape (1, num_spks, time) — single tensor with both sources
            output_names = ["sources"]
        else:
            output_names = ["source1", "source2"]

        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["mixture"],
            output_names=output_names,
            dynamic_axes={"mixture": {1: "time"},
                          output_names[0]: {-1: "time"}},
            opset_version=17,
            do_constant_folding=True,
        )
        print("ONNX export OK")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("\nTrying with torch.jit.trace ...")
        try:
            traced = torch.jit.trace(model, dummy)
            torch.onnx.export(
                traced,
                dummy,
                output_path,
                input_names=["mixture"],
                output_names=["sources"],
                dynamic_axes={"mixture": {1: "time"},
                              "sources": {-1: "time"}},
                opset_version=17,
                do_constant_folding=True,
            )
            print("ONNX export (traced) OK")
        except Exception as e2:
            print(f"Traced export also failed: {e2}")
            sys.exit(1)

    # FP16 conversion if requested
    if args.fp16:
        try:
            import onnx
            from onnxconverter_common import float16
            print("\nConverting to FP16 ...")
            model_fp32 = onnx.load(output_path)
            model_fp16 = float16.convert_float_to_float16(model_fp32)
            fp16_path = output_path.replace(".onnx", "_fp16.onnx")
            onnx.save(model_fp16, fp16_path)
            print(f"FP16 model: {fp16_path} ({os.path.getsize(fp16_path)/1e6:.1f} MB)")
        except ImportError:
            print("Note: install 'onnxconverter-common' for FP16 conversion")

    # Validate
    print("\nValidating ONNX model ...")
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    inp = np.random.randn(1, 32000).astype(np.float32)
    ort_out = sess.run(None, {"mixture": inp})
    for i, o in enumerate(ort_out):
        print(f"ONNX output[{i}]: shape={o.shape}, range=[{o.min():.4f}, {o.max():.4f}]")

    fsize = os.path.getsize(output_path)
    print(f"\nModel file: {output_path} ({fsize / 1e6:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
