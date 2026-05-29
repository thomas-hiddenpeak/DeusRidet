#!/usr/bin/env python3
"""P1a-step2a bit-equality driver: DiariZen WavLM-pruned CNN feature extractor.

Mechanical comparison only (cosine + max-abs-diff against the reference
``cnn_out`` tap). This is a physical bit-equality check, not a semantic
quality score, so a numeric verdict is admissible per
workflow.instructions.md.

Flow:
  1. Load the reference .npz (produced by tools/diarizen_dump_reference.py).
  2. Write ``wave_in`` to a raw float32 file.
  3. Run the native CUDA harness (build/test_diarizen_cnn_biteq).
  4. Load its [T,211] output and compare against ``cnn_out``.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        default="tests/fixtures/diarizen_p1a_reference.npz",
        help="reference fixture path",
    )
    ap.add_argument(
        "--weights",
        default="/home/rm01/models/dev/diarizen_v2/wavlm_pruned.safetensors",
    )
    ap.add_argument("--bin", default="build/test_diarizen_cnn_biteq")
    ap.add_argument("--tol-cos", type=float, default=0.999)
    ap.add_argument(
        "--tap",
        choices=["cnn", "tap0"],
        default="cnn",
        help="cnn = CNN feature extractor (cnn_out, 211 ch); "
        "tap0 = encoder front end vs layer_hiddens[0] (1024 ch)",
    )
    args = ap.parse_args()

    data = np.load(args.npz)
    wave = np.asarray(data["wave_in"], dtype=np.float32).reshape(-1)
    if args.tap == "tap0":
        lh = np.asarray(data["layer_hiddens"], dtype=np.float32)  # [25,1,T,1024]
        ref = lh[0].reshape(lh.shape[-2], lh.shape[-1])           # [T, 1024]
        ref_name = "layer_hiddens[0]"
    else:
        ref = np.asarray(data["cnn_out"], dtype=np.float32)       # [1, T, 211]
        ref = ref.reshape(ref.shape[-2], ref.shape[-1])          # [T, 211]
        ref_name = "cnn_out"
    n = wave.size
    print(f"wave_in: {n} samples  {ref_name}: {ref.shape}")

    with tempfile.TemporaryDirectory() as td:
        pcm_path = os.path.join(td, "wave_in.bin")
        out_path = os.path.join(td, "cnn_out.bin")
        wave.tofile(pcm_path)

        cmd = [args.bin, args.weights, pcm_path, str(n), out_path]
        if args.tap == "tap0":
            cmd.append("--tap0")
        print("running:", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"harness failed: rc={r.returncode}")
            return 1

        got = np.fromfile(out_path, dtype=np.float32)

    T, C = ref.shape
    if got.size != T * C:
        print(f"size mismatch: got {got.size} expected {T * C} (T={T}, C={C})")
        return 1
    got = got.reshape(T, C)

    a = got.reshape(-1)
    b = ref.reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    max_abs = float(np.max(np.abs(a - b)))
    mean_abs = float(np.mean(np.abs(a - b)))
    print(f"cosine={cos:.6f}  max_abs_diff={max_abs:.6e}  mean_abs_diff={mean_abs:.6e}")

    ok = cos >= args.tol_cos
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
