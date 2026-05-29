#!/usr/bin/env python3
"""P1b bit-equality driver: DiariZen Conformer head.

Mechanical comparison only (cosine + max-abs-diff against the reference
``conformer_out`` / ``classifier_logits`` / ``classifier_probs`` taps). This is
a physical bit-equality check, not a semantic quality score, so a numeric
verdict is admissible per workflow.instructions.md.

Flow:
  1. Load the reference .npz (produced by tools/diarizen_dump_reference.py).
  2. Write ``wavlm_lnorm_out`` [T,256] to a raw float32 file.
  3. Run the native CUDA harness (build/test_diarizen_conformer_biteq).
  4. Load its output and compare against the requested tap.
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
        default="/home/rm01/models/dev/diarizen_v2/conformer_head.safetensors",
    )
    ap.add_argument("--bin", default="build/test_diarizen_conformer_biteq")
    ap.add_argument("--tol-cos", type=float, default=0.999)
    ap.add_argument(
        "--tap",
        default="conformer",
        choices=["conformer", "logits", "probs"],
        help="conformer = conformer_out [T,256]; logits = classifier_logits "
        "[T,16]; probs = classifier_probs [T,16] (log-softmax)",
    )
    args = ap.parse_args()

    data = np.load(args.npz)
    feat = np.asarray(data["wavlm_lnorm_out"], dtype=np.float32)  # [1, T, 256]
    feat = feat.reshape(feat.shape[-2], feat.shape[-1])           # [T, 256]
    T, C = feat.shape

    if args.tap == "logits":
        ref = np.asarray(data["classifier_logits"], dtype=np.float32)
        ref_name = "classifier_logits"
        flag = "--logits"
    elif args.tap == "probs":
        ref = np.asarray(data["classifier_probs"], dtype=np.float32)
        ref_name = "classifier_probs"
        flag = "--probs"
    else:
        ref = np.asarray(data["conformer_out"], dtype=np.float32)
        ref_name = "conformer_out"
        flag = "--conformer"
    ref = ref.reshape(ref.shape[-2], ref.shape[-1])
    print(f"feat: {feat.shape}  {ref_name}: {ref.shape}")

    with tempfile.TemporaryDirectory() as td:
        feat_path = os.path.join(td, "feat.bin")
        out_path = os.path.join(td, "out.bin")
        feat.tofile(feat_path)

        cmd = [args.bin, args.weights, feat_path, str(T), str(C), out_path, flag]
        print("running:", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"harness failed: rc={r.returncode}")
            return 1

        got = np.fromfile(out_path, dtype=np.float32)

    Tr, Cr = ref.shape
    if got.size != Tr * Cr:
        print(f"size mismatch: got {got.size} expected {Tr * Cr} (T={Tr}, C={Cr})")
        return 1
    got = got.reshape(Tr, Cr)

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
