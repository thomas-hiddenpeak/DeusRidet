#!/usr/bin/env python3
"""P1c bit-equality driver: DiariZen segmentation orchestrator.

Mechanical comparison only (exact-match fraction + cosine against the reference
``seg_raw`` / ``seg_med`` taps). This is a physical bit-equality check, not a
semantic quality score, so a numeric verdict is admissible per
workflow.instructions.md.

Flow:
  1. Load the reference .npz (tools/diarizen_dump_segmentation.py).
  2. Write ``wave_full`` to a raw float32 file.
  3. Run the native CUDA harness (build/test_diarizen_segmentation_biteq).
  4. Load its [num_chunks, 799, 4] output (3-int32 header) and compare.

The multilabel maps are binary, so the primary metric is the exact-match
fraction; cosine is reported as a secondary sanity number.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz", default="tests/fixtures/diarizen_p1c_segmentation.npz")
    ap.add_argument(
        "--wavlm",
        default="/home/rm01/models/dev/diarizen_v2/wavlm_pruned.safetensors")
    ap.add_argument(
        "--conformer",
        default="/home/rm01/models/dev/diarizen_v2/conformer_head.safetensors")
    ap.add_argument("--bin", default="build/test_diarizen_segmentation_biteq")
    ap.add_argument("--tap", default="med", choices=["med", "raw"],
                    help="med = post median filter (seg_med); raw = seg_raw")
    ap.add_argument("--tol-match", type=float, default=0.9999)
    args = ap.parse_args()

    data = np.load(args.npz)
    wave = np.asarray(data["wave_full"], dtype=np.float32).reshape(-1)
    ref = np.asarray(data["seg_med" if args.tap == "med" else "seg_raw"],
                     dtype=np.float32)
    n = wave.size
    print(f"wave_full: {n} samples  ref({args.tap}): {ref.shape}")

    with tempfile.TemporaryDirectory() as td:
        wave_path = os.path.join(td, "wave.bin")
        out_path = os.path.join(td, "seg.bin")
        wave.tofile(wave_path)

        cmd = [args.bin, args.wavlm, args.conformer, wave_path, str(n), out_path]
        if args.tap == "raw":
            cmd.append("--raw")
        print("running:", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"harness failed: rc={r.returncode}")
            return 1

        with open(out_path, "rb") as fh:
            hdr = fh.read(12)
            nch, nfr, nsp = struct.unpack("<iii", hdr)
            got = np.frombuffer(fh.read(), dtype=np.float32)

    print(f"native: chunks={nch} frames={nfr} speakers={nsp}")
    if (nch, nfr, nsp) != ref.shape:
        print(f"shape mismatch: native {(nch, nfr, nsp)} vs ref {ref.shape}")
        return 1
    got = got.reshape(nch, nfr, nsp)

    a = got.reshape(-1)
    b = ref.reshape(-1)
    match = float(np.mean(a == b))
    diff = int(np.sum(a != b))
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"exact_match={match:.6f}  mismatched_frames={diff}/{a.size}  "
          f"cosine={cos:.6f}")

    ok = match >= args.tol_match
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
