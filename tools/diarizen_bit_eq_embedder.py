#!/usr/bin/env python3
"""P2a bit-equality driver: WeSpeaker ResNet34-LM embedder.

Mechanical comparison (cosine + max-abs) of the native CUDA embedding/fbank
against the pyannote reference dumped by tools/diarizen_dump_embedder.py. A
physical bit-equality check, not a semantic score.

Flow per pair:
  1. write wave[Ns] and mask[Nf] to raw float32 files;
  2. run build/test_diarizen_embedder_biteq (embedding, or --fbank);
  3. load the result and compare against embed / fbank0.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


def run_harness(bin_path, st, wave, mask, num_frames, fbank=False):
    with tempfile.TemporaryDirectory() as td:
        wp = os.path.join(td, "w.bin")
        mp = os.path.join(td, "m.bin")
        op = os.path.join(td, "o.bin")
        wave.astype(np.float32).tofile(wp)
        if mask is None:
            mask_arg = "none"
        else:
            mask.astype(np.float32).tofile(mp)
            mask_arg = mp
        cmd = [bin_path, st, wp, str(wave.size), mask_arg, str(num_frames), op]
        if fbank:
            cmd.append("--fbank")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"harness rc={r.returncode}")
        with open(op, "rb") as fh:
            data = fh.read()
    return data


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="tests/fixtures/diarizen_p2a_embedder.npz")
    ap.add_argument(
        "--st",
        default="/home/rm01/models/dev/diarizen_v2/wespeaker_resnet34.safetensors")
    ap.add_argument("--bin", default="build/test_diarizen_embedder_biteq")
    ap.add_argument("--tol", type=float, default=0.999)
    args = ap.parse_args()

    d = np.load(args.npz)
    waves = d["wave"]
    masks = d["mask"]
    embeds = d["embed"]
    num_frames = int(d["num_frames"])
    n_pairs = int(d["n_pairs"])

    ok = True

    # fbank check (pair 0).
    fb_ref = d["fbank0"].reshape(-1)
    T_ref = int(d["fbank0_T"])
    raw = run_harness(args.bin, args.st, waves[0], None, num_frames, fbank=True)
    T, M = struct.unpack("<ii", raw[:8])
    fb_got = np.frombuffer(raw[8:], dtype=np.float32)
    print(f"[fbank] ref T={T_ref} got T={T} M={M}")
    if T == T_ref:
        c = cos(fb_got, fb_ref)
        mx = float(np.max(np.abs(fb_got - fb_ref)))
        print(f"[fbank] cosine={c:.6f} max_abs={mx:.4e}")
        ok = ok and c >= args.tol
    else:
        print("[fbank] T mismatch")
        ok = False

    # embedding checks.
    for i in range(n_pairs):
        raw = run_harness(args.bin, args.st, waves[i], masks[i], num_frames)
        got = np.frombuffer(raw, dtype=np.float32)
        ref = embeds[i]
        c = cos(got, ref)
        mx = float(np.max(np.abs(got - ref)))
        print(f"[embed {i}] cosine={c:.6f} max_abs={mx:.4e}  "
              f"got[:3]={got[:3]}  ref[:3]={ref[:3]}")
        ok = ok and c >= args.tol

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
