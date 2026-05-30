#!/usr/bin/env python3
"""P2b bit-equality driver for the DiariZen VBx clustering stage.

Stages: fea (PLDA feature transform), ahc (AHC labels), hard (full
hard_clusters). The fea transform is compared per-column with sign alignment
because the generalized eigensolver fixes eigenvectors only up to a sign (the
VBx EM and final hard_clusters are invariant to per-column fea sign flips).
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


def run(bin_path, plda_dir, mode, args):
    with tempfile.TemporaryDirectory() as td:
        op = os.path.join(td, "o.bin")
        cmd = [bin_path, plda_dir, mode, op] + [str(x) for x in args]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"harness rc={r.returncode}")
        with open(op, "rb") as fh:
            return fh.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="tests/fixtures/diarizen_p2b_clustering.npz")
    ap.add_argument("--plda", default="/home/rm01/models/dev/diarizen_v2",
                    help="dir containing xvec_transform.npz and plda.npz")
    ap.add_argument("--bin", default="build/test_diarizen_clustering_biteq")
    ap.add_argument("--stage", default="fea", choices=("fea", "ahc", "hard"))
    args = ap.parse_args()

    d = np.load(args.npz)
    ok = True

    if args.stage == "fea":
        te = d["train_emb"].astype(np.float32)
        N, X = te.shape
        with tempfile.TemporaryDirectory() as td:
            wp = os.path.join(td, "te.bin")
            te.tofile(wp)
            raw = run(args.bin, args.plda, "--fea", [wp, N, X])
        got = np.frombuffer(raw, dtype=np.float32).reshape(N, -1)
        ref = d["fea"].astype(np.float32)
        # sign-align per column (eigenvector sign ambiguity).
        sign = np.sign(np.sum(got * ref, axis=0))
        sign[sign == 0] = 1.0
        got_a = got * sign[None, :]
        mx = float(np.max(np.abs(got_a - ref)))
        cs = float(np.mean([np.dot(got_a[i], ref[i]) /
                            (np.linalg.norm(got_a[i]) * np.linalg.norm(ref[i]) + 1e-12)
                            for i in range(N)]))
        print(f"[fea] N={N} pdim={got.shape[1]} mean_row_cosine={cs:.6f} "
              f"max_abs(sign-aligned)={mx:.4e}")
        ok = cs >= 0.999

    elif args.stage == "ahc":
        te = d["train_emb"].astype(np.float32)
        N, X = te.shape
        with tempfile.TemporaryDirectory() as td:
            wp = os.path.join(td, "te.bin")
            te.tofile(wp)
            raw = run(args.bin, args.plda, "--ahc", [wp, N, X])
        got = np.frombuffer(raw, dtype=np.int32)
        ref = d["ahc"].astype(np.int32)
        # compare as partitions (label ids are canonical 0..K-1 by first-seen)
        agree = float(np.mean(got == ref))
        print(f"[ahc] N={N} exact_label_agreement={agree:.6f} "
              f"K_got={int(got.max())+1 if got.size else 0} K_ref={int(ref.max())+1}")
        ok = agree == 1.0

    else:  # hard
        emb = d["embeddings"].astype(np.float32)
        seg = d["seg"].astype(np.float32)
        C, S, D = emb.shape
        F = seg.shape[1]
        with tempfile.TemporaryDirectory() as td:
            ep = os.path.join(td, "e.bin")
            sp = os.path.join(td, "s.bin")
            emb.tofile(ep)
            seg.tofile(sp)
            raw = run(args.bin, args.plda, "--hard", [ep, C, S, D, sp, F])
        got = np.frombuffer(raw, dtype=np.int8).reshape(C, S)
        ref = d["hard"].astype(np.int8)
        agree = float(np.mean(got == ref))
        print(f"[hard] C={C} S={S} exact_agreement={agree:.6f}")
        ok = agree == 1.0

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
