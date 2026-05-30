#!/usr/bin/env python3
"""P3a get_embeddings bit-equality driver for the native DiariZen pipeline.

Exports the waveform + binarized segmentation from a diarizen_dump_pipeline
fixture, runs the native ``test_diarizen_pipeline_biteq`` harness, and compares
the resulting per-(chunk, speaker) embeddings against the fixture's reference
embeddings (cosine + max-abs-diff, split by active/inactive speaker rows).

Mechanical comparison only (cosine of fp32 vectors against a deterministic
reference): permitted under workflow.instructions.md because this is a
bit-equality check against a fixed Python baseline, not a semantic quality
score.

Usage:
  python3 tools/diarizen_bit_eq_pipeline.py [--fixture PATH] [--harness PATH]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


def _cos(a, b):
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / (na * nb + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fixture",
        default="tests/fixtures/diarizen_p3a_pipeline.npz",
        help="diarizen_dump_pipeline.py fixture npz",
    )
    ap.add_argument(
        "--harness",
        default="build/test_diarizen_pipeline_biteq",
        help="native get_embeddings harness binary",
    )
    args = ap.parse_args()

    d = np.load(args.fixture)
    wave = np.ascontiguousarray(d["wave_in"], dtype=np.float32)
    seg = np.ascontiguousarray(d["seg_data"], dtype=np.float32)  # [C, F, S]
    ref = np.ascontiguousarray(d["embeddings"], dtype=np.float32)  # [C, S, D]
    C, F, S = seg.shape
    D = ref.shape[-1]

    with tempfile.TemporaryDirectory() as tmp:
        wp = os.path.join(tmp, "wave.bin")
        sp = os.path.join(tmp, "seg.bin")
        ep = os.path.join(tmp, "emb.bin")
        wave.tofile(wp)
        seg.tofile(sp)
        r = subprocess.run(
            [args.harness, "embeddings", wp, sp, str(C), str(F), str(S), ep],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            return 1
        emb = np.fromfile(ep, dtype=np.float32).reshape(C, S, D)

        # --- P3a-4: reconstruct + speaker_count + to_diarization -------------
        hard = np.ascontiguousarray(d["hard_clusters"], dtype=np.int32)  # [C,S]
        hp = os.path.join(tmp, "hard.bin")
        cp = os.path.join(tmp, "count.bin")
        bp = os.path.join(tmp, "binary.bin")
        hard.tofile(hp)
        r2 = subprocess.run(
            [args.harness, "postproc", sp, hp, str(C), str(F), str(S), cp, bp],
            capture_output=True,
            text=True,
        )
        if r2.returncode != 0:
            sys.stderr.write(r2.stderr)
            return 1
        ref_count = np.ascontiguousarray(d["count_data"]).reshape(-1)
        ref_disc = np.ascontiguousarray(d["discrete_data"])  # [nf, ncl]
        nf, ncl = ref_disc.shape
        nat_count = np.fromfile(cp, dtype=np.float32).astype(np.uint8)
        nat_disc = np.fromfile(bp, dtype=np.float32).reshape(nf, ncl)

        # --- P3a-5: end-to-end diarize from wave_in --------------------------
        op = os.path.join(tmp, "segs.bin")
        r3 = subprocess.run(
            [args.harness, "diarize", wp, op],
            capture_output=True,
            text=True,
        )
        e2e_ok = r3.returncode == 0
        e2e_msg = ""
        if e2e_ok:
            nat_seg = np.fromfile(op, dtype=np.float32).reshape(-1, 3)
            ref_seg = np.ascontiguousarray(d["segments"])
            ref_lab = np.ascontiguousarray(d["segment_labels"])
            no = np.lexsort((nat_seg[:, 2], nat_seg[:, 1], nat_seg[:, 0]))
            bo = np.lexsort((ref_lab, ref_seg[:, 1], ref_seg[:, 0]))
            nat_seg = nat_seg[no]
            rs, rl = ref_seg[bo], ref_lab[bo]
            same_n = len(nat_seg) == len(rs)
            if same_n:
                lab_ok = np.array_equal(nat_seg[:, 2].astype(int), rl.astype(int))
                bdiff = float(
                    max(
                        np.abs(nat_seg[:, 0] - rs[:, 0]).max(),
                        np.abs(nat_seg[:, 1] - rs[:, 1]).max(),
                    )
                )
                e2e_ok = lab_ok and bdiff <= 0.021  # <=1 frame (0.02 s)
                e2e_msg = (
                    f"segs={len(nat_seg)}/{len(rs)} labels_match={lab_ok} "
                    f"max_boundary_diff={bdiff:.4f}s"
                )
            else:
                e2e_ok = False
                e2e_msg = f"segment count mismatch {len(nat_seg)} vs {len(rs)}"
        else:
            e2e_msg = r3.stderr.strip().splitlines()[-1] if r3.stderr else "fail"

    active = seg.sum(axis=1) > 0  # [C, S]
    cs = _cos(emb.reshape(-1, D), ref.reshape(-1, D)).reshape(C, S)
    ca, ci = cs[active], cs[~active]
    print(f"chunks={C} speakers={S} dim={D}")
    print(
        f"active   rows={int(active.sum()):3d} "
        f"min_cos={float(ca.min()):.6f} mean_cos={float(ca.mean()):.6f} "
        f"max_abs_diff={float(np.abs(emb[active]-ref[active]).max()):.3e}"
    )
    print(
        f"inactive rows={int((~active).sum()):3d} "
        f"min_cos={float(ci.min()):.6f} mean_cos={float(ci.mean()):.6f} "
        f"max_abs_diff={float(np.abs(emb[~active]-ref[~active]).max()):.3e}"
    )
    count_ok = np.array_equal(nat_count, ref_count.astype(np.uint8))
    disc_diff = int((nat_disc != ref_disc).sum())
    print(
        f"postproc count_match={count_ok} discrete_diff={disc_diff} "
        f"(nf={nf} ncl={ncl})"
    )
    print(f"end2end  ok={e2e_ok} {e2e_msg}")
    ok = (
        float(ca.min()) >= 0.999
        and float(ci.min()) >= 0.999
        and count_ok
        and disc_diff == 0
        and e2e_ok
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
