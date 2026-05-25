#!/usr/bin/env python3
"""Phase 12 — fused fixture dumper for overlap-aware (dominant-source) PCM.

Identical to dump_fused_fixture.py, but reads
tests/fixtures/{cam,wl}_embeddings_v2.{f32,meta.json} (re-extracted from
/tmp/test_mp3_16k_mono_overlap_dom.f32, where 193 overlap segments have
been replaced by their MossFormer2 dominant-source channel) and writes
tests/fixtures/fused_v2_dominant.bin.
"""
import json, struct, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIX  = ROOT / "tests" / "fixtures"

CAM_F32  = FIX / "cam_embeddings_v2.f32"
CAM_META = FIX / "cam_embeddings_v2.meta.json"
WL_F32   = FIX / "wl_embeddings_v2.f32"
WL_META  = FIX / "wl_embeddings_v2.meta.json"
OUT      = FIX / "fused_v2_dominant.bin"

def load(meta_path, f32_path):
    meta = json.loads(meta_path.read_text())
    n = meta["n_segments"]
    s = len(meta["strategies"])
    d = meta["dim"]
    arr = np.fromfile(f32_path, dtype=np.float32).reshape(n, s, d)
    return meta, arr

def l2(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)

def main():
    cam_meta, cam = load(CAM_META, CAM_F32)
    wl_meta,  wl  = load(WL_META,  WL_F32)
    if cam_meta["strategies"] != wl_meta["strategies"]:
        print("ERROR: strategy lists differ", file=sys.stderr); sys.exit(1)
    if cam_meta["n_segments"] != wl_meta["n_segments"]:
        print("ERROR: n_segments differ", file=sys.stderr); sys.exit(1)
    if cam_meta["dim"] != wl_meta["dim"]:
        print("ERROR: dim differ", file=sys.stderr); sys.exit(1)

    strat = "full"
    si = cam_meta["strategies"].index(strat)
    cam_v = l2(cam[:, si, :])
    wl_v  = l2(wl[:,  si, :])

    fused = np.concatenate([cam_v, wl_v], axis=1).astype(np.float32)
    fused = l2(fused)
    dim = fused.shape[1]
    n   = fused.shape[0]
    assert dim == 384

    segs = cam_meta["segments"]
    spk_names = sorted({s["speaker"] for s in segs})
    spk_to_idx = {name: i for i, name in enumerate(spk_names)}

    order = sorted(range(n), key=lambda i: segs[i]["start_ms"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as f:
        f.write(struct.pack("<IiiiI", 0x4F524554, n, dim, len(spk_names), si))
        for i in order:
            s = segs[i]
            t_start = s["start_ms"] / 1000.0
            t_end   = s["end_ms"]   / 1000.0
            t_center = 0.5 * (t_start + t_end)
            gt = spk_to_idx[s["speaker"]]
            f.write(struct.pack("<dddi", t_center, t_start, t_end, gt))
            fused[i].tofile(f)

    sidx = FIX / "fused_v2_dominant.speakers.txt"
    sidx.write_text("\n".join(spk_names) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes), n={n} dim={dim} K_gt={len(spk_names)}")
    print(f"wrote {sidx}: {spk_names}")

if __name__ == "__main__":
    main()
