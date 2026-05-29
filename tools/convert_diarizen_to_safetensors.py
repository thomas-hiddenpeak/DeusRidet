#!/usr/bin/env python3
"""Convert DiariZen-v2 + WeSpeaker-ResNet34-LM weights to project format.

P0 deliverable from docs/{en,zh}/architecture/12-diarizen.md.

Reads HF checkpoints (must be cached locally; we don't network here).
Splits the monolithic 599-tensor BUT-FIT/diarizen-v2 .bin into two
logical safetensors files and copies the PLDA bundle verbatim:

  ~/models/dev/diarizen_v2/
      wavlm_pruned.safetensors        WavLM-large encoder (structured-
                                       pruned by BUT-FIT) + weight_sum
                                       + proj + lnorm prefixes
      conformer_head.safetensors      4-layer Conformer + classifier
      wespeaker_resnet34.safetensors  pyannote/wespeaker-voxceleb-resnet34-LM
      plda.bin                        verbatim copy of HF `plda` artifact
      shapes.json                     per-tensor shape index for the
                                       C++ loader (so we don't need
                                       safetensors header parsing yet)

Run inside the existing env:
    /home/rm01/miniconda3/envs/py310_diarizen/bin/python \\
        tools/convert_diarizen_to_safetensors.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import torch  # noqa: E402

# torch>=2.6 forces weights_only=True; BUT-FIT bin trips on TorchVersion
_orig_load = torch.load
def _load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _load  # noqa: E305

from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

OUT_ROOT = Path(os.environ.get(
    "DIARIZEN_V2_OUT", "/home/rm01/models/dev/diarizen_v2"
)).expanduser()

DIARIZEN_REPO = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
WESPEAKER_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"


def split_diarizen(state: dict) -> tuple[dict, dict]:
    """Partition BUT-FIT v2 state-dict into (wavlm_part, head_part).

    wavlm_part: wavlm_model.*, weight_sum, proj.*, lnorm.* (encoder side)
    head_part:  conformer.*, classifier.*                  (decision side)
    """
    wavlm, head = {}, {}
    for k, v in state.items():
        if k.startswith("wavlm_model.") or k.startswith("weight_sum") or \
           k.startswith("proj.") or k.startswith("lnorm."):
            wavlm[k] = v.contiguous().to(torch.float16)
        elif k.startswith("conformer.") or k.startswith("classifier."):
            head[k] = v.contiguous().to(torch.float16)
        else:
            raise ValueError(f"unexpected key prefix: {k}")
    return wavlm, head


def dump_shapes(*groups: dict, path: Path) -> None:
    """Write a JSON index of (name → shape, dtype) for the C++ loader."""
    index: dict[str, dict] = {}
    for g in groups:
        for k, v in g.items():
            index[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
    path.write_text(json.dumps(index, indent=2, sort_keys=True))


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[out] {OUT_ROOT}", flush=True)

    # ------------------------------------------------------------------
    # DiariZen-v2
    # ------------------------------------------------------------------
    print(f"[diarizen] snapshot {DIARIZEN_REPO}", flush=True)
    d_snap = Path(snapshot_download(DIARIZEN_REPO))
    bin_path = d_snap / "pytorch_model.bin"
    plda_path = d_snap / "plda"
    if not bin_path.is_file():
        print(f"[err] missing {bin_path}", file=sys.stderr)
        return 2

    state = torch.load(bin_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    wavlm, head = split_diarizen(state)
    print(f"[diarizen] wavlm tensors={len(wavlm)}  head tensors={len(head)}",
          flush=True)

    wavlm_out = OUT_ROOT / "wavlm_pruned.safetensors"
    head_out = OUT_ROOT / "conformer_head.safetensors"
    save_file(wavlm, str(wavlm_out))
    save_file(head, str(head_out))
    print(f"[diarizen] wrote {wavlm_out} ({wavlm_out.stat().st_size/1e6:.1f} MB)",
          flush=True)
    print(f"[diarizen] wrote {head_out} ({head_out.stat().st_size/1e6:.1f} MB)",
          flush=True)

    # PLDA bundle: it's a directory containing two small .npz files.
    # They are the input to VBx clustering (xvec LDA + PLDA priors).
    # Copy verbatim; the C++ side will unpack them when first used.
    if plda_path.is_dir():
        for name in ("xvec_transform.npz", "plda.npz"):
            src = plda_path / name
            if not src.is_file():
                print(f"[plda] WARNING: {src} missing", flush=True)
                continue
            dst = OUT_ROOT / name
            shutil.copyfile(src, dst)
            print(f"[plda] copied {dst} ({dst.stat().st_size} B)", flush=True)
    else:
        print(f"[plda] WARNING: {plda_path} missing or not a directory",
              flush=True)

    # ------------------------------------------------------------------
    # WeSpeaker ResNet34-LM
    # ------------------------------------------------------------------
    print(f"[wespeaker] snapshot {WESPEAKER_REPO}", flush=True)
    w_snap = Path(snapshot_download(WESPEAKER_REPO))
    w_bin = w_snap / "pytorch_model.bin"
    if not w_bin.is_file():
        print(f"[err] missing {w_bin}", file=sys.stderr)
        return 3
    w_state = torch.load(w_bin, map_location="cpu")
    if isinstance(w_state, dict) and "state_dict" in w_state:
        w_state = w_state["state_dict"]
    w_state_fp16 = {k: v.contiguous().to(torch.float16) for k, v in w_state.items()}
    w_out = OUT_ROOT / "wespeaker_resnet34.safetensors"
    save_file(w_state_fp16, str(w_out))
    print(f"[wespeaker] wrote {w_out} ({w_out.stat().st_size/1e6:.1f} MB)  "
          f"tensors={len(w_state_fp16)}", flush=True)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    idx = OUT_ROOT / "shapes.json"
    dump_shapes(wavlm, head, w_state_fp16, path=idx)
    print(f"[idx] wrote {idx}", flush=True)

    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
