#!/usr/bin/env python3
# TODO(native-cuda-port): produces the bit-equality ground truth for
# diarizen_wavlm_pruned (P1a), diarizen_conformer_head (P1b), and
# diarizen_classifier (P1c). Lives under tools/ because it runs in the
# py310_diarizen env, not in awaken. See docs/{en,zh}/architecture/
# 12-diarizen.md "Architectural anchor".
"""Dump reference activations from the live DiariZen-v2 pipeline.

Produces a single .npz file containing intermediate tensors that the
native CUDA port must reproduce. Input is a fixed 16-second window
from tests/test.mp3 starting at a configurable offset (default 0.0 s)
to keep the fixture small and deterministic.

The keys written into the .npz are the C++ verification targets:

    wave_in              float32 [1, 256000]   16 kHz mono input
    wavlm_lnorm_out      float32 [1, T, 256]   WavLM-pruned final tap
                                                (post weight_sum + proj +
                                                 lnorm) - P1a target
    conformer_out        float32 [1, T, 256]   Conformer head output -
                                                P1b target
    classifier_logits    float32 [1, T, 16]    Powerset logits -
                                                P1c target (pre-sigmoid)
    classifier_probs     float32 [1, T, 16]    After sigmoid - what the
                                                pipeline downstream uses
    layer_hiddens        float32 [25, 1, T, 1024]  CNN + 24 transformer
                                                taps before weight_sum.
                                                Used for P1a per-layer
                                                drift inspection.

Run inside the py310_diarizen env:

    conda activate py310_diarizen
    python tools/diarizen_dump_reference.py \
        --audio tests/test.mp3 \
        --offset 0.0 --duration 16.0 \
        --out tests/fixtures/diarizen_p1a_reference.npz

The output file is NOT committed - it is regenerable. The script is
committed because it pins the exact recipe.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audio", required=True, help="path to test.mp3 (or wav)")
    p.add_argument(
        "--model",
        default="BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        help="HF model id used by DiariZenPipeline.from_pretrained",
    )
    p.add_argument("--offset", type=float, default=0.0, help="start in seconds")
    p.add_argument(
        "--duration",
        type=float,
        default=16.0,
        help="window length in seconds (default matches segmentation chunk)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="output .npz path (overwritten)",
    )
    p.add_argument(
        "--dump-heads",
        default=None,
        help="if set, also write the per-layer remaining_heads table to "
             "this JSON path (24 int lists). These indices are NOT in the "
             "safetensors but the native CUDA forward needs them to align "
             "the gated relative-position bias with the pruned attention "
             "heads. Anti-entropy: the loader reads this sidecar instead of "
             "hardcoding the table.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="torch device (default cuda)",
    )
    return p.parse_args()


def _load_audio_window(path: str, offset: float, duration: float):
    """Load mono 16 kHz float32 window via soundfile (no torchaudio shim)."""
    import numpy as np
    import soundfile as sf

    info = sf.info(path)
    if info.samplerate != 16000:
        # Resample on the fly via librosa to stay deterministic.
        import librosa

        wav, _ = librosa.load(
            path,
            sr=16000,
            mono=True,
            offset=offset,
            duration=duration,
        )
    else:
        start = int(round(offset * 16000))
        nsamp = int(round(duration * 16000))
        wav, _ = sf.read(path, start=start, frames=nsamp, dtype="float32",
                         always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32)


def main() -> int:
    args = _parse_args()
    audio_path = Path(args.audio).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch

    # The DiariZen pipeline does torch.load with weights_only default True
    # in newer torch; the worker patches this. Mirror that patch here so
    # the dump tool runs standalone.
    _orig_torch_load = torch.load

    def _patched_load(*a, **kw):  # noqa: ANN001
        kw["weights_only"] = False
        return _orig_torch_load(*a, **kw)

    torch.load = _patched_load  # type: ignore[assignment]

    from diarizen.pipelines.inference import DiariZenPipeline  # noqa: PLC0415

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    print(f"[dump] loading {args.model} on {device}", file=sys.stderr)
    pipe = DiariZenPipeline.from_pretrained(args.model).to(device)
    # `pipe.model` is the DiariZen segmentation Model (BaseModel subclass).
    model = pipe.model
    model.eval()

    # Optionally export the per-layer remaining_heads table. These are the
    # original (0..15) head indices kept after structured pruning; they are
    # non-contiguous and absent from the safetensors. The native CUDA loader
    # reads this sidecar to select the gated relative-position bias slices
    # that align with each layer's surviving q/k/v heads.
    if args.dump_heads:
        import json  # noqa: PLC0415

        heads_path = Path(args.dump_heads).resolve()
        layers = model.wavlm_model.encoder.transformer.layers
        table = []
        for layer in layers:
            att = getattr(layer, "attention", None)
            if att is None:
                table.append([])  # attention sub-block fully pruned
            else:
                table.append([int(h) for h in att.remaining_heads])
        heads_path.parent.mkdir(parents=True, exist_ok=True)
        with open(heads_path, "w", encoding="utf-8") as fh:
            json.dump({"remaining_heads": table}, fh, indent=2)
        print(f"[dump] wrote remaining_heads -> {heads_path}", file=sys.stderr)

    print(f"[dump] loading audio window: offset={args.offset}s "
          f"duration={args.duration}s", file=sys.stderr)
    wav = _load_audio_window(str(audio_path), args.offset, args.duration)
    # DiariZen expects shape (batch, channel, sample); selected_channel
    # picks channel 0. Use 1-channel here.
    wav_t = torch.from_numpy(wav).to(device).unsqueeze(0).unsqueeze(0)  # [1,1,N]

    # Hook collection ----------------------------------------------------
    captured: dict[str, torch.Tensor] = {}

    def _cap(name: str):
        def _h(_mod, _inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            captured[name] = t.detach().to("cpu", torch.float32)
        return _h

    handles = [
        model.wavlm_model.feature_extractor.register_forward_hook(
            _cap("cnn_out")),
        model.lnorm.register_forward_hook(_cap("wavlm_lnorm_out")),
        model.conformer.register_forward_hook(_cap("conformer_out")),
        model.classifier.register_forward_hook(_cap("classifier_logits")),
        model.activation.register_forward_hook(_cap("classifier_probs")),
    ]

    # We also want per-layer WavLM reps. extract_features returns
    # (layer_reps_list, padding_mask); stash by re-calling the same path.
    with torch.no_grad():
        layer_reps, _ = model.wavlm_model.extract_features(
            wav_t[:, model.selected_channel, :]
        )
        # layer_reps: tuple of 25 tensors of shape [1, T, 1024]
        layer_hiddens = torch.stack(layer_reps, dim=0)  # [25, 1, T, 1024]

        # Full forward to fire hooks.
        _ = model(wav_t)

    for h in handles:
        h.remove()

    # Save --------------------------------------------------------------
    out = {
        "wave_in": wav_t.detach().to("cpu", torch.float32).numpy(),
        "layer_hiddens": layer_hiddens.detach().to("cpu", torch.float32).numpy(),
    }
    for k, t in captured.items():
        out[k] = t.numpy()

    np.savez_compressed(out_path, **out)

    print(f"[dump] wrote {out_path}", file=sys.stderr)
    for k, v in out.items():
        print(f"        {k:24s} shape={list(v.shape)} dtype={v.dtype}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
