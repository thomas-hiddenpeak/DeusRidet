#!/usr/bin/env python3
"""Run upstream ClearVoice/FunASR/3D-Speaker against Auditus clips."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]*\|>")
STREAMS = ("mix", "src1", "src2")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_clean_state(skip: bool) -> None:
    if skip:
        return
    subprocess.run("sudo kill -9 $(pgrep -f deusridet) 2>/dev/null || true", shell=True, check=False)
    subprocess.run("sudo fuser -k 8080/tcp 2>/dev/null || true", shell=True, check=False)
    subprocess.run("echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null", shell=True, check=False)
    check = subprocess.run("pgrep -x deusridet", shell=True, text=True, capture_output=True, check=False)
    if check.stdout.strip():
        raise RuntimeError(f"deusridet process still present: {check.stdout.strip()}")


def clip_base(clip_name: str) -> str:
    return clip_name[:-4] if clip_name.endswith(".wav") else clip_name


def stream_name(path: Path) -> str:
    name = path.name
    for stream in STREAMS:
        if name.endswith(f"_{stream}.wav"):
            return stream
    return "unknown"


def clip_from_stream_path(path: Path) -> str:
    name = path.name
    for stream in STREAMS:
        suffix = f"_{stream}.wav"
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".wav"
    return name


def audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def load_manifest(path: Path, clips_dir: Path, limit: int) -> list[dict[str, Any]]:
    rows = sorted(read_jsonl(path), key=lambda item: int(item.get("rank", 0)))
    if limit > 0:
        rows = [row for row in rows if int(row.get("rank", 0)) <= limit]
    for row in rows:
        local_clip = clips_dir / str(row["clip_name"])
        if local_clip.is_file():
            row["resolved_clip_path"] = str(local_clip)
        else:
            row["resolved_clip_path"] = str(row.get("clip_path", local_clip))
    return rows


def stage_clearvoice_inputs(manifest: list[dict[str, Any]], out_dir: Path) -> Path:
    input_dir = out_dir / "clearvoice_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        src = Path(str(row["resolved_clip_path"]))
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = input_dir / str(row["clip_name"])
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    return input_dir


def run_clearvoice(args: argparse.Namespace, manifest: list[dict[str, Any]]) -> Path:
    audio_dir = args.out_dir / "upstream_clearvoice" / "audio"
    expected = [audio_dir / f"{clip_base(row['clip_name'])}_src2.wav" for row in manifest]
    if not args.force and expected and all(path.is_file() for path in expected):
        print(f"[clearvoice] reuse {audio_dir}")
        return audio_dir

    input_dir = stage_clearvoice_inputs(manifest, args.out_dir)
    raw_out = args.out_dir / "upstream_clearvoice" / "clearvoice_raw"
    if raw_out.exists() and args.force:
        shutil.rmtree(raw_out)
    audio_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.upstream_root / "ClearerVoice-Studio" / "clearvoice"))
    import torch
    from clearvoice import ClearVoice
    import clearvoice.networks as networks

    if args.force_clearvoice_gpu:
        networks.SpeechModel.get_free_gpu = lambda self: 0 if torch.cuda.is_available() else None

    cwd = Path.cwd()
    os.chdir(args.upstream_root / "ClearerVoice-Studio" / "clearvoice")
    try:
        model = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
        model(input_path=str(input_dir), online_write=True, output_path=str(raw_out))
    finally:
        os.chdir(cwd)

    produced_dir = raw_out / "MossFormer2_SS_16K"
    for row in manifest:
        base = clip_base(str(row["clip_name"]))
        mix_dst = audio_dir / f"{base}_mix.wav"
        shutil.copy2(str(row["resolved_clip_path"]), mix_dst)
        for index, stream in [(1, "src1"), (2, "src2")]:
            src = produced_dir / f"{base}_s{index}.wav"
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, audio_dir / f"{base}_{stream}.wav")
    print(f"[clearvoice] wrote {audio_dir}")
    return audio_dir


def collect_variant_audio(audio_dir: Path, manifest: list[dict[str, Any]], include_mix: bool) -> list[Path]:
    paths: list[Path] = []
    wanted = STREAMS if include_mix else ("src1", "src2")
    for row in manifest:
        base = clip_base(str(row["clip_name"]))
        for stream in wanted:
            path = audio_dir / f"{base}_{stream}.wav"
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
    return paths


def clean_asr_text(raw_text: Any) -> str:
    text = SENSEVOICE_TAG_RE.sub("", str(raw_text or ""))
    return re.sub(r"\s+", " ", text).strip()


def run_funasr_variant(args: argparse.Namespace, variant: str, audio_dir: Path,
                       manifest: list[dict[str, Any]]) -> Path:
    out_dir = args.out_dir / variant
    jsonl_path = out_dir / "asr_sources.jsonl"
    if jsonl_path.is_file() and not args.force:
        print(f"[funasr:{variant}] reuse {jsonl_path}")
        return jsonl_path
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.upstream_root / "FunASR"))
    import torch
    from funasr import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("FunASR would run without CUDA; aborting")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        device=args.device,
        disable_update=True,
        log_level="ERROR",
        ncpu=args.ncpu,
    )
    paths = collect_variant_audio(audio_dir, manifest, include_mix=True)
    with jsonl_path.open("w", encoding="utf-8") as output:
        for index, path in enumerate(paths, 1):
            start = time.perf_counter()
            result = model.generate(
                input=str(path),
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            total_ms = (time.perf_counter() - start) * 1000.0
            raw_text = ""
            if result and isinstance(result, list):
                raw_text = str(result[0].get("text", ""))
            text = clean_asr_text(raw_text)
            row = {
                "clip": clip_from_stream_path(path),
                "stream": stream_name(path),
                "file": str(path),
                "duration_sec": round(audio_duration(path), 3),
                "text": text,
                "raw_text": raw_text,
                "total_ms": round(total_ms, 3),
                "tokens": len(text),
                "engine": "FunASR/SenseVoiceSmall",
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[funasr:{variant} {index:03d}/{len(paths):03d}] {row['clip']}/{row['stream']} {text[:80]}")
    return jsonl_path


def ffmpeg_pcm(path: Path) -> np.ndarray:
    data = subprocess.check_output([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-",
    ])
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def center_window(start: int, count: int, window_sec: float, total: int) -> tuple[int, int]:
    left = max(0, start)
    right = min(total, start + count)
    clipped = max(0, right - left)
    wanted = int(window_sec * SAMPLE_RATE)
    if window_sec <= 0.0 or clipped <= wanted:
        return left, clipped
    center = left + clipped // 2
    out_start = max(0, center - wanted // 2)
    if out_start + wanted > total:
        out_start = max(0, total - wanted)
    return out_start, min(wanted, total - out_start)


def source_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class ThreeDSpeaker:
    def __init__(self, args: argparse.Namespace):
        sys.path.insert(0, str(args.upstream_root / "3D-Speaker"))
        import torch
        from modelscope.hub.snapshot_download import snapshot_download
        from speakerlab.process.processor import FBank
        from speakerlab.utils.builder import dynamic_import

        infer_mod = load_module("upstream_3dspeaker_infer_sv", args.upstream_root / "3D-Speaker" / "speakerlab" / "bin" / "infer_sv.py")
        conf = infer_mod.supports[args.speaker_model_id]
        cache_dir = Path(snapshot_download(args.speaker_model_id, revision=conf["revision"]))
        model_path = cache_dir / conf["model_pt"]
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("3D-Speaker would run without CUDA; aborting")
        self.torch = torch
        self.feature_extractor = FBank(80, sample_rate=SAMPLE_RATE, mean_nor=True)
        model_conf = conf["model"]
        self.model = dynamic_import(model_conf["obj"])(**model_conf["args"])
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

    def embedding(self, samples: np.ndarray) -> np.ndarray:
        wav = self.torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        feat = self.feature_extractor(wav).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            emb = self.model(feat).detach().squeeze(0).cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / max(norm, 1e-12)


def load_wav_16k(path: Path) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio.functional as F

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.ascontiguousarray(data[:, 0])
    if sample_rate != SAMPLE_RATE:
        wav = torch.from_numpy(mono).unsqueeze(0)
        mono = F.resample(wav, sample_rate, SAMPLE_RATE).squeeze(0).numpy()
    return mono.astype(np.float32, copy=False)


def gt_overlaps_manifest(row: dict[str, Any], windows: list[tuple[float, float]]) -> bool:
    start = float(row["start_ms"]) / 1000.0
    end = float(row["end_ms"]) / 1000.0
    return any(overlap_sec(start, end, left, right) >= 0.05 for left, right in windows)


def gt_has_other_overlap(row: dict[str, Any], gt_rows: list[dict[str, Any]]) -> bool:
    start = float(row["start_ms"]) / 1000.0
    end = float(row["end_ms"]) / 1000.0
    speaker = str(row["speaker"])
    idx = int(row["idx"])
    for other in gt_rows:
        if int(other["idx"]) == idx or str(other["speaker"]) == speaker:
            continue
        other_start = float(other["start_ms"]) / 1000.0
        other_end = float(other["end_ms"]) / 1000.0
        if overlap_sec(start, end, other_start, other_end) >= 0.05:
            return True
    return False


def build_reference_bank(args: argparse.Namespace, manifest: list[dict[str, Any]], encoder: ThreeDSpeaker) -> dict[str, list[np.ndarray]]:
    gt_rows = read_jsonl(args.gt)
    windows = [(float(row["clip_start_sec"]), float(row["clip_end_sec"])) for row in manifest]
    reference_pcm = ffmpeg_pcm(args.reference_audio)
    bank: dict[str, list[np.ndarray]] = defaultdict(list)
    attempted = 0
    for row in gt_rows:
        speaker = str(row["speaker"])
        if len(bank[speaker]) >= args.max_ref_per_speaker:
            continue
        duration_ms = int(row["end_ms"]) - int(row["start_ms"])
        if duration_ms < args.ref_min_ms:
            continue
        if gt_overlaps_manifest(row, windows) or gt_has_other_overlap(row, gt_rows):
            continue
        start, count = center_window(
            int(int(row["start_ms"]) * SAMPLE_RATE / 1000),
            int(duration_ms * SAMPLE_RATE / 1000),
            args.ref_window_sec,
            int(reference_pcm.size),
        )
        segment = reference_pcm[start:start + count]
        if segment.size < 4800 or source_rms(segment) < args.min_rms:
            continue
        attempted += 1
        bank[speaker].append(encoder.embedding(segment))
    print(f"[3dspeaker:refs] attempted={attempted} used={sum(len(v) for v in bank.values())} speakers={len(bank)}")
    for speaker, refs in sorted(bank.items()):
        print(f"[3dspeaker:ref] {speaker}={len(refs)}")
    return bank


def best_speaker(query: np.ndarray, bank: dict[str, list[np.ndarray]]) -> tuple[str, float, str, float]:
    scored: list[tuple[float, str]] = []
    for speaker, refs in bank.items():
        best = max(float(np.dot(query, ref)) for ref in refs)
        scored.append((best, speaker))
    scored.sort(reverse=True)
    best_score, best_name = scored[0] if scored else (0.0, "?")
    second_score, second_name = scored[1] if len(scored) > 1 else (0.0, "?")
    return best_name, best_score, second_name, second_score


def run_spkid_variant(args: argparse.Namespace, variant: str, audio_dir: Path,
                      manifest: list[dict[str, Any]], encoder: ThreeDSpeaker,
                      bank: dict[str, list[np.ndarray]]) -> Path:
    out_dir = args.out_dir / variant
    jsonl_path = out_dir / "spkid_sources.jsonl"
    if jsonl_path.is_file() and not args.force:
        print(f"[3dspeaker:{variant}] reuse {jsonl_path}")
        return jsonl_path
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_variant_audio(audio_dir, manifest, include_mix=False)
    with jsonl_path.open("w", encoding="utf-8") as output:
        for index, path in enumerate(paths, 1):
            samples = load_wav_16k(path)
            total = int(samples.size)
            start, count = center_window(0, total, args.source_window_sec, total)
            segment = samples[start:start + count]
            rms_value = source_rms(segment)
            extracted = False
            best_name, best_score, second_name, second_score = "?", 0.0, "?", 0.0
            if segment.size >= 4800 and rms_value >= args.min_rms:
                query = encoder.embedding(segment)
                best_name, best_score, second_name, second_score = best_speaker(query, bank)
                extracted = True
            margin = best_score - second_score
            accepted = extracted and best_score >= args.match_threshold and margin >= args.min_margin
            row = {
                "clip": clip_from_stream_path(path),
                "stream": stream_name(path),
                "file": str(path),
                "duration_sec": round(total / SAMPLE_RATE, 3),
                "window_sec": round(segment.size / SAMPLE_RATE, 3),
                "rms": round(rms_value, 6),
                "extracted": extracted,
                "accepted": accepted,
                "speaker_id": -1,
                "pred_speaker": best_name if accepted else "?",
                "best_raw_speaker": best_name,
                "similarity": round(best_score, 6),
                "second_speaker": second_name,
                "second_similarity": round(second_score, 6),
                "margin": round(margin, 6),
                "match_threshold": args.match_threshold,
                "min_margin": args.min_margin,
                "engine": "3D-Speaker/CAM++",
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[3dspeaker:{variant} {index:03d}/{len(paths):03d}] {row['clip']}/{row['stream']} raw={best_name} sim={best_score:.3f} margin={margin:.3f}")
    return jsonl_path


def summarize_variant(args: argparse.Namespace, variant: str, asr_jsonl: Path, spkid_jsonl: Path) -> dict[str, Any]:
    asr_tool = load_module("summarize_separatio_asr", args.repo_root / "tools" / "summarize_separatio_asr.py")
    identity_tool = load_module("summarize_separatio_identity", args.repo_root / "tools" / "summarize_separatio_identity.py")
    out_dir = args.out_dir / variant
    manifest_map = asr_tool.build_manifest(args.manifest)
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in asr_tool.read_jsonl(asr_jsonl):
        by_clip[str(row.get("clip", ""))].append(row)
    asr_rows = [asr_tool.summarize_clip(clip, rows, manifest_map) for clip, rows in sorted(by_clip.items())]
    asr_compare = out_dir / "asr_gt_compare.json"
    write_json(asr_compare, asr_rows)
    asr_tool.write_markdown(out_dir / "asr_gt_compare.md", asr_rows, asr_jsonl, args.manifest)

    identity_report = identity_tool.summarize(asr_rows, identity_tool.read_jsonl(spkid_jsonl), args.match_threshold, args.min_margin)
    write_json(out_dir / "identity_gt_compare.json", identity_report)
    identity_tool.write_markdown(out_dir / "identity_gt_compare.md", identity_report, asr_compare, spkid_jsonl)
    return {"asr": asr_rows, "identity": identity_report}


def asr_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "clips": len(rows),
        "two_speaker": sum(1 for row in rows if row.get("separation_ok")),
        "one_good": sum(1 for row in rows if row.get("one_good_source")),
    }


def clip_successes_asr(rows: list[dict[str, Any]], key: str) -> set[int]:
    return {int(row.get("rank") or 0) for row in rows if row.get(key)}


def clip_successes_identity(report: dict[str, Any], key: str) -> set[int]:
    return {int(row.get("rank") or 0) for row in report.get("clips", []) if row.get(key)}


def threshold_sweep(args: argparse.Namespace, asr_rows: list[dict[str, Any]], spkid_jsonl: Path) -> list[dict[str, Any]]:
    identity_tool = load_module("summarize_separatio_identity_sweep", args.repo_root / "tools" / "summarize_separatio_identity.py")
    spkid_rows = identity_tool.read_jsonl(spkid_jsonl)
    rows: list[dict[str, Any]] = []
    for threshold in args.sweep_thresholds:
        for margin in args.sweep_margins:
            report = identity_tool.summarize(asr_rows, spkid_rows, threshold, margin)
            summary = dict(report["summary"])
            summary["match_threshold"] = threshold
            summary["min_margin"] = margin
            rows.append(summary)
    rows.sort(key=lambda item: (
        item["identity_two_speaker_clips"], item["timeline_recall"],
        item["source_decided_accuracy"], -item["source_wrong"], -item["source_abstain"],
    ), reverse=True)
    return rows


def write_comparison(args: argparse.Namespace, reports: dict[str, dict[str, Any]], spkid_paths: dict[str, Path]) -> None:
    comparison: dict[str, Any] = {"variants": {}, "deltas": {}}
    lines = [
        "# Upstream Audio Probe Comparison",
        "",
        "Same downstream stack for both variants: FunASR SenseVoiceSmall for ASR and 3D-Speaker CAM++ for speaker evidence.",
        "",
        "## Variant Summary",
        "",
        "| Variant | ASR Two | ASR One | ID Two | ID One | Src Acc | Coverage | Timeline | Wrong | Abstain |",
        "|---------|--------:|--------:|-------:|-------:|--------:|---------:|---------:|------:|--------:|",
    ]
    for variant, report in reports.items():
        asr_summary = asr_counts(report["asr"])
        identity_summary = report["identity"]["summary"]
        sweep = threshold_sweep(args, report["asr"], spkid_paths[variant])
        comparison["variants"][variant] = {
            "asr": asr_summary,
            "identity_default": identity_summary,
            "identity_best_sweep": sweep[0] if sweep else {},
        }
        lines.append(
            f"| {variant} | {asr_summary['two_speaker']} | {asr_summary['one_good']} | "
            f"{identity_summary['identity_two_speaker_clips']} | {identity_summary['identity_one_source_clips']} | "
            f"{identity_summary['source_decided_accuracy']:.3f} | {identity_summary['source_decided_coverage']:.3f} | "
            f"{identity_summary['timeline_recall']:.3f} | {identity_summary['source_wrong']} | {identity_summary['source_abstain']} |"
        )
    variants = list(reports)
    if len(variants) >= 2:
        left, right = variants[0], variants[1]
        left_asr_two = clip_successes_asr(reports[left]["asr"], "separation_ok")
        right_asr_two = clip_successes_asr(reports[right]["asr"], "separation_ok")
        left_id_two = clip_successes_identity(reports[left]["identity"], "identity_two_speaker_ok")
        right_id_two = clip_successes_identity(reports[right]["identity"], "identity_two_speaker_ok")
        comparison["deltas"] = {
            f"{left}_asr_two_only": sorted(left_asr_two - right_asr_two),
            f"{right}_asr_two_only": sorted(right_asr_two - left_asr_two),
            f"{left}_identity_two_only": sorted(left_id_two - right_id_two),
            f"{right}_identity_two_only": sorted(right_id_two - left_id_two),
        }
        lines.extend([
            "",
            "## Rank Deltas",
            "",
            f"- ASR two-speaker only in `{left}`: {comparison['deltas'][f'{left}_asr_two_only']}",
            f"- ASR two-speaker only in `{right}`: {comparison['deltas'][f'{right}_asr_two_only']}",
            f"- identity two-speaker only in `{left}`: {comparison['deltas'][f'{left}_identity_two_only']}",
            f"- identity two-speaker only in `{right}`: {comparison['deltas'][f'{right}_identity_two_only']}",
        ])
    lines.extend(["", "## Best 3D-Speaker Threshold Sweep", ""])
    for variant, item in comparison["variants"].items():
        best = item["identity_best_sweep"]
        lines.append(
            f"- `{variant}`: threshold={best.get('match_threshold')} margin={best.get('min_margin')} "
            f"two={best.get('identity_two_speaker_clips')} one={best.get('identity_one_source_clips')} "
            f"timeline={best.get('timeline_recall')} wrong={best.get('source_wrong')} abstain={best.get('source_abstain')}"
        )
    write_json(args.out_dir / "upstream_vs_cpp_comparison.json", comparison)
    (args.out_dir / "upstream_vs_cpp_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare upstream audio stack with C++ Auditus separation outputs.")
    parser.add_argument("--repo-root", type=Path, default=Path("/home/rm01/DeusRidet"))
    parser.add_argument("--upstream-root", type=Path, default=Path("/home/rm01/upstream-audio-labs"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/segment_homogeneity_clips_r3/clip_manifest.jsonl"))
    parser.add_argument("--clips-dir", type=Path, default=Path("logs/segment_homogeneity_clips_r3/clips"))
    parser.add_argument("--cpp-audio-dir", type=Path, default=Path("logs/separatio_param_sweep_r1/separatio_raw_o16000/audio"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/upstream_audio_probe_r1"))
    parser.add_argument("--gt", type=Path, default=Path("tests/fixtures/test_ground_truth_v1.jsonl"))
    parser.add_argument("--reference-audio", type=Path, default=Path("tests/test.mp3"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ncpu", type=int, default=4)
    parser.add_argument("--speaker-model-id", default="iic/speech_campplus_sv_zh-cn_16k-common")
    parser.add_argument("--max-ref-per-speaker", type=int, default=8)
    parser.add_argument("--ref-min-ms", type=int, default=1200)
    parser.add_argument("--ref-window-sec", type=float, default=2.5)
    parser.add_argument("--source-window-sec", type=float, default=0.0)
    parser.add_argument("--min-rms", type=float, default=0.005)
    parser.add_argument("--match-threshold", type=float, default=0.35)
    parser.add_argument("--min-margin", type=float, default=0.03)
    parser.add_argument("--sweep-thresholds", type=parse_csv_floats, default=parse_csv_floats("0.25,0.30,0.35,0.40,0.45,0.50"))
    parser.add_argument("--sweep-margins", type=parse_csv_floats, default=parse_csv_floats("0.00,0.03,0.05,0.08,0.10"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--skip-clearvoice", action="store_true")
    parser.add_argument("--force-clearvoice-gpu", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    os.chdir(args.repo_root)
    args.manifest = args.repo_root / args.manifest if not args.manifest.is_absolute() else args.manifest
    args.clips_dir = args.repo_root / args.clips_dir if not args.clips_dir.is_absolute() else args.clips_dir
    args.cpp_audio_dir = args.repo_root / args.cpp_audio_dir if not args.cpp_audio_dir.is_absolute() else args.cpp_audio_dir
    args.out_dir = args.repo_root / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    args.gt = args.repo_root / args.gt if not args.gt.is_absolute() else args.gt
    args.reference_audio = args.repo_root / args.reference_audio if not args.reference_audio.is_absolute() else args.reference_audio
    return args


def main() -> int:
    args = parse_args()
    run_clean_state(args.skip_cleanup)
    manifest = load_manifest(args.manifest, args.clips_dir, args.limit)
    if not manifest:
        raise RuntimeError("empty manifest selection")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    upstream_audio_dir = args.out_dir / "upstream_clearvoice" / "audio"
    if not args.skip_clearvoice:
        upstream_audio_dir = run_clearvoice(args, manifest)
    cpp_audio_dir = args.cpp_audio_dir

    variant_audio = {
        "upstream_clearvoice": upstream_audio_dir,
        "cpp_raw_o16000": cpp_audio_dir,
    }
    asr_paths: dict[str, Path] = {}
    for variant, audio_dir in variant_audio.items():
        asr_paths[variant] = run_funasr_variant(args, variant, audio_dir, manifest)

    encoder = ThreeDSpeaker(args)
    bank = build_reference_bank(args, manifest, encoder)
    spkid_paths: dict[str, Path] = {}
    for variant, audio_dir in variant_audio.items():
        spkid_paths[variant] = run_spkid_variant(args, variant, audio_dir, manifest, encoder, bank)

    reports: dict[str, dict[str, Any]] = {}
    for variant in variant_audio:
        reports[variant] = summarize_variant(args, variant, asr_paths[variant], spkid_paths[variant])
    write_comparison(args, reports, spkid_paths)
    print(f"[out] {args.out_dir / 'upstream_vs_cpp_comparison.md'}")
    print(f"[out] {args.out_dir / 'upstream_vs_cpp_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())