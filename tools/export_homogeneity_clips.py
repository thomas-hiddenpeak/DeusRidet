#!/usr/bin/env python3
"""Export Step 18 homogeneity candidates as audio clips for human review.

This is an offline evidence-export tool. It does not score accuracy and does
not modify any online Auditus behavior.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def probe_duration(audio_path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        text=True,
    ).strip()
    return float(out)


def time_token(value: float) -> str:
    return f"{value:08.3f}".replace(".", "p")


def compact_text(value: Any, limit: int = 96) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def clip_name(rank: int, candidate: dict[str, Any]) -> str:
    vad_idx = int(candidate.get("vad_idx", rank - 1))
    start = float(candidate.get("start_sec", 0.0))
    end = float(candidate.get("end_sec", start))
    return f"rank_{rank:02d}_vad_{vad_idx:03d}_{time_token(start)}_{time_token(end)}.wav"


def run_ffmpeg(audio_path: Path, out_path: Path, start_sec: float, duration_sec: float, sample_rate: int) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration_sec:.3f}",
        "-i", str(audio_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def format_gt_rows(candidate: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for gt in candidate.get("gt_rows", [])[:8]:
        lines.append(
            f"- {float(gt.get('start_sec', 0.0)):.2f}-{float(gt.get('end_sec', 0.0)):.2f}s "
            f"{gt.get('speaker', '?')} ov={gt.get('overlap_sec', '?')} "
            f"{compact_text(gt.get('text'))}"
        )
    return lines or ["- none"]


def format_asr_rows(candidate: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for asr in candidate.get("asr", [])[:6]:
        lines.append(
            f"- {float(asr.get('start_sec', 0.0)):.2f}-{float(asr.get('end_sec', 0.0)):.2f}s "
            f"id={asr.get('speaker_id', '?')} src={asr.get('speaker_source', '?')} "
            f"ov={asr.get('overlap_sec', '?')} {compact_text(asr.get('text'))}"
        )
    return lines or ["- none"]


def format_speaker_rows(candidate: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for event in candidate.get("near_speaker_events", [])[:6]:
        lines.append(
            f"- t={event.get('t_close_sec', '?')} id={event.get('id', '?')} "
            f"sim={event.get('sim', '?')} amended={event.get('amended', False)}"
        )
    return lines or ["- none"]


def write_review_sheet(path: Path, manifest: list[dict[str, Any]], args: argparse.Namespace, audio_duration: float) -> None:
    lines = [
        "# Segment Homogeneity Clip Review",
        "",
        "Offline listening sheet for Step 18b-A. Fill the review fields while listening; this file is an artifact, not a score.",
        "",
        "Label definitions:",
        "- 音频多人: the audio itself contains more than one speaker inside this VAD container.",
        "- 换人清楚: the speaker change point is audible and probably splittable.",
        "- ASR合并多人: ASR emitted one transcript that combines words from multiple speakers.",
        "- GT边界可疑: the reference timestamp boundary appears offset or questionable after listening.",
        "- 无害语气词: the extra speaker content is mostly filler such as 嗯/啊/哦 and low priority.",
        "- 关键短表态: short content carries a decision/stance, such as 可以/不行/不对/不要.",
        "- 需要切分探针: action label; this case would benefit from a split/change-point probe.",
        "",
        f"- source_audio: `{args.audio}`",
        f"- candidates: `{args.candidates}`",
        f"- audio_duration_sec: `{audio_duration:.3f}`",
        f"- pad_sec: `{args.pad_sec:.3f}`",
        f"- clips: `{len(manifest)}`",
        "",
    ]
    for row in manifest:
        candidate = row["candidate"]
        speakers = ", ".join(candidate.get("gt_speakers", [])) or "?"
        reasons = ", ".join(candidate.get("reasons", [])) or "none"
        lines.extend([
            f"## {row['rank']:02d}. VAD #{candidate.get('vad_idx')} {candidate.get('start_sec'):.2f}-{candidate.get('end_sec'):.2f}s",
            "",
            f"Clip: [{row['clip_name']}](clips/{row['clip_name']})",
            f"Window: {row['clip_start_sec']:.2f}-{row['clip_end_sec']:.2f}s; risk={candidate.get('risk_score')}; GT speakers={speakers}",
            f"Reasons: {reasons}",
            "",
            "Review:",
            "- labels: [ ] 音频多人  [ ] 换人清楚  [ ] ASR合并多人  [ ] GT边界可疑  [ ] 无害语气词  [ ] 关键短表态  [ ] 需要切分探针",
            "- primary_failure / 主要问题: ",
            "- speaker_attribution_note / 归属备注: ",
            "",
            "GT overlaps:",
            *format_gt_rows(candidate),
            "",
            "ASR overlaps:",
            *format_asr_rows(candidate),
            "",
            "Nearby speaker events:",
            *format_speaker_rows(candidate),
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export homogeneity-audit candidates into audio clips and a review sheet.")
    parser.add_argument("--candidates", default="/tmp/segment_homogeneity_audit_r3/homogeneity_candidates.jsonl")
    parser.add_argument("--audio", default="tests/test.mp3")
    parser.add_argument("--out-dir", default="/tmp/segment_homogeneity_clips_r3")
    parser.add_argument("--pad-sec", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    audio_path = Path(args.audio)
    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if not candidates_path.is_file():
        raise SystemExit(f"ERROR: candidates not found: {candidates_path}")
    if not audio_path.is_file():
        raise SystemExit(f"ERROR: audio not found: {audio_path}")

    candidates = read_jsonl(candidates_path)
    if args.limit > 0:
        candidates = candidates[: args.limit]
    audio_duration = probe_duration(audio_path)

    manifest: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        start = float(candidate.get("start_sec", 0.0))
        end = float(candidate.get("end_sec", start))
        clip_start = max(0.0, start - args.pad_sec)
        clip_end = min(audio_duration, end + args.pad_sec)
        duration = max(0.05, clip_end - clip_start)
        name = clip_name(rank, candidate)
        clip_path = clips_dir / name
        run_ffmpeg(audio_path, clip_path, clip_start, duration, args.sample_rate)
        manifest.append({
            "rank": rank,
            "clip_name": name,
            "clip_path": str(clip_path),
            "clip_start_sec": round(clip_start, 3),
            "clip_end_sec": round(clip_end, 3),
            "clip_duration_sec": round(duration, 3),
            "candidate": candidate,
            "review": {
                "label": "",
                "primary_failure": "",
                "speaker_attribution_note": "",
            },
        })

    manifest_path = out_dir / "clip_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    review_path = out_dir / "hearing_review.md"
    write_review_sheet(review_path, manifest, args, audio_duration)

    print(f"[input] candidates={len(candidates)} audio_duration={audio_duration:.3f}s")
    print(f"[out] clips_dir={clips_dir}")
    print(f"[out] manifest={manifest_path}")
    print(f"[out] review={review_path}")
    for row in manifest[:10]:
        candidate = row["candidate"]
        print(
            f"#{row['rank']:02d} {row['clip_name']} "
            f"clip={row['clip_start_sec']:.2f}-{row['clip_end_sec']:.2f}s "
            f"vad={candidate.get('start_sec'):.2f}-{candidate.get('end_sec'):.2f}s "
            f"risk={candidate.get('risk_score')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())