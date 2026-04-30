#!/usr/bin/env python3
"""Attribute reviewed ASR-merge cases to ASR-span or VAD-container causes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_timeline_vad_segments(path: Path, sample_rate: int = 16000) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    open_start: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("t") != "vad":
            continue
        audio_t1 = int(obj.get("audio_t1", 0))
        if obj.get("event") == "start":
            open_start = audio_t1
        elif obj.get("event") == "end" and open_start is not None:
            segments.append({
                "vad_idx": len(segments),
                "start_sec": open_start / float(sample_rate),
                "end_sec": audio_t1 / float(sample_rate),
            })
            open_start = None
    return segments


def overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def row_start(row: dict[str, Any]) -> float:
    if "start_sec" in row:
        return float(row["start_sec"])
    return float(row.get("start_ms", 0)) / 1000.0


def row_end(row: dict[str, Any]) -> float:
    if "end_sec" in row:
        return float(row["end_sec"])
    return float(row.get("end_ms", 0)) / 1000.0


def overlaps(start: float, end: float, rows: list[dict[str, Any]], min_overlap: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        ov = overlap_sec(start, end, row_start(row), row_end(row))
        if ov >= min_overlap:
            copied = dict(row)
            copied["overlap_sec"] = round(ov, 3)
            out.append(copied)
    return out


def compact_text(text: Any, limit: int = 120) -> str:
    value = str(text or "").replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def classify_asr(asr_row: dict[str, Any], gt_rows: list[dict[str, Any]], vad_rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = row_start(asr_row)
    end = row_end(asr_row)
    gt_ov = overlaps(start, end, gt_rows, 0.08)
    vad_ov = overlaps(start, end, vad_rows, 0.08)
    gt_speakers = sorted({g.get("speaker", "") for g in gt_ov if g.get("speaker")})
    gt_turns = sorted({int(g.get("idx", -1)) for g in gt_ov if int(g.get("idx", -1)) >= 0})
    if len(vad_ov) >= 2 and len(gt_speakers) >= 2:
        cause = "asr_spans_multiple_vad"
    elif len(vad_ov) <= 1 and len(gt_speakers) >= 2:
        cause = "single_vad_mixed_speakers"
    elif len(gt_speakers) < 2:
        cause = "not_multi_speaker_by_gt"
    else:
        cause = "ambiguous"
    return {
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "duration_sec": round(max(0.0, end - start), 3),
        "speaker_id": asr_row.get("speaker_id"),
        "speaker_source": asr_row.get("speaker_source"),
        "text": compact_text(asr_row.get("text")),
        "gt_speakers": gt_speakers,
        "gt_turn_count": len(gt_turns),
        "vad_count": len(vad_ov),
        "vad_indices": [v.get("vad_idx") for v in vad_ov],
        "cause": cause,
    }


def analyze(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    review_rows = read_jsonl(Path(args.review_labels))
    candidates = read_jsonl(Path(args.candidates))
    asr_rows = read_jsonl(Path(args.replay_dir) / "asr_transcripts.jsonl")
    gt_rows = read_jsonl(Path(args.gt_v1))
    vad_rows = parse_timeline_vad_segments(Path(args.timeline))
    candidates_by_rank = {idx + 1: row for idx, row in enumerate(candidates)}

    cases: list[dict[str, Any]] = []
    cause_counts: Counter[str] = Counter()
    for review in review_rows:
        labels = set(review.get("labels", []))
        if "ASR合并多人" not in labels:
            continue
        rank = int(review["rank"])
        candidate = candidates_by_rank.get(rank, {})
        start = float(review["start_sec"])
        end = float(review["end_sec"])
        asr_ov = overlaps(start, end, asr_rows, 0.08)
        analyzed_asr = [classify_asr(row, gt_rows, vad_rows) for row in asr_ov]
        local_causes = Counter(row["cause"] for row in analyzed_asr)
        if local_causes:
            cause = local_causes.most_common(1)[0][0]
        else:
            cause = "no_overlapping_asr"
        cause_counts[cause] += 1
        cases.append({
            "rank": rank,
            "vad_idx": review["vad_idx"],
            "start_sec": review["start_sec"],
            "end_sec": review["end_sec"],
            "labels": review.get("labels", []),
            "risk_score": candidate.get("risk_score"),
            "candidate_reasons": candidate.get("reasons", []),
            "case_cause": cause,
            "asr": analyzed_asr,
        })
    summary = {
        "n_reviewed_asr_merge": len(cases),
        "cause_counts": dict(cause_counts.most_common()),
        "timeline": args.timeline,
        "replay_dir": args.replay_dir,
    }
    return cases, summary


def write_markdown(path: Path, cases: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# ASR Merge Review Analysis",
        "",
        "This is a read-only diagnostic over human-reviewed homogeneity clips.",
        "",
        "## Summary",
        "",
        f"- ASR-merge reviewed cases: {summary['n_reviewed_asr_merge']}",
    ]
    for cause, count in summary["cause_counts"].items():
        lines.append(f"- {cause}: {count}")
    lines.extend(["", "## Cases", ""])
    for case in cases:
        lines.append(
            f"### {case['rank']:02d}. VAD #{case['vad_idx']} "
            f"{case['start_sec']:.2f}-{case['end_sec']:.2f}s — {case['case_cause']}"
        )
        lines.append("")
        lines.append(f"Labels: {', '.join(case['labels'])}")
        lines.append(f"Candidate reasons: {', '.join(case['candidate_reasons'])}")
        for asr in case["asr"]:
            lines.append(
                f"- ASR {asr['start_sec']:.2f}-{asr['end_sec']:.2f}s "
                f"dur={asr['duration_sec']:.2f}s cause={asr['cause']} "
                f"vad_count={asr['vad_count']} gt_speakers={','.join(asr['gt_speakers']) or '?'} "
                f"text={asr['text']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze human-reviewed ASR merge cases against VAD, ASR, and GT intervals.")
    parser.add_argument("--review-labels", default="/tmp/segment_homogeneity_clips_r3/review_labels.jsonl")
    parser.add_argument("--candidates", default="/tmp/segment_homogeneity_audit_r3/homogeneity_candidates.jsonl")
    parser.add_argument("--timeline", default="logs/timeline/tl_20260426_182711.jsonl")
    parser.add_argument("--replay-dir", default="/tmp/replay_retro_amend_r3")
    parser.add_argument("--gt-v1", default="tests/fixtures/test_ground_truth_v1.jsonl")
    parser.add_argument("--out-dir", default="/tmp/segment_homogeneity_clips_r3")
    args = parser.parse_args()

    cases, summary = analyze(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / "asr_merge_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    summary_path = out_dir / "asr_merge_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / "asr_merge_summary.md"
    write_markdown(md_path, cases, summary)

    print(f"[cases] {summary['n_reviewed_asr_merge']}")
    for cause, count in summary["cause_counts"].items():
        print(f"[cause] {cause}={count}")
    print(f"[out] {cases_path}")
    print(f"[out] {summary_path}")
    print(f"[out] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())