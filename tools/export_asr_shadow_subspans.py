#!/usr/bin/env python3
"""Export shadow speaker sub-spans for high-risk ASR transcripts.

This tool tests the metadata shape for a future online-safe broadcast. It does
not split text at word level; every sub-span references the parent transcript.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
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


def interval_start(row: dict[str, Any]) -> float:
    return float(row.get("start_sec", row.get("src_start_sec", 0.0)))


def interval_end(row: dict[str, Any]) -> float:
    return float(row.get("end_sec", row.get("src_end_sec", interval_start(row))))


def pair_vad_with_speaker_events(vad_segments: list[dict[str, Any]], speaker_events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    vad_sorted = sorted(vad_segments, key=lambda row: row["end_sec"])
    used = [False] * len(vad_sorted)
    paired: dict[int, dict[str, Any]] = {}
    for event in speaker_events:
        close_time = float(event.get("t_close_sec", 0.0))
        best_index = -1
        best_dt = float("inf")
        for index, vad in enumerate(vad_sorted):
            if used[index]:
                continue
            dt = abs(float(vad["end_sec"]) - close_time)
            if dt < best_dt:
                best_dt = dt
                best_index = index
            if float(vad["end_sec"]) > close_time + 5.0:
                break
        if best_index < 0:
            continue
        used[best_index] = True
        vad = vad_sorted[best_index]
        paired[int(vad["vad_idx"])] = {
            "speaker_id": int(event.get("id", -1)),
            "speaker_name": event.get("name", ""),
            "speaker_sim": float(event.get("sim", 0.0)),
            "t_close_sec": close_time,
            "pair_dt_sec": round(best_dt, 3),
            "amended": bool(event.get("amended", False)),
        }
    return paired


def overlapping_rows(start: float, end: float, rows: list[dict[str, Any]], min_overlap: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        row_start = interval_start(row)
        row_end = interval_end(row)
        overlap = overlap_sec(start, end, row_start, row_end)
        if overlap >= min_overlap:
            copied = dict(row)
            copied["overlap_sec"] = round(overlap, 3)
            out.append(copied)
    return out


def parent_review_flags(start: float, end: float, review_rows: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
    ranks: list[int] = []
    labels: set[str] = set()
    for row in review_rows:
        overlap = overlap_sec(start, end, float(row["start_sec"]), float(row["end_sec"]))
        if overlap < 0.08:
            continue
        ranks.append(int(row["rank"]))
        labels.update(str(label) for label in row.get("labels", []))
    return sorted(set(ranks)), sorted(labels)


def gt_speakers_for_span(start: float, end: float, gt_rows: list[dict[str, Any]]) -> list[str]:
    speakers = {
        str(row.get("speaker", "")) for row in gt_rows
        if overlap_sec(start, end, float(row.get("start_ms", 0)) / 1000.0, float(row.get("end_ms", 0)) / 1000.0) >= 0.08
        and row.get("speaker")
    }
    return sorted(speakers)


def build_subspans(
    asr_row: dict[str, Any],
    vad_segments: list[dict[str, Any]],
    vad_speakers: dict[int, dict[str, Any]],
    min_overlap: float,
) -> list[dict[str, Any]]:
    start = interval_start(asr_row)
    end = interval_end(asr_row)
    subspans: list[dict[str, Any]] = []
    for vad in overlapping_rows(start, end, vad_segments, min_overlap):
        vad_idx = int(vad["vad_idx"])
        span_start = max(start, float(vad["start_sec"]))
        span_end = min(end, float(vad["end_sec"]))
        speaker = vad_speakers.get(vad_idx)
        if speaker is None:
            speaker_id = -1
            source = "vad_unpaired"
            speaker_sim = 0.0
            pair_dt = None
            amended = False
        else:
            speaker_id = int(speaker["speaker_id"])
            source = "vad_speaker_event" if speaker_id >= 0 else "vad_speaker_abstain"
            speaker_sim = float(speaker["speaker_sim"])
            pair_dt = speaker["pair_dt_sec"]
            amended = bool(speaker["amended"])
        subspans.append({
            "start_sec": round(span_start, 3),
            "end_sec": round(span_end, 3),
            "duration_sec": round(max(0.0, span_end - span_start), 3),
            "vad_idx": vad_idx,
            "vad_overlap_sec": vad["overlap_sec"],
            "speaker_id": speaker_id,
            "speaker_sim": round(speaker_sim, 3),
            "source": source,
            "pair_dt_sec": pair_dt,
            "amended": amended,
        })
    return subspans


def risk_reason(asr_row: dict[str, Any], subspans: list[dict[str, Any]], review_labels: list[str], gt_speakers: list[str]) -> list[str]:
    reasons: list[str] = []
    if len(subspans) >= 2:
        reasons.append(f"multi_vad:{len(subspans)}")
    paired_ids = sorted({s["speaker_id"] for s in subspans if int(s["speaker_id"]) >= 0})
    if len(paired_ids) >= 2:
        reasons.append("multi_runtime_speaker:" + "/".join(str(item) for item in paired_ids))
    if "ASR合并多人" in review_labels:
        reasons.append("human_asr_merge")
    if "音频多人" in review_labels:
        reasons.append("human_audio_multi")
    if len(gt_speakers) >= 2:
        reasons.append("gt_multi_speaker:" + "/".join(gt_speakers))
    if float(asr_row.get("audio_sec", interval_end(asr_row) - interval_start(asr_row))) >= 5.0:
        reasons.append("long_asr")
    return reasons


def compact_text(text: Any, limit: int = 140) -> str:
    value = str(text or "").replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def export(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay_dir = Path(args.replay_dir)
    asr_rows = read_jsonl(replay_dir / "asr_transcripts.jsonl")
    speaker_events = read_jsonl(replay_dir / "speaker_events.jsonl")
    vad_segments = parse_timeline_vad_segments(Path(args.timeline))
    review_rows = read_jsonl(Path(args.review_labels))
    gt_rows = read_jsonl(Path(args.gt_v1))
    if args.max_sec > 0:
        asr_rows = [row for row in asr_rows if interval_start(row) <= args.max_sec + 2.0]
        speaker_events = [row for row in speaker_events if float(row.get("t_close_sec", 0.0)) <= args.max_sec + 2.0]
        vad_segments = [row for row in vad_segments if float(row["end_sec"]) <= args.max_sec + 2.0]
        gt_rows = [row for row in gt_rows if float(row.get("end_ms", 0.0)) <= args.max_sec * 1000.0]
    vad_speakers = pair_vad_with_speaker_events(vad_segments, speaker_events)

    exports: list[dict[str, Any]] = []
    for index, asr_row in enumerate(asr_rows):
        start = interval_start(asr_row)
        end = interval_end(asr_row)
        subspans = build_subspans(asr_row, vad_segments, vad_speakers, args.min_overlap)
        reviewed_ranks, labels = parent_review_flags(start, end, review_rows)
        gt_speakers = gt_speakers_for_span(start, end, gt_rows)
        reasons = risk_reason(asr_row, subspans, labels, gt_speakers)
        paired = sum(1 for span in subspans if int(span["speaker_id"]) >= 0)
        unknown = len(subspans) - paired
        high_risk = (
            len(subspans) >= args.min_vad_count
            and ("human_asr_merge" in reasons or "human_audio_multi" in reasons or len(gt_speakers) >= 2 or paired >= 2)
        )
        if args.high_risk_only and not high_risk:
            continue
        exports.append({
            "asr_index": index,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(max(0.0, end - start), 3),
            "text": asr_row.get("text", ""),
            "text_preview": compact_text(asr_row.get("text", "")),
            "text_policy": "parent_transcript_reference_only_no_word_timestamps",
            "original_speaker_id": int(asr_row.get("speaker_id", -1)),
            "original_speaker_source": asr_row.get("speaker_source", ""),
            "trigger": asr_row.get("trigger", ""),
            "subspan_count": len(subspans),
            "paired_subspan_count": paired,
            "unknown_subspan_count": unknown,
            "review_ranks": reviewed_ranks,
            "review_labels": labels,
            "gt_speakers": gt_speakers,
            "risk_reasons": reasons,
            "high_risk": high_risk,
            "subspans": subspans,
        })

    reason_counts: Counter[str] = Counter()
    for row in exports:
        reason_counts.update(row["risk_reasons"])
    summary = {
        "n_asr": len(asr_rows),
        "n_exported": len(exports),
        "n_high_risk": sum(1 for row in exports if row["high_risk"]),
        "n_multi_vad_exported": sum(1 for row in exports if row["subspan_count"] >= 2),
        "n_with_any_paired_subspan": sum(1 for row in exports if row["paired_subspan_count"] > 0),
        "total_subspans": sum(row["subspan_count"] for row in exports),
        "paired_subspans": sum(row["paired_subspan_count"] for row in exports),
        "unknown_subspans": sum(row["unknown_subspan_count"] for row in exports),
        "reason_counts": dict(reason_counts.most_common()),
        "timeline": args.timeline,
        "replay_dir": args.replay_dir,
    }
    return exports, summary


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# ASR Shadow Subspans",
        "",
        "Read-only metadata prototype. Subspans are speaker-attribution intervals; text is not split at word level.",
        "",
        "## Summary",
        "",
        f"- ASR transcripts: {summary['n_asr']}",
        f"- exported transcripts: {summary['n_exported']}",
        f"- high-risk exported: {summary['n_high_risk']}",
        f"- total subspans: {summary['total_subspans']}",
        f"- paired subspans: {summary['paired_subspans']}",
        f"- unknown subspans: {summary['unknown_subspans']}",
        "",
        "## Reasons",
        "",
    ]
    for reason, count in summary["reason_counts"].items():
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Transcripts", ""])
    for row in rows[:40]:
        lines.append(
            f"### ASR #{row['asr_index']} {row['start_sec']:.2f}-{row['end_sec']:.2f}s "
            f"orig={row['original_speaker_id']} subspans={row['subspan_count']} paired={row['paired_subspan_count']}"
        )
        lines.append("")
        lines.append(f"Text: {row['text_preview']}")
        lines.append(f"Risk: {', '.join(row['risk_reasons']) or 'none'}")
        if row["review_ranks"]:
            lines.append(f"Review ranks: {row['review_ranks']} labels={row['review_labels']}")
        if row["gt_speakers"]:
            lines.append(f"GT speakers: {', '.join(row['gt_speakers'])}")
        lines.append("Subspans:")
        for span in row["subspans"]:
            lines.append(
                f"- {span['start_sec']:.2f}-{span['end_sec']:.2f}s vad=#{span['vad_idx']} "
                f"spk={span['speaker_id']} sim={span['speaker_sim']:.3f} src={span['source']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export shadow subspan metadata for high-risk ASR transcripts.")
    parser.add_argument("--timeline", default="logs/timeline/tl_20260426_182711.jsonl")
    parser.add_argument("--replay-dir", default="/tmp/replay_retro_amend_r3")
    parser.add_argument("--review-labels", default="/tmp/segment_homogeneity_clips_r3/review_labels.jsonl")
    parser.add_argument("--gt-v1", default="tests/fixtures/test_ground_truth_v1.jsonl")
    parser.add_argument("--out-dir", default="/tmp/segment_homogeneity_clips_r3")
    parser.add_argument("--max-sec", type=float, default=600.0)
    parser.add_argument("--min-overlap", type=float, default=0.08)
    parser.add_argument("--min-vad-count", type=int, default=2)
    parser.add_argument("--all", action="store_true", help="export all ASR transcripts, not just high-risk ones")
    args = parser.parse_args()
    args.high_risk_only = not args.all

    rows, summary = export(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "asr_shadow_subspans.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = out_dir / "asr_shadow_subspans_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / "asr_shadow_subspans.md"
    write_markdown(md_path, rows, summary)

    print(f"[asr] {summary['n_asr']} exported={summary['n_exported']} high_risk={summary['n_high_risk']}")
    print(f"[subspans] total={summary['total_subspans']} paired={summary['paired_subspans']} unknown={summary['unknown_subspans']}")
    for reason, count in list(summary["reason_counts"].items())[:12]:
        print(f"[reason] {reason}={count}")
    print(f"[out] {jsonl_path}")
    print(f"[out] {summary_path}")
    print(f"[out] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())