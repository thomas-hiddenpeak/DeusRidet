#!/usr/bin/env python3
"""Evaluate a shadow ASR-boundary split without changing online behavior.

This diagnostic compares current transcript-level speaker attribution against
ASR spans split by runtime VAD containers. It is not the acceptance benchmark;
it answers whether assigning speaker labels at ASR sub-span granularity is a
promising online-safe lever.
"""
from __future__ import annotations

import argparse
import json
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


def parse_timeline_vad_segments(timeline_path: Path, sample_rate: int = 16000) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    open_start: int | None = None
    for line in timeline_path.read_text(encoding="utf-8").splitlines():
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


def pair_vad_with_speaker_events(vad_segments: list[dict[str, Any]], speaker_events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    vad_sorted = sorted(vad_segments, key=lambda item: item["end_sec"])
    used = [False] * len(vad_sorted)
    paired: dict[int, dict[str, Any]] = {}
    for event in speaker_events:
        t_close = float(event.get("t_close_sec", 0.0))
        best_index = -1
        best_dt = float("inf")
        for index, vad in enumerate(vad_sorted):
            if used[index]:
                continue
            dt = abs(float(vad["end_sec"]) - t_close)
            if dt < best_dt:
                best_dt = dt
                best_index = index
            if float(vad["end_sec"]) > t_close + 5.0:
                break
        if best_index < 0:
            continue
        used[best_index] = True
        vad = vad_sorted[best_index]
        paired[int(vad["vad_idx"])] = {
            "cluster": int(event.get("id", -1)),
            "sim": float(event.get("sim", 0.0)),
            "t_close_sec": t_close,
            "pair_dt": round(best_dt, 3),
            "amended": bool(event.get("amended", False)),
        }
    return paired


def overlapping_vads(start: float, end: float, vad_segments: list[dict[str, Any]], min_overlap: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for vad in vad_segments:
        ov = overlap_sec(start, end, float(vad["start_sec"]), float(vad["end_sec"]))
        if ov >= min_overlap:
            copied = dict(vad)
            copied["overlap_sec"] = round(ov, 3)
            out.append(copied)
    return out


def baseline_transcript_segments(asr_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for index, row in enumerate(asr_rows):
        start = float(row.get("start_sec", row.get("src_start_sec", 0.0)))
        end = float(row.get("end_sec", row.get("src_end_sec", start)))
        if end <= start:
            continue
        segments.append({
            "start_sec": start,
            "end_sec": end,
            "cluster": int(row.get("speaker_id", -1)),
            "source": "asr_whole",
            "asr_index": index,
            "text": row.get("text", ""),
        })
    return segments


def split_transcript_segments(
    asr_rows: list[dict[str, Any]],
    vad_segments: list[dict[str, Any]],
    vad_speakers: dict[int, dict[str, Any]],
    fallback_original: bool,
    min_overlap: float,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for index, row in enumerate(asr_rows):
        asr_start = float(row.get("start_sec", row.get("src_start_sec", 0.0)))
        asr_end = float(row.get("end_sec", row.get("src_end_sec", asr_start)))
        original_cluster = int(row.get("speaker_id", -1))
        vads = overlapping_vads(asr_start, asr_end, vad_segments, min_overlap)
        if not vads and asr_end > asr_start:
            segments.append({
                "start_sec": asr_start,
                "end_sec": asr_end,
                "cluster": original_cluster if fallback_original else -1,
                "source": "asr_no_vad_fallback" if fallback_original else "asr_no_vad_abstain",
                "asr_index": index,
                "text": row.get("text", ""),
            })
            continue
        for vad in vads:
            vad_idx = int(vad["vad_idx"])
            start = max(asr_start, float(vad["start_sec"]))
            end = min(asr_end, float(vad["end_sec"]))
            if end <= start:
                continue
            speaker = vad_speakers.get(vad_idx)
            if speaker is None:
                cluster = original_cluster if fallback_original else -1
                source = "vad_split_fallback_original" if fallback_original else "vad_split_unpaired_abstain"
            else:
                cluster = int(speaker.get("cluster", -1))
                source = "vad_split_speaker"
            segments.append({
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "cluster": cluster,
                "source": source,
                "asr_index": index,
                "vad_idx": vad_idx,
                "text": row.get("text", ""),
            })
    segments.sort(key=lambda item: (item["start_sec"], item["end_sec"], item.get("vad_idx", -1)))
    return segments


def score_by_overlap(runtime_segments: list[dict[str, Any]], gt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtime_sorted = sorted(runtime_segments, key=lambda item: item["start_sec"])

    def best_match(start: float, end: float) -> tuple[dict[str, Any] | None, float]:
        best = None
        best_overlap = 0.0
        for row in runtime_sorted:
            if float(row["end_sec"]) <= start:
                continue
            if float(row["start_sec"]) >= end:
                break
            ov = overlap_sec(start, end, float(row["start_sec"]), float(row["end_sec"]))
            if ov > best_overlap:
                best_overlap = ov
                best = row
        return best, best_overlap

    matched: list[dict[str, Any]] = []
    for gt in gt_rows:
        start = float(gt["start_ms"]) / 1000.0
        end = float(gt["end_ms"]) / 1000.0
        row, ov = best_match(start, end)
        cluster = int(row["cluster"]) if row else -1
        matched.append({
            "gt_idx": gt["idx"],
            "gt_start": start,
            "gt_end": end,
            "gt_speaker": gt["speaker"],
            "rt_cluster": cluster,
            "rt_start": row["start_sec"] if row else None,
            "rt_end": row["end_sec"] if row else None,
            "overlap": round(ov, 3),
            "status": "decided" if cluster >= 0 else ("abstain" if row else "no_segment"),
        })

    mapping: dict[int, str] = {}
    for row in matched:
        cluster = int(row["rt_cluster"])
        if cluster >= 0:
            mapping.setdefault(cluster, str(row["gt_speaker"]))

    speakers = sorted({str(row["gt_speaker"]) for row in matched})
    per_total = {speaker: 0 for speaker in speakers}
    per_correct = {speaker: 0 for speaker in speakers}
    per_decided = {speaker: 0 for speaker in speakers}
    per_decided_correct = {speaker: 0 for speaker in speakers}
    n_no_segment = 0
    n_abstain = 0
    for row in matched:
        truth = str(row["gt_speaker"])
        per_total[truth] += 1
        if row["status"] == "no_segment":
            n_no_segment += 1
            continue
        if row["status"] == "abstain":
            n_abstain += 1
            continue
        per_decided[truth] += 1
        pred = mapping.get(int(row["rt_cluster"]), "__unk__")
        if pred == truth:
            per_correct[truth] += 1
            per_decided_correct[truth] += 1

    total = max(1, sum(per_total.values()))
    decided_total = max(1, sum(per_decided.values()))
    per_spk = {speaker: per_correct[speaker] / max(1, per_total[speaker]) for speaker in speakers}
    per_spk_decided = {
        speaker: per_decided_correct[speaker] / per_decided[speaker]
        if per_decided[speaker] else 0.0
        for speaker in speakers
    }
    return {
        "macro": sum(per_spk.values()) / max(1, len(per_spk)),
        "micro": sum(per_correct.values()) / total,
        "decided_macro": sum(per_spk_decided.values()) / max(1, len(per_spk_decided)),
        "decided_micro": sum(per_decided_correct.values()) / decided_total,
        "coverage": sum(per_decided.values()) / total,
        "n_gt": len(matched),
        "n_no_segment": n_no_segment,
        "n_abstain": n_abstain,
        "n_decided": sum(per_decided.values()),
        "mapping": mapping,
        "per_spk": per_spk,
        "per_spk_decided": per_spk_decided,
        "matched": matched,
    }


def source_counts(segments: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for segment in segments:
        key = str(segment.get("source", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_markdown(path: Path, results: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    lines = [
        "# Shadow ASR Boundary Split",
        "",
        "Read-only diagnostic: compare current ASR whole-transcript speaker labels with VAD-subspan labels.",
        "",
        "## Scores",
        "",
        "| Variant | Micro | Macro | Coverage | Decided micro | Decided macro | Decided / GT |",
        "|---------|------:|------:|---------:|--------------:|--------------:|-------------:|",
    ]
    for name in ["asr_whole", "vad_split_strict", "vad_split_fallback"]:
        score = results[name]["score"]
        lines.append(
            f"| {name} | {score['micro']:.3f} | {score['macro']:.3f} | "
            f"{score['coverage']:.3f} | {score['decided_micro']:.3f} | "
            f"{score['decided_macro']:.3f} | {score['n_decided']}/{score['n_gt']} |"
        )
    lines.extend(["", "## Segment Counts", ""])
    for name in ["asr_whole", "vad_split_strict", "vad_split_fallback"]:
        lines.append(f"- {name}: {results[name]['n_segments']} segments, sources={results[name]['source_counts']}")
    lines.extend(["", "## ASR Merge Examples", ""])
    for example in examples[:12]:
        lines.append(
            f"- rank {example['rank']:02d} VAD #{example['vad_idx']} "
            f"{example['start_sec']:.2f}-{example['end_sec']:.2f}s: "
            f"whole={example['whole_cluster']} strict={example['strict_clusters']} "
            f"fallback={example['fallback_clusters']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_examples(review_labels: list[dict[str, Any]], asr_rows: list[dict[str, Any]], split_strict: list[dict[str, Any]], split_fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for review in review_labels:
        labels = set(review.get("labels", []))
        if "ASR合并多人" not in labels:
            continue
        start = float(review["start_sec"])
        end = float(review["end_sec"])
        whole_clusters = sorted({
            int(row.get("speaker_id", -1)) for row in asr_rows
            if overlap_sec(start, end, float(row.get("start_sec", 0.0)), float(row.get("end_sec", 0.0))) >= 0.08
        })
        strict_clusters = sorted({
            int(row.get("cluster", -1)) for row in split_strict
            if overlap_sec(start, end, float(row.get("start_sec", 0.0)), float(row.get("end_sec", 0.0))) >= 0.08
        })
        fallback_clusters = sorted({
            int(row.get("cluster", -1)) for row in split_fallback
            if overlap_sec(start, end, float(row.get("start_sec", 0.0)), float(row.get("end_sec", 0.0))) >= 0.08
        })
        examples.append({
            "rank": review["rank"],
            "vad_idx": review["vad_idx"],
            "start_sec": start,
            "end_sec": end,
            "whole_cluster": whole_clusters,
            "strict_clusters": strict_clusters,
            "fallback_clusters": fallback_clusters,
        })
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate shadow ASR boundary splitting against replay artifacts.")
    parser.add_argument("--timeline", default="logs/timeline/tl_20260426_182711.jsonl")
    parser.add_argument("--replay-dir", default="/tmp/replay_retro_amend_r3")
    parser.add_argument("--gt-v1", default="tests/fixtures/test_ground_truth_v1.jsonl")
    parser.add_argument("--review-labels", default="/tmp/segment_homogeneity_clips_r3/review_labels.jsonl")
    parser.add_argument("--out-dir", default="/tmp/segment_homogeneity_clips_r3")
    parser.add_argument("--max-sec", type=float, default=600.0)
    parser.add_argument("--min-overlap", type=float, default=0.08)
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    vad_segments = parse_timeline_vad_segments(Path(args.timeline))
    speaker_events = read_jsonl(replay_dir / "speaker_events.jsonl")
    asr_rows = read_jsonl(replay_dir / "asr_transcripts.jsonl")
    gt_rows = read_jsonl(Path(args.gt_v1))
    review_labels = read_jsonl(Path(args.review_labels))
    if args.max_sec > 0:
        vad_segments = [row for row in vad_segments if float(row["end_sec"]) <= args.max_sec + 2.0]
        speaker_events = [row for row in speaker_events if float(row.get("t_close_sec", 0.0)) <= args.max_sec + 2.0]
        asr_rows = [row for row in asr_rows if float(row.get("start_sec", 0.0)) <= args.max_sec + 2.0]
        gt_rows = [row for row in gt_rows if float(row.get("end_ms", 0.0)) <= args.max_sec * 1000.0]

    vad_speakers = pair_vad_with_speaker_events(vad_segments, speaker_events)
    whole_segments = baseline_transcript_segments(asr_rows)
    split_strict = split_transcript_segments(asr_rows, vad_segments, vad_speakers, False, args.min_overlap)
    split_fallback = split_transcript_segments(asr_rows, vad_segments, vad_speakers, True, args.min_overlap)

    results = {
        "inputs": {
            "timeline": args.timeline,
            "replay_dir": args.replay_dir,
            "gt_v1": args.gt_v1,
            "asr": len(asr_rows),
            "vad": len(vad_segments),
            "speaker_events": len(speaker_events),
            "paired_vad_speakers": len(vad_speakers),
        },
        "asr_whole": {
            "n_segments": len(whole_segments),
            "source_counts": source_counts(whole_segments),
            "score": score_by_overlap(whole_segments, gt_rows),
        },
        "vad_split_strict": {
            "n_segments": len(split_strict),
            "source_counts": source_counts(split_strict),
            "score": score_by_overlap(split_strict, gt_rows),
        },
        "vad_split_fallback": {
            "n_segments": len(split_fallback),
            "source_counts": source_counts(split_fallback),
            "score": score_by_overlap(split_fallback, gt_rows),
        },
    }
    examples = build_examples(review_labels, asr_rows, split_strict, split_fallback)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "shadow_asr_split_summary.json"
    compact_results = json.loads(json.dumps(results, ensure_ascii=False))
    for variant in ["asr_whole", "vad_split_strict", "vad_split_fallback"]:
        compact_results[variant]["score"].pop("matched", None)
    compact_results["asr_merge_examples"] = examples
    json_path.write_text(json.dumps(compact_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / "shadow_asr_split_summary.md"
    write_markdown(md_path, results, examples)
    for name, segments in [
        ("asr_whole_segments.jsonl", whole_segments),
        ("vad_split_strict_segments.jsonl", split_strict),
        ("vad_split_fallback_segments.jsonl", split_fallback),
    ]:
        with (out_dir / name).open("w", encoding="utf-8") as handle:
            for segment in segments:
                handle.write(json.dumps(segment, ensure_ascii=False) + "\n")

    print(f"[inputs] asr={len(asr_rows)} vad={len(vad_segments)} speakers={len(speaker_events)} paired_vad={len(vad_speakers)}")
    for name in ["asr_whole", "vad_split_strict", "vad_split_fallback"]:
        score = results[name]["score"]
        print(
            f"[{name}] segments={results[name]['n_segments']} "
            f"micro={score['micro']:.3f} macro={score['macro']:.3f} "
            f"coverage={score['coverage']:.3f} decided_micro={score['decided_micro']:.3f} "
            f"decided={score['n_decided']}/{score['n_gt']}"
        )
    print(f"[out] {json_path}")
    print(f"[out] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())