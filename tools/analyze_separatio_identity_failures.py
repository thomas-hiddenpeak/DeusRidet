#!/usr/bin/env python3
"""Analyze why separated-source identity recovery is low."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def label_key(labels: list[str]) -> str:
    return ",".join(labels) if labels else "none"


def energy_bucket(value: float) -> str:
    if value < 0.02:
        return "<0.02"
    if value < 0.05:
        return "0.02-0.05"
    if value < 0.15:
        return "0.05-0.15"
    if value < 0.30:
        return "0.15-0.30"
    return ">=0.30"


def duration_bucket(seconds: float) -> str:
    if seconds < 0.5:
        return "<0.5s"
    if seconds < 1.0:
        return "0.5-1.0s"
    if seconds < 1.5:
        return "1.0-1.5s"
    if seconds < 2.5:
        return "1.5-2.5s"
    return ">=2.5s"


def source_state(source: dict[str, Any]) -> str:
    if source.get("correct"):
        return "correct"
    if source.get("wrong"):
        return "wrong"
    if source.get("abstain"):
        return "abstain"
    return "no_asr_match"


def abstain_reason(spkid: dict[str, Any] | None) -> str:
    if not spkid or not spkid.get("extracted", False):
        return "no_embedding_low_rms"
    if float(spkid.get("similarity", 0.0)) < float(spkid.get("match_threshold", 0.35)):
        return "below_similarity"
    if float(spkid.get("margin", 0.0)) < float(spkid.get("min_margin", 0.03)):
        return "below_margin"
    return "other_abstain"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    identity = load_json(Path(args.identity))
    asr_rows = load_json(Path(args.asr_compare))
    sep_rows = read_jsonl(Path(args.separatio_jsonl))
    spkid_rows = read_jsonl(Path(args.spkid_jsonl))
    review_rows = read_jsonl(Path(args.review_labels))
    asr_by_clip = {str(row.get("clip", "")): row for row in asr_rows}
    sep_by_clip = {str(row.get("clip", "")): row for row in sep_rows}
    spkid_by_key = {(str(row.get("clip", "")), str(row.get("stream", ""))): row for row in spkid_rows}
    review_by_rank = {int(row.get("rank", 0)): row for row in review_rows}

    source_counts: Counter[str] = Counter()
    gt_failure_counts: Counter[str] = Counter()
    wrong_pairs: Counter[str] = Counter()
    abstain_counts: Counter[str] = Counter()
    clip_funnel: Counter[str] = Counter()
    speaker_count_clips: Counter[str] = Counter()
    duration_stats: dict[str, Counter[str]] = defaultdict(Counter)
    energy_stats: dict[str, Counter[str]] = defaultdict(Counter)
    label_stats: dict[str, Counter[str]] = defaultdict(Counter)
    clip_rows: list[dict[str, Any]] = []

    for clip in identity.get("clips", []):
        clip_name = str(clip.get("clip", ""))
        rank = int(clip.get("rank") or 0)
        asr = asr_by_clip.get(clip_name, {})
        sep = sep_by_clip.get(clip_name, {})
        labels = list(review_by_rank.get(rank, {}).get("labels", []))
        gt_rows = {
            int(row.get("idx", -1)): row
            for row in asr.get("gt_rows", [])
            if int(row.get("idx", -1)) >= 0
        }
        gt_speakers = sorted({str(row.get("speaker", "?")) for row in gt_rows.values()})
        speaker_count = len(gt_speakers)
        speaker_count_clips[str(speaker_count)] += 1
        two_active = bool(sep.get("two_active", False))
        asr_two = bool(asr.get("separation_ok", False))
        id_two = bool(clip.get("identity_two_speaker_ok", False))
        clip_funnel[f"multi={speaker_count >= 2}|two_active={two_active}|asr_two={asr_two}|id_two={id_two}"] += 1

        recovered_gt: set[int] = set()
        attempts_by_gt: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for source in clip.get("source_evals", []):
            state = source_state(source)
            if state == "no_asr_match":
                continue
            source_counts[state] += 1
            target_idx = int(source.get("target_gt_idx", -1))
            attempts_by_gt[target_idx].append(source)
            spkid = spkid_by_key.get((clip_name, str(source.get("stream", ""))))
            if state == "correct":
                recovered_gt.add(target_idx)
            elif state == "wrong":
                wrong_pairs[f"{source.get('target_speaker')}->{source.get('pred_speaker')}"] += 1
            elif state == "abstain":
                abstain_counts[abstain_reason(spkid)] += 1

        clip_failures: Counter[str] = Counter()
        for gt_idx, row in gt_rows.items():
            duration = float(row.get("end_sec", 0.0)) - float(row.get("start_sec", 0.0))
            dur_bucket = duration_bucket(duration)
            energy = float(sep.get("energy_balance", 0.0))
            eng_bucket = energy_bucket(energy)
            if gt_idx in recovered_gt:
                gt_failure_counts["recovered"] += 1
                duration_stats[dur_bucket]["recovered"] += 1
                energy_stats[eng_bucket]["recovered"] += 1
                for label in labels or ["none"]:
                    label_stats[label]["recovered"] += 1
                continue
            if gt_idx not in attempts_by_gt:
                reason = "asr_or_separator_missing_gt_row"
            else:
                states = {source_state(source) for source in attempts_by_gt[gt_idx]}
                if "wrong" in states:
                    reason = "speaker_wrong"
                elif "abstain" in states:
                    reason = "speaker_abstain"
                else:
                    reason = "speaker_other"
            gt_failure_counts[reason] += 1
            clip_failures[reason] += 1
            duration_stats[dur_bucket][reason] += 1
            energy_stats[eng_bucket][reason] += 1
            for label in labels or ["none"]:
                label_stats[label][reason] += 1

        clip_rows.append({
            "rank": rank,
            "clip": clip_name,
            "gt_speakers": gt_speakers,
            "labels": labels,
            "two_active": two_active,
            "asr_two": asr_two,
            "id_two": id_two,
            "one_id": bool(clip.get("identity_one_source_ok", False)),
            "energy_balance": round(float(sep.get("energy_balance", 0.0)), 4),
            "timeline_recovered": int(clip.get("timeline_gt_recovered", 0)),
            "timeline_total": int(clip.get("timeline_gt_total", 0)),
            "failure_counts": dict(clip_failures),
        })

    report = {
        "inputs": {
            "identity": args.identity,
            "asr_compare": args.asr_compare,
            "separatio_jsonl": args.separatio_jsonl,
            "spkid_jsonl": args.spkid_jsonl,
            "review_labels": args.review_labels,
        },
        "headline": {
            "clips": identity["summary"]["clips"],
            "gt_rows": identity["summary"]["timeline_gt_total"],
            "gt_rows_recovered": identity["summary"]["timeline_gt_recovered"],
            "timeline_recall": identity["summary"]["timeline_recall"],
            "identity_two_speaker_clips": identity["summary"]["identity_two_speaker_clips"],
            "identity_one_source_clips": identity["summary"]["identity_one_source_clips"],
        },
        "source_counts": dict(source_counts),
        "gt_row_outcomes": dict(gt_failure_counts),
        "wrong_pairs": dict(wrong_pairs.most_common()),
        "abstain_reasons": dict(abstain_counts.most_common()),
        "clip_funnel": dict(clip_funnel),
        "speaker_count_clips": dict(speaker_count_clips),
        "duration_stats": {key: dict(value) for key, value in duration_stats.items()},
        "energy_stats": {key: dict(value) for key, value in energy_stats.items()},
        "label_stats": {key: dict(value) for key, value in label_stats.items()},
        "clips": clip_rows,
    }
    return report


def count_total(row: dict[str, int]) -> int:
    return sum(row.values())


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    headline = report["headline"]
    lines = [
        "# Separatio Identity Failure Analysis",
        "",
        "This report explains why separated-source ASR + speaker ID recovery remains low on the high-risk clip set.",
        "",
        "## Headline",
        "",
        f"- clips: {headline['clips']}",
        f"- GT timeline rows: {headline['gt_rows']}",
        f"- recovered GT rows: {headline['gt_rows_recovered']} ({headline['timeline_recall']:.3f})",
        f"- two-ID clips: {headline['identity_two_speaker_clips']} / {headline['clips']}",
        f"- one-ID clips: {headline['identity_one_source_clips']} / {headline['clips']}",
        "",
        "## Funnel",
        "",
        "| Stage | Count |",
        "|-------|------:|",
    ]
    funnel = report["clip_funnel"]
    multi_clips = sum(count for key, count in funnel.items() if key.startswith("multi=True"))
    multi_two_active = sum(count for key, count in funnel.items() if key.startswith("multi=True|two_active=True"))
    multi_asr_two = sum(count for key, count in funnel.items() if "multi=True" in key and "asr_two=True" in key)
    multi_id_two = sum(count for key, count in funnel.items() if "multi=True" in key and "id_two=True" in key)
    lines.extend([
        f"| multi-speaker GT clips | {multi_clips} |",
        f"| separator two-active among multi-speaker clips | {multi_two_active} |",
        f"| ASR two-speaker among multi-speaker clips | {multi_asr_two} |",
        f"| speaker-ID two-ID among multi-speaker clips | {multi_id_two} |",
        "",
        "## GT Row Outcomes",
        "",
        "| Outcome | Rows |",
        "|---------|-----:|",
    ])
    for key, value in sorted(report["gt_row_outcomes"].items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Source Outcomes", "", "| Outcome | Sources |", "|---------|--------:|"])
    for key, value in sorted(report["source_counts"].items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Speaker-ID Abstain Reasons", "", "| Reason | Sources |", "|--------|--------:|"])
    for key, value in report["abstain_reasons"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Wrong Speaker Pairs", "", "| Pair | Count |", "|------|------:|"])
    for key, value in report["wrong_pairs"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Energy Balance", "", "| Balance | Total | Recovered | Missing ASR/Sep | Spk Abstain | Spk Wrong |", "|---------|------:|----------:|----------------:|------------:|----------:|"])
    for key in ["<0.02", "0.02-0.05", "0.05-0.15", "0.15-0.30", ">=0.30"]:
        row = report["energy_stats"].get(key, {})
        if not row:
            continue
        lines.append(
            f"| {key} | {count_total(row)} | {row.get('recovered', 0)} | "
            f"{row.get('asr_or_separator_missing_gt_row', 0)} | {row.get('speaker_abstain', 0)} | {row.get('speaker_wrong', 0)} |"
        )
    lines.extend(["", "## Duration", "", "| Duration | Total | Recovered | Missing ASR/Sep | Spk Abstain | Spk Wrong |", "|----------|------:|----------:|----------------:|------------:|----------:|"])
    for key in ["<0.5s", "0.5-1.0s", "1.0-1.5s", "1.5-2.5s", ">=2.5s"]:
        row = report["duration_stats"].get(key, {})
        if not row:
            continue
        lines.append(
            f"| {key} | {count_total(row)} | {row.get('recovered', 0)} | "
            f"{row.get('asr_or_separator_missing_gt_row', 0)} | {row.get('speaker_abstain', 0)} | {row.get('speaker_wrong', 0)} |"
        )
    lines.extend(["", "## Review Labels", "", "| Label | Total | Recovered | Missing ASR/Sep | Spk Abstain | Spk Wrong |", "|-------|------:|----------:|----------------:|------------:|----------:|"])
    for key, row in sorted(report["label_stats"].items(), key=lambda item: -count_total(item[1])):
        lines.append(
            f"| {key} | {count_total(row)} | {row.get('recovered', 0)} | "
            f"{row.get('asr_or_separator_missing_gt_row', 0)} | {row.get('speaker_abstain', 0)} | {row.get('speaker_wrong', 0)} |"
        )
    lines.extend(["", "## Per Clip", ""])
    for row in report["clips"]:
        if not row["failure_counts"]:
            continue
        labels = ",".join(row["labels"]) if row["labels"] else "none"
        failures = ", ".join(f"{key}:{value}" for key, value in row["failure_counts"].items()) or "none"
        lines.append(
            f"- rank {row['rank']:02d}: rec={row['timeline_recovered']}/{row['timeline_total']} "
            f"two_active={row['two_active']} asr_two={row['asr_two']} id_two={row['id_two']} "
            f"energy={row['energy_balance']:.3f} speakers={','.join(row['gt_speakers'])} "
            f"labels={labels} failures={failures}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze separated-source identity failure modes.")
    parser.add_argument("--identity", default="logs/separatio_param_sweep_r1/asr_raw_o16000/identity_gt_compare.json")
    parser.add_argument("--asr-compare", default="logs/separatio_param_sweep_r1/asr_raw_o16000/asr_gt_compare.json")
    parser.add_argument("--spkid-jsonl", default="logs/separatio_param_sweep_r1/asr_raw_o16000/spkid_sources.jsonl")
    parser.add_argument("--separatio-jsonl", default="logs/separatio_param_sweep_r1/separatio_raw_o16000/separatio_results.jsonl")
    parser.add_argument("--review-labels", default="logs/segment_homogeneity_clips_r3/review_labels.jsonl")
    parser.add_argument("--out-dir", default="logs/separatio_identity_failure_r1")
    args = parser.parse_args()

    report = analyze(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "separatio_identity_failure_analysis.json"
    md_path = out_dir / "separatio_identity_failure_analysis.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    headline = report["headline"]
    print(
        f"[headline] recovered={headline['gt_rows_recovered']}/{headline['gt_rows']} "
        f"two_id={headline['identity_two_speaker_clips']}/{headline['clips']} "
        f"one_id={headline['identity_one_source_clips']}/{headline['clips']}"
    )
    print(f"[gt_rows] {report['gt_row_outcomes']}")
    print(f"[sources] {report['source_counts']}")
    print(f"[abstain] {report['abstain_reasons']}")
    print(f"[wrong] {report['wrong_pairs']}")
    print(f"[out] {json_path}")
    print(f"[out] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())