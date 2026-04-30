#!/usr/bin/env python3
"""Summarize human labels from a Step 18 homogeneity listening sheet."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


LABELS = [
    "音频多人",
    "换人清楚",
    "ASR合并多人",
    "GT边界可疑",
    "无害语气词",
    "关键短表态",
    "需要切分探针",
]

SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+VAD\s+#(\d+)\s+([0-9.]+)-([0-9.]+)s")
LABEL_RE = re.compile(r"\[([xX ])\]\s*([^\[]+?)(?=\s+\[[xX ]\]|$)")


def parse_labels(line: str) -> list[str]:
    checked: list[str] = []
    for state, label_text in LABEL_RE.findall(line):
        if state.lower() != "x":
            continue
        normalized = label_text.strip()
        for label in LABELS:
            if normalized.startswith(label):
                checked.append(label)
                break
    return checked


def parse_review(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        match = SECTION_RE.match(line)
        if match:
            if current is not None:
                rows.append(current)
            rank, vad_idx, start_sec, end_sec = match.groups()
            current = {
                "rank": int(rank),
                "vad_idx": int(vad_idx),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "labels": [],
                "primary_failure": "",
                "speaker_attribution_note": "",
            }
            continue
        if current is None:
            continue
        if line.startswith("- labels:"):
            current["labels"] = parse_labels(line)
        elif line.startswith("- primary_failure"):
            current["primary_failure"] = line.split(":", 1)[1].strip()
        elif line.startswith("- speaker_attribution_note"):
            current["speaker_attribution_note"] = line.split(":", 1)[1].strip()
    if current is not None:
        rows.append(current)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    blank = 0
    for row in rows:
        labels = list(row.get("labels", []))
        if not labels:
            blank += 1
        label_counts.update(labels)
        for a, b in combinations(sorted(labels), 2):
            pair_counts[f"{a}+{b}"] += 1
    return {
        "n_items": len(rows),
        "n_unlabeled": blank,
        "label_counts": dict(label_counts.most_common()),
        "pair_counts": dict(pair_counts.most_common()),
        "audio_multi_asr_merge": [
            r["rank"] for r in rows
            if "音频多人" in r.get("labels", []) and "ASR合并多人" in r.get("labels", [])
        ],
        "gt_boundary": [r["rank"] for r in rows if "GT边界可疑" in r.get("labels", [])],
        "key_stance": [r["rank"] for r in rows if "关键短表态" in r.get("labels", [])],
        "split_probe": [r["rank"] for r in rows if "需要切分探针" in r.get("labels", [])],
    }


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], source: Path) -> None:
    lines = [
        "# Homogeneity Review Summary",
        "",
        f"Source: `{source}`",
        "",
        "## Counts",
        "",
        f"- items: {summary['n_items']}",
        f"- unlabeled: {summary['n_unlabeled']}",
        "",
        "## Labels",
        "",
    ]
    for label, count in summary["label_counts"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Common Pairs", ""])
    for pair, count in list(summary["pair_counts"].items())[:12]:
        lines.append(f"- {pair}: {count}")
    lines.extend([
        "",
        "## Action Buckets",
        "",
        f"- 音频多人 + ASR合并多人: {summary['audio_multi_asr_merge']}",
        f"- GT边界可疑: {summary['gt_boundary']}",
        f"- 关键短表态: {summary['key_stance']}",
        f"- 需要切分探针: {summary['split_probe']}",
        "",
        "## Per Candidate",
        "",
    ])
    for row in rows:
        labels = ", ".join(row.get("labels", [])) or "none"
        lines.append(
            f"- {row['rank']:02d} VAD #{row['vad_idx']} "
            f"{row['start_sec']:.2f}-{row['end_sec']:.2f}s: {labels}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize filled Step 18b homogeneity hearing-review labels.")
    parser.add_argument("--review", default="/tmp/segment_homogeneity_clips_r3/hearing_review.md")
    parser.add_argument("--out-dir", default="/tmp/segment_homogeneity_clips_r3")
    args = parser.parse_args()

    review_path = Path(args.review)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_review(review_path)
    summary = summarize(rows)

    labels_path = out_dir / "review_labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_json_path = out_dir / "review_summary.json"
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md_path = out_dir / "review_summary.md"
    write_markdown(summary_md_path, rows, summary, review_path)

    print(f"[input] {review_path}")
    print(f"[items] {summary['n_items']} unlabeled={summary['n_unlabeled']}")
    print(f"[out] {labels_path}")
    print(f"[out] {summary_json_path}")
    print(f"[out] {summary_md_path}")
    for label, count in summary["label_counts"].items():
        print(f"[label] {label}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())