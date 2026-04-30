#!/usr/bin/env python3
"""Monitor separated-source speaker identity metrics across variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    name, path_text = value.split("=", 1)
    return name.strip(), Path(path_text)


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    return (
        float(row["identity_two_speaker_clips"]),
        float(row["timeline_recall"]),
        float(row["source_decided_accuracy"]),
        float(row["source_decided_coverage"]),
        -float(row["source_wrong"]),
        str(row["variant"]),
    )


def build_monitor(variants: list[tuple[str, Path]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    best_by_clip: dict[str, list[dict[str, Any]]] = {}
    for name, path in variants:
        report = load_report(path)
        summary = dict(report["summary"])
        summary["variant"] = name
        summary["path"] = str(path)
        summaries.append(summary)
        for clip in report.get("clips", []):
            score = {
                "variant": name,
                "rank": clip.get("rank"),
                "clip": clip.get("clip"),
                "two": bool(clip.get("identity_two_speaker_ok")),
                "one": bool(clip.get("identity_one_source_ok")),
                "recovered": int(clip.get("timeline_gt_recovered", 0)),
                "total": int(clip.get("timeline_gt_total", 0)),
                "correct_speakers": clip.get("correct_speakers", []),
            }
            best_by_clip.setdefault(str(clip.get("clip", "")), []).append(score)
    primary = max(summaries, key=rank_key)
    fallback = max(summaries, key=lambda row: (row["identity_one_source_clips"], row["timeline_recall"], row["source_decided_accuracy"]))
    rows: list[dict[str, Any]] = []
    for clip_name, candidates in sorted(best_by_clip.items()):
        best = max(candidates, key=lambda item: (item["two"], item["one"], item["recovered"], item["variant"]))
        rows.append({"clip": clip_name, "best": best, "candidates": candidates})
    return {
        "summary": summaries,
        "recommendation": {
            "primary_identity_variant": primary["variant"],
            "fallback_identity_variant": fallback["variant"],
        },
        "best_by_clip": rows,
    }


def write_markdown(path: Path, monitor: dict[str, Any]) -> None:
    rec = monitor["recommendation"]
    lines = [
        "# Separatio Identity Monitor",
        "",
        "Cross-variant monitor for separated-source speaker ID aligned to ASR+GT timeline rows.",
        "",
        "## Recommendation",
        "",
        f"- primary identity variant: `{rec['primary_identity_variant']}`",
        f"- fallback identity variant: `{rec['fallback_identity_variant']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Clips | Two-ID | One-ID | Src Acc | Src Total Acc | Coverage | Timeline Recall | Wrong | Abstain |",
        "|---------|------:|-------:|-------:|--------:|--------------:|---------:|----------------:|------:|--------:|",
    ]
    for row in sorted(monitor["summary"], key=rank_key, reverse=True):
        lines.append(
            f"| {row['variant']} | {row['clips']} | {row['identity_two_speaker_clips']} | "
            f"{row['identity_one_source_clips']} | {row['source_decided_accuracy']:.3f} | "
            f"{row['source_total_accuracy']:.3f} | {row['source_decided_coverage']:.3f} | "
            f"{row['timeline_recall']:.3f} | {row['source_wrong']} | {row['source_abstain']} |"
        )
    lines.extend(["", "## Best By Clip", ""])
    for item in monitor["best_by_clip"]:
        best = item["best"]
        lines.append(
            f"- rank {int(best.get('rank') or 0):02d}: `{best['variant']}` "
            f"two={best['two']} one={best['one']} recovered={best['recovered']}/{best['total']} "
            f"speakers={','.join(best['correct_speakers']) or '?'}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor separated-source identity summaries across variants.")
    parser.add_argument("--variant", action="append", required=True, help="name=path/to/identity_gt_compare.json")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    variants = [parse_variant(value) for value in args.variant]
    monitor = build_monitor(variants)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "separatio_identity_monitor.json"
    md_path = out_dir / "separatio_identity_monitor.md"
    json_path.write_text(json.dumps(monitor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, monitor)
    rec = monitor["recommendation"]
    print(f"[recommend] primary={rec['primary_identity_variant']} fallback={rec['fallback_identity_variant']}")
    for row in sorted(monitor["summary"], key=rank_key, reverse=True):
        print(
            f"[{row['variant']}] two={row['identity_two_speaker_clips']} one={row['identity_one_source_clips']} "
            f"acc={row['source_decided_accuracy']:.3f} coverage={row['source_decided_coverage']:.3f} "
            f"timeline={row['timeline_recall']:.3f} wrong={row['source_wrong']} abstain={row['source_abstain']}"
        )
    print(f"[out] {json_path}")
    print(f"[out] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())