#!/usr/bin/env python3
"""Monitor separated-source ASR outputs across separation variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GOOD_SCORE = 0.45


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def compact(text: Any, limit: int = 72) -> str:
    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 1] + "..."


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    name, path_text = value.split("=", 1)
    return name.strip(), Path(path_text)


def source_infos(row: dict[str, Any]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    streams = row.get("streams", {})
    for stream in ["src1", "src2"]:
        info = streams.get(stream)
        if not info:
            continue
        best = info.get("best_gt", {})
        score = float(best.get("score", 0.0))
        infos.append({
            "stream": stream,
            "text": info.get("text", ""),
            "score": score,
            "speaker": str(best.get("speaker", "?")),
            "gt_text": best.get("text", ""),
            "usable": score >= GOOD_SCORE and bool(info.get("text")),
        })
    return infos


def clip_metrics(row: dict[str, Any], variant: str) -> dict[str, Any]:
    infos = source_infos(row)
    usable = [info for info in infos if info["usable"]]
    speakers = sorted({info["speaker"] for info in usable if info["speaker"] != "?"})
    score_sum = sum(info["score"] for info in usable)
    best_source_score = max((info["score"] for info in infos), default=0.0)
    two_speaker = len(usable) >= 2 and len(speakers) >= 2
    one_source = bool(usable)
    if two_speaker:
        state = "two_speaker"
        state_rank = 2
    elif one_source:
        state = "one_source"
        state_rank = 1
    else:
        state = "unreliable"
        state_rank = 0
    return {
        "variant": variant,
        "clip": row.get("clip", ""),
        "rank": row.get("rank"),
        "state": state,
        "state_rank": state_rank,
        "matched_speakers": speakers,
        "source_score_sum": round(score_sum, 3),
        "best_source_score": round(best_source_score, 3),
        "sources": infos,
    }


def rank_key(item: dict[str, Any]) -> tuple[float, float, float, float, str]:
    view_preference = 0.0 if "official_rms" in str(item["variant"]) else 1.0
    return (
        float(item["state_rank"]),
        float(len(item["matched_speakers"])),
        float(item["source_score_sum"]),
        view_preference,
        str(item["variant"]),
    )


def summarize_variant(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [clip_metrics(row, name) for row in rows]
    clips = len(metrics)
    two = sum(1 for row in metrics if row["state"] == "two_speaker")
    one = sum(1 for row in metrics if row["state"] in {"two_speaker", "one_source"})
    avg_best = sum(float(row["best_source_score"]) for row in metrics) / max(1, clips)
    avg_sum = sum(float(row["source_score_sum"]) for row in metrics) / max(1, clips)
    return {
        "variant": name,
        "clips": clips,
        "two_speaker": two,
        "one_or_two_source": one,
        "unreliable": clips - one,
        "avg_best_source_score": round(avg_best, 3),
        "avg_usable_source_score_sum": round(avg_sum, 3),
        "two_speaker_ranks": [row["rank"] for row in metrics if row["state"] == "two_speaker"],
    }


def build_monitor(variants: list[tuple[str, Path]]) -> dict[str, Any]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    by_clip: dict[str, list[dict[str, Any]]] = {}
    for name, path in variants:
        rows = load_json(path)
        loaded[name] = rows
        summaries.append(summarize_variant(name, rows))
        for row in rows:
            metric = clip_metrics(row, name)
            by_clip.setdefault(str(row.get("clip", "")), []).append(metric)

    best_rows: list[dict[str, Any]] = []
    for clip, candidates in sorted(by_clip.items()):
        best = max(candidates, key=rank_key)
        best_rows.append({
            "clip": clip,
            "rank": best.get("rank"),
            "best_variant": best["variant"],
            "state": best["state"],
            "matched_speakers": best["matched_speakers"],
            "source_score_sum": best["source_score_sum"],
            "best_source_score": best["best_source_score"],
            "candidates": sorted(candidates, key=rank_key, reverse=True),
        })

    best_two = max(summaries, key=lambda row: (row["two_speaker"], row["one_or_two_source"], row["avg_usable_source_score_sum"]))
    best_fallback = max(summaries, key=lambda row: (row["one_or_two_source"], row["two_speaker"], row["avg_best_source_score"]))
    return {
        "variants": [{"name": name, "path": str(path)} for name, path in variants],
        "summary": summaries,
        "recommendation": {
            "primary_two_speaker_variant": best_two["variant"],
            "fallback_one_source_variant": best_fallback["variant"],
        },
        "best_by_clip": best_rows,
    }


def write_markdown(path: Path, monitor: dict[str, Any]) -> None:
    lines = [
        "# Separatio ASR Monitor",
        "",
        "Cross-variant monitor for separated-source ASR outputs. It picks the currently most usable output per clip without changing online behavior.",
        "",
        "## Recommendation",
        "",
        f"- primary two-speaker variant: `{monitor['recommendation']['primary_two_speaker_variant']}`",
        f"- fallback one-source variant: `{monitor['recommendation']['fallback_one_source_variant']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Clips | Two-speaker | One-or-two | Unreliable | Avg best score | Avg usable sum |",
        "|---------|------:|------------:|-----------:|-----------:|---------------:|---------------:|",
    ]
    for row in monitor["summary"]:
        lines.append(
            f"| {row['variant']} | {row['clips']} | {row['two_speaker']} | {row['one_or_two_source']} | "
            f"{row['unreliable']} | {row['avg_best_source_score']:.3f} | {row['avg_usable_source_score_sum']:.3f} |"
        )
    lines.extend(["", "## Two-Speaker Ranks", ""])
    for row in monitor["summary"]:
        ranks = ", ".join(str(item) for item in row["two_speaker_ranks"]) or "none"
        lines.append(f"- {row['variant']}: {ranks}")
    lines.extend(["", "## Best By Clip", ""])
    for row in monitor["best_by_clip"]:
        lines.append(
            f"- rank {int(row.get('rank') or 0):02d}: `{row['best_variant']}` "
            f"state={row['state']} speakers={','.join(row['matched_speakers']) or '?'} "
            f"score_sum={row['source_score_sum']:.3f}"
        )
        best_candidate = row["candidates"][0]
        for source in best_candidate.get("sources", []):
            if not source.get("usable"):
                continue
            lines.append(
                f"  - {source['stream']} -> {source['speaker']} score={source['score']:.3f}: "
                f"{compact(source.get('text'), 96)}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_variants() -> list[tuple[str, Path]]:
    candidates = [
        ("raw_window", Path("logs/separatio_asr_window_r1/asr_gt_compare.json")),
        ("frcrn_window", Path("logs/separatio_asr_frcrn_window_r1/asr_gt_compare.json")),
    ]
    return [(name, path) for name, path in candidates if path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor separated-source ASR results across variants.")
    parser.add_argument("--variant", action="append", default=[], help="name=path/to/asr_gt_compare.json")
    parser.add_argument("--out-dir", default="logs/separatio_asr_monitor_r1")
    args = parser.parse_args()

    variants = [parse_variant(value) for value in args.variant] if args.variant else default_variants()
    if not variants:
        raise SystemExit("ERROR: no variants found; pass --variant name=path/to/asr_gt_compare.json")

    monitor = build_monitor(variants)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "separatio_asr_monitor.json"
    md_path = out_dir / "separatio_asr_monitor.md"
    json_path.write_text(json.dumps(monitor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, monitor)

    print(f"[variants] {len(variants)}")
    for row in monitor["summary"]:
        print(
            f"[{row['variant']}] clips={row['clips']} two={row['two_speaker']} "
            f"one_or_two={row['one_or_two_source']} unreliable={row['unreliable']}"
        )
    rec = monitor["recommendation"]
    print(f"[recommend] primary={rec['primary_two_speaker_variant']} fallback={rec['fallback_one_source_variant']}")
    print(f"[out] {json_path}")
    print(f"[out] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
