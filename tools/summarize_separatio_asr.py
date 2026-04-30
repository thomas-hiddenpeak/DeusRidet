#!/usr/bin/env python3
"""Compare separated-source ASR transcripts against homogeneity GT windows."""
from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PUNCT_RE = re.compile(r"[\s\u3000，。！？；：、,.!?;:'\"“”‘’（）()\[\]{}<>《》【】·—_\-]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def norm_text(text: Any) -> str:
    return PUNCT_RE.sub("", str(text or "")).lower()


def sim(a: str, b: str) -> float:
    a_n = norm_text(a)
    b_n = norm_text(b)
    if not a_n or not b_n:
        return 0.0
    return max(difflib.SequenceMatcher(None, a_n, b_n).ratio(), partial_sim(a_n, b_n))


def partial_sim(a: str, b: str) -> float:
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if not short or not long:
        return 0.0
    if short in long:
        return 1.0
    best = 0.0
    min_window = max(1, len(short) - 2)
    max_window = min(len(long), len(short) + 8)
    for window in range(min_window, max_window + 1):
        for start in range(0, len(long) - window + 1):
            part = long[start:start + window]
            matcher = difflib.SequenceMatcher(None, short, part)
            matched = sum(block.size for block in matcher.get_matching_blocks())
            coverage = matched / max(1, len(short))
            score = max(matcher.ratio(), coverage)
            if score > best:
                best = score
    return best


def compact(text: Any, limit: int = 80) -> str:
    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 1] + "..."


def build_manifest(path: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        manifest[str(row.get("clip_name", ""))] = row
    return manifest


def best_match(text: str, gt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {"score": 0.0, "speaker": "?", "text": "", "idx": -1}
    for gt in gt_rows:
        score = sim(text, str(gt.get("text", "")))
        if score > best["score"]:
            best = {
                "score": score,
                "speaker": gt.get("speaker", "?"),
                "text": gt.get("text", ""),
                "idx": gt.get("idx", -1),
                "overlap_sec": gt.get("overlap_sec", 0.0),
            }
    return best


def summarize_clip(clip: str, rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item = manifest.get(clip, {})
    candidate = item.get("candidate", {})
    gt_rows = candidate.get("gt_rows", [])
    streams: dict[str, dict[str, Any]] = {}
    for row in rows:
        stream = str(row.get("stream", "unknown"))
        match = best_match(str(row.get("text", "")), gt_rows)
        streams[stream] = {
            "text": row.get("text", ""),
            "raw_text": row.get("raw_text", ""),
            "tokens": row.get("tokens", 0),
            "total_ms": row.get("total_ms", 0.0),
            "best_gt": match,
        }

    src_matches = [streams.get("src1"), streams.get("src2")]
    decided = [m for m in src_matches if m and m["best_gt"]["score"] >= 0.45 and m.get("text")]
    matched_speakers = {str(m["best_gt"]["speaker"]) for m in decided}
    distinct_gt = len({str(gt.get("speaker", "?")) for gt in gt_rows})
    separation_ok = len(decided) >= 2 and len(matched_speakers) >= 2
    one_good = any(m["best_gt"]["score"] >= 0.45 for m in src_matches if m)

    return {
        "clip": clip,
        "rank": item.get("rank"),
        "clip_start_sec": item.get("clip_start_sec"),
        "clip_end_sec": item.get("clip_end_sec"),
        "gt_speakers": candidate.get("gt_speakers", []),
        "gt_rows": gt_rows,
        "streams": streams,
        "distinct_gt_speakers": distinct_gt,
        "matched_speakers": sorted(matched_speakers),
        "separation_ok": separation_ok,
        "one_good_source": one_good,
    }


def write_markdown(path: Path, summaries: list[dict[str, Any]], source_path: Path, manifest_path: Path) -> None:
    counts = Counter()
    for row in summaries:
        counts["clips"] += 1
        counts["separation_ok"] += int(bool(row["separation_ok"]))
        counts["one_good_source"] += int(bool(row["one_good_source"]))

    lines = [
        "# Separatio ASR Summary",
        "",
        "Offline diagnostic: ASR over separated `src1/src2` streams, compared against the homogeneity clip GT rows.",
        "",
        "## Inputs",
        "",
        f"- asr_jsonl: `{source_path}`",
        f"- manifest: `{manifest_path}`",
        "",
        "## Summary",
        "",
        f"- clips: {counts['clips']}",
        f"- clips with two distinct matched GT speakers: {counts['separation_ok']}",
        f"- clips with at least one strong source match: {counts['one_good_source']}",
        "",
        "## Per Clip",
        "",
    ]
    for row in summaries:
        lines.extend([
            f"### Rank {int(row.get('rank') or 0):02d} — {row['clip']}",
            "",
            f"- window: {row.get('clip_start_sec')}-{row.get('clip_end_sec')}s",
            f"- gt_speakers: {', '.join(row.get('gt_speakers') or []) or '?'}",
            f"- separation_ok: {row['separation_ok']}",
            f"- matched_speakers: {', '.join(row['matched_speakers']) or '?'}",
            "",
            "GT rows:",
        ])
        for gt in row.get("gt_rows", []):
            lines.append(
                f"- {gt.get('speaker', '?')} {float(gt.get('start_sec', 0.0)):.2f}-{float(gt.get('end_sec', 0.0)):.2f}s: {compact(gt.get('text'), 100)}"
            )
        lines.append("")
        for stream in ["mix", "src1", "src2"]:
            info = row["streams"].get(stream)
            if not info:
                continue
            best = info["best_gt"]
            lines.extend([
                f"{stream}:",
                f"- ASR: {compact(info.get('text'), 120)}",
                f"- best_gt: {best.get('speaker')} score={float(best.get('score', 0.0)):.3f} text={compact(best.get('text'), 100)}",
                "",
            ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize separated-source ASR against clip GT rows.")
    parser.add_argument("--asr-jsonl", default="logs/separatio_asr_window_r1/asr_sources.jsonl")
    parser.add_argument("--manifest", default="logs/segment_homogeneity_clips_r3/clip_manifest.jsonl")
    parser.add_argument("--out-dir", default="logs/separatio_asr_window_r1")
    args = parser.parse_args()

    asr_path = Path(args.asr_jsonl)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(manifest_path)
    by_clip: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(asr_path):
        by_clip.setdefault(str(row.get("clip", "")), []).append(row)

    summaries = [summarize_clip(clip, rows, manifest) for clip, rows in sorted(by_clip.items())]
    (out_dir / "asr_gt_compare.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "asr_gt_compare.md", summaries, asr_path, manifest_path)

    ok = sum(1 for row in summaries if row["separation_ok"])
    one = sum(1 for row in summaries if row["one_good_source"])
    print(f"[summary] clips={len(summaries)} separation_ok={ok} one_good_source={one}")
    print(f"[out] {out_dir / 'asr_gt_compare.json'}")
    print(f"[out] {out_dir / 'asr_gt_compare.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
