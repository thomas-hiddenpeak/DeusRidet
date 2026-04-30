#!/usr/bin/env python3
"""Aggregate runtime Auditus fusion-canary reports into a regression matrix."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    name, path_text = value.split("=", 1)
    return name.strip(), Path(path_text)


def report_name(path: Path) -> str:
    if path.name == "runtime_shadow_batch_report.json":
        return path.parent.name
    return path.stem


def unique_reports(values: list[str], globs: list[str]) -> list[tuple[str, Path]]:
    reports: list[tuple[str, Path]] = [parse_report(value) for value in values]
    for pattern in globs:
        for match in sorted(glob.glob(pattern)):
            path = Path(match)
            reports.append((report_name(path), path))
    seen: set[Path] = set()
    out: list[tuple[str, Path]] = []
    for name, path in reports:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append((name, path))
    return out


def summary_int(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key, 0)
    if value is None:
        return 0
    return int(value)


def derive_clip_canary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    if "clip_canary_would_apply_gt_wrong" in summary:
        return {
            "candidates": summary_int(summary, "clip_canary_candidates"),
            "would_apply": summary_int(summary, "clip_canary_would_apply"),
            "hit": summary_int(summary, "clip_canary_would_apply_gt_hit"),
            "wrong": summary_int(summary, "clip_canary_would_apply_gt_wrong"),
            "unmapped": summary_int(summary, "clip_canary_would_apply_unmapped"),
            "wrong_events": wrong_events(report),
        }

    id_map = {int(key): value for key, value in summary.get("id_map", {}).items()}
    candidates = would_apply = hit = wrong = unmapped = 0
    derived_wrong: list[dict[str, Any]] = []
    for clip in report.get("clips", []):
        ledger = clip.get("ledger", {})
        if ledger.get("canary_candidate"):
            candidates += 1
        if not ledger.get("canary_would_apply"):
            continue
        would_apply += 1
        candidate_id = int(ledger.get("candidate_speaker_id", -1) or -1)
        mapped = id_map.get(candidate_id, "") if candidate_id >= 0 else ""
        gt_speakers = set(clip.get("gt_speakers", []))
        if not mapped:
            unmapped += 1
        elif mapped in gt_speakers:
            hit += 1
        else:
            wrong += 1
            derived_wrong.append(wrong_event(clip, candidate_id, mapped))
    return {
        "candidates": candidates,
        "would_apply": would_apply,
        "hit": hit,
        "wrong": wrong,
        "unmapped": unmapped,
        "wrong_events": derived_wrong,
    }


def wrong_event(clip: dict[str, Any], candidate_id: int, mapped: str) -> dict[str, Any]:
    return {
        "event_index": clip.get("event_index"),
        "clip_name": clip.get("clip_name"),
        "rank": clip.get("rank"),
        "candidate_id": candidate_id,
        "mapped_speaker": mapped,
        "gt_speakers": clip.get("gt_speakers", []),
        "timeline_mapped_speaker": clip.get("timeline_mapped_speaker", ""),
    }


def wrong_events(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip in report.get("clips", []):
        if clip.get("canary_status") != "would_apply_gt_wrong":
            continue
        rows.append(wrong_event(
            clip,
            int(clip.get("canary_candidate_id", -1) or -1),
            str(clip.get("canary_mapped_speaker", "") or ""),
        ))
    return rows


def births(summary: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for birth in summary.get("id_births", []):
        out.append(f"{birth.get('speaker_id')}:{birth.get('speaker') or '?'}@{birth.get('event_time_sec')}")
    return out


def verdict(row: dict[str, Any]) -> str:
    if row["authority_violations"]:
        return "fail_authority"
    if row["clip_canary_wrong"]:
        return "fail_wrong"
    if row["clip_canary_unmapped"]:
        return "fail_unmapped"
    return "pass"


def build_matrix(reports: list[tuple[str, Path]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, path in reports:
        report = load_json(path)
        summary = report.get("summary", {})
        clip = derive_clip_canary(report)
        row = {
            "run": name,
            "path": str(path),
            "raw_rows": summary_int(summary, "raw_rows"),
            "transcripts": summary_int(summary, "transcripts"),
            "fusion_shadow_events": summary_int(summary, "fusion_shadow_events"),
            "clip_fusion_events": summary_int(summary, "clip_fusion_events"),
            "births": births(summary),
            "accepted_hit": summary_int(summary, "source_mapped_gt_hit"),
            "accepted_wrong": summary_int(summary, "source_mapped_gt_wrong"),
            "accepted_unmapped": summary_int(summary, "source_accepted_unmapped"),
            "stable_hit": summary_int(summary, "source_stable_mapped_gt_hit"),
            "stable_wrong": summary_int(summary, "source_stable_mapped_gt_wrong"),
            "stable_unmapped": summary_int(summary, "source_stable_unmapped"),
            "ledger_canary_would_apply": summary_int(summary, "ledger_canary_would_apply"),
            "clip_canary_candidates": int(clip["candidates"]),
            "clip_canary_would_apply": int(clip["would_apply"]),
            "clip_canary_hit": int(clip["hit"]),
            "clip_canary_wrong": int(clip["wrong"]),
            "clip_canary_unmapped": int(clip["unmapped"]),
            "authority_violations": summary_int(summary, "ledger_authority_violations"),
            "wrong_events": clip["wrong_events"],
        }
        row["verdict"] = verdict(row)
        rows.append(row)
    failing = [row for row in rows if row["verdict"] != "pass"]
    return {
        "summary": {
            "runs": len(rows),
            "pass": len(rows) - len(failing),
            "fail": len(failing),
            "wrong_runs": [row["run"] for row in rows if row["clip_canary_wrong"]],
            "unmapped_runs": [row["run"] for row in rows if row["clip_canary_unmapped"]],
            "authority_violation_runs": [row["run"] for row in rows if row["authority_violations"]],
        },
        "runs": rows,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_markdown(matrix: dict[str, Any]) -> str:
    summary = matrix["summary"]
    lines = [
        "# Auditus Canary Regression Matrix",
        "",
        "Telemetry-only gate for runtime fusion canary reports. Pass requires zero clip-level would-apply wrong, zero would-apply unmapped, and zero authority violations.",
        "",
        "## Summary",
        "",
        f"- runs: {summary['runs']}",
        f"- pass: {summary['pass']}",
        f"- fail: {summary['fail']}",
        f"- wrong runs: {summary['wrong_runs']}",
        f"- unmapped runs: {summary['unmapped_runs']}",
        f"- authority violation runs: {summary['authority_violation_runs']}",
        "",
        "## Runs",
        "",
        "| Run | Verdict | Births | Stable H/W/U | Accepted H/W/U | Clip Canary H/W/U | Clip Would Apply | Authority |",
        "|-----|---------|-------:|-------------:|---------------:|------------------:|-----------------:|----------:|",
    ]
    for row in matrix["runs"]:
        lines.append(
            f"| {row['run']} | {row['verdict']} | {len(row['births'])} | "
            f"{row['stable_hit']}/{row['stable_wrong']}/{row['stable_unmapped']} | "
            f"{row['accepted_hit']}/{row['accepted_wrong']}/{row['accepted_unmapped']} | "
            f"{row['clip_canary_hit']}/{row['clip_canary_wrong']}/{row['clip_canary_unmapped']} | "
            f"{row['clip_canary_would_apply']} | {row['authority_violations']} |"
        )
    failed = [row for row in matrix["runs"] if row["wrong_events"]]
    if failed:
        lines.extend(["", "## Wrong Would-Apply Events", ""])
        for row in failed:
            for event in row["wrong_events"]:
                lines.append(
                    f"- {row['run']} event={event.get('event_index')} rank={event.get('rank')} "
                    f"candidate={event.get('candidate_id')} mapped={event.get('mapped_speaker') or '?'} "
                    f"gt={','.join(event.get('gt_speakers', [])) or '?'} clip={event.get('clip_name') or '?'}"
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", default=[], help="name=path or path to runtime_shadow_batch_report.json")
    parser.add_argument("--reports-glob", action="append", default=[], help="glob for runtime_shadow_batch_report.json files")
    parser.add_argument("--out-json", default="logs/fusion_canary_regression/canary_regression_matrix.json")
    parser.add_argument("--out-md", default="logs/fusion_canary_regression/canary_regression_matrix.md")
    parser.add_argument("--fail-on-violation", action="store_true")
    args = parser.parse_args()

    reports = unique_reports(args.report, args.reports_glob)
    if not reports:
        parser.error("provide at least one --report or --reports-glob")
    matrix = build_matrix(reports)
    write_json(Path(args.out_json), matrix)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(matrix), encoding="utf-8")
    summary = matrix["summary"]
    print(f"[canary] runs={summary['runs']} pass={summary['pass']} fail={summary['fail']}")
    for row in matrix["runs"]:
        print(
            f"[{row['verdict']}] {row['run']} clip_canary="
            f"{row['clip_canary_hit']}/{row['clip_canary_wrong']}/{row['clip_canary_unmapped']} "
            f"authority={row['authority_violations']}"
        )
    print(f"[out] {args.out_json}")
    print(f"[out] {args.out_md}")
    if args.fail_on_violation and summary["fail"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())