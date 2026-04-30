#!/usr/bin/env python3
"""Fuse separated-source speaker evidence across Auditus separator variants."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MONITORS = [
    "logs/separatio_param_sweep_r1/identity_monitor/separatio_identity_monitor.json",
    "logs/separatio_param_sweep_r2/identity_monitor/separatio_identity_monitor.json",
]
SIM_GRID = [0.30, 0.35, 0.40, 0.45, 0.50]
MARGIN_GRID = [0.00, 0.03, 0.05, 0.08, 0.10]


@dataclass(frozen=True)
class Variant:
    name: str
    path: Path


@dataclass(frozen=True)
class SelectorConfig:
    mode: str
    min_similarity: float
    min_margin: float
    max_speakers: int
    min_text_chars: int


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_variant(value: str) -> Variant:
    if "=" not in value:
        path = Path(value)
        return Variant(path.parent.name, path)
    name, path_text = value.split("=", 1)
    return Variant(name.strip(), Path(path_text.strip()))


def variants_from_monitor(path: Path) -> list[Variant]:
    data = load_json(path)
    variants: list[Variant] = []
    for row in data.get("summary", []):
        variants.append(Variant(str(row["variant"]), Path(str(row["path"]))))
    return variants


def collect_variants(monitors: list[str], explicit: list[str]) -> list[Variant]:
    variants: list[Variant] = []
    for monitor in monitors:
        variants.extend(variants_from_monitor(Path(monitor)))
    variants.extend(parse_variant(value) for value in explicit)

    deduped: list[Variant] = []
    seen: set[str] = set()
    for variant in variants:
        key = str(variant.path)
        if key in seen:
            continue
        if not variant.path.is_file():
            raise FileNotFoundError(variant.path)
        seen.add(key)
        deduped.append(variant)
    return deduped


def text_chars(value: Any) -> int:
    return len(str(value or "").strip())


def runtime_score(row: dict[str, Any]) -> float:
    similarity = float(row["similarity"])
    margin = max(0.0, float(row["margin"]))
    rms = max(0.0, float(row["rms"]))
    chars = min(20, int(row["text_chars"]))
    return similarity + 0.5 * margin + 0.02 * chars + 0.10 * math.log1p(rms * 100.0)


def evidence_from_reports(variants: list[Variant]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    clips: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for variant in variants:
        report = load_json(variant.path)
        for clip in report.get("clips", []):
            clip_name = str(clip.get("clip", ""))
            clips.setdefault(
                clip_name,
                {
                    "clip": clip_name,
                    "rank": clip.get("rank"),
                    "gt_speakers": clip.get("gt_speakers", []),
                    "timeline_gt_total": int(clip.get("timeline_gt_total", 0)),
                },
            )
            for source in clip.get("source_evals", []):
                pred_speaker = str(source.get("pred_speaker", "?"))
                if pred_speaker == "?":
                    continue
                chars = text_chars(source.get("asr_text", ""))
                row = {
                    "variant": variant.name,
                    "clip": clip_name,
                    "rank": clip.get("rank"),
                    "stream": str(source.get("stream", "")),
                    "pred_speaker": pred_speaker,
                    "accepted": bool(source.get("accepted", False)),
                    "similarity": float(source.get("similarity", 0.0)),
                    "margin": float(source.get("margin", 0.0)),
                    "rms": float(source.get("rms", 0.0)),
                    "text_chars": chars,
                    "asr_text": source.get("asr_text", ""),
                    "has_asr_match": bool(source.get("has_asr_match", False)),
                    "target_gt_idx": int(source.get("target_gt_idx", -1) or -1),
                    "target_speaker": str(source.get("target_speaker", "?")),
                    "gt_text": source.get("gt_text", ""),
                }
                row["runtime_score"] = round(runtime_score(row), 6)
                evidence.append(row)
    return clips, evidence


def evidence_allowed(row: dict[str, Any], config: SelectorConfig) -> bool:
    return (
        int(row["text_chars"]) >= config.min_text_chars
        and float(row["similarity"]) >= config.min_similarity
        and float(row["margin"]) >= config.min_margin
    )


def group_score(rows: list[dict[str, Any]]) -> float:
    scores = [float(row["runtime_score"]) for row in rows]
    support = len({str(row["variant"]) for row in rows})
    return 0.65 * max(scores) + 0.35 * (sum(scores) / len(scores)) + 0.15 * math.log1p(support)


def selected_eval_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["runtime_score"]), row["variant"], row["stream"]), reverse=True)
    if mode == "single_best":
        return ordered[:1]
    if mode == "support_union":
        return ordered
    raise ValueError(f"unknown selector mode: {mode}")


def evaluate(clips: dict[str, dict[str, Any]], evidence: list[dict[str, Any]], config: SelectorConfig) -> dict[str, Any]:
    by_clip: dict[str, list[dict[str, Any]]] = {clip: [] for clip in clips}
    for row in evidence:
        if evidence_allowed(row, config):
            by_clip.setdefault(str(row["clip"]), []).append(row)

    clip_rows: list[dict[str, Any]] = []
    timeline_total = 0
    timeline_recovered = 0
    decision_correct = 0
    decision_wrong = 0
    decision_unknown = 0
    support_count = 0

    for clip_name, meta in sorted(clips.items(), key=lambda item: int(item[1].get("rank") or 0)):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in by_clip.get(clip_name, []):
            groups.setdefault(str(row["pred_speaker"]), []).append(row)

        candidates = []
        for speaker, rows in groups.items():
            candidates.append((group_score(rows), speaker, rows))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        selected: list[dict[str, Any]] = []
        recovered_ids: set[int] = set()
        correct_speakers: set[str] = set()
        wrong_speakers: set[str] = set()
        timeline_total += int(meta.get("timeline_gt_total", 0))

        for score, speaker, rows in candidates[: config.max_speakers]:
            eval_rows = selected_eval_rows(rows, config.mode)
            correct_matches = [
                row for row in eval_rows
                if bool(row["has_asr_match"]) and str(row["target_speaker"]) == speaker
            ]
            wrong_matches = [
                row for row in eval_rows
                if bool(row["has_asr_match"]) and str(row["target_speaker"]) != speaker
            ]
            support_count += len(eval_rows)
            if correct_matches:
                decision_correct += 1
                correct_speakers.add(speaker)
                for row in correct_matches:
                    target_idx = int(row.get("target_gt_idx", -1))
                    if target_idx >= 0:
                        recovered_ids.add(target_idx)
            elif wrong_matches:
                decision_wrong += 1
                wrong_speakers.add(speaker)
            else:
                decision_unknown += 1
            selected.append(
                {
                    "pred_speaker": speaker,
                    "group_score": round(score, 6),
                    "support_variants": sorted({str(row["variant"]) for row in rows}),
                    "support_count": len(rows),
                    "eval_support_count": len(eval_rows),
                    "correct": bool(correct_matches),
                    "wrong": bool(wrong_matches) and not correct_matches,
                    "recovered_gt_ids": sorted({int(row["target_gt_idx"]) for row in correct_matches if int(row["target_gt_idx"]) >= 0}),
                    "top_evidence": format_evidence(sorted(rows, key=runtime_score, reverse=True)[0]),
                    "eval_evidence": [format_evidence(row) for row in eval_rows[:8]],
                }
            )

        timeline_recovered += len(recovered_ids)
        clip_rows.append(
            {
                "clip": clip_name,
                "rank": meta.get("rank"),
                "gt_speakers": meta.get("gt_speakers", []),
                "selected": selected,
                "correct_speakers": sorted(correct_speakers),
                "wrong_speakers": sorted(wrong_speakers),
                "identity_two_speaker_ok": len(correct_speakers) >= 2,
                "identity_one_source_ok": bool(correct_speakers),
                "timeline_gt_total": int(meta.get("timeline_gt_total", 0)),
                "timeline_gt_recovered": len(recovered_ids),
            }
        )

    decisions = decision_correct + decision_wrong
    selected_total = decision_correct + decision_wrong + decision_unknown
    return {
        "config": config.__dict__,
        "summary": {
            "clips": len(clip_rows),
            "identity_two_speaker_clips": sum(1 for row in clip_rows if row["identity_two_speaker_ok"]),
            "identity_one_source_clips": sum(1 for row in clip_rows if row["identity_one_source_ok"]),
            "decision_correct": decision_correct,
            "decision_wrong": decision_wrong,
            "decision_unknown": decision_unknown,
            "decision_accuracy": round(decision_correct / decisions, 3) if decisions else 0.0,
            "selected_groups": selected_total,
            "avg_eval_support_per_group": round(support_count / selected_total, 3) if selected_total else 0.0,
            "timeline_gt_total": timeline_total,
            "timeline_gt_recovered": timeline_recovered,
            "timeline_recall": round(timeline_recovered / timeline_total, 3) if timeline_total else 0.0,
        },
        "clips": clip_rows,
    }


def format_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": row["variant"],
        "stream": row["stream"],
        "pred_speaker": row["pred_speaker"],
        "similarity": round(float(row["similarity"]), 3),
        "margin": round(float(row["margin"]), 3),
        "rms": round(float(row["rms"]), 5),
        "text_chars": int(row["text_chars"]),
        "runtime_score": round(float(row["runtime_score"]), 3),
        "asr_text": row.get("asr_text", ""),
        "target_speaker": row.get("target_speaker", "?"),
        "target_gt_idx": row.get("target_gt_idx", -1),
        "gt_text": row.get("gt_text", ""),
    }


def baseline_rows(variants: list[Variant]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        summary = load_json(variant.path).get("summary", {})
        rows.append({"variant": variant.name, "path": str(variant.path), **summary})
    return rows


def baseline_rank(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        float(row.get("identity_two_speaker_clips", 0)),
        float(row.get("timeline_recall", 0.0)),
        float(row.get("source_decided_accuracy", 0.0)),
        -float(row.get("source_wrong", 0)),
        str(row.get("variant", "")),
    )


def sweep_configs(max_speakers: int, min_text_chars: int) -> list[SelectorConfig]:
    configs: list[SelectorConfig] = []
    for mode in ["support_union", "single_best"]:
        for min_similarity in SIM_GRID:
            for min_margin in MARGIN_GRID:
                configs.append(SelectorConfig(mode, min_similarity, min_margin, max_speakers, min_text_chars))
    return configs


def sweep_rank(report: dict[str, Any]) -> tuple[float, float, float, float, float]:
    summary = report["summary"]
    return (
        float(summary["timeline_recall"]),
        float(summary["identity_two_speaker_clips"]),
        float(summary["decision_accuracy"]),
        -float(summary["decision_wrong"]),
        -float(summary["selected_groups"]),
    )


def compact(text: Any, limit: int = 72) -> str:
    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 1] + "..."


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    single = report["single_best"]
    baselines = sorted(report["baselines"], key=baseline_rank, reverse=True)
    best_baseline = baselines[0]
    sweep = report.get("sweep", [])
    lines = [
        "# Auditus Separatio Fusion Selector",
        "",
        "Offline selector/fusion report over existing separated-source ASR + speaker-ID artifacts.",
        "GT fields are used only for scoring after runtime-visible evidence has been selected.",
        "",
        "## Runtime Selection Signals",
        "",
        "- predicted speaker identity",
        "- speaker similarity and margin",
        "- separated-source RMS",
        "- separated-source ASR text length",
        "- cross-variant support for the same predicted speaker",
        "",
        "## Summary",
        "",
        "| Method | Two-ID | One-ID | Timeline | Correct Decisions | Wrong | Decision Acc | Avg Evidence |",
        "|--------|-------:|-------:|---------:|------------------:|------:|-------------:|-------------:|",
        summary_row("fusion/support_union", selected["summary"]),
        summary_row("fusion/single_best", single["summary"]),
        baseline_summary_row(f"baseline/{best_baseline['variant']}", best_baseline),
        "",
        "## Baselines",
        "",
        "| Variant | Two-ID | One-ID | Timeline | Correct | Wrong | Abstain | Decided Acc |",
        "|---------|-------:|-------:|---------:|--------:|------:|--------:|------------:|",
    ]
    for row in baselines:
        lines.append(
            f"| {row['variant']} | {row['identity_two_speaker_clips']} | {row['identity_one_source_clips']} | "
            f"{float(row['timeline_recall']):.3f} | {row['source_correct']} | {row['source_wrong']} | "
            f"{row['source_abstain']} | {float(row['source_decided_accuracy']):.3f} |"
        )

    if sweep:
        lines.extend([
            "",
            "## Threshold Sweep",
            "",
            "| Rank | Mode | Min Sim | Min Margin | Two-ID | One-ID | Timeline | Wrong | Decision Acc |",
            "|-----:|------|--------:|-----------:|-------:|-------:|---------:|------:|-------------:|",
        ])
        for idx, item in enumerate(sweep[:10], start=1):
            cfg = item["config"]
            summary = item["summary"]
            lines.append(
                f"| {idx} | {cfg['mode']} | {cfg['min_similarity']:.2f} | {cfg['min_margin']:.2f} | "
                f"{summary['identity_two_speaker_clips']} | {summary['identity_one_source_clips']} | "
                f"{summary['timeline_recall']:.3f} | {summary['decision_wrong']} | {summary['decision_accuracy']:.3f} |"
            )

    lines.extend(["", "## Per Clip Fusion Decisions", ""])
    for clip in selected["clips"]:
        lines.append(
            f"- rank {int(clip.get('rank') or 0):02d}: two={clip['identity_two_speaker_ok']} "
            f"one={clip['identity_one_source_ok']} recovered={clip['timeline_gt_recovered']}/{clip['timeline_gt_total']} "
            f"speakers={','.join(clip['correct_speakers']) or '?'}"
        )
        for item in clip["selected"]:
            top = item["top_evidence"]
            mark = "ok" if item["correct"] else ("wrong" if item["wrong"] else "unknown")
            lines.append(
                f"  - {item['pred_speaker']} {mark}: score={item['group_score']:.3f} "
                f"hypotheses={item['support_count']} variants={','.join(item['support_variants'])} "
                f"top={top['variant']}/{top['stream']} sim={top['similarity']:.3f} "
                f"margin={top['margin']:.3f} text={compact(top['asr_text'])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_row(name: str, summary: dict[str, Any]) -> str:
    return (
        f"| {name} | {summary['identity_two_speaker_clips']} | {summary['identity_one_source_clips']} | "
        f"{summary['timeline_recall']:.3f} ({summary['timeline_gt_recovered']}/{summary['timeline_gt_total']}) | "
        f"{summary['decision_correct']} | {summary['decision_wrong']} | {summary['decision_accuracy']:.3f} | "
        f"{summary['avg_eval_support_per_group']:.3f} |"
    )


def baseline_summary_row(name: str, row: dict[str, Any]) -> str:
    return (
        f"| {name} | {row['identity_two_speaker_clips']} | {row['identity_one_source_clips']} | "
        f"{float(row['timeline_recall']):.3f} ({row['timeline_gt_recovered']}/{row['timeline_gt_total']}) | "
        f"{row['source_correct']} | {row['source_wrong']} | {float(row['source_decided_accuracy']):.3f} | 1.000 |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse separated-source identity evidence across Auditus variants.")
    parser.add_argument("--monitor", action="append", default=[], help="separatio_identity_monitor.json input")
    parser.add_argument("--variant", action="append", default=[], help="name=path/to/identity_gt_compare.json")
    parser.add_argument("--out-dir", default="logs/separatio_fusion_selector_r1")
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--max-speakers", type=int, default=2)
    parser.add_argument("--min-text-chars", type=int, default=1)
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()

    monitors = args.monitor or DEFAULT_MONITORS
    variants = collect_variants(monitors, args.variant)
    clips, evidence = evidence_from_reports(variants)
    selected_config = SelectorConfig(
        "support_union", args.min_similarity, args.min_margin, args.max_speakers, args.min_text_chars
    )
    single_config = SelectorConfig(
        "single_best", args.min_similarity, args.min_margin, args.max_speakers, args.min_text_chars
    )
    selected = evaluate(clips, evidence, selected_config)
    single = evaluate(clips, evidence, single_config)

    sweep: list[dict[str, Any]] = []
    if not args.no_sweep:
        for config in sweep_configs(args.max_speakers, args.min_text_chars):
            item = evaluate(clips, evidence, config)
            sweep.append({"config": config.__dict__, "summary": item["summary"]})
        sweep.sort(key=sweep_rank, reverse=True)

    report = {
        "inputs": {
            "monitors": monitors,
            "variants": [{"name": variant.name, "path": str(variant.path)} for variant in variants],
        },
        "selection_rule": {
            "uses_gt_for_selection": False,
            "runtime_signals": ["pred_speaker", "similarity", "margin", "rms", "text_chars", "variant_support"],
        },
        "selected": selected,
        "single_best": single,
        "baselines": baseline_rows(variants),
        "sweep": sweep,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fusion_selector_report.json"
    md_path = out_dir / "fusion_selector_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, report)

    selected_summary = selected["summary"]
    single_summary = single["summary"]
    print(
        f"[fusion] support_union two={selected_summary['identity_two_speaker_clips']} "
        f"one={selected_summary['identity_one_source_clips']} "
        f"timeline={selected_summary['timeline_recall']:.3f} "
        f"wrong={selected_summary['decision_wrong']} acc={selected_summary['decision_accuracy']:.3f}"
    )
    print(
        f"[fusion] single_best two={single_summary['identity_two_speaker_clips']} "
        f"one={single_summary['identity_one_source_clips']} "
        f"timeline={single_summary['timeline_recall']:.3f} "
        f"wrong={single_summary['decision_wrong']} acc={single_summary['decision_accuracy']:.3f}"
    )
    if sweep:
        best = sweep[0]
        cfg = best["config"]
        summary = best["summary"]
        print(
            f"[sweep-best] mode={cfg['mode']} min_sim={cfg['min_similarity']:.2f} "
            f"min_margin={cfg['min_margin']:.2f} two={summary['identity_two_speaker_clips']} "
            f"timeline={summary['timeline_recall']:.3f} wrong={summary['decision_wrong']}"
        )
    print(f"[out] {json_path}")
    print(f"[out] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())