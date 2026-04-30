#!/usr/bin/env python3
"""Join separated-source ASR and speaker ID against GT timeline rows."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GOOD_SCORE = 0.45


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(text: Any, limit: int = 88) -> str:
    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 1] + "..."


def spkid_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row.get("clip", "")), str(row.get("stream", ""))): row for row in rows}


def speaker_decision(spkid: dict[str, Any] | None,
                     match_threshold: float | None,
                     min_margin: float | None) -> tuple[str, bool]:
    if not spkid:
        return "?", False
    if match_threshold is None or min_margin is None:
        pred_speaker = str(spkid.get("pred_speaker", "?"))
        accepted = bool(spkid.get("accepted", False)) and pred_speaker != "?"
        return pred_speaker if accepted else "?", accepted
    raw_speaker = str(spkid.get("best_raw_speaker", "?"))
    similarity = float(spkid.get("similarity", 0.0))
    margin = float(spkid.get("margin", 0.0))
    accepted = raw_speaker != "?" and similarity >= match_threshold and margin >= min_margin
    return raw_speaker if accepted else "?", accepted


def evaluate_source(clip: str, stream: str, asr_info: dict[str, Any], spkid: dict[str, Any] | None,
                    match_threshold: float | None, min_margin: float | None) -> dict[str, Any]:
    best = asr_info.get("best_gt", {})
    score = float(best.get("score", 0.0))
    target_speaker = str(best.get("speaker", "?"))
    target_idx = int(best.get("idx", -1) or -1)
    has_asr_match = score >= GOOD_SCORE and bool(asr_info.get("text")) and target_speaker != "?"
    pred_speaker, accepted = speaker_decision(spkid, match_threshold, min_margin)
    correct = has_asr_match and accepted and pred_speaker == target_speaker
    wrong = has_asr_match and accepted and pred_speaker != target_speaker
    abstain = has_asr_match and not accepted
    return {
        "clip": clip,
        "stream": stream,
        "asr_text": asr_info.get("text", ""),
        "asr_score": round(score, 3),
        "target_gt_idx": target_idx,
        "target_speaker": target_speaker,
        "pred_speaker": pred_speaker,
        "accepted": accepted,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "has_asr_match": has_asr_match,
        "similarity": round(float((spkid or {}).get("similarity", 0.0)), 3),
        "margin": round(float((spkid or {}).get("margin", 0.0)), 3),
        "rms": round(float((spkid or {}).get("rms", 0.0)), 5),
        "gt_text": best.get("text", ""),
    }


def summarize(asr_rows: list[dict[str, Any]], spkid_rows: list[dict[str, Any]],
              match_threshold: float | None, min_margin: float | None) -> dict[str, Any]:
    spkids = spkid_map(spkid_rows)
    clips: list[dict[str, Any]] = []
    source_total = source_correct = source_wrong = source_abstain = 0
    timeline_total = timeline_recovered = 0
    per_speaker: dict[str, Counter[str]] = defaultdict(Counter)

    for row in asr_rows:
        clip = str(row.get("clip", ""))
        source_evals: list[dict[str, Any]] = []
        recovered_gt: set[int] = set()
        gt_rows = row.get("gt_rows", [])
        gt_ids = {int(gt.get("idx", -1)) for gt in gt_rows if int(gt.get("idx", -1)) >= 0}
        timeline_total += len(gt_ids)
        for stream in ["src1", "src2"]:
            asr_info = row.get("streams", {}).get(stream)
            if not asr_info:
                continue
            source_eval = evaluate_source(
                clip, stream, asr_info, spkids.get((clip, stream)), match_threshold, min_margin
            )
            source_evals.append(source_eval)
            if source_eval["has_asr_match"]:
                source_total += 1
                speaker = source_eval["target_speaker"]
                per_speaker[speaker]["total"] += 1
                if source_eval["correct"]:
                    source_correct += 1
                    per_speaker[speaker]["correct"] += 1
                    if source_eval["target_gt_idx"] >= 0:
                        recovered_gt.add(source_eval["target_gt_idx"])
                elif source_eval["wrong"]:
                    source_wrong += 1
                    per_speaker[speaker]["wrong"] += 1
                elif source_eval["abstain"]:
                    source_abstain += 1
                    per_speaker[speaker]["abstain"] += 1
        timeline_recovered += len(recovered_gt & gt_ids)
        correct_speakers = sorted({item["target_speaker"] for item in source_evals if item["correct"]})
        clips.append({
            "clip": clip,
            "rank": row.get("rank"),
            "gt_speakers": row.get("gt_speakers", []),
            "source_evals": source_evals,
            "correct_speakers": correct_speakers,
            "identity_two_speaker_ok": len(correct_speakers) >= 2,
            "identity_one_source_ok": bool(correct_speakers),
            "timeline_gt_total": len(gt_ids),
            "timeline_gt_recovered": len(recovered_gt & gt_ids),
        })

    decided = source_correct + source_wrong
    summary = {
        "clips": len(clips),
        "identity_two_speaker_clips": sum(1 for item in clips if item["identity_two_speaker_ok"]),
        "identity_one_source_clips": sum(1 for item in clips if item["identity_one_source_ok"]),
        "source_eval_total": source_total,
        "source_correct": source_correct,
        "source_wrong": source_wrong,
        "source_abstain": source_abstain,
        "source_decided_accuracy": round(source_correct / decided, 3) if decided else 0.0,
        "source_total_accuracy": round(source_correct / source_total, 3) if source_total else 0.0,
        "source_decided_coverage": round(decided / source_total, 3) if source_total else 0.0,
        "timeline_gt_total": timeline_total,
        "timeline_gt_recovered": timeline_recovered,
        "timeline_recall": round(timeline_recovered / timeline_total, 3) if timeline_total else 0.0,
        "per_speaker": {
            speaker: {
                "total": counts["total"],
                "correct": counts["correct"],
                "wrong": counts["wrong"],
                "abstain": counts["abstain"],
                "decided_accuracy": round(counts["correct"] / max(1, counts["correct"] + counts["wrong"]), 3),
            }
            for speaker, counts in sorted(per_speaker.items())
        },
        "speaker_acceptance": {
            "match_threshold": match_threshold,
            "min_margin": min_margin,
        },
    }
    return {"summary": summary, "clips": clips}


def write_markdown(path: Path, report: dict[str, Any], asr_path: Path, spkid_path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Separatio Identity Summary",
        "",
        "This metric requires both separated-source ASR and separated-source speaker ID to agree with the same GT timeline row.",
        "",
        "## Inputs",
        "",
        f"- asr_compare: `{asr_path}`",
        f"- spkid_jsonl: `{spkid_path}`",
        "",
        "## Summary",
        "",
        f"- clips with two correctly identified GT speakers: {summary['identity_two_speaker_clips']} / {summary['clips']}",
        f"- clips with at least one correctly identified GT speaker: {summary['identity_one_source_clips']} / {summary['clips']}",
        f"- source decided accuracy: {summary['source_decided_accuracy']:.3f}",
        f"- source total accuracy: {summary['source_total_accuracy']:.3f}",
        f"- source decided coverage: {summary['source_decided_coverage']:.3f}",
        f"- GT timeline recall: {summary['timeline_recall']:.3f} ({summary['timeline_gt_recovered']}/{summary['timeline_gt_total']})",
        "",
        "## Per Speaker",
        "",
        "| Speaker | Total | Correct | Wrong | Abstain | Decided Acc |",
        "|---------|------:|--------:|------:|--------:|------------:|",
    ]
    for speaker, item in summary["per_speaker"].items():
        lines.append(
            f"| {speaker} | {item['total']} | {item['correct']} | {item['wrong']} | "
            f"{item['abstain']} | {item['decided_accuracy']:.3f} |"
        )
    lines.extend(["", "## Per Clip", ""])
    for clip in report["clips"]:
        lines.append(
            f"- rank {int(clip.get('rank') or 0):02d}: two={clip['identity_two_speaker_ok']} "
            f"one={clip['identity_one_source_ok']} recovered={clip['timeline_gt_recovered']}/{clip['timeline_gt_total']} "
            f"speakers={','.join(clip['correct_speakers']) or '?'}"
        )
        for source in clip["source_evals"]:
            if not source["has_asr_match"]:
                continue
            mark = "ok" if source["correct"] else ("wrong" if source["wrong"] else "abstain")
            lines.append(
                f"  - {source['stream']} {mark}: pred={source['pred_speaker']} "
                f"gt={source['target_speaker']} asr={source['asr_score']:.3f} "
                f"sim={source['similarity']:.3f} margin={source['margin']:.3f} "
                f"text={compact(source['asr_text'])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize separated-source ASR + speaker ID against GT timeline rows.")
    parser.add_argument("--asr-compare", required=True)
    parser.add_argument("--spkid-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--match-threshold", type=float, default=None)
    parser.add_argument("--min-margin", type=float, default=None)
    args = parser.parse_args()

    asr_path = Path(args.asr_compare)
    spkid_path = Path(args.spkid_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = summarize(load_json(asr_path), read_jsonl(spkid_path), args.match_threshold, args.min_margin)
    json_path = out_dir / "identity_gt_compare.json"
    md_path = out_dir / "identity_gt_compare.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, report, asr_path, spkid_path)
    summary = report["summary"]
    print(
        f"[identity] clips={summary['clips']} two={summary['identity_two_speaker_clips']} "
        f"one={summary['identity_one_source_clips']} decided_acc={summary['source_decided_accuracy']:.3f} "
        f"coverage={summary['source_decided_coverage']:.3f} timeline={summary['timeline_recall']:.3f}"
    )
    print(f"[out] {json_path}")
    print(f"[out] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())