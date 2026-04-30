#!/usr/bin/env python3
"""Prepare and review runtime Auditus fusion-shadow batch captures."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPEAKER_ORDER = ["朱杰", "徐子景", "石一", "唐云峰"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def speaker_name(value: Any) -> str:
    return str(value or "").strip()


def int_field(data: dict[str, Any], key: str, default: int = -1) -> int:
    value = data.get(key, default)
    if value is None:
        return default
    return int(value)


def overlap_sec(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-hide_banner", "-loglevel", "error",
        "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
        str(path),
    ], text=True)
    return float(out.strip())


def extract_wav(source: Path, start_sec: float, end_sec: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
        "-i", str(source), "-ar", "16000", "-ac", "1", str(out_path),
    ])


def normalize_wav(source: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-ar", "16000", "-ac", "1", str(out_path),
    ])


def make_silence(out_path: Path, seconds: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", f"{seconds:.3f}",
        "-i", "anullsrc=r=16000:cl=mono", str(out_path),
    ])


def concat_wavs(paths: list[Path], out_path: Path) -> None:
    concat_path = out_path.with_suffix(".concat.txt")
    concat_path.write_text("".join(f"file '{path.resolve()}'\n" for path in paths), encoding="utf-8")
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-ar", "16000", "-ac", "1", str(out_path),
    ])


def gt_has_other_overlap(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    start = float(row["start_ms"]) / 1000.0
    end = float(row["end_ms"]) / 1000.0
    speaker = speaker_name(row.get("speaker"))
    for other in rows:
        if int(other.get("idx", -1)) == int(row.get("idx", -2)):
            continue
        if speaker_name(other.get("speaker")) == speaker:
            continue
        other_start = float(other["start_ms"]) / 1000.0
        other_end = float(other["end_ms"]) / 1000.0
        if overlap_sec(start, end, other_start, other_end) >= 0.05:
            return True
    return False


def row_overlaps_windows(row: dict[str, Any], windows: list[tuple[float, float]]) -> bool:
    start = float(row["start_ms"]) / 1000.0
    end = float(row["end_ms"]) / 1000.0
    return any(overlap_sec(start, end, left, right) >= 0.05 for left, right in windows)


def select_refs(gt_rows: list[dict[str, Any]], windows: list[tuple[float, float]],
                per_speaker: int, min_sec: float, max_sec: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gt_rows:
        speaker = speaker_name(row.get("speaker"))
        dur = float(row.get("duration_ms", 0.0)) / 1000.0
        if speaker not in SPEAKER_ORDER or dur < min_sec:
            continue
        if gt_has_other_overlap(row, gt_rows) or row_overlaps_windows(row, windows):
            continue
        by_speaker[speaker].append(row)
    for speaker in SPEAKER_ORDER:
        rows = sorted(by_speaker.get(speaker, []), key=lambda item: (-float(item.get("duration_ms", 0.0)), int(item["start_ms"])))
        for row in rows[:per_speaker]:
            start = float(row["start_ms"]) / 1000.0
            end = float(row["end_ms"]) / 1000.0
            if end - start > max_sec:
                center = (start + end) * 0.5
                start = max(0.0, center - max_sec * 0.5)
                end = start + max_sec
            copied = dict(row)
            copied["speaker"] = speaker
            copied["extract_start_sec"] = round(start, 3)
            copied["extract_end_sec"] = round(end, 3)
            selected.append(copied)
    return selected


def clip_gt_rows(clip: dict[str, Any]) -> list[dict[str, Any]]:
    rows = clip.get("candidate", {}).get("gt_rows", [])
    out = []
    for row in rows:
        copied = dict(row)
        copied["speaker"] = speaker_name(copied.get("speaker"))
        out.append(copied)
    return out


def prepare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    work_dir = out_dir / "work"
    gt_rows = read_jsonl(Path(args.gt))
    clips = read_jsonl(Path(args.manifest))[: args.clip_limit]
    windows = [(float(row["clip_start_sec"]), float(row["clip_end_sec"])) for row in clips]
    refs = select_refs(gt_rows, windows, args.ref_per_speaker, args.ref_min_sec, args.ref_max_sec)

    silence = work_dir / "silence_gap.wav"
    tail = work_dir / "tail_silence.wav"
    make_silence(silence, args.gap_sec)
    make_silence(tail, args.tail_sec)

    concat_paths: list[Path] = []
    manifest_items: list[dict[str, Any]] = []
    cursor = 0.0

    def add_item(item: dict[str, Any], path: Path) -> None:
        nonlocal cursor
        item_duration = duration(path)
        item["batch_start_sec"] = round(cursor, 3)
        item["batch_end_sec"] = round(cursor + item_duration, 3)
        item["duration_sec"] = round(item_duration, 3)
        manifest_items.append(item)
        concat_paths.append(path)
        cursor += item_duration
        concat_paths.append(silence)
        cursor += args.gap_sec

    ref_index = 0
    for row in refs:
        speaker = speaker_name(row["speaker"])
        ref_index += 1
        out_path = work_dir / f"ref_{ref_index:02d}_{speaker}.wav"
        extract_wav(Path(args.source_audio), row["extract_start_sec"], row["extract_end_sec"], out_path)
        for repeat in range(args.ref_repeat):
            add_item({
                "kind": "ref",
                "speaker": speaker,
                "ref_repeat": repeat + 1,
                "gt_idx": int(row["idx"]),
                "source_start_sec": row["extract_start_sec"],
                "source_end_sec": row["extract_end_sec"],
                "file": str(out_path),
            }, out_path)

    for clip in clips:
        source_path = Path(args.clips_dir) / str(clip["clip_name"])
        out_path = work_dir / str(clip["clip_name"])
        normalize_wav(source_path, out_path)
        add_item({
            "kind": "clip",
            "rank": int(clip["rank"]),
            "clip_name": clip["clip_name"],
            "source_start_sec": float(clip["clip_start_sec"]),
            "source_end_sec": float(clip["clip_end_sec"]),
            "gt_speakers": sorted({speaker_name(row.get("speaker")) for row in clip_gt_rows(clip)}),
            "gt_rows": clip_gt_rows(clip),
            "file": str(out_path),
        }, out_path)

    concat_paths.append(tail)
    cursor += args.tail_sec
    out_wav = out_dir / "runtime_shadow_batch.wav"
    concat_wavs(concat_paths, out_wav)
    write_json(out_dir / "batch_manifest.json", {
        "audio": str(out_wav),
        "duration_sec": round(duration(out_wav), 3),
        "gap_sec": args.gap_sec,
        "tail_sec": args.tail_sec,
        "items": manifest_items,
    })
    print(f"[prepare] audio={out_wav} duration={duration(out_wav):.1f}s items={len(manifest_items)} refs={len(refs)}x{args.ref_repeat} clips={len(clips)}")
    return 0


def best_item(items: list[dict[str, Any]], start: float, end: float) -> dict[str, Any] | None:
    best = None
    best_overlap = 0.0
    for item in items:
        ov = overlap_sec(start, end, float(item["batch_start_sec"]), float(item["batch_end_sec"]))
        if ov > best_overlap:
            best = item
            best_overlap = ov
    if best and best_overlap >= 0.05:
        copied = dict(best)
        copied["overlap_sec"] = round(best_overlap, 3)
        return copied
    return None


def nearest_item_at_time(items: list[dict[str, Any]], time_sec: float, tolerance_sec: float) -> dict[str, Any] | None:
    best = None
    best_distance = tolerance_sec
    for item in items:
        start = float(item["batch_start_sec"])
        end = float(item["batch_end_sec"])
        if start <= time_sec <= end:
            copied = dict(item)
            copied["distance_sec"] = 0.0
            return copied
        distance = min(abs(time_sec - start), abs(time_sec - end))
        if distance <= best_distance:
            best = item
            best_distance = distance
    if not best:
        return None
    copied = dict(best)
    copied["distance_sec"] = round(best_distance, 3)
    return copied


def event_audio_time(rows: list[dict[str, Any]], index: int) -> float | None:
    for right in range(index, min(len(rows), index + 8)):
        row = rows[right]
        if row.get("type") == "pipeline_stats" and row.get("audio_t1") is not None:
            return float(row["audio_t1"]) / 16000.0
    for left in range(index - 1, max(-1, index - 8), -1):
        row = rows[left]
        if row.get("type") == "pipeline_stats" and row.get("audio_t1") is not None:
            return float(row["audio_t1"]) / 16000.0
    return None


def collect_birth_votes(rows: list[dict[str, Any]], items: list[dict[str, Any]]) -> tuple[dict[int, dict[str, float]], list[dict[str, Any]]]:
    votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    births: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("type") != "speaker" or not row.get("new"):
            continue
        event_time = event_audio_time(rows, index)
        if event_time is None:
            continue
        item = nearest_item_at_time(items, event_time, 1.25)
        birth = {
            "speaker_id": int_field(row, "id"),
            "sim": float(row.get("sim", 0.0) or 0.0),
            "event_time_sec": round(event_time, 3),
            "item_kind": item.get("kind") if item else "",
            "speaker": item.get("speaker", "") if item else "",
        }
        births.append(birth)
        if item and item.get("kind") == "ref" and birth["speaker_id"] >= 0:
            votes[birth["speaker_id"]][str(item["speaker"])] += 10.0
    return votes, births


def collect_ref_source_votes(rows: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        start = float(row.get("stream_start_sec", 0.0))
        end = float(row.get("stream_end_sec", 0.0))
        item = best_item(items, start, end)
        if not item or item.get("kind") != "ref":
            continue
        speaker = str(item["speaker"])
        if row.get("type") == "asr_log" and row.get("stage") == "fusion_shadow":
            for source in ("src1", "src2"):
                spk = row.get(source, {}).get("speaker", {})
                speaker_id = int_field(spk, "speaker_id")
                if spk.get("accepted") and speaker_id >= 0:
                    votes[speaker_id][speaker] += max(0.5, float(spk.get("similarity", 0.0)))
    return votes


def resolve_id_map(votes: dict[int, dict[str, float]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for speaker_id, by_speaker in votes.items():
        if not by_speaker:
            continue
        mapping[speaker_id] = max(by_speaker.items(), key=lambda item: (item[1], item[0]))[0]
    return mapping


def summarize(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.batch_manifest).read_text(encoding="utf-8"))
    items = list(manifest["items"])
    raw_rows = read_jsonl(Path(args.raw_events))
    fusion = [row for row in raw_rows if row.get("type") == "asr_log" and row.get("stage") == "fusion_shadow"]
    transcripts = [row for row in raw_rows if row.get("type") == "asr_transcript"]
    id_votes, id_births = collect_birth_votes(raw_rows, items)
    ref_source_votes = collect_ref_source_votes(fusion, items)
    if not id_votes:
        id_votes = ref_source_votes
    id_map = resolve_id_map(id_votes)

    clip_rows: list[dict[str, Any]] = []
    source_total = source_accepted = source_mapped_hit = source_mapped_wrong = source_unmapped = 0
    source_stable = source_stable_mapped_hit = source_stable_mapped_wrong = source_stable_unmapped = 0
    arbitrium_actions: Counter[str] = Counter()
    arbitrium_reasons: Counter[str] = Counter()
    arbitrium_contradictions = 0
    ledger_blockers: Counter[str] = Counter()
    ledger_missing = 0
    ledger_canary_candidates = 0
    ledger_canary_would_apply = 0
    ledger_authority_violations = 0
    for event in fusion:
        arbitrium = event.get("arbitrium", {})
        action = str(arbitrium.get("action", "<missing>") or "<missing>")
        reason = str(arbitrium.get("reason", "<missing>") or "<missing>")
        arbitrium_actions[action] += 1
        arbitrium_reasons[reason] += 1
        arbitrium_contradictions += int(bool(arbitrium.get("contradiction")))
        ledger = event.get("ledger", {})
        if not ledger:
            ledger_missing += 1
        else:
            blocker = str(ledger.get("canary_blocker", "<missing>") or "<missing>")
            ledger_blockers[blocker] += 1
            ledger_canary_candidates += int(bool(ledger.get("canary_candidate")))
            ledger_canary_would_apply += int(bool(ledger.get("canary_would_apply")))
            if ledger.get("authority") != "shadow" or not bool(ledger.get("shadow_only")):
                ledger_authority_violations += 1
    for index, event in enumerate(fusion, 1):
        item = best_item(items, float(event.get("stream_start_sec", 0.0)), float(event.get("stream_end_sec", 0.0)))
        if not item or item.get("kind") != "clip":
            continue
        sources = []
        gt_speakers = set(item.get("gt_speakers", []))
        for source_name in ("src1", "src2"):
            source = event.get(source_name, {})
            speaker = source.get("speaker", {})
            accepted = bool(speaker.get("accepted"))
            stable = bool(speaker.get("stable"))
            speaker_id = int_field(speaker, "speaker_id")
            mapped = id_map.get(speaker_id, "") if speaker_id >= 0 else ""
            status = "abstain"
            if accepted:
                source_accepted += 1
                if not mapped:
                    source_unmapped += 1
                    status = "accepted_unmapped"
                elif mapped in gt_speakers:
                    source_mapped_hit += 1
                    status = "mapped_gt_hit"
                else:
                    source_mapped_wrong += 1
                    status = "mapped_gt_wrong"
            if stable:
                source_stable += 1
                if not mapped:
                    source_stable_unmapped += 1
                elif mapped in gt_speakers:
                    source_stable_mapped_hit += 1
                else:
                    source_stable_mapped_wrong += 1
            source_total += 1
            sources.append({
                "source": source_name,
                "text": source.get("text", ""),
                "text_nonempty": bool(source.get("text_nonempty")),
                "accepted": accepted,
                "stable": stable,
                "speaker_id": speaker_id,
                "mapped_speaker": mapped,
                "exemplar_count": int_field(speaker, "exemplar_count", 0),
                "match_count": int_field(speaker, "match_count", 0),
                "similarity": float(speaker.get("similarity", 0.0) or 0.0),
                "margin": float(speaker.get("margin", 0.0) or 0.0),
                "reason": speaker.get("reason", ""),
                "status": status,
            })
        clip_rows.append({
            "event_index": index,
            "stream_start_sec": event.get("stream_start_sec"),
            "stream_end_sec": event.get("stream_end_sec"),
            "clip_name": item.get("clip_name"),
            "rank": item.get("rank"),
            "gt_speakers": sorted(gt_speakers),
            "mix_text": event.get("mix_text", ""),
            "timeline_speaker_id": event.get("timeline_speaker_id"),
            "timeline_mapped_speaker": id_map.get(int_field(event, "timeline_speaker_id"), ""),
            "arbitrium": event.get("arbitrium", {}),
            "ledger": event.get("ledger", {}),
            "sources": sources,
        })

    summary = {
        "raw_rows": len(raw_rows),
        "transcripts": len(transcripts),
        "fusion_shadow_events": len(fusion),
        "clip_fusion_events": len(clip_rows),
        "id_births": id_births,
        "ref_source_votes": {str(k): dict(v) for k, v in sorted(ref_source_votes.items())},
        "id_votes": {str(k): dict(v) for k, v in sorted(id_votes.items())},
        "id_map": {str(k): v for k, v in sorted(id_map.items())},
        "source_total": source_total,
        "source_accepted": source_accepted,
        "source_abstain": source_total - source_accepted,
        "source_mapped_gt_hit": source_mapped_hit,
        "source_mapped_gt_wrong": source_mapped_wrong,
        "source_accepted_unmapped": source_unmapped,
        "source_stable": source_stable,
        "source_stable_abstain": source_total - source_stable,
        "source_stable_mapped_gt_hit": source_stable_mapped_hit,
        "source_stable_mapped_gt_wrong": source_stable_mapped_wrong,
        "source_stable_unmapped": source_stable_unmapped,
        "arbitrium_actions": dict(sorted(arbitrium_actions.items())),
        "arbitrium_reasons": dict(sorted(arbitrium_reasons.items())),
        "arbitrium_contradictions": arbitrium_contradictions,
        "ledger_missing": ledger_missing,
        "ledger_canary_candidates": ledger_canary_candidates,
        "ledger_canary_would_apply": ledger_canary_would_apply,
        "ledger_blockers": dict(sorted(ledger_blockers.items())),
        "ledger_authority_violations": ledger_authority_violations,
    }
    report = {"summary": summary, "clips": clip_rows}
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    write_json(out_json, report)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"[summary] {summary}")
    print(f"[out] {out_json}")
    print(f"[out] {out_md}")
    return 0


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Runtime Fusion Shadow Batch Review",
        "",
        "This is telemetry for the shadow path, not the final speaker-attribution score.",
        "",
        "## Summary",
        "",
        f"- raw rows: {summary['raw_rows']}",
        f"- transcripts: {summary['transcripts']}",
        f"- fusion shadow events: {summary['fusion_shadow_events']}",
        f"- clip fusion events: {summary['clip_fusion_events']}",
        f"- id map: {summary['id_map']}",
        f"- sources: total={summary['source_total']} accepted={summary['source_accepted']} "
        f"abstain={summary['source_abstain']} mapped_hit={summary['source_mapped_gt_hit']} "
        f"mapped_wrong={summary['source_mapped_gt_wrong']} unmapped={summary['source_accepted_unmapped']}",
        f"- stable sources: stable={summary['source_stable']} "
        f"abstain={summary['source_stable_abstain']} "
        f"mapped_hit={summary['source_stable_mapped_gt_hit']} "
        f"mapped_wrong={summary['source_stable_mapped_gt_wrong']} "
        f"unmapped={summary['source_stable_unmapped']}",
        f"- arbitrium actions: {summary['arbitrium_actions']}",
        f"- arbitrium reasons: {summary['arbitrium_reasons']}",
        f"- arbitrium contradictions: {summary['arbitrium_contradictions']}",
        f"- ledger: missing={summary['ledger_missing']} "
        f"canary_candidates={summary['ledger_canary_candidates']} "
        f"canary_would_apply={summary['ledger_canary_would_apply']} "
        f"authority_violations={summary['ledger_authority_violations']} "
        f"blockers={summary['ledger_blockers']}",
        "",
        "## Events",
        "",
    ]
    for row in report["clips"]:
        lines.append(f"### Event {row['event_index']} — {row['clip_name']} (rank {row['rank']})")
        lines.append(f"- GT speakers: {', '.join(row['gt_speakers']) or '?'}")
        lines.append(f"- mix: {row['mix_text']}")
        lines.append(f"- timeline: id={row['timeline_speaker_id']} mapped={row['timeline_mapped_speaker'] or '?'}")
        arbitrium = row.get("arbitrium", {})
        lines.append(
            f"- arbitrium: action={arbitrium.get('action', '?')} "
            f"reason={arbitrium.get('reason', '?')} "
            f"stable_sources={arbitrium.get('stable_text_sources', '?')} "
            f"contradiction={arbitrium.get('contradiction', False)}"
        )
        ledger = row.get("ledger", {})
        lines.append(
            f"- ledger: canary={ledger.get('canary_candidate', False)} "
            f"would_apply={ledger.get('canary_would_apply', False)} "
            f"blocker={ledger.get('canary_blocker', '?')} "
            f"stable_ids={ledger.get('stable_speaker_ids', [])}"
        )
        for source in row["sources"]:
            lines.append(
                f"- {source['source']}: status={source['status']} stable={source['stable']} "
                f"id={source['speaker_id']} "
                f"mapped={source['mapped_speaker'] or '?'} sim={source['similarity']:.3f} "
                f"margin={source['margin']:.3f} ex={source['exemplar_count']} "
                f"match={source['match_count']} reason={source['reason']} text={source['text']}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", default="logs/fusion_shadow_batch_r1")
    prep.add_argument("--manifest", default="logs/segment_homogeneity_clips_r3/clip_manifest.jsonl")
    prep.add_argument("--clips-dir", default="logs/segment_homogeneity_clips_r3/clips")
    prep.add_argument("--gt", default="tests/fixtures/test_ground_truth_v1.jsonl")
    prep.add_argument("--source-audio", default="tests/test.mp3")
    prep.add_argument("--clip-limit", type=int, default=12)
    prep.add_argument("--ref-per-speaker", type=int, default=2)
    prep.add_argument("--ref-repeat", type=int, default=1)
    prep.add_argument("--ref-min-sec", type=float, default=2.0)
    prep.add_argument("--ref-max-sec", type=float, default=4.0)
    prep.add_argument("--gap-sec", type=float, default=1.0)
    prep.add_argument("--tail-sec", type=float, default=3.0)
    prep.set_defaults(func=prepare)

    summ = sub.add_parser("summarize")
    summ.add_argument("--batch-manifest", default="logs/fusion_shadow_batch_r1/batch_manifest.json")
    summ.add_argument("--raw-events", default="logs/fusion_shadow_batch_r1/live_capture/raw_events.jsonl")
    summ.add_argument("--out-json", default="logs/fusion_shadow_batch_r1/runtime_shadow_batch_report.json")
    summ.add_argument("--out-md", default="logs/fusion_shadow_batch_r1/runtime_shadow_batch_report.md")
    summ.set_defaults(func=summarize)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
