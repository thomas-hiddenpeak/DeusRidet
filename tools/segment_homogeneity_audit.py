#!/usr/bin/env python3
"""
segment_homogeneity_audit.py — Step 18a candidate generator.

This is a triage/export tool, not an accuracy scorer. It ranks VAD segments
that are likely not speaker-homogeneous by comparing runtime VAD boundaries
against GT time spans, ASR transcript spans, and speaker broadcasts captured by
online_replay_score.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

STANCE_TERMS = [
    "可以", "不行", "不对", "不是", "同意", "不同意", "接受", "拒绝",
    "别", "不要", "就这样", "这么定", "不能", "要", "必须", "我来",
]
FILLER_TERMS = ["嗯", "哦", "啊", "呃", "额"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_coarse_gt(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(u["idx"]): u for u in data.get("utterances", [])}


def parse_timeline_vad_segments(path: Path, sample_rate: int = 16000) -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    open_start: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("t") != "vad":
            continue
        event = obj.get("event")
        audio_t1 = int(obj.get("audio_t1", 0))
        if event == "start":
            open_start = audio_t1
        elif event == "end" and open_start is not None:
            idx = len(segs)
            segs.append({
                "vad_idx": idx,
                "start_sec": open_start / float(sample_rate),
                "end_sec": audio_t1 / float(sample_rate),
            })
            open_start = None
    return segs


def overlap_sec(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def row_start(row: dict[str, Any]) -> float:
    if "start_sec" in row:
        return float(row["start_sec"])
    return float(row.get("start_ms", 0)) / 1000.0


def row_end(row: dict[str, Any]) -> float:
    if "end_sec" in row:
        return float(row["end_sec"])
    return float(row.get("end_ms", 0)) / 1000.0


def text_flags(text: str) -> tuple[bool, bool, list[str]]:
    stance_hits = [t for t in STANCE_TERMS if t in text]
    has_stance = bool(stance_hits)
    stripped = text.strip(" ，。！？,.!?\t\n")
    filler_only = bool(stripped) and all(ch in "嗯哦啊呃额" for ch in stripped)
    if any(t in text for t in FILLER_TERMS) and len(stripped) <= 2:
        filler_only = True
    return has_stance, filler_only, stance_hits


def overlapping_rows(start: float, end: float, rows: list[dict[str, Any]], min_ov: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        ov = overlap_sec(start, end, row_start(row), row_end(row))
        if ov >= min_ov:
            copied = dict(row)
            copied["overlap_sec"] = round(ov, 3)
            out.append(copied)
    return out


def nearest_speaker_events(start: float, end: float, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    center = (start + end) * 0.5
    nearby = []
    for ev in events:
        close = float(ev.get("t_close_sec", 0.0))
        if start - 1.0 <= close <= end + 1.0:
            copied = dict(ev)
            copied["distance_to_center"] = round(abs(close - center), 3)
            nearby.append(copied)
    nearby.sort(key=lambda e: (e.get("distance_to_center", 0.0), e.get("order", 0)))
    return nearby[:5]


def load_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    vad = parse_timeline_vad_segments(Path(args.timeline))
    gt_v1 = read_jsonl(Path(args.gt_v1))
    coarse = read_coarse_gt(Path(args.gt_coarse))
    replay_dir = Path(args.replay_dir)
    asr = read_jsonl(replay_dir / "asr_transcripts.jsonl")
    speakers = read_jsonl(replay_dir / "speaker_events.jsonl")
    if args.max_sec > 0:
        vad = [v for v in vad if v["end_sec"] <= args.max_sec + 2.0]
        gt_v1 = [g for g in gt_v1 if row_start(g) <= args.max_sec + 2.0]
        asr = [a for a in asr if row_start(a) <= args.max_sec + 2.0]
        speakers = [s for s in speakers if float(s.get("t_close_sec", 0.0)) <= args.max_sec + 2.0]
    return vad, gt_v1, asr, speakers, coarse


def build_candidate(vad: dict[str, Any], gt_rows: list[dict[str, Any]], asr_rows: list[dict[str, Any]], speaker_events: list[dict[str, Any]], coarse: dict[int, dict[str, Any]]) -> dict[str, Any]:
    start = float(vad["start_sec"])
    end = float(vad["end_sec"])
    duration = max(0.0, end - start)
    gt_ov = overlapping_rows(start, end, gt_rows, 0.08)
    asr_ov = overlapping_rows(start, end, asr_rows, 0.08)
    spk_near = nearest_speaker_events(start, end, speaker_events)

    speakers = sorted({g.get("speaker", "") for g in gt_ov if g.get("speaker")})
    src_utts = sorted({int(g.get("src_utt_idx", -1)) for g in gt_ov if int(g.get("src_utt_idx", -1)) >= 0})
    texts = []
    local_stance_hits: list[str] = []
    coarse_stance_hits: list[str] = []
    filler_only_count = 0
    for src_idx in src_utts:
        coarse_row = coarse.get(src_idx, {})
        text = str(coarse_row.get("text", ""))
        if text:
            texts.append(text)
            has_stance, filler_only, hits = text_flags(text)
            if has_stance:
                coarse_start = float(coarse_row.get("t0_start_sec", 0.0))
                coarse_end = float(coarse_row.get("t0_end_sec", coarse_start))
                if coarse_end - coarse_start <= 5.0:
                    local_stance_hits.extend(hits)
                else:
                    coarse_stance_hits.extend(hits)
            if filler_only:
                filler_only_count += 1
    asr_speakers = sorted({int(a.get("speaker_id", -1)) for a in asr_ov})
    asr_unknown = sum(1 for a in asr_ov if int(a.get("speaker_id", -1)) < 0)
    for asr_row in asr_ov:
        has_stance, _, hits = text_flags(str(asr_row.get("text", "")))
        if has_stance:
            local_stance_hits.extend(hits)
    speaker_ids = sorted({int(s.get("id", -1)) for s in spk_near})
    amended = sum(1 for s in spk_near if s.get("amended"))

    score = 0.0
    reasons: list[str] = []
    if len(speakers) >= 2:
        score += 40.0 + 8.0 * (len(speakers) - 2)
        reasons.append(f"multi_gt_speaker:{'/'.join(speakers)}")
    if len(src_utts) >= 2:
        score += min(20.0, 5.0 * (len(src_utts) - 1))
        reasons.append(f"multi_gt_turn:{len(src_utts)}")
    if local_stance_hits and duration <= 3.0:
        score += 18.0
        reasons.append("short_stance:" + "/".join(sorted(set(local_stance_hits))[:4]))
    elif local_stance_hits:
        score += 8.0
        reasons.append("stance:" + "/".join(sorted(set(local_stance_hits))[:4]))
    elif coarse_stance_hits:
        score += 3.0
        reasons.append("coarse_stance:" + "/".join(sorted(set(coarse_stance_hits))[:4]))
    if asr_unknown:
        score += 8.0
        reasons.append(f"asr_unknown:{asr_unknown}")
    if len(asr_speakers) >= 2:
        score += 12.0
        reasons.append(f"asr_speaker_conflict:{asr_speakers}")
    if len(speaker_ids) >= 2:
        score += 10.0
        reasons.append(f"near_speaker_conflict:{speaker_ids}")
    if amended:
        score += 6.0
        reasons.append(f"retro_amended_nearby:{amended}")
    if duration >= 6.0 and len(src_utts) >= 2:
        score += min(10.0, math.log(duration + 1.0) * 3.0)
        reasons.append(f"long_container:{duration:.1f}s")
    if filler_only_count and not local_stance_hits:
        score -= min(8.0, 4.0 * filler_only_count)
        reasons.append("mostly_filler")

    return {
        "vad_idx": vad["vad_idx"],
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "duration_sec": round(duration, 3),
        "risk_score": round(score, 2),
        "reasons": reasons,
        "gt_speakers": speakers,
        "gt_turn_count": len(src_utts),
        "gt_rows": [
            {
                "idx": g.get("idx"),
                "speaker": g.get("speaker"),
                "start_sec": round(row_start(g), 3),
                "end_sec": round(row_end(g), 3),
                "src_utt_idx": g.get("src_utt_idx"),
                "overlap_sec": g.get("overlap_sec"),
                "text": coarse.get(int(g.get("src_utt_idx", -1)), {}).get("text", "")[:120],
            }
            for g in gt_ov
        ],
        "asr": [
            {
                "start_sec": round(row_start(a), 3),
                "end_sec": round(row_end(a), 3),
                "speaker_id": a.get("speaker_id"),
                "speaker_source": a.get("speaker_source"),
                "text": a.get("text", "")[:120],
                "overlap_sec": a.get("overlap_sec"),
            }
            for a in asr_ov
        ],
        "near_speaker_events": [
            {
                "id": s.get("id"),
                "sim": s.get("sim"),
                "t_close_sec": s.get("t_close_sec"),
                "amended": bool(s.get("amended")),
                "distance_to_center": s.get("distance_to_center"),
            }
            for s in spk_near
        ],
    }


def write_markdown(path: Path, candidates: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Segment Homogeneity Audit",
        "",
        "This is a candidate list for human listening review. It is not an accuracy score.",
        "",
        f"- timeline: `{args.timeline}`",
        f"- replay_dir: `{args.replay_dir}`",
        f"- max_sec: `{args.max_sec}`",
        f"- candidates: `{len(candidates)}`",
        "",
    ]
    for c in candidates:
        lines.append(
            f"## #{c['vad_idx']}  {c['start_sec']:.2f}-{c['end_sec']:.2f}s  "
            f"risk={c['risk_score']:.1f}"
        )
        lines.append("")
        lines.append(f"Reasons: {', '.join(c['reasons']) or 'none'}")
        lines.append(f"GT speakers: {', '.join(c['gt_speakers']) or '?'}; GT turns: {c['gt_turn_count']}")
        lines.append("")
        lines.append("GT overlaps:")
        for g in c["gt_rows"][:8]:
            text = str(g.get("text", "")).replace("\n", " ")
            lines.append(
                f"- {g['start_sec']:.2f}-{g['end_sec']:.2f}s {g.get('speaker')} "
                f"ov={g.get('overlap_sec')} text={text}"
            )
        if len(c["gt_rows"]) > 8:
            lines.append(f"- ... {len(c['gt_rows']) - 8} more GT rows")
        lines.append("")
        lines.append("ASR overlaps:")
        for a in c["asr"][:6]:
            text = str(a.get("text", "")).replace("\n", " ")
            lines.append(
                f"- {a['start_sec']:.2f}-{a['end_sec']:.2f}s id={a.get('speaker_id')} "
                f"src={a.get('speaker_source')} ov={a.get('overlap_sec')} text={text}"
            )
        lines.append("")
        lines.append("Nearby speaker events:")
        for s in c["near_speaker_events"]:
            lines.append(
                f"- t={s.get('t_close_sec')} id={s.get('id')} sim={s.get('sim')} "
                f"amended={s.get('amended')} dt={s.get('distance_to_center')}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank VAD segments likely to contain rapid speaker switches or mixed speakers.")
    ap.add_argument("--timeline", required=True, help="TimelineLogger jsonl for the replay run")
    ap.add_argument("--replay-dir", required=True, help="online_replay_score output directory")
    ap.add_argument("--gt-v1", default="tests/fixtures/test_ground_truth_v1.jsonl")
    ap.add_argument("--gt-coarse", default="tests/fixtures/test_ground_truth.json")
    ap.add_argument("--max-sec", type=float, default=600.0)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--out-dir", default="/tmp/segment_homogeneity_audit")
    args = ap.parse_args()

    vad, gt_v1, asr, speakers, coarse = load_inputs(args)
    candidates = [build_candidate(v, gt_v1, asr, speakers, coarse) for v in vad]
    candidates = [c for c in candidates if c["risk_score"] > 0]
    candidates.sort(key=lambda c: (-c["risk_score"], c["start_sec"]))
    top = candidates[: args.top_n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "homogeneity_candidates.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for c in top:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    md_path = out_dir / "homogeneity_candidates.md"
    write_markdown(md_path, top, args)

    print(f"[inputs] vad={len(vad)} gt_v1={len(gt_v1)} asr={len(asr)} speaker_events={len(speakers)}")
    print(f"[rank] candidates={len(candidates)} top={len(top)}")
    print(f"[out] {jsonl_path}")
    print(f"[out] {md_path}")
    for c in top[:10]:
        print(
            f"#{c['vad_idx']:03d} {c['start_sec']:7.2f}-{c['end_sec']:7.2f}s "
            f"risk={c['risk_score']:5.1f} speakers={','.join(c['gt_speakers']) or '?'} "
            f"turns={c['gt_turn_count']} reasons={';'.join(c['reasons'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
