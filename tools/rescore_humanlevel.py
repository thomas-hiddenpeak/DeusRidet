#!/usr/bin/env python3
"""
rescore_humanlevel.py — Re-score a completed replay run against the
human transcript tests/test.txt (utterance-level, 1 s precision),
instead of the derived VAD-level GT tests/fixtures/test_ground_truth_v1.jsonl.

Rationale:
    The derived GT fragments each utterance into 1-4 sub-segments so a
    "no_segment" on a 300 ms VAD crumb counts the same as missing a
    full sentence. The raw transcript carries the human intuition:
    "this is one thing a person said." Scoring at that granularity
    answers "did the system correctly label this speaking turn?"
    instead of "did it label every VAD crumb?".

Method (parallels online_replay_score.py --- same first-seen mapping,
same macro/decided_macro accounting, different GT unit):
    1. Parse test.txt into (start_sec, speaker) utterances.
    2. For utterance u, define window = [u.start, next.start - eps]
       (or +max_window_s if it's the last one, default 30 s).
    3. Collect every speaker_events entry whose t_close_sec falls in
       the window. Aggregate cluster IDs by majority vote (abstain
       votes with id<0 are dropped; if no valid vote, the utterance
       is "abstain"; if no events fall in the window at all,
       "no_segment").
    4. First-seen mapping cluster -> name in utterance order.
    5. Report coverage, macro, micro, decided_macro, decided_micro,
       per-speaker breakdown, cluster count.

Usage:
    python3 tools/rescore_humanlevel.py \
        --transcript tests/test.txt \
        --events runs/online_step4c_v4_600s/speaker_events.jsonl \
        --max-sec 600 \
        --out runs/online_step4c_v4_600s/humanlevel_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\s+(\S+)\s*$")


def parse_transcript(path: Path, max_sec: float):
    """Return list of {idx, start_sec, end_sec, speaker} utterances."""
    utts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = TS_RE.match(line)
        if not m:
            continue
        hh, mm, ss, spk = m.groups()
        t = int(hh) * 3600 + int(mm) * 60 + int(ss)
        utts.append({"start_sec": float(t), "speaker": spk})
    # Clip to first max_sec seconds of source audio.
    utts = [u for u in utts if u["start_sec"] <= max_sec]
    # Assign end_sec = next start (capped at max_sec) or start+30 s for tail.
    for i, u in enumerate(utts):
        if i + 1 < len(utts):
            u["end_sec"] = min(utts[i + 1]["start_sec"], max_sec)
        else:
            u["end_sec"] = min(u["start_sec"] + 30.0, max_sec)
        u["idx"] = i
    return utts


def load_events(path: Path):
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        events.append(e)
    events.sort(key=lambda e: e.get("t_close_sec", 0.0))
    return events


def score(utts, events):
    """Assign each utterance a cluster via majority vote inside window."""
    # Binary-search-ish linear scan since events sorted.
    ei = 0
    results = []
    for u in utts:
        votes = []   # list of cluster ids falling in window
        while ei < len(events) and events[ei].get("t_close_sec", 0.0) < u["start_sec"]:
            ei += 1
        j = ei
        while j < len(events) and events[j].get("t_close_sec", 0.0) < u["end_sec"]:
            cid = events[j].get("id", -1)
            votes.append(cid)
            j += 1
        if not votes:
            status = "no_segment"
            cid = -1
        else:
            valid = [c for c in votes if c >= 0]
            if not valid:
                status = "abstain"
                cid = -1
            else:
                cid = Counter(valid).most_common(1)[0][0]
                status = "decided"
        results.append({
            "gt_idx":     u["idx"],
            "gt_start":   u["start_sec"],
            "gt_end":     u["end_sec"],
            "gt_speaker": u["speaker"],
            "rt_cluster": cid,
            "status":     status,
            "n_events":   len(votes),
        })
    return results


def compute(matched):
    # First-seen mapping cluster -> GT speaker name, utterance order.
    mapping = {}
    for m in matched:
        c = m["rt_cluster"]
        if c < 0:
            continue
        mapping.setdefault(c, m["gt_speaker"])

    speakers = sorted({m["gt_speaker"] for m in matched})
    per_total = {s: 0 for s in speakers}
    per_decided = {s: 0 for s in speakers}
    per_correct = {s: 0 for s in speakers}
    per_decided_correct = {s: 0 for s in speakers}
    n_no_seg = n_abstain = 0
    for m in matched:
        s = m["gt_speaker"]
        per_total[s] += 1
        if m["status"] == "no_segment":
            n_no_seg += 1
        elif m["status"] == "abstain":
            n_abstain += 1
        else:
            per_decided[s] += 1
            pred = mapping.get(m["rt_cluster"], "__unk__")
            if pred == s:
                per_correct[s] += 1
                per_decided_correct[s] += 1
    per_spk = {
        s: per_correct[s] / per_total[s] if per_total[s] else 0.0
        for s in speakers
    }
    per_spk_decided = {
        s: (per_decided_correct[s] / per_decided[s])
            if per_decided[s] else 0.0
        for s in speakers
    }
    n_gt = len(matched)
    n_decided = sum(per_decided.values())
    macro = sum(per_spk.values()) / max(1, len(per_spk))
    micro = sum(per_correct.values()) / max(1, n_gt)
    decided_macro = (
        sum(per_spk_decided.values()) / max(1, len(per_spk_decided))
    )
    decided_micro = (
        sum(per_decided_correct.values()) / max(1, n_decided)
    )
    return {
        "method": "humanlevel_transcript_first_seen",
        "n_gt": n_gt,
        "n_decided": n_decided,
        "n_abstain": n_abstain,
        "n_no_seg": n_no_seg,
        "coverage": round(n_decided / max(1, n_gt), 4),
        "macro": round(macro, 4),
        "micro": round(micro, 4),
        "decided_macro": round(decided_macro, 4),
        "decided_micro": round(decided_micro, 4),
        "per_spk": {s: round(v, 4) for s, v in per_spk.items()},
        "per_spk_decided": {s: round(v, 4) for s, v in per_spk_decided.items()},
        "per_spk_total": per_total,
        "per_spk_decided_count": per_decided,
        "mapping": {str(k): v for k, v in mapping.items()},
        "n_clusters": len(mapping),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--events",     required=True)
    ap.add_argument("--max-sec",    type=float, default=600.0)
    ap.add_argument("--out",        default=None)
    args = ap.parse_args()

    utts = parse_transcript(Path(args.transcript), args.max_sec)
    events = load_events(Path(args.events))
    matched = score(utts, events)
    summary = compute(matched)
    summary["args"] = {
        "transcript": args.transcript,
        "events":     args.events,
        "max_sec":    args.max_sec,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Also dump per-utterance matched file next to it.
        mpath = outp.with_suffix(".matched.jsonl")
        with mpath.open("w", encoding="utf-8") as fh:
            for m in matched:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"\n[out] {outp}\n[out] {mpath}")


if __name__ == "__main__":
    main()
