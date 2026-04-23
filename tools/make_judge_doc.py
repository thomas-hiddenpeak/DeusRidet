#!/usr/bin/env python3
"""Produce a T0-aligned side-by-side Markdown judge document from
ground_truth.json + hyp.jsonl (Step 15c).

This is purely a layout tool. It performs NO scoring -- per
benchmarks.instructions.md scoring is done by the agent reading the
document directly. The Markdown is structured so the reader can scan
a single timeline and see, at every instant, what the human annotator
wrote vs what the pipeline said.

Strategy
========
* Build a merged list of events keyed on t0_start_sec:
    - GT rows:  {kind: "gt",  t0_start, t0_end, speaker, text, idx}
    - HYP rows: {kind: "hyp", t0_start, t0_end, speaker, speaker_id,
                 speaker_sim, speaker_confidence, speaker_source,
                 trigger, text}
* Sort ascending by t0_start.
* Emit one Markdown row per event, with a "what it is" column so the
  agent can always tell whether a line came from GT or HYP.
* Also emit a block header every 60 s of source audio so long runs
  stay navigable (anchor links: `## 00:03:00 - 00:04:00`).

Output file: runs/<ts>/judge.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def mmss(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def load_gt(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for u in data["utterances"]:
        out.append(
            {
                "kind":     "gt",
                "t0_start": float(u["t0_start_sec"]),
                "t0_end":   float(u["t0_end_sec"]),
                "speaker":  u["speaker"],
                "text":     u["text"],
                "idx":      int(u["idx"]),
            }
        )
    return out


def load_hyp(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(
            {
                "kind":       "hyp",
                "t0_start":   float(d.get("t0_start_sec", 0.0)),
                "t0_end":     float(d.get("t0_end_sec", 0.0)),
                "speaker":    d.get("speaker_name", "") or f"spk{d.get('speaker_id', -1)}",
                "speaker_id": int(d.get("speaker_id", -1)),
                "sim":        float(d.get("speaker_sim", 0.0)),
                "conf":       float(d.get("speaker_confidence", 0.0)),
                "source":     d.get("speaker_source", ""),
                "trigger":    d.get("trigger", ""),
                "text":       d.get("text", ""),
                "latency_ms": float(d.get("latency_ms", 0.0)),
            }
        )
    return out


def _diarization_summary(gt: list[dict], hyp: list[dict]) -> list[str]:
    """Diarization-focused aggregation. NOT scoring — counts + majority-vote
    id→name mapping so the human-style reader can focus on failure modes
    instead of bookkeeping. The agent still judges correctness by reading
    the timeline below.
    """
    from collections import Counter, defaultdict
    out: list[str] = []
    out.append("## Diarization summary (Step 16)")
    out.append("")
    gt_counts = Counter(r["speaker"] for r in gt)
    out.append(f"- GT utterances: {len(gt)} across {len(gt_counts)} speakers")
    for name, n in gt_counts.most_common():
        out.append(f"    - {name}: {n}")
    hyp_counts = Counter(r["speaker_id"] for r in hyp)
    out.append(f"- HYP segments: {len(hyp)} across {len(hyp_counts)} ids "
               f"(including id=-1 = unidentified)")
    for sid, n in sorted(hyp_counts.items()):
        out.append(f"    - spk{sid}: {n}")
    out.append("")

    # Majority-vote HYP_id → GT_name: for each HYP segment, find GT utterance
    # whose [t0_start, t0_end] contains the HYP midpoint (first containing match).
    gt_sorted = sorted(gt, key=lambda r: r["t0_start"])
    def find_gt(t: float) -> dict | None:
        # linear is fine for ~100 rows; robust against overlapping GT.
        for u in gt_sorted:
            if u["t0_start"] <= t < u["t0_end"]:
                return u
        # Fall back: closest by midpoint distance.
        best = None; best_d = 1e9
        for u in gt_sorted:
            d = min(abs(t - u["t0_start"]), abs(t - u["t0_end"]))
            if d < best_d: best_d, best = d, u
        return best if best_d < 2.0 else None

    votes: dict[int, Counter] = defaultdict(Counter)
    matched_pairs = []
    for h in hyp:
        mid = 0.5 * (h["t0_start"] + h["t0_end"])
        u = find_gt(mid)
        if u is None:
            matched_pairs.append((h, None))
            continue
        votes[h["speaker_id"]][u["speaker"]] += 1
        matched_pairs.append((h, u))

    id2name: dict[int, str] = {}
    out.append("- HYP id → GT name (majority vote over overlapping HYP segments):")
    for sid in sorted(votes.keys()):
        total = sum(votes[sid].values())
        top_name, top_n = votes[sid].most_common(1)[0]
        id2name[sid] = top_name
        share = f"{top_n}/{total} = {100.0*top_n/total:.0f}%"
        rest = ", ".join(f"{n}:{c}" for n, c in votes[sid].most_common()[1:])
        rest = f" (rest: {rest})" if rest else ""
        out.append(f"    - spk{sid} → **{top_name}** [{share}]{rest}")
    out.append("")

    # Per-HYP match count against its mapped name.
    hyp_ok = sum(1 for h, u in matched_pairs
                 if u is not None and id2name.get(h["speaker_id"]) == u["speaker"])
    hyp_total = sum(1 for _, u in matched_pairs if u is not None)
    out.append(f"- HYP-side accuracy (mapped id matches overlapping GT speaker): "
               f"**{hyp_ok}/{hyp_total} = "
               f"{100.0*hyp_ok/max(hyp_total,1):.1f}%**")

    # Per-GT coverage: any HYP segment overlaps this GT, and its mapped name matches.
    # Stricter: this is the metric the user cares about ("who is speaking right now").
    gt_covered = 0; gt_correct = 0
    for u in gt:
        overlapping = [h for h in hyp
                       if h["t0_end"] > u["t0_start"] and h["t0_start"] < u["t0_end"]]
        if not overlapping: continue
        gt_covered += 1
        # Majority vote of the mapped names across overlapping HYP segments.
        names = [id2name.get(h["speaker_id"], f"spk{h['speaker_id']}")
                 for h in overlapping]
        majority = Counter(names).most_common(1)[0][0]
        if majority == u["speaker"]:
            gt_correct += 1
    out.append(f"- GT-side accuracy (GT utterances whose overlapping HYP majority "
               f"maps to the right speaker): **{gt_correct}/{gt_covered} = "
               f"{100.0*gt_correct/max(gt_covered,1):.1f}%**  "
               f"(of {len(gt)} total GT; "
               f"{len(gt)-gt_covered} had no overlapping HYP)")
    out.append("")
    out.append("Target: ≥90% GT-side accuracy before advancing to the next "
               "10-minute segment.")
    out.append("")
    return out


def render(gt: list[dict], hyp: list[dict], gt_meta: dict,
           hyp_meta: dict, out_path: Path, window: tuple[float, float] | None) -> None:
    if window is not None:
        wa, wb = window
        gt  = [r for r in gt  if r["t0_start"] < wb and r["t0_end"] > wa]
        hyp = [r for r in hyp if r["t0_start"] < wb and r["t0_end"] > wa]

    merged = sorted(gt + hyp, key=lambda r: (r["t0_start"], 0 if r["kind"] == "gt" else 1))

    lines: list[str] = []
    lines.append("# Auditus judge — side-by-side timeline")
    lines.append("")
    lines.append(f"- source audio: `{gt_meta.get('source_audio', '?')}`")
    lines.append(f"- duration: {gt_meta.get('duration_sec', 0.0):.3f} s")
    if window is not None:
        lines.append(f"- window: [{window[0]:.1f}s, {window[1]:.1f}s)")
    lines.append(f"- GT utterances: {len(gt)}")
    lines.append(f"- HYP transcripts: {len(hyp)}")
    if hyp_meta:
        lines.append(f"- HYP replay speed: {hyp_meta.get('speed', '?')}x")
        lines.append(f"- HYP captured at:  {hyp_meta.get('started_at', '?')}")
    lines.append("")
    lines.append("Legend:")
    lines.append("- **GT**  = ground truth (human annotation from tests/test.txt).")
    lines.append("- **HYP** = DeusRidet pipeline output (ASR + speaker attribution).")
    lines.append("- Times are in source-audio seconds (T0).")
    lines.append("")
    lines.extend(_diarization_summary(gt, hyp))
    lines.append("Reader instructions (Step 16):")
    lines.append("1. Read the per-minute tables below. For each GT row, find the "
                 "nearest HYP row(s) overlapping its `[t0_start, t0_end]` window.")
    lines.append("2. Mark speaker ✅/❌ using the id→name mapping in the summary "
                 "above. Transcription can be ignored for Step 16 (we are "
                 "tuning SAAS, not the ASR).")
    lines.append("3. Flag the failure modes you see: cold-start stickiness, "
                 "short-interjection drift, identity swap, unidentified (-1), etc.")
    lines.append("4. Roll findings into docs/{en,zh}/devlog/<date>.md.")
    lines.append("")

    current_bucket = -1
    table_open = False

    def open_table() -> None:
        nonlocal table_open
        if not table_open:
            lines.append("| T0 start | T0 end | Kind | Speaker | Extra | Text |")
            lines.append("|---------:|-------:|:-----|:--------|:------|:-----|")
            table_open = True

    def close_table() -> None:
        nonlocal table_open
        if table_open:
            lines.append("")
            table_open = False

    for r in merged:
        bucket = int(r["t0_start"] // 60)
        if bucket != current_bucket:
            close_table()
            beg = bucket * 60
            end = beg + 60
            lines.append(f"## {mmss(float(beg))} – {mmss(float(end))}")
            lines.append("")
            current_bucket = bucket
        open_table()

        t_start = f"{r['t0_start']:8.2f}"
        t_end   = f"{r['t0_end']:8.2f}"
        text    = r["text"].replace("|", "\\|").replace("\n", " ")

        if r["kind"] == "gt":
            row = (
                f"| {t_start} | {t_end} | **GT** | {r['speaker']} "
                f"| idx={r['idx']} | {text} |"
            )
        else:
            extra = (
                f"sim={r['sim']:.2f} conf={r['conf']:.2f} "
                f"src={r['source']} trig={r['trigger']} "
                f"lat={r['latency_ms']:.0f}ms"
            )
            row = (
                f"| {t_start} | {t_end} | HYP   | {r['speaker']} "
                f"| {extra} | {text} |"
            )
        lines.append(row)

    close_table()

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", default="tests/fixtures/test_ground_truth.json")
    ap.add_argument("--run-dir", required=True,
                    help="directory containing hyp.jsonl and run_meta.json")
    ap.add_argument("--out", default=None,
                    help="output markdown path (default: <run-dir>/judge.md)")
    ap.add_argument("--window", default=None,
                    help="restrict to [start,end) source-audio seconds, "
                         "e.g. --window 0,600")
    args = ap.parse_args()

    window = None
    if args.window:
        a, b = args.window.split(",")
        window = (float(a), float(b))

    gt_path  = Path(args.gt)
    run_dir  = Path(args.run_dir)
    hyp_path = run_dir / "hyp.jsonl"
    meta_path = run_dir / "run_meta.json"
    out_path = Path(args.out) if args.out else run_dir / "judge.md"

    gt_meta  = json.loads(gt_path.read_text(encoding="utf-8"))
    hyp_meta = {}
    if meta_path.is_file():
        hyp_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    gt  = load_gt(gt_path)
    hyp = load_hyp(hyp_path)

    render(gt, hyp, gt_meta, hyp_meta, out_path, window)
    print(f"Wrote {out_path} ({len(gt)} GT + {len(hyp)} HYP rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
