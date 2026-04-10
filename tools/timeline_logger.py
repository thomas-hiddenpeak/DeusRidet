#!/usr/bin/env python3
"""
timeline_analyze.py — Summary & analysis tool for DeusRidet timeline logs.

The main C++ program writes JSONL timeline logs to logs/timeline/ automatically.
This tool reads those logs and prints structured summaries for analysis.

Usage:
  python3 tools/timeline_logger.py --summary         # analyze latest log
  python3 tools/timeline_logger.py --summary <file>  # analyze specific log
  python3 tools/timeline_logger.py --list             # list available logs
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LOG_DIR = PROJECT_DIR / "logs" / "timeline"


def latest_log():
    if not LOG_DIR.exists():
        return None
    files = sorted(LOG_DIR.glob("tl_*.jsonl"))
    return files[-1] if files else None


# ─── Summary / Analysis ───

def print_summary(log_path):
    """Print a structured summary of a timeline log file."""
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return

    events = []
    header = footer = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("t") == "header":
                header = obj
            elif obj.get("t") == "footer":
                footer = obj
            else:
                events.append(obj)

    print(f"═══ Timeline Log: {os.path.basename(log_path)} ═══")
    if header:
        print(f"  Started: {header.get('started', '?')}")
    if footer:
        print(f"  Duration: {footer.get('duration_sec', '?')}s")
        print(f"  Events: {footer.get('counts', {})}")
    print(f"  Total lines: {len(events)}")
    print()

    # Stats summary
    stats = [e for e in events if e.get("t") == "stats"]
    if stats:
        stream_range = (stats[0].get("s", 0), stats[-1].get("s", 0))
        print(f"─── Pipeline Stats ({len(stats)} ticks) ───")
        print(f"  Stream time: {stream_range[0]:.1f}s → {stream_range[1]:.1f}s ({stream_range[1] - stream_range[0]:.1f}s span)")

        # VAD intervals
        vad_starts = []
        vad_ends = []
        prev_sp = None
        for e in stats:
            sp = e.get("speech", False)
            if sp and not prev_sp:
                vad_starts.append(e["s"])
            elif not sp and prev_sp:
                vad_ends.append(e["s"])
            prev_sp = sp
        if prev_sp:
            vad_ends.append(stats[-1]["s"])
        print(f"  VAD intervals: {len(vad_starts)}")
        for i, (vs, ve) in enumerate(zip(vad_starts, vad_ends)):
            print(f"    [{i}] {vs:.1f}s – {ve:.1f}s ({ve - vs:.1f}s)")
            if i >= 30:
                print(f"    ... ({len(vad_starts) - 30} more)")
                break

        # Tracker spans
        trk_checks = [e for e in stats if e.get("trk_check")]
        if trk_checks:
            print(f"  Tracker checks: {len(trk_checks)}")
            prev_id = None
            prev_state = None
            spans = []
            for e in trk_checks:
                tid = e.get("trk_id", -1)
                tst = e.get("trk_state", -1)
                if tid != prev_id or tst != prev_state:
                    if spans:
                        spans[-1]["end"] = e["s"]
                    spans.append({"start": e["s"], "id": tid, "name": e.get("trk_name", ""),
                                  "state": tst, "sim": e.get("trk_sim")})
                    prev_id = tid
                    prev_state = tst
            if spans:
                spans[-1]["end"] = trk_checks[-1]["s"]
            STATE_NAMES = {0: "SIL", 1: "TRK", 2: "TRANS", 3: "OVLP", 4: "UNK"}
            print(f"  Tracker spans: {len(spans)}")
            for sp in spans:
                end = sp.get("end")
                if end is not None:
                    dur = f"{end - sp['start']:.1f}s"
                    end_s = f"{end:.1f}"
                else:
                    dur = "?"
                    end_s = "?"
                sname = STATE_NAMES.get(sp["state"], str(sp["state"]))
                label = sp.get("name") or (f"T{sp['id']}" if sp["id"] >= 0 else "?")
                sim_s = f" sim={sp['sim']:.3f}" if sp.get("sim") is not None else ""
                print(f"    {sp['start']:.1f}s – {end_s}s ({dur}) [{sname}] {label}{sim_s}")

            # Switches
            switches_vals = [e.get("trk_switches", 0) for e in trk_checks]
            if switches_vals:
                print(f"  Tracker switches: {switches_vals[0]} → {switches_vals[-1]} (+{switches_vals[-1] - switches_vals[0]})")

            # Registrations
            regs = [e for e in stats if e.get("trk_reg")]
            if regs:
                print(f"  Tracker registrations: {len(regs)}")
                for e in regs:
                    print(f"    {e['s']:.1f}s: id={e.get('trk_reg_id')} name={e.get('trk_reg_name', '')}")

    # ASR summary
    asr_events = [e for e in events if e.get("t") == "asr"]
    if asr_events:
        print(f"\n─── ASR Transcripts ({len(asr_events)}) ───")
        for i, e in enumerate(asr_events):
            s, end = e.get("s", 0), e.get("e", 0)
            spk = e.get("spk_name") or (f"S{e.get('spk_id', '?')}" if e.get("spk_id", -1) >= 0 else "?")
            trk = e.get("trk_name") or (f"T{e.get('trk_id', '?')}" if e.get("trk_id", -1) >= 0 else "?")
            src = e.get("spk_src", "")
            trig = e.get("trigger", "")
            lat = e.get("latency", 0)
            text = e.get("text", "")
            print(f"  [{s:.1f}–{end:.1f}s] SPK={spk}({src}) TRK={trk} trig={trig} lat={lat:.0f}ms")
            print(f"    \"{text}\"")

    # Cross-lane alignment check
    if asr_events and stats:
        print(f"\n─── Alignment Check ───")
        stats_range = (stats[0]["s"], stats[-1]["s"])
        for e in asr_events:
            s, end = e.get("s", 0), e.get("e", 0)
            in_range = stats_range[0] <= s and end <= stats_range[1] + 1
            status = "OK" if in_range else "OUTSIDE stats range"
            # Find matching VAD interval
            vad_match = False
            for vs, ve in zip(vad_starts, vad_ends):
                if vs <= end and ve >= s:
                    vad_match = True
                    break
            vad_s = "VAD✓" if vad_match else "VAD✗"
            print(f"  ASR [{s:.1f}–{end:.1f}s] {status} | {vad_s}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeusRidet Timeline Log Analyzer")
    parser.add_argument("--summary", nargs="?", const="__latest__", help="Print summary (latest or specific file)")
    parser.add_argument("--list", action="store_true", help="List available log files")
    args = parser.parse_args()

    if args.list:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(LOG_DIR.glob("tl_*.jsonl"))
        if not files:
            print("No timeline logs found.")
        else:
            for f in files:
                size = f.stat().st_size / 1024
                print(f"  {f.name}  ({size:.0f} KB)")
        sys.exit(0)

    if args.summary is not None:
        if args.summary == "__latest__":
            lf = latest_log()
            if not lf:
                print("No timeline logs found.")
                sys.exit(1)
            print_summary(str(lf))
        else:
            # Check if it's a bare filename or full path
            p = Path(args.summary)
            if not p.exists():
                p = LOG_DIR / args.summary
            print_summary(str(p))
        sys.exit(0)

    # Default: show summary of latest log.
    lf = latest_log()
    if lf:
        print_summary(str(lf))
    else:
        parser.print_help()
        print("\nNo timeline logs found. Logs are created automatically by the main program.")
