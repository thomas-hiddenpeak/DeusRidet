#!/usr/bin/env python3
"""
timeline_logger.py — Persistent timeline data recorder for DeusRidet.

Connects to the WebSocket, records ALL timeline-relevant events (VAD, ASR, SAAS,
Tracker) as JSONL (one JSON object per line). Each run creates a new timestamped
log file in logs/timeline/.

Output file: logs/timeline/tl_YYYYMMDD_HHMMSS.jsonl

Each line is one of:
  {"t":"stats", "s":<stream_sec>, ...}   — pipeline_stats snapshot
  {"t":"asr",   "s":<stream_start>, ...} — ASR transcript
  {"t":"vad",   ...}                     — VAD event

Usage:
  python3 tools/timeline_logger.py                  # run until Ctrl+C
  python3 tools/timeline_logger.py --duration 300   # run for 5 minutes
  python3 tools/timeline_logger.py --summary         # print summary of latest log
  python3 tools/timeline_logger.py --summary <file>  # print summary of specific log
"""

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LOG_DIR = PROJECT_DIR / "logs" / "timeline"
WS_URI = "ws://localhost:8080/ws"


def new_log_path():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"tl_{ts}.jsonl"


def latest_log():
    if not LOG_DIR.exists():
        return None
    files = sorted(LOG_DIR.glob("tl_*.jsonl"))
    return files[-1] if files else None


# ─── Recording ───

async def record(duration=None):
    try:
        import websockets
    except ImportError:
        print("ERROR: pip install websockets")
        sys.exit(1)

    log_path = new_log_path()
    print(f"Recording to: {log_path}")
    print(f"WebSocket:    {WS_URI}")
    if duration:
        print(f"Duration:     {duration}s")
    print("Press Ctrl+C to stop.\n")

    counts = {"stats": 0, "asr": 0, "vad": 0, "other": 0}
    wall_t0 = time.time()
    stop = asyncio.Event()

    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    with open(log_path, "w") as f:
        # Write header comment
        f.write(json.dumps({
            "t": "header",
            "version": 1,
            "started": datetime.now().isoformat(),
            "ws_uri": WS_URI,
        }) + "\n")

        try:
            async with websockets.connect(WS_URI, max_size=4 * 1024 * 1024) as ws:
                print("Connected.")

                while not stop.is_set():
                    if duration and (time.time() - wall_t0) >= duration:
                        break

                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    if not isinstance(msg, str):
                        continue

                    try:
                        obj = json.loads(msg)
                    except json.JSONDecodeError:
                        continue

                    wall = round(time.time() - wall_t0, 4)
                    msg_type = obj.get("type", "")

                    if msg_type == "pipeline_stats":
                        pcm = obj.get("pcm_samples", 0)
                        rec = {
                            "t": "stats",
                            "w": wall,
                            "s": round(pcm / 16000.0, 4),
                            # VAD
                            "speech": obj.get("is_speech", False),
                            "energy": obj.get("energy"),
                            "rms": obj.get("rms"),
                            "silero_p": obj.get("silero_prob"),
                            "silero_sp": obj.get("silero_speech"),
                            "fsmn_p": obj.get("fsmn_prob"),
                            "fsmn_sp": obj.get("fsmn_speech"),
                            # WL-ECAPA (SAAS)
                            "wle_active": obj.get("wlecapa_active"),
                            "wle_id": obj.get("wlecapa_id"),
                            "wle_name": obj.get("wlecapa_name"),
                            "wle_sim": obj.get("wlecapa_sim"),
                            "wle_early": obj.get("wlecapa_is_early"),
                            "wle_margin": obj.get("wlecapa_margin"),
                            "change_sim": obj.get("change_similarity"),
                            # Tracker
                            "trk_state": obj.get("tracker_state"),
                            "trk_id": obj.get("tracker_spk_id"),
                            "trk_name": obj.get("tracker_spk_name"),
                            "trk_sim": obj.get("tracker_sim_to_ref"),
                            "trk_avg": obj.get("tracker_sim_avg"),
                            "trk_conf": obj.get("tracker_confidence"),
                            "trk_check": obj.get("tracker_check_active"),
                            "trk_switches": obj.get("tracker_switches"),
                            "trk_f0": obj.get("tracker_f0_hz"),
                            "trk_jitter": obj.get("tracker_f0_jitter"),
                            "trk_reg": obj.get("tracker_reg_event"),
                            "trk_reg_id": obj.get("tracker_reg_id"),
                            "trk_reg_name": obj.get("tracker_reg_name"),
                            # ASR buffer state
                            "asr_buf": obj.get("asr_buf_sec"),
                            "asr_buf_sp": obj.get("asr_buf_has_speech"),
                            "asr_busy": obj.get("asr_busy"),
                            "asr_sil_ms": obj.get("asr_current_silence_ms"),
                            "asr_eff_sil": obj.get("asr_effective_silence_ms"),
                        }
                        # Strip None values
                        rec = {k: v for k, v in rec.items() if v is not None}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts["stats"] += 1

                    elif msg_type == "asr_transcript":
                        rec = {
                            "t": "asr",
                            "w": wall,
                            "s": obj.get("stream_start_sec", 0),
                            "e": obj.get("stream_end_sec", 0),
                            "text": obj.get("text", ""),
                            "trigger": obj.get("trigger", ""),
                            "latency": obj.get("latency_ms"),
                            "audio": obj.get("audio_sec"),
                            "mel_ms": obj.get("mel_ms"),
                            "enc_ms": obj.get("encoder_ms"),
                            "dec_ms": obj.get("decode_ms"),
                            "tokens": obj.get("tokens"),
                            # SAAS
                            "spk_id": obj.get("speaker_id"),
                            "spk_name": obj.get("speaker_name"),
                            "spk_sim": obj.get("speaker_sim"),
                            "spk_conf": obj.get("speaker_confidence"),
                            "spk_src": obj.get("speaker_source"),
                            # Tracker
                            "trk_id": obj.get("tracker_id"),
                            "trk_name": obj.get("tracker_name"),
                            "trk_sim": obj.get("tracker_sim"),
                        }
                        rec = {k: v for k, v in rec.items() if v is not None}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts["asr"] += 1

                    elif msg_type == "vad":
                        rec = {
                            "t": "vad",
                            "w": wall,
                            "event": obj.get("event"),
                            "speech": obj.get("speech"),
                            "frame": obj.get("frame"),
                            "energy": obj.get("energy"),
                        }
                        rec = {k: v for k, v in rec.items() if v is not None}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts["vad"] += 1

                    elif msg_type in ("speaker", "speaker_debug"):
                        rec = {"t": msg_type, "w": wall}
                        rec.update({k: v for k, v in obj.items() if k != "type"})
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts["other"] += 1

                    # Periodic status
                    total = sum(counts.values())
                    if total % 100 == 0 and total > 0:
                        elapsed = time.time() - wall_t0
                        print(f"  {elapsed:.0f}s | stats={counts['stats']} asr={counts['asr']} vad={counts['vad']} | {os.path.getsize(log_path) / 1024:.0f} KB")

        except Exception as e:
            print(f"\nConnection error: {e}")

        # Write footer
        f.write(json.dumps({
            "t": "footer",
            "ended": datetime.now().isoformat(),
            "duration_sec": round(time.time() - wall_t0, 1),
            "counts": counts,
        }) + "\n")

    elapsed = round(time.time() - wall_t0, 1)
    size_kb = os.path.getsize(log_path) / 1024
    print(f"\nDone. {elapsed}s, {sum(counts.values())} events, {size_kb:.0f} KB")
    print(f"  stats={counts['stats']}  asr={counts['asr']}  vad={counts['vad']}  other={counts['other']}")
    print(f"  Saved: {log_path}")
    return log_path


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
    parser = argparse.ArgumentParser(description="DeusRidet Timeline Logger")
    parser.add_argument("--duration", type=int, help="Recording duration in seconds")
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

    asyncio.run(record(duration=args.duration))
