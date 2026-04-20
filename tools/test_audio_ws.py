#!/usr/bin/env python3
"""Send an audio file to DeusRidet via WebSocket and print speaker + ASR events.

Usage:
    python3 tools/test_audio_ws.py <audio_file> [--url ws://localhost:8080/ws] [--speed 1.0]

Supports any format ffmpeg can decode (mp3, wav, flac, ogg, m4a, ...).
Audio is converted to 16kHz mono int16 PCM and streamed in 512-sample chunks
at real-time pace (adjustable via --speed).
"""

import argparse
import json
import subprocess
import struct
import sys
import threading
import time
from typing import Optional

import websocket


# Runtime shared state for lightweight timeline logging.
_SENT_SAMPLES = 0
_STATS_LOG_FH: Optional[object] = None
_LAST_MULTI_STATE: Optional[bool] = None


def decode_to_pcm(path: str) -> bytes:
    """Decode any audio file to raw 16kHz mono int16 PCM via ffmpeg."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "pipe:1"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def on_message(ws, message):
    """Handle text messages from the server (JSON events)."""
    try:
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        data = json.loads(message)
        msg_type = data.get("type", "")

        if msg_type == "asr_transcript":
            spk = data.get("speaker_name") or f"spk{data.get('speaker_id', '?')}"
            text = data.get("text", "")
            trigger = data.get("trigger", "")
            src = data.get("speaker_source", "")
            sim = data.get("speaker_sim", 0)
            conf = data.get("speaker_confidence", 0)
            t0 = data.get("stream_start_sec", 0)
            t1 = data.get("stream_end_sec", 0)
            print(f"  ASR [{spk}] ({t0:.1f}-{t1:.1f}s sim={sim:.2f} conf={conf:.2f} src={src} trigger={trigger}) {text}")

        elif msg_type == "asr_partial":
            text = data.get("text", "")
            audio_sec = data.get("audio_sec", 0)
            print(f"  ASR(partial, {audio_sec:.1f}s): {text}")

        elif msg_type == "speaker":
            sid = data.get("id", -1)
            name = data.get("name", "")
            sim = data.get("sim", 0)
            is_new = data.get("new", False)
            label = name if name else f"spk{sid}"
            tag = "NEW " if is_new else ""
            print(f"  SPK: {tag}{label} (id={sid} sim={sim:.3f})")

        elif msg_type == "pipeline_stats":
            global _LAST_MULTI_STATE

            audio_sec = _SENT_SAMPLES / 16000.0
            multi = bool(data.get("multi_speaker", False))
            multi_score = float(data.get("multi_score", 0.0) or 0.0)
            multi_source = data.get("multi_source", "")

            if _STATS_LOG_FH is not None:
                rec = {
                    "type": "pipeline_stats",
                    "audio_sec": round(audio_sec, 3),
                    "recv_ts": round(time.time(), 3),
                    "multi_speaker": multi,
                    "multi_score": multi_score,
                    "multi_source": multi_source,
                }
                _STATS_LOG_FH.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _STATS_LOG_FH.flush()

            if _LAST_MULTI_STATE is None or _LAST_MULTI_STATE != multi:
                print(f"  MULTI: {'ON' if multi else 'OFF'} @~{audio_sec:.1f}s "
                      f"(score={multi_score:.2f} src={multi_source})")
                _LAST_MULTI_STATE = multi

            # Periodic stats — only print speaker-relevant fields if active.
            wl_active = data.get("wlecapa_active", False)
            if wl_active:
                sid = data.get("wlecapa_id", -1)
                name = data.get("wlecapa_name", "")
                sim = data.get("wlecapa_sim", 0)
                label = name if name else f"spk{sid}"
                lat = data.get("wlecapa_lat_total_ms", 0)
                early = data.get("wlecapa_is_early", False)
                stage = "EARLY" if early else "FULL"
                print(f"  STATS: {label} id={sid} sim={sim:.3f} lat={lat:.0f}ms {stage}")

        elif msg_type == "asr_log":
            # ASR pipeline debug log.
            inner = data.get("data", "")
            if inner:
                try:
                    d = json.loads(inner) if isinstance(inner, str) else inner
                    stage = d.get("stage", "")
                    if stage == "trigger":
                        reason = d.get("reason", "")
                        buf_sec = d.get("buf_sec", 0)
                        print(f"  ASR-TRIGGER: {reason} buf={buf_sec:.2f}s")
                except (json.JSONDecodeError, AttributeError):
                    pass

    except json.JSONDecodeError:
        pass


def on_error(ws, error):
    print(f"WS error: {error}", file=sys.stderr)


def on_close(ws, close_status_code, close_msg):
    print(f"\n--- WebSocket closed ---")


def on_open(ws):
    print("--- WebSocket connected ---")


def main():
    global _SENT_SAMPLES
    global _STATS_LOG_FH

    parser = argparse.ArgumentParser(description="Send audio to DeusRidet WS")
    parser.add_argument("audio_file", help="Path to audio file (mp3, wav, etc.)")
    parser.add_argument("--url", default="ws://localhost:8080/ws",
                        help="WebSocket URL (default: ws://localhost:8080/ws)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0 = real-time)")
    parser.add_argument("--chunk", type=int, default=512,
                        help="Samples per WS frame (default: 512 = 32ms)")
    parser.add_argument("--stats-log", default="",
                        help="Optional JSONL file to save pipeline multi-speaker timeline")
    args = parser.parse_args()

    if args.stats_log:
        _STATS_LOG_FH = open(args.stats_log, "w", encoding="utf-8")

    print(f"Decoding {args.audio_file} ...")
    pcm = decode_to_pcm(args.audio_file)
    n_samples = len(pcm) // 2
    duration = n_samples / 16000.0
    print(f"  {n_samples} samples, {duration:.1f}s")

    chunk_samples = args.chunk
    chunk_bytes = chunk_samples * 2  # int16 = 2 bytes
    chunk_duration = chunk_samples / 16000.0  # seconds per chunk
    n_chunks = (len(pcm) + chunk_bytes - 1) // chunk_bytes

    # Connect WebSocket with message receiver on a separate thread.
    ws = websocket.WebSocketApp(
        args.url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=lambda: ws.run_forever(skip_utf8_validation=True), daemon=True)
    ws_thread.start()

    # Wait for connection.
    for _ in range(50):
        if ws.sock and ws.sock.connected:
            break
        time.sleep(0.1)
    else:
        print("Failed to connect to WebSocket", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming {n_chunks} chunks ({chunk_samples} samples each, "
          f"speed={args.speed:.1f}x) ...")
    print("=" * 60)

    sleep_per_chunk = chunk_duration / args.speed
    t0 = time.time()
    sent = 0

    try:
        for i in range(n_chunks):
            start = i * chunk_bytes
            end = min(start + chunk_bytes, len(pcm))
            chunk = pcm[start:end]

            # Pad last chunk if needed.
            if len(chunk) < chunk_bytes:
                chunk = chunk + b'\x00' * (chunk_bytes - len(chunk))

            ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
            sent += chunk_samples
            _SENT_SAMPLES = sent

            # Real-time pacing.
            elapsed = time.time() - t0
            expected = (i + 1) * sleep_per_chunk
            if expected > elapsed:
                time.sleep(expected - elapsed)

            # Progress every 5 seconds.
            pos_sec = sent / 16000.0
            if i > 0 and i % int(5.0 / chunk_duration) == 0:
                print(f"  ... {pos_sec:.1f}s / {duration:.1f}s")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    print("=" * 60)
    print(f"Done streaming. Waiting 5s for final ASR results ...")
    time.sleep(5)

    ws.close()
    if _STATS_LOG_FH is not None:
        _STATS_LOG_FH.close()
    print("Finished.")


if __name__ == "__main__":
    main()
