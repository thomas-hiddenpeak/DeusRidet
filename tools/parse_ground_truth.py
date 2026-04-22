#!/usr/bin/env python3
"""Parse tests/test.txt into a canonical ground-truth JSON keyed on T0.

Part of Step 15a (Auditus Evaluation Harness). This is a pure conversion
tool -- it does NOT compute any metrics. Per
.github/instructions/benchmarks.instructions.md, automated scoring
scripts are forbidden: the agent reads the paired timeline directly.
This file only produces the structured substrate on which the agent
will later do human-style reading.

Input  : tests/test.txt  (hand-transcribed, HH:MM:SS <speaker>\\n<text>\\n\\n ...)
Output : tests/fixtures/test_ground_truth.json

Output schema:
{
  "source_audio": "tests/test.mp3",
  "duration_sec": <float>,
  "speakers": [<name>, ...],
  "utterances": [
    {
      "idx":           <int>,           # 0-based stable index
      "t0_start_sec":  <float>,         # HH:MM:SS parsed to seconds
      "t0_end_sec":    <float>,         # next utt start, or duration for tail
      "speaker":       <str>,           # one of the four labels
      "text":          <str>            # normalised single-line text
    }, ...
  ]
}
"""
from __future__ import annotations

import json
import re
import sys
import subprocess
from pathlib import Path


TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\s+(\S+)\s*$")
AI_FOOTER = "【内容由 AI 生成，仅供参考】"


def hhmmss_to_sec(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


def probe_audio_duration(path: Path) -> float:
    """Ask ffprobe for the audio duration in seconds."""
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def parse(txt_path: Path, audio_path: Path) -> dict:
    raw = txt_path.read_text(encoding="utf-8")
    # Strip trailing AI-generated disclaimer if present.
    raw = raw.replace(AI_FOOTER, "").rstrip() + "\n"

    lines = raw.splitlines()
    utterances: list[dict] = []
    speakers_seen: list[str] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        m = TIMESTAMP_RE.match(line)
        if not m:
            i += 1
            continue
        h, mm, ss, speaker = m.group(1), m.group(2), m.group(3), m.group(4)
        start_sec = float(hhmmss_to_sec(h, mm, ss))

        # Collect text lines until the next timestamp line or EOF.
        text_parts: list[str] = []
        j = i + 1
        while j < n:
            next_line = lines[j]
            if TIMESTAMP_RE.match(next_line.strip()):
                break
            stripped = next_line.strip()
            if stripped:
                text_parts.append(stripped)
            j += 1
        text = "".join(text_parts).strip()  # Chinese: no space join

        if speaker not in speakers_seen:
            speakers_seen.append(speaker)

        utterances.append(
            {
                "idx": len(utterances),
                "t0_start_sec": start_sec,
                "t0_end_sec": None,  # filled after traversal
                "speaker": speaker,
                "text": text,
            }
        )
        i = j

    # Fill end times from successor starts; tail uses audio duration.
    duration = probe_audio_duration(audio_path)
    for k, utt in enumerate(utterances):
        if k + 1 < len(utterances):
            utt["t0_end_sec"] = utterances[k + 1]["t0_start_sec"]
        else:
            utt["t0_end_sec"] = round(duration, 3)

    return {
        "source_audio": str(audio_path.as_posix()),
        "duration_sec": round(duration, 3),
        "speakers": sorted(speakers_seen),
        "utterances": utterances,
    }


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    txt = repo / "tests" / "test.txt"
    audio = repo / "tests" / "test.mp3"
    out_dir = repo / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_ground_truth.json"

    if not txt.is_file():
        print(f"ERROR: {txt} not found", file=sys.stderr)
        return 1
    if not audio.is_file():
        print(f"ERROR: {audio} not found", file=sys.stderr)
        return 1

    data = parse(txt, audio)
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    n = len(data["utterances"])
    dur = data["duration_sec"]
    print(
        f"Wrote {out_path.relative_to(repo)}: "
        f"{n} utterances, {len(data['speakers'])} speakers, "
        f"duration={dur:.3f}s"
    )
    # Sanity summary
    per_spk: dict[str, int] = {}
    for u in data["utterances"]:
        per_spk[u["speaker"]] = per_spk.get(u["speaker"], 0) + 1
    for spk, cnt in sorted(per_spk.items(), key=lambda kv: -kv[1]):
        print(f"  {spk}: {cnt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
