#!/usr/bin/env python3
"""Pre-enhance audio through FRCRN for offline A/B testing.

Processes a full audio file through FRCRN speech enhancement model,
producing a clean WAV file that can be fed through the pipeline
for speaker accuracy comparison.

Usage:
    python3 tools/frcrn_enhance_audio.py tests/test.mp3 -o tests/test_frcrn.wav
    
Then compare:
    # Original
    python3 tools/test_audio_ws.py tests/test.mp3 --speed 10
    # Enhanced
    python3 tools/test_audio_ws.py tests/test_frcrn.wav --speed 10
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np


def decode_to_pcm(path: str) -> np.ndarray:
    """Decode any audio file to 16kHz mono float32 PCM via ffmpeg."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", "16000", "-ac", "1",
        "pipe:1"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    return np.frombuffer(result.stdout, dtype=np.float32)


def save_wav(path: str, pcm: np.ndarray, sr: int = 16000):
    """Save float32 PCM to 16-bit WAV via ffmpeg."""
    # Convert to int16
    pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "s16le", "-ar", str(sr), "-ac", "1",
        "-i", "pipe:0",
        "-acodec", "pcm_s16le",
        path
    ]
    proc = subprocess.run(cmd, input=pcm_int16.tobytes(), capture_output=True)
    if proc.returncode != 0:
        print(f"ffmpeg error: {proc.stderr.decode()}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="FRCRN offline audio enhancement")
    parser.add_argument("input", help="Input audio file (any format)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output WAV file (default: <input>_frcrn.wav)")
    parser.add_argument("--model", default=os.path.expanduser("~/models/dev/vad/frcrn_se_16k.onnx"),
                        help="FRCRN ONNX model path")
    parser.add_argument("--chunk-sec", type=float, default=3.0,
                        help="Processing chunk size in seconds (default: 3.0)")
    args = parser.parse_args()

    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = base + "_frcrn.wav"

    # Decode input.
    print(f"Decoding {args.input} ...")
    pcm = decode_to_pcm(args.input)
    duration = len(pcm) / 16000
    print(f"  {len(pcm)} samples, {duration:.1f}s")

    # Load FRCRN model.
    print(f"Loading FRCRN model: {args.model} ...")
    import onnxruntime as ort
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    print(f"  Model loaded")

    # Process in chunks with progress.
    chunk_samples = int(args.chunk_sec * 16000)
    # Align to hop (320 samples)
    chunk_samples = (chunk_samples // 320) * 320

    enhanced = np.zeros_like(pcm)
    n_chunks = (len(pcm) + chunk_samples - 1) // chunk_samples
    total_inference_ms = 0.0

    print(f"Processing {n_chunks} chunks ({args.chunk_sec:.1f}s each) ...")
    for i in range(n_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, len(pcm))
        chunk = pcm[start:end]

        # Pad to hop alignment.
        padded_len = len(chunk)
        if padded_len % 320 != 0:
            padded_len = ((padded_len // 320) + 1) * 320
        if padded_len < 640:
            padded_len = 640

        chunk_padded = np.zeros(padded_len, dtype=np.float32)
        chunk_padded[:len(chunk)] = chunk

        # Run FRCRN.
        t0 = time.time()
        input_data = chunk_padded.reshape(1, -1)
        output = sess.run(None, {"audio": input_data})[0]
        t1 = time.time()
        total_inference_ms += (t1 - t0) * 1000

        # Copy output (may be shorter than padded).
        out_flat = output.flatten()
        copy_len = min(end - start, len(out_flat))
        enhanced[start:start + copy_len] = out_flat[:copy_len]

        # Progress.
        if (i + 1) % 10 == 0 or (i + 1) == n_chunks:
            pct = (i + 1) / n_chunks * 100
            elapsed = total_inference_ms / 1000
            processed_sec = (start + copy_len) / 16000
            rtf = elapsed / processed_sec if processed_sec > 0 else 0
            print(f"  [{pct:5.1f}%] {processed_sec:.0f}/{duration:.0f}s "
                  f"(inference: {elapsed:.1f}s, RTF={rtf:.3f})")

    # Save output.
    print(f"Saving to {args.output} ...")
    save_wav(args.output, enhanced)
    file_size = os.path.getsize(args.output) / 1024 / 1024
    print(f"  {file_size:.1f} MB")

    print(f"\nDone! Total inference: {total_inference_ms/1000:.1f}s "
          f"(RTF={total_inference_ms/1000/duration:.3f})")
    print(f"\nTo test accuracy:")
    print(f"  1. Start server: ./build/deusridet awaken")
    print(f"  2. Run original:  python3 tools/test_audio_ws.py {args.input} --speed 10")
    print(f"  3. Run enhanced:  python3 tools/test_audio_ws.py {args.output} --speed 10")
    print(f"  4. Compare:       python3 tools/eval_speaker_accuracy.py")


if __name__ == "__main__":
    main()
