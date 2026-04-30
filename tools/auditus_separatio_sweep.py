#!/usr/bin/env python3
"""Run separatio parameter sweeps through ASR+speaker-ID+GT monitoring.

This is an offline production-tuning harness: each variant runs separation,
ASR, speaker ID, GT comparison, and then cross-variant monitors. It does not
change online Auditus behavior.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = [
    {
        "name": "raw_o4000",
        "frcrn": False,
        "overlap": 4000,
    },
    {
        "name": "raw_o8000",
        "frcrn": False,
        "overlap": 8000,
        "reuse_sep": "logs/separatio_examen_window_r1",
        "reuse_asr": "logs/separatio_asr_window_r1",
    },
    {
        "name": "raw_o12000",
        "frcrn": False,
        "overlap": 12000,
    },
    {
        "name": "raw_o16000",
        "frcrn": False,
        "overlap": 16000,
    },
    {
        "name": "frcrn_o8000",
        "frcrn": True,
        "overlap": 8000,
        "reuse_sep": "logs/separatio_examen_frcrn_window_r1",
        "reuse_asr": "logs/separatio_asr_frcrn_window_r1",
    },
    {
        "name": "frcrn_o12000",
        "frcrn": True,
        "overlap": 12000,
    },
]


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def run_capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def cleanup_runtime() -> None:
    subprocess.run(["pkill", "-9", "-x", "deusridet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["fuser", "-k", "8080/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["sudo", "tee", "/proc/sys/vm/drop_caches"], input="3\n", text=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    alive = subprocess.run(["pgrep", "-x", "deusridet"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if alive.stdout.strip():
        raise RuntimeError("deusridet still alive after cleanup")


def load_variant_file(path: Path) -> list[dict[str, Any]]:
    variants = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(variants, list):
        raise ValueError("variant file must contain a JSON list")
    return variants


def variant_dirs(root: Path, variant: dict[str, Any]) -> tuple[Path, Path]:
    name = str(variant["name"])
    sep_dir = Path(str(variant.get("reuse_sep", ""))) if variant.get("reuse_sep") else root / f"separatio_{name}"
    asr_dir = Path(str(variant.get("reuse_asr", ""))) if variant.get("reuse_asr") else root / f"asr_{name}"
    return sep_dir, asr_dir


def run_separation(args: argparse.Namespace, variant: dict[str, Any], sep_dir: Path, log_dir: Path) -> None:
    name = str(variant["name"])
    if (sep_dir / "audio").is_dir() and not args.force:
        print(f"[reuse-sep] {name}: {sep_dir}")
        return
    cleanup_runtime()
    cmd = [
        "./build/auditus_separatio_examen",
        "--clips-dir", args.clips_dir,
        "--out-dir", str(sep_dir),
        "--limit", str(args.limit),
        "--max-chunk", str(variant.get("max_chunk", args.max_chunk)),
        "--overlap-samples", str(variant.get("overlap", args.overlap)),
        "--two-source-balance", str(variant.get("two_source_balance", args.two_source_balance)),
        "--two-source-corr", str(variant.get("two_source_corr", args.two_source_corr)),
    ]
    if variant.get("frcrn"):
        cmd.append("--frcrn")
    print(f"[sep] {name}: overlap={variant.get('overlap', args.overlap)} frcrn={bool(variant.get('frcrn'))}")
    run_logged(cmd, log_dir / f"{name}_separatio.log")


def run_asr(args: argparse.Namespace, variant: dict[str, Any], sep_dir: Path, asr_dir: Path, log_dir: Path) -> None:
    name = str(variant["name"])
    compare_path = asr_dir / "asr_gt_compare.json"
    if compare_path.is_file() and not args.force:
        print(f"[reuse-asr] {name}: {compare_path}")
        return
    cleanup_runtime()
    audio_dir = sep_dir / "audio"
    cmd = [
        "./build/auditus_separatio_asr",
        "--audio-dir", str(audio_dir),
        "--out", str(asr_dir / "asr_sources.jsonl"),
        "--limit", str(args.limit),
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    print(f"[asr] {name}: {audio_dir}")
    run_logged(cmd, log_dir / f"{name}_asr.log")
    run_logged([
        "python3", "tools/summarize_separatio_asr.py",
        "--asr-jsonl", str(asr_dir / "asr_sources.jsonl"),
        "--out-dir", str(asr_dir),
    ], log_dir / f"{name}_summary.log")


def run_spkid(args: argparse.Namespace, variant: dict[str, Any], sep_dir: Path, asr_dir: Path, log_dir: Path) -> None:
    name = str(variant["name"])
    spkid_path = asr_dir / "spkid_sources.jsonl"
    identity_path = asr_dir / "identity_gt_compare.json"
    if spkid_path.is_file() and identity_path.is_file() and not args.force_spkid:
        print(f"[reuse-spkid] {name}: {identity_path}")
        return
    cleanup_runtime()
    print(f"[spkid] {name}: {sep_dir / 'audio'}")
    run_logged([
        "./build/auditus_separatio_spkid",
        "--audio-dir", str(sep_dir / "audio"),
        "--out", str(spkid_path),
        "--limit", str(args.limit),
        "--match-threshold", str(args.spkid_match_threshold),
        "--min-margin", str(args.spkid_min_margin),
        "--source-window-sec", str(args.spkid_source_window_sec),
    ], log_dir / f"{name}_spkid.log")
    run_logged([
        "python3", "tools/summarize_separatio_identity.py",
        "--asr-compare", str(asr_dir / "asr_gt_compare.json"),
        "--spkid-jsonl", str(spkid_path),
        "--out-dir", str(asr_dir),
    ], log_dir / f"{name}_identity.log")


def run_monitor(root: Path, variants: list[dict[str, Any]], args: argparse.Namespace) -> None:
    cmd = ["python3", "tools/auditus_separatio_monitor.py", "--out-dir", str(root / "monitor")]
    for variant in variants:
        _, asr_dir = variant_dirs(root, variant)
        compare_path = asr_dir / "asr_gt_compare.json"
        if compare_path.is_file():
            cmd.extend(["--variant", f"{variant['name']}={compare_path}"])
    run_logged(cmd, root / "logs" / "monitor.log")
    print(run_capture(["sed", "-n", "1,45p", str(root / "monitor" / "separatio_asr_monitor.md")]))

    identity_cmd = ["python3", "tools/auditus_separatio_identity_monitor.py", "--out-dir", str(root / "identity_monitor")]
    for variant in variants:
        _, asr_dir = variant_dirs(root, variant)
        identity_path = asr_dir / "identity_gt_compare.json"
        if identity_path.is_file():
            identity_cmd.extend(["--variant", f"{variant['name']}={identity_path}"])
    run_logged(identity_cmd, root / "logs" / "identity_monitor.log")
    print(run_capture(["sed", "-n", "1,45p", str(root / "identity_monitor" / "separatio_identity_monitor.md")]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run separatio parameter variants through ASR+GT monitor.")
    parser.add_argument("--out-root", default="logs/separatio_param_sweep_r1")
    parser.add_argument("--clips-dir", default="logs/segment_homogeneity_clips_r3/clips")
    parser.add_argument("--variant-file", default="")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-chunk", type=int, default=32000)
    parser.add_argument("--overlap", type=int, default=8000)
    parser.add_argument("--two-source-balance", type=float, default=0.05)
    parser.add_argument("--two-source-corr", type=float, default=0.92)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--spkid-match-threshold", type=float, default=0.35)
    parser.add_argument("--spkid-min-margin", type=float, default=0.03)
    parser.add_argument("--spkid-source-window-sec", type=float, default=0.0)
    parser.add_argument("--skip-spkid", action="store_true")
    parser.add_argument("--force-spkid", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    log_dir = root / "logs"
    variants = load_variant_file(Path(args.variant_file)) if args.variant_file else list(DEFAULT_VARIANTS)
    (root / "sweep_variants.json").write_text(json.dumps(variants, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for variant in variants:
        sep_dir, asr_dir = variant_dirs(root, variant)
        run_separation(args, variant, sep_dir, log_dir)
        run_asr(args, variant, sep_dir, asr_dir, log_dir)
        if not args.skip_spkid:
            run_spkid(args, variant, sep_dir, asr_dir, log_dir)
    run_monitor(root, variants, args)
    cleanup_runtime()
    print(f"[done] {root / 'monitor' / 'separatio_asr_monitor.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())