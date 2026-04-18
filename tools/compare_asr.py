#!/usr/bin/env python3
"""Compare ASR pipeline output against reference transcript.

Usage:
    python3 tools/compare_asr.py /tmp/asr_test_output.txt tests/asrTest2Final.txt
"""

import re
import sys
from collections import defaultdict


def parse_asr_output(path):
    """Parse ASR output lines into structured records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Match: ASR [spkX] (t0-t1s sim=... conf=... src=... trigger=...) text
            m = re.match(
                r'ASR \[([^\]]+)\] \((\d+\.?\d*)-(\d+\.?\d*)s '
                r'sim=(\d+\.?\d*) conf=(\d+\.?\d*) '
                r'src=(\S+) trigger=(\S+)\) (.+)',
                line
            )
            if m:
                records.append({
                    'spk': m.group(1),
                    't0': float(m.group(2)),
                    't1': float(m.group(3)),
                    'sim': float(m.group(4)),
                    'conf': float(m.group(5)),
                    'src': m.group(6),
                    'trigger': m.group(7),
                    'text': m.group(8),
                })
    return records


def parse_reference(path):
    """Parse reference transcript into structured records.
    Format:
        HH:MM:SS speaker_name
        text content...
    """
    records = []
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'(\d{2}):(\d{2}):(\d{2})\s+(.+)', line)
        if m:
            h, mi, s, spk = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4).strip()
            t = h * 3600 + mi * 60 + s
            # Collect text lines until next timestamp
            text_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if re.match(r'\d{2}:\d{2}:\d{2}\s+', next_line):
                    break
                if next_line:
                    text_lines.append(next_line)
                i += 1
            text = ''.join(text_lines)
            if text:
                records.append({'t': t, 'spk': spk, 'text': text})
        else:
            i += 1
    return records


def cer(ref_text, hyp_text):
    """Character Error Rate using edit distance."""
    ref = list(ref_text.replace(' ', '').replace('，', '').replace('。', '').replace('？', '').replace('！', '').replace('、', ''))
    hyp = list(hyp_text.replace(' ', '').replace('，', '').replace('。', '').replace('？', '').replace('！', '').replace('、', ''))

    n = len(ref)
    m = len(hyp)
    if n == 0:
        return 1.0 if m > 0 else 0.0

    # DP edit distance
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)

    return d[n][m] / n


def map_speakers(asr_records, ref_records):
    """Map ASR speaker IDs to reference names by time overlap."""
    # For each ASR record, find overlapping reference record
    spk_votes = defaultdict(lambda: defaultdict(float))

    for asr in asr_records:
        asr_mid = (asr['t0'] + asr['t1']) / 2
        best_ref = None
        best_dist = float('inf')
        for ref in ref_records:
            dist = abs(ref['t'] - asr_mid)
            if dist < best_dist:
                best_dist = dist
                best_ref = ref
        if best_ref and best_dist < 15:  # within 15 seconds
            duration = asr['t1'] - asr['t0']
            spk_votes[asr['spk']][best_ref['spk']] += duration

    # Greedy assignment
    mapping = {}
    used_refs = set()
    # Sort by total voting weight
    sorted_spks = sorted(spk_votes.keys(),
                        key=lambda s: max(spk_votes[s].values()) if spk_votes[s] else 0,
                        reverse=True)

    for spk in sorted_spks:
        votes = spk_votes[spk]
        for ref_name in sorted(votes, key=votes.get, reverse=True):
            if ref_name not in used_refs:
                mapping[spk] = ref_name
                used_refs.add(ref_name)
                break

    return mapping, spk_votes


def align_segments(asr_records, ref_records):
    """Align ASR segments to reference segments by time proximity."""
    # Merge consecutive ASR segments from same speaker within a reference window
    alignments = []

    for ref in ref_records:
        ref_t = ref['t']
        # Find next reference timestamp for end bound
        ref_idx = ref_records.index(ref)
        if ref_idx + 1 < len(ref_records):
            ref_end = ref_records[ref_idx + 1]['t']
        else:
            ref_end = ref_t + 30

        # Collect ASR segments in this window
        matched = []
        for asr in asr_records:
            if asr['t0'] >= ref_t - 2 and asr['t0'] < ref_end + 2:
                matched.append(asr)

        if matched:
            merged_text = ''.join(a['text'] for a in matched)
            alignments.append({
                'ref': ref,
                'asr_segments': matched,
                'merged_text': merged_text,
            })
        else:
            alignments.append({
                'ref': ref,
                'asr_segments': [],
                'merged_text': '',
            })

    return alignments


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <asr_output> <reference>")
        sys.exit(1)

    asr_path, ref_path = sys.argv[1], sys.argv[2]

    print("Parsing ASR output...")
    asr_records = parse_asr_output(asr_path)
    print(f"  {len(asr_records)} ASR segments")

    print("Parsing reference...")
    ref_records = parse_reference(ref_path)
    print(f"  {len(ref_records)} reference segments")

    # Speaker mapping
    print("\n" + "=" * 70)
    print("SPEAKER MAPPING")
    print("=" * 70)
    mapping, votes = map_speakers(asr_records, ref_records)
    for spk, ref_name in sorted(mapping.items()):
        vote_str = ', '.join(f"{n}={v:.1f}s" for n, v in
                            sorted(votes[spk].items(), key=lambda x: -x[1])[:3])
        count = sum(1 for a in asr_records if a['spk'] == spk)
        print(f"  {spk:>8} → {ref_name:<8} ({count} segments, votes: {vote_str})")

    # Show unmapped
    unmapped = set(a['spk'] for a in asr_records) - set(mapping.keys())
    for spk in unmapped:
        count = sum(1 for a in asr_records if a['spk'] == spk)
        print(f"  {spk:>8} → ???      ({count} segments, unmapped)")

    # Speaker accuracy per reference segment
    print("\n" + "=" * 70)
    print("SPEAKER IDENTIFICATION ACCURACY")
    print("=" * 70)
    reverse_map = {v: k for k, v in mapping.items()}

    correct_spk = 0
    total_spk = 0
    spk_accuracy_per_person = defaultdict(lambda: [0, 0])  # [correct, total]

    for ref in ref_records:
        ref_t = ref['t']
        expected_spk = reverse_map.get(ref['spk'])
        if not expected_spk:
            continue

        # Find ASR segments near this reference time
        near = [a for a in asr_records if abs(a['t0'] - ref_t) < 10 or abs((a['t0']+a['t1'])/2 - ref_t) < 10]
        if near:
            # Pick the closest by start time
            closest = min(near, key=lambda a: abs(a['t0'] - ref_t))
            total_spk += 1
            spk_accuracy_per_person[ref['spk']][1] += 1
            if closest['spk'] == expected_spk:
                correct_spk += 1
                spk_accuracy_per_person[ref['spk']][0] += 1

    if total_spk > 0:
        print(f"  Overall: {correct_spk}/{total_spk} = {100*correct_spk/total_spk:.1f}%")
        for name, (c, t) in sorted(spk_accuracy_per_person.items()):
            print(f"  {name:<8}: {c}/{t} = {100*c/t:.1f}%")

    # ASR text quality - sample comparison
    print("\n" + "=" * 70)
    print("ASR TEXT QUALITY (sample segments)")
    print("=" * 70)

    total_cer = 0
    cer_count = 0
    cer_per_speaker = defaultdict(lambda: [0.0, 0])

    # For each reference segment, find temporally closest ASR and compute CER
    for ref in ref_records:
        ref_t = ref['t']
        # Find ASR segments within a window
        ref_idx = ref_records.index(ref)
        if ref_idx + 1 < len(ref_records):
            window_end = ref_records[ref_idx + 1]['t']
        else:
            window_end = ref_t + 30

        matched = [a for a in asr_records if a['t0'] >= ref_t - 3 and a['t1'] <= window_end + 3]
        if not matched:
            continue

        hyp_text = ''.join(a['text'] for a in matched)
        segment_cer = cer(ref['text'], hyp_text)
        total_cer += segment_cer
        cer_count += 1
        cer_per_speaker[ref['spk']][0] += segment_cer
        cer_per_speaker[ref['spk']][1] += 1

    if cer_count > 0:
        print(f"  Overall CER: {100*total_cer/cer_count:.1f}% ({cer_count} aligned segments)")
        for name, (c, t) in sorted(cer_per_speaker.items()):
            print(f"  {name:<8} CER: {100*c/t:.1f}% ({t} segments)")

    # Show first 20 aligned examples
    print("\n" + "=" * 70)
    print("ALIGNMENT EXAMPLES (first 20)")
    print("=" * 70)
    shown = 0
    for ref in ref_records[:40]:
        ref_t = ref['t']
        ref_idx = ref_records.index(ref)
        if ref_idx + 1 < len(ref_records):
            window_end = ref_records[ref_idx + 1]['t']
        else:
            window_end = ref_t + 30

        matched = [a for a in asr_records if a['t0'] >= ref_t - 3 and a['t1'] <= window_end + 3]
        if not matched:
            continue

        hyp_text = ''.join(a['text'] for a in matched)
        spk_ids = set(a['spk'] for a in matched)
        mapped_names = [mapping.get(s, s) for s in spk_ids]

        mm = int(ref_t // 60)
        ss = int(ref_t % 60)
        segment_cer = cer(ref['text'], hyp_text)

        print(f"\n  [{mm:02d}:{ss:02d}] REF ({ref['spk']}): {ref['text'][:80]}")
        print(f"         ASR ({','.join(mapped_names)}): {hyp_text[:80]}")
        print(f"         CER: {100*segment_cer:.0f}%")
        shown += 1
        if shown >= 20:
            break

    # Overall summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Audio duration:     ~60 min")
    print(f"  ASR segments:       {len(asr_records)}")
    print(f"  Reference segments: {len(ref_records)}")
    print(f"  Speakers detected:  {len(set(a['spk'] for a in asr_records))}")
    print(f"  Speakers expected:  {len(set(r['spk'] for r in ref_records))}")
    if cer_count > 0:
        print(f"  Average CER:        {100*total_cer/cer_count:.1f}%")
    if total_spk > 0:
        print(f"  Speaker accuracy:   {100*correct_spk/total_spk:.1f}%")

    # Time coverage
    if asr_records:
        asr_total_dur = sum(a['t1'] - a['t0'] for a in asr_records)
        print(f"  ASR total speech:   {asr_total_dur:.0f}s ({asr_total_dur/60:.1f} min)")
        print(f"  ASR time range:     {asr_records[0]['t0']:.1f}s - {asr_records[-1]['t1']:.1f}s")


if __name__ == '__main__':
    main()
