#!/usr/bin/env python3
"""Evaluate speaker identification accuracy against human-calibrated ground truth.

Compares system ASR output (with speaker labels) against ground truth transcript.
Uses time-overlap matching: for each system segment, find the ground truth speaker
that owns that time range, then check if the system's speaker label is correct.

Usage:
    python3 eval_speaker_accuracy.py <ground_truth.txt> <system_output.txt>
"""

import re
import sys
from collections import defaultdict, Counter


def parse_ground_truth(path):
    """Parse ground truth file: 'HH:MM:SS speaker_name\\ntext...'
    Returns list of (start_sec, speaker_name, text) sorted by time.
    """
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^(\d{2}):(\d{2}):(\d{2})\s+(.+)$', line)
        if m:
            h, mi, s, speaker = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4).strip()
            t = h * 3600 + mi * 60 + s
            # Next line(s) are the text until next timestamp
            text_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if re.match(r'^\d{2}:\d{2}:\d{2}\s+', next_line):
                    break
                if next_line:
                    text_lines.append(next_line)
                i += 1
            entries.append((t, speaker, ' '.join(text_lines)))
        else:
            i += 1

    return entries


def gt_to_intervals(entries):
    """Convert GT entries to intervals: (start, end, speaker).
    End time = start of next entry (or +30s for last).
    """
    intervals = []
    for i, (t, spk, text) in enumerate(entries):
        if i + 1 < len(entries):
            end = entries[i + 1][0]
        else:
            end = t + 30
        intervals.append((t, end, spk))
    return intervals


def parse_system_output(path):
    """Parse system ASR output lines.
    Format: ASR [spkN] (start-ends sim=... ...) text
    Returns list of (start_sec, end_sec, spk_id, text).
    """
    segments = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r'ASR \[(spk\d+)\] \((\d+\.?\d*)-(\d+\.?\d*)s', line)
            if m:
                spk = m.group(1)
                start = float(m.group(2))
                end = float(m.group(3))
                # Extract text after the closing paren
                text_m = re.search(r'\)\s*(.+)$', line)
                text = text_m.group(1).strip() if text_m else ''
                segments.append((start, end, spk, text))
    return segments


def find_gt_speaker(gt_intervals, t_start, t_end):
    """Find ground truth speaker for a given time range.
    Uses midpoint matching against GT intervals.
    """
    mid = (t_start + t_end) / 2.0
    for gt_start, gt_end, spk in gt_intervals:
        if gt_start <= mid < gt_end:
            return spk
    # Fallback: closest GT entry by start time
    best = None
    best_dist = float('inf')
    for gt_start, gt_end, spk in gt_intervals:
        dist = abs(gt_start - mid)
        if dist < best_dist:
            best_dist = dist
            best = spk
    return best if best_dist < 10 else None


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ground_truth.txt> <system_output.txt>")
        sys.exit(1)

    gt_path = sys.argv[1]
    sys_path = sys.argv[2]

    gt_entries = parse_ground_truth(gt_path)
    gt_intervals = gt_to_intervals(gt_entries)
    sys_segments = parse_system_output(sys_path)

    print(f"Ground truth: {len(gt_entries)} entries, {len(set(s for _,s,_ in gt_entries))} speakers")
    print(f"  Speakers: {dict(Counter(s for _,s,_ in gt_entries))}")
    print(f"  Time range: {gt_entries[0][0]}s - {gt_entries[-1][0]}s")
    print(f"System output: {len(sys_segments)} segments")
    print(f"  Time range: {sys_segments[0][0]}s - {sys_segments[-1][1]}s")
    print()

    # Map system spk IDs to GT names by majority vote
    spk_gt_counts = defaultdict(lambda: Counter())  # spk_id -> Counter(gt_name)
    matched = []

    for start, end, spk_id, text in sys_segments:
        gt_spk = find_gt_speaker(gt_intervals, start, end)
        if gt_spk:
            spk_gt_counts[spk_id][gt_spk] += 1
            matched.append((start, end, spk_id, gt_spk, text))

    # Build mapping: each system spk -> most frequent GT name
    spk_mapping = {}
    print("=" * 60)
    print("Speaker Mapping (system → ground truth, by majority vote):")
    print("=" * 60)
    for spk_id in sorted(spk_gt_counts.keys()):
        counts = spk_gt_counts[spk_id]
        total = sum(counts.values())
        best_gt, best_count = counts.most_common(1)[0]
        purity = best_count / total * 100
        spk_mapping[spk_id] = best_gt
        print(f"  {spk_id} → {best_gt}  ({best_count}/{total} = {purity:.1f}% purity)")
        if len(counts) > 1:
            for gt_name, cnt in counts.most_common():
                if gt_name != best_gt:
                    print(f"         also matched: {gt_name} ({cnt} times)")
    print()

    # Check for GT speakers mapped to multiple system IDs
    gt_to_sys = defaultdict(list)
    for sys_id, gt_name in spk_mapping.items():
        gt_to_sys[gt_name].append(sys_id)
    
    print("=" * 60)
    print("Reverse Mapping (ground truth → system):")
    print("=" * 60)
    for gt_name in sorted(gt_to_sys.keys()):
        sys_ids = gt_to_sys[gt_name]
        print(f"  {gt_name} → {', '.join(sys_ids)}")
    print()

    # Calculate accuracy
    correct = 0
    wrong = 0
    confusion = defaultdict(lambda: Counter())  # (gt_name) -> Counter(predicted_gt_name)

    for start, end, spk_id, gt_spk, text in matched:
        predicted_gt = spk_mapping[spk_id]
        if predicted_gt == gt_spk:
            correct += 1
        else:
            wrong += 1
            confusion[gt_spk][predicted_gt] += 1

    total = correct + wrong
    accuracy = correct / total * 100 if total > 0 else 0

    print("=" * 60)
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print("=" * 60)
    print()

    # Per-speaker accuracy
    print("Per-Speaker Accuracy:")
    print("-" * 60)
    per_spk = defaultdict(lambda: {'correct': 0, 'total': 0})
    for start, end, spk_id, gt_spk, text in matched:
        predicted_gt = spk_mapping[spk_id]
        per_spk[gt_spk]['total'] += 1
        if predicted_gt == gt_spk:
            per_spk[gt_spk]['correct'] += 1

    for gt_name in sorted(per_spk.keys()):
        d = per_spk[gt_name]
        acc = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
        print(f"  {gt_name}: {d['correct']}/{d['total']} = {acc:.1f}%")
        if gt_name in confusion:
            for wrong_name, cnt in confusion[gt_name].most_common():
                print(f"    → misclassified as {wrong_name}: {cnt} times")
    print()

    # Duration-weighted accuracy
    correct_dur = 0.0
    total_dur = 0.0
    for start, end, spk_id, gt_spk, text in matched:
        dur = end - start
        total_dur += dur
        if spk_mapping[spk_id] == gt_spk:
            correct_dur += dur

    dur_acc = correct_dur / total_dur * 100 if total_dur > 0 else 0
    print(f"Duration-Weighted Accuracy: {correct_dur:.1f}s / {total_dur:.1f}s = {dur_acc:.1f}%")
    print()

    # Show some misclassified examples
    print("=" * 60)
    print("Sample Misclassifications (first 20):")
    print("=" * 60)
    mis_count = 0
    for start, end, spk_id, gt_spk, text in matched:
        predicted_gt = spk_mapping[spk_id]
        if predicted_gt != gt_spk:
            mis_count += 1
            if mis_count <= 20:
                print(f"  {start:.1f}-{end:.1f}s: sys={spk_id}→{predicted_gt}, GT={gt_spk}")
                print(f"    \"{text[:60]}\"")

    # ---- Turn-level accuracy ----
    # For each GT turn (interval), collect all system segments that overlap,
    # find the majority system speaker label, and check if it maps correctly.
    # This measures "can the system correctly identify who is speaking for
    # each distinct speaking turn?" — more relevant than per-segment accuracy.
    print()
    print("=" * 60)
    print("Turn-Level Accuracy (majority system label per GT turn):")
    print("=" * 60)

    turn_correct = 0
    turn_total = 0
    turn_per_spk = defaultdict(lambda: {'correct': 0, 'total': 0})
    turn_consistency_scores = []

    for gt_start, gt_end, gt_spk in gt_intervals:
        gt_dur = gt_end - gt_start
        if gt_dur < 2:  # skip very short turns (< 2s)
            continue

        # Find system segments overlapping this GT turn.
        sys_labels = Counter()
        for s_start, s_end, s_spk, s_text in sys_segments:
            ov_start = max(s_start, gt_start)
            ov_end = min(s_end, gt_end)
            if ov_end > ov_start:
                ov_dur = ov_end - ov_start
                sys_labels[s_spk] += ov_dur

        if not sys_labels:
            continue

        turn_total += 1
        turn_per_spk[gt_spk]['total'] += 1

        # Majority system label for this turn.
        majority_sys, majority_dur = sys_labels.most_common(1)[0]
        total_dur_in_turn = sum(sys_labels.values())

        # Consistency: fraction of duration covered by majority label.
        consistency = majority_dur / total_dur_in_turn if total_dur_in_turn > 0 else 0
        turn_consistency_scores.append(consistency)

        predicted_gt = spk_mapping.get(majority_sys, None)
        if predicted_gt == gt_spk:
            turn_correct += 1
            turn_per_spk[gt_spk]['correct'] += 1

    if turn_total > 0:
        turn_acc = turn_correct / turn_total * 100
        avg_consistency = sum(turn_consistency_scores) / len(turn_consistency_scores) * 100
        print(f"  Turns evaluated: {turn_total} (>= 2s duration)")
        print(f"  Turn-level accuracy: {turn_correct}/{turn_total} = {turn_acc:.1f}%")
        print(f"  Avg label consistency within turns: {avg_consistency:.1f}%")
        print()
        for gt_name in sorted(turn_per_spk.keys()):
            d = turn_per_spk[gt_name]
            acc = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
            print(f"    {gt_name}: {d['correct']}/{d['total']} = {acc:.1f}%")


if __name__ == '__main__':
    main()
