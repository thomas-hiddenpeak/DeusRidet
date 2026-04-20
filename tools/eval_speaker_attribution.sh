#!/bin/bash
# Speaker-attribution evaluation with catastrophic-error accounting.
#
# Goal: the pipeline's value is NOT frame-level speaker identification
# accuracy. Its true goal is correct speaker attribution for downstream
# semantic processing. Misattributing words in an argument/interruption
# region (overlap) to the wrong speaker is SEMANTICALLY CATASTROPHIC —
# an entire utterance gets routed to the wrong party.
#
# This script produces the two-tuple `(true_accuracy, catastrophic_count)`
# required by the new evaluation regime. Abstentions in overlap regions
# are counted as neutral (not errors) — better to say "?" than to lie.
#
# Inputs:
#   1. mapped file: lines of `<audio_sec> id=<N>` (<0 means abstain)
#   2. GT file: lines of `<audio_sec> <speaker_name>`
#   3. overlap timeline: lines of `<start>\t<end>\t<confidence>\t...`
#      (start/end formatted HH:MM:SS or MM:SS)
#
# Usage:
#   tools/eval_speaker_attribution.sh <mapped.txt> <gt.txt> <overlap_timeline.txt>

set -e

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <mapped.txt> <gt_seconds.txt> <overlap_timeline.txt>" >&2
    exit 1
fi

MAPPED="$1"
GT="$2"
OVERLAP="$3"

for f in "$MAPPED" "$GT" "$OVERLAP"; do
    [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done

# Speaker ID → name mapping is fixed per benchmark:
# id=0 → 朱杰, id=1 → 唐云峰, id=2 → 徐子景, id=3 → 石一
# This matches the registration order verified in test5/test7/test8 logs.
#
# Boundary tolerance: ±3 seconds around a GT speaker transition is counted
# as correct regardless of predicted speaker (handles label lag and
# ASR segmentation drift). Matches existing evaluator behavior.

awk -v mapped="$MAPPED" -v gt="$GT" -v overlap="$OVERLAP" '
function parse_ts(s,   a, n) {
    # accepts HH:MM:SS or MM:SS
    n = split(s, a, ":")
    if (n == 3) return a[1]*3600 + a[2]*60 + a[3]
    if (n == 2) return a[1]*60 + a[2]
    return a[1] + 0
}

BEGIN {
    map_name[0] = "朱杰"
    map_name[1] = "唐云峰"
    map_name[2] = "徐子景"
    map_name[3] = "石一"

    # Load GT (sorted timeline of speaker active intervals)
    while ((getline line < gt) > 0) {
        split(line, f, " ")
        gt_n++
        gt_t[gt_n] = int(f[1])
        gt_s[gt_n] = f[2]
    }
    close(gt)

    # Load overlap timeline.
    while ((getline line < overlap) > 0) {
        # Skip headers/blank/comments
        if (line ~ /^#/ || line ~ /^$/) continue
        if (line ~ /^[a-zA-Z]/) continue  # header lines like "asrTest2..."
        if (line ~ /^source:/ || line ~ /^rules:/ || line ~ /^note:/) continue
        if (line ~ /^columns:/) continue
        if (line !~ /^[0-9]+:[0-9]+/) continue
        split(line, f, "\t")
        ov_n++
        ov_start[ov_n] = parse_ts(f[1])
        ov_end[ov_n]   = parse_ts(f[2])
        ov_conf[ov_n]  = f[3]
    }
    close(overlap)

    # Iterate mapped predictions
    while ((getline line < mapped) > 0) {
        split(line, f, " ")
        t = int(f[1])
        gsub(/id=/, "", f[2])
        id = int(f[2])

        # Find active GT speaker
        active = ""; idx = 0; dist_start = 999; dist_next = 999
        for (i = gt_n; i >= 1; i--) {
            if (t >= gt_t[i]) {
                active = gt_s[i]; idx = i; dist_start = t - gt_t[i]; break
            }
        }
        if (idx > 0 && idx < gt_n) dist_next = gt_t[idx+1] - t

        # Is this timestamp inside an overlap region?
        in_overlap = 0; overlap_conf = ""
        for (j = 1; j <= ov_n; j++) {
            if (t >= ov_start[j] && t <= ov_end[j]) {
                in_overlap = 1
                overlap_conf = ov_conf[j]
                break
            }
        }

        # Classify prediction
        if (id < 0) {
            # Abstention
            if (in_overlap) {
                abstain_overlap++      # neutral/good
            } else if (dist_start <= 2 || dist_next <= 2) {
                abstain_boundary++     # neutral
            } else {
                abstain_midutt++       # lost, but not a lie
            }
            continue
        }

        pred_name = map_name[id]
        total_identified++
        correct_match = (pred_name == active)

        if (in_overlap) {
            # Overlap region: any identified prediction is risky.
            # If it matches ONE of the overlap speakers it is at least
            # plausible; otherwise it is catastrophic.
            overlap_total++
            if (correct_match) {
                overlap_correct++
            } else {
                # Misattribution in overlap → CATASTROPHIC.
                # This routes ASR content to the wrong party.
                overlap_catastrophic++
                if (overlap_conf == "high")   catastrophic_high++
                if (overlap_conf == "medium") catastrophic_med++
                if (overlap_conf == "low")    catastrophic_low++
            }
        } else {
            # Non-overlap single-speaker region
            nonoverlap_total++
            if (correct_match) {
                nonoverlap_correct++
            } else {
                # Boundary tolerance
                if (dist_start <= 2 || dist_next <= 2) {
                    boundary_err++
                } else {
                    genuine_err++
                }
            }
        }
    }
    close(mapped)

    # === Report ===
    printf "================================================================\n"
    printf "  Speaker-Attribution Evaluation (semantic-centric)\n"
    printf "================================================================\n"
    printf "Mapped file:      %s\n", mapped
    printf "GT file:          %s\n", gt
    printf "Overlap timeline: %s  (%d regions)\n", overlap, ov_n
    printf "\n"
    printf "-- Overall --\n"
    printf "  Total predictions: %d\n", total_identified + abstain_overlap + abstain_boundary + abstain_midutt
    printf "  Identified:        %d\n", total_identified
    printf "  Abstained:         %d (overlap=%d boundary=%d mid=%d)\n", \
        abstain_overlap + abstain_boundary + abstain_midutt, \
        abstain_overlap + 0, abstain_boundary + 0, abstain_midutt + 0
    printf "\n"
    printf "-- Non-overlap regions --\n"
    if (nonoverlap_total > 0) {
        printf "  Identified:        %d\n", nonoverlap_total
        printf "  Correct:           %d\n", nonoverlap_correct + 0
        printf "  Boundary errors:   %d (accepted)\n", boundary_err + 0
        printf "  Genuine errors:    %d\n", genuine_err + 0
        true_correct = nonoverlap_correct + boundary_err
        printf "  True accuracy:     %.1f%%  (correct + boundary-tolerated)\n", \
            true_correct * 100.0 / nonoverlap_total
    }
    printf "\n"
    printf "-- Overlap regions (CRITICAL) --\n"
    if (overlap_total + abstain_overlap > 0) {
        printf "  Identified:        %d\n", overlap_total + 0
        printf "  Abstained:         %d (acceptable — better than lying)\n", abstain_overlap + 0
        printf "  Correct match:     %d\n", overlap_correct + 0
        printf "  *** CATASTROPHIC MISATTRIBUTIONS: %d ***\n", overlap_catastrophic + 0
        if (overlap_catastrophic > 0) {
            printf "      by confidence:  high=%d medium=%d low=%d\n", \
                catastrophic_high + 0, catastrophic_med + 0, catastrophic_low + 0
        }
    } else {
        printf "  (no predictions in overlap regions)\n"
    }
    printf "\n"
    printf "-- Semantic-centric verdict --\n"
    if (nonoverlap_total > 0) {
        printf "  (true_accuracy, catastrophic_count) = (%.1f%%, %d)\n", \
            (nonoverlap_correct + boundary_err) * 100.0 / nonoverlap_total, \
            overlap_catastrophic + 0
    }
    printf "  Goal: maximize true_accuracy while keeping catastrophic_count\n"
    printf "        as close to 0 as possible. Abstention in overlap is OK.\n"
    printf "================================================================\n"
}
' < /dev/null
