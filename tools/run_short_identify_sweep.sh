#!/bin/bash
# Step 19d — SHORT-IDENTIFY × MULTI-GATE combined sweep.
# Usage: bash tools/run_short_identify_sweep.sh <tag> <si_enable> <si_thresh> <si_margin> [min_fbank_ident] [max_sec]
# MULTI-GATE stays at its default (enable=1 thr=0.58 min_fb=250).
# max_sec (default 600) controls how much of tests/test.mp3 is replayed —
# Step 22+ uses 1800 s (3x baseline, ~3× decision points) to drop
# run-to-run variance on coverage from ~0.055 to a usable signal floor.
set -e
cd /home/rm01/DeusRidet

TAG="${1:-baseline}"
SI_ENABLE="${2:-0}"
SI_THRESH="${3:-0.55}"
SI_MARGIN="${4:-0.05}"
SI_MIN_FB="${5:-50}"
MAX_SEC="${6:-600}"
OUT_DIR="/tmp/replay_step19d_${TAG}"

echo "[sweep] tag=${TAG} si_enable=${SI_ENABLE} si_thresh=${SI_THRESH} si_margin=${SI_MARGIN} si_min_fb=${SI_MIN_FB} max_sec=${MAX_SEC} out=${OUT_DIR}"

python3 - <<PY
import re
p='configs/auditus.conf'
s=open(p).read()
s=re.sub(r'^speaker_short_identify_enable\s*=.*$',     f'speaker_short_identify_enable     = ${SI_ENABLE}', s, flags=re.M)
s=re.sub(r'^speaker_short_identify_threshold\s*=.*$',  f'speaker_short_identify_threshold  = ${SI_THRESH}', s, flags=re.M)
s=re.sub(r'^speaker_short_identify_margin\s*=.*$',     f'speaker_short_identify_margin     = ${SI_MARGIN}', s, flags=re.M)
s=re.sub(r'^speaker_min_fbank_frames_identify\s*=.*$', f'speaker_min_fbank_frames_identify = ${SI_MIN_FB}', s, flags=re.M)
open(p,'w').write(s)
print('[cfg] patched')
PY

grep -E '^speaker_short_identify|^speaker_min_fbank_frames_identify|^speaker_multi_gate' configs/auditus.conf

sudo kill -9 $(pgrep -f deusridet) 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true
sleep 1
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
rm -f /tmp/spk_embeddings.bin

rm -rf "$OUT_DIR"
LOG_FILE="/tmp/awaken_19d_${TAG}.log"
DEUSRIDET_TEST_WS_ENABLE_ASR=1 ./build/deusridet awaken > "$LOG_FILE" 2>&1 &
AWAKEN_PID=$!
echo "[awaken] pid=$AWAKEN_PID log=$LOG_FILE"

for i in $(seq 1 60); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/ 2>/dev/null | grep -q 200; then
        echo "[awaken] ready after ${i}s"
        break
    fi
    sleep 1
done

python3 tools/online_replay_score.py \
    --audio tests/test.mp3 \
    --gt tests/fixtures/test_ground_truth_v1.jsonl \
    --speed 1.0 --drain-sec 60 --max-sec "${MAX_SEC}" \
    --out-dir "$OUT_DIR" \
    --enable-asr 2>&1 | tee "$OUT_DIR.replay.log" | tail -30

sudo kill -9 $AWAKEN_PID 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true

echo "--- summary ${TAG} ---"
python3 -c "import json; s=json.load(open('${OUT_DIR}/summary.json')); print(json.dumps({k:s[k] for k in ['n_gt','n_decided','n_abstain','n_no_seg','coverage','macro','decided_macro','per_spk','per_spk_decided']}, indent=2, ensure_ascii=False))"
echo "--- SI events ---"
grep -c "SHORT-IDENTIFY match" "$LOG_FILE" 2>/dev/null || true
echo "--- MULTI-GATE events ---"
grep -c "MULTI-GATE flagged" "$LOG_FILE" 2>/dev/null || true
