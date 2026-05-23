#!/bin/bash
# Step 19c — multi-gate sweep harness.
# Usage: bash tools/run_multi_gate_sweep.sh <tag> <enable> <threshold>
set -e
cd /home/rm01/DeusRidet

TAG="${1:-baseline}"
ENABLE="${2:-0}"
THRESH="${3:-0.58}"
OUT_DIR="/tmp/replay_step19c_${TAG}"

echo "[sweep] tag=${TAG} enable=${ENABLE} thresh=${THRESH} out=${OUT_DIR}"

# Patch config
python3 - <<PY
import re
p='configs/auditus.conf'
s=open(p).read()
s=re.sub(r'^speaker_multi_gate_enable\s*=.*$',     f'speaker_multi_gate_enable         = ${ENABLE}', s, flags=re.M)
s=re.sub(r'^speaker_multi_gate_threshold\s*=.*$',  f'speaker_multi_gate_threshold      = ${THRESH}', s, flags=re.M)
open(p,'w').write(s)
print('[cfg] patched')
PY

# Verify state
grep -E '^speaker_multi_gate' configs/auditus.conf

# Kill + clean
sudo kill -9 $(pgrep -f deusridet) 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true
sleep 1
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
rm -f /tmp/spk_embeddings.bin

# Start awaken in background
rm -rf "$OUT_DIR"
LOG_FILE="/tmp/awaken_${TAG}.log"
DEUSRIDET_TEST_WS_ENABLE_ASR=1 ./build/deusridet awaken > "$LOG_FILE" 2>&1 &
AWAKEN_PID=$!
echo "[awaken] pid=$AWAKEN_PID log=$LOG_FILE"

# Wait for HTTP ready
for i in $(seq 1 60); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/ 2>/dev/null | grep -q 200; then
        echo "[awaken] ready after ${i}s"
        break
    fi
    sleep 1
done

# Replay (10-min slice @ 1x to match 19b baseline)
python3 tools/online_replay_score.py \
    --audio tests/test.mp3 \
    --gt tests/fixtures/test_ground_truth_v1.jsonl \
    --speed 1.0 --drain-sec 60 --max-sec 600 \
    --out-dir "$OUT_DIR" \
    --enable-asr 2>&1 | tee "$OUT_DIR.replay.log" | tail -30

# Kill awaken
sudo kill -9 $AWAKEN_PID 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true

# Summary
echo "--- summary ${TAG} ---"
python3 -c "import json; s=json.load(open('${OUT_DIR}/summary.json')); print(json.dumps({k:s[k] for k in ['n_gt','n_decided','n_abstain','n_no_seg','coverage','macro','decided_macro','per_spk','per_spk_decided']}, indent=2, ensure_ascii=False))"
