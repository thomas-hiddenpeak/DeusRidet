#!/usr/bin/env bash
# Baseline cell: reclusterer disabled. Captures raw_id behaviour end-to-end.
# Usage: tools/run_recluster_off.sh <cell_name>
set -euo pipefail
CELL="${1:?cell name required, e.g. OFF_baseline}"
ROOT="/home/rm01/DeusRidet"
OUT="$ROOT/runs/matrix_v1/$CELL"
AWAKEN_LOG="/tmp/awaken_${CELL}.log"
BINARY="$ROOT/build/deusridet"
cd "$ROOT"
if [ -e "$OUT" ]; then echo "exists" >&2; exit 2; fi
mkdir -p "$OUT"
sudo kill -9 $(pgrep -f deusridet) 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
sleep 1
COMMIT=$(git rev-parse HEAD)
BIN_SHA=$(sha256sum "$BINARY" | awk '{print $1}')
START_TS=$(date -u +%FT%TZ); START_EPOCH=$(date +%s)
nohup env DEUSRIDET_RECLUSTERER_ENABLE=0 "$BINARY" awaken > "$AWAKEN_LOG" 2>&1 &
PID=$!; disown $PID
for i in $(seq 1 60); do
  sleep 1
  curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/ | grep -q 200 && break
done
python3 tools/replay_to_transcript.py --audio tests/test.mp3 --gt tests/fixtures/test_ground_truth.json --max-sec 3615 --speed 2.0 --drain-sec 60 --out-dir "$OUT" > "$OUT/runner.log" 2>&1
REPLAY_EXIT=$?
END_TS=$(date -u +%FT%TZ); END_EPOCH=$(date +%s); WALL=$((END_EPOCH-START_EPOCH))
kill $PID 2>/dev/null || true; sleep 2
pgrep -af deusridet >/dev/null && sudo kill -9 $(pgrep -f deusridet) || true
cp "$AWAKEN_LOG" "$OUT/awaken.log"
OVF=$(grep -c "Ring buffer overflow" "$AWAKEN_LOG" || true)
cat > "$OUT/manifest.json" <<JSON
{"cell":"$CELL","matrix":"v1","commit":"$COMMIT","binary_sha256":"$BIN_SHA","env":{"DEUSRIDET_RECLUSTERER_ENABLE":"0"},"fixed":{"audio":"tests/test.mp3","speed":2.0,"max_sec":3615},"start_ts":"$START_TS","end_ts":"$END_TS","wall_sec":$WALL,"replay_exit":$REPLAY_EXIT,"ring_buffer_overflows":$OVF}
JSON
echo "=== cell $CELL done ==="
