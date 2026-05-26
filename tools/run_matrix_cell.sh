#!/usr/bin/env bash
# Run one cell of matrix_v1.
# Usage: tools/run_matrix_cell.sh <cell_name> <link_thresh> <accept_abstained>
# Example: tools/run_matrix_cell.sh L060_A1 0.60 1
#
# Fixed (matrix_v1 invariants):
#   audio = tests/test.mp3
#   gt    = tests/fixtures/test_ground_truth.json
#   speed = 2.0   drain = 60s   max-sec = 3615
#   W     = 180s  (config default, not tunable via env yet)
#   S     = 30s   ema = 0.20    K ∈ [2,6]   max_globals default
#
# Output: runs/matrix_v1/<cell_name>/{runner.log, awaken.log, runtime_segments.json,
#   relabel_log.json, forward_map.json, transcript.md, asr_events.json, manifest.json}

set -euo pipefail

CELL="${1:?cell name required, e.g. L060_A1}"
LINK="${2:?link threshold required, e.g. 0.60}"
ACCEPT="${3:?accept_abstained 0 or 1 required}"

ROOT="/home/rm01/DeusRidet"
OUT="$ROOT/runs/matrix_v1/$CELL"
AWAKEN_LOG="/tmp/awaken_matrix_${CELL}.log"
BINARY="$ROOT/build/deusridet"

cd "$ROOT"

if [ -e "$OUT" ]; then
    echo "ERROR: $OUT already exists; refusing to overwrite" >&2
    exit 2
fi
mkdir -p "$OUT"

# --- clean-state ritual (mandatory) ---
sudo kill -9 $(pgrep -f deusridet) 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
sleep 1
if pgrep -af deusridet >/dev/null; then
    echo "ERROR: deusridet still running after kill" >&2
    exit 3
fi

# --- start awaken ---
COMMIT=$(git rev-parse HEAD)
BIN_SHA=$(sha256sum "$BINARY" | awk '{print $1}')
START_TS=$(date -u +%FT%TZ)
START_EPOCH=$(date +%s)

nohup env \
    DEUSRIDET_RECLUSTERER_ENABLE=1 \
    DEUSRIDET_RECLUSTERER_LINK_THRESH="$LINK" \
    DEUSRIDET_RECLUSTERER_ACCEPT_ABSTAINED="$ACCEPT" \
    "$BINARY" awaken > "$AWAKEN_LOG" 2>&1 &
AWAKEN_PID=$!
disown $AWAKEN_PID

# wait for awaken to be ready
for i in $(seq 1 60); do
    sleep 1
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/ | grep -q 200; then
        break
    fi
done
if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/ | grep -q 200; then
    echo "ERROR: awaken did not reach HTTP 200" >&2
    kill $AWAKEN_PID 2>/dev/null || true
    exit 4
fi

# --- run replay ---
python3 tools/replay_to_transcript.py \
    --audio tests/test.mp3 \
    --gt tests/fixtures/test_ground_truth.json \
    --max-sec 3615 \
    --speed 2.0 \
    --drain-sec 60 \
    --out-dir "$OUT" > "$OUT/runner.log" 2>&1
REPLAY_EXIT=$?

END_TS=$(date -u +%FT%TZ)
END_EPOCH=$(date +%s)
WALL_SEC=$((END_EPOCH - START_EPOCH))

# --- stop awaken ---
kill $AWAKEN_PID 2>/dev/null || true
sleep 2
pgrep -af deusridet >/dev/null && sudo kill -9 $(pgrep -f deusridet) || true

cp "$AWAKEN_LOG" "$OUT/awaken.log"

OVERFLOWS=$(grep -c "Ring buffer overflow" "$AWAKEN_LOG" || true)

# --- compute K_pred final + abstain% ---
python3 - <<PY >"$OUT/_summary.json"
import json, sys
seg = json.load(open("$OUT/runtime_segments.json"))
def final_id(s):
    cur = s["current_id"]
    for r in s.get("relabel_chain", []):
        if r["old"] == cur:
            cur = r["new"]
    return cur
from collections import Counter
c = Counter(final_id(s) for s in seg)
abstain = c.get(-1, 0)
non_abstain = {k: v for k, v in c.items() if k != -1}
K = len(non_abstain)
print(json.dumps({
    "n_segments": len(seg),
    "abstain_count": abstain,
    "abstain_pct": round(abstain / max(1, len(seg)) * 100, 2),
    "K_pred_final": K,
    "id_distribution": dict(sorted(non_abstain.items(), key=lambda x: -x[1])),
}, ensure_ascii=False, indent=2))
PY

# --- manifest ---
cat > "$OUT/manifest.json" <<JSON
{
  "cell": "$CELL",
  "matrix": "v1",
  "commit": "$COMMIT",
  "binary_sha256": "$BIN_SHA",
  "env": {
    "DEUSRIDET_RECLUSTERER_ENABLE": "1",
    "DEUSRIDET_RECLUSTERER_LINK_THRESH": "$LINK",
    "DEUSRIDET_RECLUSTERER_ACCEPT_ABSTAINED": "$ACCEPT"
  },
  "fixed": {
    "audio": "tests/test.mp3",
    "gt": "tests/fixtures/test_ground_truth.json",
    "speed": 2.0,
    "drain_sec": 60,
    "max_sec": 3615,
    "window_sec_default": 180.0,
    "interval_sec_default": 30.0,
    "ema_default": 0.20,
    "k_range": [2, 6]
  },
  "start_ts": "$START_TS",
  "end_ts": "$END_TS",
  "wall_sec": $WALL_SEC,
  "replay_exit": $REPLAY_EXIT,
  "ring_buffer_overflows": $OVERFLOWS,
  "summary": $(cat "$OUT/_summary.json")
}
JSON

rm -f "$OUT/_summary.json"

echo "=== cell $CELL done ==="
cat "$OUT/manifest.json"
