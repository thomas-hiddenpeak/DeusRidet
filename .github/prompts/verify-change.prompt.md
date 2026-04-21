---
mode: agent
description: Run the full post-change verification checklist (build, kill, drop caches, start, HTTP 200, WS 101).
---

# Verify Change

Run the mandatory post-change verification checklist defined in
`.github/instructions/workflow.instructions.md`. Do not skip steps. Do not
declare success until both HTTP 200 and WS 101 are confirmed.

## Steps (execute in order)

1. **Build**
   ```bash
   cd /home/rm01/DeusRidet/build && make -j$(nproc)
   ```
   If build fails, stop and diagnose. Do not proceed.

2. **Kill previous + free port**
   ```bash
   sudo kill -9 $(pgrep -f deusridet) 2>/dev/null
   sudo fuser -k 8080/tcp 2>/dev/null
   ```

3. **Drop page caches**
   ```bash
   echo 3 | sudo tee /proc/sys/vm/drop_caches
   ```

4. **Start service (async)**
   ```bash
   cd /home/rm01/DeusRidet && ./build/deusridet awaken
   ```
   Wait until the log shows `WebUI server listening on :8080` or equivalent.

5. **HTTP 200 check**
   ```bash
   curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/
   ```
   Must print `200`.

6. **Asset check** — for every JS/CSS file added or touched in this change:
   ```bash
   curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/js/app.js
   curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/<new-asset-path>
   ```
   Each must print `200`.

7. **WS 101 check**
   ```bash
   curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
     -H "Upgrade: websocket" -H "Connection: Upgrade" \
     -H "Sec-WebSocket-Key: dGVzdA==" -H "Sec-WebSocket-Version: 13" \
     http://localhost:8080/ws
   ```
   Must print `101`.

8. **Kill the server** to leave the system clean:
   ```bash
   sudo kill -9 $(pgrep -f deusridet) 2>/dev/null
   ```

## Report

Report the exact HTTP codes observed for steps 5/6/7. Only declare success
if all three are as required. On any failure, do not guess — read the server
log and fix the root cause before retrying.
