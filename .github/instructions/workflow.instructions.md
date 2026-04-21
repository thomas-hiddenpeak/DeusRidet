---
applyTo: "**"
---

# Workflow — Verification, Pacing, Git Discipline

## Implementation Pacing

**Never attempt to generate very long files in a single step.** Break large
file creation into incremental stages — define structures first, then
implement functions one at a time. Long generation attempts will time out and
waste the entire output. Prefer multiple small, compilable commits over one
monolithic generation. Each step must be independently verifiable (compiles,
runs, or at minimum has no syntax errors).

## Testing Discipline (mandatory before every test/benchmark)

1. Kill all previous test processes: `sudo kill -9 $(pgrep -f deusridet) 2>/dev/null`
2. Drop page caches: `echo 3 | sudo tee /proc/sys/vm/drop_caches`
3. Verify clean state before proceeding.

No exceptions. Every invocation, not just the first one.

## Post-Change Verification (mandatory after every build-affecting change)

1. Build: `cd build && make -j$(nproc)`
2. Kill + free port: `sudo kill -9 $(pgrep -f deusridet) 2>/dev/null; sudo fuser -k 8080/tcp 2>/dev/null`
3. Drop page caches: `echo 3 | sudo tee /proc/sys/vm/drop_caches`
4. Start service: `cd /home/rm01/DeusRidet && ./build/deusridet awaken`
5. Verify WebUI: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/` → **200**
6. Verify key assets load (at least `app.js` and newly added component JS/CSS) → **200**
7. Verify WebSocket: 
   ```
   curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
     -H "Upgrade: websocket" -H "Connection: Upgrade" \
     -H "Sec-WebSocket-Key: dGVzdA==" -H "Sec-WebSocket-Version: 13" \
     http://localhost:8080/ws
   ```
   → **101**

Task is not complete until step 5 confirms 200 and step 7 confirms 101.
Use the `/verify-change` prompt to run this checklist.

## Git Discipline

- Atomic commits with descriptive messages.
- Every experimental attempt gets its own commit — including failures.
  Recording what didn't work is as valuable as recording what did.
- Conventional commit prefixes: `feat:`, `fix:`, `perf:`, `test:`, `docs:`,
  `refactor:`, `experiment:`.
- `main` is stable. Experiments on feature branches. Merge only after
  verification passes.
- All commit messages in English.

## File-Size Limits (R1)

| Kind | Soft limit | Hard limit |
|------|------------|------------|
| `.cpp` / `.h` / `.hpp` | 400 lines | 500 lines |
| `.cu` / `.cuh` | 600 lines | 800 lines |

Exceeding hard limit triggers mandatory split. No "just this once" exceptions.
Current violations are tracked in `docs/en/architecture/00-overview.md`
under "Refactor Backlog".

## Observability

Every internal process must be inspectable from the WebUI. Every concurrent
Decode branch carries a unique trace ID. Every event, memory record, and log
entry carries a full `tempus::TimeStamp` triple (see `09-tempus.md`).

## Error Handling

Validate at system boundaries only. No defensive coding in hot inner loops.
Boundary = external input (WS message, file read, CLI arg, model weight load).
Interior = trusted arena.
