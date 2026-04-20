#!/usr/bin/env python3
"""server.log -> mapped.txt  (FULL + FULL margin-abstain).

Anchor audio time to the wall-clock of the LAST `[WS] Client connected`
event BEFORE the first speaker event. Earlier WS connects (HTTP probes
etc.) are skipped.
"""
import re, sys

def parse_wall(line):
    m = re.match(r'\[(\d{2}):(\d{2}):(\d{2})\.(\d+)\]', line)
    if not m:
        return None
    h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h*3600 + mi*60 + s + ms/1000.0

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <server.log> <mapped.txt> [speed=2.0]", file=sys.stderr)
        sys.exit(1)
    log_path, out_path = sys.argv[1], sys.argv[2]
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    pat_full   = re.compile(r'FULL: id=(-?\d+) sim=([\d.]+)')
    pat_margin = re.compile(r'FULL margin-abstain: id=(-?\d+) sim=([\d.]+)')
    pat_ws     = re.compile(r'\[WS\] Client connected')

    last_ws = None
    anchor = None
    rows = []
    with open(log_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if pat_ws.search(line):
                last_ws = parse_wall(line)
                continue
            m = pat_full.search(line) or pat_margin.search(line)
            if not m:
                continue
            if anchor is None:
                # Freeze anchor to the most recent WS connect event.
                anchor = last_ws if last_ws is not None else parse_wall(line)
            wall = parse_wall(line)
            if wall is None:
                continue
            rows.append(((wall - anchor) * speed, int(m.group(1))))
    rows.sort(key=lambda r: r[0])
    with open(out_path, 'w') as f:
        for a, sid in rows:
            f.write(f"{a:.3f} id={sid}\n")
    identified = sum(1 for _, sid in rows if sid >= 0)
    print(f"Parsed {len(rows)} events -> {out_path}  (anchor={anchor})")
    print(f"  identified: {identified}  abstained: {len(rows)-identified}")

if __name__ == '__main__':
    main()
