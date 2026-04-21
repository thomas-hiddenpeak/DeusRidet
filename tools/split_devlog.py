#!/usr/bin/env python3
"""Split docs/{en,zh}/DEVLOG.md into docs/{en,zh}/devlog/YYYY-MM-DD.md daily files.

Multiple `## YYYY-MM-DD — title` sections sharing a date are merged into one
daily file (in original order). The top-level DEVLOG.md is replaced by an
index that links to each daily file in reverse chronological order.
"""
from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict, OrderedDict

ROOT = Path(__file__).resolve().parent.parent
HEADING_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2}) — (.+)$")

LANG_TITLES = {
    "en": ("DeusRidet Development Log", "DEVLOG", "Daily entries (newest first)"),
    "zh": ("DeusRidet 开发日志", "开发日志", "按日条目（最新在前）"),
}


def split_devlog(lang: str) -> None:
    src = ROOT / "docs" / lang / "DEVLOG.md"
    out_dir = ROOT / "docs" / lang / "devlog"
    out_dir.mkdir(exist_ok=True)

    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    # Strip the H1 header (first non-empty `# ...` line).
    i = 0
    while i < len(lines) and not lines[i].startswith("## "):
        i += 1
    body = lines[i:]

    # Walk sections.
    sections: list[tuple[str, str, list[str]]] = []  # (date, title, body_lines)
    current: list[str] | None = None
    cur_date = cur_title = ""
    for line in body:
        m = HEADING_RE.match(line.rstrip("\n"))
        if m:
            if current is not None:
                sections.append((cur_date, cur_title, current))
            cur_date, cur_title = m.group(1), m.group(2).strip()
            current = []
        else:
            if current is not None:
                current.append(line)
    if current is not None:
        sections.append((cur_date, cur_title, current))

    # Bucket by date (preserve insertion order).
    by_date: "OrderedDict[str, list[tuple[str, list[str]]]]" = OrderedDict()
    for date, title, content in sections:
        by_date.setdefault(date, []).append((title, content))

    # Write daily files.
    for date, entries in by_date.items():
        path = out_dir / f"{date}.md"
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# DEVLOG — {date}\n\n")
            for idx, (title, content) in enumerate(entries):
                f.write(f"## {title}\n")
                # content already includes its leading blank line typically.
                f.writelines(content)
                if idx != len(entries) - 1 and not content[-1].endswith("\n\n"):
                    f.write("\n")

    # Build index.
    h1, idx_h2, idx_intro = LANG_TITLES[lang]
    sorted_dates = sorted(by_date.keys(), reverse=True)
    with src.open("w", encoding="utf-8") as f:
        f.write(f"# {h1}\n\n")
        f.write(f"{idx_intro}. ")
        if lang == "en":
            f.write("Each link opens that day's full notes.\n\n")
            f.write("To add a new entry, create `devlog/YYYY-MM-DD.md` and prepend it here.\n\n")
        else:
            f.write("点击日期查看当日完整记录。\n\n")
            f.write("新增条目：在 `devlog/YYYY-MM-DD.md` 中创建文件，并在下方列表顶端追加链接。\n\n")
        f.write(f"## {idx_h2}\n\n")
        for date in sorted_dates:
            titles = " ; ".join(t for t, _ in by_date[date])
            f.write(f"- [{date}](devlog/{date}.md) — {titles}\n")
        f.write("\n")

    print(f"[{lang}] {len(sections)} sections → {len(by_date)} daily files")


if __name__ == "__main__":
    for lang in ("en", "zh"):
        split_devlog(lang)
