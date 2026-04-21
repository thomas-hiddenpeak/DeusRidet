#!/usr/bin/env python3
"""Split src/actus/actus.cpp: one cmd_* / print_* function per file.

Strategy:
- Preserve lines 1-42 (header + includes) as the common prelude.
- Keep actus.cpp as a tiny "registry" file containing:
    * the common prelude
    * namespace open
    * module-scope constants (VERSION, BUILD_DATE)
    * g_shutdown_requested definition
    * print_version() and print_usage()
    * namespace close
- For every `cmd_*` entry point, emit src/actus/<name>.cpp with:
    * an @file + @philosophical_role doxygen header
    * the common prelude verbatim
    * namespace deusridet {
    * the extracted function body verbatim
    * } // namespace deusridet

Invariant: byte-wise concatenation of all emitted function bodies (after
stripping per-file wrappers) must equal the original function bodies.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "actus" / "actus.cpp"

FN_RE = re.compile(r"^(?:int|void) ((?:cmd_|print_)\w+)\b")


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Locate the prelude end: line containing `namespace deusridet {`.
    ns_open = next(i for i, l in enumerate(lines) if l.startswith("namespace deusridet {"))
    prelude = "".join(lines[:ns_open])  # everything before (incl. blank line after includes)
    # Locate namespace close at end-of-file.
    ns_close = next(i for i in range(len(lines) - 1, -1, -1)
                    if lines[i].startswith("} // namespace deusridet"))

    # Find function start lines. A function begins on a line matching FN_RE,
    # and the preceding block (comment banner + possibly blank lines) belongs
    # to the function emotionally — but for a clean split we only extract the
    # function itself (signature through closing `}` at column 0). A
    # top-level function ends at the next line starting with `}` in column 0.
    body_start = ns_open + 1
    body_end = ns_close  # exclusive
    body_lines = lines[body_start:body_end]

    # Find `^}` at column 0 — each marks end of a top-level definition.
    def is_close(line: str) -> bool:
        return line == "}\n" or line == "}"

    # Walk forward, grouping functions.
    fns: list[tuple[str, int, int]] = []  # (name, start_idx_in_body, end_idx_exclusive)
    i = 0
    while i < len(body_lines):
        m = FN_RE.match(body_lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        start = i
        # Find closing brace at column 0.
        j = i + 1
        while j < len(body_lines) and not is_close(body_lines[j]):
            j += 1
        assert j < len(body_lines), f"no close for {name}"
        fns.append((name, start, j + 1))
        i = j + 1

    print(f"Found {len(fns)} top-level functions:")
    for name, s, e in fns:
        print(f"  {name:30s} body[{s}..{e})  {e-s} lines")

    # Identify extractables (cmd_*) and residents (print_*, and anything else).
    residents = {n for n, _, _ in fns if n.startswith("print_")}
    extracted = [(n, s, e) for n, s, e in fns if not n.startswith("print_")]

    # Build the residual actus.cpp: prelude + namespace + constants +
    # shutdown flag + print_version + print_usage + namespace close.
    # We rebuild by walking body_lines and keeping only the non-extracted
    # regions.
    keep = [True] * len(body_lines)
    for name, s, e in fns:
        if name not in residents:
            for k in range(s, e):
                keep[k] = False
            # Also skip trailing blank line(s) to keep output tidy.
            k = e
            while k < len(body_lines) and body_lines[k].strip() == "":
                keep[k] = False
                k += 1

    # Also scrub the two "section banner" comments that belong to extracted
    # functions. We leave the `version/usage` banner intact because that
    # section is staying.
    # Strategy: drop any comment-banner immediately preceding an extracted
    # function start.
    for name, s, e in extracted:
        # Walk backward to find a `// ====...` banner and blank line before.
        k = s - 1
        # Skip one blank line
        while k >= 0 and body_lines[k].strip() == "":
            keep[k] = False
            k -= 1
        # Skip a comment banner block (lines starting with `// `)
        while k >= 0 and body_lines[k].lstrip().startswith("//"):
            keep[k] = False
            k -= 1
        # Skip one blank line before that banner
        while k >= 0 and body_lines[k].strip() == "":
            keep[k] = False
            k -= 1

    residual_body = "".join(l for l, k in zip(body_lines, keep) if k).rstrip() + "\n"

    # Update header of actus.cpp.
    new_header = """/**
 * @file actus.cpp
 * @philosophical_role Registry of the Actus subsystem. Holds the module-scope
 *         shutdown flag (the one shared datum between the signal handler and
 *         every running command) plus the two non-command entry points
 *         `print_version()` and `print_usage()`.
 * @serves main.cpp. Every `cmd_*` lives in its own translation unit in
 *         src/actus/; see src/actus/README.la for the subsystem-level anchor.
 *
 * Rationale: an act is transient, so the Actus subsystem holds almost no
 * state. What little remains — the shutdown flag, the version strings, the
 * usage banner — lives here because it is shared across all acts.
 */

"""
    includes_and_namespace = "".join(lines[ns_open - 0:ns_open + 1])  # just the namespace line
    # Reconstruct includes portion (between end of original doxygen header
    # and `namespace deusridet {`).
    orig_doxy_end = 0
    for i, l in enumerate(lines):
        if l.startswith(" */"):
            orig_doxy_end = i + 1
            break
    includes_block = "".join(lines[orig_doxy_end:ns_open])
    new_actus = (
        new_header
        + includes_block
        + "namespace deusridet {\n"
        + residual_body
        + "\n} // namespace deusridet\n"
    )
    SRC.write_text(new_actus, encoding="utf-8")
    print(f"Rewrote {SRC} -> {len(new_actus.splitlines())} lines")

    # Emit extracted function files.
    out_dir = SRC.parent
    for name, s, e in extracted:
        body = "".join(body_lines[s:e]).rstrip() + "\n"
        # Find a short role summary from the function name.
        role = f"External command `{name}`."
        header = f"""/**
 * @file {name}.cpp
 * @philosophical_role {role} An Actus function — one CLI verb, one finite
 *         act, one return code.
 * @serves main.cpp dispatch (declaration in actus.h).
 */

"""
        content = (
            header
            + includes_block
            + "namespace deusridet {\n\n"
            + body
            + "\n} // namespace deusridet\n"
        )
        (out_dir / f"{name}.cpp").write_text(content, encoding="utf-8")
        print(f"  wrote {name}.cpp ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
