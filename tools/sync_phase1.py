#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import yaml

STATUS = Path("status/phase1.yml")
PHASE1 = Path("PHASE1.md")
if not PHASE1.exists():
    PHASE1 = Path("docs/PHASE1.md")

BEGIN = "<!-- STATUS:PHASE1:BEGIN -->"
END = "<!-- STATUS:PHASE1:END -->"


def load_status():
    with STATUS.open() as f:
        return yaml.safe_load(f)


def make_table(data):
    steps = data.get("steps", [])
    done = sum(1 for s in steps if s.get("status") == "done")
    ip = sum(1 for s in steps if s.get("status") == "in_progress")
    blk = sum(1 for s in steps if s.get("status") == "blocked")
    total = len(steps)
    lines = []
    lines.append(f"Progress: {done}/{total} steps done · {ip} in progress · {blk} blocked")
    lines.append("")
    lines.append("| Step | Status | Owner | Notes |")
    lines.append("| --- | --- | --- | --- |")
    for s in steps:
        owner = s.get("owner") or ""
        notes = s.get("notes") or ""
        status = s.get("status", "pending")
        lines.append(f"| {s['id']} | {status} | {owner} | {notes} |")
    return "\n".join(lines)


def ensure_markers(text):
    if BEGIN in text and END in text:
        return text
    lines = text.splitlines()
    insert_idx = 1 if lines and lines[0].startswith("# ") else 0
    block = [BEGIN, "", "(auto-generated; do not edit)", "", END]
    new = lines[: insert_idx + 1] + [""] + block + [""] + lines[insert_idx + 1 :]
    return "\n".join(new) + ("\n" if not text.endswith("\n") else "")


def build_updated_markdown():
    data = load_status()
    md = PHASE1.read_text()
    md = ensure_markers(md)
    table = make_table(data)
    pattern = re.compile(re.escape(BEGIN) + r"[\s\S]*?" + re.escape(END), re.M)
    replacement = BEGIN + "\n\n" + table + "\n\n" + END
    return pattern.sub(replacement, md)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sync Phase 1 status block")
    parser.add_argument("--check", action="store_true", help="Check if status block is up to date")
    args = parser.parse_args(argv)

    updated = build_updated_markdown()
    current = PHASE1.read_text()

    if args.check:
        if updated != current:
            sys.stderr.write("PHASE1.md status block is out of date. Run without --check to update.\n")
            return 1
        print("PHASE1.md status block is up to date.")
        return 0

    PHASE1.write_text(updated)
    print("Updated PHASE1.md status block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
