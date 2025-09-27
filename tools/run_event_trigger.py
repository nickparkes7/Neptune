#!/usr/bin/env python3
"""Run the hybrid anomaly trigger and emit SuspectedSpillEvent JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anomaly import generate_events_from_ndjson, SuspectedSpillEvent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="SeaOWL NDJSON stream to score")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path (defaults to stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indentation",
    )
    return parser


def serialize_events(events: Sequence[SuspectedSpillEvent]) -> list[dict]:
    return [event.model_dump(mode="json") for event in events]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    events = generate_events_from_ndjson(args.input)
    payload = serialize_events(events)
    text = json.dumps(payload, indent=2 if args.pretty else None)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + ("\n" if not text.endswith("\n") else ""))
    else:
        print(text)


if __name__ == "__main__":
    main()
