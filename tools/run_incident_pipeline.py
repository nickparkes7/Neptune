#!/usr/bin/env python3
"""Run the full anomaly → incident pipeline and emit lifecycle transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anomaly import PipelineConfig, generate_transitions_from_ndjson


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="SeaOWL NDJSON stream to process")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--flush-after",
        type=float,
        default=None,
        help="Seconds after the final event to flush/close incidents (defaults to incident_ttl_s + 1)",
    )
    return parser


def serialize_transition(transition) -> Dict[str, Any]:
    incident_payload = transition.incident.model_dump(mode="json")
    trigger_payload = (
        transition.trigger_event.model_dump(mode="json") if transition.trigger_event else None
    )
    return {
        "kind": transition.kind,
        "at": transition.at.isoformat().replace("+00:00", "Z"),
        "reason": transition.reason,
        "allow_tasking": transition.allow_tasking,
        "incident": incident_payload,
        "trigger_event": trigger_payload,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig(flush_after_s=args.flush_after)
    result = generate_transitions_from_ndjson(args.input, config=config)
    payload = [serialize_transition(t) for t in result.transitions]
    text = json.dumps(payload, indent=2 if args.pretty else None)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + ("\n" if not text.endswith("\n") else ""))
    else:
        print(text)


if __name__ == "__main__":
    main()
