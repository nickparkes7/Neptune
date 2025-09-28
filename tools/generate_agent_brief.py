#!/usr/bin/env python3
"""Generate a visually rich agent brief from cached SeaOWL evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.briefing import (  # noqa: E402
    build_agent_brief,
    brief_to_markdown,
    render_brief_media,
    score_stream,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="seaowl_demo",
        help="Scenario identifier captured in the brief metadata.",
    )
    parser.add_argument(
        "--stream",
        type=Path,
        default=ROOT / "data/ship/seaowl_live.ndjson",
        help="SeaOWL NDJSON stream to summarise.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "artifacts/briefs",
        help="Directory for brief outputs (JSON/Markdown/media).",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional ISO timestamp override for generated_at field (UTC).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.stream.exists():
        raise SystemExit(f"Stream not found: {args.stream}")

    outdir: Path = args.outdir
    media_dir = outdir / "media"
    outdir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    frame = score_stream(args.stream)
    artifacts = render_brief_media(
        frame,
        media_dir,
        media_url_base="/agent-brief/media",
    )

    generated_at = (
        datetime.fromisoformat(args.timestamp)
        if args.timestamp
        else datetime.now(timezone.utc)
    )

    brief = build_agent_brief(
        scenario_id=args.scenario,
        stream_path=args.stream,
        frame=frame,
        artifacts=artifacts,
        hero_artifact_key="seaowl_timeseries",
        generated_at=generated_at,
    )

    json_path = outdir / "latest.json"
    md_path = outdir / "latest.md"
    scenario_json_path = outdir / f"{args.scenario}.json"

    payload = brief.model_dump(mode="json")
    json_path.write_text(json.dumps(payload, indent=2))
    scenario_json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(brief_to_markdown(brief))

    if not args.quiet:
        print(f"Generated {json_path.relative_to(ROOT)}")
        print(f"Generated {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
