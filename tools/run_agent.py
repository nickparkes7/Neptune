"""CLI entry point to run the GPT-5 agent on a saved event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anomaly.events import SuspectedSpillEvent

from agent import AgentConfig, GPTAgentModel, RuleBasedAgentModel, run_agent_for_event
from tools.load_env import load_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", type=Path, help="Path to a SuspectedSpillEvent JSON file")
    parser.add_argument("--artifact-root", type=Path, default=AgentConfig().artifact_root)
    parser.add_argument("--followup-store", type=Path, default=AgentConfig().followup_store)
    parser.add_argument(
        "--rule-based",
        action="store_true",
        help="Use the deterministic fallback model instead of GPT-5 (offline/testing).",
    )
    args = parser.parse_args(argv)

    load_env()
    payload = json.loads(args.event.read_text())
    event = SuspectedSpillEvent.model_validate(payload)
    config = AgentConfig(artifact_root=args.artifact_root, followup_store=args.followup_store)
    model = RuleBasedAgentModel() if args.rule_based else GPTAgentModel()
    result = run_agent_for_event(event, model=model, config=config)

    print(json.dumps({
        "plan": json.loads(result.plan.json()),
        "synopsis": json.loads(result.synopsis.json()),
        "artifacts": [str(path) for path in result.artifacts],
        "trace": str(result.trace_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
