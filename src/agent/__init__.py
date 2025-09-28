"""GPT-5 agent orchestration package for Neptune Phase 1."""

from importlib import import_module
from typing import Any

__all__ = [
    "AgentModel",
    "GPTAgentModel",
    "RuleBasedAgentModel",
    "AgentConfig",
    "AgentRunResult",
    "AgentPlan",
    "IncidentSynopsis",
    "AgentBrief",
    "run_agent_for_event",
]


_MODEL_EXPORTS = {"AgentModel", "GPTAgentModel", "RuleBasedAgentModel"}
_RUNNER_EXPORTS = {"AgentConfig", "AgentRunResult", "run_agent_for_event"}
_SCHEMA_EXPORTS = {"AgentPlan", "IncidentSynopsis", "AgentBrief"}


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        module = import_module(".model", __name__)
        return getattr(module, name)
    if name in _RUNNER_EXPORTS:
        module = import_module(".runner", __name__)
        return getattr(module, name)
    if name in _SCHEMA_EXPORTS:
        module = import_module(".schemas", __name__)
        return getattr(module, name)
    raise AttributeError(f"module 'agent' has no attribute '{name}'")
