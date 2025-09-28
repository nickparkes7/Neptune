"""End-to-end SeaOWL anomaly pipeline: scorer → events → incident transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .events import EventExtractorConfig, SuspectedSpillEvent, extract_events
from .hybrid import HybridOilAlertScorer
from .incidents import IncidentManager, IncidentManagerConfig, IncidentTransition

from agent import AgentConfig as GPTAgentConfig, AgentModel, GPTAgentModel, run_agent_for_event
from agent.runner import AgentRunResult
from cerulean import CeruleanClient

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "generate_transitions_from_ndjson",
    "run_pipeline",
]


@dataclass
class PipelineConfig:
    """Bundle configuration knobs for scorer → event → incident pipeline."""

    scorer: Optional[HybridOilAlertScorer] = None
    event_config: Optional[EventExtractorConfig] = None
    incident_config: Optional[IncidentManagerConfig] = None
    flush_after_s: Optional[float] = None
    agent_enabled: bool = False
    agent_model: Optional[AgentModel] = None
    agent_config: Optional[GPTAgentConfig] = None
    agent_client: Optional[CeruleanClient] = None
    agent_trigger_kinds: Tuple[str, ...] = ("opened", "updated")
    agent_sink: Optional[Callable[[AgentRunResult], None]] = None


@dataclass
class PipelineResult:
    """Structured output listing all intermediate artifacts."""

    events: List[SuspectedSpillEvent]
    transitions: List[IncidentTransition]
    agent_runs: List[AgentRunResult] = field(default_factory=list)


def generate_transitions_from_ndjson(
    path: Path,
    config: Optional[PipelineConfig] = None,
    sink: Optional[Callable[[IncidentTransition], None]] = None,
) -> PipelineResult:
    """Run the full pipeline for a SeaOWL NDJSON payload."""

    cfg = config or PipelineConfig()
    scorer = cfg.scorer or HybridOilAlertScorer()
    df = scorer.score_ndjson(path)
    events = extract_events(df, config=cfg.event_config)
    transitions, agent_runs = run_pipeline(events, cfg, sink=sink)
    return PipelineResult(events=events, transitions=transitions, agent_runs=agent_runs)


def run_pipeline(
    events: Sequence[SuspectedSpillEvent],
    config: Optional[PipelineConfig] = None,
    sink: Optional[Callable[[IncidentTransition], None]] = None,
) -> Tuple[List[IncidentTransition], List[AgentRunResult]]:
    """Feed scored events into the incident manager and collect transitions."""

    cfg = config or PipelineConfig()
    manager = IncidentManager(cfg.incident_config)
    transitions: List[IncidentTransition] = []
    agent_runs: List[AgentRunResult] = []

    agent_enabled = cfg.agent_enabled or cfg.agent_model is not None
    agent_model: Optional[AgentModel] = None
    agent_config = cfg.agent_config
    agent_client = cfg.agent_client
    trigger_set = set(cfg.agent_trigger_kinds)

    for event in sorted(events, key=lambda e: e.ts_start):
        for transition in manager.process_event(event):
            transitions.append(transition)
            if sink:
                sink(transition)
            if agent_enabled and _should_run_agent(transition, trigger_set):
                agent_model = agent_model or cfg.agent_model or GPTAgentModel()
                agent_client = agent_client or CeruleanClient()
                agent_config = agent_config or GPTAgentConfig()
                run = run_agent_for_event(
                    transition.incident,
                    model=agent_model,
                    client=agent_client,
                    config=agent_config,
                    timestamp=transition.at,
                )
                agent_runs.append(run)
                if cfg.agent_sink:
                    cfg.agent_sink(run)

    if events:
        last_ts = events[-1].ts_end
        flush_after = cfg.flush_after_s or manager.cfg.incident_ttl_s + 1.0
        flush_time = last_ts + timedelta(seconds=flush_after)
        for transition in manager.flush(flush_time):
            transitions.append(transition)
            if sink:
                sink(transition)
            if agent_enabled and _should_run_agent(transition, trigger_set):
                agent_model = agent_model or cfg.agent_model or GPTAgentModel()
                agent_client = agent_client or CeruleanClient()
                agent_config = agent_config or GPTAgentConfig()
                run = run_agent_for_event(
                    transition.incident,
                    model=agent_model,
                    client=agent_client,
                    config=agent_config,
                    timestamp=transition.at,
                )
                agent_runs.append(run)
                if cfg.agent_sink:
                    cfg.agent_sink(run)
        if manager.is_active:
            for transition in manager.finalize(at=flush_time):
                transitions.append(transition)
                if sink:
                    sink(transition)
                if agent_enabled and _should_run_agent(transition, trigger_set):
                    agent_model = agent_model or cfg.agent_model or GPTAgentModel()
                    agent_client = agent_client or CeruleanClient()
                    agent_config = agent_config or GPTAgentConfig()
                    run = run_agent_for_event(
                        transition.incident,
                        model=agent_model,
                        client=agent_client,
                        config=agent_config,
                        timestamp=transition.at,
                    )
                    agent_runs.append(run)
                    if cfg.agent_sink:
                        cfg.agent_sink(run)

    return transitions, agent_runs


def _should_run_agent(transition: IncidentTransition, trigger_kinds: Sequence[str]) -> bool:
    return transition.allow_tasking and transition.kind in trigger_kinds
