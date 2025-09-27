"""End-to-end SeaOWL anomaly pipeline: scorer → events → incident transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from .events import EventExtractorConfig, SuspectedSpillEvent, extract_events
from .hybrid import HybridOilAlertScorer
from .incidents import IncidentManager, IncidentManagerConfig, IncidentTransition

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


@dataclass
class PipelineResult:
    """Structured output listing all intermediate artifacts."""

    events: List[SuspectedSpillEvent]
    transitions: List[IncidentTransition]


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
    transitions = run_pipeline(events, cfg, sink=sink)
    return PipelineResult(events=events, transitions=transitions)


def run_pipeline(
    events: Sequence[SuspectedSpillEvent],
    config: Optional[PipelineConfig] = None,
    sink: Optional[Callable[[IncidentTransition], None]] = None,
) -> List[IncidentTransition]:
    """Feed scored events into the incident manager and collect transitions."""

    cfg = config or PipelineConfig()
    manager = IncidentManager(cfg.incident_config)
    transitions: List[IncidentTransition] = []

    for event in sorted(events, key=lambda e: e.ts_start):
        for transition in manager.process_event(event):
            transitions.append(transition)
            if sink:
                sink(transition)

    if events:
        last_ts = events[-1].ts_end
        flush_after = cfg.flush_after_s or manager.cfg.incident_ttl_s + 1.0
        flush_time = last_ts + timedelta(seconds=flush_after)
        for transition in manager.flush(flush_time):
            transitions.append(transition)
            if sink:
                sink(transition)
        if manager.is_active:
            for transition in manager.finalize(at=flush_time):
                transitions.append(transition)
                if sink:
                    sink(transition)

    return transitions
