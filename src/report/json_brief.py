"""Utilities to generate machine-readable incident briefs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from anomaly.events import SuspectedSpillEvent
from agent.schemas import AgentPlan, IncidentSynopsis
from cerulean import CeruleanQueryResult


def write_incident_brief(
    event: SuspectedSpillEvent,
    *,
    plan: AgentPlan,
    synopsis: IncidentSynopsis,
    cerulean: CeruleanQueryResult,
    artifact_dir: Path,
    generated_at: Optional[datetime] = None,
) -> Path:
    """Create a JSON brief summarising the incident and agent output.

    Parameters
    ----------
    event:
        The incident snapshot that triggered the agent run.
    plan:
        The agent's chosen Cerulean parameters.
    synopsis:
        Structured synopsis returned by the agent.
    cerulean:
        Raw Cerulean query result for additional context.
    artifact_dir:
        Directory where the brief should be written.
    generated_at:
        Optional override for the timestamp; defaults to ``datetime.utcnow``.
    """

    artifact_dir.mkdir(parents=True, exist_ok=True)
    generated_at = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)

    brief = {
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "incident": {
            "event_id": event.event_id,
            "ts_start": event.ts_start.isoformat().replace("+00:00", "Z"),
            "ts_end": event.ts_end.isoformat().replace("+00:00", "Z"),
            "ts_peak": event.ts_peak.isoformat().replace("+00:00", "Z"),
            "lat": event.lat,
            "lon": event.lon,
            "duration_s": event.duration_s,
            "aoi_bbox": event.aoi_bbox,
            "platform_id": event.platform_id,
            "sensor_id": event.sensor_id,
        },
        "scenario": synopsis.scenario,
        "confidence": synopsis.confidence,
        "summary": synopsis.summary,
        "rationale": synopsis.rationale,
        "recommended_actions": synopsis.recommended_actions,
        "metrics": synopsis.metrics.model_dump(mode="json"),
        "cerulean_query": {
            "padding_km": plan.padding_km,
            "lookback_hours": plan.lookback_hours,
            "lookahead_hours": plan.lookahead_hours,
            "min_source_score": plan.min_source_score,
            "only_active": plan.only_active,
            "filters": plan.filters,
            "sort_by": plan.sort_by,
            "limit": plan.limit,
            "followup_delay_hours": plan.followup_delay_hours,
        },
        "cerulean_result": {
            "number_matched": cerulean.number_matched,
            "number_returned": cerulean.number_returned,
        },
        "follow_up": {
            "scheduled": synopsis.followup_scheduled,
            "eta": synopsis.followup_eta.isoformat().replace("+00:00", "Z")
            if synopsis.followup_eta
            else None,
        },
        "artifacts": {k: str(v) for k, v in synopsis.artifacts.items()},
    }

    brief_path = artifact_dir / "incident_brief.json"
    brief_path.write_text(json.dumps(brief, indent=2))
    return brief_path
