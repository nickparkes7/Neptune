"""Agent runner that coordinates GPT-5 planning with Cerulean tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from anomaly.events import SuspectedSpillEvent
from cerulean import (
    CeruleanClient,
    CeruleanQueryResult,
    build_feature_collection,
    schedule_followup,
    summarize_slicks,
)

from .model import AgentModel, GPTAgentModel, RuleBasedAgentModel
from .schemas import ActionRecord, AgentPlan, IncidentSynopsis, QueryBounds

DEFAULT_ARTIFACT_ROOT = Path("artifacts")


@dataclass
class AgentConfig:
    """Configuration for the orchestration agent."""

    query_bounds: QueryBounds = field(default_factory=QueryBounds)
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    followup_store: Path = field(default_factory=lambda: Path("data/cerulean/followups.ndjson"))
    cerulean_limit_default: int = 100


@dataclass
class AgentRunResult:
    plan: AgentPlan
    synopsis: IncidentSynopsis
    cerulean_result: CeruleanQueryResult
    artifacts: List[Path]
    trace_path: Path


def run_agent_for_event(
    event: SuspectedSpillEvent,
    *,
    model: Optional[AgentModel] = None,
    client: Optional[CeruleanClient] = None,
    config: Optional[AgentConfig] = None,
    timestamp: Optional[datetime] = None,
) -> AgentRunResult:
    """Execute the agent for a single spill event."""

    model = model or GPTAgentModel()
    cfg = config or AgentConfig()
    client = client or CeruleanClient()
    now = _ensure_utc(timestamp or datetime.now(timezone.utc))

    trace: List[ActionRecord] = []

    plan = model.generate_plan(event, cfg.query_bounds)
    trace.append(
        ActionRecord(
            timestamp=now,
            action="plan_generated",
            payload=json.loads(plan.json()),
        )
    )

    event_end = _ensure_utc(event.ts_end)
    bbox = _expand_bbox(event.aoi_bbox, plan.padding_km)
    lookback = timedelta(hours=plan.lookback_hours)
    lookahead = timedelta(hours=plan.lookahead_hours)
    window_start = event_end - lookback
    window_end = event_end + lookahead

    trace.append(
        ActionRecord(
            timestamp=now,
            action="cerulean_query_parameters",
            payload={
                "bbox": bbox,
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
                "min_source_score": plan.min_source_score,
                "only_active": plan.only_active,
                "filters": plan.filters,
                "sort_by": plan.sort_by,
                "limit": plan.limit,
            },
        )
    )

    result = client.query_slicks(
        bbox,
        start=window_start,
        end=window_end,
        limit=plan.limit,
        min_source_score=plan.min_source_score,
        only_active=plan.only_active,
        sort=plan.sort_by,
        extra_filters=plan.filters,
    )

    trace.append(
        ActionRecord(
            timestamp=_ensure_utc(datetime.now(timezone.utc)),
            action="cerulean_response",
            payload={
                "number_matched": result.number_matched,
                "number_returned": result.number_returned,
            },
        )
    )

    artifact_dir = cfg.artifact_root / (event.event_id or _slugify(event.ts_peak.isoformat()))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[Path] = []

    summary = summarize_slicks(result.slicks)
    summary_path = artifact_dir / "cerulean_summary.json"
    summary_path.write_text(
        _json_dumps(
            {
                "slick_count": summary.count,
                "active_count": summary.active_count,
                "total_area_km2": summary.total_area_km2,
                "avg_machine_confidence": summary.avg_machine_confidence,
                "max_source_collated_score": summary.max_source_collated_score,
                "source_counts": summary.source_counts,
            }
        )
    )
    artifacts.append(summary_path)

    geojson_path = None
    if result.slicks:
        geojson = build_feature_collection(result.slicks)
        geojson_path = artifact_dir / "cerulean.geojson"
        geojson_path.write_text(_json_dumps(geojson))
        artifacts.append(geojson_path)

    followup_scheduled = False
    followup_eta: Optional[datetime] = None
    if not result.slicks and plan.followup_delay_hours > 0:
        followup = schedule_followup(
            event.event_id or _slugify(event.ts_peak.isoformat()),
            reason="cerulean_refresh",
            delay=timedelta(hours=plan.followup_delay_hours),
            store=cfg.followup_store,
        )
        followup_scheduled = True
        followup_eta = followup.run_at
        trace.append(
            ActionRecord(
                timestamp=_ensure_utc(datetime.now(timezone.utc)),
                action="followup_scheduled",
                payload={"task_id": followup.task_id, "run_at": followup.run_at.isoformat()},
            ),
        )

    synopsis = model.synthesize(
        event,
        plan,
        result,
        followup_scheduled,
        followup_eta,
    )

    # update artifact references
    artifact_map = {}
    for path in artifacts:
        artifact_map[path.name] = str(path)

    synopsis.artifacts.update(artifact_map)

    synopsis_path = artifact_dir / "incident_synopsis.json"
    synopsis_path.write_text(_json_dumps(synopsis.model_dump(mode="json")))
    artifacts.append(synopsis_path)

    trace_path = artifact_dir / "agent_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for entry in trace:
            handle.write(_json_dumps(entry.model_dump(mode="json")) + "\n")

    return AgentRunResult(
        plan=plan,
        synopsis=synopsis,
        cerulean_result=result,
        artifacts=artifacts,
        trace_path=trace_path,
    )


def _expand_bbox(bbox: tuple[float, float, float, float], padding_km: float) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    if padding_km <= 0:
        return bbox
    # Approximate conversion: 1 degree lat ≈ 111 km
    lat_padding = padding_km / 111.0
    # For longitude, scale by cos(latitude) using midpoint
    mid_lat_rad = (min_lat + max_lat) / 2.0
    lon_scale = max(0.1, abs(_cosd(mid_lat_rad)))
    lon_padding = padding_km / (111.0 * lon_scale)
    return (
        min_lon - lon_padding,
        min_lat - lat_padding,
        max_lon + lon_padding,
        max_lat + lat_padding,
    )


def _cosd(deg: float) -> float:
    from math import cos, radians

    return cos(radians(deg))


def _slugify(text: str) -> str:
    return (
        text.replace(":", "")
        .replace("-", "")
        .replace("T", "")
        .replace("Z", "")
        .replace(".", "")
    )


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return obj


def _json_dumps(data) -> str:
    return json.dumps(data, indent=2, default=_json_default)
