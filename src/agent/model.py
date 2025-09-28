"""Agent model interfaces and default rule-based fallback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Protocol

from anomaly.events import SuspectedSpillEvent
from cerulean import CeruleanQueryResult, summarize_slicks
try:  # Optional import so tests without the package still run
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore[assignment]

from .schemas import AgentPlan, IncidentSynopsis, QueryBounds, ScenarioLabel, SynopsisMetrics


class AgentModel(Protocol):
    """Protocol for GPT-5 backed agent models."""

    def generate_plan(self, event: SuspectedSpillEvent, bounds: QueryBounds) -> AgentPlan:
        ...

    def synthesize(
        self,
        event: SuspectedSpillEvent,
        plan: AgentPlan,
        result: CeruleanQueryResult,
        followup_scheduled: bool,
        followup_eta: datetime | None,
    ) -> IncidentSynopsis:
        ...


@dataclass
class RuleBasedAgentModel:
    """Deterministic fallback when GPT-5 is unavailable.

    The heuristics are intentionally simple but respect the same schema as the
    GPT-backed model so integration points remain identical.
    """

    default_bounds: QueryBounds = field(default_factory=QueryBounds)

    def generate_plan(self, event: SuspectedSpillEvent, bounds: QueryBounds | None = None) -> AgentPlan:
        bounds = bounds or self.default_bounds
        oil_peak = event.oil_stats.max_z
        duration_hours = max(event.duration_s / 3600.0, 0.1)

        padding = min(bounds.padding_km_max, max(bounds.padding_km_min, oil_peak * 2))
        lookback = min(bounds.lookback_hours_max, max(bounds.lookback_hours_min, int(12 + duration_hours * 2)))
        lookahead = min(bounds.lookahead_hours_max, bounds.lookahead_hours_min + 6)
        min_score = 0.0 if oil_peak >= 2.0 else 0.2
        limit = min(bounds.limit_max, max(bounds.limit_min, int(100 + duration_hours * 20)))
        filters: list[str] = []
        rationale = (
            "Rule-based fallback: padding scaled with oil peak, "
            "retaining active slicks and broadening lookback for longer events."
        )
        return AgentPlan(
            padding_km=padding,
            lookback_hours=lookback,
            lookahead_hours=lookahead,
            min_source_score=min_score,
            only_active=False,
            filters=filters,
            sort_by="-max_source_collated_score",
            limit=limit,
            followup_delay_hours=24,
            rationale=rationale,
        )

    def synthesize(
        self,
        event: SuspectedSpillEvent,
        plan: AgentPlan,
        result: CeruleanQueryResult,
        followup_scheduled: bool,
        followup_eta: datetime | None,
    ) -> IncidentSynopsis:
        slicks = result.slicks
        metrics = summarize_slicks(slicks)
        metrics_model = SynopsisMetrics(
            slick_count=metrics.count,
            active_count=metrics.active_count,
            total_area_km2=metrics.total_area_km2,
            avg_machine_confidence=metrics.avg_machine_confidence,
            max_source_collated_score=metrics.max_source_collated_score,
            source_counts=metrics.source_counts,
        )
        scenario: ScenarioLabel
        confidence: float
        summary_lines = []
        if slicks:
            scenario = "validation_context"
            confidence = min(1.0, 0.6 + (metrics.max_source_collated_score or 0) * 0.4)
            summary_lines.append(
                f"Cerulean returned {metrics.count} slick(s) with max source score "
                f"{metrics.max_source_collated_score or 'n/a'}; in-situ peak z={event.oil_stats.max_z:.2f}."
            )
            summary_lines.append("Treat as validated spill; coordinate with response.")
            actions = [
                "Fuse onboard samples with Cerulean polygons for localized response",
                "Notify ops with attached synopsis and overlays",
            ]
        else:
            scenario = "first_discovery"
            confidence = 0.7
            summary_lines.append(
                "No Cerulean slick matched the AOI/time window; onboard sensors remain elevated."
            )
            summary_lines.append("Flag as first discovery and monitor follow-up query once Cerulean refreshes.")
            actions = [
                "Maintain onboard sampling cadence",
                "Inspect upcoming Cerulean run once follow-up is due",
            ]
        if followup_scheduled and followup_eta is not None:
            actions.append("Re-run Cerulean query at scheduled follow-up time.")

        return IncidentSynopsis(
            scenario=scenario,
            confidence=confidence,
            summary=" ".join(summary_lines),
            rationale=plan.rationale,
            recommended_actions=actions,
            metrics=metrics_model,
            followup_scheduled=followup_scheduled,
            followup_eta=followup_eta,
        )


@dataclass
class GPTAgentModel:
    """OpenAI GPT-5 powered agent model implementation."""

    model_name: str = "gpt-5-mini"
    temperature: float = 1.0
    max_retries: int = 2
    api_key_env: str = "OPENAI_API_KEY"
    client: Any | None = None

    def _get_client(self):
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' Python package is not installed. Run 'uv sync' and ensure OPENAI_API_KEY is set."
            )
        if self.client is not None:
            return self.client
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenAI API key. Set {self.api_key_env} before running the agent."
            )
        self.client = OpenAI(api_key=api_key)
        return self.client

    def generate_plan(self, event: SuspectedSpillEvent, bounds: QueryBounds | None = None) -> AgentPlan:
        bounds = bounds or QueryBounds()
        payload = self._plan_payload(event, bounds)
        prompt = self._plan_prompt(payload)
        response = self._invoke_model(prompt)
        plan_dict = response.get("plan") if isinstance(response, dict) else response
        if not isinstance(plan_dict, dict):
            raise RuntimeError("GPT-5 returned invalid plan payload")
        plan = AgentPlan(**plan_dict)
        return self._clamp_plan(plan, bounds)

    def synthesize(
        self,
        event: SuspectedSpillEvent,
        plan: AgentPlan,
        result: CeruleanQueryResult,
        followup_scheduled: bool,
        followup_eta: datetime | None,
    ) -> IncidentSynopsis:
        payload = self._synopsis_payload(event, plan, result, followup_scheduled, followup_eta)
        prompt = self._synopsis_prompt(payload)
        response = self._invoke_model(prompt)
        data = response.get("synopsis") if isinstance(response, dict) else response
        if not isinstance(data, dict):
            raise RuntimeError("GPT-5 returned invalid synopsis payload")
        if "metrics" not in data:
            data["metrics"] = {}
        if "recommended_actions" not in data:
            data["recommended_actions"] = []
        if "artifacts" not in data:
            data["artifacts"] = {}
        # Ensure all artifact values are strings
        if "artifacts" in data and isinstance(data["artifacts"], dict):
            for key, value in data["artifacts"].items():
                if not isinstance(value, str):
                    data["artifacts"][key] = json.dumps(value)
        if "followup_scheduled" not in data:
            data["followup_scheduled"] = followup_scheduled
        if "followup_eta" not in data and followup_eta is not None:
            data["followup_eta"] = followup_eta.isoformat()
        synopsis = IncidentSynopsis(**data)
        synopsis.confidence = max(0.0, min(1.0, synopsis.confidence))
        return synopsis

    # ------------------------------------------------------------------
    # GPT helpers
    # ------------------------------------------------------------------

    def _invoke_model(self, prompt: Dict[str, str]) -> Dict[str, Any]:
        client = self._get_client()
        errors = []
        for attempt in range(self.max_retries):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                )
                content = completion.choices[0].message.content.strip()
                return json.loads(content)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        raise RuntimeError(f"GPT-5 call failed after {self.max_retries} attempts: {errors}")

    def _plan_payload(self, event: SuspectedSpillEvent, bounds: QueryBounds) -> Dict[str, Any]:
        return {
            "event": {
                "event_id": event.event_id,
                "lat": event.lat,
                "lon": event.lon,
                "duration_s": event.duration_s,
                "oil_stats": event.oil_stats.model_dump(mode="json"),
                "context_channels": {k: v.model_dump(mode="json") for k, v in event.context_channels.items()},
                "aoi_bbox": event.aoi_bbox,
                "ts_start": event.ts_start.isoformat(),
                "ts_end": event.ts_end.isoformat(),
            },
            "bounds": bounds.model_dump(mode="json") if hasattr(bounds, "model_dump") else bounds.dict(),
        }

    def _plan_prompt(self, payload: Dict[str, Any]) -> Dict[str, str]:
        schema = {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "properties": {
                        "padding_km": {"type": "number"},
                        "lookback_hours": {"type": "integer"},
                        "lookahead_hours": {"type": "integer"},
                        "min_source_score": {"type": "number"},
                        "only_active": {"type": "boolean"},
                        "filters": {"type": "array", "items": {"type": "string"}},
                        "sort_by": {"type": "string"},
                        "limit": {"type": "integer"},
                        "followup_delay_hours": {"type": "integer"},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "padding_km",
                        "lookback_hours",
                        "lookahead_hours",
                        "min_source_score",
                        "only_active",
                        "filters",
                        "sort_by",
                        "limit",
                        "followup_delay_hours",
                        "rationale",
                    ],
                }
            },
        }
        return {
            "system": (
                "You are Neptune's GPT-5 planning agent. Respond only with valid JSON matching the provided schema. "
                "Respect the bounds for query parameters and explain rationale succinctly."
            ),
            "user": json.dumps({"payload": payload, "schema": schema}, indent=2),
        }

    def _synopsis_payload(
        self,
        event: SuspectedSpillEvent,
        plan: AgentPlan,
        result: CeruleanQueryResult,
        followup_scheduled: bool,
        followup_eta: datetime | None,
    ) -> Dict[str, Any]:
        summary = summarize_slicks(result.slicks)
        return {
            "event": {
                "event_id": event.event_id,
                "lat": event.lat,
                "lon": event.lon,
                "duration_s": event.duration_s,
                "oil_stats": event.oil_stats.model_dump(mode="json"),
            },
            "plan": plan.model_dump(mode="json"),
            "cerulean": {
                "number_matched": result.number_matched,
                "number_returned": result.number_returned,
                "summary": {
                    "slick_count": summary.count,
                    "active_count": summary.active_count,
                    "total_area_km2": summary.total_area_km2,
                    "avg_machine_confidence": summary.avg_machine_confidence,
                    "max_source_collated_score": summary.max_source_collated_score,
                    "source_counts": summary.source_counts,
                },
                "samples": [
                    {
                        "id": slick.id,
                        "area": slick.area,
                        "machine_confidence": slick.machine_confidence,
                        "max_source_collated_score": slick.max_source_collated_score,
                        "active": slick.active,
                        "source_type_1_ids": slick.source_type_1_ids,
                        "source_type_2_ids": slick.source_type_2_ids,
                        "source_type_3_ids": slick.source_type_3_ids,
                    }
                    for slick in result.slicks[:5]
                ],
            },
            "followup": {
                "scheduled": followup_scheduled,
                "eta": followup_eta.isoformat() if followup_eta else None,
            },
        }

    def _synopsis_prompt(self, payload: Dict[str, Any]) -> Dict[str, str]:
        schema = {
            "type": "object",
            "properties": {
                "synopsis": {
                    "type": "object",
                    "properties": {
                        "scenario": {"type": "string"},
                        "confidence": {"type": "number"},
                        "summary": {"type": "string"},
                        "rationale": {"type": "string"},
                        "recommended_actions": {"type": "array", "items": {"type": "string"}},
                        "metrics": {"type": "object"},
                        "artifacts": {"type": "object"},
                        "followup_scheduled": {"type": "boolean"},
                        "followup_eta": {"type": ["string", "null"]},
                    },
                    "required": [
                        "scenario",
                        "confidence",
                        "summary",
                        "rationale",
                        "recommended_actions",
                        "metrics",
                        "followup_scheduled",
                    ],
                }
            },
        }
        return {
            "system": (
                "You are Neptune's GPT-5 incident analyst. Choose scenario 'validation_context' or 'first_discovery'. "
                "Provide concise summary, rationales, and recommended actions. Respond only with JSON matching schema."
            ),
            "user": json.dumps({"payload": payload, "schema": schema}, indent=2),
        }

    def _clamp_plan(self, plan: AgentPlan, bounds: QueryBounds) -> AgentPlan:
        return AgentPlan(
            padding_km=_clamp(plan.padding_km, bounds.padding_km_min, bounds.padding_km_max),
            lookback_hours=int(_clamp(plan.lookback_hours, bounds.lookback_hours_min, bounds.lookback_hours_max)),
            lookahead_hours=int(_clamp(plan.lookahead_hours, bounds.lookahead_hours_min, bounds.lookahead_hours_max)),
            min_source_score=_clamp(plan.min_source_score, bounds.min_source_score_min, bounds.min_source_score_max),
            only_active=plan.only_active,
            filters=plan.filters,
            sort_by=plan.sort_by,
            limit=int(_clamp(plan.limit, bounds.limit_min, bounds.limit_max)),
            followup_delay_hours=int(max(0, plan.followup_delay_hours)),
            rationale=plan.rationale,
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
