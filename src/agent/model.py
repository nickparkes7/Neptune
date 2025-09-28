"""Agent model interfaces and default rule-based fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from anomaly.events import SuspectedSpillEvent
from cerulean import CeruleanQueryResult, summarize_slicks

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

    default_bounds: QueryBounds = QueryBounds()

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
            only_active=True,
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
