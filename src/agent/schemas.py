"""Typed schemas for GPT-5 agent planning and synopsis outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import Field, validator

from common.pydantic_compat import CompatBaseModel

ScenarioLabel = Literal["validation_context", "first_discovery"]


class QueryBounds(CompatBaseModel):
    """Configurable limits for agent-selected Cerulean parameters."""

    padding_km_min: float = 5.0
    padding_km_max: float = 30.0
    lookback_hours_min: int = 12
    lookback_hours_max: int = 96
    lookahead_hours_min: int = 0
    lookahead_hours_max: int = 48
    min_source_score_min: float = 0.0
    min_source_score_max: float = 0.9
    limit_min: int = 25
    limit_max: int = 500


class AgentPlan(CompatBaseModel):
    """Actionable plan decoded from the GPT-5 response."""

    padding_km: float = Field(..., description="Additional AOI padding in km")
    lookback_hours: int = Field(..., description="Hours before event end to include")
    lookahead_hours: int = Field(..., description="Hours after event end to include")
    min_source_score: float = Field(..., description="Minimum Cerulean source score")
    only_active: bool = Field(False, description="Only return active slicks")
    filters: List[str] = Field(default_factory=list, description="Additional CQL filters")
    sort_by: str = Field("-max_source_collated_score", description="Sort expression")
    limit: int = Field(100, description="Maximum slicks to return")
    followup_delay_hours: int = Field(24, description="Delay before re-query scheduling")
    rationale: str = Field("GPT-5 plan rationale", description="Natural language reasoning")

    @validator("padding_km", "min_source_score", "limit")
    def _non_negative(cls, value):  # noqa: D417
        if isinstance(value, (int, float)) and value < 0:
            raise ValueError("plan parameters must be non-negative")
        return value

    @validator("lookback_hours", "lookahead_hours", "followup_delay_hours")
    def _non_negative_int(cls, value):  # noqa: D417
        if value < 0:
            raise ValueError("plan hours must be non-negative")
        return value


class SynopsisMetrics(CompatBaseModel):
    """Structured metrics for dashboards."""

    slick_count: int = 0
    active_count: int = 0
    total_area_km2: float = 0.0
    avg_machine_confidence: Optional[float] = None
    max_source_collated_score: Optional[float] = None
    source_counts: Dict[str, int] = Field(default_factory=dict)


class IncidentSynopsis(CompatBaseModel):
    """Structured summary produced by the agent."""

    scenario: ScenarioLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    rationale: str
    recommended_actions: List[str]
    metrics: SynopsisMetrics
    artifacts: Dict[str, str] = Field(default_factory=dict)
    followup_scheduled: bool = False
    followup_eta: Optional[datetime] = None


class ActionRecord(CompatBaseModel):
    """Trace entry for agent steps."""

    timestamp: datetime
    action: str
    payload: Dict[str, object]
