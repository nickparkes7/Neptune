"""Typed schemas for GPT-5 agent planning and synopsis outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import Field, validator

from common.pydantic_compat import CompatBaseModel

ScenarioLabel = Literal["validation_context", "first_discovery"]


class QueryBounds(CompatBaseModel):
    """Configurable limits for agent-selected Cerulean parameters."""

    padding_km_min: float = 8.0
    padding_km_max: float = 12.0
    lookback_hours_min: int = 24
    lookback_hours_max: int = 36
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


class BriefMedia(CompatBaseModel):
    """Visual artifact linked to an observation in the agent brief."""

    label: str
    path: str
    asset_path: Optional[str] = None
    thumbnail: Optional[str] = None
    kind: Literal["image", "plot", "map", "document"] = "image"


class BriefObservation(CompatBaseModel):
    """Key evidence-backed finding in the agent brief."""

    id: str
    title: str
    summary: str
    impact: Optional[str] = None
    evidence: List[BriefMedia] = Field(default_factory=list)


class BriefAction(CompatBaseModel):
    """Recommended action for operators reviewing the brief."""

    id: str
    title: str
    summary: str
    urgency: Literal["low", "medium", "high", "critical"] = "high"


class BriefCitation(CompatBaseModel):
    """Mapping between claims and their supporting artifacts."""

    claim_id: str
    label: str
    path: str


class AgentBrief(CompatBaseModel):
    """Structured agent brief for rapid anomaly handoff."""

    scenario_id: str
    generated_at: datetime
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_label: Literal["low", "medium", "high", "critical"]
    headline: str
    summary: str
    hero_image: Optional[str] = None
    hero_caption: Optional[str] = None
    observations: List[BriefObservation] = Field(default_factory=list)
    recommended_actions: List[BriefAction] = Field(default_factory=list)
    citations: List[BriefCitation] = Field(default_factory=list)
    metrics: Dict[str, Union[float, int, str]] = Field(default_factory=dict)
    data_sources: Dict[str, Union[str, List[str]]] = Field(default_factory=dict)

    @validator("risk_label")
    def _ensure_label_matches_score(cls, value, values):  # noqa: D417
        score = values.get("risk_score")
        if score is None:
            return value
        tiers = {
            "low": (0.0, 0.34),
            "medium": (0.34, 0.67),
            "high": (0.67, 0.85),
            "critical": (0.85, 1.01),
        }
        for label, (lower, upper) in tiers.items():
            if lower <= score < upper:
                if value != label:
                    raise ValueError(
                        f"risk_label '{value}' inconsistent with score {score:.2f}; expected '{label}'"
                    )
                break
        return value
