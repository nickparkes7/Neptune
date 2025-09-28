"""Utilities to assemble visually rich agent briefs from cached evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anomaly.hybrid import HybridOilAlertScorer
from anomaly.events import SuspectedSpillEvent

from .schemas import (
    AgentBrief,
    BriefAction,
    BriefCitation,
    BriefMedia,
    BriefObservation,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ArtifactRef:
    """Reference to a generated visual artifact."""

    label: str
    url: str
    asset_path: Path
    kind: str = "image"
    thumbnail: Optional[str] = None

    def to_media(self) -> BriefMedia:
        return BriefMedia(
            label=self.label,
            path=self.url,
            asset_path=_relative_str(self.asset_path),
            thumbnail=self.thumbnail,
            kind=self.kind,  # type: ignore[arg-type]
        )


def score_stream(stream_path: Path) -> pd.DataFrame:
    """Score the SeaOWL stream and return a feature-rich DataFrame."""

    scorer = HybridOilAlertScorer()
    frame = scorer.score_ndjson(stream_path)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame.sort_values("ts").reset_index(drop=True)
    frame.attrs.update(
        {
            "z_warn": scorer.cfg.z_warn,
            "z_alarm": scorer.cfg.z_alarm,
            "abs_warn": scorer.cfg.abs_warn,
            "abs_alarm": scorer.cfg.abs_alarm,
        }
    )
    return frame


def render_brief_media(
    frame: pd.DataFrame,
    media_dir: Path,
    *,
    media_url_base: str,
) -> Dict[str, ArtifactRef]:
    """Render the core visual assets for the agent brief."""

    media_dir.mkdir(parents=True, exist_ok=True)

    timeseries_path = media_dir / "seaowl_timeseries.png"
    track_path = media_dir / "seaowl_track.png"
    hybrid_values_path = media_dir / "hybrid_values.png"
    hybrid_scores_path = media_dir / "hybrid_scores.png"

    _render_timeseries_plot(frame, timeseries_path)
    _render_track_plot(frame, track_path)
    _render_hybrid_values_plot(frame, hybrid_values_path)
    _render_hybrid_scores_plot(frame, hybrid_scores_path, frame.attrs.get("z_warn"), frame.attrs.get("z_alarm"))

    def _url(name: str) -> str:
        return f"{media_url_base.rstrip('/')}/{name}"

    return {
        "seaowl_timeseries": ArtifactRef(
            label="SeaOWL time-series",
            url=_url("seaowl_timeseries.png"),
            asset_path=timeseries_path,
            kind="plot",
        ),
        "seaowl_track": ArtifactRef(
            label="Track heatmap",
            url=_url("seaowl_track.png"),
            asset_path=track_path,
            kind="map",
        ),
        "hybrid_values": ArtifactRef(
            label="Hybrid alert values",
            url=_url("hybrid_values.png"),
            asset_path=hybrid_values_path,
            kind="plot",
        ),
        "hybrid_scores": ArtifactRef(
            label="Hybrid alert z-scores",
            url=_url("hybrid_scores.png"),
            asset_path=hybrid_scores_path,
            kind="plot",
        ),
    }


def slice_frame_for_event(
    frame: pd.DataFrame,
    event: SuspectedSpillEvent,
    *,
    pre_seconds: int = 600,
    post_seconds: int = 600,
) -> pd.DataFrame:
    """Return a window around the event to keep plots focused."""

    if frame.empty:
        return frame

    work = frame.copy()
    work.attrs = frame.attrs.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    work = work.dropna(subset=["ts"]).reset_index(drop=True)

    start = (_ensure_utc(event.ts_start) - timedelta(seconds=pre_seconds)).to_pydatetime()
    end = (_ensure_utc(event.ts_end) + timedelta(seconds=post_seconds)).to_pydatetime()

    mask = (work["ts"] >= start) & (work["ts"] <= end)
    sliced = work.loc[mask].reset_index(drop=True)
    sliced.attrs = frame.attrs.copy()
    if sliced.empty:
        return work
    return sliced


def generate_agent_brief_for_event(
    event: SuspectedSpillEvent,
    *,
    stream_path: Path,
    output_dir: Path,
    media_url_base: str,
    pre_seconds: int = 600,
    post_seconds: int = 600,
) -> Tuple[AgentBrief, List[Path]]:
    """Generate and persist an agent brief for a specific incident."""

    frame = score_stream(stream_path)
    sliced = slice_frame_for_event(frame, event, pre_seconds=pre_seconds, post_seconds=post_seconds)
    if sliced.empty:
        sliced = frame

    media_dir = output_dir / "media"
    artifacts = render_brief_media(sliced, media_dir, media_url_base=media_url_base)

    scenario_id = event.event_id or "seaowl_event"
    brief = build_agent_brief(
        scenario_id=scenario_id,
        stream_path=stream_path,
        frame=sliced,
        artifacts=artifacts,
        hero_artifact_key="seaowl_timeseries",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "brief.json"
    md_path = output_dir / "brief.md"
    json_path.write_text(json.dumps(brief.model_dump(mode="json"), indent=2))
    md_path.write_text(brief_to_markdown(brief))

    return brief, [json_path, md_path]


def build_agent_brief(
    *,
    scenario_id: str,
    stream_path: Path,
    frame: pd.DataFrame,
    artifacts: Mapping[str, ArtifactRef],
    hero_artifact_key: str,
    generated_at: Optional[datetime] = None,
) -> AgentBrief:
    """Create an :class:`AgentBrief` from scored telemetry and visual assets."""

    if frame.empty:
        raise ValueError("frame must contain scored telemetry")

    generated_at = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)

    oil_series = pd.to_numeric(frame.get("oil_fluor_ppb"), errors="coerce")
    baseline_series = pd.to_numeric(frame.get("oil_baseline"), errors="coerce")
    z_series = pd.to_numeric(frame.get("oil_z_eff"), errors="coerce")

    peak_idx = int(oil_series.idxmax()) if not oil_series.isna().all() else frame.index[-1]
    peak_row = frame.iloc[peak_idx]
    peak_ppb = float(oil_series.max()) if not oil_series.isna().all() else float("nan")
    peak_z = float(z_series.max()) if not z_series.isna().all() else float("nan")
    baseline_at_peak = float(baseline_series.iloc[peak_idx]) if not baseline_series.isna().all() else float("nan")

    warn_count = int(pd.Series(frame.get("oil_warn", []), dtype=float).fillna(0).sum())
    alarm_count = int(pd.Series(frame.get("oil_alarm", []), dtype=float).fillna(0).sum())

    event_mask = _event_mask(frame)
    event_frame = frame.loc[event_mask]
    if event_frame.empty:
        event_frame = frame.tail(min(len(frame), 120))

    event_duration_s = _event_duration_seconds(event_frame)
    event_duration_min = event_duration_s / 60.0 if event_duration_s else 0.0

    lat = _safe_float(peak_row.get("lat"))
    lon = _safe_float(peak_row.get("lon"))
    drift_m = _event_drift(event_frame)

    risk_score = _risk_score(peak_z, alarm_count, event_duration_min)
    risk_label = _risk_label(risk_score)

    headline = _headline(lat, lon)
    summary = _summary_text(
        peak_ppb=peak_ppb,
        peak_z=peak_z,
        event_duration_min=event_duration_min,
        drift_m=drift_m,
        alarm_count=alarm_count,
        lat=lat,
        lon=lon,
    )

    hero_artifact = artifacts.get(hero_artifact_key)
    hero_image = hero_artifact.url if hero_artifact else None

    observations = _build_observations(
        frame=frame,
        peak_ppb=peak_ppb,
        peak_z=peak_z,
        baseline_at_peak=baseline_at_peak,
        drift_m=drift_m,
        alarm_count=alarm_count,
        artifacts=artifacts,
        lat=lat,
        lon=lon,
        event_duration_min=event_duration_min,
    )

    actions = _build_actions(alarm_count, drift_m, risk_label)

    citations = _build_citations(observations)

    metrics = _metrics_dict(
        peak_ppb=peak_ppb,
        peak_z=peak_z,
        baseline=baseline_at_peak,
        alarm_count=alarm_count,
        warn_count=warn_count,
        event_duration_min=event_duration_min,
        drift_m=drift_m,
    )

    data_sources: Dict[str, str | List[str]] = {
        "stream": _relative_str(stream_path),
        "frames": f"{len(frame)} samples",
    }

    artifact_paths = [_relative_str(ref.asset_path) for ref in artifacts.values()]
    if artifact_paths:
        data_sources["artifacts"] = sorted(set(artifact_paths))

    brief = AgentBrief(
        scenario_id=scenario_id,
        generated_at=generated_at,
        risk_score=risk_score,
        risk_label=risk_label,
        headline=headline,
        summary=summary,
        hero_image=hero_image,
        hero_caption="SeaOWL fluorescence with hybrid alerts highlighted",
        observations=observations,
        recommended_actions=actions,
        citations=citations,
        metrics=metrics,
        data_sources=data_sources,
    )
    return brief


def brief_to_markdown(brief: AgentBrief) -> str:
    """Render a human-friendly Markdown version of the brief."""

    generated = brief.generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    lines.append(f"# Agent Brief — {brief.headline}")
    lines.append("")
    lines.append(
        f"**Risk: {brief.risk_label.title()} ({brief.risk_score:.0%}) · Scenario: {brief.scenario_id} · Generated {generated}**"
    )
    lines.append("")
    lines.append(brief.summary)
    lines.append("")

    if brief.hero_image:
        lines.append(f"![Hero image]({brief.hero_image})")
        if brief.hero_caption:
            lines.append(f"*{brief.hero_caption}*")
        lines.append("")

    if brief.metrics:
        lines.append("## Key Metrics")
        for key, value in brief.metrics.items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        lines.append("")

    if brief.observations:
        lines.append("## Observations")
        for obs in brief.observations:
            lines.append(f"### {obs.title}")
            lines.append(obs.summary)
            if obs.impact:
                lines.append(f"*Impact: {obs.impact}*")
            for media in obs.evidence:
                lines.append("")
                lines.append(f"![{media.label}]({media.path})")
            lines.append("")

    if brief.recommended_actions:
        lines.append("## Recommended Actions")
        for action in brief.recommended_actions:
            lines.append(f"- **{action.title}** ({action.urgency.title()}): {action.summary}")
        lines.append("")

    if brief.data_sources:
        lines.append("## Data Sources")
        for key, value in brief.data_sources.items():
            if isinstance(value, list):
                joined = ", ".join(str(item) for item in value)
                lines.append(f"- {key}: {joined}")
            else:
                lines.append(f"- {key}: {value}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _relative_str(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _safe_float(value) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


def _render_timeseries_plot(frame: pd.DataFrame, path: Path) -> None:
    ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    oil = pd.to_numeric(frame.get("oil_fluor_ppb"), errors="coerce")
    baseline = pd.to_numeric(frame.get("oil_baseline"), errors="coerce")
    warn = frame.get("oil_warn", pd.Series([False] * len(frame))).astype(bool)
    alarm = frame.get("oil_alarm", pd.Series([False] * len(frame))).astype(bool)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(ts, oil, color="#38bdf8", linewidth=2.2, label="oil fluorescence")

    if not baseline.isna().all():
        ax.plot(ts, baseline, color="#f97316", linewidth=1.2, alpha=0.85, label="adaptive baseline")

    if warn.any():
        ax.scatter(ts[warn], oil[warn], color="#facc15", s=18, label="warn")
    if alarm.any():
        ax.scatter(ts[alarm], oil[alarm], color="#f87171", s=22, label="alarm", zorder=4)

    event_mask = _event_mask(frame)
    if event_mask.any():
        ymin, ymax = ax.get_ylim()
        ax.fill_between(ts, ymin, ymax, where=event_mask, color="#0ea5e9", alpha=0.12, linewidth=0)
        ax.set_ylim(ymin, ymax)

    ax.set_title("SeaOWL Oil Fluorescence")
    ax.set_ylabel("ppb")
    ax.grid(color="#1e293b", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _render_track_plot(frame: pd.DataFrame, path: Path) -> None:
    lat = pd.to_numeric(frame.get("lat"), errors="coerce")
    lon = pd.to_numeric(frame.get("lon"), errors="coerce")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    valid = ~(lat.isna() | lon.isna())
    lat = lat[valid]
    lon = lon[valid]

    if lat.empty:
        ax.text(0.5, 0.5, "No track data", ha="center", va="center", color="#94a3b8")
    else:
        ax.plot(lon, lat, color="#38bdf8", linewidth=2.0, alpha=0.85)
        sc = ax.scatter(lon, lat, c=np.linspace(0, 1, len(lat)), cmap="viridis", s=18)
        fig.colorbar(sc, ax=ax, label="Track progression")
        event_mask = _event_mask(frame.loc[valid])
        event_lat = lat[event_mask]
        event_lon = lon[event_mask]
        if not event_lat.empty:
            ax.scatter(event_lon, event_lat, color="#f87171", s=26, label="anomaly window")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="lower right")

    ax.set_title("Track Trace (last window)")
    ax.grid(color="#1e293b", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _render_hybrid_values_plot(frame: pd.DataFrame, path: Path) -> None:
    ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    oil = pd.to_numeric(frame.get("oil_fluor_ppb"), errors="coerce")
    baseline = pd.to_numeric(frame.get("oil_baseline"), errors="coerce")
    warn = frame.get("oil_warn", pd.Series([False] * len(frame))).astype(bool)
    alarm = frame.get("oil_alarm", pd.Series([False] * len(frame))).astype(bool)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(ts, oil, color="#38bdf8", linewidth=2.0, label="oil")
    if not baseline.isna().all():
        ax.plot(ts, baseline, color="#f97316", linewidth=1.0, alpha=0.8, label="baseline")
    if warn.any():
        ax.scatter(ts[warn], oil[warn], color="#facc15", s=16, label="warn")
    if alarm.any():
        ax.scatter(ts[alarm], oil[alarm], color="#f87171", s=20, label="alarm")
    ax.set_title("Hybrid Alert — Oil Fluorescence vs Baseline")
    ax.set_ylabel("ppb")
    ax.grid(color="#1e293b", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _render_hybrid_scores_plot(
    frame: pd.DataFrame,
    path: Path,
    z_warn: Optional[float],
    z_alarm: Optional[float],
) -> None:
    ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    zeff = pd.to_numeric(frame.get("oil_z_eff"), errors="coerce")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(ts, zeff, color="#c4b5fd", linewidth=1.8, label="z_eff")
    if z_warn is not None:
        ax.axhline(z_warn, color="#facc15", linestyle="--", linewidth=1.2, label="warn threshold")
    if z_alarm is not None:
        ax.axhline(z_alarm, color="#f87171", linestyle="--", linewidth=1.2, label="alarm threshold")
    ax.set_title("Hybrid Alert — Effective Z-Score")
    ax.set_ylabel("z-score")
    ax.grid(color="#1e293b", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
def _event_mask(frame: pd.DataFrame) -> pd.Series:
    warn = pd.Series(frame.get("oil_warn", False)).astype(bool)
    alarm = pd.Series(frame.get("oil_alarm", False)).astype(bool)
    phase = pd.Series(frame.get("event_phase", 0.0))
    return warn | alarm | (pd.to_numeric(phase, errors="coerce") > 0.05)


def _event_duration_seconds(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    timestamps = pd.to_datetime(frame["ts"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return 0.0
    delta = timestamps.iloc[-1] - timestamps.iloc[0]
    return max(delta.total_seconds(), 0.0)


def _event_drift(frame: pd.DataFrame) -> float:
    if frame.empty or "lat" not in frame or "lon" not in frame:
        return 0.0
    lat = pd.to_numeric(frame["lat"], errors="coerce").dropna()
    lon = pd.to_numeric(frame["lon"], errors="coerce").dropna()
    if lat.empty or lon.empty:
        return 0.0
    start = (lat.iloc[0], lon.iloc[0])
    end = (lat.iloc[-1], lon.iloc[-1])
    return _haversine_meters(start, end)


def _haversine_meters(a: Iterable[float], b: Iterable[float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    if any(np.isnan([lat1, lon1, lat2, lon2])):
        return 0.0
    r = 6371_000  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(max(h, 0.0)))


def _risk_score(peak_z: float, alarm_count: int, event_duration_min: float) -> float:
    score = 0.35
    if not math.isnan(peak_z):
        score += min(0.45, max(peak_z, 0.0) / 200.0)
    if alarm_count > 0:
        score += min(0.15, alarm_count / 200.0)
    if event_duration_min > 6:
        score += 0.05
    return max(0.05, min(score, 0.98))


def _risk_label(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def _headline(lat: float, lon: float) -> str:
    if math.isnan(lat) or math.isnan(lon):
        return "SeaOWL anomaly detected"
    return f"SeaOWL anomaly near {lat:.4f}°N, {lon:.4f}°E"


def _summary_text(
    *,
    peak_ppb: float,
    peak_z: float,
    event_duration_min: float,
    drift_m: float,
    alarm_count: int,
    lat: float,
    lon: float,
) -> str:
    location = "unknown location"
    if not math.isnan(lat) and not math.isnan(lon):
        location = f"{lat:.4f}°N, {lon:.4f}°E"

    peak_desc = f"peak oil fluorescence at {peak_ppb:.1f} ppb" if not math.isnan(peak_ppb) else "oil fluorescence spike"
    z_desc = f" (z≈{peak_z:.1f})" if not math.isnan(peak_z) else ""
    duration_desc = f" over {event_duration_min:.1f} min" if event_duration_min else ""
    drift_desc = f", drifting {drift_m/1000:.1f} km" if drift_m else ""
    alarm_desc = f"; hybrid alert captured {alarm_count} alarm frames" if alarm_count else ""

    return (
        f"SeaOWL detected {peak_desc}{z_desc}{duration_desc} near {location}{drift_desc}{alarm_desc}."
    )


def _build_observations(
    *,
    frame: pd.DataFrame,
    peak_ppb: float,
    peak_z: float,
    baseline_at_peak: float,
    drift_m: float,
    alarm_count: int,
    artifacts: Mapping[str, ArtifactRef],
    lat: float,
    lon: float,
    event_duration_min: float,
) -> List[BriefObservation]:
    observations: List[BriefObservation] = []

    baseline_text = "baseline" if math.isnan(baseline_at_peak) else f"baseline {baseline_at_peak:.2f} ppb"
    obs1_summary = (
        f"Oil fluorescence climbed to {peak_ppb:.1f} ppb (z≈{peak_z:.1f}) from {baseline_text}"
        if not math.isnan(peak_ppb) and not math.isnan(peak_z)
        else "Oil fluorescence exceeded configured thresholds"
    )
    obs1_summary += f" with {alarm_count} alarm frames." if alarm_count else "."

    obs1 = BriefObservation(
        id="obs_fluorescence_peak",
        title="Sustained fluorescence spike",
        summary=obs1_summary,
        impact="Indicates dense oil-like material persisting above both absolute and adaptive thresholds.",
        evidence=[artifacts["seaowl_timeseries"].to_media()] if "seaowl_timeseries" in artifacts else [],
    )
    observations.append(obs1)

    location_text = "position fix unavailable"
    if not math.isnan(lat) and not math.isnan(lon):
        location_text = f"clustered around {lat:.4f}°N, {lon:.4f}°E"

    motion_text = "minimal drift" if drift_m < 200 else f"~{drift_m/1000:.1f} km drift"
    obs2 = BriefObservation(
        id="obs_track_behavior",
        title="Track compresses inside AOI",
        summary=(
            f"Track points remained {location_text} with {motion_text} during the anomaly window"
            + (f" (~{event_duration_min:.1f} min)." if event_duration_min else ".")
        ),
        impact="Suggests vessel loitering or slow maneuver near slick extent.",
        evidence=[artifacts["seaowl_track"].to_media()] if "seaowl_track" in artifacts else [],
    )
    observations.append(obs2)

    obs3 = BriefObservation(
        id="obs_hybrid_context",
        title="Hybrid alert confidence",
        summary=(
            "Hybrid scorer maintained elevated z-eff values"
            + (f" peaking at {peak_z:.1f}" if not math.isnan(peak_z) else " throughout the event")
            + ", reinforcing the alarm classification."
        ),
        impact="Cross-validated anomaly reduces false positives from biogenic fluorescence spikes.",
        evidence=[artifacts["hybrid_values"].to_media()] if "hybrid_values" in artifacts else [],
    )
    observations.append(obs3)

    return observations


def _build_actions(alarm_count: int, drift_m: float, risk_label: str) -> List[BriefAction]:
    actions: List[BriefAction] = []

    urgency = "critical" if risk_label == "critical" else "high"
    actions.append(
        BriefAction(
            id="act_monitor",
            title="Maintain incident watch",
            summary="Keep SeaOWL telemetry pinned and log manual observations for the next 30 minutes.",
            urgency=urgency,
        )
    )

    actions.append(
        BriefAction(
            id="act_task_sat",
            title="Queue Sentinel-1 follow-up",
            summary="Submit repeat-pass request covering the anomaly coordinates for the next available orbit.",
            urgency="high",
        )
    )

    drift_text = "Perform AIS/vessel context query within 25 km." if drift_m < 1000 else "Run drift backtrack and broaden AIS search radius to 40 km."
    actions.append(
        BriefAction(
            id="act_context",
            title="Broaden context filters",
            summary=drift_text,
            urgency="medium",
        )
    )

    if alarm_count > 50:
        actions.append(
            BriefAction(
                id="act_notify",
                title="Notify regional response",
                summary="Escalate to regional response lead with brief and supporting visuals attached.",
                urgency="high",
            )
        )

    return actions


def _build_citations(observations: List[BriefObservation]) -> List[BriefCitation]:
    citations: List[BriefCitation] = []
    for obs in observations:
        for media in obs.evidence:
            citations.append(BriefCitation(claim_id=obs.id, label=media.label, path=media.path))
    return citations


def _metrics_dict(
    *,
    peak_ppb: float,
    peak_z: float,
    baseline: float,
    alarm_count: int,
    warn_count: int,
    event_duration_min: float,
    drift_m: float,
) -> Dict[str, str | float | int]:
    metrics: Dict[str, str | float | int] = {}
    if not math.isnan(peak_ppb):
        metrics["peak_ppb"] = round(peak_ppb, 2)
    if not math.isnan(peak_z):
        metrics["peak_z_eff"] = round(peak_z, 2)
    if not math.isnan(baseline):
        metrics["baseline_ppb"] = round(baseline, 3)
    metrics["alarm_frames"] = alarm_count
    metrics["warn_frames"] = warn_count
    metrics["event_duration_min"] = round(event_duration_min, 2)
    metrics["drift_km"] = round(drift_m / 1000.0, 2)
    return metrics


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
