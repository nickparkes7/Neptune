"""Event schema and trigger helpers for SeaOWL anomaly alerts."""

from __future__ import annotations

import math
from pathlib import Path
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import Field

from common.pydantic_compat import CompatBaseModel

__all__ = [
    "ChannelStats",
    "OilStats",
    "SuspectedSpillEvent",
    "EventExtractorConfig",
    "extract_events",
    "generate_events_from_ndjson",
    "publish_events",
]


class ChannelStats(CompatBaseModel):
    """Summary statistics for a single scalar channel."""

    min: float = Field(..., description="Minimum observed value during the event")
    max: float = Field(..., description="Maximum observed value during the event")
    mean: float = Field(..., description="Mean value over the event window")
    median: float = Field(..., description="Median value over the event window")


class OilStats(ChannelStats):
    """Extended statistics for the oil fluorescence signal."""

    max_z: float = Field(..., description="Peak effective z-score during the event")
    mean_z: float = Field(..., description="Mean effective z-score during the event")


class SuspectedSpillEvent(CompatBaseModel):
    """Structured representation of an onboard anomaly trigger."""

    event_id: str = Field(..., description="Stable identifier for the event")
    ts_start: datetime = Field(..., description="UTC timestamp of the first alerting sample")
    ts_end: datetime = Field(..., description="UTC timestamp of the final alerting sample")
    ts_peak: datetime = Field(..., description="Timestamp of the peak oil response")
    lat: float = Field(..., description="Latitude at peak response (degrees)")
    lon: float = Field(..., description="Longitude at peak response (degrees)")
    duration_s: float = Field(..., ge=0.0, description="Duration of the event window in seconds")
    sample_count: int = Field(..., ge=1, description="Number of samples contributing to the event")
    platform_id: Optional[str] = Field(None, description="Platform identifier if available")
    sensor_id: Optional[str] = Field(None, description="Sensor identifier if available")
    oil_stats: OilStats = Field(..., description="Oil fluorescence statistics")
    context_channels: Dict[str, ChannelStats] = Field(
        default_factory=dict,
        description="Summary stats for additional SeaOWL channels",
    )
    aoi_bbox: Tuple[float, float, float, float] = Field(
        ...,
        description="Bounding box (min_lon, min_lat, max_lon, max_lat) expanded for satellite tasking",
    )


@dataclass
class EventExtractorConfig:
    """Controls event segmentation and spatial envelope generation."""

    require_alarm: bool = True
    min_duration_s: float = 60.0
    min_samples: int = 30
    bbox_padding_km: float = 15.0
    context_channels: Tuple[str, ...] = (
        "chlorophyll_ug_per_l",
        "backscatter_m-1_sr-1",
    )


def extract_events(frame: pd.DataFrame, config: Optional[EventExtractorConfig] = None) -> List[SuspectedSpillEvent]:
    """Convert a scored SeaOWL frame into structured spill events.

    Parameters
    ----------
    frame:
        Dataframe returned by :class:`HybridOilAlertScorer` containing alert flags
        and supporting columns (``oil_warn``, ``oil_alarm``, ``oil_z_eff``).
    config:
        Optional :class:`EventExtractorConfig` controlling segmentation rules.
    """

    cfg = config or EventExtractorConfig()

    if "oil_warn" not in frame or "oil_alarm" not in frame:
        raise KeyError("frame must contain 'oil_warn' and 'oil_alarm' columns")

    work = frame.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    work = work.sort_values("ts").reset_index(drop=True)

    warn = work["oil_warn"].astype(bool).to_numpy()
    alarm = work["oil_alarm"].astype(bool).to_numpy()
    active = np.logical_or(warn, alarm)

    events: List[SuspectedSpillEvent] = []

    for start, end in _iterate_windows(active):
        if cfg.require_alarm and not alarm[start:end].any():
            continue
        segment = work.iloc[start:end]
        # Skip segments that do not meet minimum duration/sample requirements
        if segment.empty:
            continue
        if cfg.min_samples and len(segment) < cfg.min_samples:
            continue
        duration = _compute_duration_s(segment["ts"].to_numpy())
        if duration < cfg.min_duration_s:
            continue

        event = _build_event(segment, cfg)
        if event is not None:
            events.append(event)

    return events


def generate_events_from_ndjson(
    path: Path,
    scorer: Optional["HybridOilAlertScorer"] = None,
    config: Optional[EventExtractorConfig] = None,
    sink: Optional[Callable[[SuspectedSpillEvent], None]] = None,
) -> List[SuspectedSpillEvent]:
    """Shortcut to score an NDJSON stream and emit ``SuspectedSpillEvent`` objects."""

    from .hybrid import HybridOilAlertScorer  # Lazy import to avoid circular dependency

    scorer = scorer or HybridOilAlertScorer()
    df = scorer.score_ndjson(path)
    events = extract_events(df, config=config)
    if sink is not None:
        publish_events(events, sink)
    return events


def publish_events(events: Sequence[SuspectedSpillEvent], sink: Callable[[SuspectedSpillEvent], None]) -> None:
    """Send events to a sink callback (MVP in-process publisher)."""

    for event in events:
        sink(event)


# Internal helpers -----------------------------------------------------------------

def _iterate_windows(mask: np.ndarray) -> Iterable[Tuple[int, int]]:
    start: Optional[int] = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, len(mask)


def _compute_duration_s(timestamps: np.ndarray) -> float:
    if timestamps.size == 0:
        return 0.0
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    ts_series = pd.Series(ts).dropna().sort_values().reset_index(drop=True)
    if ts_series.empty:
        return 0.0
    duration = (ts_series.iloc[-1] - ts_series.iloc[0]).total_seconds()
    deltas = ts_series.diff().dropna().dt.total_seconds()
    deltas = deltas[(deltas > 0) & deltas.notna()]
    if not deltas.empty:
        duration += float(deltas.median())
    return float(max(duration, 0.0))


def _build_event(segment: pd.DataFrame, cfg: EventExtractorConfig) -> Optional[SuspectedSpillEvent]:
    ts_values = segment["ts"].to_numpy()
    ts_values = ts_values[~pd.isna(ts_values)]
    if ts_values.size == 0:
        return None

    ts_start = _ts_to_datetime(segment["ts"].iloc[0])
    ts_end = _ts_to_datetime(segment["ts"].iloc[-1])

    oil_z = pd.to_numeric(segment.get("oil_z_eff"), errors="coerce")
    if oil_z.isna().all():
        return None
    peak_idx = oil_z.idxmax()
    peak_row = segment.loc[peak_idx]

    lat = _safe_float(peak_row.get("lat"))
    lon = _safe_float(peak_row.get("lon"))

    # Fallback to mean if peak position is missing
    if lat is None or lon is None:
        lat = _safe_float(segment["lat"].astype(float).mean())
        lon = _safe_float(segment["lon"].astype(float).mean())

    if lat is None or lon is None:
        return None

    oil_stats = _compute_oil_stats(segment)
    context = _compute_context_stats(segment, cfg.context_channels)
    aoi_bbox = _compute_bbox(segment, lat, lon, cfg.bbox_padding_km)

    duration_s = _compute_duration_s(segment["ts"].to_numpy())
    sample_count = len(segment)

    platform_id = _first_non_null(segment.get("platform_id"))
    sensor_id = _first_non_null(segment.get("sensor_id"))

    return SuspectedSpillEvent(
        event_id=uuid.uuid4().hex,
        ts_start=ts_start,
        ts_end=ts_end,
        ts_peak=_ts_to_datetime(peak_row.get("ts")),
        lat=lat,
        lon=lon,
        duration_s=duration_s,
        sample_count=sample_count,
        platform_id=platform_id,
        sensor_id=sensor_id,
        oil_stats=oil_stats,
        context_channels=context,
        aoi_bbox=aoi_bbox,
    )


def _ts_to_datetime(value: object) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    raise TypeError("timestamp column must contain datetime-like values")


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _compute_oil_stats(segment: pd.DataFrame) -> OilStats:
    oil = pd.to_numeric(segment.get("oil_fluor_ppb"), errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(segment.get("oil_z_eff"), errors="coerce").to_numpy(dtype=float)

    oil = oil[np.isfinite(oil)]
    z = z[np.isfinite(z)]

    if oil.size == 0 or z.size == 0:
        raise ValueError("segment lacks valid oil data")

    return OilStats(
        min=float(np.min(oil)),
        max=float(np.max(oil)),
        mean=float(np.mean(oil)),
        median=float(np.median(oil)),
        max_z=float(np.max(z)),
        mean_z=float(np.mean(z)),
    )


def _compute_context_stats(segment: pd.DataFrame, channels: Sequence[str]) -> Dict[str, ChannelStats]:
    stats: Dict[str, ChannelStats] = {}
    for name in channels:
        if name not in segment:
            continue
        arr = pd.to_numeric(segment[name], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        stats[name] = ChannelStats(
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
        )
    return stats


def _compute_bbox(segment: pd.DataFrame, peak_lat: float, peak_lon: float, padding_km: float) -> Tuple[float, float, float, float]:
    lat_series = pd.to_numeric(segment.get("lat"), errors="coerce").to_numpy(dtype=float)
    lon_series = pd.to_numeric(segment.get("lon"), errors="coerce").to_numpy(dtype=float)

    lat_vals = lat_series[np.isfinite(lat_series)]
    lon_vals = lon_series[np.isfinite(lon_series)]

    if lat_vals.size == 0:
        lat_vals = np.array([peak_lat])
    if lon_vals.size == 0:
        lon_vals = np.array([peak_lon])

    min_lat = float(np.min(lat_vals))
    max_lat = float(np.max(lat_vals))
    min_lon = float(np.min(lon_vals))
    max_lon = float(np.max(lon_vals))

    mean_lat = float(np.mean(lat_vals))
    padding_deg_lat = padding_km / 111.0
    cos_lat = math.cos(math.radians(mean_lat))
    cos_lat = max(cos_lat, 1e-6)
    padding_deg_lon = padding_km / (111.0 * cos_lat)

    min_lat -= padding_deg_lat
    max_lat += padding_deg_lat
    min_lon -= padding_deg_lon
    max_lon += padding_deg_lon

    min_lat = max(min_lat, -90.0)
    max_lat = min(max_lat, 90.0)
    min_lon = ((min_lon + 180.0) % 360.0) - 180.0
    max_lon = ((max_lon + 180.0) % 360.0) - 180.0

    return (min_lon, min_lat, max_lon, max_lat)


def _first_non_null(series: Optional[pd.Series]) -> Optional[str]:
    if series is None:
        return None
    for value in series:
        if value:
            return str(value)
    return None
