"""Incident lifecycle management for SeaOWL spill detection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Literal
from uuid import uuid4


from .events import ChannelStats, OilStats, SuspectedSpillEvent

TransitionKind = Literal["opened", "updated", "heartbeat", "closed"]

__all__ = [
    "IncidentManagerConfig",
    "IncidentTransition",
    "IncidentManager",
]


@dataclass(slots=True)
class IncidentManagerConfig:
    """Tunable parameters for :class:`IncidentManager`."""

    merge_gap_s: float = 600.0
    merge_distance_km: float = 5.0
    clear_hold_s: float = 600.0
    rearm_distance_km: float = 5.0
    heartbeat_interval_s: float = 900.0
    tasking_cooldown_s: float = 21600.0
    incident_ttl_s: float = 7200.0
    significant_oil_increase_pct: float = 0.2
    significant_oil_mean_increase_pct: float = 0.1
    significant_bbox_expand_km: float = 5.0

    def __post_init__(self) -> None:
        if self.clear_hold_s < self.merge_gap_s:
            raise ValueError("clear_hold_s must be >= merge_gap_s to avoid re-arm thrash")


@dataclass(slots=True)
class IncidentTransition:
    """Lifecycle notification emitted by the incident manager."""

    kind: TransitionKind
    at: datetime
    incident: SuspectedSpillEvent
    reason: str
    trigger_event: Optional[SuspectedSpillEvent] = None
    allow_tasking: bool = False


@dataclass(slots=True)
class _ContextAccumulator:
    min: float
    max: float
    sum: float
    median: float
    count: int

    def update(self, stats: ChannelStats, samples: int) -> None:
        self.min = min(self.min, stats.min)
        self.max = max(self.max, stats.max)
        self.sum += stats.mean * samples
        self.count += samples
        self.median = stats.median

    def to_stats(self) -> ChannelStats:
        mean = self.sum / self.count if self.count else self.median
        return ChannelStats(min=self.min, max=self.max, mean=mean, median=self.median)


@dataclass
class _IncidentAccumulator:
    incident_id: str
    ts_start: datetime
    ts_end: datetime
    ts_peak: datetime
    lat: float
    lon: float
    sample_count: int
    oil_min: float
    oil_max: float
    oil_sum: float
    oil_z_sum: float
    oil_median: float
    oil_max_z: float
    context: Dict[str, _ContextAccumulator] = field(default_factory=dict)
    aoi_bbox: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)
    platform_id: Optional[str] = None
    sensor_id: Optional[str] = None

    @classmethod
    def from_event(cls, event: SuspectedSpillEvent, incident_id: Optional[str] = None) -> "_IncidentAccumulator":
        incident_id = incident_id or event.event_id or uuid4().hex
        ctx = {
            name: _ContextAccumulator(
                min=stats.min,
                max=stats.max,
                sum=stats.mean * event.sample_count,
                median=stats.median,
                count=event.sample_count,
            )
            for name, stats in event.context_channels.items()
        }
        return cls(
            incident_id=incident_id,
            ts_start=event.ts_start,
            ts_end=event.ts_end,
            ts_peak=event.ts_peak,
            lat=event.lat,
            lon=event.lon,
            sample_count=event.sample_count,
            oil_min=event.oil_stats.min,
            oil_max=event.oil_stats.max,
            oil_sum=event.oil_stats.mean * event.sample_count,
            oil_z_sum=event.oil_stats.mean_z * event.sample_count,
            oil_median=event.oil_stats.median,
            oil_max_z=event.oil_stats.max_z,
            context=ctx,
            aoi_bbox=event.aoi_bbox,
            platform_id=event.platform_id,
            sensor_id=event.sensor_id,
        )

    def update(self, event: SuspectedSpillEvent) -> None:
        self.ts_end = max(self.ts_end, event.ts_end)
        if event.oil_stats.max_z >= self.oil_max_z:
            self.oil_max_z = event.oil_stats.max_z
            self.ts_peak = event.ts_peak
            self.lat = event.lat
            self.lon = event.lon
        self.sample_count += event.sample_count
        self.oil_min = min(self.oil_min, event.oil_stats.min)
        self.oil_max = max(self.oil_max, event.oil_stats.max)
        self.oil_sum += event.oil_stats.mean * event.sample_count
        self.oil_z_sum += event.oil_stats.mean_z * event.sample_count
        self.oil_median = event.oil_stats.median
        self.aoi_bbox = _merge_bbox(self.aoi_bbox, event.aoi_bbox)
        if self.platform_id is None:
            self.platform_id = event.platform_id
        if self.sensor_id is None:
            self.sensor_id = event.sensor_id
        for name, stats in event.context_channels.items():
            samples = event.sample_count
            if name in self.context:
                self.context[name].update(stats, samples)
            else:
                self.context[name] = _ContextAccumulator(
                    min=stats.min,
                    max=stats.max,
                    sum=stats.mean * samples,
                    median=stats.median,
                    count=samples,
                )

    def snapshot(self) -> SuspectedSpillEvent:
        duration_s = max((self.ts_end - self.ts_start).total_seconds(), 0.0)
        oil_mean = self.oil_sum / self.sample_count if self.sample_count else self.oil_max
        oil_mean_z = self.oil_z_sum / self.sample_count if self.sample_count else self.oil_max_z
        oil_stats = OilStats(
            min=self.oil_min,
            max=self.oil_max,
            mean=oil_mean,
            median=self.oil_median,
            max_z=self.oil_max_z,
            mean_z=oil_mean_z,
        )
        context = {name: acc.to_stats() for name, acc in self.context.items()}
        return SuspectedSpillEvent(
            event_id=self.incident_id,
            ts_start=self.ts_start,
            ts_end=self.ts_end,
            ts_peak=self.ts_peak,
            lat=self.lat,
            lon=self.lon,
            duration_s=duration_s,
            sample_count=self.sample_count,
            platform_id=self.platform_id,
            sensor_id=self.sensor_id,
            oil_stats=oil_stats,
            context_channels=context,
            aoi_bbox=self.aoi_bbox,
        )


@dataclass
class _IncidentState:
    accumulator: _IncidentAccumulator
    opened_at: datetime
    last_update: datetime
    last_snapshot: SuspectedSpillEvent
    last_heartbeat: datetime
    last_tasking: datetime


class IncidentManager:
    """Debounce and lifecycle management for spill incidents."""

    def __init__(
        self,
        config: Optional[IncidentManagerConfig] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.cfg = config or IncidentManagerConfig()
        self._now = now_fn or datetime.utcnow
        self._current: Optional[_IncidentState] = None

    @property
    def is_active(self) -> bool:
        return self._current is not None

    def process_event(self, event: SuspectedSpillEvent) -> List[IncidentTransition]:
        transitions: List[IncidentTransition] = []
        if self._current is None:
            return self._open_incident(event, reason="new_event")

        current = self._current
        gap = (event.ts_start - current.accumulator.ts_end).total_seconds()
        if gap < 0:
            gap = 0.0
        distance = _haversine_km(current.last_snapshot.lat, current.last_snapshot.lon, event.lat, event.lon)

        if gap >= self.cfg.clear_hold_s and distance >= self.cfg.rearm_distance_km:
            transitions.extend(self._close(reason="clear_gap"))
            transitions.extend(self._open_incident(event, reason="rearm"))
            return transitions

        if gap > self.cfg.merge_gap_s and distance > self.cfg.merge_distance_km:
            transitions.extend(self._close(reason="gap_or_distance_exceeded"))
            transitions.extend(self._open_incident(event, reason="new_location"))
            return transitions

        prev_snapshot = current.last_snapshot
        current.accumulator.update(event)
        snapshot = current.accumulator.snapshot()
        current.last_snapshot = snapshot
        current.last_update = event.ts_end

        allow_tasking = False
        if (event.ts_end - current.last_tasking).total_seconds() >= self.cfg.tasking_cooldown_s:
            allow_tasking = True
            current.last_tasking = event.ts_end

        significant, reasons = _compute_significance(prev_snapshot, snapshot, self.cfg)
        now = event.ts_end
        if significant:
            transitions.append(
                IncidentTransition(
                    kind="updated",
                    at=now,
                    incident=snapshot,
                    reason=", ".join(reasons),
                    trigger_event=event,
                    allow_tasking=allow_tasking,
                )
            )
            current.last_heartbeat = now
        elif (now - current.last_heartbeat).total_seconds() >= self.cfg.heartbeat_interval_s:
            transitions.append(
                IncidentTransition(
                    kind="heartbeat",
                    at=now,
                    incident=snapshot,
                    reason="heartbeat_interval",
                    trigger_event=event,
                    allow_tasking=allow_tasking,
                )
            )
            current.last_heartbeat = now
        elif allow_tasking:
            transitions.append(
                IncidentTransition(
                    kind="heartbeat",
                    at=now,
                    incident=snapshot,
                    reason="tasking_cooldown_elapsed",
                    trigger_event=event,
                    allow_tasking=True,
                )
            )
            current.last_heartbeat = now

        return transitions

    def flush(self, now: Optional[datetime] = None) -> List[IncidentTransition]:
        if self._current is None:
            return []
        now = now or self._now()
        if (now - self._current.last_update).total_seconds() >= self.cfg.incident_ttl_s:
            return self._close(reason="ttl", at=now)
        return []

    def active_snapshot(self) -> Optional[SuspectedSpillEvent]:
        if self._current is None:
            return None
        return self._current.last_snapshot

    def finalize(self, reason: str = "finalize", at: Optional[datetime] = None) -> List[IncidentTransition]:
        """Force close the active incident (used for offline batch runs)."""
        return self._close(reason=reason, at=at)

    # Internal helpers -----------------------------------------------------------------

    def _open_incident(self, event: SuspectedSpillEvent, reason: str) -> List[IncidentTransition]:
        acc = _IncidentAccumulator.from_event(event)
        snapshot = acc.snapshot()
        state = _IncidentState(
            accumulator=acc,
            opened_at=event.ts_start,
            last_update=event.ts_end,
            last_snapshot=snapshot,
            last_heartbeat=event.ts_start,
            last_tasking=event.ts_end,
        )
        self._current = state
        transition = IncidentTransition(
            kind="opened",
            at=event.ts_end,
            incident=snapshot,
            reason=reason,
            trigger_event=event,
            allow_tasking=True,
        )
        return [transition]

    def _close(self, reason: str, at: Optional[datetime] = None) -> List[IncidentTransition]:
        if self._current is None:
            return []
        snapshot = self._current.accumulator.snapshot()
        at = at or self._current.last_update
        transition = IncidentTransition(
            kind="closed",
            at=at,
            incident=snapshot,
            reason=reason,
            trigger_event=None,
            allow_tasking=False,
        )
        self._current = None
        return [transition]


def _compute_significance(
    previous: SuspectedSpillEvent,
    current: SuspectedSpillEvent,
    cfg: IncidentManagerConfig,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    prev_max_z = previous.oil_stats.max_z
    curr_max_z = current.oil_stats.max_z
    if prev_max_z > 0:
        delta = (curr_max_z - prev_max_z) / prev_max_z
    else:
        delta = float("inf") if curr_max_z > 0 else 0.0
    if delta >= cfg.significant_oil_increase_pct:
        reasons.append("oil_max_z")

    prev_mean_z = previous.oil_stats.mean_z
    curr_mean_z = current.oil_stats.mean_z
    if prev_mean_z > 0:
        delta_mean = (curr_mean_z - prev_mean_z) / prev_mean_z
    else:
        delta_mean = float("inf") if curr_mean_z > 0 else 0.0
    if delta_mean >= cfg.significant_oil_mean_increase_pct:
        reasons.append("oil_mean_z")

    bbox_growth = _bbox_growth_km(previous.aoi_bbox, current.aoi_bbox)
    if bbox_growth >= cfg.significant_bbox_expand_km:
        reasons.append("aoi_expand")

    return (len(reasons) > 0, reasons)


def _bbox_growth_km(prev_bbox: Tuple[float, float, float, float], new_bbox: Tuple[float, float, float, float]) -> float:
    prev_lat_km, prev_lon_km = _bbox_span_km(prev_bbox)
    new_lat_km, new_lon_km = _bbox_span_km(new_bbox)
    return max(new_lat_km - prev_lat_km, new_lon_km - prev_lon_km)


def _bbox_span_km(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_span_km = abs(max_lat - min_lat) * 111.0
    mean_lat = (max_lat + min_lat) / 2.0
    lon_span_km = abs(max_lon - min_lon) * 111.0 * max(math.cos(math.radians(mean_lat)), 1e-6)
    return lat_span_km, lon_span_km


def _merge_bbox(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c
