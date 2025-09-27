from __future__ import annotations

import unittest
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from anomaly import (
    IncidentManager,
    IncidentManagerConfig,
    SuspectedSpillEvent,
    generate_events_from_ndjson,
)

DATA_PATH = Path("data/ship/seaowl_sample.ndjson")


def _scaled_event(
    base: SuspectedSpillEvent,
    *,
    start_delta: timedelta,
    duration: timedelta,
    lat_delta: float = 0.0,
    lon_delta: float = 0.0,
    max_z_factor: float = 1.0,
    mean_z_factor: float = 1.0,
    bbox_expand_deg: float = 0.0,
) -> SuspectedSpillEvent:
    ts_start = base.ts_end + start_delta
    ts_end = ts_start + duration
    ts_peak = ts_start + duration / 2
    sample_count = max(int(duration.total_seconds()), 1)

    oil_stats = base.oil_stats.model_copy(
        update={
            "min": base.oil_stats.min,
            "max": base.oil_stats.max * max_z_factor,
            "mean": base.oil_stats.mean * ((max_z_factor + mean_z_factor) / 2),
            "median": base.oil_stats.median,
            "max_z": base.oil_stats.max_z * max_z_factor,
            "mean_z": base.oil_stats.mean_z * mean_z_factor,
        }
    )

    context_channels = {
        name: stats.model_copy(
            update={
                "min": stats.min,
                "max": stats.max,
                "mean": stats.mean,
                "median": stats.median,
            }
        )
        for name, stats in base.context_channels.items()
    }

    min_lon, min_lat, max_lon, max_lat = base.aoi_bbox
    expanded_bbox = (
        min_lon - bbox_expand_deg,
        min_lat - bbox_expand_deg,
        max_lon + bbox_expand_deg,
        max_lat + bbox_expand_deg,
    )

    return base.model_copy(
        update={
            "event_id": uuid4().hex,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "ts_peak": ts_peak,
            "lat": base.lat + lat_delta,
            "lon": base.lon + lon_delta,
            "duration_s": duration.total_seconds(),
            "sample_count": sample_count,
            "oil_stats": oil_stats,
            "context_channels": context_channels,
            "aoi_bbox": expanded_bbox,
        }
    )


class IncidentManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        events = generate_events_from_ndjson(DATA_PATH)
        if not events:
            self.skipTest("sample NDJSON did not produce events")
        self.base_event = events[0]

    def test_open_and_update_with_significant_change(self) -> None:
        cfg = IncidentManagerConfig(
            merge_gap_s=600,
            merge_distance_km=8.0,
            clear_hold_s=900,
            rearm_distance_km=6.0,
            heartbeat_interval_s=600,
            tasking_cooldown_s=99999,
            significant_oil_increase_pct=0.2,
            significant_oil_mean_increase_pct=0.2,
            significant_bbox_expand_km=1.0,
        )
        manager = IncidentManager(cfg)

        opened = manager.process_event(self.base_event)
        self.assertEqual(1, len(opened))
        self.assertEqual("opened", opened[0].kind)

        follow_on = _scaled_event(
            self.base_event,
            start_delta=timedelta(seconds=120),
            duration=timedelta(minutes=15),
            max_z_factor=1.6,
            mean_z_factor=1.4,
            bbox_expand_deg=0.02,
        )
        transitions = manager.process_event(follow_on)
        kinds = {t.kind for t in transitions}
        self.assertIn("updated", kinds)
        self.assertTrue(any("oil_max_z" in t.reason for t in transitions if t.kind == "updated"))

    def test_rearm_after_clear_gap_and_distance(self) -> None:
        cfg = IncidentManagerConfig(
            merge_gap_s=600,
            merge_distance_km=8.0,
            clear_hold_s=900,
            rearm_distance_km=3.0,
            heartbeat_interval_s=600,
            tasking_cooldown_s=99999,
        )
        manager = IncidentManager(cfg)
        manager.process_event(self.base_event)

        distant = _scaled_event(
            self.base_event,
            start_delta=timedelta(seconds=1200),
            duration=timedelta(minutes=10),
            lat_delta=0.12,
            lon_delta=0.12,
        )
        transitions = manager.process_event(distant)
        kinds = [t.kind for t in transitions]
        self.assertEqual(["closed", "opened"], kinds)

    def test_heartbeat_when_change_is_small(self) -> None:
        cfg = IncidentManagerConfig(
            merge_gap_s=600,
            merge_distance_km=8.0,
            clear_hold_s=900,
            rearm_distance_km=6.0,
            heartbeat_interval_s=300,
            tasking_cooldown_s=99999,
            significant_oil_increase_pct=0.8,
            significant_oil_mean_increase_pct=0.8,
            significant_bbox_expand_km=5.0,
        )
        manager = IncidentManager(cfg)
        manager.process_event(self.base_event)

        mild = _scaled_event(
            self.base_event,
            start_delta=timedelta(seconds=240),
            duration=timedelta(minutes=12),
            max_z_factor=1.0,
            mean_z_factor=1.0,
        )
        transitions = manager.process_event(mild)
        kinds = {t.kind for t in transitions}
        self.assertIn("heartbeat", kinds)

    def test_finalize_force_closes_active_incident(self) -> None:
        cfg = IncidentManagerConfig(incident_ttl_s=9999)
        manager = IncidentManager(cfg)
        manager.process_event(self.base_event)
        transitions = manager.finalize(at=self.base_event.ts_end)
        self.assertEqual(1, len(transitions))
        self.assertEqual("closed", transitions[0].kind)

    def test_flush_closes_after_ttl(self) -> None:
        cfg = IncidentManagerConfig(
            merge_gap_s=600,
            merge_distance_km=8.0,
            clear_hold_s=900,
            rearm_distance_km=6.0,
            heartbeat_interval_s=99999,
            tasking_cooldown_s=99999,
            incident_ttl_s=300,
            significant_oil_increase_pct=0.8,
            significant_oil_mean_increase_pct=0.8,
            significant_bbox_expand_km=10.0,
        )
        manager = IncidentManager(cfg)
        manager.process_event(self.base_event)

        now = self.base_event.ts_end + timedelta(seconds=400)
        transitions = manager.flush(now)
        self.assertEqual(1, len(transitions))
        self.assertEqual("closed", transitions[0].kind)


if __name__ == "__main__":
    unittest.main()
