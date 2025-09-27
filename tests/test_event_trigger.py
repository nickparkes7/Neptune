from __future__ import annotations

import unittest
from pathlib import Path

from anomaly import HybridOilAlertScorer
from anomaly.events import EventExtractorConfig, SuspectedSpillEvent, extract_events, generate_events_from_ndjson

DATA_PATH = Path("data/ship/seaowl_sample.ndjson")


class EventTriggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = HybridOilAlertScorer()

    def test_extract_events_from_dataframe(self) -> None:
        df = self.scorer.score_ndjson(DATA_PATH)
        events = extract_events(df, EventExtractorConfig())
        self.assertGreaterEqual(len(events), 1, "expected at least one spill event")
        event = events[0]
        self._assert_event(event)

    def test_generate_events_from_ndjson(self) -> None:
        events = generate_events_from_ndjson(DATA_PATH, scorer=self.scorer)
        self.assertGreaterEqual(len(events), 1)
        self._assert_event(events[0])

    def _assert_event(self, event: SuspectedSpillEvent) -> None:
        self.assertIsInstance(event, SuspectedSpillEvent)
        self.assertGreater(event.oil_stats.max, event.oil_stats.mean)
        self.assertGreater(event.oil_stats.max_z, event.oil_stats.mean_z)
        bbox = event.aoi_bbox
        self.assertEqual(len(bbox), 4)
        min_lon, min_lat, max_lon, max_lat = bbox
        self.assertLess(min_lat, max_lat)
        if min_lon <= max_lon:
            self.assertGreaterEqual(event.lon, min_lon)
            self.assertLessEqual(event.lon, max_lon)
        else:  # dateline wrap
            self.assertTrue(event.lon >= min_lon or event.lon <= max_lon)
        self.assertGreaterEqual(event.lat, min_lat)
        self.assertLessEqual(event.lat, max_lat)
        self.assertIn("chlorophyll_ug_per_l", event.context_channels)
        self.assertGreater(event.duration_s, 60.0)
        payload = event.model_dump(mode="json")
        self.assertEqual(payload["event_id"], event.event_id)


if __name__ == "__main__":
    unittest.main()
