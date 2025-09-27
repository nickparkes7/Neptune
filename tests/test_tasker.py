from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import unittest

from satellite.tasker import (
    SceneRef,
    TaskRequest,
    TaskerConfig,
    filter_catalog,
    load_catalog,
    task_satellite,
    _parse_bbox,
    _bbox_intersects,
)


class SatelliteTaskerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog_path = Path("configs/s1_catalog.json")
        if not self.catalog_path.exists():
            self.skipTest("default catalog missing")

    def test_load_catalog(self) -> None:
        scenes = load_catalog(self.catalog_path)
        self.assertGreaterEqual(len(scenes), 1)
        self.assertIsInstance(scenes[0], SceneRef)

    def test_filter_by_bbox_and_time(self) -> None:
        scenes = load_catalog(self.catalog_path)
        request = TaskRequest(
            bbox=(-74.3, 40.5, -73.5, 41.2),
            start="2024-09-01T00:00:00Z",
            end="2024-09-04T00:00:00Z",
        )
        filtered = filter_catalog(scenes, request, TaskerConfig(max_results=5))
        self.assertTrue(all(_bbox_intersects(scene.bbox, request.bbox) for scene in filtered))
        self.assertTrue(all(request.start <= scene.acquired <= request.end for scene in filtered))

    def test_task_satellite_returns_sorted_scenes(self) -> None:
        request = TaskRequest(
            bbox=(-74.3, 40.5, -73.5, 41.2),
            start="2024-09-01T00:00:00Z",
            end="2024-09-04T00:00:00Z",
        )
        scenes = task_satellite(request, TaskerConfig(max_results=1))
        self.assertEqual(1, len(scenes))
        self.assertLessEqual(scenes[0].acquired, request.end)

    def test_parse_bbox_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _parse_bbox("1,2,1,3")

if __name__ == "__main__":
    unittest.main()
