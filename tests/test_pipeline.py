from __future__ import annotations

import unittest
from pathlib import Path

from anomaly import PipelineConfig, generate_transitions_from_ndjson

DATA_PATH = Path("data/ship/seaowl_sample.ndjson")


class PipelineIntegrationTest(unittest.TestCase):
    def test_pipeline_generates_incident_transitions(self) -> None:
        result = generate_transitions_from_ndjson(DATA_PATH, config=PipelineConfig(flush_after_s=1200))
        self.assertGreater(len(result.events), 0)
        self.assertGreater(len(result.transitions), 0)
        kinds = {t.kind for t in result.transitions}
        self.assertIn("opened", kinds)
        self.assertIn("closed", kinds)
        self.assertTrue(any(t.allow_tasking for t in result.transitions if t.kind == "opened"))


if __name__ == "__main__":
    unittest.main()
