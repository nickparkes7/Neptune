from __future__ import annotations

import unittest
from pathlib import Path

from anomaly.hybrid import HybridOilAlertConfig, HybridOilAlertScorer

DATA_PATH = Path("data/ship/seaowl_sample.ndjson")


class HybridOilAlertTest(unittest.TestCase):
    def test_hybrid_alert_triggers_on_demo_stream(self) -> None:
        cfg = HybridOilAlertConfig()
        cfg.use_chl = False
        cfg.use_back = False
        scorer = HybridOilAlertScorer(cfg)
        df = scorer.score_ndjson(DATA_PATH)
        self.assertFalse(df.empty)
        self.assertGreater(df["oil_alarm"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
