from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from anomaly.events import SuspectedSpillEvent
from cerulean import CeruleanQueryResult, CeruleanSlick

from agent import AgentConfig, RuleBasedAgentModel, run_agent_for_event


class StubCeruleanClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    def query_slicks(self, bbox, start=None, end=None, limit=100, min_source_score=0.0, only_active=True, sort=None, extra_filters=None):
        self.calls.append(
            {
                "bbox": bbox,
                "start": start,
                "end": end,
                "limit": limit,
                "min_source_score": min_source_score,
                "only_active": only_active,
                "sort": sort,
                "filters": list(extra_filters or []),
            }
        )
        return CeruleanQueryResult.from_payload(self.payload)


@pytest.fixture
def sample_event() -> SuspectedSpillEvent:
    payload = json.loads(Path("tests/data/sample_event.json").read_text())
    return SuspectedSpillEvent.model_validate(payload)


def test_agent_with_matches(tmp_path: Path, sample_event: SuspectedSpillEvent):
    payload = json.loads(Path("tests/data/cerulean_slicks.json").read_text())
    client = StubCeruleanClient(payload)
    config = AgentConfig(artifact_root=tmp_path / "artifacts", followup_store=tmp_path / "followups.ndjson")
    result = run_agent_for_event(
        sample_event,
        model=RuleBasedAgentModel(),
        client=client,
        config=config,
        timestamp=datetime(2024, 6, 1, 12, 20, tzinfo=timezone.utc),
    )

    assert result.cerulean_result.number_returned == payload["numberReturned"]
    synopsis = result.synopsis
    assert synopsis.scenario == "validation_context"
    assert synopsis.followup_scheduled is False
    geojson_path = next(p for p in result.artifacts if p.name == "cerulean.geojson")
    assert geojson_path.exists()
    assert client.calls
    call = client.calls[0]
    assert call["limit"] >= 100
    assert call["min_source_score"] == 0.0


def test_agent_with_no_matches_schedules_followup(tmp_path: Path, sample_event: SuspectedSpillEvent):
    payload = {
        "type": "FeatureCollection",
        "numberMatched": 0,
        "numberReturned": 0,
        "features": []
    }
    client = StubCeruleanClient(payload)
    followups_path = tmp_path / "followups.ndjson"
    config = AgentConfig(artifact_root=tmp_path / "artifacts", followup_store=followups_path)
    result = run_agent_for_event(
        sample_event,
        model=RuleBasedAgentModel(),
        client=client,
        config=config,
        timestamp=datetime(2024, 6, 1, 12, 20, tzinfo=timezone.utc),
    )

    assert result.synopsis.scenario == "first_discovery"
    assert result.synopsis.followup_scheduled is True
    assert followups_path.exists()
    lines = followups_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event_id"] == "test-event"
    summary_path = next(p for p in result.artifacts if p.name == "cerulean_summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["slick_count"] == 0
