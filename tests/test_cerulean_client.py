import json
from datetime import datetime, timedelta, timezone

import pytest

from cerulean import (
    CeruleanClient,
    CeruleanError,
    CeruleanQueryResult,
    CeruleanSlick,
    build_feature_collection,
    summarize_slicks,
)


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if status_code == 200 else "error"

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, response):
        self._response = response
        self.last_url = None
        self.last_params = None
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        self.last_url = url
        self.last_params = params or {}
        return self._response


@pytest.fixture(scope="module")
def sample_payload():
    with open("tests/data/cerulean_slicks.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_query_slicks_builds_expected_params(sample_payload):
    session = DummySession(DummyResponse(sample_payload))
    client = CeruleanClient(base_url="https://api.test", session=session)

    bbox = (-74.3, 40.5, -73.9, 41.2)
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=48)

    result = client.query_slicks(
        bbox,
        start=start,
        end=end,
        limit=25,
        min_source_score=0.0,
    )

    assert session.calls == 1
    assert session.last_url == "https://api.test/collections/public.slick_plus/items"
    params = session.last_params
    assert params["bbox"].startswith("-74.300000,40.500000")
    assert params["limit"] == "25"
    assert params["datetime"] == "2024-06-01T00:00:00Z/2024-06-03T00:00:00Z"
    if "filter" in params:
        assert "active = true" in params["filter"]
        assert "max_source_collated_score GTE 0.0" in params["filter"]
    assert params.get("sortby") in {None, "-max_source_collated_score"}

    assert isinstance(result, CeruleanQueryResult)
    assert len(result.slicks) == sample_payload["numberReturned"]


def test_query_slicks_handles_http_errors(sample_payload):
    session = DummySession(DummyResponse(sample_payload, status_code=500))
    client = CeruleanClient(base_url="https://api.test", session=session)

    with pytest.raises(CeruleanError):
        client.query_slicks((-1, -1, 1, 1))


def test_slick_parsing_and_summary(sample_payload):
    result = CeruleanQueryResult.from_payload(sample_payload)
    slick = result.slicks[0]

    assert isinstance(slick, CeruleanSlick)
    assert slick.id == "slick-001"
    assert slick.active is True
    assert slick.machine_confidence == pytest.approx(0.83)
    lon, lat = slick.centroid
    assert pytest.approx(lon, rel=1e-3) == -74.075
    assert pytest.approx(lat, rel=1e-3) == 40.71

    summary = summarize_slicks(result.slicks)
    assert summary.count == 2
    assert summary.active_count == 1
    assert summary.source_counts["vessels"] == 1
    assert summary.source_counts["infrastructure"] == 1
    assert summary.source_counts["dark_vessels"] == 1
    assert summary.total_area_km2 == pytest.approx((250000 + 90000) / 1_000_000)


def test_build_feature_collection(sample_payload):
    result = CeruleanQueryResult.from_payload(sample_payload)
    feature_collection = build_feature_collection(result.slicks)

    assert feature_collection["type"] == "FeatureCollection"
    assert len(feature_collection["features"]) == result.number_returned
    geometry = feature_collection["features"][0]["geometry"]
    assert geometry["type"] == "Polygon"
    props = feature_collection["features"][0]["properties"]
    assert "geometry" not in props
    assert props["id"] == "slick-001"
