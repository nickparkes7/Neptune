"""Minimal Cerulean API client for retrieving slick detections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests
from pydantic import Field, ValidationError, validator
from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry

from common.pydantic_compat import CompatBaseModel

DEFAULT_BASE_URL = "https://api.cerulean.skytruth.org"
_SLICKS_ENDPOINT = "collections/public.slick_plus/items"


class CeruleanError(RuntimeError):
    """Raised when the Cerulean API returns an error or malformed payload."""


class CeruleanSlick(CompatBaseModel):
    """Parsed slick detection returned by Cerulean."""

    id: str = Field(..., description="Stable slick identifier")
    slick_timestamp: datetime = Field(..., description="Timestamp of the detection (UTC)")
    area: float = Field(..., description="Slick area in square meters")
    active: bool = Field(True, description="Whether the slick is marked active")
    machine_confidence: Optional[float] = Field(
        None, description="Model confidence score from Cerulean"
    )
    max_source_collated_score: Optional[float] = Field(
        None, description="Cerulean source attribution score"
    )
    s1_scene_id: Optional[str] = Field(None, description="Associated Sentinel-1 scene id")
    slick_url: Optional[str] = Field(None, description="Link to slick detail in Cerulean UI")
    source_type_1_ids: List[str] = Field(default_factory=list, description="Nearby vessels")
    source_type_2_ids: List[str] = Field(default_factory=list, description="Nearby infrastructure")
    source_type_3_ids: List[str] = Field(default_factory=list, description="Nearby dark vessels")
    geometry: BaseGeometry = Field(..., description="Slick footprint geometry")

    @validator("geometry", pre=True)
    def _coerce_geometry(cls, value: Any) -> BaseGeometry:  # noqa: D417 - short circuit
        if isinstance(value, BaseGeometry):
            return value
        if isinstance(value, Mapping):
            return shape(value)
        raise TypeError("geometry must be GeoJSON mapping or shapely geometry")

    @validator("source_type_1_ids", "source_type_2_ids", "source_type_3_ids", pre=True)
    def _coerce_optional_lists(cls, value: Any) -> List[str]:  # noqa: D417
        if value in (None, ""):
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            BaseGeometry: mapping,
            datetime: lambda dt: dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @classmethod
    def from_feature(cls, feature: Mapping[str, Any]) -> "CeruleanSlick":
        geometry = feature.get("geometry")
        if geometry is None:
            raise CeruleanError("Feature missing geometry")
        properties = dict(feature.get("properties", {}))
        # Prefer explicit property id, fall back to GeoJSON feature id.
        properties.setdefault("id", feature.get("id"))
        try:
            return cls(geometry=geometry, **properties)
        except ValidationError as exc:  # pragma: no cover - bubbles as CeruleanError
            raise CeruleanError("Invalid slick feature payload") from exc

    @property
    def centroid(self) -> Tuple[float, float]:
        geom = self.geometry
        return (geom.centroid.x, geom.centroid.y)


class CeruleanQueryResult(CompatBaseModel):
    """Structured response returned by :class:`CeruleanClient`."""

    slicks: List[CeruleanSlick]
    number_matched: int = 0
    number_returned: int = 0
    links: List[Mapping[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CeruleanQueryResult":
        features = payload.get("features", [])
        slicks = [CeruleanSlick.from_feature(feature) for feature in features]
        return cls(
            slicks=slicks,
            number_matched=int(payload.get("numberMatched", len(slicks))),
            number_returned=int(payload.get("numberReturned", len(slicks))),
            links=list(payload.get("links", [])),
        )


@dataclass
class CeruleanSummary:
    count: int
    active_count: int
    total_area_km2: float
    avg_machine_confidence: Optional[float]
    max_source_collated_score: Optional[float]
    source_counts: Dict[str, int]


def summarize_slicks(slicks: Sequence[CeruleanSlick]) -> CeruleanSummary:
    """Compute simple rollups for dashboard overlays."""

    slick_list = list(slicks)
    if not slick_list:
        return CeruleanSummary(
            count=0,
            active_count=0,
            total_area_km2=0.0,
            avg_machine_confidence=None,
            max_source_collated_score=None,
            source_counts={"vessels": 0, "infrastructure": 0, "dark_vessels": 0},
        )

    active_count = sum(1 for s in slick_list if s.active)
    total_area_km2 = sum(s.area for s in slick_list) / 1_000_000.0
    confidences = [s.machine_confidence for s in slick_list if s.machine_confidence is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    scores = [s.max_source_collated_score for s in slick_list if s.max_source_collated_score is not None]
    max_score = max(scores) if scores else None
    source_counts = {
        "vessels": sum(1 for s in slick_list if s.source_type_1_ids),
        "infrastructure": sum(1 for s in slick_list if s.source_type_2_ids),
        "dark_vessels": sum(1 for s in slick_list if s.source_type_3_ids),
    }
    return CeruleanSummary(
        count=len(slick_list),
        active_count=active_count,
        total_area_km2=total_area_km2,
        avg_machine_confidence=avg_conf,
        max_source_collated_score=max_score,
        source_counts=source_counts,
    )


def build_feature_collection(slicks: Sequence[CeruleanSlick]) -> Dict[str, Any]:
    """Return a GeoJSON FeatureCollection for map overlays."""

    features: List[Dict[str, Any]] = []
    for slick in slicks:
        props = slick.model_dump(exclude={"geometry"}, mode="json")
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(slick.geometry),
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


class CeruleanClient:
    """Tiny wrapper around the Cerulean REST API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        session: Optional[requests.Session] = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self.timeout = timeout

    def query_slicks(
        self,
        bbox: Tuple[float, float, float, float],
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
        min_source_score: Optional[float] = None,
        only_active: bool = True,
        sort: Optional[str] = "-max_source_collated_score",
        extra_filters: Optional[Iterable[str]] = None,
    ) -> CeruleanQueryResult:
        """Fetch slick detections intersecting the bounding box within a window."""

        params: Dict[str, str] = {
            "bbox": _format_bbox(bbox),
            "limit": str(limit),
        }
        if start or end:
            params["datetime"] = _format_datetime_range(start, end)

        filters: List[str] = list(extra_filters or [])
        if only_active:
            filters.append("active = true")
        if min_source_score is not None:
            filters.append(f"max_source_collated_score GTE {min_source_score}")
        if filters:
            params["filter"] = " AND ".join(filters)
        if sort:
            params["sortby"] = sort

        payload = self._get_json(_SLICKS_ENDPOINT, params=params)
        return CeruleanQueryResult.from_payload(payload)

    def _get_json(self, path: str, *, params: Optional[Mapping[str, str]] = None) -> Mapping[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self._session.get(url, params=params, timeout=self.timeout)
        if response.status_code != 200:
            raise CeruleanError(f"Cerulean API error {response.status_code}: {response.text}")
        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - network guard
            raise CeruleanError("Cerulean API returned invalid JSON") from exc


def _format_bbox(bbox: Tuple[float, float, float, float]) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox
    return f"{min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}"


def _format_datetime_range(start: Optional[datetime], end: Optional[datetime]) -> str:
    start_iso = _format_datetime(start) if start else ".."
    end_iso = _format_datetime(end) if end else ".."
    if start_iso == ".." and end_iso == "..":  # pragma: no cover - defensive
        raise ValueError("start or end datetime must be provided")
    return f"{start_iso}/{end_iso}"


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
