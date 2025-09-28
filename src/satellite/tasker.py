"""Sentinel-1 tasking utilities for Phase 1.

This module operates on a local catalog of staged Sentinel-1 scenes. Given an
area of interest (AOI) and a time window, it surfaces candidate scenes that
should already exist on disk so downstream steps (slick detection, linking)
can operate deterministically without hitting external APIs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from pydantic import Field, validator

from common.pydantic_compat import CompatBaseModel

DEFAULT_CATALOG = Path("configs/s1_catalog.json")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SceneRef(CompatBaseModel):
    """Metadata describing a locally staged Sentinel-1 scene."""

    scene_id: str = Field(..., description="Sentinel scene identifier")
    acquired: datetime = Field(..., description="Acquisition time (UTC)")
    platform: str = Field(..., description="Satellite platform (e.g., S1A)")
    polarizations: List[str] = Field(default_factory=list)
    bbox: Tuple[float, float, float, float] = Field(
        ..., description="Bounding box (min_lon, min_lat, max_lon, max_lat)"
    )
    path: Path = Field(..., description="Local raster path (GeoTIFF/COG)")

    @validator("acquired", pre=True)
    def _parse_datetime(cls, value: object) -> datetime:  # noqa: D401 - pydantic pattern
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            dt = _parse_datetime(value)
        else:
            raise TypeError("acquired must be datetime or ISO8601 string")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @validator("bbox")
    def _validate_bbox(cls, value: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        min_lon, min_lat, max_lon, max_lat = value
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bbox must be ordered as min_lon < max_lon and min_lat < max_lat")
        return value

    class Config:
        json_encoders = {Path: lambda p: str(p)}


class TaskRequest(CompatBaseModel):
    """AOI and time window describing a Sentinel-1 tasking request."""

    bbox: Tuple[float, float, float, float]
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    @validator("start", "end", pre=True)
    def _parse_optional_datetime(cls, value: object) -> Optional[datetime]:  # noqa: D401
        if value in (None, "", False):
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            dt = _parse_datetime(value)
        else:
            raise TypeError("start/end must be datetime or ISO8601 string")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @validator("bbox")
    def _validate_bbox(cls, value: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        min_lon, min_lat, max_lon, max_lat = value
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bbox must be ordered as min_lon < max_lon and min_lat < max_lat")
        return value


@dataclass
class TaskerConfig:
    """Runtime options for the tasker."""

    catalog_path: Path = DEFAULT_CATALOG
    max_results: int = 2
    allowed_platforms: Sequence[str] = ("S1", "S1A", "S1B")


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


def load_catalog(path: Path) -> List[SceneRef]:
    """Load the staged scene catalog."""

    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    data = json.loads(path.read_text())
    scenes = [SceneRef.model_validate(item) for item in data]
    return scenes


def filter_catalog(
    scenes: Iterable[SceneRef],
    request: TaskRequest,
    config: Optional[TaskerConfig] = None,
) -> List[SceneRef]:
    """Select scenes intersecting the AOI and time window."""

    cfg = config or TaskerConfig()
    filtered: List[SceneRef] = []
    for scene in scenes:
        if cfg.allowed_platforms and not any(
            scene.platform.startswith(prefix) for prefix in cfg.allowed_platforms
        ):
            continue
        if request.start and scene.acquired < request.start:
            continue
        if request.end and scene.acquired > request.end:
            continue
        if not _bbox_intersects(scene.bbox, request.bbox):
            continue
        filtered.append(scene)

    filtered.sort(key=lambda s: s.acquired)
    if cfg.max_results:
        filtered = filtered[: cfg.max_results]
    return filtered


def task_satellite(
    request: TaskRequest,
    config: Optional[TaskerConfig] = None,
) -> List[SceneRef]:
    """Return staged Sentinel-1 scenes for the provided request."""

    cfg = config or TaskerConfig()
    scenes = load_catalog(cfg.catalog_path)
    return filter_catalog(scenes, request, cfg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--start", help="Start timestamp (ISO8601)")
    parser.add_argument("--end", help="End timestamp (ISO8601)")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--max-results", type=int, default=2)
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    request = TaskRequest(bbox=_parse_bbox(args.bbox), start=args.start, end=args.end)
    config = TaskerConfig(catalog_path=args.catalog, max_results=args.max_results)
    scenes = task_satellite(request, config)
    payload = [scene.model_dump(mode="json") for scene in scenes]
    text = json.dumps(payload, indent=2 if args.pretty else None)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + ("\n" if not text.endswith("\n") else ""))
    else:
        print(text)
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_bbox(value: str) -> Tuple[float, float, float, float]:
    try:
        parts = [float(item.strip()) for item in value.split(",")]
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("bbox must be comma-separated floats") from exc
    if len(parts) != 4:
        raise ValueError("bbox must contain exactly four numbers")
    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox must be ordered as min_lon < max_lon and min_lat < max_lat")
    return min_lon, min_lat, max_lon, max_lat


def _bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _parse_datetime(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
