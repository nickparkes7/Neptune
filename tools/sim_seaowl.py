#!/usr/bin/env python3
"""Synthetic SeaOWL stream simulator.

Generates newline-delimited JSON (NDJSON) suitable for Phase 1 MVP tests.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

ISOFORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

# Derived from provided Cerulean slick MultiPolygon GeoJSON (lon/lat reordered to lat/lon).
CERULEAN_SLICK_ROUTE = [
    (28.473102, 49.617664),
    (28.473102, 49.619724),
    (28.472415, 49.619724),
    (28.472415, 49.621097),
    (28.471042, 49.621097),
    (28.471042, 49.621783),
    (28.470355, 49.621783),
    (28.470355, 49.622470),
    (28.471042, 49.622470),
    (28.471042, 49.623157),
    (28.472415, 49.623157),
    (28.472415, 49.625903),
    (28.471729, 49.625903),
    (28.471729, 49.626590),
    (28.471042, 49.626590),
    (28.471042, 49.625903),
    (28.470355, 49.625903),
    (28.470355, 49.626590),
    (28.468295, 49.626590),
    (28.468295, 49.628650),
    (28.467609, 49.628650),
    (28.467609, 49.631396),
    (28.466235, 49.631396),
    (28.466235, 49.632770),
    (28.467609, 49.632770),
    (28.467609, 49.636203),
    (28.466235, 49.636203),
    (28.466235, 49.637576),
    (28.465549, 49.637576),
    (28.465549, 49.639636),
    (28.464175, 49.639636),
    (28.464175, 49.638950),
    (28.463489, 49.638950),
    (28.463489, 49.637576),
    (28.462802, 49.637576),
    (28.462802, 49.636890),
    (28.461429, 49.636890),
    (28.461429, 49.636203),
    (28.460742, 49.636203),
    (28.460742, 49.635516),
    (28.460056, 49.635516),
    (28.460056, 49.634830),
    (28.460742, 49.634830),
    (28.460742, 49.633456),
    (28.459369, 49.633456),
    (28.459369, 49.632770),
    (28.458682, 49.632770),
    (28.458682, 49.632083),
    (28.457996, 49.632083),
    (28.457996, 49.631396),
    (28.458682, 49.631396),
    (28.458682, 49.630710),
    (28.459369, 49.630710),
    (28.459369, 49.628650),
    (28.458682, 49.628650),
    (28.458682, 49.627963),
    (28.459369, 49.627963),
    (28.459369, 49.627277),
    (28.460742, 49.627277),
    (28.460742, 49.625217),
    (28.460056, 49.625217),
    (28.460056, 49.623843),
    (28.459369, 49.623843),
    (28.459369, 49.622470),
    (28.460056, 49.622470),
    (28.460056, 49.620410),
    (28.460742, 49.620410),
    (28.460742, 49.619724),
    (28.461429, 49.619724),
    (28.461429, 49.619037),
    (28.462802, 49.619037),
    (28.462802, 49.618350),
    (28.463489, 49.618350),
    (28.463489, 49.617664),
    (28.464175, 49.617664),
    (28.464175, 49.616977),
    (28.464862, 49.616977),
    (28.464862, 49.616290),
    (28.465549, 49.616290),
    (28.465549, 49.616977),
    (28.466235, 49.616977),
    (28.466235, 49.617664),
    (28.466922, 49.617664),
    (28.466922, 49.616977),
    (28.468295, 49.616977),
    (28.468295, 49.616290),
    (28.470355, 49.616290),
    (28.470355, 49.615604),
    (28.471042, 49.615604),
    (28.471042, 49.616290),
    (28.472415, 49.616290),
    (28.472415, 49.617664),
    (28.473102, 49.617664),
    (28.484088, 49.632083),
    (28.484088, 49.635516),
    (28.482715, 49.635516),
    (28.482715, 49.634830),
    (28.483401, 49.634830),
    (28.483401, 49.633456),
    (28.482715, 49.633456),
    (28.482715, 49.634143),
    (28.482028, 49.634143),
    (28.482028, 49.632083),
    (28.484088, 49.632083),
    (28.482028, 49.634143),
    (28.482028, 49.634830),
    (28.481342, 49.634830),
    (28.481342, 49.634143),
    (28.482028, 49.634143),
    (28.473102, 49.612170),
    (28.473102, 49.614230),
    (28.472415, 49.614230),
    (28.472415, 49.615604),
    (28.471729, 49.615604),
    (28.471729, 49.614917),
    (28.471042, 49.614917),
    (28.471042, 49.613544),
    (28.471729, 49.613544),
    (28.471729, 49.612857),
    (28.472415, 49.612857),
    (28.472415, 49.612170),
    (28.473102, 49.612170),
]


@dataclass
class SimulatorConfig:
    duration_s: int = 7200
    sample_rate_hz: float = 1.0
    seed: Optional[int] = 42
    output: Path = Path("data/ship/seaowl.ndjson")
    platform_id: str = "vessel_001"
    sensor_id: str = "SeaOWL_01"
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_lat: float = 40.689415  # Brooklyn Marine Terminal
    start_lon: float = -74.006912  # Brooklyn Marine Terminal
    speed_m_s: float = 2.0
    heading_deg: float = 45.0
    event_start_s: Optional[int] = 3600
    event_duration_s: Optional[int] = 1800
    event_magnitude: float = 2.5
    sleep: bool = False
    pattern: str = "linear"


@dataclass
class Sample:
    ts: datetime
    lat: float
    lon: float
    oil_fluor: float
    chlorophyll: float
    backscatter: float
    temperature_c: float
    qc_flags: dict
    event_phase: float

    def to_dict(self, cfg: SimulatorConfig) -> dict:
        return {
            "ts": self.ts.strftime(ISOFORMAT),
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "depth_m": 0.0,
            "platform_id": cfg.platform_id,
            "sensor_id": cfg.sensor_id,
            "sensor_type": "seaowl",
            "sample_rate_hz": cfg.sample_rate_hz,
            "mode": "synthetic",
            "oil_fluor_ppb": round(self.oil_fluor, 4),
            "chlorophyll_ug_per_l": round(self.chlorophyll, 4),
            "backscatter_m-1_sr-1": round(self.backscatter, 5),
            "temperature_c": round(self.temperature_c, 3),
            "qc_flags": self.qc_flags,
            "event_phase": round(self.event_phase, 4),
        }


def parse_args() -> SimulatorConfig:
    parser = argparse.ArgumentParser(description="SeaOWL synthetic stream generator")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds (default: 60)")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Samples per second (default: 1.0)")
    parser.add_argument("--output", type=Path, default=Path("data/ship/seaowl.ndjson"), help="NDJSON output path")
    parser.add_argument("--platform-id", default="vessel_001")
    parser.add_argument("--sensor-id", default="SeaOWL_01")
    pattern_group = parser.add_mutually_exclusive_group()
    pattern_group.add_argument("--pattern", choices=["linear", "cerulean_slick"], help="Synthetic trajectory pattern (deprecated, prefer --nyc/--gulf)")
    pattern_group.add_argument("--nyc", action="store_true", help="Alias for --pattern linear")
    pattern_group.add_argument("--gulf", action="store_true", help="Alias for --pattern cerulean_slick")
    parser.add_argument("--start-lat", type=float, default=40.689415, help="Starting latitude (default: 40.689415)")
    parser.add_argument("--start-lon", type=float, default=-74.006912, help="Starting longitude (default: -74.006912)")
    parser.add_argument("--speed", type=float, default=2.0, help="Speed in m/s (default: 2.0)")
    parser.add_argument("--heading", type=float, default=45.0, help="Heading in degrees")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-start", type=int, default=20, help="Seconds from start to trigger event (use -1 for none)")
    parser.add_argument("--event-duration", type=int, default=20, help="Event duration in seconds")
    parser.add_argument("--event-magnitude", type=float, default=3.0, help="Event peak multiplier for oil fluorescence")
    parser.add_argument("--no-event", action="store_true", help="Disable event injection")
    parser.add_argument("--sleep", action="store_true", help="Sleep between samples to emulate realtime output")

    args = parser.parse_args()

    if args.pattern:
        pattern = args.pattern
    elif args.gulf:
        pattern = "cerulean_slick"
    else:
        pattern = "linear"

    default_output = parser.get_default("output")
    output_path = args.output
    if pattern == "cerulean_slick" and output_path == default_output:
        output_path = Path("data/ship/seaowl_cerulean.ndjson")

    cfg = SimulatorConfig(
        duration_s=args.duration,
        sample_rate_hz=args.sample_rate,
        seed=args.seed,
        output=output_path,
        platform_id=args.platform_id,
        sensor_id=args.sensor_id,
        start_lat=args.start_lat,
        start_lon=args.start_lon,
        speed_m_s=args.speed,
        heading_deg=args.heading,
        event_start_s=None if args.no_event or args.event_start < 0 else args.event_start,
        event_duration_s=args.event_duration if not args.no_event else None,
        event_magnitude=args.event_magnitude,
        sleep=args.sleep,
        pattern=pattern,
    )
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.start_time = datetime.now(timezone.utc)

    return cfg


def haversine_offset(lat: float, lon: float, distance_m: float, bearing_deg: float) -> tuple[float, float]:
    radius = 6378137.0
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(distance_m / radius) + math.cos(lat1) * math.sin(distance_m / radius) * math.cos(bearing))
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(distance_m / radius) * math.cos(lat1),
        math.cos(distance_m / radius) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6378137.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _signal_series(cfg: SimulatorConfig, total_steps: int, dt_s: float, rng: np.random.Generator) -> Iterable[tuple[float, float, float, float, float]]:
    oil = 0.05
    chl = 2.0
    back = 0.004
    temp = 12.5

    def ou_step(x: float, mu: float, theta: float, sigma: float) -> float:
        return x + theta * (mu - x) + sigma * rng.normal()

    event_start = cfg.event_start_s
    event_end = event_start + cfg.event_duration_s if event_start is not None and cfg.event_duration_s else None

    for step in range(total_steps):
        oil = max(0.0, ou_step(oil, mu=0.05, theta=1.0 / (10 * 60.0), sigma=0.005))
        chl = max(0.0, ou_step(chl, mu=2.0, theta=1.0 / (15 * 60.0), sigma=0.02))
        back = max(0.0, ou_step(back, mu=0.004, theta=1.0 / (20 * 60.0), sigma=0.0003))
        temp = max(0.0, temp + rng.normal(scale=0.003))

        event_phase = 0.0
        oil_evt = 0.0
        back_evt = 0.0
        if event_start is not None and event_end is not None and event_start <= step * dt_s <= event_end:
            center = event_start + (cfg.event_duration_s or 0) / 2
            width = (cfg.event_duration_s or 1) / 6
            event_phase = math.exp(-((step * dt_s - center) ** 2) / (2 * width**2))
            oil_evt = cfg.event_magnitude * event_phase
            back_evt = (cfg.event_magnitude * 0.15) * event_phase * 1e-3

        yield (
            oil + oil_evt,
            chl,
            back + back_evt,
            temp,
            event_phase,
        )


def _interpolate_route(points: List[tuple[float, float]], total_steps: int) -> List[tuple[float, float]]:
    if total_steps <= 0:
        return []
    if not points:
        raise ValueError("route must contain at least one point")
    if len(points) == 1:
        return [points[0]] * total_steps

    cumulative: List[float] = [0.0]
    for idx in range(1, len(points)):
        cumulative.append(cumulative[-1] + haversine_distance(points[idx - 1][0], points[idx - 1][1], points[idx][0], points[idx][1]))

    total_length = cumulative[-1]
    if total_length == 0:
        return [points[0]] * total_steps

    positions: List[tuple[float, float]] = []
    segment_idx = 0
    for step in range(total_steps):
        target = 0.0 if total_steps == 1 else total_length * (step / (total_steps - 1))
        while segment_idx < len(points) - 2 and cumulative[segment_idx + 1] < target:
            segment_idx += 1
        seg_length = cumulative[segment_idx + 1] - cumulative[segment_idx]
        if seg_length == 0:
            positions.append(points[segment_idx])
            continue
        ratio = (target - cumulative[segment_idx]) / seg_length
        lat = points[segment_idx][0] + ratio * (points[segment_idx + 1][0] - points[segment_idx][0])
        lon = points[segment_idx][1] + ratio * (points[segment_idx + 1][1] - points[segment_idx][1])
        positions.append((lat, lon))

    return positions


def _generate_samples_linear(cfg: SimulatorConfig) -> Iterable[Sample]:
    rng = np.random.default_rng(cfg.seed)
    total_steps = int(cfg.duration_s * cfg.sample_rate_hz)
    dt_s = 1.0 / cfg.sample_rate_hz
    lat = cfg.start_lat
    lon = cfg.start_lon
    ts = cfg.start_time

    for oil_fluor, chl, back, temp, event_phase in _signal_series(cfg, total_steps, dt_s, rng):
        distance = cfg.speed_m_s * dt_s
        lat, lon = haversine_offset(lat, lon, distance, cfg.heading_deg)
        qc_flags = {"range": 0, "spike": 0, "stuck": 0, "biofouling": 0}
        yield Sample(
            ts=ts,
            lat=lat,
            lon=lon,
            oil_fluor=oil_fluor,
            chlorophyll=chl,
            backscatter=back,
            temperature_c=temp,
            qc_flags=qc_flags,
            event_phase=event_phase,
        )
        ts += timedelta(seconds=dt_s)


def _generate_samples_cerulean(cfg: SimulatorConfig) -> Iterable[Sample]:
    rng = np.random.default_rng(cfg.seed)
    total_steps = int(cfg.duration_s * cfg.sample_rate_hz)
    dt_s = 1.0 / cfg.sample_rate_hz
    ts = cfg.start_time
    route = _interpolate_route(CERULEAN_SLICK_ROUTE, total_steps)

    for idx, measurement in enumerate(_signal_series(cfg, total_steps, dt_s, rng)):
        oil_fluor, chl, back, temp, event_phase = measurement
        lat, lon = route[idx]
        qc_flags = {"range": 0, "spike": 0, "stuck": 0, "biofouling": 0}
        yield Sample(
            ts=ts,
            lat=lat,
            lon=lon,
            oil_fluor=oil_fluor,
            chlorophyll=chl,
            backscatter=back,
            temperature_c=temp,
            qc_flags=qc_flags,
            event_phase=event_phase,
        )
        ts += timedelta(seconds=dt_s)


def generate_samples(cfg: SimulatorConfig) -> Iterable[Sample]:
    if cfg.pattern == "cerulean_slick":
        yield from _generate_samples_cerulean(cfg)
    else:
        yield from _generate_samples_linear(cfg)


def write_samples(cfg: SimulatorConfig, samples: Iterable[Sample]) -> None:
    with cfg.output.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.to_dict(cfg)) + "\n")
            if cfg.sleep:
                # Flush so downstream streamers see each sample immediately.
                fh.flush()
                time.sleep(1.0 / cfg.sample_rate_hz)


def main() -> None:
    cfg = parse_args()
    write_samples(cfg, generate_samples(cfg))
    print(f"wrote synthetic stream to {cfg.output}")


if __name__ == "__main__":
    main()
