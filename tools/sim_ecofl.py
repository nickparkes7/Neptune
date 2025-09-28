#!/usr/bin/env python3
"""Synthetic ECO FL fluorescence stream simulator."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:  # Local import to reuse routing helpers
    from sim_seaowl import CERULEAN_SLICK_ROUTE, haversine_distance, haversine_offset
except Exception:  # pragma: no cover - fallback when executed via module path
    from tools.sim_seaowl import (
        CERULEAN_SLICK_ROUTE,
        haversine_distance,
        haversine_offset,
    )


ISOFORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass
class SimulatorConfig:
    duration_s: int = 7200
    sample_rate_hz: float = 1.0
    seed: Optional[int] = 42
    output: Path = Path("data/ship/ecofl_live.ndjson")
    platform_id: str = "vessel_001"
    sensor_id: str = "ECO_FL_01"
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_lat: float = 40.689415
    start_lon: float = -74.006912
    speed_m_s: float = 2.0
    heading_deg: float = 45.0
    depth_m: float = 5.0
    event_start_s: Optional[int] = 900
    event_duration_s: Optional[int] = 900
    event_magnitude: float = 4.5
    sleep: bool = False
    pattern: str = "linear"


@dataclass
class Sample:
    ts: datetime
    lat: float
    lon: float
    temperature_c: float
    chlorophyll_a: float
    fdom: float
    phycocyanin: float
    phycoerythrin: float
    depth_m: float
    qc_flags: dict
    event_phase: float

    def to_dict(self, cfg: SimulatorConfig) -> dict:
        return {
            "timestamp": self.ts.strftime(ISOFORMAT),
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "depth": round(self.depth_m, 2),
            "platform_id": cfg.platform_id,
            "sensor_id": cfg.sensor_id,
            "sensor_type": "eco_fl",
            "sample_rate_hz": cfg.sample_rate_hz,
            "mode": "synthetic",
            "temperature": round(self.temperature_c, 3),
            "chlorophyll_a": round(self.chlorophyll_a, 4),
            "fdom": round(self.fdom, 3),
            "phycocyanin": round(self.phycocyanin, 3),
            "phycoerythrin": round(self.phycoerythrin, 3),
            "qc_flags": self.qc_flags,
            "event_phase": round(self.event_phase, 4),
        }


def parse_args() -> SimulatorConfig:
    parser = argparse.ArgumentParser(description="ECO FL synthetic stream generator")
    parser.add_argument("--duration", type=int, default=7200)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("data/ship/ecofl_live.ndjson")
    )
    parser.add_argument("--platform-id", default="vessel_001")
    parser.add_argument("--sensor-id", default="ECO_FL_01")
    parser.add_argument("--start-lat", type=float, default=40.689415)
    parser.add_argument("--start-lon", type=float, default=-74.006912)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--heading", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--event-start", type=int, default=900)
    parser.add_argument("--event-duration", type=int, default=900)
    parser.add_argument("--event-magnitude", type=float, default=4.5)
    parser.add_argument("--depth", type=float, default=5.0)
    parser.add_argument("--no-event", action="store_true")
    parser.add_argument("--sleep", action="store_true")

    pattern_group = parser.add_mutually_exclusive_group()
    pattern_group.add_argument(
        "--pattern", choices=["linear", "cerulean_slick"], help="Trajectory pattern"
    )
    pattern_group.add_argument(
        "--nyc", action="store_true", help="Alias for linear trajectory"
    )
    pattern_group.add_argument(
        "--gulf", action="store_true", help="Alias for Cerulean slick route"
    )

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
        output_path = Path("data/ship/ecofl_gulf_live.ndjson")

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
        depth_m=args.depth,
        event_start_s=None
        if args.no_event or args.event_start < 0
        else args.event_start,
        event_duration_s=args.event_duration if not args.no_event else None,
        event_magnitude=args.event_magnitude,
        sleep=args.sleep,
        pattern=pattern,
    )
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.start_time = datetime.now(timezone.utc)
    return cfg


def _signal_series(
    cfg: SimulatorConfig, total_steps: int, dt_s: float, rng: np.random.Generator
) -> Iterable[Tuple[float, float, float, float, float]]:
    """Generate baseline fluorescence channels with slow drift."""

    temp = 14.5
    chl = 3.2
    fdom = 55.0
    pc = 4.0
    pe = 2.5

    def ou_step(x: float, mu: float, theta: float, sigma: float) -> float:
        return x + theta * (mu - x) + sigma * rng.normal()

    event_start = cfg.event_start_s
    event_end = (
        event_start + cfg.event_duration_s
        if event_start is not None and cfg.event_duration_s
        else None
    )

    for step in range(total_steps):
        temp = max(0.0, ou_step(temp, mu=14.2, theta=1.0 / (40 * 60.0), sigma=0.01))
        chl = max(0.0, ou_step(chl, mu=3.0, theta=1.0 / (30 * 60.0), sigma=0.03))
        fdom = max(0.0, ou_step(fdom, mu=55.0, theta=1.0 / (45 * 60.0), sigma=0.4))
        pc = max(0.0, ou_step(pc, mu=4.5, theta=1.0 / (30 * 60.0), sigma=0.08))
        pe = max(0.0, ou_step(pe, mu=2.8, theta=1.0 / (30 * 60.0), sigma=0.05))

        event_phase = 0.0
        chl_evt = 0.0
        pc_evt = 0.0
        pe_evt = 0.0
        fdom_evt = 0.0
        if (
            event_start is not None
            and event_end is not None
            and event_start <= step * dt_s <= event_end
        ):
            center = event_start + (cfg.event_duration_s or 0) / 2
            width = max((cfg.event_duration_s or 1) / 6, 1.0)
            event_phase = math.exp(-((step * dt_s - center) ** 2) / (2 * width**2))
            bloom_gain = cfg.event_magnitude * event_phase
            chl_evt = max(0.0, 8.0 * bloom_gain)
            pc_evt = max(0.0, 30.0 * bloom_gain)
            pe_evt = max(0.0, 18.0 * bloom_gain)
            fdom_evt = max(0.0, 12.0 * bloom_gain)

        yield (
            temp,
            chl + chl_evt,
            fdom + fdom_evt,
            pc + pc_evt,
            pe + pe_evt,
            event_phase,
        )


def _interpolate_route(
    points: List[Tuple[float, float]], total_steps: int
) -> List[Tuple[float, float]]:
    if total_steps <= 0:
        return []
    if not points:
        raise ValueError("route must contain at least one point")
    if len(points) == 1:
        return [points[0]] * total_steps

    cumulative: List[float] = [0.0]
    for idx in range(1, len(points)):
        cumulative.append(
            cumulative[-1]
            + haversine_distance(
                points[idx - 1][0], points[idx - 1][1], points[idx][0], points[idx][1]
            )
        )

    total_length = cumulative[-1]
    if total_length == 0:
        return [points[0]] * total_steps

    positions: List[Tuple[float, float]] = []
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
        lat = points[segment_idx][0] + ratio * (
            points[segment_idx + 1][0] - points[segment_idx][0]
        )
        lon = points[segment_idx][1] + ratio * (
            points[segment_idx + 1][1] - points[segment_idx][1]
        )
        positions.append((lat, lon))

    return positions


def _generate_samples_linear(cfg: SimulatorConfig) -> Iterable[Sample]:
    rng = np.random.default_rng(cfg.seed)
    total_steps = int(cfg.duration_s * cfg.sample_rate_hz)
    dt_s = 1.0 / cfg.sample_rate_hz
    lat = cfg.start_lat
    lon = cfg.start_lon
    ts = cfg.start_time

    for temp, chl, fdom, pc, pe, event_phase in _signal_series(
        cfg, total_steps, dt_s, rng
    ):
        distance = cfg.speed_m_s * dt_s
        lat, lon = haversine_offset(lat, lon, distance, cfg.heading_deg)
        qc_flags = {"range": 0, "spike": 0, "stuck": 0, "biofouling": 0}
        yield Sample(
            ts=ts,
            lat=lat,
            lon=lon,
            temperature_c=temp,
            chlorophyll_a=chl,
            fdom=fdom,
            phycocyanin=pc,
            phycoerythrin=pe,
            depth_m=cfg.depth_m,
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

    for idx, values in enumerate(_signal_series(cfg, total_steps, dt_s, rng)):
        temp, chl, fdom, pc, pe, event_phase = values
        lat, lon = route[idx]
        qc_flags = {"range": 0, "spike": 0, "stuck": 0, "biofouling": 0}
        yield Sample(
            ts=ts,
            lat=lat,
            lon=lon,
            temperature_c=temp,
            chlorophyll_a=chl,
            fdom=fdom,
            phycocyanin=pc,
            phycoerythrin=pe,
            depth_m=cfg.depth_m,
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
                fh.flush()
                time.sleep(1.0 / cfg.sample_rate_hz)


def main() -> None:
    cfg = parse_args()
    write_samples(cfg, generate_samples(cfg))
    print(f"wrote ECO FL synthetic stream to {cfg.output}")


if __name__ == "__main__":
    main()
