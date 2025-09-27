#!/usr/bin/env python3
"""Synthetic SeaOWL stream simulator.

Generates newline-delimited JSON (NDJSON) suitable for Phase 1 MVP tests.
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

ISOFORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


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
    parser.add_argument("--duration", type=int, default=7200, help="Duration in seconds (default: 7200)")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Samples per second (default: 1.0)")
    parser.add_argument("--output", type=Path, default=Path("data/ship/seaowl.ndjson"), help="NDJSON output path")
    parser.add_argument("--platform-id", default="vessel_001")
    parser.add_argument("--sensor-id", default="SeaOWL_01")
    parser.add_argument("--start-lat", type=float, default=40.689415, help="Starting latitude (default: 40.689415)")
    parser.add_argument("--start-lon", type=float, default=-74.006912, help="Starting longitude (default: -74.006912)")
    parser.add_argument("--speed", type=float, default=2.0, help="Speed in m/s (default: 2.0)")
    parser.add_argument("--heading", type=float, default=45.0, help="Heading in degrees")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-start", type=int, default=3600, help="Seconds from start to trigger event (use -1 for none)")
    parser.add_argument("--event-duration", type=int, default=1800, help="Event duration in seconds")
    parser.add_argument("--event-magnitude", type=float, default=2.5, help="Event peak multiplier for oil fluorescence")
    parser.add_argument("--no-event", action="store_true", help="Disable event injection")
    parser.add_argument("--sleep", action="store_true", help="Sleep between samples to emulate realtime output")

    args = parser.parse_args()

    cfg = SimulatorConfig(
        duration_s=args.duration,
        sample_rate_hz=args.sample_rate,
        seed=args.seed,
        output=args.output,
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


def generate_samples(cfg: SimulatorConfig) -> Iterable[Sample]:
    rng = np.random.default_rng(cfg.seed)
    total_steps = int(cfg.duration_s * cfg.sample_rate_hz)
    dt_s = 1.0 / cfg.sample_rate_hz
    lat = cfg.start_lat
    lon = cfg.start_lon
    ts = cfg.start_time

    # Baselines (engineering units) anchored to plausible SeaOWL ranges
    # - oil fluorescence: ppb crude-oil equivalent, near-zero background
    # - chlorophyll-a: ug/L, coastal background O(1–5)
    # - optical backscatter (700 nm, VSF @ ~117°): m^-1 sr^-1, O(1e-3–1e-2)
    oil = 0.05
    chl = 2.0
    back = 0.004
    temp = 12.5

    # Low-frequency variability via discrete Ornstein–Uhlenbeck steps
    def ou_step(x: float, mu: float, theta: float, sigma: float) -> float:
        # x_{t+1} = x + theta*(mu - x) + sigma*N(0,1)
        return x + theta * (mu - x) + sigma * rng.normal()

    event_start = cfg.event_start_s
    event_end = event_start + cfg.event_duration_s if event_start is not None and cfg.event_duration_s else None

    for step in range(total_steps):
        # OU updates (theta ~ 1/tau). Keep means stable; small stochastic wiggle.
        # Chosen taus: oil~10 min, chl~15 min, back~20 min at 1 Hz.
        oil = max(0.0, ou_step(oil, mu=0.05, theta=1.0 / (10 * 60.0), sigma=0.005))
        chl = max(0.0, ou_step(chl, mu=2.0, theta=1.0 / (15 * 60.0), sigma=0.02))
        back = max(0.0, ou_step(back, mu=0.004, theta=1.0 / (20 * 60.0), sigma=0.0003))
        # Add a tiny drift + noise to temperature; optional diurnal would be overkill here
        temp = max(0.0, temp + rng.normal(scale=0.003))

        event_phase = 0.0
        oil_evt = 0.0
        back_evt = 0.0
        if event_start is not None and event_end is not None and event_start <= step * dt_s <= event_end:
            center = event_start + (cfg.event_duration_s or 0) / 2
            width = (cfg.event_duration_s or 1) / 6
            event_phase = math.exp(-((step * dt_s - center) ** 2) / (2 * width**2))
            # Oil-like event: strong rise in oil fluorescence; do not assume Chlorophyll rises.
            # Slight co-variation in optical backscatter can occur with droplets; keep it modest.
            oil_evt = cfg.event_magnitude * event_phase
            # Do not force chlorophyll correlation during an oil event (avoid baked-in inference)
            # backscatter: small co-variation only (absolute, not compounding)
            back_evt = (cfg.event_magnitude * 0.15) * event_phase * 1e-3

        distance = cfg.speed_m_s * dt_s
        lat, lon = haversine_offset(lat, lon, distance, cfg.heading_deg)

        # Simple QC placeholders; leave anomaly detection to downstream scorer.
        qc_flags = {"range": 0, "spike": 0, "stuck": 0, "biofouling": 0}

        yield Sample(
            ts=ts,
            lat=lat,
            lon=lon,
            oil_fluor=oil + oil_evt,
            chlorophyll=chl,
            backscatter=back + back_evt,
            temperature_c=temp,
            qc_flags=qc_flags,
            event_phase=event_phase,
        )

        ts += timedelta(seconds=dt_s)


def write_samples(cfg: SimulatorConfig, samples: Iterable[Sample]) -> None:
    with cfg.output.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.to_dict(cfg)) + "\n")
            if getattr(cfg, "sleep", False):
                import time

                time.sleep(1.0 / cfg.sample_rate_hz)


def main() -> None:
    cfg = parse_args()
    write_samples(cfg, generate_samples(cfg))
    print(f"wrote synthetic stream to {cfg.output}")


if __name__ == "__main__":
    main()
