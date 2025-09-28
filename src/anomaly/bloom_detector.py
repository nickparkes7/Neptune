"""Detection helpers for ECO FL synthetic bloom signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BloomSignal:
    """Summary of a detected algal bloom signal from ECO FL."""

    detected: bool
    peak_channel: Optional[str] = None
    peak_z: float = 0.0
    peak_value: float = 0.0
    ts_peak: Optional[datetime] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    sample_count: int = 0

    def to_trace(self) -> Dict[str, object]:
        return {
            "detected": self.detected,
            "peak_channel": self.peak_channel,
            "peak_z": self.peak_z,
            "peak_value": self.peak_value,
            "ts_peak": self.ts_peak.isoformat() if self.ts_peak else None,
            "lat": self.lat,
            "lon": self.lon,
            "metrics": self.metrics,
            "sample_count": self.sample_count,
        }


def detect_bloom_signal(
    eco_path: Path,
    *,
    window: timedelta,
    baseline: timedelta,
    start: datetime,
    end: datetime,
    z_threshold: float = 8.0,
) -> BloomSignal:
    """Detect high-z bloom events from ECO FL NDJSON stream."""

    if not eco_path.exists():
        return BloomSignal(detected=False)

    try:
        df = pd.read_json(eco_path, lines=True)
    except ValueError:
        return BloomSignal(detected=False)

    if df.empty or "timestamp" not in df:
        return BloomSignal(detected=False)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return BloomSignal(detected=False)

    window_start = (start - window).astimezone(timezone.utc)
    window_end = (end + window).astimezone(timezone.utc)
    window_df = df[
        (df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)
    ].copy()
    if window_df.empty:
        return BloomSignal(detected=False)

    baseline_end = (start - timedelta(seconds=1)).astimezone(timezone.utc)
    baseline_start = (baseline_end - baseline).astimezone(timezone.utc)
    baseline_df = df[
        (df["timestamp"] >= baseline_start) & (df["timestamp"] <= baseline_end)
    ].copy()
    if baseline_df.empty():
        baseline_df = window_df.iloc[: max(1, min(len(window_df), 600))]

    channels = ["chlorophyll_a", "phycocyanin", "phycoerythrin"]
    for channel in channels:
        window_df[channel] = pd.to_numeric(window_df.get(channel), errors="coerce")
        baseline_df[channel] = pd.to_numeric(baseline_df.get(channel), errors="coerce")

    metrics: Dict[str, float] = {}
    z_scores = {}
    for channel in channels:
        baseline_values = baseline_df[channel].dropna()
        if baseline_values.empty:
            baseline_mean = (
                float(window_df[channel].dropna().median())
                if not window_df[channel].dropna().empty
                else 0.0
            )
            baseline_std = max(abs(baseline_mean) * 0.05, 1e-3)
        else:
            baseline_mean = float(baseline_values.mean())
            baseline_std = float(baseline_values.std(ddof=0))
            if not np.isfinite(baseline_std) or baseline_std < 1e-6:
                baseline_std = max(abs(baseline_mean) * 0.05, 1e-3)

        metrics[f"{channel}_baseline_mean"] = baseline_mean
        metrics[f"{channel}_baseline_std"] = baseline_std

        channel_series = window_df[channel].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            channel_z = (channel_series - baseline_mean) / baseline_std
        window_df[f"{channel}_z"] = channel_z
        metrics[f"{channel}_max"] = (
            float(np.nanmax(channel_series)) if channel_series.size else 0.0
        )
        metrics[f"{channel}_max_z"] = (
            float(np.nanmax(channel_z)) if channel_z.size else 0.0
        )
        z_scores[channel] = metrics[f"{channel}_max_z"]

    if not z_scores:
        return BloomSignal(detected=False, metrics=metrics, sample_count=len(window_df))

    peak_channel = max(z_scores, key=z_scores.get)
    peak_z = z_scores[peak_channel]
    if not np.isfinite(peak_z) or peak_z < z_threshold:
        return BloomSignal(detected=False, metrics=metrics, sample_count=len(window_df))

    peak_idx = window_df[f"{peak_channel}_z"].idxmax()
    peak_row = window_df.loc[peak_idx]
    peak_value = float(peak_row.get(peak_channel, 0.0))
    if not np.isfinite(peak_value):
        peak_value = 0.0
    ts_peak = peak_row.get("timestamp")
    lat = peak_row.get("lat")
    lon = peak_row.get("lon")

    def _coerce_float(value) -> Optional[float]:
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            return float(value)
        except Exception:  # noqa: BLE001
            return None

    return BloomSignal(
        detected=True,
        peak_channel=peak_channel,
        peak_z=float(peak_z),
        peak_value=peak_value,
        ts_peak=ts_peak if isinstance(ts_peak, datetime) else None,
        lat=_coerce_float(lat),
        lon=_coerce_float(lon),
        metrics=metrics,
        sample_count=len(window_df),
    )


__all__ = ["BloomSignal", "detect_bloom_signal"]
