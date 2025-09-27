"""Hybrid oil anomaly detection for SeaOWL streams.

Implements an adaptive-baseline + absolute-threshold + persistence design
focused on the oil fluorescence channel, with optional cross-parameter
gating using chlorophyll/backscatter to reduce biogenic false positives.

This module is deliberately simple and testable. It avoids heavy
dependencies and operates on a pandas DataFrame, returning the same with
additional columns that describe the computed baseline, scores, and
warn/alarm flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..ingest.ship_stream import iter_ndjson, normalize_record  # type: ignore
except Exception:  # pragma: no cover
    from ingest.ship_stream import iter_ndjson, normalize_record


EPS = 1e-9


@dataclass
class HybridOilAlertConfig:
    """Configuration for :class:`HybridOilAlertScorer`.

    Defaults are chosen for the synthetic 1 Hz demo stream; field values
    should be calibrated per site and risk tolerance.
    """

    # Adaptive baseline half-life (seconds). Converted to EWMA alpha.
    baseline_halflife_s: float = 20 * 60.0
    scale_halflife_s: float = 20 * 60.0

    # Robustness: clip residuals when updating baseline/scale
    clip_k: float = 3.0

    # Relative thresholds (z)
    z_warn: float = 10.0
    z_alarm: float = 25.0
    z_clear: float = 5.0

    # Absolute thresholds (engineering units, ppb oil fluorescence)
    abs_warn: float = 1.0
    abs_alarm: float = 2.0
    abs_clear: float = 0.8

    # Persistence (samples at 1 Hz)
    k_warn: int = 5
    k_alarm: int = 10
    k_clear: int = 10

    # Cross-parameter gating (penalize oil z if chl/back scatter is elevated)
    use_chl: bool = True
    use_back: bool = False
    z_chl_low: float = 0.5
    z_back_low: float = 0.5
    w_chl: float = 0.5
    w_back: float = 0.2

    # Initialization window (seconds) used only for a conservative seed
    init_window_s: int = 120


class HybridOilAlertScorer:
    """Adaptive baseline + absolute thresholds + persistence for oil channel."""

    OIL = "oil_fluor_ppb"
    CHL = "chlorophyll_ug_per_l"
    BACK = "backscatter_m-1_sr-1"

    def __init__(self, config: Optional[HybridOilAlertConfig] = None) -> None:
        self.cfg = config or HybridOilAlertConfig()

    # region public API
    def score_ndjson(self, path: Path) -> pd.DataFrame:
        return self.score_stream(iter_ndjson(path))

    def score_stream(self, records: Iterable[dict]) -> pd.DataFrame:
        rows = [normalize_record(r) for r in records]
        if not rows:
            raise ValueError("records iterable yielded no rows")
        df = pd.DataFrame(rows)
        return self.score_dataframe(df)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("dataframe is empty")

        needed = {"ts", self.OIL}
        missing = needed.difference(df.columns)
        if missing:
            raise KeyError(f"missing required columns: {', '.join(sorted(missing))}")

        frame = df.copy()
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        frame = frame.sort_values("ts").reset_index(drop=True)

        # Prepare numeric series
        x = pd.to_numeric(frame[self.OIL], errors="coerce").to_numpy(dtype=float)
        chl = pd.to_numeric(frame.get(self.CHL, np.nan), errors="coerce").to_numpy(dtype=float)
        back = pd.to_numeric(frame.get(self.BACK, np.nan), errors="coerce").to_numpy(dtype=float)

        # Initialize arrays
        n = len(frame)
        b = np.zeros(n, dtype=float)  # baseline
        s = np.zeros(n, dtype=float)  # ewma abs residual (scale)
        z = np.zeros(n, dtype=float)  # oil z
        z_eff = np.zeros(n, dtype=float)  # gated z
        chl_z = np.zeros(n, dtype=float)
        back_z = np.zeros(n, dtype=float)
        warn = np.zeros(n, dtype=bool)
        alarm = np.zeros(n, dtype=bool)

        # EWMA alphas from half-lives (assumes dt=1s)
        alpha = np.log(2.0) / max(self.cfg.baseline_halflife_s, 1.0)
        beta = np.log(2.0) / max(self.cfg.scale_halflife_s, 1.0)

        # Seed baseline/scale using the first init_window samples (robust)
        w0 = int(min(self.cfg.init_window_s, n)) or 1
        b0 = float(np.nanmedian(x[:w0]))
        s0 = float(np.nanmedian(np.abs(x[:w0] - b0)))
        if not np.isfinite(s0) or s0 < EPS:
            s0 = 1e-3
        b[0] = b0
        s[0] = s0

        # Helper for robust EWMA updates
        def clip_residual(r: float, scale: float) -> float:
            k = self.cfg.clip_k
            limit = max(k * scale, 1e-6)
            return float(np.clip(r, -limit, limit))

        warn_ctr = 0
        alarm_ctr = 0
        clear_ctr = 0

        # Baselines for chl/back (independent, simple EWMA of level + abs dev)
        cb = float(np.nanmedian(chl[:w0])) if np.isfinite(chl[:w0]).any() else 0.0
        cs = float(np.nanmedian(np.abs(chl[:w0] - cb))) if np.isfinite(chl[:w0]).any() else 1e-3
        bb = float(np.nanmedian(back[:w0])) if np.isfinite(back[:w0]).any() else 0.0
        bs = float(np.nanmedian(np.abs(back[:w0] - bb))) if np.isfinite(back[:w0]).any() else 1e-3

        for i in range(1, n):
            # Update baselines if not in event (hysteresis via clear counter)
            updating = not warn[i - 1] and not alarm[i - 1]

            # Oil channel updates
            r = x[i] - b[i - 1]
            rc = clip_residual(r, s[i - 1])
            if updating:
                b[i] = (1.0 - alpha) * b[i - 1] + alpha * (b[i - 1] + rc)
                s[i] = (1.0 - beta) * s[i - 1] + beta * abs(rc)
            else:
                b[i] = b[i - 1]
                s[i] = s[i - 1]

            denom = max(s[i], EPS)
            z[i] = (x[i] - b[i]) / denom

            # Supporting channels
            if np.isfinite(chl[i]):
                cr = chl[i] - cb
                if updating:
                    cb = (1.0 - alpha) * cb + alpha * (cb + clip_residual(cr, cs))
                    cs = (1.0 - beta) * cs + beta * abs(clip_residual(cr, cs))
                chl_z[i] = (chl[i] - cb) / max(cs, EPS)
            else:
                chl_z[i] = 0.0

            if np.isfinite(back[i]):
                br = back[i] - bb
                if updating:
                    bb = (1.0 - alpha) * bb + alpha * (bb + clip_residual(br, bs))
                    bs = (1.0 - beta) * bs + beta * abs(clip_residual(br, bs))
                back_z[i] = (back[i] - bb) / max(bs, EPS)
            else:
                back_z[i] = 0.0

            # Cross-parameter gating
            gate = 0.0
            if self.cfg.use_chl:
                gate += self.cfg.w_chl * max(0.0, chl_z[i] - self.cfg.z_chl_low)
            if self.cfg.use_back:
                gate += self.cfg.w_back * max(0.0, back_z[i] - self.cfg.z_back_low)
            z_eff[i] = z[i] - gate

            # Threshold logic with persistence
            rel_hit_warn = z_eff[i] >= self.cfg.z_warn
            rel_hit_alarm = z_eff[i] >= self.cfg.z_alarm
            abs_hit_warn = x[i] >= self.cfg.abs_warn
            abs_hit_alarm = x[i] >= self.cfg.abs_alarm

            warn_ctr = warn_ctr + 1 if (rel_hit_warn or abs_hit_warn) else 0
            alarm_ctr = alarm_ctr + 1 if (rel_hit_alarm or abs_hit_alarm) else 0

            do_alarm = abs_hit_alarm or (alarm_ctr >= self.cfg.k_alarm)
            do_warn = (warn_ctr >= self.cfg.k_warn) or do_alarm

            alarm[i] = bool(do_alarm)
            warn[i] = bool(do_warn and not do_alarm)

            # Clear tracking: require quiet period
            if not (rel_hit_warn or abs_hit_warn or rel_hit_alarm or abs_hit_alarm):
                if (z_eff[i] < self.cfg.z_clear) and (x[i] < self.cfg.abs_clear):
                    clear_ctr += 1
                else:
                    clear_ctr = 0
                if clear_ctr >= self.cfg.k_clear:
                    warn_ctr = 0
                    alarm_ctr = 0

        out = frame.copy()
        out["oil_baseline"] = b
        out["oil_scale"] = s
        out["oil_z"] = z
        out["oil_z_eff"] = z_eff
        out["chl_z"] = chl_z
        out["back_z"] = back_z
        out["oil_warn"] = warn
        out["oil_alarm"] = alarm
        return out


__all__ = ["HybridOilAlertScorer", "HybridOilAlertConfig"]
