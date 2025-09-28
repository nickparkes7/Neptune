#!/usr/bin/env python3
"""Generate a visually rich agent brief from cached SeaOWL evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.briefing import (  # noqa: E402
    ArtifactRef,
    build_agent_brief,
    brief_to_markdown,
    score_stream,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="seaowl_demo",
        help="Scenario identifier captured in the brief metadata.",
    )
    parser.add_argument(
        "--stream",
        type=Path,
        default=ROOT / "data/ship/seaowl_live.ndjson",
        help="SeaOWL NDJSON stream to summarise.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "artifacts/briefs",
        help="Directory for brief outputs (JSON/Markdown/media).",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional ISO timestamp override for generated_at field (UTC).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.stream.exists():
        raise SystemExit(f"Stream not found: {args.stream}")

    outdir: Path = args.outdir
    media_dir = outdir / "media"
    outdir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    frame = score_stream(args.stream)

    # Render visuals emphasising contrast for the demo
    timeseries_path = media_dir / "seaowl_timeseries.png"
    track_path = media_dir / "seaowl_track.png"
    hybrid_values_path = media_dir / "hybrid_values.png"
    hybrid_scores_path = media_dir / "hybrid_scores.png"

    render_timeseries(frame, timeseries_path)
    render_track(frame, track_path)
    render_hybrid_values(frame, hybrid_values_path)
    render_hybrid_scores(frame, hybrid_scores_path)

    artifacts = {
        "seaowl_timeseries": ArtifactRef(
            label="SeaOWL time-series",
            url="/agent-brief/media/seaowl_timeseries.png",
            asset_path=timeseries_path,
            kind="plot",
        ),
        "seaowl_track": ArtifactRef(
            label="Track heatmap",
            url="/agent-brief/media/seaowl_track.png",
            asset_path=track_path,
            kind="map",
        ),
        "hybrid_values": ArtifactRef(
            label="Hybrid alert values",
            url="/agent-brief/media/hybrid_values.png",
            asset_path=hybrid_values_path,
            kind="plot",
        ),
        "hybrid_scores": ArtifactRef(
            label="Hybrid alert z-scores",
            url="/agent-brief/media/hybrid_scores.png",
            asset_path=hybrid_scores_path,
            kind="plot",
        ),
    }

    generated_at = (
        datetime.fromisoformat(args.timestamp)
        if args.timestamp
        else datetime.now(timezone.utc)
    )

    brief = build_agent_brief(
        scenario_id=args.scenario,
        stream_path=args.stream,
        frame=frame,
        artifacts=artifacts,
        hero_artifact_key="seaowl_timeseries",
        generated_at=generated_at,
    )

    json_path = outdir / "latest.json"
    md_path = outdir / "latest.md"
    scenario_json_path = outdir / f"{args.scenario}.json"

    payload = brief.model_dump(mode="json")
    json_path.write_text(json.dumps(payload, indent=2))
    scenario_json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(brief_to_markdown(brief))

    if not args.quiet:
        print(f"Generated {json_path.relative_to(ROOT)}")
        print(f"Generated {md_path.relative_to(ROOT)}")


def render_timeseries(frame: pd.DataFrame, path: Path) -> None:
    ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    oil = pd.to_numeric(frame.get("oil_fluor_ppb"), errors="coerce")
    baseline = pd.to_numeric(frame.get("oil_baseline"), errors="coerce")
    warn = frame.get("oil_warn", pd.Series([False] * len(frame))).astype(bool)
    alarm = frame.get("oil_alarm", pd.Series([False] * len(frame))).astype(bool)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(ts, oil, color="#38bdf8", linewidth=2.2, label="oil fluorescence")

    if not baseline.isna().all():
        ax.plot(ts, baseline, color="#f97316", linewidth=1.2, alpha=0.85, label="adaptive baseline")

    if warn.any():
        ax.scatter(ts[warn], oil[warn], color="#facc15", s=18, label="warn")
    if alarm.any():
        ax.scatter(ts[alarm], oil[alarm], color="#f87171", s=22, label="alarm", zorder=4)

    event_mask = _event_mask(frame)
    if event_mask.any():
        ymin, ymax = ax.get_ylim()
        ax.fill_between(ts, ymin, ymax, where=event_mask, color="#0ea5e9", alpha=0.12, linewidth=0)
        ax.set_ylim(ymin, ymax)

    ax.set_title("SeaOWL Oil Fluorescence")
    ax.set_ylabel("ppb")
    ax.grid(color="#1e293b", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_track(frame: pd.DataFrame, path: Path) -> None:
    lat = pd.to_numeric(frame.get("lat"), errors="coerce")
    lon = pd.to_numeric(frame.get("lon"), errors="coerce")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    valid = ~(lat.isna() | lon.isna())
    lat = lat[valid]
    lon = lon[valid]

    if lat.empty:
        ax.text(0.5, 0.5, "No track data", ha="center", va="center", color="#94a3b8")
    else:
        ax.plot(lon, lat, color="#38bdf8", linewidth=2.0, alpha=0.85)
        sc = ax.scatter(lon, lat, c=np.linspace(0, 1, len(lat)), cmap="viridis", s=18)
        fig.colorbar(sc, ax=ax, label="Track progression")
        event_mask = _event_mask(frame.loc[valid])
        event_lat = lat[event_mask]
        event_lon = lon[event_mask]
        if not event_lat.empty:
            ax.scatter(event_lon, event_lat, color="#f87171", s=26, label="anomaly window")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="lower right")

    ax.set_title("Track Trace (last 60 min)")
    ax.grid(color="#1e293b", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_hybrid_values(frame: pd.DataFrame, path: Path) -> None:
    ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    oil = pd.to_numeric(frame.get("oil_fluor_ppb"), errors="coerce")
    baseline = pd.to_numeric(frame.get("oil_baseline"), errors="coerce")
    warn = frame.get("oil_warn", pd.Series([False] * len(frame))).astype(bool)
    alarm = frame.get("oil_alarm", pd.Series([False] * len(frame))).astype(bool)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(ts, oil, color="#38bdf8", linewidth=2.0, label="oil")
    if not baseline.isna().all():
        ax.plot(ts, baseline, color="#f97316", linewidth=1.0, alpha=0.8, label="baseline")
    if warn.any():
        ax.scatter(ts[warn], oil[warn], color="#facc15", s=16, label="warn")
    if alarm.any():
        ax.scatter(ts[alarm], oil[alarm], color="#f87171", s=20, label="alarm")
    ax.set_title("Hybrid Alert — Oil Fluorescence vs Baseline")
    ax.set_ylabel("ppb")
    ax.grid(color="#1e293b", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_hybrid_scores(frame: pd.DataFrame, path: Path) -> None:
    ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    zeff = pd.to_numeric(frame.get("oil_z_eff"), errors="coerce")
    z_warn = frame.attrs.get("z_warn", 10.0)
    z_alarm = frame.attrs.get("z_alarm", 25.0)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(ts, zeff, color="#c4b5fd", linewidth=1.8, label="z_eff")
    ax.axhline(z_warn, color="#facc15", linestyle="--", linewidth=1.2, label="warn threshold")
    ax.axhline(z_alarm, color="#f87171", linestyle="--", linewidth=1.2, label="alarm threshold")
    ax.set_title("Hybrid Alert — Effective Z-Score")
    ax.set_ylabel("z-score")
    ax.grid(color="#1e293b", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _event_mask(frame: pd.DataFrame) -> pd.Series:
    warn = frame.get("oil_warn", pd.Series([False] * len(frame))).astype(bool)
    alarm = frame.get("oil_alarm", pd.Series([False] * len(frame))).astype(bool)
    phase = pd.Series(frame.get("event_phase", 0.0))
    return warn | alarm | (pd.to_numeric(phase, errors="coerce") > 0.05)


if __name__ == "__main__":
    main()
