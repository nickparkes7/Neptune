#!/usr/bin/env python3
"""Plot synthetic SeaOWL stream channels and save to PNG."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_stream(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def plot_timeseries(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(df["ts"], df["oil_fluor_ppb"], label="oil_fluor_ppb", color="#d62728")
    axes[0].set_ylabel("Oil (ppb)")

    axes[1].plot(df["ts"], df["chlorophyll_ug_per_l"], label="chlorophyll", color="#2ca02c")
    axes[1].set_ylabel("Chl (µg/L)")

    axes[2].plot(df["ts"], df["backscatter_m-1_sr-1"], label="backscatter", color="#1f77b4")
    axes[2].set_ylabel("Backscatter")

    axes[3].plot(df["ts"], df["temperature_c"], label="temperature", color="#ff7f0e")
    axes[3].set_ylabel("Temp (°C)")
    axes[3].set_xlabel("Time (UTC)")

    event_mask = df.get("event_phase", 0) > 0.05
    if event_mask.any():
        for ax in axes:
            ymin, ymax = ax.get_ylim()
            ax.fill_between(
                df["ts"],
                ymin,
                ymax,
                where=event_mask,
                color="gray",
                alpha=0.2,
                label="event" if ax is axes[0] else None,
            )
            ax.set_ylim(ymin, ymax)

    axes[0].legend(loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_track(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df["lon"], df["lat"], color="#1f77b4")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Ship Track")
    event_mask = df.get("event_phase", 0) > 0.05
    if event_mask.any():
        ax.scatter(df.loc[event_mask, "lon"], df.loc[event_mask, "lat"], color="#d62728", label="event", s=12)
        ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SeaOWL synthetic stream")
    parser.add_argument("ndjson", type=Path, help="NDJSON file from simulator")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/seaowl"), help="Directory for PNG outputs")
    args = parser.parse_args()

    df = load_stream(args.ndjson)
    ts_path = args.outdir / "seaowl_timeseries.png"
    track_path = args.outdir / "seaowl_track.png"
    plot_timeseries(df, ts_path)
    plot_track(df, track_path)
    print(f"Wrote {ts_path} and {track_path}")


if __name__ == "__main__":
    main()
