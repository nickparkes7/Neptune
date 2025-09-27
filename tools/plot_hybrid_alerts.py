"""Plot Hybrid Oil Alert outputs for the SeaOWL stream.

Generates two figures:
 - hybrid_values.png: oil values with adaptive baseline and warn/alarm markers
 - hybrid_scores.png: effective z-score vs thresholds
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anomaly.hybrid import HybridOilAlertConfig, HybridOilAlertScorer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="NDJSON input stream")
    p.add_argument("--output", type=Path, required=True, help="Directory for plots")
    # Optional overrides
    p.add_argument("--z-warn", type=float, default=None)
    p.add_argument("--z-alarm", type=float, default=None)
    p.add_argument("--abs-warn", type=float, default=None)
    p.add_argument("--abs-alarm", type=float, default=None)
    return p


def ensure_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    return frame.sort_values("ts").reset_index(drop=True)


def plot_values(df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(df["ts"], df["oil_fluor_ppb"], color="tab:blue", label="oil")
    ax.plot(df["ts"], df["oil_baseline"], color="tab:orange", label="baseline", linewidth=1.2)

    warn_pts = df.loc[df["oil_warn"], ["ts", "oil_fluor_ppb"]]
    alarm_pts = df.loc[df["oil_alarm"], ["ts", "oil_fluor_ppb"]]
    if not warn_pts.empty:
        ax.scatter(warn_pts["ts"], warn_pts["oil_fluor_ppb"], color="#e6b800", s=14, label="warn")
    if not alarm_pts.empty:
        ax.scatter(alarm_pts["ts"], alarm_pts["oil_fluor_ppb"], color="#d62728", s=16, label="alarm")

    ax.set_ylabel("oil_fluor_ppb")
    ax.set_xlabel("timestamp (UTC)")
    ax.set_title("Hybrid oil alert: values + baseline")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    path = outdir / "hybrid_values.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_scores(df: pd.DataFrame, cfg: HybridOilAlertConfig, outdir: Path) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(df["ts"], df["oil_z_eff"], color="tab:purple", label="z_eff")
    ax.axhline(cfg.z_warn, color="#e6b800", linestyle="--", label="z_warn")
    ax.axhline(cfg.z_alarm, color="#d62728", linestyle="--", label="z_alarm")
    ax.set_ylabel("z")
    ax.set_xlabel("timestamp (UTC)")
    ax.set_title("Hybrid oil alert: z-scores")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    path = outdir / "hybrid_scores.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    args = build_parser().parse_args()
    outdir: Path = args.output
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = HybridOilAlertConfig()
    if args.z_warn is not None:
        cfg.z_warn = args.z_warn
    if args.z_alarm is not None:
        cfg.z_alarm = args.z_alarm
    if args.abs_warn is not None:
        cfg.abs_warn = args.abs_warn
    if args.abs_alarm is not None:
        cfg.abs_alarm = args.abs_alarm

    scorer = HybridOilAlertScorer(config=cfg)
    df = scorer.score_ndjson(args.input)
    df = ensure_datetime(df)

    vp = plot_values(df, outdir)
    sp = plot_scores(df, cfg, outdir)

    (outdir / "README.txt").write_text(
        "\n".join(
            [
                f"Hybrid plots generated from {args.input.name}",
                f"z_warn={cfg.z_warn}, z_alarm={cfg.z_alarm}, abs_warn={cfg.abs_warn}, abs_alarm={cfg.abs_alarm}",
                f"Values: {vp.name}",
                f"Scores: {sp.name}",
            ]
        )
    )


if __name__ == "__main__":
    main()

