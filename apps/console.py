"""Unified Streamlit console with Telemetry + Incidents views.

Features
- Telemetry view: tails live NDJSON (1 Hz), renders rolling charts & map,
  computes warn/alarm (HybridOilAlertScorer), optionally auto-runs the
  incident pipeline+agent on alarm episodes, and lists detected incidents.
- Incidents view: lists incidents and renders details from saved artifacts
  (synopsis, Cerulean polygons, brief download, trace tail).

This consolidates the two workflows into a single app with a sidebar
view switch, matching a “two tabs” UX in one Streamlit app.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anomaly.hybrid import HybridOilAlertScorer
from anomaly import PipelineConfig, generate_transitions_from_ndjson
from agent import GPTAgentModel, RuleBasedAgentModel
from ingest.ship_stream import normalize_record


DATA_PATH = Path("data/ship/seaowl_live.ndjson")
ARTIFACTS_ROOT = Path("artifacts")
REFRESH_MS = 1000
BUFFER_SECONDS = 30 * 60


@dataclass
class SimConfig:
    duration: int = 3600
    sample_rate: float = 1.0
    event_start: int = 600
    event_duration: int = 300
    event_magnitude: float = 2.5


def _init_state() -> None:
    st.session_state.setdefault("tail_pos", 0)
    st.session_state.setdefault("buffer_df", pd.DataFrame())
    st.session_state.setdefault("sim_process", None)
    st.session_state.setdefault("selected_incident_id", None)
    st.session_state.setdefault("last_live_run_ts", 0.0)
    # Always start on telemetry view, only switch to incident when one is selected


def _start_simulator(cfg: SimConfig, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "sim_seaowl.py"),
        "--duration",
        str(cfg.duration),
        "--sample-rate",
        str(cfg.sample_rate),
        "--output",
        str(output),
        "--event-start",
        str(cfg.event_start),
        "--event-duration",
        str(cfg.event_duration),
        "--event-magnitude",
        str(cfg.event_magnitude),
        "--sleep",
    ]
    proc = subprocess.Popen(cmd)  # noqa: S603
    st.session_state["sim_process"] = proc
    return proc.pid


def _stop_simulator() -> None:
    proc = st.session_state.get("sim_process")
    if proc and isinstance(proc, subprocess.Popen):
        try:
            proc.terminate()
        except Exception:
            pass
    st.session_state["sim_process"] = None


def _tail_ndjson(path: Path, max_rows: int) -> pd.DataFrame:
    buf: pd.DataFrame = st.session_state.get("buffer_df")
    tail_pos: int = int(st.session_state.get("tail_pos", 0))

    if not path.exists():
        return buf

    size = path.stat().st_size
    if size < tail_pos:
        tail_pos = 0
        buf = pd.DataFrame()

    with path.open("r", encoding="utf-8") as fh:
        fh.seek(tail_pos)
        lines = fh.readlines()
        tail_pos = fh.tell()

    if lines:
        rows = []
        for ln in lines:
            if ln.strip():
                try:
                    rows.append(normalize_record(json.loads(ln)))
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
        if rows:
            df_new = pd.DataFrame(rows)
            df_new["ts"] = pd.to_datetime(df_new["ts"], utc=True, errors="coerce")
            buf = pd.concat([buf, df_new], ignore_index=True)
            buf = buf.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if len(buf) > max_rows:
        buf = buf.iloc[-max_rows:].reset_index(drop=True)

    st.session_state["buffer_df"] = buf
    st.session_state["tail_pos"] = tail_pos
    return buf


def _score_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    scorer = HybridOilAlertScorer()
    try:
        scored = scorer.score_dataframe(df)
    except Exception:
        return df
    return scored


def _auto_run_pipeline(live_path: Path, use_gpt: bool) -> None:
    import time as _t

    now = _t.time()
    if now - float(st.session_state.get("last_live_run_ts", 0.0)) < 3.0:
        return
    if not live_path.exists():
        return
    cfg = PipelineConfig(
        flush_after_s=1800,
        agent_enabled=True,
        agent_model=GPTAgentModel() if use_gpt else RuleBasedAgentModel(),
    )
    try:
        generate_transitions_from_ndjson(live_path, config=cfg)
        st.session_state["last_live_run_ts"] = now
    except Exception:  # noqa: BLE001
        # Silently skip pipeline errors (usually from malformed JSON data)
        # Users can manually run pipeline if needed
        pass


def _list_incidents(artifacts_root: Path) -> List[Dict]:
    incidents: List[Dict] = []
    if not artifacts_root.exists():
        return incidents
    for d in sorted(artifacts_root.iterdir()):
        if not d.is_dir():
            continue
        syn = d / "incident_synopsis.json"
        if syn.exists():
            try:
                payload = json.loads(syn.read_text())
                incidents.append(
                    {
                        "incident_id": d.name,
                        "scenario": payload.get("scenario"),
                        "confidence": payload.get("confidence"),
                        "summary": payload.get("summary"),
                        "ts_peak": payload.get("event", {}).get("ts_peak"),
                        "path": d,
                    }
                )
            except Exception:
                continue
    incidents.sort(key=lambda r: r.get("ts_peak") or "", reverse=True)
    return incidents


def _render_timeseries(df: pd.DataFrame) -> None:
    st.subheader("Telemetry (rolling)")
    if df.empty:
        st.info("Waiting for live telemetry…")
        return

    # Define metrics with display names and units
    metrics = [
        ("oil_fluor_ppb", "Oil Fluorescence", "ppb"),
        ("chlorophyll_ug_per_l", "Chlorophyll", "μg/L"),
        ("backscatter_m-1_sr-1", "Backscatter", "m⁻¹sr⁻¹")
    ]

    # Create separate charts for each metric
    for col, display_name, unit in metrics:
        if col in df.columns:
            metric_df = df[["ts", col]].set_index("ts")
            st.caption(f"{display_name} ({unit})")
            st.line_chart(metric_df[[col]], height=180)

    # Show alarms/warnings at the bottom
    if "oil_alarm" in df.columns and df["oil_alarm"].iloc[-1]:
        st.error("ALARM: oil channel elevated")
    elif "oil_warn" in df.columns and df["oil_warn"].iloc[-1]:
        st.warning("WARN: oil channel elevated")


def _render_track(df: pd.DataFrame) -> None:
    st.subheader("Ship Track (recent)")
    if df.empty or ("lat" not in df.columns) or ("lon" not in df.columns):
        st.info("No geolocated samples yet…")
        return
    mdf = df[["lat", "lon"]].dropna()
    if mdf.empty:
        st.info("No geolocated samples yet…")
        return
    layers = [
        pdk.Layer(
            "TileLayer",
            data="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
        )
    ]
    layers.append(
        pdk.Layer(
            "PathLayer",
            data=[{"path": mdf[["lon", "lat"]].values.tolist(), "name": "track"}],
            get_path="path",
            width_scale=1,
            width_min_pixels=2,
            get_color=[66, 135, 245],
        )
    )
    last = mdf.iloc[-1]
    view_state = pdk.ViewState(latitude=float(last["lat"]), longitude=float(last["lon"]), zoom=11)
    deck = pdk.Deck(map_style=None, initial_view_state=view_state, layers=layers)
    st.pydeck_chart(deck)


def _render_incident_list(incidents: List[Dict]) -> None:
    st.subheader("Incidents")
    if not incidents:
        st.info("No incidents yet. When an alarm occurs, the agent will generate one.")
        return
    for item in incidents[:50]:
        cols = st.columns([2, 1, 5, 1])
        with cols[0]:
            st.write(item.get("ts_peak", ""))
        with cols[1]:
            conf = item.get("confidence")
            st.write(f"{conf:.2f}" if isinstance(conf, (float, int)) else "-")
        with cols[2]:
            st.write(item.get("summary", ""))
        with cols[3]:
            if st.button("Open", key=f"open_{item['incident_id']}"):
                st.session_state["selected_incident_id"] = item["incident_id"]
                st.rerun()


def _render_incident_detail(incident_id: Optional[str]) -> None:
    st.subheader("Incident Details")
    if not incident_id:
        st.info("Select an incident from the Telemetry view or list below.")
        return
    d = ARTIFACTS_ROOT / incident_id
    syn = d / "incident_synopsis.json"
    brief = d / "incident_brief.json"
    geo = d / "cerulean.geojson"
    trace = d / "agent_trace.jsonl"
    if not syn.exists():
        st.error(f"Synopsis not found for {incident_id}")
        return
    payload = json.loads(syn.read_text())
    st.write(f"Scenario: {payload.get('scenario')}  ·  Confidence: {payload.get('confidence')}")
    st.write(payload.get("summary", ""))
    if brief.exists():
        st.download_button(
            "Download JSON Brief",
            data=brief.read_bytes(),
            file_name=brief.name,
            mime="application/json",
        )
    if geo.exists():
        try:
            gj = json.loads(geo.read_text())
            layers = [
                pdk.Layer(
                    "TileLayer",
                    data="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                    min_zoom=0,
                    max_zoom=19,
                    tile_size=256,
                ),
                pdk.Layer(
                    "GeoJsonLayer",
                    data=gj,
                    stroked=True,
                    filled=True,
                    get_fill_color="[0, 121, 191, 70]",
                    get_line_color="[0, 121, 191, 220]",
                    line_width_min_pixels=2,
                ),
            ]
            # Compute a view from features if possible
            view_state = pdk.ViewState(latitude=0, longitude=0, zoom=3)
            deck = pdk.Deck(map_style=None, initial_view_state=view_state, layers=layers)
            st.pydeck_chart(deck)
        except Exception:
            st.caption("GeoJSON present but failed to render; see artifact file.")
    if trace.exists():
        lines = trace.read_text().strip().splitlines()[-20:]
        st.caption("Agent trace (tail)")
        st.code("\n".join(lines), language="json")


def _render_sidebar() -> tuple[Path, bool]:
    # No view selector - navigation is handled by incident selection

    st.sidebar.header("Live Source")
    live_path = Path(st.sidebar.text_input("NDJSON path", value=str(DATA_PATH)))

    st.sidebar.header("Simulator (optional)")
    duration = st.sidebar.number_input("Duration (s)", value=3600, step=60)
    sample_rate = st.sidebar.number_input("Sample rate (Hz)", value=1.0, step=0.5, format="%.1f")
    event_start = st.sidebar.number_input("Event start (s)", value=600, step=30)
    event_duration = st.sidebar.number_input("Event duration (s)", value=300, step=30)
    event_mag = st.sidebar.number_input("Event magnitude (x)", value=2.5, step=0.1)
    cols = st.sidebar.columns(2)
    with cols[0]:
        if st.button("Start sim", use_container_width=True):
            _stop_simulator()
            _start_simulator(
                SimConfig(
                    duration=int(duration),
                    sample_rate=float(sample_rate),
                    event_start=int(event_start),
                    event_duration=int(event_duration),
                    event_magnitude=float(event_mag),
                ),
                live_path,
            )
    with cols[1]:
        if st.button("Stop sim", use_container_width=True):
            _stop_simulator()

    use_gpt = True
    if not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning("OPENAI_API_KEY missing; agent runs will fail.")

    st.sidebar.checkbox("Auto refresh", value=True, key="auto_refresh")
    if st.sidebar.button("Manual Refresh", use_container_width=True):
        st.rerun()
    st.sidebar.checkbox("Auto-run agent on alarm", value=True, key="auto_run_agent")
    return live_path, use_gpt


def main() -> None:
    st.set_page_config(page_title="Neptune Console", layout="wide")
    st.title("🌊 Neptune Console")
    _init_state()

    live_path, use_gpt = _render_sidebar()

    # Auto-refresh mechanism for live telemetry (only when not viewing incidents)
    if not st.session_state.get("selected_incident_id") and st.session_state.get("auto_refresh", True):
        # Use streamlit-autorefresh for smooth real-time updates (1 second interval)
        refresh_count = st_autorefresh(interval=1000, limit=None, key="datarefresh")

        # Show refresh indicator in sidebar
        if refresh_count > 0:
            st.sidebar.caption(f"🔄 Updates: {refresh_count}")

    # Show incident detail if one is selected, otherwise show telemetry
    if st.session_state.get("selected_incident_id"):
        # Incident view with back button
        if st.button("← Back to Live Telemetry", type="primary"):
            st.session_state["selected_incident_id"] = None
            st.rerun()

        _render_incident_detail(st.session_state["selected_incident_id"])

    else:
        # Telemetry view (default)
        max_rows = int(BUFFER_SECONDS)
        df = _tail_ndjson(live_path, max_rows=max_rows)
        if not df.empty:
            df = _score_frame(df)

        col1, col2 = st.columns((2, 1))
        with col1:
            _render_timeseries(df)
        with col2:
            _render_track(df)

        if st.session_state.get("auto_run_agent", True):
            # Trigger agent when in warn/alarm or periodically
            try:
                if ("oil_alarm" in df.columns and bool(df["oil_alarm"].iloc[-1])) or (
                    "oil_warn" in df.columns and bool(df["oil_warn"].iloc[-1])
                ):
                    _auto_run_pipeline(live_path, use_gpt)
            except Exception:
                pass

        incidents = _list_incidents(ARTIFACTS_ROOT)
        _render_incident_list(incidents)


if __name__ == "__main__":
    main()

