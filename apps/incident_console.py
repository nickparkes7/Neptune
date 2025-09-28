"""Streamlit dashboard for the Neptune Phase 1 demo."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import pydeck as pdk
import streamlit as st

from anomaly import PipelineConfig, generate_transitions_from_ndjson
from anomaly.events import SuspectedSpillEvent
from agent import AgentRunResult, GPTAgentModel, RuleBasedAgentModel
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from tools.load_env import load_env

load_env()

st.set_page_config(page_title="Neptune Incident Console", layout="wide")
st.title("🛰️ Neptune Incident Console")

DATA_DIR = Path("data/ship")
ARTIFACT_ROOT = Path("artifacts")
TRACE_LIMIT = 15


@st.cache_data(show_spinner=False)
def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, orient="records", lines=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> dict:
    return json.loads(path.read_text())


def run_pipeline(
    input_path: Path,
    use_gpt: bool,
    logger: Optional[object] = None,
) -> tuple[pd.DataFrame, AgentRunResult, SuspectedSpillEvent]:
    if logger:
        logger.write("Loading SeaOWL telemetry…")
    df = load_timeseries(input_path)
    if logger:
        logger.write(f"Loaded {len(df):,} samples from {input_path.name}")

    if logger:
        logger.write("Executing incident pipeline…")
    config = PipelineConfig(
        flush_after_s=1800,
        agent_enabled=True,
        agent_model=GPTAgentModel() if use_gpt else RuleBasedAgentModel(),
    )
    result = generate_transitions_from_ndjson(input_path, config=config)
    if logger:
        logger.write(f"Generated {len(result.transitions)} transitions")
    if not result.agent_runs:
        raise RuntimeError("Agent did not produce output. Check pipeline configuration.")
    agent_run = result.agent_runs[-1]
    if logger:
        scenario = agent_run.synopsis.scenario.replace("_", " ")
        logger.write(f"Agent scenario: {scenario} (confidence {agent_run.synopsis.confidence:.2f})")

    incident: Optional[SuspectedSpillEvent] = None
    for transition in result.transitions:
        if transition.allow_tasking:
            incident = transition.incident
            break
    if incident is None:
        incident = result.transitions[-1].incident if result.transitions else agent_run.synopsis
    return df, agent_run, incident


def render_timeseries(df: pd.DataFrame) -> None:
    st.subheader("SeaOWL Telemetry")
    metrics = [col for col in ["oil_fluor_ppb", "chlorophyll_ug_per_l", "backscatter_m-1_sr-1"] if col in df.columns]
    if not metrics:
        st.warning("No measurements found in the selected file.")
        return
    chart_df = df.set_index("ts")[metrics]
    st.line_chart(chart_df, height=260)


def render_incident_summary(agent_run: AgentRunResult, incident: SuspectedSpillEvent) -> None:
    synopsis = agent_run.synopsis
    brief_path = Path(synopsis.artifacts.get("incident_brief", ""))
    st.subheader("Incident Summary")
    st.metric("Scenario", synopsis.scenario.replace("_", " ").title())
    st.metric("Confidence", f"{synopsis.confidence * 100:.1f}%")
    st.write(synopsis.summary)
    st.write("**Recommended Actions**")
    for action in synopsis.recommended_actions:
        st.write(f"- {action}")
    if synopsis.followup_scheduled:
        eta = synopsis.followup_eta.isoformat().replace("+00:00", "Z") if synopsis.followup_eta else "next Cerulean update"
        st.info(f"Follow-up scheduled: {eta}")
    if brief_path.exists():
        st.download_button(
            "Download JSON Brief",
            data=brief_path.read_bytes(),
            file_name=brief_path.name,
            mime="application/json",
        )


def render_map(agent_run: AgentRunResult, incident: SuspectedSpillEvent) -> None:
    st.subheader("Incident Map")
    layers = []
    # Vessel location
    point_df = pd.DataFrame(
        {
            "lat": [incident.lat],
            "lon": [incident.lon],
            "label": [incident.event_id or "incident"],
        }
    )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=point_df,
            get_position="[lon, lat]",
            get_color="[255, 100, 50]",
            get_radius=500,
            pickable=True,
        )
    )
    # Cerulean polygons
    geojson_path = agent_run.synopsis.artifacts.get("cerulean.geojson")
    if geojson_path and Path(geojson_path).exists():
        geojson = load_geojson(Path(geojson_path))
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=geojson,
                stroked=True,
                filled=True,
                get_fill_color="[0, 121, 191, 40]",
                get_line_color="[0, 121, 191]",
                line_width_min_pixels=2,
            )
        )
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state=pdk.ViewState(latitude=incident.lat, longitude=incident.lon, zoom=8),
        layers=layers,
        tooltip={"text": "{label}"},
    )
    st.pydeck_chart(deck)


def render_agent_details(agent_run: AgentRunResult) -> None:
    st.subheader("Agent Plan & Trace")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Cerulean plan")
        st.json(agent_run.plan.model_dump(mode="json"))
    with col2:
        st.caption("Synopsis metrics")
        st.json(agent_run.synopsis.metrics.model_dump(mode="json"))

    trace_path = agent_run.trace_path
    if trace_path.exists():
        trace_lines = trace_path.read_text().strip().splitlines()[-TRACE_LIMIT:]
        st.caption("Agent trace (tail)")
        st.code("\n".join(trace_lines), language="json")


def main() -> None:
    st.sidebar.header("Configuration")
    files = sorted(DATA_DIR.glob("*.ndjson"))
    if not files:
        st.sidebar.error("No SeaOWL NDJSON files found under data/ship.")
        return

    selected_name = st.sidebar.selectbox("SeaOWL stream", [f.name for f in files])
    selected_file = DATA_DIR / selected_name

    use_gpt_default = bool(os.getenv("OPENAI_API_KEY"))
    use_gpt = st.sidebar.checkbox("Use GPT-5 agent", value=use_gpt_default)
    if use_gpt and not use_gpt_default:
        st.sidebar.warning("OPENAI_API_KEY missing in environment. Falling back to rule-based agent.")
        use_gpt = False

    rerun_required = st.session_state.get("selected_file") != selected_file
    if rerun_required:
        st.session_state.pop("timeseries_df", None)
        st.session_state.pop("agent_run", None)
        st.session_state.pop("incident", None)
    if st.sidebar.button("Run pipeline", type="primary") or rerun_required:
        with st.status("Running incident pipeline…", expanded=True) as status_box:
            try:
                df, agent_run, incident = run_pipeline(selected_file, use_gpt, logger=status_box)
                st.session_state["timeseries_df"] = df
                st.session_state["agent_run"] = agent_run
                st.session_state["incident"] = incident
                st.session_state["selected_file"] = selected_file
                status_box.update(label="Pipeline complete ✅", state="complete")
            except Exception as exc:  # noqa: BLE001
                status_box.update(label="Pipeline failed", state="error")
                status_box.write(str(exc))
                return

    if "agent_run" not in st.session_state:
        st.info("Select a stream and run the pipeline to generate an incident view.")
        return

    df = st.session_state["timeseries_df"]
    agent_run = st.session_state["agent_run"]
    incident = st.session_state["incident"]

    ts_col, summary_col = st.columns((2, 1))
    with ts_col:
        render_timeseries(df)
    with summary_col:
        render_incident_summary(agent_run, incident)

    render_map(agent_run, incident)
    render_agent_details(agent_run)


if __name__ == "__main__":
    main()
