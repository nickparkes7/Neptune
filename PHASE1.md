# Phase 1 MVP Execution Plan

- Executes the Phase 1 scope in `docs/DEVELOPMENT_STAGES.md`.
- Focus: continuous SeaOWL monitoring, on-demand Sentinel‑1 pulls, fast SAR slick detection, agent brief, Streamlit UI.

<!-- STATUS:PHASE1:BEGIN -->

Progress: 4/12 steps done · 0 in progress · 0 blocked

| Step | Status | Owner | Notes |
| --- | --- | --- | --- |
| 1_bootstrap | done | nicholas | uv environment + enforcement committed |
| 2_simulator | done | nicholas | simulator emits NDJSON; parquet batches at data/ship/parquet |
| 3_anomaly | done | nicholas | Hybrid oil alert implemented; plots in artifacts/hybrid; test=tests/test_hybrid_scorer.py |
| 4_events | done | nicholas | End-to-end anomaly trigger: schema (src/anomaly/events.py), incident manager + pipeline (src/anomaly/incidents.py, src/anomaly/pipeline.py), CLI tools/run_event_trigger.py + tools/run_incident_pipeline.py; tests=test_event_trigger.py,test_incident_manager.py,test_pipeline.py |
| 5_tasker | pending |  |  |
| 6_detector | pending |  |  |
| 7_linking | pending |  |  |
| 8_agent | pending |  |  |
| 9_brief | pending |  |  |
| 10_streamlit | pending |  |  |
| 11_demo | pending |  |  |
| 12_qa | pending |  |  |

<!-- STATUS:PHASE1:END -->

## 12‑Hour War‑Room Plan (Timeboxed)

- This compresses the MVP to essentials. We keep the SeaOWL feed synthetic but use real web-fetched Sentinel‑1 scenes while shipping a deterministic demo path.
- Key cuts: no fvdb integration, PDF optional. Real S‑1 via API, synthetic onboard stream, local JSON/PNG artifacts.

| Time (hrs) | Task | Output |
| --- | --- | --- |
| 0.0–0.5 | Env bootstrap, repo scaffolding, config stubs | venv ready, folders, configs |
| 0.5–2.0 | SeaOWL simulator + anomaly scorer | NDJSON stream, alert firing on inject |
| 2.0–3.0 | Event schema + trigger wiring | `SuspectedSpillEvent` emitted with AOI |
| 3.0–4.5 | Satellite tasker (real S‑1 fetch) | 1–2 Sentinel‑1 scenes downloaded & indexed |
| 4.5–5.5 | Slick detector (dark-spot + morphology) | Polygons + scores over real rasters |
| 5.5–6.5 | Temporal linking + drift estimate (2 passes) | Track + drift vector |
| 6.5–8.5 | Streamlit MVP (map, timeseries, incident pane) | Interactive demo UI |
| 8.5–9.5 | Agent orchestration (rule-based) | Text summary + 3 waypoints |
| 9.5–11.0 | Brief export (JSON + PNG snapshots; PDF optional) | `artifacts/event_id/*.json, *.png` (PDF if time) |
| 11.0–12.0 | E2E demo script, tuning, dry run, backups | `tools/run_demo.py`, stable narrative |

## Scope Adjustments (12h Reality)

- Must‑have: SeaOWL sim + anomaly, event wiring, real Sentinel‑1 fetch, slick polygons, Streamlit visualization, JSON brief.
- Nice‑to‑have if time: PDF brief, drift vectors, shoreline mask, confidence score, agent chat UI polish.
- Deferred: fvdb cube/ops, VIIRS/S2 integrations, advanced texture models.

## Outcomes

- Detect oil-like anomalies from SeaOWL at 1 Hz, trigger a real Sentinel‑1 fetch via web API, outline slick polygons, summarize origin/propagation, generate a JSON brief (+PNG snapshots; PDF optional), and visualize in Streamlit.

## Definition of Done

- Live/simulated SeaOWL stream triggers an alert.
- Sentinel‑1 stack downloaded for the AOI/time window (real rasters).
- Slick polygons detected and linked across passes, with drift vector.
- Streamlit shows timeseries, map overlays, and incident pane.
- Agent tools produce a coherent “incident brief” (JSON + PNG; PDF if time permits).

## Repo Setup

- Python 3.10+ (or 3.11) managed with `uv` (https://github.com/astral-sh/uv).
- Folder layout:
  - `src/anomaly/`, `src/satellite/`, `src/agent/`, `src/report/`, `apps/streamlit/`, `tools/`, `configs/`, `data/ship/`, `data/s1/`
- Dependencies: `numpy`, `pandas`, `pyarrow`, `rasterio`, `rio-cogeo` (optional), `shapely`, `geopandas`, `scikit-image`, `opencv-python`, `matplotlib`, `pydeck` or `folium`, `streamlit`, `reportlab` or `weasyprint`, `pyyaml`, `pydantic`, `asf-search` (or `sentinelsat`)

## Step-by-Step Plan

1) Bootstrap Environment (0.0–0.5h)
- Tasks
  - Initialize `pyproject.toml` (optional) and `requirements.txt` or `requirements.lock` for `uv`.
  - Use `uv` to manage virtual env and installs.
  - Add `make setup` (runs `uv sync`), `make run-app`, and `make demo` in `Makefile`.
- Acceptance
  - `uv venv` creates `.venv` (or rely on `uv` implicit env).
  - `uv pip install -r requirements.txt` completes (or `uv sync` if using lockfile).

2) SeaOWL Stream Simulator and Ingestion (0.5–2.0h)
- Files
  - `tools/sim_seaowl.py` – emits NDJSON lines at 1 Hz with fields: `ts, lat, lon, oil_fluor, chlorophyll, backscatter, qc_flags`.
  - `src/ingest/ship_stream.py` – reader that yields records and writes rolling Parquet.
- How
  - Model baseline with AR(1) noise; inject 15–30 min “oil-like” events: oil_fluor +2–3σ; modest turbidity/foam guards using chlorophyll/backscatter.
   - Save to `data/ship/seaowl.ndjson` and periodic Parquet shards.
- Acceptance
   - `uv run tools/sim_seaowl.py --aoi configs/aoi.yaml` runs and writes ~1 Hz.

3) Anomaly Scorer (robust z-score + MAD) (parallel within 0.5–2.0h or 1.5–2.5h)
- Files
  - `src/anomaly/scorer.py` with class `SeaOWLAnomaly(window_s, z_thresh, hold_s)`.
  - `tests/test_anomaly.py` for synthetic sequences (alert/no-alert).
- How
  - Maintain rolling median/MAD for `oil_fluor`; compute robust z-score.
  - Cross-check: suppress if chlorophyll spikes similarly and backscatter unchanged, to cut bio/foam FPs.
  - Alert when score > τ for N seconds; emit `SuspectedSpillEvent`.
- Acceptance
   - Deterministic alerts on injected events in simulator playback (`uv run tests/test_anomaly.py`).

4) Event Schema + Trigger Wire-up (2.0–3.0h)
- Files
  - `src/anomaly/events.py` – `SuspectedSpillEvent` Pydantic schema:
    - `{ event_id, ts_start, ts_peak, lat, lon, duration_s, oil_stats, context_channels, aoi_bbox }`
- How
  - When alert holds, assemble event window and compute a tight `aoi_bbox` (e.g., ±10–20 km).
  - Publish in-process (function call) for MVP; no external broker needed.
- Acceptance
  - Event dict validates; downstream receives the same object.

4b) Incident Lifecycle Manager (debounce + heartbeat)
- Files
  - `src/anomaly/incidents.py` – `IncidentManager` state machine consolidating alert windows into a single active incident and managing lifecycle transitions.
  - `tests/test_incident_manager.py` – scenarios covering continuous slick traversal, material updates, and re-arm after clear gaps.
- How
  - Merge contiguous or nearby alarm windows into one incident; extend duration, update stats, and grow the AOI envelope as the vessel moves.
  - Gate new incident creation on both time-clear (`clear_hold_s`) and spatial separation (`rearm_distance_km`) to avoid thrash when sitting inside a slick.
  - Emit paced transitions (`opened`, `updated`, `heartbeat`, `closed`) and expose cooldown knobs (e.g., tasking cooldown) so downstream components throttle heavy work.
- Acceptance
  - Long-duration alarm segments yield a single incident with periodic heartbeats instead of repeated opens.
  - Incident only closes and re-arms after the configured clear gap and distance are exceeded.
  - CLI `tools/run_incident_pipeline.py` produces lifecycle transitions from an NDJSON stream.
  - Unit tests cover the transition behavior and cooldown settings.

5) Satellite Tasker (Real S‑1 Fetch) (3.0–4.5h)
- Files
  - `src/satellite/tasker.py` – `task_satellite(aoi, time_range, sensors=['S1']) -> list[Scene]`
  - `configs/s1_paths.yaml` – local COG paths or SAFE/GeoTIFFs with timestamps and footprints.
- How
  - Use `asf_search` (preferred) or `sentinelsat` to query & download 1–2 recent Sentinel‑1 IW GRD scenes intersecting the AOI/time window (limit search to small bbox to keep downloads < 1 GB).
  - Extract VV (and VH if available) to cropped AOI windows using `rasterio.windows` and write to `data/s1/processed/<scene_id>_crop.tif` (COG optional for fast reads).
  - Cache scene metadata (ID, acquisition time, relative orbit, polarization) in `configs/s1_catalog.json` so reruns use local data if already present.
- Acceptance
  - Given AOI/time, returns scenes with valid bounds and timestamps from the downloaded catalog; `uv run src/satellite/tasker.py --smoke` downloads/caches when missing and exits cleanly when data already present.

6) Slick Detector (dark-spot + morphology) (4.5–5.5h)
- Files
  - `src/satellite/sar_slick.py` – `detect_slicks(scene) -> GeoDataFrame(polygons, scores)`
- How
  - Read cropped VV (and VH) rasters; convert to dB; apply Lee or median filter to reduce speckle.
  - Dark-spot detect: percentile threshold (e.g., <= p10) combined with local background subtraction.
  - Morphology: open/close; filter by area and elongation; optional quick shoreline mask using Natural Earth coastline if time allows.
  - Texture (optional if time): local variance/GLCM metrics to boost confidence.
  - Score = w1*(darkness) + w2*(low-variance) + w3*(elongation).
- Acceptance
  - Produces polygons with scores; QC plot overlay saved to `artifacts/scene_id_overlay.png`.

7) Temporal Linking and Drift Estimation (5.5–6.5h)
- Files
  - `src/satellite/track.py` – `link_polygons(scenes_polys) -> tracks, drift_vectors`
- How
  - Associate polygons across passes via centroid distance, IoU, and score consistency.
  - Estimate drift vector (bearing/speed) from sequential centroids; compute growth factor (area change).
- Acceptance
  - Outputs track list and mean drift vector; plot saved.

8) Minimal Agent Orchestration + Tools (8.5–9.5h)
- Files
  - `src/agent/tools.py` – function stubs per Phase 1: `get_ship_window`, `task_satellite`, `detect_slicks`, `make_brief`.
  - `src/agent/runner.py` – orchestrates: on event → task satellite → detect slicks → link → summarize.
- How
  - Keep “agent” local and deterministic for MVP; simple rule-based prompt template.
  - Return a structured explanation string with metrics and recommendations (waypoints along drift).
- Acceptance
  - Given an event, returns a summary + 3 waypoint suggestions.

9) Incident Brief: JSON + PNG (PDF optional) (9.5–11.0h)
- Files
  - `src/report/brief.py` – `make_brief(event, tracks, drift, confidence, assets) -> (json_path, png_paths[, pdf_path])`
  - `configs/brief_template.yaml` – organization/contact/logo.
- How
  - Assemble key metrics, map thumbnails, timeseries snapshot, polygons overlay, drift arrow, recommended actions.
  - Produce JSON for downstream systems and save PNG snapshots; generate PDF only if time remains.
- Acceptance
  - Valid JSON and PNGs saved to `artifacts/event_id/` (PDF optional).

10) Streamlit App (6.5–8.5h)
- Files
  - `apps/streamlit/App.py`
- How
  - Panels:
    - Map (pydeck): ship track, anomaly markers; post-trigger SAR polygons and drift arrows.
    - Timeseries: oil_fluor, chlorophyll, backscatter with thresholds.
    - Incident pane: agent explanation, confidence, waypoints, brief download.
    - Controls: “Start simulation”, “Inject event”, “Re-run detection”.
- Acceptance
   - End-to-end flow visible; brief downloadable (`uv run streamlit run apps/streamlit/App.py`).

11) E2E Demo Script and Data Staging (11.0–12.0h)
- Files
  - `tools/run_demo.py`
  - `configs/demo.yaml` – AOI, timeline, scene IDs to use.
- How
  - Start simulator; wait for trigger; run the pipeline; open Streamlit with prewired event.
  - Stage downloaded S‑1 tiles in `data/s1/processed/` with small AOI to keep runtime fast (script should skip re-downloads if cached).
- Acceptance
   - Single command runs a complete demo path (`uv run tools/run_demo.py`).

12) QA, Metrics, and Tuning
- Checks
  - Latency (alert→polygons): < 30 s with staged data.
  - False positives: anomaly gate suppressed by chlorophyll/backscatter cross-check in baseline period.
  - Visual QA: overlays look sensible; no land polygons; timestamps consistent.
- Artifacts
  - `artifacts/qa/` with comparison plots and metrics CSV.

## Data Preparation

- SeaOWL synthetic
  - Use `tools/sim_seaowl.py` to generate a 2–3 hr stream with one injected event.
- Sentinel‑1 tiles
  - Use `uv run src/satellite/tasker.py --bootstrap` (or `asf_search` CLI) to download 1–2 IW GRD scenes intersecting AOI; store in `data/s1/raw/`.
  - Crop to AOI and convert to lightweight GeoTIFF/COG in `data/s1/processed/`; record metadata in `configs/s1_catalog.json`.

## Command Cheatsheet

- `make setup` – `uv sync` to install deps.
- `uv run tools/sim_seaowl.py --duration 7200 --inject 3600,1800`
- `uv run src/satellite/tasker.py --bootstrap` – downloads/crops Sentinel‑1 scenes for AOI.
- `uv run tools/run_demo.py` – runs full pipeline headless (expects cached scenes).
- `uv run streamlit run apps/streamlit/App.py`

## Owner Matrix

- Anomaly + Simulator: Data/ML engineer
- SAR Detector + Tasker + Tracking: Remote sensing engineer
- Agent + Brief: Backend engineer
- Streamlit: Frontend/data viz
- QA + Demo: All hands

## 12‑Hour Timeline (Single Dev; parallelize if possible)

- 0.0–0.5h: Step 1
- 0.5–2.0h: Steps 2–3
- 2.0–4.5h: Steps 4–5 (includes download/crop)
- 4.5–6.5h: Steps 6–7
- 6.5–8.5h: Step 10
- 8.5–9.5h: Step 8
- 9.5–11.0h: Step 9
- 11.0–12.0h: Step 11 + QA/tuning

## Risks and Mitigations

- SAR false positives: use texture + morphology; filter by shoreline mask; cap polygon size.
- Timing drift: ensure simulator and scene times align; allow ±Δt padding in tasker query.
- Performance: pre-crop tiles to AOI; cache preprocessed rasters; keep GLCM window small.
