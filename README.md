# Neptune

This repo contains the Phase 1 prototype for the Neptune spill response pipeline. The workflow revolves around a synthetic SeaOWL stream, anomaly scoring, and visual artifacts that help interpret the signal. Minimal update: after a SeaOWL anomaly, the agent first performs a lightweight Cerulean query for recent satellite‑detected slicks; if none are found we characterize with onboard data only, mark it as a "first discovery," and schedule a next‑day Cerulean re‑query. Direct satellite fetching is retained as a future extension for data layers not covered by Cerulean (e.g., algal blooms), not as a fallback.

## Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) for environment management (preferred)

Create or refresh the environment:

```cmd
uv venv
uv sync
```

## Core Pipeline Commands

1. **Simulate SeaOWL stream** — regenerates the canonical NDJSON sample.

   ```cmd
   uv run tools/sim_seaowl.py --output data/ship/seaowl_sample.ndjson
   ```

2. **Plot baseline telemetry + track** — produces `artifacts/seaowl/seaowl_timeseries.png` and `seaowl_track.png` for quick visual inspection.

   ```cmd
   uv run python tools/plot_seaowl.py data/ship/seaowl_sample.ndjson --outdir artifacts/seaowl
   ```

3. **Run Hybrid Oil Alert** — adaptive baseline + absolute limits + persistence with visuals under `artifacts/hybrid/`.

   ```cmd
   uv run python tools/plot_hybrid_alerts.py \
     --input data/ship/seaowl_sample.ndjson \
     --output artifacts/hybrid
   ```

4. **Emit Suspected Spill events** — validates the trigger wiring and produces structured payloads for downstream steps.

   ```cmd
   uv run python tools/run_event_trigger.py data/ship/seaowl_sample.ndjson --pretty
   ```

5. **Run incident pipeline** — end-to-end scorer → event → lifecycle transitions with cooldown-aware state machine.

   ```cmd
   uv run python tools/run_incident_pipeline.py data/ship/seaowl_sample.ndjson --pretty --flush-after 1800
   # Include the GPT-5 agent (requires .env with OPENAI_API_KEY)
   uv run python tools/run_incident_pipeline.py data/ship/seaowl_sample.ndjson --run-agent --agent-mode gpt --pretty
   ```

6. **Task Sentinel-1 scenes** — filters the local catalog to the incident AOI/time window.

   ```cmd
   uv run python src/satellite/tasker.py --bbox "-74.3,40.5,-73.5,41.2" --start 2024-09-01T00:00:00Z --end 2024-09-04T00:00:00Z --pretty
   ```

7. **Execute smoke tests** — validates the scorer, trigger wiring, incident manager, and tasker.

   ```cmd
   uv run python -m unittest tests.test_hybrid_scorer tests.test_event_trigger tests.test_incident_manager tests.test_pipeline tests.test_tasker
   ```

8. **Run agent orchestration** — feeds a stored `SuspectedSpillEvent` through the GPT-5 agent (requires `OPENAI_API_KEY`).

```cmd
echo "OPENAI_API_KEY=your_key_here" > .env
uv run python tools/run_agent.py tests/data/sample_event.json --artifact-root artifacts/agent_demo --followup-store data/cerulean/followups.ndjson
```

> `tools/load_env.py` automatically loads `.env` for most entry points. When running custom scripts, call it before constructing the agent if needed.

9. **Launch Streamlit incident console** — explore telemetry, maps, and agent output interactively.

```cmd
uv run streamlit run apps/incident_console.py
```

The sidebar lets you pick a SeaOWL stream, toggle GPT-5 (falls back to the rule-based agent if `OPENAI_API_KEY` is missing), and run the pipeline on demand. The app visualises timeseries data, Cerulean overlays, incident synopsis, and offers a download button for the JSON brief.

## Cerulean Follow-ups

- Scheduled re-query tasks are stored at `data/cerulean/followups.ndjson` (see `configs/cerulean.yml`).
- Run `uv run pytest -q tests/test_cerulean_followup.py` to exercise the scheduling utilities.

## Incident Briefs

- Each agent run writes an `incident_brief.json` alongside the synopsis and overlay artifacts (see `artifacts/<event_id>/`).
- Briefs capture the scenario, confidence, chosen Cerulean parameters, summary metrics, follow-up plan, and artifact paths—ideal for system hand-offs or audit trails.

## Status Tracking

Update the Phase 1 tracker when starting/stopping tasks, then sync the public status block.

```cmd
uv run tools/status.py set <step_id> <status> --owner "nicholas"
uv run tools/sync_phase1.py
```

Refer to `PHASE1.md` and `status/phase1.yml` for live progress snapshots.
