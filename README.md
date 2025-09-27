# Neptune

This repo contains the Phase 1 prototype for the Neptune spill response pipeline. The workflow revolves around a synthetic SeaOWL stream, anomaly scoring, and visual artifacts that help interpret the signal.

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

4. **Execute smoke tests** — validates the hybrid scorer against the sample stream.

   ```cmd
  uv run python -m unittest tests.test_hybrid_scorer
   ```

## Status Tracking

Update the Phase 1 tracker when starting/stopping tasks, then sync the public status block.

```cmd
uv run tools/status.py set <step_id> <status> --owner "nicholas"
uv run tools/sync_phase1.py
```

Refer to `PHASE1.md` and `status/phase1.yml` for live progress snapshots.
