# GPT‑5 Agent: Demo Playbook

This is a concise, presentation‑ready reference for positioning the agent, justifying its role, and making it pop in the demo. Use it as your final pre‑brief.

## Why It Matters
- Collapse detection → decision: Converts tracks, radar, and features into a ranked, explainable incident brief an operator can act on.
- Cognitive load reducer: Automates triage, context‑stitching, and narrative building so analysts decide, not dig.
- Evidence‑first and auditable: Every claim links to concrete artifacts and data paths for reproducibility.
- Glue across the stack: Orchestrates simulator, detector, and console so the system feels like a product, not parts.

## Judge Justifications
- Decision support, not chat: Outputs schema‑bound briefs with citations, not freeform text.
- Trust via traceability: Assertions reference exact files and plots (track, time‑series, feature bars).
- Faster time‑to‑insight: Auto‑collects multi‑sensor evidence, highlights change points, ranks risk, and proposes next actions.
- Deterministic demo: Uses cached `data/` and `artifacts/` for consistent, reproducible outputs.
- Extensible orchestration: Same scaffolding later drives tasking and multi‑source linking.

## Two‑Hour Upgrades (High Impact, Low Risk)

### 1) Incident Brief Generator (deterministic)
- Build `tools/generate_incident_brief.py` → writes `artifacts/briefs/latest.{json,md}` using cached evidence.
- Inputs: `artifacts/seaowl/seaowl_track.png`, `artifacts/seaowl/seaowl_timeseries.png`, `artifacts/hybrid/hybrid_values.png`, config + scenario metadata.
- Output schema: overview, risk score, key observations, recommended actions, and citations with file paths.
- Wire simple summarization in `src/agent/model.py` so it runs offline if LLM is unavailable.
- Run:

```cmd
uv run tools/generate_incident_brief.py --scenario seaowl_demo
```

### 2) Console “Explain Selection” panel
- Add right‑side “Agent Brief” drawer in `neptune-console/src/components/Sidebar.tsx` that loads `artifacts/briefs/latest.json`.
- From `neptune-console/src/components/ShipMap.tsx`, add a button on selected track: “Explain anomaly”.
- Render bullets + inline thumbnails; bullets link to artifact paths and can highlight on the map.

### 3) CLI fallback for judges
- Add `uv run apps/console.py brief --ship <id>` to print a condensed brief with file links for terminal demos.
- Touchpoints: `apps/console.py`, `src/agent/model.py`.

## Demo Flow (60–90 seconds)
- Select suspicious track on the map.
- Click “Explain anomaly” → Agent Brief opens (3–5 bullets, risk score) with:
  - Track plot: `artifacts/seaowl/seaowl_track.png`
  - Time‑series with change points: `artifacts/seaowl/seaowl_timeseries.png`
  - Feature bars: `artifacts/hybrid/hybrid_values.png`
- Click “Export brief” → saves `artifacts/briefs/latest.md` (optional PDF later).
- Toggle “Show sources” → shows exact file paths and config used.

## What To Implement Where
- `src/agent/model.py`: Add `build_brief(evidence) -> dict` returning schema‑bound JSON with citations; provide offline fallback summarization.
- `tools/generate_incident_brief.py`: Gather evidence, call agent, write JSON + Markdown.
- `neptune-console/src/components/Sidebar.tsx`: New Agent Brief drawer that reads `artifacts/briefs/latest.json`.
- `neptune-console/src/components/ShipMap.tsx`: Add “Explain anomaly” button to trigger brief load for selected ship.
- `apps/console.py`: `brief` command to print top bullets with paths.

## Brief JSON Schema (minimal, judge‑friendly)

Example shape:

```json
{
  "scenario_id": "seaowl_demo",
  "generated_at": "2025-09-28T12:34:56Z",
  "risk_score": 0.82,
  "summary": "Vessel exhibits speed variance and heading zig-zag near AOI boundary.",
  "observations": [
    {"id": "obs_track_pattern", "text": "Track shows sharp course changes over 15 min."},
    {"id": "obs_speed_variance", "text": "Speed variance exceeds 95th percentile baseline."},
    {"id": "obs_hybrid", "text": "Hybrid feature composite above risk threshold."}
  ],
  "recommended_actions": [
    {"id": "act_watch", "text": "Increase monitoring for next 30 min."},
    {"id": "act_tasking", "text": "Queue repeat pass if pattern persists."}
  ],
  "citations": [
    {"claim_id": "obs_track_pattern", "path": "artifacts/seaowl/seaowl_track.png"},
    {"claim_id": "obs_speed_variance", "path": "artifacts/seaowl/seaowl_timeseries.png"},
    {"claim_id": "obs_hybrid", "path": "artifacts/hybrid/hybrid_values.png"}
  ],
  "sources": {
    "data_root": "data/",
    "artifacts_root": "artifacts/",
    "config": "configs/seaowl_demo.yaml"
  }
}
```

Notes:
- Keep values deterministic (e.g., seeded thresholds) for stable demos.
- Prefer IDs for cross‑reference between observations and citations.

## React Drawer Props (minimal)

```ts
// Sidebar.tsx
interface AgentBriefProps {
  briefPath?: string; // default: artifacts/briefs/latest.json
  open: boolean;
  onClose: () => void;
}
```

Render bullets with thumbnails; each bullet’s click opens its `path` in a new tab and can signal the map to highlight the related area.

## CLI Command (console)

```cmd
uv run apps/console.py brief --ship <id> [--json artifacts/briefs/latest.json]
```

- Prints: scenario, risk score, top 3 observations, and file paths.
- If `--json` is present, it reads and prints that brief; otherwise triggers generation if missing.

## Metrics To Call Out
- Time‑to‑brief: “<10s from selection to brief”.
- Evidence coverage: “3 modalities, 100% claims cited to artifacts/”.
- Determinism: “No network needed; all from cached `data/`”.

## Talking Points (Lightning Round)
- “This is decision support, not chat.”
- “Every claim is cited to a file you can click.”
- “Same agent will later drive tasking and cross‑source linking.”

---

Maintenance notes:
- Keep brief generation offline‑capable; only optional LLM calls.
- Avoid inventing paths; only use those in `PHASE1.md` and `configs/*`.
- When tasks change, update `status/phase1.yml` and run sync:

```cmd
uv run tools/sync_phase1.py
```
