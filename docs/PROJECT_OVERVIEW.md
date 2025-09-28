# 1) Refined Problem & MVP

**Problem:** Detect and track water-quality anomalies (esp. oil slicks + algal blooms) by fusing **shipboard optical sensors** with **SAR/optical/ocean-color satellites**, then guide sampling/response.

**MVP (demoable in hackathon time):**

- A **simulated ship track** streaming Sea-Bird-like sensor channels (absorption/attenuation, backscatter, chlorophyll proxy, fluorescence).
- **Satellite swaths** along the track:
  - **Sentinel-1 (SAR)** for oil slick surface roughness.
  - **Sentinel-2 (optical)** for inland/coastal water quality & turbidity/chlorophyll indices.
  - Stretch: **VIIRS / PACE** ocean color scene(s) to validate chlorophyll patterns.
- **Anomaly pipeline:** detect anomalies on the ship stream → pull co-temporal satellite imagery for that segment → score spectral/textural similarity around the anomaly → track evolution over time with repeat acquisitions.
- **Streamlit dashboard** with a **GPT-5 agent** that explains anomalies, answers questions, and auto-generates incident reports.
- **fvdb** does the heavy lifting to co-register multi-modal, multi-resolution data for ML.

# 2) Data plan (open + synthetic)

- **Open datasets for references/priors**
  - Public Sentinel-1/2 tiles for a known coastal region with historical slicks/blooms.
  - A few VIIRS/PACE ocean-color granules for the same area/date window (if handy).
- **Synthetic to fill gaps**
  - Generate shipboard time-series that statistically mimic Sea-Bird ACS/Hyper-BB/ECO Puck channels; parameterize with distributions derived from the open scenes (so the synthetic looks like the region/time you’re demoing).
  - Create controllable “events” (e.g., oil-like drop in surface roughness + fluorescence spike) to stress-test the anomaly logic and make the live demo exciting.

# 3) Architecture (end-to-end)

**Ingestion**

- Ship stream → Kafka/WebSocket (or simple async generator) emitting lat/lon, time, sensor channels.
- Satellite fetcher → small wrapper that grabs pre-downloaded Sentinel-1/2/VIIRS tiles for the ship’s spatiotemporal window.

**Fusion (fvdb)**

- Define a common spatial index (e.g., tiled grid) and temporal index.
- Register modalities:
  - Vector: ship track.
  - Rasters: S1 (VV/VH backscatter), S2 (bands for water indices), VIIRS/PACE (chlorophyll/OC).
- fvdb handles different resolutions & projections; expose **query(window, time)** to pull aligned tensors for model/UX.

**Models**

- **Anomaly detection (fast):**
  - Univariate + multivariate z-score on shipboard channels (fluorescence, backscatter, a, c).
  - Optional compact **autoencoder** over the fused feature vector (ship + nearest-neighbor satellite patch stats).
- **Oil slick heuristic (SAR):**
  - Texture/contrast features on S1 (GLCM-style) around the anomaly location; threshold or tiny logistic head.
- **Spectral similarity (S2/OC):**
  - Compare anomaly patch spectra to nearby patches; rank “similar areas” for follow-up.

**Triggering**

- When anomaly score > τ (SeaOWL is always the trigger):
  1. **Query Cerulean first (cheap):** `query_cerulean(aoi, last≈48h)` to see if any recent satellite‑detected slicks exist near the event. If matches are found, import polygons + probable source hints and proceed with the Validation & Contextualization playbook.
  2. **If no match (First Discovery):** characterize using onboard data only and explicitly schedule a next‑day Cerulean recheck after the model updates using our follow‑up scheduler. No local satellite processing in Phase 1.
  3. Keep a direct satellite query capability as a future extension for other layers (e.g., S2 for blooms) or research comparisons.
  4. The incident pipeline automatically invokes the GPT-5 agent on taskable transitions, producing artifacts and scheduling follow-ups.
  5. Each agent run emits an `incident_brief.json` containing scenario, confidence, Cerulean parameters, summary metrics, follow-up plan, and artifact paths for downstream integration.

# 4) GPT-5 Agent (make this shine)

**Agent roles (tool‑calling, typed JSON)**

1. **Data Concierge** – given a region/time, composes the raster/vector queries to fvdb and returns aligned tensors.
2. **Explainer** – translates detections into plain language with uncertainty (“Probable oil-like surface dampening detected; S1 VV contrast −1.8σ, fluorescence +2.3σ”).
3. **Investigator** – runs “playbooks” and selects parameters:
   - _Validation & Contextualization (Cerulean match):_ use Cerulean polygons + source hints, ground‑truth with onboard data; returns a structured IncidentSynopsis with confidence.
   - _First Discovery (no match):_ onboard‑only characterization; schedules a Cerulean recheck after the next daily update via `schedule_followup` and logs planned actions.
   - _Future extension:_ direct satellite fetch/processing for layers not covered by Cerulean (e.g., S2 blooms).
   - _Bloom playbook:_ cross-check S2/OC indices vs. ship chlorophyll proxy → estimate extent and trend.
4. **Report Writer** – produces a one-click PDF “incident brief” and a JSON handoff for ops.
5. **Code Helper (Codex in Dev)** – on demand, writes small analysis snippets (e.g., a new plot, a filter) live in the demo.

**Why judges care:** Not just chatbot text—GPT‑5 produces typed JSON plans, selects Cerulean parameters (AOI/time/filter/sort), calls tools, synthesizes evidence into a confidence‑scored IncidentSynopsis, and schedules future actions.

# 5) Streamlit demo flow (3–5 minutes)

1. **Hook** – “We fuse ship sensors with satellites to spot and track spills/blooms in real time.”
2. **Live route** – show ship path; sensors streaming.
3. **Anomaly!** – synthetic event triggers; dashboard highlights segment; mini-charts spike.
4. **Agent explains** – ask: “What happened at 13:42 near 36.78N, −122.1W?”
   - GPT-5 explains with metrics, confidence, and likely cause (oil-like or bloom-like).
5. **Cerulean overlay** – agent shows returned polygons + source hints on the map; if no match, it explains the observational gap and schedules a recheck.
6. **Actionables** – agent proposes 3 sampling waypoints + a JSON incident brief (downloadable in the Streamlit console).
7. **Codex cameo** – you ask for a new plot (“contrast vs. wind speed” or “bbp vs. chlorophyll”), Codex writes it live.

# 6) What we’ll build this week (checklist)

- [ ] **fvdb schema** for: ship_timeseries, S1 VV/VH patches, S2 bands, optional VIIRS/PACE layers.
- [ ] **Data loaders**: prefetch tiles; synthetic ship generator seeded from open scenes.
- [ ] **Anomaly module**: simple stats + optional tiny autoencoder; SAR texture scorer.
- [ ] **Tasking simulator**: given anomaly time/lat-lon, lookup next-pass imagery timestamps and present “acquisition plan.”
- [ ] **GPT-5 agent** with tools: `query_fvdb()`, `run_playbook(kind)`, `generate_report()`, `make_plot(code)`.
- [ ] **Streamlit UI**: map + time slider; charts; agent chat; “similar areas” list; report download.
- [ ] **Demo data bundle**: small curated set of S1/S2 tiles + 1–2 OC scenes for a coastal AOI.

# 7) How this maps to judging

**Codex in Development (25%)**

- Live generate analysis/plot code; show Codex-authored ingestion/fusion snippets and tests.

**GPT-5 in Project (25%)**

- Typed‑JSON tool agent that selects query parameters, fuses onboard + Cerulean signals into a confidence‑scored synopsis, schedules follow‑ups, and generates narratives—auditable via an action trace.

**Live Demo (25%)**

- Clear narrative with a real anomaly, satellite overlays, similarity search, and report generation.

**Technicality (25%)**

- fvdb-backed multi-modal fusion, SAR texture features, spectral similarity, anomaly thresholds/AE, and simulated tasking pipeline.

# 8) Stretch goals (if time allows)

- Simple **drift model** (wind/current-aware) to predict spill extent between passes.
- **Few-shot classifier** primed with known slick/bloom patches from open datasets.
- **pySAS/above-water spec** placeholder channel in the schema (so you can brag about readiness for that sensor).
