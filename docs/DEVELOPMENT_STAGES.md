Love this framing. Let’s pivot the plan so the **onboard sensor(s) run continuously**, the **GPT-5 agent** is the brains “in the loop,” and **satellite data is pulled on-demand** only when needed—to trace **origin** and **propagation** of a suspected slick.

Below is a tight, staged build that gets you a slick MVP quickly and then layers sophistication.

# Phase 1 — MVP (continuous onboard + triggered satellite)

**Goal:** Detect oil-like anomalies from the ship’s live stream and, when triggered, fetch a small stack of satellite scenes to locate origin and project near-term spread.

## Inputs

- **Onboard (continuous, cheap):**

  - **SeaOWL UV-A**: oil-in-water fluorescence (EX/EM 370/460 nm), with chlorophyll and 700 nm backscatter channels for discrimination; oil sensitivity down to ~3 ppb, LOD < 80 ppb.

- **Satellite (on-demand, heavy):**

  - **Sentinel-1 SAR** tiles (IW GRD) for surface slick morphology around alert windows.
  - (Optional, if clear skies) **Sentinel-2 optical** frames for context/visualization.

## System behavior (end-to-end)

1. **Continuous monitor**

   - A lightweight process samples SeaOWL UV-A at 1 Hz (or your chosen rate).
   - Compute rolling baselines and an **anomaly score** (e.g., robust z-score + median absolute deviation) on the **oil channel**, with cross-checks on **chlorophyll** and **backscatter** to reduce biological/foam false positives.

2. **Agent-in-the-loop trigger**

   - When score > τ for ≥ N seconds, the **GPT-5 agent** is invoked with a “suspected spill” event (lat/lon, time, sensor snippets, wind/current if available).
   - The agent decides to **call a SatelliteTasking tool** with a tight spatiotemporal envelope:

     - **Historical window** (e.g., −7 days → now) to find the **origin**.
     - **Forward window** (next pass predictions) to track **propagation**.

3. **Targeted satellite fetch** (on demand)

   - Pull a **small stack**: prior S-1 passes along the ship track and up-current sector; nearest post-trigger pass when available.
   - Run a **fast dark-spot + texture** detector to outline candidate slick polygons in each pass.
   - Link polygons through time to estimate **drift vector** and **growth**.

4. **Agent explanations & actionables**

   - GPT-5 generates a short “incident brief”: location, magnitude, confidence, probable **origin window**, **likely transport direction**, and **recommended waypoints** for follow-up sampling.
   - One-click PDF export + JSON handoff.

## Streamlit MVP UI

- **Map:** live ship track, SeaOWL anomaly markers, satellite polygons (only after trigger).
- **Timeseries:** oil-fluor signal + context channels.
- **Incident pane:** origin/propagation summary, confidence, waypoints.
- **Chat (GPT-5):** “Explain this event,” “show most likely source area,” “propose 3 sampling stops in the next 2 hrs.”

## Why this nails your narrative

- **Cheap continuous monitoring**: only SeaOWL runs nonstop.
- **Expensive data only on demand**: satellite pulled _after_ anomaly—exactly your point.
- **Root-cause & projection**: the historical stack finds where it started; sequential passes estimate spread.

---

# Phase 2 — Better science (still lightweight)

- **Wind/current assist:** ingest gridded winds/currents (simple vector advection) to refine propagation between satellite passes.
- **Confidence model:** combine SeaOWL peak, SAR polygon contrast/shape metrics, and met-ocean factors into a composite confidence score.
- **False-positive guardrails:** use SeaOWL **chlorophyll/backscatter** channels to argue against purely biological films when oil channel is weak.

---

# Phase 3 — Add algal blooms (parallel “playbook”)

- Reuse the same architecture: the agent triggers a **BloomTasking** flow if chlorophyll spikes with no oil channel rise.
- Pull **Sentinel-2** + ocean-color frames to map bloom extent; report uses the “bloom playbook.”
- Your dashboard simply shows a **different overlay** and a different explanation; same UX.

---

# Phase 4 — Where **fvdb** shines (justify it clearly)

When you move beyond a single sensor + a couple satellite layers, pandas/xarray glue starts to creak. **fvdb** gives you a scalable, GPU-friendly **spatiotemporal index** with operators for sampling, splatting, sparse conv, and **fast ray-marching**—ideal for **multi-modal**, **multi-resolution** earth-observation fusion:

- **Why it matters here (concrete):**

  1. **Query unification:** one call to retrieve _aligned_ tensors from SeaOWL (vector time series) **and** co-registered raster patches (S-1/S-2/OC) over a 3D cube (x, y, t). fvdb’s **IndexGrid/JaggedTensor** abstraction is built for variable-density voxels across batches.
  2. **Operator speed at scale:** GPU-accelerated grid building, **HDDA ray-marching** to skip empty space, and high-efficiency **sparse conv** can outpace hash-grid engines and handle **much larger inputs** (lower memory footprint, faster ray traversal).
  3. **Modeling fused data:** once you store SeaOWL + SAR/optical stacks, you can train small sparse-CNNs over spatiotemporal bricks (slick evolution, source-attribution classifiers) leveraging fvdb’s **sparse convolution + attention** kernels.

> Judge-friendly line: _“We start with cheap continuous sensing, and when it matters we ‘zoom out’ in space and time. fvdb is our scalable substrate to unify and learn from these heterogeneous data cubes when we graduate from a single ship to a fleet and from one sensor to many.”_

---

# GPT-5 Agent: minimal tools (Phase 1) → richer (later)

**Phase 1 tools**

- `get_ship_window(start, end, bbox)` → SeaOWL snippets & stats.
- `task_satellite(aoi, time_range, sensors=['S1'])` → returns scene IDs and rasters.
- `detect_slicks(rasters)` → polygons + scores.
- `make_brief(event_id, polygons, winds)` → PDF/JSON.

**Phase 3–4 additions**

- `advect(polygons, winds_currents, dt)` → forecast polygons.
- `fvdb_query(cube_spec)` → aligned tensors for ML/analytics.
- `similar_patches(polygon, stack)` → historical look-alikes (origin hints).

---

# Implementation checklist (you can build this fast)

## MVP (this week)

- [ ] **SeaOWL stream sim** (readings at 1 Hz; inject events).
- [ ] **Anomaly scorer** (rolling baseline + MAD; configurable τ).
- [ ] **Satellite tasker stub**: function that returns pre-downloaded S-1 tiles for aoi/time.
- [ ] **SAR slick detector**: dark-spot threshold + simple texture filters → polygon.
- [ ] **Streamlit app**: live timeseries, map, incident panel, GPT-5 chat.
- [ ] **Agent prompt + tools** wired to the above; one-click PDF brief.

## Nice-to-have in MVP+

- [ ] **Wind vector** ingestion; arrow overlay; naive advection forecast.
- [ ] **Confidence score**: sensor spike + polygon metrics.
- [ ] **Origin finder**: backward scan across stack; earliest confident polygon.

## fvdb pilot (Phase 4)

- [ ] **Define cube schema** (x, y, t) with SeaOWL resampled to grid; S-1/S-2 tiles as channels.
- [ ] **Load a tiny cube** and demo one **fvdb query** + a simple **sparse conv** over (x,y,t) to classify “slick/no-slick” patches.
- [ ] Show **HDDA**-accelerated sampling or a **JaggedTensor** batch to argue scalability.

---

# Streamlit demo flow (aligned to your story)

1. **Idle**: SeaOWL timeseries humming along (cheap, continuous).
2. **Spike**: oil channel rises → alert.
3. **Agent fires tasking**: grabs recent S-1 stack, outlines polygons, shows earliest appearance (origin).
4. **Propagation**: shows drift arrow and a 2-pass comparison; proposes sampling waypoints.
5. **Brief**: click to export PDF & JSON.
6. (If time) **fvdb button**: “Run fused-cube analysis (pilot)” → shows aligned tensors over the event and a tiny model score.

---

## Sensor notes (for reviewers)

- **SeaOWL UV-A** is designed for oil-in-water detection and includes **chlorophyll** and **backscattering** channels specifically to **discriminate crude oil** from natural FDOM/phytoplankton—ideal for your trigger logic.

---

If you want, I’ll draft:

- The **Streamlit skeleton** (pages + components).
- A **Python stub** for `task_satellite` (reading local S-1 scenes), `detect_slicks`, and the **anomaly scorer**.
- The **agent tool schema** as Python function signatures you can drop in.

And when you’re ready for the fvdb pilot, I’ll propose the **cube layout** and a **toy sparse-CNN** over (x,y,t) to demo why fvdb helps at scale.
