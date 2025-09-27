# Synthetic Data Plan

## 1. Purpose

- Provide realistic time-aligned sensor and satellite feeds for the demo when open data are missing or not timely enough.
- Stress-test the anomaly detection, tasking simulator, and GPT-5 agent workflows without relying on live operations.
- Keep control over event frequency, severity, and metadata so the narrative stays crisp during the presentation.

## 2. Target Scenarios

- **Oil slick event**
  - Shipboard: drop in surface roughness proxies, elevated fluorescence/fDOM.
  - Satellite: Sentinel-1 VV/VH texture dampening, slight Sentinel-2 turbidity change.
- **Algal bloom flare-up**
  - Shipboard: chlorophyll proxy spikes, absorption/attenuation spectra shape change.
  - Satellite: Sentinel-2/VIIRS chlorophyll indices high, nearby similar spectral patches.
- **Background baseline**
  - Quiescent periods with nominal variability to train/validate anomaly thresholds.

## 3. Synthetic Components

- **Ship sensors**
  - ECO V2 4-channel stream @ 4 Hz (scattering_700nm, turbidity, chlorophyll_a, fDOM).
  - ECO FL single-channel (fDOM) redundancy when needed.
  - ac-s spectral curves (80 wavelengths, 4 Hz) co-sampled with ship GPS.
- **Satellite swaths**
  - Pre-cut Sentinel-1 patches (VV/VH backscatter) every ~6 hours.
  - Sentinel-2 Level-2A tiles with relevant bands (B2, B3, B4, B8, B11) and derived indices.
  - Optional VIIRS/PACE chlorophyll rasters for validation overlays.
- **Environmental context**
  - Wind/current metadata (from reanalysis or simple parametric model) to justify drift narratives.
  - Tasking metadata: simulated revisit times and acquisition IDs.

## 4. Generation Approach

1. **Baseline synthesis**
   - Fit distributions and temporal correlations from the open reference scenes noted in the data plan.
   - Sample multi-channel shipboard data with autoregressive noise to maintain realistic dynamics.
2. **Event injection**
   - Oil slick: impose 15–30 minute window with dampened backscatter, +2–3 fluorescence/fDOM, slight turbidity rise.
   - Bloom: apply gradual chlorophyll increase, spectral shape shifts on ac-s, matching Sentinel-2 chlorophyll index bumps.
   - Ensure events align with satellite pass windows to enable fusion storytelling.
3. **Satellite fabrication**
   - Start from real imagery when possible; otherwise warp baseline tiles using texture/spectral transforms.
   - Generate anomaly masks and inject into raster bands while preserving radiometric ranges.
4. **Packaging**
   - Emit shipboard streams as newline JSON and Parquet snapshots for replay.
   - Store satellite patches as Cloud-Optimized GeoTIFF (COG) or fvdb-native tensors keyed by `(lat, lon, time)`.

## 5. Validation & QA

- Statistical checks: compare synthetic channel distributions vs. observed priors (mean, variance, autocorrelation).
- Visual QA: quicklook plots for key events (ship time series, SAR texture heatmaps, spectral curves).
- Scenario review: confirm anomalies trip the same thresholds as expected by the anomaly module.
- Metadata sanity: verify timestamps, geolocation continuity, and fvdb registration hashes.

## 6. Integration Plan

- Register synthetic assets in fvdb alongside open-source reference layers (flagged with `source="synthetic"`).
- Provide scripted replay tool (`python tools/replay_ship_stream.py`) to feed the Streamlit demo.
- Precompute incident playbook outputs (e.g., predicted drift) to ensure GPT-5 agent responses stay deterministic.
- Bundle a markdown briefing and JSON manifest describing files, events, and calibration coefficients.

## 7. Timeline & Owners

| Week | Focus | Owner |
| --- | --- | --- |
| W1 | Fit priors, generate baseline ship streams, label QA checks | Data synthesis lead |
| W2 | Inject oil/bloom scenarios, craft satellite counterparts | Remote sensing lead |
| W3 | fvdb registration, replay tooling, end-to-end dry run | Platform engineer |

## 8. Open Questions

- Confirm final channel list and sampling rates for the hardware emulation.
- Decide whether VIIRS/PACE layers are stretch or must-have for the demo script.
- Determine acceptable synthetic vs. real data blend for judging (percentage per segment).
- Validate whether we need day/night variations or cloud cover scenarios for resiliency messaging.
