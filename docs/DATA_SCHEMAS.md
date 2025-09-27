# 1) Sensors & exact data forms

## A) Sea-Bird **ECO V2 (2–4 channels)**

- **What it can measure (per channel)**: scattering (m⁻¹ sr⁻¹), turbidity (NTU), chlorophyll-a (µg/L), fDOM (ppb), phycocyanin/phycoerythrin (ppb). 16-bit digital, user-selectable 0.5–8 Hz sampling, RS-232 @ 19 200 baud; optional 0–5 V analog; on-board data-validation flags.
- **File/stream form**: onboard **binary** that you convert to CSV with Sea-Bird **UCI** software _or_ direct **RS-232 streaming** to your logger. ([Seabird][1])
- **Counts→engineering units** (for ECO-class optics): usually `(counts − dark) × scale_factor`, stored per-channel in the device file. ([Seabird][2])

**Data shape per sample (ECO V2):**

```text
timestamp, lat, lon, depth?, temp_C?, channel_1_value, channel_2_value, ...
+ qc_flags (bitfield), raw_counts? (optional), dark?, scale_factor?
```

---

## B) Sea-Bird **ac-s** (spectral absorption & attenuation)

- **What it measures**: spectra of **a(λ)** and **c(λ)** at ~**80 wavelengths** from **400–730 nm** (≈4 nm spacing), **4 Hz** sampling; 10 cm or 25 cm pathlength; RS-232/-422/-485.
- **Output/processing**: supports several **serial output protocols** and vendor tools (**WETView / UCI**) for acquisition & conversion; the **ac-meter protocol** doc covers collection & processing. ([NERC Field Spectroscopy Facility][3])
- **Important corrections** (for usable a(λ)/c(λ)): pure-water & temperature/salinity corrections, and scattering/stray-light handling—standardized in ac-meter/QA guides. ([Seabird][4])

**Data shape per sample (ac-s):**

```text
timestamp, lat, lon, temp_C?, pathlength_cm, wavelengths[n≈80],
a_m-1[n], c_m-1[n], qc_flags, (optional: raw detector channels)
```

---

## C) Sea-Bird **ECO FL** (single-channel fluorometer)

- **What it measures** (choose filter set): chlorophyll-a (470/695 nm), fDOM (370/460 nm), uranine, rhodamine, phycocyanin/phycoerythrin; **14-bit** digital; **RS-232 19 200 baud**, up to **8 Hz**; analog 0–5 V optional; battery/memory options.
- **Counts→engineering units**: same ECO formula `(counts − dark) × scale_factor` (per-sensor device file). ([Seabird][2])

**Data shape per sample (ECO FL):**

```text
timestamp, lat, lon, meas_type (e.g., chlorophyll_a), value, units,
raw_counts?, dark?, scale_factor?, qc_flags
```

---

# 2) Gaps we still need (to ensure the data are scientifically usable)

**Applies to ECO V2 / ECO FL**

- **Calibration sheet (.dev/.cal)** with **per-channel scale factors & dark offsets** (needed for counts→units). ([Seabird][2])
- Exact **channel configuration** you ordered (e.g., [Chl-a, fDOM, turbidity, BB-700] vs another set).
- Whether you’ll run **RS-232 streaming** or **internal logging → UCI CSV export**. ([Seabird][1])
- **Bio-wiper** presence & duty cycle (affects QA/QC flags and maintenance intervals).

**Applies to ac-s**

- The chosen **output protocol** (ASCII line format vs other) so we parse correctly. ([NERC Field Spectroscopy Facility][3])
- **Pure-water calibration file** and **T/S coefficients** for a(λ)/c(λ) correction. ([Seabird][4])
- **Plumbing volume / flow rate** from intake → ac-s (needed to estimate **transit time** for time/space alignment with ship GPS). (Best practice; referenced in QA/ops guides.) ([cdn.ioos.noaa.gov][5])

**Common ops gaps**

- **Clock sync** plan (GPS/NTP) for all instruments.
- **Georeferencing** source (ship GPS/NMEA sentence) and how it’s timestamped.
- **QC playbook** (stuck-value, range, spike, rate-of-change tests) per QARTOD ocean optics. ([cdn.ioos.noaa.gov][5])

---

# 3) Ingestion plan (how we’ll read the data)

- **Realtime (preferred for demo):**

  - Pull **RS-232 (19 200 baud, 8-N-1)** from each sensor into a small Python serial daemon; write **newline-delimited JSON** or **CSV**. (Sea-Bird notes RS-232 sensors have unique frame formats—our parser will be per-instrument.) ([Sea-Bird Scientific Blog][6])

- **Batch (fallback):**

  - Use vendor tools (UCI / WETView) to convert **binary logs → CSV**; we then ingest the CSV. ([Seabird][1])

We’ll standardize on **Parquet** for storage and register everything in **fvdb** as:

- a **vector timeseries** (ship track) with attributes (ECO\*/ac-s) and
- **queryable tensors** keyed by `(time, lat, lon)` (so models can pull aligned windows). (fVDB supports multi-modal, differentiable ops over spatial data.) ([arXiv][7])

---

# 4) Schemas (canonical + per-sensor)

## 4.1 Canonical “measurements” record (applies to every sample)

```json
{
  "ts": "UTC ISO8601",
  "lat": <float>, "lon": <float>, "depth_m": <float|null>,
  "platform_id": "vessel_001", "sensor_id": "ECO_V2_01",
  "sensor_type": "eco_v2|eco_fl|acs",
  "sample_rate_hz": <float>, "mode": "serial|logged",
  "qc_flags": { "range":0|1, "spike":0|1, "stuck":0|1, "biofouling":0|1 }
}
```

## 4.2 ECO V2 payload

```json
{
  "eco_v2": {
    "channels": ["scattering_700nm","turbidity","chlorophyll_a","fDOM"],
    "values": [<m^-1 sr^-1>, <NTU>, <ug/L>, <ppb>],
    "raw_counts": [<int>...], "dark": [<int>...], "scale": [<float>...],
    "instrument_temp_C": <float|null>, "wiper_active": <bool|null>
  }
}
```

## 4.3 ECO FL payload

```json
{
  "eco_fl": {
    "meas_type": "chlorophyll_a|fDOM|... ",
    "value": <float>, "units": "ug/L|ppb",
    "raw_counts": <int>, "dark": <int>, "scale": <float>,
    "instrument_temp_C": <float|null>
  }
}
```

## 4.4 ac-s payload (spectral)

Two options; pick **(A)** for ML-friendliness, **(B)** for analytics/SQL.

**A) Wide (arrays)**

```json
{
  "acs": {
    "pathlength_cm": 25,
    "wavelength_nm": [400,404,...,730],
    "a_m-1": [ ... n≈80 ... ],
    "c_m-1": [ ... n≈80 ... ],
    "instrument_temp_C": <float|null>
  }
}
```

**B) Long (normalized)**

```
(ts, lat, lon, wl_nm, a_m-1, c_m-1, pathlength_cm, temp_C, qc_flags)
```

---

# 5) What I’ll assume (unless you tell me otherwise)

- **ECO V2 4-channel config**: `scattering_700nm`, `turbidity`, `chlorophyll_a`, `fDOM` (best coverage for anomalies/blooms & oil-like signals).
- **ECO FL**: configure for **chlorophyll-a** if ECO V2 doesn’t already include it; otherwise set to **fDOM** for redundancy and sensitivity.
- **ac-s**: **25 cm pathlength**, **ASCII protocol**, 4 Hz; we’ll apply water/T-S corrections during ingestion. ([Seabird][4])

---

## TL;DR — what to procure/confirm now

1. **Calibration/device files** for each ECO channel (scale & dark), and **ac-s** water/T-S coefficients. ([Seabird][2])
2. Exact **ECO V2 channel set** and **ECO FL filter set**.
3. **Output mode** for each instrument (RS-232 vs internal logging) and a **clock-sync plan**. ([Seabird][1])
4. **Transit-time** metadata (intake→sensor flow rate & tubing volume) for time alignment. ([cdn.ioos.noaa.gov][5])

If you’re good with these assumptions, I’ll draft the **parsers + schema classes** and the small **serial daemons** to start capturing data in exactly this shape.

[1]: https://www.seabird.com/asset-get.download.jsa?id=70469884051&utm_source=chatgpt.com "User manual ECO V2"
[2]: https://www.seabird.com/asset-get.download.jsa?id=69833855317&utm_source=chatgpt.com "User manual ECO fluorometers and scattering sensors"
[3]: https://fsf.nerc.ac.uk/assets/documents/AC-S/WET%20Labs%20AC-S%20User%20Guide%20Rev%20M.pdf?utm_source=chatgpt.com "Spectral Absorption and Attenuation Sensor"
[4]: https://www.seabird.com/asset-get.download.jsa?id=69833849025&utm_source=chatgpt.com "ac Meter Protocol Document"
[5]: https://cdn.ioos.noaa.gov/media/2017/12/qartod_ocean_optics_manual.pdf?utm_source=chatgpt.com "Manual for - Real-Time Quality Control of Ocean Optics Data"
[6]: https://blog.seabird.com/rs232-integration/?utm_source=chatgpt.com "RS-232 Sensor Integration Guide"
[7]: https://arxiv.org/abs/2407.01781?utm_source=chatgpt.com "fVDB: A Deep-Learning Framework for Sparse, Large ..."
