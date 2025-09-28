# Synthetic Telemetry Streams

This note captures how the Phase 1 simulators synthesize telemetry and how that
data flows into the realtime stack.

## SeaOWL oil-fluorescence generator

`tools/sim_seaowl.py` produces newline-delimited JSON with 1 Hz samples.  The
generator follows either a straight-line drift from Brooklyn or the Cerulean
slick polygon, computes per-sample latitude/longitude with haversine math, and
injects an oil-fluorescence anomaly by multiplying the oil channel with a
Gaussian-shaped pulse.  Chlorophyll, backscatter, and temperature channels use
Ornstein–Uhlenbeck steps to stay within realistic ranges, and optional
`--sleep` output keeps the file tail streaming friendly.

## ECO FL algae-bloom generator

`tools/sim_ecofl.py` mirrors the SeaOWL routing so the two sensors stay co-
located.  Baseline temperature, chlorophyll, FDOM, phycocyanin, and
phycoerythrin channels are also driven by mean-reverting noise.  During a bloom
window the simulator boosts the pigment channels, providing a spike that the
agent can pick up via z-score analysis.

## Batch runner

`run-simulator.sh` launches both simulators together.  It keeps their file
targets aligned for either the NYC or Gulf routes, restarts them in a loop, and
exposes `--output` / `--eco-output` overrides for custom paths.

## Streaming server integration

`streaming-data-server.py` monitors the SeaOWL NDJSON for primary telemetry and
tracks the ECO FL file as a secondary sensor.  The server publishes sensor
heartbeats to the UI so each instrument gets a live status indicator, and it
feeds the ECO file path to the agent orchestration pipeline.

## Agent and anomaly handling

`src/anomaly/bloom_detector.py` calculates a rolling baseline for the ECO
channels and flags z-score peaks that indicate a bloom.  The agent runner calls
this detector before crafting a Cerulean plan; if the bloom branch fires it
skips the satellite query, emits a bloom-specific synopsis, and instructs the
crew accordingly.
