# Agent Working Agreement (Repo‑wide)

This file defines persistent instructions for any agent working in this repository. Agents must follow these rules when reading/writing code and docs.

## Core Behaviors

- Source of truth for Phase 1 progress is `status/phase1.yml`.
- Keep `PHASE1.md` in sync by running `uv run tools/sync_phase1.py` after changes to `status/phase1.yml`.
- Use `uv` for Python env and execution (`uv venv`, `uv sync`, `uv run`).
- Prefer minimal, focused changes; don’t refactor unrelated code.
- Never invent data paths; use those declared in `PHASE1.md` and `configs/*`.

## Documentation Discipline

- `PHASE1.md` carries a live status block between the markers:
  - `<!-- STATUS:PHASE1:BEGIN -->` and `<!-- STATUS:PHASE1:END -->`
- The content inside these markers is auto‑generated. Do not hand‑edit between them.
- Update `status/phase1.yml` for all task state changes; then sync `PHASE1.md`.

## Task Tracking

- Step IDs (must be used in commits/PRs):
  - 1_bootstrap, 2_simulator, 3_anomaly, 4_events, 5_tasker,
  - 6_detector, 7_linking, 8_agent, 9_brief, 10_streamlit,
  - 11_demo, 12_qa
- Allowed statuses: `pending`, `in_progress`, `blocked`, `done`.

## Commands

- `uv run tools/status.py list` → status summary.
- `uv run tools/status.py set <step_id> <status> [--owner you] [--notes text]` → update.
- `uv run tools/sync_phase1.py` → regenerate status block in `PHASE1.md`.

## PR Requirements

- PRs must include updated `status/phase1.yml` and a regenerated `PHASE1.md` status block when tasks change.
- Use the PR template in `.github/pull_request_template.md` and list Step IDs touched.

## Scope Guardrails (Phase 1)

- Onboard SeaOWL data is synthetic; Sentinel‑1 is real via web API.
- No fvdb integration in Phase 1. PDF brief is optional.
- Keep demo deterministic with cached data under `data/`.

If a conflict exists between this file and an explicit user instruction, the user instruction wins.
