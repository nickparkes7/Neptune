.PHONY: setup status sync demo

setup:
	uv sync

status:
	uv run tools/status.py list

sync:
	uv run tools/sync_phase1.py

demo:
	uv run tools/run_demo.py

