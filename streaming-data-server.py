#!/usr/bin/env python3
"""WebSocket streaming data server for real-time telemetry at 1Hz."""

import argparse
import asyncio
import json
import mimetypes
import os
import time
import websockets
from pathlib import Path
from typing import Set, Dict, Any, Optional, Tuple
import threading
import queue
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from anomaly import PipelineConfig
from anomaly.events import ChannelStats, EventExtractorConfig, OilStats, SuspectedSpillEvent
from agent.briefing import generate_agent_brief_for_event
from agent.runner import AgentConfig, run_agent_for_event
from agent.model import GPTAgentModel, RuleBasedAgentModel
from cerulean.client import CeruleanClient, CeruleanQueryResult, CeruleanError

INCIDENT_CACHE: Dict[str, Dict[str, Any]] = {}
INCIDENT_CACHE_LOCK = threading.Lock()
AGENT_BRIEF_ROOT = Path("artifacts/briefs")


def _slugify(text: str) -> str:
    return (
        text.replace(":", "")
        .replace("-", "")
        .replace("T", "")
        .replace("Z", "")
        .replace(".", "")
    )


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _incident_id_for_snapshot(snapshot, fallback: Optional[datetime] = None) -> str:
    if getattr(snapshot, "event_id", None):
        return snapshot.event_id  # type: ignore[attr-defined]
    ts = getattr(snapshot, "ts_peak", None) or fallback
    if ts is None:
        ts = datetime.now(timezone.utc)
    iso_ts = _to_iso(ts) or ts.isoformat()
    return _slugify(iso_ts)


def _event_payload(snapshot) -> Dict[str, Any]:
    return {
        "event_id": getattr(snapshot, "event_id", None),
        "ts_start": _to_iso(getattr(snapshot, "ts_start", None)),
        "ts_end": _to_iso(getattr(snapshot, "ts_end", None)),
        "ts_peak": _to_iso(getattr(snapshot, "ts_peak", None)),
        "lat": getattr(snapshot, "lat", None),
        "lon": getattr(snapshot, "lon", None),
    }


def _update_incident_cache_from_transition(transition) -> None:
    snapshot = transition.incident
    incident_id = _incident_id_for_snapshot(snapshot, fallback=transition.at)
    status_map = {
        "opened": "analyzing",
        "updated": "ready",
        "heartbeat": None,
        "closed": "closed",
    }
    status = status_map.get(transition.kind)

    with INCIDENT_CACHE_LOCK:
        record = INCIDENT_CACHE.setdefault(
            incident_id,
            {
                "incident_id": incident_id,
                "summary": "Analyzing SeaOWL anomaly…",
                "scenario": "pending_analysis",
                "confidence": 0.0,
            },
        )

        if status:
            record["status"] = status
        else:
            record.setdefault("status", "analyzing")

        event_payload = _event_payload(snapshot)
        record["event"] = event_payload
        if event_payload.get("ts_peak"):
            record["ts_peak"] = event_payload["ts_peak"]
        else:
            record.setdefault("ts_peak", _to_iso(transition.at))

        record["updated_at"] = _to_iso(transition.at)
        record.setdefault("summary", "Analyzing SeaOWL anomaly…")
        record.setdefault("scenario", "pending_analysis")
        record.setdefault("confidence", 0.0)


def _update_incident_cache_status(incident_id: str, status: str, *, event: Optional[Dict[str, Any]] = None) -> None:
    with INCIDENT_CACHE_LOCK:
        record = INCIDENT_CACHE.setdefault(
            incident_id,
            {
                "incident_id": incident_id,
                "summary": "Analyzing SeaOWL anomaly…",
                "scenario": "pending_analysis",
                "confidence": 0.0,
            },
        )
        record["status"] = status
        if event:
            record["event"] = event
            if event.get("ts_peak"):
                record["ts_peak"] = event["ts_peak"]
        record.setdefault("updated_at", _to_iso(datetime.now(timezone.utc)))


DEFAULT_DATA_PATH_NYC = Path("data/ship/seaowl_live.ndjson")
DEFAULT_DATA_PATH_GULF = Path("data/ship/seaowl_gulf_live.ndjson")


class TelemetryStreamer:
    def __init__(self, data_path: Path):
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.data_queue = queue.Queue()
        self.incident_queue = queue.Queue()
        self.last_position = 0
        self.data_path = data_path
        self.running = True

    async def register_client(self, websocket):
        """Register a new WebSocket client."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")

        # Send initial data to new client
        try:
            initial_data = self.get_recent_data(100)  # Last 100 points
            if initial_data:
                await websocket.send(json.dumps({
                    "type": "initial",
                    "data": initial_data
                }))
        except Exception as e:
            print(f"Error sending initial data: {e}")

    async def unregister_client(self, websocket):
        """Unregister a WebSocket client."""
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")

    def get_recent_data(self, count: int = 100) -> list:
        """Get the most recent data points."""
        if not self.data_path.exists():
            return []

        try:
            with open(self.data_path, 'r') as f:
                # Read from end efficiently
                f.seek(0, 2)
                file_size = f.tell()

                # Estimate bytes needed for last N lines
                chunk_size = min(file_size, count * 200)
                f.seek(max(0, file_size - chunk_size))

                content = f.read()
                lines = content.strip().split('\n')

                # Take last N lines and parse
                recent_lines = lines[-count:] if len(lines) > count else lines
                data = []

                for line in recent_lines:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            data.append(record)
                        except json.JSONDecodeError:
                            continue

                return data
        except Exception as e:
            print(f"Error reading data: {e}")
            return []

    def watch_file_changes(self):
        """Watch for new data in the file and queue it."""
        print("Starting file watcher...")

        while self.running:
            try:
                if not self.data_path.exists():
                    time.sleep(0.1)
                    continue

                # Check file size
                current_size = self.data_path.stat().st_size

                if current_size > self.last_position:
                    # Read new data
                    with open(self.data_path, 'r') as f:
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        self.last_position = f.tell()

                    # Parse and queue new records
                    for line in new_lines:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                self.data_queue.put(record)
                            except json.JSONDecodeError:
                                continue
                elif current_size < self.last_position:
                    # File was truncated/reset
                    self.last_position = 0

                time.sleep(0.1)  # Check every 100ms for responsiveness

            except Exception as e:
                print(f"Error in file watcher: {e}")
                time.sleep(1)

    async def broadcast_new_data(self):
        """Broadcast new data to all connected clients."""
        print("Starting data broadcaster...")

        while self.running:
            try:
                # Collect any queued data points
                new_data = []
                while not self.data_queue.empty():
                    try:
                        record = self.data_queue.get_nowait()
                        new_data.append(record)
                    except queue.Empty:
                        break

                pending_messages = []
                if new_data:
                    pending_messages.append({"type": "update", "data": new_data})

                while not self.incident_queue.empty():
                    try:
                        pending_messages.append(self.incident_queue.get_nowait())
                    except queue.Empty:
                        break

                if pending_messages and self.clients:
                    disconnected = set()
                    for client in self.clients.copy():
                        for payload in pending_messages:
                            try:
                                await client.send(json.dumps(payload))
                            except websockets.exceptions.ConnectionClosed:
                                disconnected.add(client)
                                break
                            except Exception as e:
                                print(f"Error sending to client: {e}")
                                disconnected.add(client)
                                break

                    self.clients -= disconnected

                    if new_data:
                        print(f"Broadcasted {len(new_data)} data points to {len(self.clients)} clients")

                await asyncio.sleep(0.05)  # 20Hz check rate for smooth streaming

            except Exception as e:
                print(f"Error in broadcaster: {e}")
                await asyncio.sleep(1)

    async def handle_websocket(self, websocket):
        """Handle WebSocket connections."""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                # Handle any client messages if needed
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)


class IncidentPipelineMonitor:
    """Background worker that runs the anomaly → incident pipeline."""

    def __init__(
        self,
        data_path: Path,
        *,
        poll_interval: float = 5.0,
        min_duration_s: float = 5.0,
        min_samples: int = 5,
        flush_after_s: float = 900.0,
        incident_queue: Optional[queue.Queue] = None,
        agent_enabled: bool = True,
    ) -> None:
        self.data_path = data_path
        self.poll_interval = poll_interval
        self.pipeline_config = PipelineConfig(
            agent_enabled=False,
            event_config=EventExtractorConfig(
                require_alarm=True,
                min_duration_s=min_duration_s,
                min_samples=min_samples,
            ),
            flush_after_s=flush_after_s,
        )
        self.agent_config = AgentConfig()
        self.agent_config.artifact_root.mkdir(parents=True, exist_ok=True)
        self._agent_model: Optional[object] = None
        self._seen_transitions: Set[str] = set()
        self._processed_signatures: Dict[str, Set[str]] = {}
        self._last_stat: Optional[Tuple[int, float]] = None
        self._lock = threading.Lock()
        self.running = True
        self._incident_queue = incident_queue
        self._agent_enabled = agent_enabled
        self._prime_existing_transitions()

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()
        return thread

    def _run_loop(self) -> None:
        while self.running:
            try:
                self._process_once()
            except Exception as exc:  # noqa: BLE001
                print(f"[IncidentPipelineMonitor] Error: {exc}")
            time.sleep(self.poll_interval)

    def _process_once(self) -> None:
        if not self.data_path.exists():
            return

        stat = self.data_path.stat()
        current_sig = (stat.st_size, stat.st_mtime_ns)
        if self._last_stat == current_sig:
            return
        self._last_stat = current_sig

        from anomaly import generate_transitions_from_ndjson

        result = generate_transitions_from_ndjson(self.data_path, config=self.pipeline_config)
        transitions = result.transitions

        now_utc = datetime.now(timezone.utc)
        new_transitions = []
        for transition in transitions:
            if transition.kind == "closed":
                at = transition.at
                if at.tzinfo is None:
                    at = at.replace(tzinfo=timezone.utc)
                else:
                    at = at.astimezone(timezone.utc)
                if at > now_utc:
                    # Ignore speculative closures scheduled in the future (streaming pipeline artifact).
                    continue

            key = self._transition_key(transition)
            if key in self._seen_transitions:
                continue
            self._seen_transitions.add(key)
            sig = self._transition_signature(transition)
            kinds = self._processed_signatures.setdefault(sig, set())
            if transition.kind in kinds:
                continue
            kinds.add(transition.kind)
            new_transitions.append(transition)

        if not new_transitions:
            return

        print(
            f"[IncidentPipelineMonitor] New transitions: "
            f"{', '.join(f'{t.kind}:{t.incident.event_id}' for t in new_transitions)}"
        )

        for transition in new_transitions:
            _update_incident_cache_from_transition(transition)

        if self._incident_queue is not None:
            payload = {
                "type": "incident_transition",
                "transitions": [
                    {
                        "kind": transition.kind,
                        "at": transition.at.isoformat().replace("+00:00", "Z"),
                        "incident_id": transition.incident.event_id,
                        "ts_peak": transition.incident.ts_peak.isoformat().replace("+00:00", "Z"),
                        "lat": transition.incident.lat,
                        "lon": transition.incident.lon,
                        "oil_max": transition.incident.oil_stats.max,
                        "oil_max_z": transition.incident.oil_stats.max_z,
                    }
                    for transition in new_transitions
                ],
            }
            self._incident_queue.put(payload)

        for transition in new_transitions:
            if not transition.allow_tasking:
                continue
            self._dispatch_agent(transition)

    def _dispatch_agent(self, transition) -> None:
        if not self._agent_enabled:
            return
        incident = transition.incident
        artifact_dir = self.agent_config.artifact_root / (incident.event_id or self._slugify(incident.ts_peak.isoformat()))
        synopsis_path = artifact_dir / "incident_synopsis.json"
        if synopsis_path.exists():
            print(
                f"[IncidentPipelineMonitor] Synopsis already exists for {incident.event_id}; skipping agent run."
            )
            return

        model = self._ensure_agent_model()
        primary_client = self._cerulean_client()
        primary_label = "stub" if getattr(primary_client, "_is_stub", False) else "live"
        clients_to_try = [(primary_client, primary_label)]
        if primary_label != "stub":
            clients_to_try.append((self._cerulean_client(force_stub=True), "stub"))

        last_error: Optional[BaseException] = None
        for client, label in clients_to_try:
            try:
                run_agent_for_event(
                    incident,
                    model=model,
                    client=client,
                    config=self.agent_config,
                    timestamp=transition.at,
                    stream_path=self.data_path,
                )
                note = " (Cerulean fallback)" if label == "stub" and last_error else ""
                print(
                    "[IncidentPipelineMonitor] Agent run complete for incident "
                    f"{incident.event_id}{note}."
                )
                if label == "stub" and last_error:
                    self._annotate_synopsis_with_fallback(
                        synopsis_path,
                        last_error,
                    )
                _update_incident_cache_status(
                    incident.event_id or self._slugify(incident.ts_peak.isoformat()),
                    "ready",
                    event=_event_payload(incident),
                )
                return
            except CeruleanError as exc:
                last_error = exc
                print(
                    "[IncidentPipelineMonitor] Cerulean error for incident "
                    f"{incident.event_id} using {label} client: {exc}"
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(
                    "[IncidentPipelineMonitor] Agent run failed for incident "
                    f"{incident.event_id} using {label} client: {exc}"
                )

        if last_error is not None:
            print(
                "[IncidentPipelineMonitor] Exhausted Cerulean fallbacks for incident "
                f"{incident.event_id}: {last_error}"
            )

    def _annotate_synopsis_with_fallback(self, path: Path, reason: BaseException) -> None:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(
                "[IncidentPipelineMonitor] Failed to annotate synopsis fallback for "
                f"{path.parent.name}: {exc}"
            )
            return

        metadata = payload.setdefault("metadata", {})
        metadata["cerulean_client"] = "stub"
        metadata["cerulean_fallback_reason"] = str(reason)

        try:
            path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:  # noqa: BLE001
            print(
                "[IncidentPipelineMonitor] Failed to persist synopsis fallback annotation for "
                f"{path.parent.name}: {exc}"
            )

    def _ensure_agent_model(self):
        with self._lock:
            if self._agent_model is not None:
                return self._agent_model
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self._agent_model = GPTAgentModel()
                    # Trigger lazy client init to validate key quickly
                    self._agent_model._get_client()  # type: ignore[attr-defined]
                    print("[IncidentPipelineMonitor] Using GPTAgentModel for agent runs.")
                    return self._agent_model
                except Exception as exc:  # noqa: BLE001
                    print(
                        "[IncidentPipelineMonitor] Failed to initialize GPT agent, falling back to rule-based model: "
                        f"{exc}"
                    )
                    self._agent_model = None
            self._agent_model = RuleBasedAgentModel()
            print("[IncidentPipelineMonitor] Using RuleBasedAgentModel for agent runs.")
            return self._agent_model

    @staticmethod
    def _cerulean_client(*, force_stub: bool = False):
        if force_stub or os.getenv("CERULEAN_STUB") == "1":
            class _EmptyCeruleanClient:
                _is_stub = True

                def query_slicks(
                    self,
                    *_,
                    **__,
                ) -> CeruleanQueryResult:
                    return CeruleanQueryResult(slicks=[], number_matched=0, number_returned=0, links=[])

            return _EmptyCeruleanClient()
        client = CeruleanClient()
        setattr(client, "_is_stub", False)
        return client

    @staticmethod
    def _transition_key(transition) -> str:
        incident_id = transition.incident.event_id or "unknown"
        return f"{incident_id}:{transition.kind}:{transition.at.isoformat()}"

    @staticmethod
    def _transition_signature(transition) -> str:
        ts_peak = transition.incident.ts_peak.isoformat() if transition.incident.ts_peak else transition.at.isoformat()
        max_oil = f"{transition.incident.oil_stats.max:.6f}" if transition.incident.oil_stats else "0.0"
        return f"{ts_peak}:{max_oil}"

    @staticmethod
    def _slugify(text: str) -> str:
        return (
            text.replace(":", "")
            .replace("-", "")
            .replace("T", "")
            .replace("Z", "")
            .replace(".", "")
        )

    def _prime_existing_transitions(self) -> None:
        if not self.data_path.exists():
            return
        try:
            from anomaly import generate_transitions_from_ndjson, PipelineConfig

            seed_config = PipelineConfig(
                scorer=self.pipeline_config.scorer,
                event_config=self.pipeline_config.event_config,
                incident_config=self.pipeline_config.incident_config,
                flush_after_s=self.pipeline_config.flush_after_s,
                agent_enabled=False,
            )

            result = generate_transitions_from_ndjson(self.data_path, config=seed_config)
            for transition in result.transitions:
                key = self._transition_key(transition)
                self._seen_transitions.add(key)
                sig = self._transition_signature(transition)
                kinds = self._processed_signatures.setdefault(sig, set())
                kinds.add(transition.kind)
        except Exception as exc:  # noqa: BLE001
            print(f"[IncidentPipelineMonitor] Failed to seed transitions: {exc}")

# HTTP server for REST endpoints (incidents, health, etc.)
class HTTPHandler(BaseHTTPRequestHandler):
    DATA_STREAM_PATH: Optional[Path] = None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == '/health':
            self._send_json({"status": "ok", "streaming": True})
            return

        if path == '/incidents':
            self._send_json(self._collect_incidents())
            return

        if path.startswith('/incidents/'):
            parts = [segment for segment in path.split('/') if segment]
            if len(parts) == 2:
                detail = self._collect_incident_detail(parts[1])
                if detail is None:
                    self._send_json({"error": "Incident not found"}, status=404)
                else:
                    self._send_json(detail)
                return
            if len(parts) == 3 and parts[2] == 'brief':
                self._serve_artifact(parts[1], 'incident_brief.json', download_name='incident_brief.json')
                return
            if len(parts) == 3 and parts[2] == 'trace':
                self._serve_artifact(parts[1], 'agent_trace.jsonl', content_type='application/x-ndjson')
                return
            if len(parts) >= 3 and parts[2] == 'agent-brief':
                if len(parts) == 3:
                    refresh = params.get('refresh', ['0'])[0] in {'1', 'true', 'True'}
                    self._serve_incident_agent_brief_json(parts[1], refresh=refresh)
                    return
                if len(parts) == 4 and parts[3] == 'markdown':
                    self._serve_incident_agent_brief_markdown(parts[1])
                    return
                if len(parts) == 5 and parts[3] == 'media':
                    self._serve_incident_agent_brief_media(parts[1], parts[4])
                    return

        if path == '/agent-brief':
            self._serve_agent_brief_json()
            return

        if path == '/agent-brief/markdown':
            self._serve_agent_brief_markdown()
            return

        if path.startswith('/agent-brief/media/'):
            slug = path[len('/agent-brief/media/'):]
            self._serve_agent_brief_media(slug)
            return

        self._send_json({"error": "Use WebSocket for telemetry data"}, status=404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_json(self, payload: Dict[str, Any] | list, *, status: int = 200) -> None:
        self.send_response(status)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def _serve_agent_brief_json(self) -> None:
        json_path = AGENT_BRIEF_ROOT / "latest.json"
        if not json_path.exists():
            self._send_json({"error": "Agent brief not available"}, status=404)
            return
        try:
            payload = json.loads(json_path.read_text())
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)
            return
        self._send_json(payload)

    def _serve_agent_brief_markdown(self) -> None:
        md_path = AGENT_BRIEF_ROOT / "latest.md"
        if not md_path.exists():
            self._send_json({"error": "Agent brief markdown not available"}, status=404)
            return
        self._send_file(md_path, content_type='text/markdown; charset=utf-8')

    def _serve_agent_brief_media(self, slug: str) -> None:
        media_root = (AGENT_BRIEF_ROOT / "media").resolve()
        target = (media_root / slug).resolve()
        if not str(target).startswith(str(media_root)):
            self._send_json({"error": "Invalid media path"}, status=400)
            return
        if not target.exists():
            self._send_json({"error": "Media not found"}, status=404)
            return
        content_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        self._send_file(target, content_type=content_type)

    def _serve_incident_agent_brief_json(self, incident_id: str, *, refresh: bool = False) -> None:
        if not self._ensure_incident_agent_brief(incident_id, refresh=refresh):
            self._send_json({"error": "Agent brief not available"}, status=404)
            return
        brief_path = Path("artifacts") / incident_id / "agent_brief" / "brief.json"
        try:
            payload = json.loads(brief_path.read_text())
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)
            return
        self._send_json(payload)

    def _serve_incident_agent_brief_markdown(self, incident_id: str) -> None:
        if not self._ensure_incident_agent_brief(incident_id, refresh=False):
            self._send_json({"error": "Agent brief not available"}, status=404)
            return
        md_path = Path("artifacts") / incident_id / "agent_brief" / "brief.md"
        if not md_path.exists():
            self._send_json({"error": "Agent brief markdown not available"}, status=404)
            return
        self._send_file(md_path, content_type='text/markdown; charset=utf-8')

    def _serve_incident_agent_brief_media(self, incident_id: str, slug: str) -> None:
        if not slug:
            self._send_json({"error": "Missing media name"}, status=400)
            return
        media_root = (Path("artifacts") / incident_id / "agent_brief" / "media").resolve()
        target = (media_root / slug).resolve()
        if not str(target).startswith(str(media_root)):
            self._send_json({"error": "Invalid media path"}, status=400)
            return
        if not target.exists():
            self._send_json({"error": "Media not found"}, status=404)
            return
        content_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        self._send_file(target, content_type=content_type)

    def _ensure_incident_agent_brief(self, incident_id: str, *, refresh: bool) -> bool:
        brief_json = Path("artifacts") / incident_id / "agent_brief" / "brief.json"
        if brief_json.exists() and not refresh:
            return True
        return self._generate_incident_agent_brief(incident_id)

    def _generate_incident_agent_brief(self, incident_id: str) -> bool:
        incident_dir = Path("artifacts") / incident_id
        synopsis_path = incident_dir / "incident_synopsis.json"
        if not synopsis_path.exists():
            return False
        try:
            synopsis = json.loads(synopsis_path.read_text())
        except Exception:  # noqa: BLE001
            return False

        event_payload = synopsis.get("event")
        if not event_payload:
            return False

        event = self._event_from_payload(event_payload)
        if event is None:
            return False

        stream_path = self.DATA_STREAM_PATH
        if stream_path is None or not Path(stream_path).exists():
            return False

        try:
            generate_agent_brief_for_event(
                event,
                stream_path=Path(stream_path),
                output_dir=incident_dir / "agent_brief",
                media_url_base=f"/incidents/{incident_id}/agent-brief/media",
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _event_from_payload(payload: Dict[str, Any]) -> Optional[SuspectedSpillEvent]:
        try:
            oil_stats_payload = payload.get("oil_stats") or {}
            oil_stats = OilStats(**oil_stats_payload)
            context_channels_payload = payload.get("context_channels", {})
            context = {
                key: ChannelStats(**value)
                for key, value in context_channels_payload.items()
                if isinstance(value, dict)
            }
            aoi_bbox = payload.get("aoi_bbox")
            if not aoi_bbox or len(aoi_bbox) != 4:
                aoi_bbox = (payload.get("lon", 0.0), payload.get("lat", 0.0), payload.get("lon", 0.0), payload.get("lat", 0.0))

            return SuspectedSpillEvent(
                event_id=payload.get("event_id", ""),
                ts_start=_parse_iso_ts(payload.get("ts_start")),
                ts_end=_parse_iso_ts(payload.get("ts_end")),
                ts_peak=_parse_iso_ts(payload.get("ts_peak")),
                lat=float(payload.get("lat", 0.0)),
                lon=float(payload.get("lon", 0.0)),
                duration_s=float(payload.get("duration_s", 0.0)),
                sample_count=int(payload.get("sample_count", 1)),
                platform_id=payload.get("platform_id"),
                sensor_id=payload.get("sensor_id"),
                oil_stats=oil_stats,
                context_channels=context,
                aoi_bbox=tuple(aoi_bbox),
            )
        except Exception:  # noqa: BLE001
            return None

    def _send_file(
        self,
        path: Path,
        *,
        content_type: str,
        download_name: Optional[str] = None,
    ) -> None:
        try:
            payload = path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)
            return

        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-type', content_type)
        if download_name:
            self.send_header('Content-Disposition', f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(payload)

    def _collect_incidents(self) -> list:
        artifacts_root = Path("artifacts")
        incidents_by_id: Dict[str, Dict[str, Any]] = {}

        if artifacts_root.exists():
            for incident_dir in sorted(artifacts_root.iterdir()):
                if not incident_dir.is_dir():
                    continue
                synopsis_path = incident_dir / "incident_synopsis.json"
                if not synopsis_path.exists():
                    continue
                try:
                    synopsis = json.loads(synopsis_path.read_text())
                except Exception:
                    continue

                event_payload = synopsis.get("event") or {}
                ts_peak = event_payload.get("ts_peak")
                if not ts_peak:
                    brief_path = incident_dir / "incident_brief.json"
                    if brief_path.exists():
                        try:
                            brief = json.loads(brief_path.read_text())
                            ts_peak = brief.get("incident", {}).get("ts_peak")
                        except Exception:
                            ts_peak = None

                incidents_by_id[incident_dir.name] = {
                    "incident_id": incident_dir.name,
                    "scenario": synopsis.get("scenario"),
                    "confidence": synopsis.get("confidence"),
                    "summary": synopsis.get("summary"),
                    "ts_peak": ts_peak,
                    "path": str(incident_dir),
                    "status": synopsis.get("status", "ready"),
                    "agent_brief_available": False,
                }

                brief_json = incident_dir / "agent_brief" / "brief.json"
                if brief_json.exists():
                    try:
                        brief_payload = json.loads(brief_json.read_text())
                        incidents_by_id[incident_dir.name]["agent_brief_available"] = True
                        incidents_by_id[incident_dir.name]["agent_brief"] = {
                            "headline": brief_payload.get("headline"),
                            "risk_label": brief_payload.get("risk_label"),
                            "risk_score": brief_payload.get("risk_score"),
                            "generated_at": brief_payload.get("generated_at"),
                            "hero_image": brief_payload.get("hero_image"),
                        }
                    except Exception:
                        incidents_by_id[incident_dir.name]["agent_brief_available"] = False

        with INCIDENT_CACHE_LOCK:
            for incident_id, cached in INCIDENT_CACHE.items():
                record = incidents_by_id.get(incident_id)
                if record is None:
                    incidents_by_id[incident_id] = {
                        "incident_id": incident_id,
                        "scenario": cached.get("scenario", "pending_analysis"),
                        "confidence": cached.get("confidence", 0.0),
                        "summary": cached.get("summary", "Analyzing SeaOWL anomaly…"),
                        "ts_peak": cached.get("ts_peak"),
                        "path": str(artifacts_root / incident_id) if (artifacts_root / incident_id).exists() else "",
                        "status": cached.get("status", "analyzing"),
                        "agent_brief_available": False,
                    }
                else:
                    if cached.get("status"):
                        record["status"] = cached["status"]
                    if not record.get("ts_peak") and cached.get("ts_peak"):
                        record["ts_peak"] = cached["ts_peak"]
                    if not record.get("summary") and cached.get("summary"):
                        record["summary"] = cached["summary"]

        incidents = list(incidents_by_id.values())
        incidents.sort(key=lambda record: record.get("ts_peak") or "", reverse=True)
        return incidents

    def _collect_incident_detail(self, incident_id: str) -> Optional[Dict[str, Any]]:
        incident_dir = Path("artifacts") / incident_id
        synopsis_path = incident_dir / "incident_synopsis.json"
        brief_path = incident_dir / "incident_brief.json"
        trace_path = incident_dir / "agent_trace.jsonl"

        with INCIDENT_CACHE_LOCK:
            cached_record = INCIDENT_CACHE.get(incident_id)

        if not synopsis_path.exists():
            if cached_record is None:
                return None
            event_payload = cached_record.get("event")
            return {
                "incident_id": incident_id,
                "scenario": cached_record.get("scenario", "pending_analysis"),
                "confidence": cached_record.get("confidence", 0.0),
                "summary": cached_record.get("summary", "Analyzing SeaOWL anomaly…"),
                "rationale": None,
                "recommended_actions": [],
                "followup_scheduled": False,
                "followup_eta": None,
                "metrics": {},
                "artifacts": {},
                "event": event_payload,
                "status": cached_record.get("status", "analyzing"),
                "brief_available": False,
                "trace_available": False,
            }

        try:
            synopsis = json.loads(synopsis_path.read_text())
        except Exception:
            return None

        event_payload = synopsis.get("event")
        if event_payload is None and brief_path.exists():
            try:
                brief = json.loads(brief_path.read_text())
                event_payload = brief.get("incident")
            except Exception:
                event_payload = None

        detail = {
            "incident_id": incident_id,
            "scenario": synopsis.get("scenario"),
            "confidence": synopsis.get("confidence"),
            "summary": synopsis.get("summary"),
            "rationale": synopsis.get("rationale"),
            "recommended_actions": synopsis.get("recommended_actions", []),
            "followup_scheduled": synopsis.get("followup_scheduled", False),
            "followup_eta": synopsis.get("followup_eta"),
            "metrics": synopsis.get("metrics", {}),
            "artifacts": synopsis.get("artifacts", {}),
            "event": event_payload,
            "status": synopsis.get("status", "ready"),
            "brief_available": brief_path.exists(),
            "trace_available": trace_path.exists(),
            "agent_brief_available": False,
        }

        if cached_record:
            if cached_record.get("status"):
                detail["status"] = cached_record["status"]
            if not detail.get("event") and cached_record.get("event"):
                detail["event"] = cached_record.get("event")

        agent_brief_json = incident_dir / "agent_brief" / "brief.json"
        if agent_brief_json.exists():
            detail["agent_brief_available"] = True
            try:
                payload = json.loads(agent_brief_json.read_text())
                detail["agent_brief"] = {
                    "headline": payload.get("headline"),
                    "risk_label": payload.get("risk_label"),
                    "risk_score": payload.get("risk_score"),
                    "generated_at": payload.get("generated_at"),
                    "hero_image": payload.get("hero_image"),
                    "summary": payload.get("summary"),
                }
            except Exception:
                detail["agent_brief_available"] = False

        return detail

    def _serve_artifact(
        self,
        incident_id: str,
        artifact_name: str,
        *,
        download_name: Optional[str] = None,
        content_type: str = 'application/json',
    ) -> None:
        incident_dir = Path("artifacts") / incident_id
        artifact_path = incident_dir / artifact_name
        if not artifact_path.exists():
            self._send_json({"error": "Artifact not found"}, status=404)
            return

        try:
            payload = artifact_path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)
            return

        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-type', content_type)
        if download_name:
            self.send_header('Content-Disposition', f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(payload)


def _parse_iso_ts(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neptune streaming data server")
    pattern_group = parser.add_mutually_exclusive_group()
    pattern_group.add_argument("--nyc", action="store_true", help="Use NYC SeaOWL stream (default)")
    pattern_group.add_argument("--gulf", action="store_true", help="Use Gulf/Cerulean SeaOWL stream")
    parser.add_argument("--data-path", type=Path, help="Explicit path to SeaOWL NDJSON stream")
    parser.add_argument("--no-agent", action="store_true", help="Disable agent runs (artifacts not generated)")
    return parser.parse_args()


async def main():
    """Main server function."""
    args = parse_args()

    if args.data_path is not None:
        data_path = args.data_path
    elif args.gulf:
        data_path = DEFAULT_DATA_PATH_GULF
    else:
        data_path = DEFAULT_DATA_PATH_NYC

    data_path = data_path.resolve()
    print(f"📁 Monitoring telemetry file: {data_path}")
    print(f"🤖 Agent runs: {'enabled' if not args.no_agent else 'disabled'}")

    streamer = TelemetryStreamer(data_path)
    incident_monitor = IncidentPipelineMonitor(
        streamer.data_path,
        incident_queue=streamer.incident_queue,
        agent_enabled=not args.no_agent,
    )
    HTTPHandler.DATA_STREAM_PATH = streamer.data_path

    # Start file watcher in background thread
    file_watcher_thread = threading.Thread(target=streamer.watch_file_changes, daemon=True)
    file_watcher_thread.start()

    # Start pipeline monitor thread
    incident_monitor.start()

    # Start HTTP server in background thread
    def run_http_server():
        httpd = HTTPServer(('', 8000), HTTPHandler)
        print("HTTP server running on http://localhost:8000")
        httpd.serve_forever()

    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    print("🚀 Starting Neptune Streaming Data Server")
    print("📡 WebSocket server: ws://localhost:8001")
    print("🌐 HTTP server: http://localhost:8000")
    print("📊 Streaming telemetry data at real-time rates")

    # Start WebSocket server and data broadcaster
    async with websockets.serve(streamer.handle_websocket, "localhost", 8001):
        # Start the data broadcaster
        broadcaster_task = asyncio.create_task(streamer.broadcast_new_data())

        print("✅ Server ready - waiting for connections...")

        try:
            await broadcaster_task
        except KeyboardInterrupt:
            print("\n🛑 Shutting down streaming server...")
            streamer.running = False
            incident_monitor.running = False

if __name__ == "__main__":
    asyncio.run(main())
