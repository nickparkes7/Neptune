"""Follow-up scheduling utilities for Cerulean re-queries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Literal, Optional
from uuid import uuid4

from pydantic import ValidationError

from common.pydantic_compat import CompatBaseModel

DEFAULT_FOLLOWUP_PATH = Path("data/cerulean/followups.ndjson")
FollowUpStatus = Literal["pending", "completed", "skipped"]


class FollowUpError(RuntimeError):
    """Raised when a follow-up entry cannot be persisted or parsed."""


class _FollowUpModel(CompatBaseModel):
    task_id: str
    event_id: str
    created_at: datetime
    run_at: datetime
    status: FollowUpStatus
    reason: str
    notes: Optional[str] = None


@dataclass
class FollowUpTask:
    task_id: str
    event_id: str
    created_at: datetime
    run_at: datetime
    status: FollowUpStatus = "pending"
    reason: str = ""
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        model = _FollowUpModel(
            task_id=self.task_id,
            event_id=self.event_id,
            created_at=_ensure_utc(self.created_at),
            run_at=_ensure_utc(self.run_at),
            status=self.status,
            reason=self.reason,
            notes=self.notes,
        )
        data = model.model_dump()
        data["created_at"] = _to_iso(model.created_at)
        data["run_at"] = _to_iso(model.run_at)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FollowUpTask":
        try:
            model = _FollowUpModel.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - data corruption guard
            raise FollowUpError("Invalid follow-up payload") from exc
        return cls(
            task_id=model.task_id,
            event_id=model.event_id,
            created_at=_ensure_utc(model.created_at),
            run_at=_ensure_utc(model.run_at),
            status=model.status,
            reason=model.reason,
            notes=model.notes,
        )

    @classmethod
    def build(cls, event_id: str, *, delay: timedelta = timedelta(days=1), reason: str = "") -> "FollowUpTask":
        now = datetime.now(timezone.utc)
        run_at = now + delay
        return cls(task_id=uuid4().hex, event_id=event_id, created_at=now, run_at=run_at, reason=reason)


def schedule_followup(
    event_id: str,
    *,
    reason: str,
    delay: timedelta = timedelta(days=1),
    store: Path = DEFAULT_FOLLOWUP_PATH,
) -> FollowUpTask:
    """Create and persist a follow-up task to re-query Cerulean."""

    task = FollowUpTask.build(event_id, delay=delay, reason=reason)
    append_followups([task], store=store)
    return task


def append_followups(tasks: Iterable[FollowUpTask], *, store: Path = DEFAULT_FOLLOWUP_PATH) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict()) + "\n")


def load_followups(*, store: Path = DEFAULT_FOLLOWUP_PATH) -> List[FollowUpTask]:
    if not store.exists():
        return []
    tasks: List[FollowUpTask] = []
    with store.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tasks.append(FollowUpTask.from_dict(data))
    return tasks


def iter_due_followups(
    *,
    as_of: Optional[datetime] = None,
    store: Path = DEFAULT_FOLLOWUP_PATH,
) -> Iterable[FollowUpTask]:
    now = _ensure_utc(as_of or datetime.now(timezone.utc))
    for task in load_followups(store=store):
        if task.status == "pending" and task.run_at <= now:
            yield task


def update_followup_status(
    task_id: str,
    *,
    status: FollowUpStatus,
    notes: Optional[str] = None,
    store: Path = DEFAULT_FOLLOWUP_PATH,
) -> FollowUpTask:
    tasks = load_followups(store=store)
    updated: Optional[FollowUpTask] = None
    for task in tasks:
        if task.task_id == task_id:
            task.status = status
            task.notes = notes
            updated = task
            break
    if updated is None:
        raise FollowUpError(f"Follow-up task {task_id} not found")
    _write_all(tasks, store)
    return updated


def _write_all(tasks: List[FollowUpTask], store: Path) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict()) + "\n")


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_iso(value: datetime) -> str:
    return _ensure_utc(value).replace(microsecond=0).isoformat().replace("+00:00", "Z")
