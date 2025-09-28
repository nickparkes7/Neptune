from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cerulean import (
    FollowUpError,
    FollowUpTask,
    iter_due_followups,
    load_followups,
    schedule_followup,
    update_followup_status,
)


def test_schedule_and_load_followup(tmp_path: Path):
    store = tmp_path / "followups.ndjson"
    task = schedule_followup(
        "event-123",
        reason="cerulean gap",
        delay=timedelta(hours=2),
        store=store,
    )
    assert task.event_id == "event-123"
    loaded = load_followups(store=store)
    assert len(loaded) == 1
    assert loaded[0].task_id == task.task_id
    assert loaded[0].reason == "cerulean gap"
    assert loaded[0].status == "pending"


def test_iter_due_followups(tmp_path: Path):
    store = tmp_path / "followups.ndjson"
    # schedule one due in past, one in future
    schedule_followup(
        "event-due",
        reason="due soon",
        delay=timedelta(minutes=-5),
        store=store,
    )
    schedule_followup(
        "event-future",
        reason="future",
        delay=timedelta(hours=1),
        store=store,
    )
    due = list(iter_due_followups(as_of=datetime.now(timezone.utc), store=store))
    assert len(due) == 1
    assert due[0].event_id == "event-due"


def test_update_followup_status(tmp_path: Path):
    store = tmp_path / "followups.ndjson"
    task = schedule_followup(
        "event-456",
        reason="state change",
        delay=timedelta(hours=1),
        store=store,
    )
    updated = update_followup_status(
        task.task_id,
        status="completed",
        notes="queried cerulean",
        store=store,
    )
    assert updated.status == "completed"
    assert updated.notes == "queried cerulean"
    round_trip = load_followups(store=store)
    assert round_trip[0].status == "completed"
    with pytest.raises(FollowUpError):
        update_followup_status("missing", status="skipped", store=store)
