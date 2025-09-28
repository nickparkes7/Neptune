"""Lightweight Cerulean API client and data helpers for Phase 1."""

from .client import (
    CeruleanClient,
    CeruleanError,
    CeruleanQueryResult,
    CeruleanSlick,
    CeruleanSummary,
    build_feature_collection,
    summarize_slicks,
)
from .followup import (
    FollowUpError,
    FollowUpTask,
    iter_due_followups,
    load_followups,
    schedule_followup,
    update_followup_status,
)

__all__ = [
    "CeruleanClient",
    "CeruleanError",
    "CeruleanQueryResult",
    "CeruleanSlick",
    "CeruleanSummary",
    "build_feature_collection",
    "FollowUpError",
    "FollowUpTask",
    "iter_due_followups",
    "load_followups",
    "schedule_followup",
    "summarize_slicks",
    "update_followup_status",
]
