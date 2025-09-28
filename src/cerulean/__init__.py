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

__all__ = [
    "CeruleanClient",
    "CeruleanError",
    "CeruleanQueryResult",
    "CeruleanSlick",
    "CeruleanSummary",
    "build_feature_collection",
    "summarize_slicks",
]
