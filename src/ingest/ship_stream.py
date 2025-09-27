"""Helpers to read synthetic SeaOWL streams and persist Parquet batches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pandas as pd

QC_FIELDS = ["range", "spike", "stuck", "biofouling"]
BASE_FIELDS = [
    "ts",
    "lat",
    "lon",
    "depth_m",
    "platform_id",
    "sensor_id",
    "sensor_type",
    "sample_rate_hz",
    "mode",
    "oil_fluor_ppb",
    "chlorophyll_ug_per_l",
    "backscatter_m-1_sr-1",
    "temperature_c",
    "event_phase",
]


def iter_ndjson(path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from an NDJSON file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_record(record: dict) -> dict:
    data = {field: record.get(field) for field in BASE_FIELDS}
    qc = record.get("qc_flags", {}) or {}
    for key in QC_FIELDS:
        data[f"qc_{key}"] = qc.get(key)
    return data


@dataclass
class ParquetWriter:
    output_dir: Path
    batch_size: int = 600
    prefix: str = "seaowl"

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rows: List[dict] = []
        self._count = 0

    def add(self, record: dict) -> Optional[Path]:
        self._rows.append(normalize_record(record))
        if len(self._rows) >= self.batch_size:
            return self.flush()
        return None

    def flush(self) -> Optional[Path]:
        if not self._rows:
            return None
        df = pd.DataFrame(self._rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        chunk_id = self._count
        self._count += 1
        path = self.output_dir / f"{self.prefix}_{chunk_id:05d}.parquet"
        df.to_parquet(path)
        self._rows.clear()
        return path


def write_parquet_batches(records: Iterable[dict], output_dir: Path, batch_size: int = 600) -> List[Path]:
    writer = ParquetWriter(output_dir=output_dir, batch_size=batch_size)
    paths: List[Path] = []
    for record in records:
        path = writer.add(record)
        if path:
            paths.append(path)
    final = writer.flush()
    if final:
        paths.append(final)
    return paths


def replay_to_parquet(ndjson_path: Path, output_dir: Path, batch_size: int = 600) -> List[Path]:
    return write_parquet_batches(iter_ndjson(ndjson_path), output_dir=output_dir, batch_size=batch_size)


__all__ = [
    "iter_ndjson",
    "write_parquet_batches",
    "replay_to_parquet",
    "ParquetWriter",
]
