"""Shared local file iteration helpers for corpus adapters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def iter_local_json_rows(directory: Path) -> Iterator[Dict[str, Any]]:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        yield row
                    else:
                        yield {"text": str(row)}
            elif isinstance(payload, dict):
                if isinstance(payload.get("data"), list):
                    for row in payload["data"]:
                        yield row if isinstance(row, dict) else {"text": str(row)}
                else:
                    yield payload
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)
        elif path.suffix == ".csv":
            with path.open("r", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    yield dict(row)
        elif path.suffix == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield {"text": line.strip()}


def take(limit: Optional[int], rows: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    if limit is None:
        yield from rows
        return
    for i, row in enumerate(rows):
        if i >= limit:
            break
        yield row


def first_present(row: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None
