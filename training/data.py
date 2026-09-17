"""Dataset loading via repository abstractions (JSON / JSONL / CSV).

Training depends on ``DatasetRepository`` — not on a concrete database.
MongoDB / Postgres repositories can be added later without changing trainers.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset

from models.backends.base import BaseModelBackend
from repositories.base import DatasetRepository


@dataclass
class TextPairExample:
    """One AI-generated source (+ optional human reference) for humanization."""

    source_text: str
    target_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _parse_records(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [r if isinstance(r, dict) else {"source_text": str(r)} for r in raw]
    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            return _parse_records(raw["data"])
        return [raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        # JSONL
        if "\n" in text and text.lstrip().startswith("{"):
            rows = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
            return rows
        # CSV
        if "," in text.splitlines()[0]:
            reader = csv.DictReader(text.splitlines())
            return [dict(row) for row in reader]
        try:
            return _parse_records(json.loads(text))
        except json.JSONDecodeError:
            return [{"source_text": line} for line in text.splitlines() if line.strip()]
    raise TypeError(f"Unsupported dataset payload type: {type(raw)!r}")


def records_to_examples(records: Sequence[Dict[str, Any]]) -> List[TextPairExample]:
    examples: List[TextPairExample] = []
    for row in records:
        source = (
            row.get("source_text")
            or row.get("source")
            or row.get("ai_text")
            or row.get("text")
            or row.get("input")
        )
        if source is None:
            raise KeyError(f"Dataset row missing source text field: {row!r}")
        target = row.get("target_text") or row.get("target") or row.get("human_text") or row.get("output")
        meta = {k: v for k, v in row.items() if k not in {
            "source_text", "source", "ai_text", "text", "input",
            "target_text", "target", "human_text", "output",
        }}
        examples.append(TextPairExample(source_text=str(source), target_text=str(target) if target else None, metadata=meta or None))
    return examples


class HumanizationDataset(Dataset):
    """Torch dataset of text pairs loaded through a DatasetRepository."""

    def __init__(self, examples: Sequence[TextPairExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TextPairExample:
        return self.examples[idx]

    @classmethod
    def from_repository(cls, repo: DatasetRepository, name: str) -> "HumanizationDataset":
        raw = repo.load(name)
        # Support path-like payloads saved as files via repo
        if isinstance(raw, (str, Path)) and Path(str(raw)).exists() and Path(str(raw)).is_file():
            path = Path(str(raw))
            if path.suffix == ".json":
                raw = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".jsonl":
                raw = path.read_text(encoding="utf-8")
            elif path.suffix == ".csv":
                raw = path.read_text(encoding="utf-8")
        records = _parse_records(raw)
        return cls(records_to_examples(records))

    @classmethod
    def from_records(cls, records: Sequence[Dict[str, Any]]) -> "HumanizationDataset":
        return cls(records_to_examples(records))


class BackendCollator:
    """Tokenize text pairs using ``BaseModelBackend`` (never a concrete backend)."""

    def __init__(
        self,
        backend: BaseModelBackend,
        max_input_length: int,
        max_output_length: int,
    ) -> None:
        self.backend = backend
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __call__(self, batch: Sequence[TextPairExample]) -> Dict[str, Any]:
        sources = [ex.source_text for ex in batch]
        encoded = self.backend.tokenize(
            sources,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
        )
        out: Dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded.get("attention_mask"),
            "source_texts": sources,
        }
        targets = [ex.target_text for ex in batch if ex.target_text]
        if len(targets) == len(batch):
            tgt = self.backend.tokenize(
                [ex.target_text or "" for ex in batch],
                max_length=self.max_output_length,
                padding=True,
                truncation=True,
            )
            out["decoder_input_ids"] = tgt["input_ids"]
            labels = tgt["input_ids"].clone()
            pad_id = getattr(self.backend.tokenizer, "pad_token_id", 0) or 0
            labels[labels == pad_id] = -100
            out["labels"] = labels
        return out


def build_dataloader(
    dataset: HumanizationDataset,
    backend: BaseModelBackend,
    batch_size: int,
    max_input_length: int,
    max_output_length: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader; tokenization goes through ``BaseModelBackend`` only."""
    collator = BackendCollator(backend, max_input_length, max_output_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
