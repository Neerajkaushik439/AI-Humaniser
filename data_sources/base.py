"""Dataset source abstractions for AI-Humanizer corpora.

Named ``data_sources`` (not ``datasets``) to avoid clashing with the
Hugging Face ``datasets`` package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class DatasetRole(str, Enum):
    """How a corpus is used in the humanization pipeline."""

    HUMAN_STYLE = "human_style"
    HUMAN_STYLE_LONG = "human_style_long"
    AI_VS_HUMAN = "ai_vs_human"
    PARAPHRASE = "paraphrase"
    AI_HUMAN_PAIRS_ROUGH = "ai_human_pairs_rough"
    CUSTOM = "custom"


@dataclass
class DatasetSpec:
    """Declarative description of a training corpus (from catalog YAML)."""

    name: str
    role: DatasetRole
    description: str
    enabled: bool = True
    hf_id: Optional[str] = None
    local_subdir: Optional[str] = None
    split_default: str = "train"
    notes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetRecord:
    """Normalized row used across heterogeneous corpora."""

    text: Optional[str] = None
    ai_text: Optional[str] = None
    human_text: Optional[str] = None
    label: Optional[str] = None  # e.g. "ai" | "human"
    source_dataset: Optional[str] = None
    split: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_training_row(self) -> Dict[str, Any]:
        """Map into fields understood by ``training.data.records_to_examples``.

        Humanization convention:
          source_text = AI text (to humanize)
          target_text = human reference (optional)
        """
        row: Dict[str, Any] = {"metadata": dict(self.metadata)}
        if self.source_dataset:
            row["source_dataset"] = self.source_dataset
        if self.split:
            row["split"] = self.split
        if self.label:
            row["label"] = self.label

        if self.ai_text is not None and self.human_text is not None:
            row["source_text"] = self.ai_text
            row["target_text"] = self.human_text
            row["ai_text"] = self.ai_text
            row["human_text"] = self.human_text
            return row

        if self.ai_text is not None:
            row["source_text"] = self.ai_text
            row["ai_text"] = self.ai_text
            return row

        if self.human_text is not None:
            # Human-only corpora: pairing (AI rewrite) comes later.
            row["human_text"] = self.human_text
            row["text"] = self.human_text
            row["source_text"] = self.human_text
            return row

        if self.text is not None:
            row["source_text"] = self.text
            row["text"] = self.text
            return row

        raise ValueError("DatasetRecord has no usable text fields")


class BaseDatasetSource(ABC):
    """Adapter for one external/local corpus. No training logic here."""

    name: str = "base"

    def __init__(self, spec: DatasetSpec, data_root: Path | str = "data") -> None:
        self.spec = spec
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw" / (spec.local_subdir or spec.name)
        self.processed_dir = self.data_root / "processed" / (spec.local_subdir or spec.name)

    @abstractmethod
    def is_available(self) -> bool:
        """True if local files are present (HF download is opt-in later)."""

    @abstractmethod
    def iter_records(
        self, split: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterator[DatasetRecord]:
        ...

    def load_records(
        self, split: Optional[str] = None, limit: Optional[int] = None
    ) -> List[DatasetRecord]:
        return list(self.iter_records(split=split, limit=limit))

    def to_training_rows(
        self, split: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return [r.to_training_row() for r in self.iter_records(split=split, limit=limit)]

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
