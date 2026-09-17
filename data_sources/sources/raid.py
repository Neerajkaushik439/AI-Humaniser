"""RAID — AI vs human detection corpus."""

from __future__ import annotations

from typing import Iterator, Optional

from data_sources.base import BaseDatasetSource, DatasetRecord
from data_sources.local_io import first_present, iter_local_json_rows, take


class RAIDSource(BaseDatasetSource):
    name = "raid"

    def is_available(self) -> bool:
        return self.raw_dir.exists() and any(
            p.is_file() and not p.name.startswith(".") for p in self.raw_dir.rglob("*")
        )

    def iter_records(
        self, split: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterator[DatasetRecord]:
        split = split or self.spec.split_default
        for row in take(limit, iter_local_json_rows(self.raw_dir)):
            text = first_present(row, ["text", "content", "generation", "document"])
            label = first_present(row, ["label", "source", "model", "generator"])
            if text is None:
                continue
            normalized = None
            if label is not None:
                low = label.lower()
                if low in {"human", "humans", "0"}:
                    normalized = "human"
                elif "ai" in low or "gpt" in low or "model" in low or low in {"1", "machine"}:
                    normalized = "ai"
                else:
                    normalized = low
            rec = DatasetRecord(
                text=text,
                label=normalized,
                source_dataset=self.name,
                split=split,
                metadata={"role": self.spec.role.value, "raw_label": label},
            )
            if normalized == "human":
                rec.human_text = text
            elif normalized == "ai":
                rec.ai_text = text
            yield rec
