"""PAR-3 — paraphrase pairs (DIPPER-related meaning preservation)."""

from __future__ import annotations

from typing import Iterator, Optional

from data_sources.base import BaseDatasetSource, DatasetRecord
from data_sources.local_io import first_present, iter_local_json_rows, take


class PAR3Source(BaseDatasetSource):
    name = "par3"

    def is_available(self) -> bool:
        return self.raw_dir.exists() and any(
            p.is_file() and not p.name.startswith(".") for p in self.raw_dir.rglob("*")
        )

    def iter_records(
        self, split: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterator[DatasetRecord]:
        split = split or self.spec.split_default
        for row in take(limit, iter_local_json_rows(self.raw_dir)):
            # Paraphrase corpora vary; keep both sides when present.
            left = first_present(row, ["source", "original", "text1", "input", "src"])
            right = first_present(row, ["paraphrase", "target", "text2", "output", "tgt", "rewrite"])
            if left is None and right is None:
                text = first_present(row, ["text", "content"])
                if text is None:
                    continue
                yield DatasetRecord(
                    text=text,
                    source_dataset=self.name,
                    split=split,
                    metadata={"role": self.spec.role.value},
                )
                continue
            # For semantic preservation: treat rewrite as AI-like source, original as human-like target
            # when both exist — provisional until true AI→humanized pairs are built.
            yield DatasetRecord(
                ai_text=right or left,
                human_text=left if right else None,
                text=left or right,
                source_dataset=self.name,
                split=split,
                metadata={"role": self.spec.role.value, "pair_type": "paraphrase"},
            )
