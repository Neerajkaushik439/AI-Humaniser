"""HC3 — rough ChatGPT vs human answer pairs (older, imperfect)."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

from data_sources.base import BaseDatasetSource, DatasetRecord
from data_sources.local_io import first_present, iter_local_json_rows, take


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    return [str(value)] if str(value).strip() else []


class HC3Source(BaseDatasetSource):
    name = "hc3"

    def is_available(self) -> bool:
        return self.raw_dir.exists() and any(
            p.is_file() and not p.name.startswith(".") for p in self.raw_dir.rglob("*")
        )

    def iter_records(
        self, split: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterator[DatasetRecord]:
        split = split or self.spec.split_default
        count = 0
        for row in take(None, iter_local_json_rows(self.raw_dir)):
            humans = _as_list(row.get("human_answers") or row.get("human") or row.get("human_text"))
            ais = _as_list(row.get("chatgpt_answers") or row.get("ai_answers") or row.get("ai") or row.get("ai_text"))
            question = first_present(row, ["question", "query", "prompt"])

            if humans and ais:
                # Emit pairwise combinations (bounded) as rough AI→human pairs.
                for ai in ais[:3]:
                    for human in humans[:3]:
                        if limit is not None and count >= limit:
                            return
                        yield DatasetRecord(
                            ai_text=ai,
                            human_text=human,
                            source_dataset=self.name,
                            split=split,
                            metadata={
                                "role": self.spec.role.value,
                                "hf_id": self.spec.hf_id,
                                "question": question,
                                "pair_quality": "rough_hc3",
                            },
                        )
                        count += 1
                continue

            text = first_present(row, ["text", "answer", "content"])
            if text is None:
                continue
            if limit is not None and count >= limit:
                return
            yield DatasetRecord(
                text=text,
                source_dataset=self.name,
                split=split,
                metadata={"role": self.spec.role.value, "question": question},
            )
            count += 1
