"""Blog Authorship Corpus — short human posts (style)."""

from __future__ import annotations

from typing import Iterator, Optional

from data_sources.base import BaseDatasetSource, DatasetRecord
from data_sources.local_io import first_present, iter_local_json_rows, take


class BlogAuthorshipSource(BaseDatasetSource):
    name = "blog_authorship"

    def is_available(self) -> bool:
        return self.raw_dir.exists() and any(self.raw_dir.rglob("*"))

    def iter_records(
        self, split: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterator[DatasetRecord]:
        split = split or self.spec.split_default
        for row in take(limit, iter_local_json_rows(self.raw_dir)):
            text = first_present(row, ["text", "content", "post", "body", "human_text"])
            if text is None:
                continue
            yield DatasetRecord(
                human_text=text,
                label="human",
                source_dataset=self.name,
                split=split,
                metadata={"role": self.spec.role.value, **{k: v for k, v in row.items() if k not in {"text", "content", "post", "body", "human_text"}}},
            )
