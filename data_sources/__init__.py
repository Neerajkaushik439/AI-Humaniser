"""Corpus adapters for planned AI-Humanizer training datasets."""

from data_sources.base import BaseDatasetSource, DatasetRecord, DatasetRole, DatasetSpec
from data_sources.catalog import (
    SOURCE_CLASSES,
    build_source,
    export_to_repository,
    future_pair_construction_plan,
    list_all_datasets,
    list_enabled_datasets,
    load_catalog,
    parse_specs,
)

__all__ = [
    "SOURCE_CLASSES",
    "BaseDatasetSource",
    "DatasetRecord",
    "DatasetRole",
    "DatasetSpec",
    "build_source",
    "export_to_repository",
    "future_pair_construction_plan",
    "list_all_datasets",
    "list_enabled_datasets",
    "load_catalog",
    "parse_specs",
]
