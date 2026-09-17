"""Load dataset catalog from YAML and build source adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from data_sources.base import BaseDatasetSource, DatasetRole, DatasetSpec
from data_sources.sources.blog_authorship import BlogAuthorshipSource
from data_sources.sources.hc3 import HC3Source
from data_sources.sources.mage import MAGESource
from data_sources.sources.par3 import PAR3Source
from data_sources.sources.pg19 import PG19Source
from data_sources.sources.raid import RAIDSource

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = ROOT / "configs" / "datasets.yaml"

SOURCE_CLASSES = {
    "blog_authorship": BlogAuthorshipSource,
    "pg19": PG19Source,
    "raid": RAIDSource,
    "mage": MAGESource,
    "par3": PAR3Source,
    "hc3": HC3Source,
}

ROLE_MAP = {
    "human_style": DatasetRole.HUMAN_STYLE,
    "human_style_long": DatasetRole.HUMAN_STYLE_LONG,
    "ai_vs_human": DatasetRole.AI_VS_HUMAN,
    "paraphrase": DatasetRole.PARAPHRASE,
    "ai_human_pairs_rough": DatasetRole.AI_HUMAN_PAIRS_ROUGH,
    "custom": DatasetRole.CUSTOM,
}


def load_catalog(path: Optional[Path | str] = None) -> Dict[str, Any]:
    catalog_path = Path(path) if path else DEFAULT_CATALOG_PATH
    with catalog_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError(f"Dataset catalog at {catalog_path} must contain a 'datasets' mapping.")
    return data


def parse_specs(catalog: Optional[Mapping[str, Any]] = None) -> Dict[str, DatasetSpec]:
    raw = catalog if catalog is not None else load_catalog()
    datasets = raw.get("datasets") or {}
    specs: Dict[str, DatasetSpec] = {}
    for name, cfg in datasets.items():
        if not isinstance(cfg, Mapping):
            raise TypeError(f"Dataset entry {name!r} must be a mapping")
        role_key = str(cfg.get("role", "custom"))
        role = ROLE_MAP.get(role_key, DatasetRole.CUSTOM)
        specs[name] = DatasetSpec(
            name=name,
            role=role,
            description=str(cfg.get("description", "")),
            enabled=bool(cfg.get("enabled", True)),
            hf_id=cfg.get("hf_id"),
            local_subdir=cfg.get("local_subdir") or name,
            split_default=str(cfg.get("split_default", "train")),
            notes=list(cfg.get("notes") or []),
            extra={
                k: v
                for k, v in cfg.items()
                if k
                not in {
                    "role",
                    "description",
                    "enabled",
                    "hf_id",
                    "local_subdir",
                    "split_default",
                    "notes",
                }
            },
        )
    return specs


def build_source(
    name: str,
    data_root: Path | str = "data",
    catalog_path: Optional[Path | str] = None,
) -> BaseDatasetSource:
    specs = parse_specs(load_catalog(catalog_path))
    if name not in specs:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(specs)}")
    if name not in SOURCE_CLASSES:
        raise KeyError(
            f"No source adapter registered for {name!r}. "
            f"Known adapters: {sorted(SOURCE_CLASSES)}"
        )
    return SOURCE_CLASSES[name](specs[name], data_root=data_root)


def list_enabled_datasets(catalog_path: Optional[Path | str] = None) -> List[str]:
    return [n for n, s in parse_specs(load_catalog(catalog_path)).items() if s.enabled]


def list_all_datasets(catalog_path: Optional[Path | str] = None) -> List[str]:
    return sorted(parse_specs(load_catalog(catalog_path)))


def future_pair_construction_plan(catalog_path: Optional[Path | str] = None) -> Dict[str, Any]:
    catalog = load_catalog(catalog_path)
    return dict(catalog.get("future_pair_construction") or {})


def export_to_repository(
    name: str,
    repository,
    *,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    data_root: Path | str = "data",
    processed_name: Optional[str] = None,
) -> str:
    """Materialize source rows into a DatasetRepository entry (JSON-friendly list).

    Does not download remote corpora. Requires local files under data/raw/<name>/.
    """
    source = build_source(name, data_root=data_root)
    if not source.is_available():
        raise FileNotFoundError(
            f"Dataset '{name}' has no local files under {source.raw_dir}. "
            "Place JSON/JSONL/CSV/TXT there before exporting."
        )
    rows = source.to_training_rows(split=split, limit=limit)
    key = processed_name or name
    repository.save(key, rows)
    return key
