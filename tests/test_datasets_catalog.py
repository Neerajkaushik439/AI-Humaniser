"""Tests for planned training dataset catalog and source adapters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config
from data_sources import (
    build_source,
    export_to_repository,
    future_pair_construction_plan,
    list_all_datasets,
    list_enabled_datasets,
    load_catalog,
    parse_specs,
)
from repositories import build_repositories


EXPECTED = {
    "blog_authorship",
    "pg19",
    "raid",
    "mage",
    "par3",
    "hc3",
}


def test_catalog_lists_planned_datasets():
    names = set(list_all_datasets())
    assert EXPECTED.issubset(names)
    assert set(list_enabled_datasets()) == EXPECTED


def test_catalog_roles_match_plan():
    specs = parse_specs()
    assert specs["blog_authorship"].role.value == "human_style"
    assert specs["pg19"].role.value == "human_style_long"
    assert specs["raid"].role.value == "ai_vs_human"
    assert specs["mage"].role.value == "ai_vs_human"
    assert specs["par3"].role.value == "paraphrase"
    assert specs["hc3"].role.value == "ai_human_pairs_rough"


def test_future_pair_construction_disabled():
    plan = future_pair_construction_plan()
    assert plan.get("enabled") is False
    assert plan.get("strategy") == "human_to_ai_rewrite"


def test_app_config_includes_datasets_section():
    cfg = load_config()
    assert cfg.datasets.catalog_path.endswith("datasets.yaml")
    assert cfg.datasets.pair_construction.enabled is False


def test_hc3_source_reads_local_json(tmp_path):
    raw = tmp_path / "raw" / "hc3"
    raw.mkdir(parents=True)
    (raw / "sample.json").write_text(
        json.dumps(
            [
                {
                    "question": "What is gravity?",
                    "human_answers": ["Gravity pulls things down."],
                    "chatgpt_answers": ["Gravity is a force that attracts masses."],
                }
            ]
        ),
        encoding="utf-8",
    )
    source = build_source("hc3", data_root=tmp_path)
    assert source.is_available()
    rows = source.to_training_rows(limit=1)
    assert rows[0]["source_text"].startswith("Gravity is a force")
    assert rows[0]["target_text"].startswith("Gravity pulls")


def test_blog_authorship_human_only(tmp_path):
    raw = tmp_path / "raw" / "blog_authorship"
    raw.mkdir(parents=True)
    (raw / "posts.jsonl").write_text(
        '{"text": "My weekend was quiet."}\n',
        encoding="utf-8",
    )
    source = build_source("blog_authorship", data_root=tmp_path)
    recs = source.load_records(limit=1)
    assert recs[0].human_text == "My weekend was quiet."
    assert recs[0].label == "human"


def test_export_to_memory_repository(tmp_path):
    raw = tmp_path / "raw" / "raid"
    raw.mkdir(parents=True)
    (raw / "a.jsonl").write_text(
        '{"text": "AI wrote this.", "label": "ai"}\n'
        '{"text": "A person wrote this.", "label": "human"}\n',
        encoding="utf-8",
    )
    repos = build_repositories("memory", str(tmp_path), "ckpt", "exp")
    key = export_to_repository("raid", repos.datasets, data_root=tmp_path, limit=10)
    loaded = repos.datasets.load(key)
    assert len(loaded) == 2
