"""Tests for main dry-run verification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dry_run import run_dry_run


def test_full_stack_dry_run():
    report = run_dry_run()
    assert report.ok, f"Dry-run failed checks: {report.errors}"
    required = {
        "configuration",
        "model_backend",
        "tokenizer",
        "gumbel_softmax",
        "differentiable_embedding",
        "composite_loss",
        "composite_reward",
        "rl_trainer",
        "checkpoint_manager",
        "logging",
        "gradient_flow",
        "training_no_argmax",
        "inference_generation",
    }
    assert required.issubset(report.checks.keys())
    assert all(report.checks[k] for k in required)
