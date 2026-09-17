"""Smoke tests for configuration and loss extensibility."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config
from losses.base import LOSS_REGISTRY, BaseLoss, register_loss
from losses.composite import CompositeHumanizationLoss
from repositories import build_repositories


def test_load_default_config():
    cfg = load_config()
    assert cfg.model_backend == "flan_t5"
    assert cfg.gumbel_temperature > 0
    assert "lambda_ppl" in cfg.losses
    assert cfg.losses["lambda_ppl"] < cfg.losses["lambda_semantic"]


def test_repositories_health():
    cfg = load_config()
    repos = build_repositories(
        kind="memory",
        data_dir=cfg.repositories.data_dir,
        checkpoint_dir=cfg.repositories.checkpoint_dir,
        experiment_dir=cfg.repositories.experiment_dir,
    )
    assert all(repos.healthcheck().values())


def test_composite_loss_discovers_new_lambda():
    @register_loss(name="new_loss")
    class NewLoss(BaseLoss):
        name = "new_loss"

        def forward(self, **batch):
            import torch

            return torch.tensor(1.0)

    weights = {
        "lambda_semantic": 1.0,
        "lambda_style": 1.0,
        "lambda_contrastive": 1.0,
        "lambda_adversarial": 0.5,
        "lambda_diversity": 0.5,
        "lambda_ppl": 0.05,
        "lambda_new_loss": 0.25,
    }
    composite = CompositeHumanizationLoss(weights=weights)
    assert "new_loss" in composite.losses
    assert composite.active_weights()["new_loss"] == 0.25
    LOSS_REGISTRY.unregister("new_loss")
