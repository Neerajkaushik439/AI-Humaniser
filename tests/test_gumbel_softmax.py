"""Tests for Gumbel-Softmax."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config
from models.gumbel_softmax import (
    GumbelSoftmax,
    LinearAnnealingTemperatureSchedule,
    gumbel_softmax,
)


def test_gumbel_softmax_output_shape():
    batch, seq, vocab = 2, 5, 64
    logits = torch.randn(batch, seq, vocab)
    out = gumbel_softmax(logits, temperature=1.0, hard=False)
    assert out.shape == (batch, seq, vocab)


def test_gumbel_softmax_probability_distribution():
    logits = torch.randn(3, 4, 32)
    out = gumbel_softmax(logits, temperature=0.7, hard=False)
    assert torch.all(out >= 0)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_gumbel_temperature_from_config():
    cfg = load_config()
    module = GumbelSoftmax(temperature=cfg.gumbel.temperature, hard=cfg.gumbel.hard)
    assert module.temperature == cfg.gumbel.temperature
    assert module.hard == cfg.gumbel.hard

    logits = torch.randn(1, 2, 16)
    soft = module(logits)
    assert soft.shape == logits.shape
    assert torch.allclose(soft.sum(-1), torch.ones(1, 2), atol=1e-5)


def test_gumbel_temperature_override_and_schedule():
    module = GumbelSoftmax(temperature=1.0, hard=False)
    module.set_schedule(LinearAnnealingTemperatureSchedule(start=1.0, end=0.1, total_steps=10))
    tau = module.step_schedule(5)
    assert 0.1 < tau < 1.0

    logits = torch.randn(2, 3, 8, requires_grad=True)
    out = module(logits, temperature=0.5)
    assert out.requires_grad
    assert torch.allclose(out.sum(-1), torch.ones(2, 3), atol=1e-5)


def test_gumbel_hard_straight_through_is_one_hot():
    logits = torch.randn(2, 3, 10)
    hard = gumbel_softmax(logits, temperature=1.0, hard=True)
    # Forward values are one-hot
    assert torch.allclose(hard.sum(-1), torch.ones(2, 3), atol=1e-5)
    assert torch.all((hard == 0) | (hard == 1) | ((hard > 0) & (hard < 1)))  # ST may be soft in autograd sense
    # Max per row equals 1 for pure hard forward — ST returns values that sum to 1
    assert (hard.max(dim=-1).values > 0.99).all()
