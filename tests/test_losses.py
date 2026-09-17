"""Comprehensive tests for the humanization loss architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config
from losses import (
    LOSS_REGISTRY,
    AdversarialLoss,
    BaseLoss,
    CompositeHumanizationLoss,
    ContrastiveHumanLoss,
    DiversityLoss,
    PPLLoss,
    SemanticLoss,
    StyleLoss,
    register_loss,
    validate_loss_weights,
)
from losses.reference_lm import LogitsCrossEntropyReferenceLM
from models.discriminator import HumanAIDiscriminator
from models.diversity_module import DiversityBurstinessModule
from models.semantic_encoder import BaseSemanticEncoder, SemanticEncoder
from models.style_encoder import BaseHumanStyleEncoder, HumanStyleEncoder
from models.stylometric_module import StylometricModule


DEFAULT_WEIGHTS = {
    "lambda_semantic": 1.0,
    "lambda_style": 1.0,
    "lambda_contrastive": 1.0,
    "lambda_adversarial": 0.5,
    "lambda_diversity": 0.5,
    "lambda_ppl": 0.05,
}


def _batch(dim: int = 16, feature_dim: int = 16, proj: int = 8, batch: int = 4):
    return {
        "semantic": torch.randn(batch, dim, requires_grad=True),
        "semantic_target": torch.randn(batch, dim),
        "stylometric": torch.randn(batch, feature_dim, requires_grad=True),
        "style_target": torch.randn(batch, feature_dim),
        "human_style": torch.randn(batch, proj, requires_grad=True),
        "human_style_positive": torch.randn(batch, proj),
        "human_style_negative": torch.randn(batch, proj),
        "diversity": torch.randn(batch, 12, requires_grad=True),
        "diversity_target": torch.randn(batch, 12),
        "discriminator_logits": torch.randn(batch, requires_grad=True),
        "logits": torch.randn(batch, 5, 32, requires_grad=True),
        "labels": torch.randint(0, 32, (batch, 5)),
    }


# ---------------------------------------------------------------------------
# Independent losses
# ---------------------------------------------------------------------------


def test_semantic_loss_independent():
    loss = SemanticLoss()
    s = torch.randn(3, 8, requires_grad=True)
    t = torch.randn(3, 8)
    value = loss(semantic=s, semantic_target=t)
    assert value.ndim == 0
    value.backward()
    assert s.grad is not None


def test_style_loss_independent():
    loss = StyleLoss()
    s = torch.randn(3, 8, requires_grad=True)
    value = loss(stylometric=s, style_target=torch.randn(3, 8))
    value.backward()
    assert s.grad is not None


def test_contrastive_human_loss_independent():
    loss = ContrastiveHumanLoss(temperature=0.1)
    anchor = torch.randn(4, 8, requires_grad=True)
    value = loss(
        human_style=anchor,
        human_style_positive=torch.randn(4, 8),
        human_style_negative=torch.randn(4, 8),
    )
    assert value.ndim == 0
    value.backward()
    assert anchor.grad is not None


def test_adversarial_loss_independent():
    loss = AdversarialLoss()
    logits = torch.randn(5, requires_grad=True)
    value = loss(discriminator_logits=logits, fool_discriminator=True)
    value.backward()
    assert logits.grad is not None


def test_diversity_loss_independent():
    loss = DiversityLoss(target_burstiness=1.0)
    d = torch.randn(3, 10, requires_grad=True)
    value = loss(diversity=d)
    value.backward()
    assert d.grad is not None


def test_ppl_loss_independent_and_no_hardcoded_lambda():
    lm = LogitsCrossEntropyReferenceLM()
    loss = PPLLoss(reference_lm=lm)
    logits = torch.randn(2, 4, 20, requires_grad=True)
    labels = torch.randint(0, 20, (2, 4))
    value = loss(logits=logits, labels=labels)
    # PPLLoss returns raw NLL — weighting is only in CompositeHumanizationLoss
    assert value.ndim == 0
    value.backward()
    assert logits.grad is not None


def test_ppl_uses_reference_lm_abstraction():
    class ConstLM(LogitsCrossEntropyReferenceLM):
        def negative_log_likelihood(self, **kwargs):
            return torch.tensor(2.5, requires_grad=True)

    loss = PPLLoss(reference_lm=ConstLM())
    assert float(loss(logits=torch.randn(1, 2, 3)).detach()) == 2.5


# ---------------------------------------------------------------------------
# Encoder / module abstractions (replaceable)
# ---------------------------------------------------------------------------


def test_semantic_encoder_is_replaceable_abstraction():
    assert issubclass(SemanticEncoder, BaseSemanticEncoder)
    enc = SemanticEncoder(input_size=16, hidden_size=16, output_size=8)
    out = enc(torch.randn(2, 5, 16))
    assert out.shape == (2, 8)


def test_human_style_encoder_replaceable():
    assert issubclass(HumanStyleEncoder, BaseHumanStyleEncoder)
    enc = HumanStyleEncoder(input_size=16, hidden_size=16, projection_dim=8)
    out = enc(torch.randn(2, 5, 16))
    assert out.shape == (2, 8)


def test_stylometric_module_modular_features():
    mod = StylometricModule(input_size=16, feature_dim=8, hidden_size=16)
    emb = torch.randn(2, 6, 16)
    soft = torch.softmax(torch.randn(2, 6, 32), dim=-1)
    out = mod(emb, soft_tokens=soft)
    assert out.shape == (2, 8)
    assert len(mod.extractors) >= 3


def test_diversity_module_modular_features():
    mod = DiversityBurstinessModule(input_size=16, window_size=4, hidden_size=12)
    out = mod(torch.randn(2, 8, 16), soft_tokens=torch.softmax(torch.randn(2, 8, 20), dim=-1))
    assert out.shape == (2, 12)


def test_discriminator_replaceable():
    disc = HumanAIDiscriminator(input_size=16, hidden_size=16)
    logits = disc(torch.randn(3, 5, 16))
    assert logits.shape == (3,)


# ---------------------------------------------------------------------------
# Composite + registry + extensibility
# ---------------------------------------------------------------------------


def test_composite_loss_output_dictionary():
    composite = CompositeHumanizationLoss(weights=DEFAULT_WEIGHTS)
    out = composite(**_batch())
    assert "total_loss" in out
    for key in ("semantic", "style", "contrastive", "adversarial", "diversity", "ppl"):
        assert key in out
        assert torch.is_tensor(out[key])
    assert out["total_loss"].ndim == 0
    assert out["total_loss"].requires_grad


def test_composite_matches_manual_weighted_sum():
    weights = DEFAULT_WEIGHTS
    composite = CompositeHumanizationLoss(weights=weights)
    batch = _batch()
    out = composite(**batch)

    manual = (
        weights["lambda_semantic"] * SemanticLoss()(
            semantic=batch["semantic"], semantic_target=batch["semantic_target"]
        )
        + weights["lambda_style"] * StyleLoss()(
            stylometric=batch["stylometric"], style_target=batch["style_target"]
        )
        + weights["lambda_contrastive"] * ContrastiveHumanLoss()(
            human_style=batch["human_style"],
            human_style_positive=batch["human_style_positive"],
            human_style_negative=batch["human_style_negative"],
        )
        + weights["lambda_adversarial"] * AdversarialLoss()(
            discriminator_logits=batch["discriminator_logits"]
        )
        + weights["lambda_diversity"] * DiversityLoss()(
            diversity=batch["diversity"], diversity_target=batch["diversity_target"]
        )
        + weights["lambda_ppl"] * PPLLoss()(
            logits=batch["logits"], labels=batch["labels"]
        )
    )
    assert torch.allclose(out["total_loss"], manual, rtol=1e-4, atol=1e-5)


def test_zero_weight_loss_excluded_from_total_effect():
    weights = dict(DEFAULT_WEIGHTS)
    weights["lambda_style"] = 0.0
    composite = CompositeHumanizationLoss(weights=weights)
    batch = _batch()
    # Make style term huge if it were included
    batch["stylometric"] = batch["stylometric"] * 1000
    batch["style_target"] = torch.zeros_like(batch["stylometric"])
    out = composite(**batch)
    # Unweighted style still logged
    assert out["style"].item() > 0
    assert out["weighted_style"].item() == 0.0


def test_adding_new_loss_via_registry_without_rewriting_composite():
    @register_loss(name="new_loss")
    class NewLoss(BaseLoss):
        name = "new_loss"

        def forward(self, **batch):
            ref = batch.get("semantic")
            return ref.new_ones(()) if torch.is_tensor(ref) else torch.ones(())

    weights = dict(DEFAULT_WEIGHTS)
    weights["lambda_new_loss"] = 0.25
    composite = CompositeHumanizationLoss(weights=weights)
    assert "new_loss" in composite.losses
    out = composite(**_batch())
    assert "new_loss" in out
    assert abs(float(out["weighted_new_loss"]) - 0.25) < 1e-6
    LOSS_REGISTRY.unregister("new_loss")


def test_loss_registry_contains_architecture_terms():
    for name in ("semantic", "style", "contrastive", "adversarial", "diversity", "ppl"):
        assert name in LOSS_REGISTRY


def test_gradient_flow_through_composite():
    composite = CompositeHumanizationLoss(weights=DEFAULT_WEIGHTS)
    batch = _batch()
    out = composite(**batch)
    out["total_loss"].backward()
    assert batch["semantic"].grad is not None
    assert batch["stylometric"].grad is not None
    assert batch["human_style"].grad is not None
    assert batch["diversity"].grad is not None
    assert batch["discriminator_logits"].grad is not None
    assert batch["logits"].grad is not None


def test_from_config_uses_yaml_lambdas():
    cfg = load_config()
    composite = CompositeHumanizationLoss.from_config(cfg)
    assert composite.active_weights()["ppl"] == cfg.losses["lambda_ppl"]
    assert composite.active_weights()["ppl"] < composite.active_weights()["semantic"]


# ---------------------------------------------------------------------------
# Invalid configuration
# ---------------------------------------------------------------------------


def test_invalid_config_negative_lambda():
    with pytest.raises(ValueError, match="must be >= 0"):
        validate_loss_weights({"lambda_semantic": -1.0})


def test_invalid_config_bad_key():
    with pytest.raises(ValueError, match="lambda_"):
        validate_loss_weights({"semantic": 1.0})


def test_invalid_config_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_loss_weights({})


def test_invalid_config_non_numeric():
    with pytest.raises(TypeError):
        validate_loss_weights({"lambda_semantic": "high"})


def test_composite_rejects_nonzero_unregistered_loss():
    with pytest.raises(KeyError, match="not registered"):
        CompositeHumanizationLoss(
            weights={**DEFAULT_WEIGHTS, "lambda_ghost": 1.0},
            loss_names=["ghost", "semantic", "style", "contrastive", "adversarial", "diversity", "ppl"],
            allow_unknown_weights=True,
        )


def test_composite_allow_unknown_false():
    with pytest.raises(KeyError, match="Unknown loss weight"):
        validate_loss_weights({"lambda_not_a_real_loss": 0.1}, allow_unknown=False)
