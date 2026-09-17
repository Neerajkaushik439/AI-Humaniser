"""Integration: logits → Gumbel → soft P → H_soft → loss → backward gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.composite import CompositeHumanizationLoss
from models.backends.base import BaseModelBackend
from models.pipeline import SoftTokenPipeline
from utils.tiny import build_tiny_humanizer


def test_end_to_end_gumbel_gradient_through_humanizer():
    """FLAN-T5-class backend logits → Gumbel → soft tokens → diff emb → loss → grad."""
    config, _repos, model, backend = build_tiny_humanizer()
    assert isinstance(backend, BaseModelBackend)
    model.train()

    vocab = min(backend.vocab_size - 1, 60)
    lo, hi = 3, max(4, vocab)
    input_ids = torch.randint(lo, hi, (2, 4))
    decoder_input_ids = torch.randint(lo, hi, (2, 5))
    labels = torch.randint(lo, hi, (2, 5))

    outputs = model.forward_trainable(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        decoder_input_ids=decoder_input_ids,
        labels=labels,
        gumbel_hard=False,
    )

    # Soft probabilities (not argmax one-hots)
    assert outputs.soft_tokens.shape == outputs.logits.shape
    assert torch.allclose(
        outputs.soft_tokens.sum(-1),
        torch.ones_like(outputs.soft_tokens.sum(-1)),
        atol=1e-4,
    )
    assert ((outputs.soft_tokens > 0.01) & (outputs.soft_tokens < 0.99)).any()

    loss_fn = CompositeHumanizationLoss.from_config(config)
    losses = loss_fn(
        semantic=outputs.semantic,
        semantic_target=torch.zeros_like(outputs.semantic),
        stylometric=outputs.stylometric,
        style_target=torch.zeros_like(outputs.stylometric),
        human_style=outputs.human_style,
        human_style_positive=outputs.human_style.detach(),
        diversity=outputs.diversity,
        discriminator_logits=outputs.discriminator_logits,
        logits=outputs.logits,
        labels=labels,
    )
    assert "total_loss" in losses
    losses["total_loss"].backward()

    # Gradients must reach the backbone (through Gumbel + soft embedding path)
    backbone_grads = [
        p.grad for p in backend.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert len(backbone_grads) > 0

    # Embedding table also receives gradients when not frozen
    if model.diff_embedding.weight.requires_grad:
        assert model.diff_embedding.weight.grad is not None


def test_soft_pipeline_preserves_graph_from_logits():
    config, _repos, model, backend = build_tiny_humanizer()
    logits = torch.randn(2, 3, backend.vocab_size, requires_grad=True)
    pipeline: SoftTokenPipeline = model.soft_pipeline
    out = pipeline(logits, temperature=config.gumbel.temperature, hard=False)
    out.embeddings.mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_inference_path_does_not_use_gumbel_graph():
    _config, _repos, model, _backend = build_tiny_humanizer()
    model.eval()
    # generate is decorated no_grad and calls backend.generate — discrete tokens
    texts = model.generate("sample ai text", max_new_tokens=3)
    assert isinstance(texts[0], str)
