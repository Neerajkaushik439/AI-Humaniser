"""Tests for DifferentiableEmbedding and gradient flow through the soft path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.differentiable_embedding import DifferentiableEmbedding
from models.gumbel_softmax import GumbelSoftmax
from models.pipeline import SoftTokenPipeline


def test_differentiable_embedding_shape():
    batch, seq, vocab, dim = 2, 5, 32, 16
    emb = DifferentiableEmbedding(num_embeddings=vocab, embedding_dim=dim)
    soft = torch.softmax(torch.randn(batch, seq, vocab), dim=-1)
    out = emb(soft)
    assert out.shape == (batch, seq, dim)


def test_h_soft_equals_p_times_e():
    vocab, dim = 20, 8
    weight = torch.randn(vocab, dim)
    emb = DifferentiableEmbedding(num_embeddings=vocab, embedding_dim=dim, weight=weight)
    p = torch.softmax(torch.randn(3, 4, vocab), dim=-1)
    h = emb(p)
    expected = p @ emb.weight
    assert torch.allclose(h, expected)


def test_gradient_propagation_logits_to_loss():
    """loss → H_soft → P → logits must retain a gradient on logits."""
    batch, seq, vocab, dim = 2, 4, 24, 12
    logits = torch.randn(batch, seq, vocab, requires_grad=True)
    gumbel = GumbelSoftmax(temperature=1.0, hard=False)
    emb = DifferentiableEmbedding(num_embeddings=vocab, embedding_dim=dim)

    soft = gumbel(logits)  # soft mode — no argmax
    h_soft = emb(soft)
    loss = h_soft.pow(2).mean()
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0
    assert emb.weight.grad is not None


def test_soft_pipeline_gradient_flow():
    vocab, dim = 16, 8
    pipeline = SoftTokenPipeline(
        gumbel=GumbelSoftmax(temperature=0.8, hard=False),
        embedding=DifferentiableEmbedding(num_embeddings=vocab, embedding_dim=dim),
    )
    logits = torch.randn(2, 3, vocab, requires_grad=True)
    out = pipeline(logits, hard=False)
    assert out.soft_tokens.shape == (2, 3, vocab)
    assert out.embeddings.shape == (2, 3, dim)
    out.embeddings.sum().backward()
    assert logits.grad is not None


def test_shared_weight_tying():
    base = nn.Embedding(10, 4)
    emb = DifferentiableEmbedding.from_pretrained(base, share_weights=True)
    assert emb.weight is base.weight
