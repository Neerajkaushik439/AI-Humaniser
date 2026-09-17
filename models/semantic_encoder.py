"""Semantic Encoder abstractions — replaceable, not tied to FLAN-T5."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseSemanticEncoder(ABC, nn.Module):
    """Encode soft embeddings into a semantic representation for L_semantic.

    Implementations may later wrap Sentence-Transformers, frozen encoders, etc.
    They must NOT assume a specific Transformer backbone.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Return semantic vectors ``[batch, semantic_dim]``."""


class MeanPoolSemanticEncoder(BaseSemanticEncoder):
    """Lightweight MLP semantic encoder over pooled differentiable embeddings."""

    def __init__(self, input_size: int, hidden_size: int = 768, output_size: int = 768) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        )

    def pool(self, embeddings: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if attention_mask is None:
            return embeddings.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        summed = (embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        pooled = self.pool(embeddings, attention_mask)
        return self.encoder(pooled)


# Default concrete encoder used by HumanizerModel (name preserved).
class SemanticEncoder(MeanPoolSemanticEncoder):
    """Default semantic encoder (replaceable via BaseSemanticEncoder)."""
