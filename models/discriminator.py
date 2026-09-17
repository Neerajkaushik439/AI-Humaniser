"""Human / AI Discriminator — independently replaceable adversarial head."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseDiscriminator(ABC, nn.Module):
    """Binary human (1) vs AI (0) discriminator over soft embeddings."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Return logits ``[batch]`` (pre-sigmoid)."""


class MLPDiscriminator(BaseDiscriminator):
    def __init__(self, input_size: int, hidden_size: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def pool(self, embeddings: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if attention_mask is None:
            return embeddings.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        pooled = self.pool(embeddings, attention_mask)
        return self.net(pooled).squeeze(-1)


class HumanAIDiscriminator(MLPDiscriminator):
    """Default discriminator (replaceable via BaseDiscriminator)."""
