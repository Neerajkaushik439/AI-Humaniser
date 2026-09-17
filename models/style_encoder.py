"""Human Style Encoder — replaceable encoder for contrastive human loss."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseHumanStyleEncoder(ABC, nn.Module):
    """Project soft embeddings into a contrastive style space."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        normalize: bool = True,
    ) -> Tensor:
        """Return style vectors ``[batch, projection_dim]``."""


class ProjectionHumanStyleEncoder(BaseHumanStyleEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 768,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()
        self.projection_dim = projection_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_dim),
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
        normalize: bool = True,
    ) -> Tensor:
        pooled = self.pool(embeddings, attention_mask)
        projected = self.encoder(pooled)
        if normalize:
            return F.normalize(projected, p=2, dim=-1)
        return projected


class HumanStyleEncoder(ProjectionHumanStyleEncoder):
    """Default human style encoder (replaceable via BaseHumanStyleEncoder)."""
