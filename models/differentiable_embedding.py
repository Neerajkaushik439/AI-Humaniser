"""Differentiable Embedding Layer.

Given soft token distribution ``P`` and embedding matrix ``E``:

    H_soft = P @ E

Preserves gradient flow:

    loss → H_soft → P (Gumbel-Softmax) → transformer logits

Do **not** use argmax on the training path — pass the soft distribution directly.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


class DifferentiableEmbedding(nn.Module):
    """Compute ``H_soft = P E`` for soft token distributions ``P``."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight: Optional[Tensor] = None,
        freeze: bool = False,
        padding_idx: Optional[int] = None,
        shared_weight: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        if shared_weight is not None:
            if shared_weight.shape != (num_embeddings, embedding_dim):
                raise ValueError(
                    f"shared_weight shape {tuple(shared_weight.shape)} != "
                    f"{(num_embeddings, embedding_dim)}"
                )
            # Tie to an existing Parameter (e.g. backbone input embeddings).
            self.weight = shared_weight
            if freeze:
                self.weight.requires_grad_(False)
        elif weight is not None:
            if weight.shape != (num_embeddings, embedding_dim):
                raise ValueError(
                    f"Expected weight shape {(num_embeddings, embedding_dim)}, "
                    f"got {tuple(weight.shape)}"
                )
            self.weight = nn.Parameter(weight.clone().detach(), requires_grad=not freeze)
        else:
            self.weight = nn.Parameter(
                torch.empty(num_embeddings, embedding_dim),
                requires_grad=not freeze,
            )
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        if padding_idx is not None and shared_weight is None:
            with torch.no_grad():
                self.weight[padding_idx].zero_()

    @classmethod
    def from_pretrained(
        cls,
        embedding: nn.Embedding,
        freeze: bool = False,
        share_weights: bool = False,
    ) -> "DifferentiableEmbedding":
        """Initialize from an ``nn.Embedding`` (optionally weight-tied)."""
        if share_weights:
            return cls(
                num_embeddings=embedding.num_embeddings,
                embedding_dim=embedding.embedding_dim,
                shared_weight=embedding.weight,
                freeze=freeze,
                padding_idx=embedding.padding_idx,
            )
        return cls(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            weight=embedding.weight.data,
            freeze=freeze,
            padding_idx=embedding.padding_idx,
        )

    @classmethod
    def from_matrix(
        cls,
        embedding_matrix: Union[Tensor, nn.Parameter],
        freeze: bool = False,
        share_weights: bool = False,
    ) -> "DifferentiableEmbedding":
        """Initialize from a raw ``[V, D]`` matrix."""
        if embedding_matrix.ndim != 2:
            raise ValueError(f"embedding_matrix must be 2D [V, D], got {tuple(embedding_matrix.shape)}")
        num_embeddings, embedding_dim = embedding_matrix.shape
        if share_weights:
            if not isinstance(embedding_matrix, nn.Parameter):
                embedding_matrix = nn.Parameter(embedding_matrix)
            return cls(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                shared_weight=embedding_matrix,
                freeze=freeze,
            )
        return cls(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            weight=embedding_matrix.detach(),
            freeze=freeze,
        )

    def forward(self, soft_tokens: Tensor) -> Tensor:
        """
        Args:
            soft_tokens: soft distribution ``P`` with shape ``[batch, seq, vocab]``.
                Must NOT be argmax-discretized on the training path.
        Returns:
            Differentiable embeddings ``H_soft`` with shape ``[batch, seq, embedding_dim]``.
        """
        if soft_tokens.ndim != 3:
            raise ValueError(
                f"soft_tokens must be [batch, seq, vocab], got shape {tuple(soft_tokens.shape)}"
            )
        if soft_tokens.size(-1) != self.num_embeddings:
            raise ValueError(
                f"soft_tokens vocab dim {soft_tokens.size(-1)} != "
                f"embedding table size {self.num_embeddings}"
            )
        # H_soft = P E  — keeps the graph: loss → H → P → logits
        return soft_tokens @ self.weight

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"freeze={not self.weight.requires_grad}"
        )
