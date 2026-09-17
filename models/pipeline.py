"""Core differentiable token pipeline (training path).

    token logits
      → Gumbel-Softmax
      → soft token distribution P
      → Differentiable Embedding  (H_soft = P E)

Inference / discrete generation intentionally lives on the backend
(``BaseModelBackend.generate``) and is not routed through this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.differentiable_embedding import DifferentiableEmbedding
from models.gumbel_softmax import GumbelSoftmax


@dataclass
class SoftTokenPipelineOutput:
    """Outputs of the differentiable training path (pre-downstream modules)."""

    logits: Tensor  # [B, T, V]
    soft_tokens: Tensor  # [B, T, V]  — probability distribution P
    embeddings: Tensor  # [B, T, D]  — H_soft = P E
    temperature: float


class SoftTokenPipeline(nn.Module):
    """Gumbel-Softmax + Differentiable Embedding stage."""

    def __init__(
        self,
        gumbel: GumbelSoftmax,
        embedding: DifferentiableEmbedding,
    ) -> None:
        super().__init__()
        self.gumbel = gumbel
        self.embedding = embedding

    def forward(
        self,
        logits: Tensor,
        temperature: Optional[float] = None,
        hard: Optional[bool] = None,
    ) -> SoftTokenPipelineOutput:
        """
        Training path only. Does not call ``argmax`` when ``hard`` is False
        (the default from config).
        """
        if hard is None:
            hard = False  # training default: soft distribution, never argmax
        soft_tokens = self.gumbel(logits, temperature=temperature, hard=hard)
        embeddings = self.embedding(soft_tokens)
        tau = self.gumbel.current_temperature(temperature)
        return SoftTokenPipelineOutput(
            logits=logits,
            soft_tokens=soft_tokens,
            embeddings=embeddings,
            temperature=tau,
        )
