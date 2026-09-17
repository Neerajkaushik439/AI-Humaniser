"""Contrastive human loss — L_contrastive (λ3).

Pulls generated style reps toward human writing reps (InfoNCE).
Encoder: ``models.style_encoder.HumanStyleEncoder`` (replaceable).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.base import BaseLoss, register_loss


@register_loss
class ContrastiveHumanLoss(BaseLoss):
    name = "contrastive"

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("contrastive temperature must be > 0")
        self.temperature = float(temperature)

    def forward(
        self,
        human_style: Tensor,
        human_style_positive: Optional[Tensor] = None,
        human_style_negative: Optional[Tensor] = None,
        **_: Any,
    ) -> Tensor:
        if human_style_positive is None:
            return human_style.new_zeros(())

        anchor = F.normalize(human_style, dim=-1)
        positive = F.normalize(human_style_positive, dim=-1)

        if human_style_negative is None:
            # In-batch negatives: diagonal is the positive pair
            logits = anchor @ positive.transpose(0, 1) / self.temperature
            labels = torch.arange(anchor.size(0), device=anchor.device)
            return F.cross_entropy(logits, labels)

        pos_sim = (anchor * positive).sum(dim=-1, keepdim=True) / self.temperature
        negative = F.normalize(human_style_negative, dim=-1)
        neg_sim = anchor @ negative.transpose(0, 1) / self.temperature
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)
