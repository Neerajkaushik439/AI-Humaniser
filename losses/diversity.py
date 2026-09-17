"""Diversity / Burstiness loss — L_diversity (λ5).

Features come from ``DiversityBurstinessModule`` (pluggable extractors).
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn.functional as F
from torch import Tensor

from losses.base import BaseLoss, register_loss


@register_loss
class DiversityLoss(BaseLoss):
    """Burstiness / diversity term (equation name: L_diversity)."""

    name = "diversity"

    def __init__(self, target_burstiness: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.target_burstiness = float(target_burstiness)
        self.reduction = reduction

    def forward(
        self,
        diversity: Tensor,
        diversity_target: Optional[Tensor] = None,
        **_: Any,
    ) -> Tensor:
        if diversity_target is not None:
            return F.mse_loss(diversity, diversity_target, reduction=self.reduction)
        # Encourage non-collapsed diversity representations
        norms = diversity.norm(dim=-1)
        target = norms.new_full(norms.shape, self.target_burstiness)
        return F.mse_loss(norms, target, reduction=self.reduction)
