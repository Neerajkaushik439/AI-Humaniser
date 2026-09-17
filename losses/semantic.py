"""Semantic loss — L_semantic (λ1).

Preserves meaning of humanized text vs. source semantics.
Encoder lives in ``models.semantic_encoder`` and is replaceable;
this module only compares semantic vectors.
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn.functional as F
from torch import Tensor

from losses.base import BaseLoss, register_loss


@register_loss
class SemanticLoss(BaseLoss):
    name = "semantic"

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        semantic: Tensor,
        semantic_target: Optional[Tensor] = None,
        **_: Any,
    ) -> Tensor:
        if semantic_target is None:
            return semantic.new_zeros(())
        if semantic.shape != semantic_target.shape:
            raise ValueError(
                f"semantic shape {tuple(semantic.shape)} != "
                f"semantic_target shape {tuple(semantic_target.shape)}"
            )
        return F.mse_loss(semantic, semantic_target, reduction=self.reduction)
