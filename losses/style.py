"""Style loss — L_style (λ2).

Compares stylometric feature vectors. Feature extraction is handled by
``models.stylometric_module.StylometricModule`` (pluggable features).
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn.functional as F
from torch import Tensor

from losses.base import BaseLoss, register_loss


@register_loss
class StyleLoss(BaseLoss):
    name = "style"

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        stylometric: Tensor,
        style_target: Optional[Tensor] = None,
        **_: Any,
    ) -> Tensor:
        if style_target is None:
            return stylometric.new_zeros(())
        if stylometric.shape != style_target.shape:
            raise ValueError(
                f"stylometric shape {tuple(stylometric.shape)} != "
                f"style_target shape {tuple(style_target.shape)}"
            )
        return F.mse_loss(stylometric, style_target, reduction=self.reduction)
