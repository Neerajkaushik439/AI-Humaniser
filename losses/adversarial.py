"""Adversarial loss — L_adversarial (λ4).

Generator-side objective against ``HumanAIDiscriminator`` (replaceable).
Discriminator training can use the same class with ``fool_discriminator=False``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.base import BaseLoss, register_loss


@register_loss
class AdversarialLoss(BaseLoss):
    name = "adversarial"

    def forward(
        self,
        discriminator_logits: Tensor,
        disc_labels: Optional[Tensor] = None,
        fool_discriminator: bool = True,
        **_: Any,
    ) -> Tensor:
        if disc_labels is None:
            if fool_discriminator:
                disc_labels = torch.ones_like(discriminator_logits)
            else:
                return discriminator_logits.new_zeros(())
        return F.binary_cross_entropy_with_logits(
            discriminator_logits,
            disc_labels.float(),
        )
