"""Style reward component."""

from __future__ import annotations

from typing import Any, Optional

import torch.nn.functional as F
from torch import Tensor

from rewards.base import BaseRewardComponent, register_reward


@register_reward
class StyleReward(BaseRewardComponent):
    name = "style"

    def forward(
        self,
        stylometric: Tensor,
        style_target: Optional[Tensor] = None,
        **_: Any,
    ) -> Tensor:
        if style_target is None:
            return stylometric.new_zeros(stylometric.size(0))
        dist = F.mse_loss(stylometric, style_target, reduction="none").mean(dim=-1)
        return -dist
