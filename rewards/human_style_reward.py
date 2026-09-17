"""Human-style contrastive reward (toward human writing representations)."""

from __future__ import annotations

from typing import Any, Optional

import torch.nn.functional as F
from torch import Tensor

from rewards.base import BaseRewardComponent, register_reward


@register_reward
class HumanStyleReward(BaseRewardComponent):
    """Reward higher cosine similarity to human style positives."""

    name = "human_style"

    def forward(
        self,
        human_style: Tensor,
        human_style_positive: Optional[Tensor] = None,
        **_: Any,
    ) -> Tensor:
        if human_style_positive is None:
            return human_style.new_zeros(human_style.size(0))
        anchor = F.normalize(human_style, dim=-1)
        positive = F.normalize(human_style_positive, dim=-1)
        return (anchor * positive).sum(dim=-1)
