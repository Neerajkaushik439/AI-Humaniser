"""Semantic reward component."""

from __future__ import annotations

from typing import Any, Optional

import torch.nn.functional as F
from torch import Tensor

from rewards.base import BaseRewardComponent, register_reward


@register_reward
class SemanticReward(BaseRewardComponent):
    name = "semantic"

    def forward(
        self,
        semantic: Tensor,
        semantic_target: Optional[Tensor] = None,
        **_: Any,
    ) -> Tensor:
        if semantic_target is None:
            return semantic.new_zeros(semantic.size(0))
        dist = F.mse_loss(semantic, semantic_target, reduction="none").mean(dim=-1)
        return -dist
