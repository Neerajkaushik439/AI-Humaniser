"""Diversity / burstiness reward component."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from rewards.base import BaseRewardComponent, register_reward


@register_reward
class DiversityReward(BaseRewardComponent):
    name = "diversity"

    def forward(self, diversity: Tensor, **_: Any) -> Tensor:
        return diversity.norm(dim=-1)
