"""Discriminator-based human-likeness reward."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rewards.base import BaseRewardComponent, REWARD_REGISTRY, register_reward


@register_reward
class DiscriminatorReward(BaseRewardComponent):
    """P(human) from the Human/AI discriminator logits."""

    name = "discriminator"

    def forward(self, discriminator_logits: Tensor, **_: Any) -> Tensor:
        return torch.sigmoid(discriminator_logits)


# Config historically uses lambda_human for discriminator human-likeness.
REWARD_REGISTRY.register(DiscriminatorReward, name="human")
HumanReward = DiscriminatorReward
