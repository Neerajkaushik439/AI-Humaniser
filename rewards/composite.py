"""Composite RL reward — config-weighted combination of reward components.

PPO / REINFORCE must call this module rather than hard-coding a reward formula.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch import Tensor

from rewards.base import (
    REWARD_REGISTRY,
    BaseRewardComponent,
    resolve_reward_weight,
    validate_reward_weights,
)

from rewards import diversity_reward as _diversity  # noqa: F401
from rewards import human_reward as _human  # noqa: F401
from rewards import human_style_reward as _human_style  # noqa: F401
from rewards import ppl_reward as _ppl  # noqa: F401
from rewards import semantic_reward as _semantic  # noqa: F401
from rewards import style_reward as _style  # noqa: F401


DEFAULT_REWARD_NAMES: List[str] = [
    "semantic",
    "style",
    "human_style",
    "diversity",
    "discriminator",
]


class CompositeReward(torch.nn.Module):
    def __init__(
        self,
        weights: Mapping[str, float],
        reward_names: Optional[Iterable[str]] = None,
        extra_rewards: Optional[Mapping[str, BaseRewardComponent]] = None,
        *,
        validate: bool = True,
    ) -> None:
        super().__init__()
        self.weights = validate_reward_weights(weights) if validate else dict(weights)
        names = list(reward_names) if reward_names is not None else list(DEFAULT_REWARD_NAMES)

        # Map legacy lambda_human → discriminator component if discriminator not listed
        if "lambda_human" in self.weights and "human" not in names and "discriminator" in names:
            # weight resolved via alias "human" → DiscriminatorReward also registered as human
            pass
        if "lambda_human" in self.weights and "discriminator" in names and "human" not in names:
            # ensure human alias participates when only discriminator is in the name list
            # DiscriminatorReward is created once as "discriminator"; weight key lambda_human
            # is resolved by also checking name "human" — handled in forward via dual keys.
            pass

        for key in self.weights:
            if key.startswith("lambda_"):
                suffix = key[len("lambda_") :]
                if suffix not in names and suffix in REWARD_REGISTRY:
                    names.append(suffix)

        modules: Dict[str, BaseRewardComponent] = {}
        # Avoid double-counting discriminator aliases (human ↔ discriminator)
        skip = set()
        if "discriminator" in names and "human" in names:
            skip.add("human")
        for name in names:
            if name in skip:
                continue
            if name in REWARD_REGISTRY:
                modules[name] = REWARD_REGISTRY.create(name)
        if extra_rewards:
            modules.update(extra_rewards)
        if not modules:
            raise ValueError("CompositeReward has no reward components")
        self.rewards = torch.nn.ModuleDict(modules)

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "CompositeReward":
        weights = config.rewards if hasattr(config, "rewards") else config
        return cls(weights=weights, **kwargs)

    def _weight_for(self, name: str) -> float:
        w = resolve_reward_weight(self.weights, name, default=0.0)
        # Allow lambda_human to weight the discriminator component
        if w == 0.0 and name == "discriminator":
            w = resolve_reward_weight(self.weights, "human", default=0.0)
        if w == 0.0 and name == "human":
            w = resolve_reward_weight(self.weights, "discriminator", default=0.0)
        return w

    def forward(self, **batch: Any) -> Dict[str, Tensor]:
        components: Dict[str, Tensor] = {}
        total = None
        for name, fn in self.rewards.items():
            value = fn(**batch)
            weight = self._weight_for(name)
            weighted = weight * value
            components[name] = value.detach()
            components[f"weighted_{name}"] = weighted.detach()
            # Also expose "human" alias for logging when using discriminator
            if name == "discriminator":
                components["human"] = value.detach()
                components["weighted_human"] = weighted.detach()
            total = weighted if total is None else total + weighted
        if total is None:
            total = torch.zeros(1)
        components["total"] = total
        components["total_reward"] = total
        return components

    def active_weights(self) -> Dict[str, float]:
        return {name: self._weight_for(name) for name in self.rewards.keys()}
