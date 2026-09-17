"""Reward interfaces for the RL stage (PPO / REINFORCE).

Trainers depend on ``BaseRewardComponent`` / ``CompositeReward`` only —
reward formulas are never hard-coded inside PPO / REINFORCE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Type

import torch.nn as nn
from torch import Tensor


class BaseRewardComponent(ABC, nn.Module):
    """Single named reward term returning per-sample scores ``[batch]``."""

    name: str = "base"

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, **batch: Any) -> Tensor:
        """Return per-sample rewards ``[batch]``."""

    def extra_repr(self) -> str:
        return f"name={self.name}"


# Backward-compatible alias
BaseReward = BaseRewardComponent


class RewardRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseRewardComponent]] = {}

    def register(
        self, cls: Type[BaseRewardComponent], name: Optional[str] = None
    ) -> Type[BaseRewardComponent]:
        key = (name or cls.name).lower().strip()
        if not key or key == "base":
            raise ValueError(f"Invalid reward registry name: {key!r}")
        self._registry[key] = cls
        return cls

    def unregister(self, name: str) -> None:
        self._registry.pop(name.lower(), None)

    def get(self, name: str) -> Type[BaseRewardComponent]:
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Reward {name!r} is not registered. Known: {sorted(self._registry)}")
        return self._registry[key]

    def create(self, name: str, **kwargs: Any) -> BaseRewardComponent:
        return self.get(name)(**kwargs)

    def available(self) -> Dict[str, Type[BaseRewardComponent]]:
        return dict(self._registry)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._registry


REWARD_REGISTRY = RewardRegistry()


def register_reward(
    cls: Optional[Type[BaseRewardComponent]] = None, *, name: Optional[str] = None
):
    def decorator(reward_cls: Type[BaseRewardComponent]) -> Type[BaseRewardComponent]:
        REWARD_REGISTRY.register(reward_cls, name=name)
        return reward_cls

    if cls is not None:
        return decorator(cls)
    return decorator


def resolve_reward_weight(weights: Mapping[str, float], name: str, default: float = 0.0) -> float:
    if name in weights:
        return float(weights[name])
    key = name if name.startswith("lambda_") else f"lambda_{name}"
    return float(weights.get(key, default))


def validate_reward_weights(weights: Mapping[str, Any]) -> Dict[str, float]:
    if not isinstance(weights, Mapping) or len(weights) == 0:
        raise ValueError("reward weights must be a non-empty mapping of lambda_* keys")
    cleaned: Dict[str, float] = {}
    for key, value in weights.items():
        if not isinstance(key, str) or not key.startswith("lambda_"):
            raise ValueError(f"Invalid reward weight key {key!r}; expected 'lambda_<name>'")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Reward weight {key} must be numeric")
        value_f = float(value)
        if value_f < 0:
            raise ValueError(f"Reward weight {key} must be >= 0")
        cleaned[key] = value_f
    return cleaned
