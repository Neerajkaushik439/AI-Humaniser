"""Loss interfaces and plugin-style registry.

Trainers depend only on ``BaseLoss`` / ``LossRegistry`` / ``CompositeHumanizationLoss``.
Adding a future term:

1. ``@register_loss`` a new ``BaseLoss`` subclass
2. Add ``lambda_new_loss`` in config YAML

No trainer / model / RL rewrites required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Type

import torch.nn as nn
from torch import Tensor


class BaseLoss(ABC, nn.Module):
    """Single named loss term.

    ``name`` must match the config suffix (e.g. ``name='semantic'`` → ``lambda_semantic``).
    """

    name: str = "base"

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, **batch: Any) -> Tensor:
        """Compute a scalar loss tensor (unweighted)."""

    def extra_repr(self) -> str:
        return f"name={self.name}"


class LossRegistry:
    """Global registry mapping loss names → classes."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseLoss]] = {}

    def register(self, cls: Type[BaseLoss], name: Optional[str] = None) -> Type[BaseLoss]:
        key = (name or cls.name).lower().strip()
        if not key or key == "base":
            raise ValueError(f"Invalid loss registry name: {key!r}")
        self._registry[key] = cls
        return cls

    def unregister(self, name: str) -> None:
        self._registry.pop(name.lower(), None)

    def get(self, name: str) -> Type[BaseLoss]:
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Loss {name!r} is not registered. Known: {sorted(self._registry)}")
        return self._registry[key]

    def create(self, name: str, **kwargs: Any) -> BaseLoss:
        return self.get(name)(**kwargs)

    def available(self) -> Dict[str, Type[BaseLoss]]:
        return dict(self._registry)

    def clear(self) -> None:
        self._registry.clear()

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._registry

    def __len__(self) -> int:
        return len(self._registry)


LOSS_REGISTRY = LossRegistry()


def register_loss(cls: Optional[Type[BaseLoss]] = None, *, name: Optional[str] = None):
    """Decorator: ``@register_loss`` or ``@register_loss(name='foo')``."""

    def decorator(loss_cls: Type[BaseLoss]) -> Type[BaseLoss]:
        LOSS_REGISTRY.register(loss_cls, name=name)
        return loss_cls

    if cls is not None:
        return decorator(cls)
    return decorator


def resolve_weight(weights: Mapping[str, float], loss_name: str, default: float = 0.0) -> float:
    """Resolve ``lambda_<name>`` (or raw name) from a config weight map."""
    if loss_name in weights:
        return float(weights[loss_name])
    key = loss_name if loss_name.startswith("lambda_") else f"lambda_{loss_name}"
    return float(weights.get(key, default))


def weight_key(loss_name: str) -> str:
    return loss_name if loss_name.startswith("lambda_") else f"lambda_{loss_name}"


def validate_loss_weights(
    weights: Mapping[str, Any],
    *,
    require_registered: bool = False,
    allow_unknown: bool = True,
) -> Dict[str, float]:
    """Validate λ configuration.

    Raises:
        ValueError / TypeError on invalid configuration.
    """
    if not isinstance(weights, Mapping):
        raise TypeError(f"loss weights must be a mapping, got {type(weights)!r}")
    if len(weights) == 0:
        raise ValueError("loss weights mapping is empty")

    cleaned: Dict[str, float] = {}
    for key, value in weights.items():
        if not isinstance(key, str) or not key.startswith("lambda_"):
            raise ValueError(
                f"Invalid loss weight key {key!r}. Expected keys like 'lambda_semantic'."
            )
        suffix = key[len("lambda_") :]
        if not suffix:
            raise ValueError(f"Invalid loss weight key {key!r}: missing loss name suffix.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Weight {key} must be a number, got {type(value)!r}")
        value_f = float(value)
        if value_f < 0:
            raise ValueError(f"Weight {key} must be >= 0, got {value_f}")
        if require_registered and suffix not in LOSS_REGISTRY and not allow_unknown:
            raise KeyError(f"Weight {key} refers to unregistered loss {suffix!r}")
        if not allow_unknown and suffix not in LOSS_REGISTRY:
            raise KeyError(
                f"Unknown loss weight {key}. Registered: {sorted(LOSS_REGISTRY.available())}"
            )
        cleaned[key] = value_f
    return cleaned
