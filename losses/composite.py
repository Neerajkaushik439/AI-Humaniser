"""Composite humanization objective matching the architecture equation:

    L = λ1 L_semantic + λ2 L_style + λ3 L_contrastive
      + λ4 L_adversarial + λ5 L_diversity + λ6 L_PPL

Weights come exclusively from configuration. Extra ``lambda_*`` keys
automatically instantiate matching registered losses — trainers never need
rewriting.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch import Tensor

from losses.base import (
    LOSS_REGISTRY,
    BaseLoss,
    resolve_weight,
    validate_loss_weights,
    weight_key,
)

# Import concrete losses so they self-register on package import.
from losses import adversarial as _adversarial  # noqa: F401
from losses import contrastive as _contrastive  # noqa: F401
from losses import diversity as _diversity  # noqa: F401
from losses import perplexity as _perplexity  # noqa: F401
from losses import semantic as _semantic  # noqa: F401
from losses import style as _style  # noqa: F401


DEFAULT_LOSS_NAMES: List[str] = [
    "semantic",
    "style",
    "contrastive",
    "adversarial",
    "diversity",
    "ppl",
]


class CompositeHumanizationLoss(torch.nn.Module):
    """Dynamically combine registered loss components with config λ weights.

    Returns a dict with ``total_loss`` plus every individual (unweighted) term
    for logging / experimentation.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        loss_names: Optional[Iterable[str]] = None,
        extra_losses: Optional[Mapping[str, BaseLoss]] = None,
        *,
        validate: bool = True,
        allow_unknown_weights: bool = True,
        loss_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        super().__init__()
        if validate:
            self.weights = validate_loss_weights(
                weights,
                allow_unknown=allow_unknown_weights,
            )
        else:
            self.weights = {str(k): float(v) for k, v in weights.items()}

        names = list(loss_names) if loss_names is not None else list(DEFAULT_LOSS_NAMES)

        # Discover additional λ_* keys for future losses (plugin extensibility).
        for key in self.weights:
            if not key.startswith("lambda_"):
                continue
            suffix = key[len("lambda_") :]
            if suffix not in names and suffix in LOSS_REGISTRY:
                names.append(suffix)

        kwargs_map = dict(loss_kwargs or {})
        modules: Dict[str, BaseLoss] = {}
        for name in names:
            if name in LOSS_REGISTRY:
                modules[name] = LOSS_REGISTRY.create(name, **kwargs_map.get(name, {}))
            elif name in (extra_losses or {}):
                continue
            elif resolve_weight(self.weights, name, default=0.0) != 0.0:
                raise KeyError(
                    f"Loss {name!r} has non-zero weight but is not registered. "
                    f"Registered: {sorted(LOSS_REGISTRY.available())}"
                )
        if extra_losses:
            for name, loss_fn in extra_losses.items():
                modules[name] = loss_fn

        if not modules:
            raise ValueError("CompositeHumanizationLoss has no loss components to combine.")

        self.losses = torch.nn.ModuleDict(modules)

    @classmethod
    def from_config(
        cls,
        config: Any,
        **kwargs: Any,
    ) -> "CompositeHumanizationLoss":
        """Build from ``AppConfig`` (uses ``config.losses`` λ map)."""
        weights = config.losses if hasattr(config, "losses") else config
        return cls(weights=weights, **kwargs)

    def forward(self, **batch: Any) -> Dict[str, Tensor]:
        components: Dict[str, Tensor] = {}
        total: Optional[Tensor] = None

        for name, loss_fn in self.losses.items():
            value = loss_fn(**batch)
            if not torch.is_tensor(value):
                raise TypeError(f"Loss {name!r} must return a Tensor, got {type(value)!r}")
            weight = resolve_weight(self.weights, name, default=0.0)
            weighted = value * weight
            # Unweighted terms for logging; gradient flows through total_loss only.
            components[name] = value.detach()
            components[f"weighted_{name}"] = weighted.detach()
            total = weighted if total is None else total + weighted

        assert total is not None
        components["total_loss"] = total
        components["total"] = total  # backward-compatible alias for trainers
        return components

    def active_weights(self) -> Dict[str, float]:
        return {name: resolve_weight(self.weights, name) for name in self.losses.keys()}

    def configured_lambda_keys(self) -> List[str]:
        return [weight_key(name) for name in self.losses.keys()]


# Backward-compatible alias used by existing trainers / bootstrap.
CompositeLoss = CompositeHumanizationLoss
