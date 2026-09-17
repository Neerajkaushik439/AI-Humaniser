"""Gumbel-Softmax soft token distribution.

Converts Transformer logits → differentiable soft one-hot token distributions
exactly as specified in the architecture diagram.

Temperature MUST come from configuration (or an explicit override / scheduler).
Never hard-code τ in call sites.
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemperatureSchedule(Protocol):
    """Callable schedule: ``tau = schedule(step)`` for annealing support."""

    def __call__(self, step: int) -> float: ...


class ConstantTemperatureSchedule:
    def __init__(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def __call__(self, step: int) -> float:
        return self.temperature


class LinearAnnealingTemperatureSchedule:
    """Linearly anneal temperature from ``start`` → ``end`` over ``total_steps``."""

    def __init__(self, start: float, end: float, total_steps: int) -> None:
        if start <= 0 or end <= 0:
            raise ValueError("temperatures must be > 0")
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        self.start = float(start)
        self.end = float(end)
        self.total_steps = int(total_steps)

    def __call__(self, step: int) -> float:
        if step >= self.total_steps:
            return self.end
        ratio = max(0.0, min(1.0, step / float(self.total_steps)))
        return self.start + (self.end - self.start) * ratio


def sample_gumbel(shape: torch.Size, device: torch.device, eps: float = 1e-10) -> Tensor:
    u = torch.rand(shape, device=device, dtype=torch.float32)
    return -torch.log(-torch.log(u.clamp_min(eps)) + eps)


def gumbel_softmax(
    logits: Tensor,
    temperature: float,
    hard: bool = False,
    dim: int = -1,
) -> Tensor:
    """Standard Gumbel-Softmax (Jang et al. / Maddison et al.).

    Soft mode returns a proper probability distribution over the vocabulary.
    Hard mode uses a straight-through estimator (one-hot forward, soft backward).
    Training path for soft embeddings should use ``hard=False`` (no argmax).
    """
    if temperature <= 0:
        raise ValueError(f"Gumbel temperature must be > 0, got {temperature}")

    # Compute in float32 for numerical stability, cast back to logits dtype.
    logits_f = logits.float()
    gumbel_noise = sample_gumbel(logits_f.shape, device=logits_f.device)
    y_soft = F.softmax((logits_f + gumbel_noise) / temperature, dim=dim)
    y_soft = y_soft.to(dtype=logits.dtype)

    if hard:
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
        # Straight-through: forward = hard, backward = soft
        return (y_hard - y_soft).detach() + y_soft
    return y_soft


class GumbelSoftmax(nn.Module):
    """Module wrapper with configurable τ / hard flag + optional scheduling."""

    def __init__(
        self,
        temperature: float = 1.0,
        hard: bool = False,
        schedule: Optional[TemperatureSchedule] = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        # Registered buffers so state_dict / device moves keep τ with the module,
        # while still allowing in-place schedule updates.
        self.register_buffer("_temperature", torch.tensor(float(temperature)))
        self.hard = bool(hard)
        self.schedule = schedule
        self._step = 0

    @property
    def temperature(self) -> float:
        return float(self._temperature.item())

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"temperature must be > 0, got {value}")
        self._temperature.fill_(float(value))

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def set_schedule(self, schedule: TemperatureSchedule) -> None:
        self.schedule = schedule

    def step_schedule(self, step: Optional[int] = None) -> float:
        """Advance (or set) the schedule step and update temperature."""
        if step is not None:
            self._step = int(step)
        else:
            self._step += 1
        if self.schedule is not None:
            self.set_temperature(self.schedule(self._step))
        return self.temperature

    def current_temperature(self, temperature: Optional[float] = None) -> float:
        if temperature is not None:
            return float(temperature)
        return self.temperature

    def forward(
        self,
        logits: Tensor,
        temperature: Optional[float] = None,
        hard: Optional[bool] = None,
    ) -> Tensor:
        """
        Args:
            logits: token logits ``[..., vocab]``
            temperature: optional override (else configured / scheduled τ)
            hard: optional override of soft vs straight-through hard mode
        Returns:
            Soft (or ST-hard) distribution over vocabulary, same shape as logits.
        """
        tau = self.current_temperature(temperature)
        use_hard = self.hard if hard is None else hard
        return gumbel_softmax(logits, temperature=tau, hard=use_hard)

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}, hard={self.hard}, step={self._step}"
