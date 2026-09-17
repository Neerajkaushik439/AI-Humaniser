"""TensorBoard metrics logging for losses, rewards, and training/RL stats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("ai_humanizer.training.metrics")


def _to_float(value: Union[float, int, Tensor]) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().cpu().item() if value.ndim == 0 else value.detach().mean().cpu())
    return float(value)


class MetricsLogger:
    """Structured TensorBoard logger with the required metric namespaces."""

    def __init__(self, log_dir: Union[str, Path], enabled: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.writer: Optional[SummaryWriter] = None
        if enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info("TensorBoard logging to %s", self.log_dir)

    def log_scalar(self, tag: str, value: Union[float, int, Tensor], step: int) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar(tag, _to_float(value), step)

    def log_losses(self, losses: Mapping[str, Union[float, Tensor]], step: int) -> None:
        """Log loss/total, loss/semantic, ..."""
        mapping = {
            "total_loss": "loss/total",
            "total": "loss/total",
            "semantic": "loss/semantic",
            "style": "loss/style",
            "contrastive": "loss/contrastive",
            "adversarial": "loss/adversarial",
            "diversity": "loss/diversity",
            "ppl": "loss/ppl",
        }
        logged_total = False
        for key, tag in mapping.items():
            if key not in losses:
                continue
            if tag == "loss/total" and logged_total:
                continue
            self.log_scalar(tag, losses[key], step)
            if tag == "loss/total":
                logged_total = True

    def log_rewards(self, rewards: Mapping[str, Union[float, Tensor]], step: int) -> None:
        """Log reward/total, reward/semantic, ..."""
        mapping = {
            "total_reward": "reward/total",
            "total": "reward/total",
            "semantic": "reward/semantic",
            "style": "reward/style",
            "human": "reward/human",
            "human_style": "reward/human_style",
            "diversity": "reward/diversity",
            "discriminator": "reward/discriminator",
            "ppl": "reward/ppl",
        }
        logged_total = False
        for key, tag in mapping.items():
            if key not in rewards:
                continue
            if tag == "reward/total" and logged_total:
                continue
            value = rewards[key]
            # Per-sample rewards → mean for scalar log
            self.log_scalar(tag, value, step)
            if tag == "reward/total":
                logged_total = True

    def log_training(
        self,
        *,
        step: int,
        learning_rate: Optional[float] = None,
        gumbel_temperature: Optional[float] = None,
        **extra: Union[float, Tensor],
    ) -> None:
        if learning_rate is not None:
            self.log_scalar("training/learning_rate", learning_rate, step)
        if gumbel_temperature is not None:
            self.log_scalar("training/gumbel_temperature", gumbel_temperature, step)
        for key, value in extra.items():
            self.log_scalar(f"training/{key}", value, step)

    def log_rl(self, metrics: Mapping[str, Union[float, Tensor]], step: int, algorithm: str) -> None:
        for key, value in metrics.items():
            self.log_scalar(f"rl/{algorithm}/{key}", value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
