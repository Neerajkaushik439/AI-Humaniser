"""Base trainer — depends on config, repositories, and loss/reward interfaces."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter

from configs.config import AppConfig
from losses.composite import CompositeHumanizationLoss
from models.humanizer import HumanizerModel
from repositories.base import RepositoryBundle

logger = logging.getLogger("ai_humanizer.training")


class BaseTrainer(ABC):
    def __init__(
        self,
        model: HumanizerModel,
        config: AppConfig,
        repositories: RepositoryBundle,
        loss_fn: Optional[CompositeHumanizationLoss] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.repositories = repositories
        self.loss_fn = loss_fn or CompositeHumanizationLoss.from_config(config)
        self.writer: Optional[SummaryWriter] = None

    def setup_logging(self) -> None:
        log_dir = self.config.project.log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoard logging to %s", log_dir)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train") -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, step)

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run training. Concrete trainers implement the loop; not invoked by main.py."""

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
