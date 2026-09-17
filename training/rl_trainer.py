"""Base RL trainer — algorithm-agnostic scaffolding.

Communicates with the policy exclusively through ``HumanizerModel.backend``
typed as ``BaseModelBackend`` — never imports FLANT5Backend.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from configs.config import AppConfig
from losses.composite import CompositeHumanizationLoss
from models.backends.base import BaseModelBackend
from models.humanizer import HumanizerModel, HumanizerOutput
from repositories.base import RepositoryBundle
from rewards.composite import CompositeReward
from training.checkpoint import CheckpointManager
from training.metrics import MetricsLogger
from training.trainer import BaseTrainer

logger = logging.getLogger("ai_humanizer.training.rl")


class BaseRLTrainer(BaseTrainer):
    """Shared RL loop utilities for PPO and REINFORCE."""

    def __init__(
        self,
        model: HumanizerModel,
        config: AppConfig,
        repositories: RepositoryBundle,
        reward_fn: Optional[CompositeReward] = None,
        loss_fn: Optional[CompositeHumanizationLoss] = None,
        dataloader: Any = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(model, config, repositories, loss_fn=loss_fn)
        self.reward_fn = reward_fn or CompositeReward.from_config(config)
        self.dataloader = dataloader
        self.dry_run = bool(dry_run)
        self.algorithm = config.rl.algorithm.lower()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.rl.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        warmup_steps = max(1, int(100 * config.training.warmup_ratio))
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)),
        )
        self.checkpoint_manager = CheckpointManager(
            repository=repositories.checkpoints,
            checkpoint_dir=config.project.checkpoint_dir,
        )
        self.metrics = MetricsLogger(log_dir=config.project.log_dir)
        self.global_step = 0
        self.epoch = 0

    @property
    def backend(self) -> BaseModelBackend:
        """Policy backbone — abstract interface only."""
        return self.model.backend

    def compute_rewards(self, **batch: Any) -> Dict[str, Any]:
        return self.reward_fn(**batch)

    def humanize_forward(
        self,
        batch: Dict[str, Any],
        gumbel_temperature: Optional[float] = None,
    ) -> HumanizerOutput:
        """Training forward: backend → Gumbel → soft embeddings → heads."""
        return self.model.forward_trainable(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            decoder_input_ids=batch.get("decoder_input_ids"),
            labels=batch.get("labels"),
            gumbel_temperature=gumbel_temperature,
        )

    def build_reward_batch(
        self,
        outputs: HumanizerOutput,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "semantic": outputs.semantic,
            "semantic_target": batch.get("semantic_target"),
            "stylometric": outputs.stylometric,
            "style_target": batch.get("style_target"),
            "human_style": outputs.human_style,
            "human_style_positive": batch.get("human_style_positive"),
            "diversity": outputs.diversity,
            "discriminator_logits": outputs.discriminator_logits,
            "logits": outputs.logits,
            "labels": batch.get("labels"),
            "backbone_loss": (outputs.extras or {}).get("backbone_loss"),
        }

    def build_loss_batch(
        self,
        outputs: HumanizerOutput,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "semantic": outputs.semantic,
            "semantic_target": batch.get("semantic_target"),
            "stylometric": outputs.stylometric,
            "style_target": batch.get("style_target"),
            "human_style": outputs.human_style,
            "human_style_positive": batch.get("human_style_positive"),
            "human_style_negative": batch.get("human_style_negative"),
            "diversity": outputs.diversity,
            "diversity_target": batch.get("diversity_target"),
            "discriminator_logits": outputs.discriminator_logits,
            "logits": outputs.logits,
            "labels": batch.get("labels"),
            "backbone_loss": (outputs.extras or {}).get("backbone_loss"),
        }

    def log_step(
        self,
        *,
        losses: Optional[Dict[str, Any]] = None,
        rewards: Optional[Dict[str, Any]] = None,
        rl_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        lr = float(self.optimizer.param_groups[0]["lr"])
        tau = float(self.model.gumbel.temperature)
        if losses:
            self.metrics.log_losses(losses, self.global_step)
        if rewards:
            self.metrics.log_rewards(rewards, self.global_step)
        self.metrics.log_training(
            step=self.global_step,
            learning_rate=lr,
            gumbel_temperature=tau,
        )
        if rl_metrics:
            self.metrics.log_rl(rl_metrics, self.global_step, algorithm=self.algorithm)

    def save_checkpoint(self, name: str) -> None:
        self.checkpoint_manager.save(
            name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.epoch,
            config=self.config,
            loss_weights=dict(self.config.losses),
            reward_weights=dict(self.config.rewards),
            rl_config=self.config.to_dict().get("rl", {}),
        )

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.model.parameters()).device
        moved = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        return moved

    def policy_log_probs(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Sequence log-probs via BaseModelBackend helper (backend-agnostic)."""
        return self.backend.log_probs_from_logits(logits, token_ids)

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        ...

    def close(self) -> None:
        self.metrics.close()
        super().close()


# Backward-compatible alias
RLTrainer = BaseRLTrainer
