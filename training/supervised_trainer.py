"""Supervised multi-loss trainer (pre-RL stage).

Does not start training when imported — ``train()`` must be called explicitly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW

from configs.config import AppConfig
from losses.composite import CompositeHumanizationLoss
from models.humanizer import HumanizerModel
from repositories.base import RepositoryBundle
from training.checkpoint import CheckpointManager
from training.metrics import MetricsLogger
from training.trainer import BaseTrainer

logger = logging.getLogger("ai_humanizer.training.supervised")


class SupervisedTrainer(BaseTrainer):
    def __init__(
        self,
        model: HumanizerModel,
        config: AppConfig,
        repositories: RepositoryBundle,
        loss_fn: Optional[CompositeHumanizationLoss] = None,
        dataloader: Any = None,
    ) -> None:
        super().__init__(model, config, repositories, loss_fn=loss_fn)
        self.dataloader = dataloader
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.checkpoint_manager = CheckpointManager(
            repository=repositories.checkpoints,
            checkpoint_dir=config.project.checkpoint_dir,
        )
        self.metrics = MetricsLogger(log_dir=config.project.log_dir)
        self.global_step = 0

    def train(self) -> Dict[str, Any]:
        if self.dataloader is None:
            raise RuntimeError(
                "SupervisedTrainer.train() requires a dataloader. "
                "Wire a DatasetRepository-backed loader before training."
            )

        self.setup_logging()
        self.model.train()
        global_step = 0
        history: Dict[str, Any] = {"steps": []}

        for epoch in range(self.config.training.epochs):
            for batch in self.dataloader:
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    decoder_input_ids=batch.get("decoder_input_ids"),
                    labels=batch.get("labels"),
                )
                loss_batch = {
                    "semantic": outputs.semantic,
                    "stylometric": outputs.stylometric,
                    "human_style": outputs.human_style,
                    "diversity": outputs.diversity,
                    "discriminator_logits": outputs.discriminator_logits,
                    "logits": outputs.logits,
                    "labels": batch.get("labels"),
                    "backbone_loss": (outputs.extras or {}).get("backbone_loss"),
                    "semantic_target": batch.get("semantic_target"),
                    "style_target": batch.get("style_target"),
                    "human_style_positive": batch.get("human_style_positive"),
                    "human_style_negative": batch.get("human_style_negative"),
                    "diversity_target": batch.get("diversity_target"),
                }
                losses = self.loss_fn(**loss_batch)
                total = losses.get("total_loss", losses["total"])
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm,
                )
                self.optimizer.step()

                metrics = {
                    k: float(v.detach().cpu())
                    for k, v in losses.items()
                    if torch.is_tensor(v) and k in {
                        "total_loss", "total", "semantic", "style", "contrastive",
                        "adversarial", "diversity", "ppl",
                    }
                }
                self.metrics.log_losses(losses, global_step)
                self.metrics.log_training(
                    step=global_step,
                    learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    gumbel_temperature=float(self.model.gumbel.temperature),
                )
                self.log_metrics(metrics, global_step)
                if global_step % max(1, self.config.training.logging_steps) == 0:
                    logger.info(
                        "step=%s epoch=%s total_loss=%.4f ppl=%.4f adv=%.4f div=%.4f",
                        global_step,
                        epoch + 1,
                        metrics.get("total_loss", metrics.get("total", 0.0)),
                        metrics.get("ppl", 0.0),
                        metrics.get("adversarial", 0.0),
                        metrics.get("diversity", 0.0),
                    )
                    print(
                        f"[train] step={global_step} epoch={epoch + 1} "
                        f"total_loss={metrics.get('total_loss', metrics.get('total', 0.0)):.4f} "
                        f"ppl={metrics.get('ppl', 0.0):.4f} "
                        f"adv={metrics.get('adversarial', 0.0):.4f}",
                        flush=True,
                    )
                history["steps"].append(metrics)
                global_step += 1
                if (
                    self.config.training.save_steps > 0
                    and global_step % self.config.training.save_steps == 0
                ):
                    self.checkpoint_manager.save(
                        f"supervised_step_{global_step}",
                        model=self.model,
                        optimizer=self.optimizer,
                        step=global_step,
                        epoch=epoch,
                        config=self.config,
                        loss_weights=dict(self.config.losses),
                        reward_weights=dict(self.config.rewards),
                        rl_config=self.config.to_dict().get("rl", {}),
                    )

            logger.info("Completed supervised epoch %s", epoch + 1)

        self.checkpoint_manager.save(
            "supervised_last",
            model=self.model,
            optimizer=self.optimizer,
            step=global_step,
            epoch=self.config.training.epochs,
            config=self.config,
            loss_weights=dict(self.config.losses),
            reward_weights=dict(self.config.rewards),
            rl_config=self.config.to_dict().get("rl", {}),
        )
        self.metrics.close()
        self.close()
        return history
