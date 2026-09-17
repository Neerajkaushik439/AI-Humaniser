"""REINFORCE trainer — config-selected alternative to PPO.

Policy access goes through ``BaseModelBackend`` only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from configs.config import AppConfig
from losses.composite import CompositeHumanizationLoss
from models.humanizer import HumanizerModel
from repositories.base import RepositoryBundle
from rewards.composite import CompositeReward
from training.rl_trainer import BaseRLTrainer

logger = logging.getLogger("ai_humanizer.training.reinforce")


class ReinforceTrainer(BaseRLTrainer):
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
        super().__init__(
            model=model,
            config=config,
            repositories=repositories,
            reward_fn=reward_fn,
            loss_fn=loss_fn,
            dataloader=dataloader,
            dry_run=dry_run,
        )
        self.algorithm = "reinforce"
        self.reinforce_cfg = config.rl.reinforce
        self._baseline_ema: Optional[float] = None

    def _update_baseline(self, reward_mean: float) -> float:
        cfg = self.reinforce_cfg
        if cfg.baseline == "none":
            return 0.0
        if self._baseline_ema is None:
            self._baseline_ema = reward_mean
        else:
            self._baseline_ema = 0.9 * self._baseline_ema + 0.1 * reward_mean
        return float(self._baseline_ema)

    def train(self) -> Dict[str, Any]:
        if self.dataloader is None and not self.dry_run:
            raise RuntimeError("ReinforceTrainer.train() requires a dataloader (or dry_run=True).")

        logger.info(
            "REINFORCE ready | gamma=%s baseline=%s entropy_coef=%s dry_run=%s",
            self.reinforce_cfg.gamma,
            self.reinforce_cfg.baseline,
            self.reinforce_cfg.entropy_coef,
            self.dry_run,
        )

        history: Dict[str, Any] = {
            "algorithm": "reinforce",
            "steps": [],
            "dry_run": self.dry_run,
        }

        if self.dry_run:
            history.update(self._dry_run_step())
            self.close()
            return history

        self.model.train()
        save_every = self.config.training.save_steps
        for epoch in range(self.config.rl.epochs):
            self.epoch = epoch
            for batch in self.dataloader:
                batch = self._move_batch(batch)
                metrics = self._train_step(batch)
                history["steps"].append(metrics)
                self.global_step += 1
                if save_every > 0 and self.global_step % save_every == 0:
                    self.save_checkpoint(f"reinforce_step_{self.global_step}")
            self.model.step_gumbel_schedule(self.global_step)

        self.save_checkpoint("reinforce_last")
        self.close()
        history["status"] = "completed"
        return history

    def _dry_run_step(self) -> Dict[str, Any]:
        self.model.train()
        device = next(self.model.parameters()).device
        vocab = min(self.backend.vocab_size - 1, 60)
        lo, hi = 3, max(4, vocab)
        batch = {
            "input_ids": torch.randint(lo, hi, (2, 4), device=device),
            "attention_mask": torch.ones(2, 4, dtype=torch.long, device=device),
            "decoder_input_ids": torch.randint(lo, hi, (2, 5), device=device),
            "labels": torch.randint(lo, hi, (2, 5), device=device),
        }
        metrics = self._train_step(batch, take_optimizer_step=False)
        self.save_checkpoint("reinforce_dry_run")
        return {"status": "dry_run_ok", "sample_metrics": metrics}

    def _train_step(self, batch: Dict[str, Any], take_optimizer_step: bool = True) -> Dict[str, float]:
        outputs = self.humanize_forward(batch)
        rewards = self.compute_rewards(**self.build_reward_batch(outputs, batch))
        reward_total = rewards["total"]
        if reward_total.ndim > 1:
            reward_total = reward_total.mean(dim=-1)

        token_ids = batch.get("decoder_input_ids")
        if token_ids is None:
            token_ids = outputs.soft_tokens.argmax(dim=-1)
        t = min(outputs.logits.size(1), token_ids.size(1))
        log_probs_tok = self.policy_log_probs(outputs.logits[:, :t], token_ids[:, :t])
        log_probs = log_probs_tok.mean(dim=-1)
        probs = torch.softmax(outputs.logits[:, :t], dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()

        reward_mean = float(reward_total.detach().mean().cpu())
        baseline = self._update_baseline(reward_mean)
        advantage = reward_total.detach() - baseline
        # Discounted return surrogate (single-step)
        returns = advantage * self.reinforce_cfg.gamma
        pg_loss = -(log_probs * returns).mean()
        total = pg_loss - self.reinforce_cfg.entropy_coef * entropy

        losses = self.loss_fn(**self.build_loss_batch(outputs, batch))

        if take_optimizer_step:
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()

        metrics = {
            "reward_mean": reward_mean,
            "baseline": baseline,
            "pg_loss": float(pg_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "reinforce_total": float(total.detach().cpu()),
        }
        self.log_step(losses=losses, rewards=rewards, rl_metrics=metrics)
        return metrics
