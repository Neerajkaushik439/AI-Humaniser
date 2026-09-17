"""PPO trainer — preferred RL algorithm.

Uses config-driven PPO hyperparameters. Optionally detects TRL when installed,
but never requires TRL-specific types in the public API.
Policy access goes through ``BaseModelBackend`` only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from configs.config import AppConfig
from losses.composite import CompositeHumanizationLoss
from models.humanizer import HumanizerModel
from repositories.base import RepositoryBundle
from rewards.composite import CompositeReward
from training.rl_trainer import BaseRLTrainer

logger = logging.getLogger("ai_humanizer.training.ppo")


def _maybe_import_trl() -> bool:
    try:
        import trl  # noqa: F401

        return True
    except ImportError:
        return False


class PPOTrainer(BaseRLTrainer):
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
        self.algorithm = "ppo"
        self.ppo_cfg = config.rl.ppo
        self.trl_available = _maybe_import_trl()
        hidden = self.backend.hidden_size
        self.value_head = torch.nn.Linear(hidden, 1)
        device = next(self.model.parameters()).device
        self.value_head.to(device)
        self.optimizer.add_param_group(
            {"params": self.value_head.parameters(), "lr": config.rl.learning_rate}
        )

    def _ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.ppo_cfg
        ratio = torch.exp(log_probs - old_log_probs.detach())
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - cfg.cliprange, 1.0 + cfg.cliprange) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, returns.detach())
        entropy = -(log_probs.exp() * log_probs).mean()
        kl = (old_log_probs.detach() - log_probs).mean()
        total = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
        total = total + cfg.kl_penalty * kl
        return {
            "ppo_total": total,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "kl": kl.detach(),
            "approx_ratio": ratio.mean().detach(),
        }

    def train(self) -> Dict[str, Any]:
        if self.dataloader is None and not self.dry_run:
            raise RuntimeError("PPOTrainer.train() requires a dataloader (or dry_run=True).")

        logger.info(
            "PPO ready | cliprange=%s vf_coef=%s ent_coef=%s kl_penalty=%s trl=%s dry_run=%s",
            self.ppo_cfg.cliprange,
            self.ppo_cfg.vf_coef,
            self.ppo_cfg.ent_coef,
            self.ppo_cfg.kl_penalty,
            self.trl_available,
            self.dry_run,
        )

        history: Dict[str, Any] = {"algorithm": "ppo", "steps": [], "dry_run": self.dry_run}

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
                    self.save_checkpoint(f"ppo_step_{self.global_step}")
            self.model.step_gumbel_schedule(self.global_step)

        self.save_checkpoint("ppo_last")
        self.close()
        history["status"] = "completed"
        return history

    def _dry_run_step(self) -> Dict[str, Any]:
        self.model.train()
        device = next(self.model.parameters()).device
        vocab = min(self.backend.vocab_size - 1, 60)
        lo = 3
        hi = max(lo + 1, vocab)
        batch = {
            "input_ids": torch.randint(lo, hi, (2, 4), device=device),
            "attention_mask": torch.ones(2, 4, dtype=torch.long, device=device),
            "decoder_input_ids": torch.randint(lo, hi, (2, 5), device=device),
            "labels": torch.randint(lo, hi, (2, 5), device=device),
        }
        metrics = self._train_step(batch, take_optimizer_step=False)
        self.save_checkpoint("ppo_dry_run")
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
        log_probs = self.policy_log_probs(outputs.logits[:, :t], token_ids[:, :t]).mean(dim=-1)

        pooled = outputs.embeddings.mean(dim=1)
        values = self.value_head(pooled).squeeze(-1)
        returns = reward_total.detach()
        advantages = returns - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ppo_stats = self._ppo_loss(log_probs, log_probs.detach(), advantages, values, returns)
        losses = self.loss_fn(**self.build_loss_batch(outputs, batch))
        total = ppo_stats["ppo_total"]

        if take_optimizer_step:
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.value_head.parameters()),
                self.config.training.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()

        metrics = {
            "reward_mean": float(reward_total.detach().mean().cpu()),
            "ppo_total": float(ppo_stats["ppo_total"].detach().cpu()),
            "policy_loss": float(ppo_stats["policy_loss"].cpu()),
            "value_loss": float(ppo_stats["value_loss"].cpu()),
            "entropy": float(ppo_stats["entropy"].cpu()),
            "kl": float(ppo_stats["kl"].cpu()),
        }
        self.log_step(losses=losses, rewards=rewards, rl_metrics=metrics)
        return metrics
