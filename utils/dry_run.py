"""Dry-run verification of the full AI-Humanizer stack (no training)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from configs.config import load_config
from losses.composite import CompositeHumanizationLoss
from models.backends.base import BaseModelBackend
from rewards.composite import CompositeReward
from training.checkpoint import CheckpointManager
from training.metrics import MetricsLogger
from training.pipeline import TrainingPipeline
from utils.tiny import build_tiny_humanizer


@dataclass
class DryRunReport:
    ok: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def mark(self, name: str, passed: bool, detail: Any = None) -> None:
        self.checks[name] = passed
        if detail is not None:
            self.details[name] = detail
        if not passed:
            self.ok = False
            self.errors.append(name)


def _synthetic_batch(backend: BaseModelBackend, device: torch.device) -> Dict[str, torch.Tensor]:
    vocab = min(backend.vocab_size - 1, 60)
    lo, hi = 3, max(4, vocab)
    return {
        "input_ids": torch.randint(lo, hi, (2, 4), device=device),
        "attention_mask": torch.ones(2, 4, dtype=torch.long, device=device),
        "decoder_input_ids": torch.randint(lo, hi, (2, 5), device=device),
        "labels": torch.randint(lo, hi, (2, 5), device=device),
    }


def run_dry_run() -> DryRunReport:
    """Exercise config → backend → Gumbel → embedding → losses → rewards → RL."""
    report = DryRunReport(ok=True)
    algorithm = load_config().rl.algorithm.lower()
    config, repos, model, backend = build_tiny_humanizer(algorithm=algorithm)
    device = torch.device("cpu")
    model.to(device)
    model.train()

    report.mark(
        "configuration",
        True,
        {
            "backend": config.model.backend,
            "gumbel_temperature": config.gumbel.temperature,
            "rl_algorithm": config.rl.algorithm,
            "loss_weights": dict(config.losses),
            "reward_weights": dict(config.rewards),
        },
    )

    report.mark(
        "model_backend",
        isinstance(backend, BaseModelBackend),
        {"backend_name": backend.backend_name, "type": type(backend).__name__},
    )
    try:
        tok = backend.tokenize(["dry run text"], max_length=16)
        report.mark("tokenizer", "input_ids" in tok, {"keys": list(tok.keys())})
    except Exception as exc:  # noqa: BLE001
        report.mark("tokenizer", False, str(exc))

    batch = _synthetic_batch(backend, device)
    outputs = model.forward_trainable(**batch)
    report.mark("gumbel_softmax", outputs.soft_tokens.shape == outputs.logits.shape)
    report.mark(
        "soft_distribution_valid",
        bool(
            torch.allclose(
                outputs.soft_tokens.sum(-1),
                torch.ones_like(outputs.soft_tokens.sum(-1)),
                atol=1e-4,
            )
        ),
    )
    fractional = ((outputs.soft_tokens > 0.01) & (outputs.soft_tokens < 0.99)).any().item()
    report.mark("training_no_argmax", bool(fractional))
    report.mark(
        "differentiable_embedding",
        outputs.embeddings.shape[:2] == outputs.soft_tokens.shape[:2],
        {"shape": list(outputs.embeddings.shape)},
    )

    loss_fn = CompositeHumanizationLoss.from_config(config)
    loss_batch = {
        "semantic": outputs.semantic,
        "semantic_target": outputs.semantic.detach(),
        "stylometric": outputs.stylometric,
        "style_target": outputs.stylometric.detach(),
        "human_style": outputs.human_style,
        "human_style_positive": outputs.human_style.detach(),
        "diversity": outputs.diversity,
        "discriminator_logits": outputs.discriminator_logits,
        "logits": outputs.logits,
        "labels": batch["labels"],
        "backbone_loss": (outputs.extras or {}).get("backbone_loss"),
    }
    losses = loss_fn(**loss_batch)
    report.mark("composite_loss", "total_loss" in losses)
    for name in ("semantic", "style", "contrastive", "adversarial", "diversity", "ppl"):
        report.mark(f"loss_module_{name}", name in losses and name in loss_fn.losses)

    try:
        losses["total_loss"].backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        report.mark("gradient_flow", has_grad)
    except Exception as exc:  # noqa: BLE001
        report.mark("gradient_flow", False, str(exc))

    model.zero_grad(set_to_none=True)
    outputs = model.forward_trainable(**batch)
    reward_fn = CompositeReward.from_config(config)
    rewards = reward_fn(
        semantic=outputs.semantic,
        semantic_target=outputs.semantic.detach(),
        stylometric=outputs.stylometric,
        style_target=outputs.stylometric.detach(),
        human_style=outputs.human_style,
        human_style_positive=outputs.human_style.detach(),
        diversity=outputs.diversity,
        discriminator_logits=outputs.discriminator_logits,
        logits=outputs.logits,
        labels=batch["labels"],
    )
    report.mark("composite_reward", "total_reward" in rewards or "total" in rewards)
    for name in reward_fn.rewards.keys():
        report.mark(f"reward_module_{name}", name in rewards)

    pipeline = TrainingPipeline(
        config=config,
        repositories=repos,
        model=model,
        loss_fn=loss_fn,
        reward_fn=reward_fn,
    )
    rl_result = pipeline.dry_run()
    report.mark("rl_trainer", bool(rl_result.get("ok")), rl_result.get("algorithm"))

    ckpt = CheckpointManager(repos.checkpoints)
    report.mark(
        "checkpoint_manager",
        ckpt.exists("ppo_dry_run") or ckpt.exists("reinforce_dry_run"),
    )

    metrics = MetricsLogger(log_dir=config.project.log_dir, enabled=True)
    metrics.log_losses(losses, step=0)
    metrics.log_rewards(rewards, step=0)
    metrics.log_training(
        step=0,
        learning_rate=config.rl.learning_rate,
        gumbel_temperature=config.gumbel.temperature,
    )
    metrics.close()
    report.mark("logging", True, {"log_dir": config.project.log_dir})

    model.eval()
    with torch.no_grad():
        texts = model.generate("hello world", max_new_tokens=4)
    report.mark("inference_generation", isinstance(texts, list) and len(texts) == 1)

    report.details["repositories"] = repos.healthcheck()
    report.details["humanizer"] = type(model).__name__
    return report
