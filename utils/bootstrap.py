"""Application bootstrap — wires config, repositories, model, losses, rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from configs.config import AppConfig, load_config
from losses.composite import CompositeHumanizationLoss
from models.humanizer import HumanizerModel
from repositories import RepositoryBundle, build_repositories
from rewards.composite import CompositeReward
from utils import resolve_device, set_seed, setup_logging


@dataclass
class Application:
    config: AppConfig
    repositories: RepositoryBundle
    model: HumanizerModel
    loss_fn: CompositeHumanizationLoss
    reward_fn: CompositeReward
    device: object


def build_application(
    config_path: Optional[str] = None,
    load_weights: bool = False,
    overrides: Optional[dict] = None,
) -> Application:
    """Construct the full system without starting training.

    ``load_weights=False`` (default for ``main.py``) builds the architecture
    from config only — no training loop, no checkpoint downloads required for
    local bootstrap (falls back to a minimal T5 config if HF is unreachable).
    """
    config = load_config(config_path, overrides=overrides)
    setup_logging(config.project.log_dir)
    set_seed(config.project.seed)
    device = resolve_device(config.device)

    repositories = build_repositories(
        kind=config.repositories.kind,
        data_dir=config.repositories.data_dir,
        checkpoint_dir=config.repositories.checkpoint_dir,
        experiment_dir=config.repositories.experiment_dir,
    )

    from models.backends.registry import build_backend
    from models.humanizer import _generation_params_from_config

    backend = build_backend(
        backend=config.model.backend,
        model_name=config.model.model_name,
        tokenizer_name=config.model.tokenizer_name,
        use_peft=config.model.use_peft and load_weights,
        peft_config={
            "r": config.model.peft.r,
            "lora_alpha": config.model.peft.lora_alpha,
            "lora_dropout": config.model.peft.lora_dropout,
            "target_modules": config.model.peft.target_modules,
        },
        device=device if load_weights else None,
        load_weights=load_weights,
        generation_params=_generation_params_from_config(config.model.generation),
    )
    model = HumanizerModel(config=config, backend=backend)
    model.to(device)

    loss_fn = CompositeHumanizationLoss.from_config(config)
    reward_fn = CompositeReward.from_config(config)

    return Application(
        config=config,
        repositories=repositories,
        model=model,
        loss_fn=loss_fn,
        reward_fn=reward_fn,
        device=device,
    )
