"""Shared tiny-model helpers for dry-run and tests (no large HF downloads)."""

from __future__ import annotations

from typing import Tuple

from configs.config import (
    AppConfig,
    EmbeddingConfig,
    GumbelConfig,
    ModelConfig,
    ModulesConfig,
    RepositoryConfig,
    RLConfig,
    load_config,
)
from models.backends.base import BaseModelBackend
from models.backends.registry import build_tiny_backend
from models.humanizer import HumanizerModel
from repositories import RepositoryBundle, build_repositories


def tiny_module_config() -> ModulesConfig:
    modules = ModulesConfig()
    modules.semantic_encoder.hidden_size = 32
    modules.semantic_encoder.output_size = 32
    modules.human_style_encoder.hidden_size = 32
    modules.human_style_encoder.projection_dim = 16
    modules.stylometric.hidden_size = 32
    modules.stylometric.feature_dim = 16
    modules.diversity.hidden_size = 16
    modules.discriminator.hidden_size = 32
    return modules


def build_tiny_config(algorithm: str = "ppo") -> AppConfig:
    """Config sized for a tiny randomly-initialized backend."""
    base = load_config()
    return AppConfig(
        project=base.project,
        device="cpu",
        model=ModelConfig(
            backend=base.model.backend,
            model_name="tiny-backend",
            tokenizer_name="tiny-backend",
            max_input_length=32,
            max_output_length=32,
            generation=base.model.generation,
        ),
        gumbel=GumbelConfig(temperature=base.gumbel.temperature, hard=False),
        embedding=EmbeddingConfig(freeze=False, share_weights=True),
        modules=tiny_module_config(),
        losses=dict(base.losses),
        rewards=dict(base.rewards),
        training=base.training,
        rl=RLConfig(
            algorithm=algorithm,
            learning_rate=base.rl.learning_rate,
            batch_size=2,
            mini_batch_size=1,
            epochs=1,
            ppo=base.rl.ppo,
            reinforce=base.rl.reinforce,
        ),
        repositories=RepositoryConfig(
            kind="memory",
            data_dir=base.repositories.data_dir,
            checkpoint_dir=base.repositories.checkpoint_dir,
            experiment_dir=base.repositories.experiment_dir,
        ),
    )


def build_tiny_humanizer(
    algorithm: str = "ppo",
) -> Tuple[AppConfig, RepositoryBundle, HumanizerModel, BaseModelBackend]:
    config = build_tiny_config(algorithm=algorithm)
    repos = build_repositories(
        kind=config.repositories.kind,
        data_dir=config.repositories.data_dir,
        checkpoint_dir=config.repositories.checkpoint_dir,
        experiment_dir=config.repositories.experiment_dir,
    )
    backend = build_tiny_backend(backend=config.model.backend)
    model = HumanizerModel(config=config, backend=backend)
    return config, repos, model, backend
