"""Tests for rewards, RL trainers, checkpoints, and training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import (
    AppConfig,
    EmbeddingConfig,
    GumbelConfig,
    ModelConfig,
    ModulesConfig,
    RLConfig,
    load_config,
)
from models.backends.flan_t5 import FLANT5Backend
from models.humanizer import HumanizerModel
from repositories import build_repositories
from rewards import (
    REWARD_REGISTRY,
    BaseRewardComponent,
    CompositeReward,
    DiscriminatorReward,
    DiversityReward,
    HumanStyleReward,
    PPLReward,
    SemanticReward,
    StyleReward,
)
from training import (
    CheckpointManager,
    HumanizationDataset,
    PPOTrainer,
    ReinforceTrainer,
    TrainingPipeline,
    build_rl_trainer,
)


def _tiny_config(algorithm: str = "ppo") -> AppConfig:
    modules = ModulesConfig()
    modules.semantic_encoder.hidden_size = 32
    modules.semantic_encoder.output_size = 32
    modules.human_style_encoder.hidden_size = 32
    modules.human_style_encoder.projection_dim = 16
    modules.stylometric.hidden_size = 32
    modules.stylometric.feature_dim = 16
    modules.diversity.hidden_size = 16
    modules.discriminator.hidden_size = 32
    return AppConfig(
        model=ModelConfig(backend="flan_t5", model_name="tiny-t5", tokenizer_name="tiny-t5"),
        gumbel=GumbelConfig(temperature=0.9, hard=False),
        embedding=EmbeddingConfig(freeze=False, share_weights=True),
        modules=modules,
        rl=RLConfig(algorithm=algorithm, learning_rate=1e-4, batch_size=2, epochs=1),
        rewards={
            "lambda_semantic": 1.0,
            "lambda_style": 1.0,
            "lambda_human_style": 1.0,
            "lambda_human": 1.0,
            "lambda_discriminator": 1.0,
            "lambda_diversity": 0.5,
            "lambda_ppl": 0.05,
        },
        repositories=load_config().repositories,
    )


def _tiny_app(algorithm: str = "ppo"):
    config = _tiny_config(algorithm=algorithm)
    config.repositories.kind = "memory"
    repos = build_repositories("memory", "data", "checkpoints", "outputs/experiments")
    backend = FLANT5Backend.from_tiny(vocab_size=64, d_model=32)
    model = HumanizerModel(config=config, backend=backend)
    return config, repos, model


def _reward_batch(batch: int = 4, dim: int = 8):
    return {
        "semantic": torch.randn(batch, dim),
        "semantic_target": torch.randn(batch, dim),
        "stylometric": torch.randn(batch, dim),
        "style_target": torch.randn(batch, dim),
        "human_style": torch.randn(batch, dim),
        "human_style_positive": torch.randn(batch, dim),
        "diversity": torch.randn(batch, 12),
        "discriminator_logits": torch.randn(batch),
        "logits": torch.randn(batch, 5, 16),
        "labels": torch.randint(0, 16, (batch, 5)),
    }


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------


def test_reward_components_independent():
    batch = _reward_batch()
    assert SemanticReward()(**batch).shape == (4,)
    assert StyleReward()(**batch).shape == (4,)
    assert HumanStyleReward()(**batch).shape == (4,)
    assert DiversityReward()(**batch).shape == (4,)
    assert DiscriminatorReward()(**batch).shape == (4,)
    assert PPLReward()(**batch).shape == (4,)


def test_composite_reward_output():
    weights = {
        "lambda_semantic": 1.0,
        "lambda_style": 0.5,
        "lambda_human_style": 1.0,
        "lambda_discriminator": 1.0,
        "lambda_diversity": 0.5,
        "lambda_ppl": 0.05,
    }
    composite = CompositeReward(weights=weights)
    out = composite(**_reward_batch())
    assert "total" in out and "total_reward" in out
    assert out["total"].shape == (4,)
    for key in ("semantic", "style", "human_style", "diversity", "discriminator"):
        assert key in out


def test_reward_registry_has_required_components():
    for name in ("semantic", "style", "human_style", "diversity", "discriminator", "ppl", "human"):
        assert name in REWARD_REGISTRY


def test_base_reward_component_alias():
    assert issubclass(SemanticReward, BaseRewardComponent)


# ---------------------------------------------------------------------------
# RL trainers
# ---------------------------------------------------------------------------


def test_rl_trainer_initialization_uses_base_backend():
    config, repos, model = _tiny_app("ppo")
    trainer = build_rl_trainer(model, config, repos, dry_run=True)
    from models.backends.base import BaseModelBackend

    assert isinstance(trainer.backend, BaseModelBackend)
    # Must not depend on concrete FLAN class in the trainer module's required path
    assert trainer.backend is model.backend


def test_ppo_initialization_and_dry_run():
    config, repos, model = _tiny_app("ppo")
    trainer = PPOTrainer(model=model, config=config, repositories=repos, dry_run=True)
    result = trainer.train()
    assert result["algorithm"] == "ppo"
    assert result["status"] == "dry_run_ok"
    assert repos.checkpoints.exists("ppo_dry_run")


def test_reinforce_initialization_and_dry_run():
    config, repos, model = _tiny_app("reinforce")
    trainer = ReinforceTrainer(model=model, config=config, repositories=repos, dry_run=True)
    result = trainer.train()
    assert result["algorithm"] == "reinforce"
    assert result["status"] == "dry_run_ok"


def test_build_rl_trainer_respects_config_algorithm():
    config, repos, model = _tiny_app("reinforce")
    trainer = build_rl_trainer(model, config, repos, dry_run=True)
    assert isinstance(trainer, ReinforceTrainer)
    config2, repos2, model2 = _tiny_app("ppo")
    trainer2 = build_rl_trainer(model2, config2, repos2, dry_run=True)
    assert isinstance(trainer2, PPOTrainer)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_save_load_roundtrip():
    config, repos, model = _tiny_app("ppo")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    mgr = CheckpointManager(repos.checkpoints, checkpoint_dir="checkpoints")
    mgr.save(
        "unit_ckpt",
        model=model,
        optimizer=opt,
        step=7,
        epoch=2,
        config=config,
        loss_weights=config.losses,
        reward_weights=config.rewards,
        rl_config=config.to_dict()["rl"],
    )
    payload = mgr.load("unit_ckpt")
    assert payload["step"] == 7
    assert payload["epoch"] == 2
    assert "model_state" in payload
    assert "optimizer_state" in payload
    assert "loss_weights" in payload
    assert "reward_weights" in payload
    assert "rl_configuration" in payload
    assert "configuration" in payload

    model2 = HumanizerModel(config=config, backend=FLANT5Backend.from_tiny(vocab_size=64, d_model=32))
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    restored = mgr.restore("unit_ckpt", model=model2, optimizer=opt2)
    assert restored["step"] == 7


# ---------------------------------------------------------------------------
# Dataset / pipeline
# ---------------------------------------------------------------------------


def test_dataset_from_json_records_via_repository():
    repos = build_repositories("memory", "data", "checkpoints", "outputs/experiments")
    records = [
        {"source_text": "AI wrote this.", "target_text": "A person wrote this."},
        {"ai_text": "Hello", "human_text": "Hi there"},
    ]
    repos.datasets.save("demo", records)
    ds = HumanizationDataset.from_repository(repos.datasets, "demo")
    assert len(ds) == 2
    assert ds[0].source_text == "AI wrote this."


def test_training_pipeline_initialization_and_dry_run():
    config, repos, model = _tiny_app("ppo")
    pipeline = TrainingPipeline(config=config, repositories=repos, model=model)
    components = pipeline.initialize(dry_run=True)
    assert components.rl_trainer is not None
    assert components.backend is model.backend
    result = pipeline.dry_run()
    assert result["ok"] is True
    assert result["algorithm"] == "ppo"


def test_jsonl_and_csv_parsing(tmp_path):
    from repositories.local import LocalDatasetRepository

    repo = LocalDatasetRepository(tmp_path)
    jsonl = tmp_path / "raw" / "a.jsonl"
    jsonl.write_text(
        '{"source_text":"one"}\n{"source_text":"two","target_text":"deux"}\n',
        encoding="utf-8",
    )
    rows = repo.load("a")
    assert len(rows) == 2
    csv_path = tmp_path / "raw" / "b.csv"
    csv_path.write_text("source_text,target_text\nhello,hi\n", encoding="utf-8")
    rows2 = repo.load("b")
    assert rows2[0]["source_text"] == "hello"
