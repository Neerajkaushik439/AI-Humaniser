"""End-to-end training pipeline orchestration (supervised + RL).

Separates dataset handling, model, losses, rewards, RL algorithm,
optimizer, checkpointing, and logging. Supports dry-run verification
without starting a long training job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from configs.config import AppConfig
from losses.composite import CompositeHumanizationLoss
from models.backends.base import BaseModelBackend
from models.humanizer import HumanizerModel
from repositories.base import RepositoryBundle
from rewards.composite import CompositeReward
from training.checkpoint import CheckpointManager
from training.data import HumanizationDataset, build_dataloader
from training.metrics import MetricsLogger
from training.ppo_trainer import PPOTrainer
from training.reinforce_trainer import ReinforceTrainer
from training.rl_trainer import BaseRLTrainer
from training.supervised_trainer import SupervisedTrainer

logger = logging.getLogger("ai_humanizer.training.pipeline")


@dataclass
class PipelineComponents:
    config: AppConfig
    repositories: RepositoryBundle
    model: HumanizerModel
    backend: BaseModelBackend
    loss_fn: CompositeHumanizationLoss
    reward_fn: CompositeReward
    checkpoint_manager: CheckpointManager
    metrics: MetricsLogger
    dataloader: Any = None
    supervised_trainer: Optional[SupervisedTrainer] = None
    rl_trainer: Optional[BaseRLTrainer] = None


class TrainingPipeline:
    """
    Dataset → DataLoader → Backend → Humanizer forward → Gumbel → Diff. Emb.
    → Composite Loss → Composite Reward → PPO/REINFORCE → Optimizer → Checkpoint
    """

    def __init__(
        self,
        config: AppConfig,
        repositories: RepositoryBundle,
        model: HumanizerModel,
        loss_fn: Optional[CompositeHumanizationLoss] = None,
        reward_fn: Optional[CompositeReward] = None,
    ) -> None:
        self.config = config
        self.repositories = repositories
        self.model = model
        # Mandatory: expose abstract backend only
        self.backend: BaseModelBackend = model.backend
        self.loss_fn = loss_fn or CompositeHumanizationLoss.from_config(config)
        self.reward_fn = reward_fn or CompositeReward.from_config(config)
        self.checkpoint_manager = CheckpointManager(
            repository=repositories.checkpoints,
            checkpoint_dir=config.project.checkpoint_dir,
        )
        self.metrics = MetricsLogger(log_dir=config.project.log_dir)

    def build_dataloader(self, dataset_name: Optional[str] = None, records: Optional[list] = None):
        if records is not None:
            dataset = HumanizationDataset.from_records(records)
        elif dataset_name is not None:
            dataset = HumanizationDataset.from_repository(self.repositories.datasets, dataset_name)
        else:
            return None
        return build_dataloader(
            dataset=dataset,
            backend=self.backend,
            batch_size=self.config.training.batch_size,
            max_input_length=self.config.model.max_input_length,
            max_output_length=self.config.model.max_output_length,
            shuffle=True,
        )

    def build_rl_trainer(
        self,
        dataloader: Any = None,
        dry_run: bool = False,
    ) -> BaseRLTrainer:
        algo = self.config.rl.algorithm.lower()
        kwargs = dict(
            model=self.model,
            config=self.config,
            repositories=self.repositories,
            reward_fn=self.reward_fn,
            loss_fn=self.loss_fn,
            dataloader=dataloader,
            dry_run=dry_run,
        )
        if algo == "ppo":
            return PPOTrainer(**kwargs)
        if algo == "reinforce":
            return ReinforceTrainer(**kwargs)
        raise ValueError(f"Unknown rl.algorithm={algo!r}. Use 'ppo' or 'reinforce'.")

    def initialize(
        self,
        dataset_name: Optional[str] = None,
        records: Optional[list] = None,
        dry_run: bool = False,
    ) -> PipelineComponents:
        """Wire all components without starting a long training run."""
        dataloader = None
        if not dry_run:
            dataloader = self.build_dataloader(dataset_name=dataset_name, records=records)

        supervised = SupervisedTrainer(
            model=self.model,
            config=self.config,
            repositories=self.repositories,
            loss_fn=self.loss_fn,
            dataloader=dataloader,
        )
        rl_trainer = self.build_rl_trainer(dataloader=dataloader, dry_run=dry_run)

        components = PipelineComponents(
            config=self.config,
            repositories=self.repositories,
            model=self.model,
            backend=self.backend,
            loss_fn=self.loss_fn,
            reward_fn=self.reward_fn,
            checkpoint_manager=self.checkpoint_manager,
            metrics=self.metrics,
            dataloader=dataloader,
            supervised_trainer=supervised,
            rl_trainer=rl_trainer,
        )
        logger.info(
            "Pipeline initialized | backend=%s rl=%s dry_run=%s",
            self.backend.backend_name,
            self.config.rl.algorithm,
            dry_run,
        )
        return components

    def dry_run(self) -> Dict[str, Any]:
        """Verify the full RL path on a tiny synthetic batch; do not train for real."""
        components = self.initialize(dry_run=True)
        assert components.rl_trainer is not None
        result = components.rl_trainer.train()
        self.metrics.close()
        return {
            "ok": result.get("status") == "dry_run_ok",
            "algorithm": result.get("algorithm"),
            "backend": self.backend.backend_name,
            "result": result,
        }
