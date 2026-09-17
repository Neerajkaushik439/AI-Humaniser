"""Training package — supervised, RL, pipeline, checkpoints, metrics, data."""

from training.checkpoint import CheckpointManager
from training.data import HumanizationDataset, build_dataloader
from training.metrics import MetricsLogger
from training.pipeline import PipelineComponents, TrainingPipeline
from training.ppo_trainer import PPOTrainer
from training.reinforce_trainer import ReinforceTrainer
from training.rl_trainer import BaseRLTrainer, RLTrainer
from training.supervised_trainer import SupervisedTrainer
from training.trainer import BaseTrainer


def build_rl_trainer(
    model,
    config,
    repositories,
    reward_fn=None,
    loss_fn=None,
    dataloader=None,
    dry_run: bool = False,
):
    algo = config.rl.algorithm.lower()
    kwargs = dict(
        model=model,
        config=config,
        repositories=repositories,
        reward_fn=reward_fn,
        loss_fn=loss_fn,
        dataloader=dataloader,
        dry_run=dry_run,
    )
    if algo == "ppo":
        return PPOTrainer(**kwargs)
    if algo == "reinforce":
        return ReinforceTrainer(**kwargs)
    raise ValueError(f"Unknown rl.algorithm={algo!r}. Use 'ppo' or 'reinforce'.")


__all__ = [
    "BaseRLTrainer",
    "BaseTrainer",
    "CheckpointManager",
    "HumanizationDataset",
    "MetricsLogger",
    "PPOTrainer",
    "PipelineComponents",
    "RLTrainer",
    "ReinforceTrainer",
    "SupervisedTrainer",
    "TrainingPipeline",
    "build_dataloader",
    "build_rl_trainer",
]
