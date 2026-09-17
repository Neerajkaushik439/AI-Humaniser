"""RL reward package."""

from rewards.base import (
    REWARD_REGISTRY,
    BaseReward,
    BaseRewardComponent,
    register_reward,
    resolve_reward_weight,
    validate_reward_weights,
)
from rewards.composite import DEFAULT_REWARD_NAMES, CompositeReward
from rewards.diversity_reward import DiversityReward
from rewards.human_reward import DiscriminatorReward, HumanReward
from rewards.human_style_reward import HumanStyleReward
from rewards.ppl_reward import PPLReward
from rewards.semantic_reward import SemanticReward
from rewards.style_reward import StyleReward

__all__ = [
    "BaseReward",
    "BaseRewardComponent",
    "CompositeReward",
    "DEFAULT_REWARD_NAMES",
    "DiscriminatorReward",
    "DiversityReward",
    "HumanReward",
    "HumanStyleReward",
    "PPLReward",
    "REWARD_REGISTRY",
    "SemanticReward",
    "StyleReward",
    "register_reward",
    "resolve_reward_weight",
    "validate_reward_weights",
]
