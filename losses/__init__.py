"""Loss package — extensible multi-term humanization objective."""

from losses.adversarial import AdversarialLoss
from losses.base import (
    LOSS_REGISTRY,
    BaseLoss,
    register_loss,
    resolve_weight,
    validate_loss_weights,
)
from losses.composite import DEFAULT_LOSS_NAMES, CompositeHumanizationLoss, CompositeLoss
from losses.contrastive import ContrastiveHumanLoss
from losses.diversity import DiversityLoss
from losses.perplexity import PPLLoss, PerplexityLoss
from losses.reference_lm import (
    BaseReferenceLanguageModel,
    FrozenExternalReferenceLM,
    LogitsCrossEntropyReferenceLM,
)
from losses.semantic import SemanticLoss
from losses.style import StyleLoss

__all__ = [
    "AdversarialLoss",
    "BaseLoss",
    "BaseReferenceLanguageModel",
    "CompositeHumanizationLoss",
    "CompositeLoss",
    "ContrastiveHumanLoss",
    "DEFAULT_LOSS_NAMES",
    "DiversityLoss",
    "FrozenExternalReferenceLM",
    "LOSS_REGISTRY",
    "LogitsCrossEntropyReferenceLM",
    "PPLLoss",
    "PerplexityLoss",
    "SemanticLoss",
    "StyleLoss",
    "register_loss",
    "resolve_weight",
    "validate_loss_weights",
]
