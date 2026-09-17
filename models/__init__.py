"""Models package — architecture modules + Final Humanizer."""

from models.backends import (
    BACKEND_REGISTRY,
    BaseModelBackend,
    FLANT5Backend,
    GenerationParams,
    build_backend,
)
from models.differentiable_embedding import DifferentiableEmbedding
from models.discriminator import BaseDiscriminator, HumanAIDiscriminator
from models.diversity_module import DiversityBurstinessModule
from models.gumbel_softmax import (
    ConstantTemperatureSchedule,
    GumbelSoftmax,
    LinearAnnealingTemperatureSchedule,
    gumbel_softmax,
)
from models.humanizer import HumanizerModel, HumanizerOutput
from models.pipeline import SoftTokenPipeline, SoftTokenPipelineOutput
from models.semantic_encoder import BaseSemanticEncoder, SemanticEncoder
from models.style_encoder import BaseHumanStyleEncoder, HumanStyleEncoder
from models.stylometric_module import StylometricModule

__all__ = [
    "BACKEND_REGISTRY",
    "BaseDiscriminator",
    "BaseHumanStyleEncoder",
    "BaseModelBackend",
    "BaseSemanticEncoder",
    "ConstantTemperatureSchedule",
    "DifferentiableEmbedding",
    "DiversityBurstinessModule",
    "FLANT5Backend",
    "GenerationParams",
    "GumbelSoftmax",
    "HumanAIDiscriminator",
    "HumanStyleEncoder",
    "HumanizerModel",
    "HumanizerOutput",
    "LinearAnnealingTemperatureSchedule",
    "SemanticEncoder",
    "SoftTokenPipeline",
    "SoftTokenPipelineOutput",
    "StylometricModule",
    "build_backend",
    "gumbel_softmax",
]
