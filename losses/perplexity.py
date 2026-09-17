"""PPL loss — L_PPL (λ6), optional weak regularizer.

Architecture note: λ6 MUST stay small and come from configuration.
This class never hard-codes a λ weight.

The reference LM is injected behind ``BaseReferenceLanguageModel``.
"""

from __future__ import annotations

from typing import Any, Optional

from torch import Tensor

from losses.base import BaseLoss, register_loss
from losses.reference_lm import (
    BaseReferenceLanguageModel,
    LogitsCrossEntropyReferenceLM,
)


@register_loss
class PPLLoss(BaseLoss):
    """Perplexity / NLL regularizer (not the primary objective)."""

    name = "ppl"

    def __init__(
        self,
        reference_lm: Optional[BaseReferenceLanguageModel] = None,
    ) -> None:
        super().__init__()
        self.reference_lm: BaseReferenceLanguageModel = (
            reference_lm if reference_lm is not None else LogitsCrossEntropyReferenceLM()
        )

    def forward(
        self,
        logits: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        backbone_loss: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        return self.reference_lm.negative_log_likelihood(
            logits=logits,
            labels=labels,
            backbone_loss=backbone_loss,
            **kwargs,
        )


# Backward-compatible alias (same registered class).
PerplexityLoss = PPLLoss
