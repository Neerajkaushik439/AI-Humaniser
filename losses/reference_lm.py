"""Reference language-model abstraction for the optional PPL regularizer.

PPL is NOT the primary objective — λ6 stays small in config.
The concrete LM behind this interface is swappable (backbone CE, frozen LM, …).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseReferenceLanguageModel(ABC, nn.Module):
    """Produces a scalar NLL / CE surrogate used by PPLLoss."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def negative_log_likelihood(
        self,
        *,
        logits: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        backbone_loss: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Return a scalar NLL tensor (not weighted by λ)."""


class LogitsCrossEntropyReferenceLM(BaseReferenceLanguageModel):
    """Default: token CE from provided logits / backbone loss (no extra LM)."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def negative_log_likelihood(
        self,
        *,
        logits: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        backbone_loss: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if backbone_loss is not None:
            return backbone_loss
        if logits is None:
            raise ValueError("LogitsCrossEntropyReferenceLM requires logits or backbone_loss")
        if labels is None:
            return logits.new_zeros(())
        vocab = logits.size(-1)
        return F.cross_entropy(
            logits.reshape(-1, vocab),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )


class FrozenExternalReferenceLM(BaseReferenceLanguageModel):
    """Placeholder for a future frozen external LM scorer.

    Wire a real scorer later without changing PPLLoss or the trainer.
    """

    def __init__(self, scorer: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.scorer = scorer

    def negative_log_likelihood(
        self,
        *,
        logits: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        backbone_loss: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if self.scorer is None:
            # Graceful zero until an external scorer is injected.
            ref = logits if logits is not None else backbone_loss
            if ref is None:
                return torch.zeros(())
            return ref.new_zeros(())
        return self.scorer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
