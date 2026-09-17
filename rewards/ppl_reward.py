"""Optional PPL / reference-LM reward adapter (usually small weight)."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from rewards.base import BaseRewardComponent, register_reward


@register_reward
class PPLReward(BaseRewardComponent):
    """Negative NLL reward — higher when language-model likelihood is better.

    Weight ``lambda_ppl`` should stay small when enabled (regularizer-like).
    """

    name = "ppl"

    def forward(
        self,
        logits: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        backbone_loss: Optional[Tensor] = None,
        **_: Any,
    ) -> Tensor:
        if backbone_loss is not None:
            if backbone_loss.ndim == 0:
                if logits is not None:
                    return (-backbone_loss).expand(logits.size(0))
                return -backbone_loss.unsqueeze(0)
            return -backbone_loss

        if logits is None or labels is None:
            if logits is not None:
                return logits.new_zeros(logits.size(0))
            return torch.zeros(1)

        log_probs = torch.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return token_lp.mean(dim=-1)
