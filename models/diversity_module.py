"""Diversity / Burstiness Module — extensible measurements for L_diversity.

Supports lexical diversity, length variation, token-distribution variation,
and sentence rhythm / burstiness via pluggable feature extractors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Type

import torch
import torch.nn as nn
from torch import Tensor


class BurstinessFeatureExtractor(ABC, nn.Module):
    name: str = "base"

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...

    @abstractmethod
    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        soft_tokens: Tensor | None = None,
    ) -> Tensor:
        ...


class LocalVarianceFeature(BurstinessFeatureExtractor):
    """Local embedding variance over sliding windows (burstiness)."""

    name = "local_variance"

    def __init__(self, input_size: int, window_size: int = 16) -> None:
        super().__init__()
        self.input_size = input_size
        self.window_size = int(window_size)

    @property
    def output_dim(self) -> int:
        return self.input_size

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        soft_tokens: Tensor | None = None,
    ) -> Tensor:
        seq_len = embeddings.size(1)
        window = min(self.window_size, max(seq_len, 1))
        if seq_len >= window:
            unfolded = embeddings.unfold(dimension=1, size=window, step=1)
            return unfolded.std(dim=-1, unbiased=False).mean(dim=1)
        return embeddings.std(dim=1, unbiased=False)


class TokenNormBurstFeature(BurstinessFeatureExtractor):
    """Sentence rhythm via variance of token embedding norms."""

    name = "token_norm_burst"

    @property
    def output_dim(self) -> int:
        return 1

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        soft_tokens: Tensor | None = None,
    ) -> Tensor:
        token_norms = embeddings.norm(dim=-1)
        if attention_mask is not None:
            mask = attention_mask.to(token_norms.dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
            mean_norm = (token_norms * mask).sum(dim=1, keepdim=True) / denom
            burst = ((token_norms - mean_norm).pow(2) * mask).sum(dim=1, keepdim=True) / denom
        else:
            burst = token_norms.var(dim=1, unbiased=False, keepdim=True)
        return burst


class SoftTokenDiversityFeature(BurstinessFeatureExtractor):
    """Token distribution variation (avg entropy + uniqueness proxy)."""

    name = "soft_token_diversity"

    @property
    def output_dim(self) -> int:
        return 2

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        soft_tokens: Tensor | None = None,
    ) -> Tensor:
        bsz = embeddings.size(0)
        if soft_tokens is None:
            return embeddings.new_zeros(bsz, self.output_dim)
        p = soft_tokens.clamp_min(1e-8)
        entropy = -(p * p.log()).sum(dim=-1)  # [B, T]
        # Soft uniqueness: average of max-prob (lower → more diverse)
        max_prob = soft_tokens.max(dim=-1).values
        if attention_mask is not None:
            mask = attention_mask.to(entropy.dtype)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            mean_ent = (entropy * mask).sum(dim=1) / denom
            mean_max = (max_prob * mask).sum(dim=1) / denom
        else:
            mean_ent = entropy.mean(dim=1)
            mean_max = max_prob.mean(dim=1)
        return torch.stack([mean_ent, 1.0 - mean_max], dim=-1)


class LengthVariationFeature(BurstinessFeatureExtractor):
    """Sequence length signal used as length-variation component."""

    name = "length_variation"

    @property
    def output_dim(self) -> int:
        return 1

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        soft_tokens: Tensor | None = None,
    ) -> Tensor:
        bsz, seq_len, _ = embeddings.shape
        if attention_mask is None:
            lengths = embeddings.new_full((bsz, 1), float(seq_len))
        else:
            lengths = attention_mask.to(embeddings.dtype).sum(dim=1, keepdim=True)
        return lengths / float(max(seq_len, 1))


BURSTINESS_FEATURE_REGISTRY: Dict[str, Type[BurstinessFeatureExtractor]] = {
    LocalVarianceFeature.name: LocalVarianceFeature,
    TokenNormBurstFeature.name: TokenNormBurstFeature,
    SoftTokenDiversityFeature.name: SoftTokenDiversityFeature,
    LengthVariationFeature.name: LengthVariationFeature,
}


def register_burstiness_feature(
    cls: Type[BurstinessFeatureExtractor],
) -> Type[BurstinessFeatureExtractor]:
    BURSTINESS_FEATURE_REGISTRY[cls.name] = cls
    return cls


class DiversityBurstinessModule(nn.Module):
    """Compose burstiness / diversity features for L_diversity."""

    def __init__(
        self,
        input_size: int,
        window_size: int = 16,
        hidden_size: int = 128,
        feature_names: Optional[Sequence[str]] = None,
        extra_features: Optional[Sequence[BurstinessFeatureExtractor]] = None,
    ) -> None:
        super().__init__()
        names = list(feature_names) if feature_names is not None else [
            "local_variance",
            "token_norm_burst",
            "soft_token_diversity",
            "length_variation",
        ]
        extractors: List[BurstinessFeatureExtractor] = []
        for name in names:
            if name not in BURSTINESS_FEATURE_REGISTRY:
                raise KeyError(
                    f"Unknown burstiness feature {name!r}. "
                    f"Known: {sorted(BURSTINESS_FEATURE_REGISTRY)}"
                )
            cls = BURSTINESS_FEATURE_REGISTRY[name]
            if name == "local_variance":
                extractors.append(cls(input_size=input_size, window_size=window_size))  # type: ignore[call-arg]
            else:
                extractors.append(cls())
        if extra_features:
            extractors.extend(list(extra_features))

        self.extractors = nn.ModuleList(extractors)
        in_dim = sum(e.output_dim for e in extractors)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def extract_features(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        soft_tokens: Tensor | None = None,
    ) -> Tensor:
        parts = [
            ext(embeddings, attention_mask=attention_mask, soft_tokens=soft_tokens)
            for ext in self.extractors
        ]
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        soft_tokens: Tensor | None = None,
    ) -> Tensor:
        raw = self.extract_features(embeddings, attention_mask, soft_tokens)
        return self.proj(raw)
