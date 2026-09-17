"""Stylometric Module — modular stylistic feature extraction for Style Loss.

Feature extractors are pluggable so lexical / length / punctuation / rhythm /
syntactic signals can be added later without rewriting the trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Type

import torch
import torch.nn as nn
from torch import Tensor


class StyleFeatureExtractor(ABC, nn.Module):
    """One stylometric feature family → fixed-size vector per sample."""

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
        """Return ``[batch, output_dim]``."""


class EmbeddingStatsFeature(StyleFeatureExtractor):
    """Mean / std / norm stats of soft embeddings (lexical proxy in embedding space)."""

    name = "embedding_stats"

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size

    @property
    def output_dim(self) -> int:
        return self.input_size * 2 + 1

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        soft_tokens: Tensor | None = None,
    ) -> Tensor:
        if attention_mask is None:
            mean = embeddings.mean(dim=1)
            std = embeddings.std(dim=1, unbiased=False)
            norms = embeddings.norm(dim=-1).mean(dim=1, keepdim=True)
        else:
            mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
            lengths = mask.sum(dim=1).clamp(min=1e-6)
            mean = (embeddings * mask).sum(dim=1) / lengths
            centered = (embeddings - mean.unsqueeze(1)) * mask
            var = (centered.pow(2).sum(dim=1) / lengths).clamp(min=0.0)
            std = var.sqrt()
            norms = (embeddings.norm(dim=-1) * attention_mask.to(embeddings.dtype)).sum(
                dim=1, keepdim=True
            ) / lengths
        return torch.cat([mean, std, norms], dim=-1)


class SequenceLengthFeature(StyleFeatureExtractor):
    """Sentence / sequence length signal from the attention mask."""

    name = "sequence_length"

    @property
    def output_dim(self) -> int:
        return 2

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
        # Normalized length + log length (rhythm / pacing proxy)
        return torch.cat([lengths / float(seq_len), torch.log1p(lengths)], dim=-1)


class SoftTokenEntropyFeature(StyleFeatureExtractor):
    """Lexical diversity proxy via entropy of the soft token distribution."""

    name = "soft_token_entropy"

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
        # soft_tokens: [B, T, V]
        p = soft_tokens.clamp_min(1e-8)
        token_entropy = -(p * p.log()).sum(dim=-1)  # [B, T]
        if attention_mask is not None:
            mask = attention_mask.to(token_entropy.dtype)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            mean_ent = (token_entropy * mask).sum(dim=1) / denom
            std_ent = (
                (((token_entropy - mean_ent.unsqueeze(1)).pow(2) * mask).sum(dim=1) / denom).sqrt()
            )
        else:
            mean_ent = token_entropy.mean(dim=1)
            std_ent = token_entropy.std(dim=1, unbiased=False)
        return torch.stack([mean_ent, std_ent], dim=-1)


class EmbeddingRhythmFeature(StyleFeatureExtractor):
    """Sentence rhythm / burstiness via successive embedding differences."""

    name = "embedding_rhythm"

    @property
    def output_dim(self) -> int:
        return 2

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        soft_tokens: Tensor | None = None,
    ) -> Tensor:
        if embeddings.size(1) < 2:
            return embeddings.new_zeros(embeddings.size(0), self.output_dim)
        deltas = (embeddings[:, 1:] - embeddings[:, :-1]).norm(dim=-1)  # [B, T-1]
        return torch.stack([deltas.mean(dim=1), deltas.std(dim=1, unbiased=False)], dim=-1)


STYLE_FEATURE_REGISTRY: Dict[str, Type[StyleFeatureExtractor]] = {
    EmbeddingStatsFeature.name: EmbeddingStatsFeature,
    SequenceLengthFeature.name: SequenceLengthFeature,
    SoftTokenEntropyFeature.name: SoftTokenEntropyFeature,
    EmbeddingRhythmFeature.name: EmbeddingRhythmFeature,
}


def register_style_feature(cls: Type[StyleFeatureExtractor]) -> Type[StyleFeatureExtractor]:
    STYLE_FEATURE_REGISTRY[cls.name] = cls
    return cls


class StylometricModule(nn.Module):
    """Compose pluggable stylometric features → Style Loss representation."""

    def __init__(
        self,
        input_size: int,
        feature_dim: int = 64,
        hidden_size: int = 256,
        feature_names: Optional[Sequence[str]] = None,
        extra_features: Optional[Sequence[StyleFeatureExtractor]] = None,
    ) -> None:
        super().__init__()
        names = list(feature_names) if feature_names is not None else [
            "embedding_stats",
            "sequence_length",
            "soft_token_entropy",
            "embedding_rhythm",
        ]
        extractors: List[StyleFeatureExtractor] = []
        for name in names:
            if name not in STYLE_FEATURE_REGISTRY:
                raise KeyError(
                    f"Unknown style feature {name!r}. Known: {sorted(STYLE_FEATURE_REGISTRY)}"
                )
            cls = STYLE_FEATURE_REGISTRY[name]
            if name == "embedding_stats":
                extractors.append(cls(input_size=input_size))  # type: ignore[call-arg]
            else:
                extractors.append(cls())
        if extra_features:
            extractors.extend(list(extra_features))

        self.extractors = nn.ModuleList(extractors)
        in_dim = sum(e.output_dim for e in extractors)
        self.feature_dim = feature_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, feature_dim),
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
