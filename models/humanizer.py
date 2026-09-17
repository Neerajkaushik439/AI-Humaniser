"""Final Humanizer — wires backbone + differentiable pipeline + objective heads.

TRAINING path (differentiable):
  text → tokenize → backbone logits → Gumbel-Softmax → soft P
       → Differentiable Embedding (H_soft = P E) → downstream modules

INFERENCE path (discrete):
  text → tokenize → backbone.generate → discrete tokens → text

The inference path never routes through Gumbel-Softmax / soft embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from configs.config import AppConfig, GenerationConfig
from models.backends.base import BaseModelBackend, GenerationParams
from models.backends.registry import build_backend
from models.differentiable_embedding import DifferentiableEmbedding
from models.discriminator import HumanAIDiscriminator
from models.diversity_module import DiversityBurstinessModule
from models.gumbel_softmax import (
    ConstantTemperatureSchedule,
    GumbelSoftmax,
    LinearAnnealingTemperatureSchedule,
)
from models.pipeline import SoftTokenPipeline, SoftTokenPipelineOutput
from models.semantic_encoder import SemanticEncoder
from models.style_encoder import HumanStyleEncoder
from models.stylometric_module import StylometricModule


@dataclass
class HumanizerOutput:
    """Full training-path output including soft tokens and downstream heads."""

    logits: Tensor
    soft_tokens: Tensor
    embeddings: Tensor
    semantic: Tensor
    stylometric: Tensor
    human_style: Tensor
    diversity: Tensor
    discriminator_logits: Tensor
    temperature: float
    extras: Optional[Dict[str, Any]] = None


def _generation_params_from_config(gen: GenerationConfig) -> GenerationParams:
    return GenerationParams(
        max_new_tokens=gen.max_new_tokens,
        num_beams=gen.num_beams,
        do_sample=gen.do_sample,
        temperature=gen.temperature,
        top_p=gen.top_p,
        top_k=gen.top_k,
        repetition_penalty=gen.repetition_penalty,
        length_penalty=gen.length_penalty,
        early_stopping=gen.early_stopping,
        no_repeat_ngram_size=gen.no_repeat_ngram_size,
    )


def _build_gumbel_from_config(config: AppConfig) -> GumbelSoftmax:
    g = config.gumbel
    schedule = None
    if g.schedule_enabled:
        schedule = LinearAnnealingTemperatureSchedule(
            start=g.temperature,
            end=g.temperature_end,
            total_steps=g.schedule_steps,
        )
    else:
        schedule = ConstantTemperatureSchedule(g.temperature)
    return GumbelSoftmax(
        temperature=g.temperature,
        hard=g.hard,
        schedule=schedule,
    )


class HumanizerModel(nn.Module):
    """Composable Final Humanizer matching the architecture specification."""

    def __init__(self, config: AppConfig, backend: Optional[BaseModelBackend] = None) -> None:
        super().__init__()
        self.config = config

        if backend is None:
            backend = build_backend(
                backend=config.model.backend,
                model_name=config.model.model_name,
                tokenizer_name=config.model.tokenizer_name,
                use_peft=config.model.use_peft,
                peft_config={
                    "r": config.model.peft.r,
                    "lora_alpha": config.model.peft.lora_alpha,
                    "lora_dropout": config.model.peft.lora_dropout,
                    "target_modules": config.model.peft.target_modules,
                },
                load_weights=False,
                generation_params=_generation_params_from_config(config.model.generation),
            )
        else:
            backend.set_generation_params(_generation_params_from_config(config.model.generation))

        self.backend = backend
        hidden = self.backend.hidden_size
        vocab = self.backend.vocab_size

        gumbel = _build_gumbel_from_config(config)
        diff_embedding = DifferentiableEmbedding.from_pretrained(
            self.backend.get_input_embeddings(),
            freeze=config.embedding.freeze,
            share_weights=config.embedding.share_weights,
        )
        self.soft_pipeline = SoftTokenPipeline(gumbel=gumbel, embedding=diff_embedding)

        # Convenience aliases used throughout the codebase / tests
        self.gumbel = gumbel
        self.diff_embedding = diff_embedding

        m = config.modules
        self.semantic_encoder = SemanticEncoder(
            input_size=hidden,
            hidden_size=m.semantic_encoder.hidden_size,
            output_size=m.semantic_encoder.output_size,
        )
        self.stylometric = StylometricModule(
            input_size=hidden,
            feature_dim=m.stylometric.feature_dim,
            hidden_size=m.stylometric.hidden_size,
        )
        self.human_style_encoder = HumanStyleEncoder(
            input_size=hidden,
            hidden_size=m.human_style_encoder.hidden_size,
            projection_dim=m.human_style_encoder.projection_dim,
        )
        self.diversity_module = DiversityBurstinessModule(
            input_size=hidden,
            window_size=m.diversity.window_size,
            hidden_size=m.diversity.hidden_size,
        )
        self.discriminator = HumanAIDiscriminator(
            input_size=hidden,
            hidden_size=m.discriminator.hidden_size,
            dropout=m.discriminator.dropout,
        )

        self._vocab_size = vocab
        self._hidden_size = hidden

    @property
    def tokenizer(self):
        return self.backend.tokenizer

    # ==================================================================
    # TRAINING PATH — differentiable soft tokens (no argmax)
    # ==================================================================

    def forward_trainable(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        gumbel_temperature: Optional[float] = None,
        gumbel_hard: Optional[bool] = None,
    ) -> HumanizerOutput:
        """Differentiable training forward.

        text ids → backbone logits → Gumbel-Softmax → P → H_soft = P E
        → downstream objective modules.
        """
        backbone_out = self.backend.forward_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        # Soft path only — hard defaults to False for gradient-friendly training.
        soft_out: SoftTokenPipelineOutput = self.soft_pipeline(
            backbone_out.logits,
            temperature=gumbel_temperature if gumbel_temperature is not None else None,
            hard=False if gumbel_hard is None else gumbel_hard,
        )

        emb_mask = self._embedding_mask(
            soft_tokens=soft_out.soft_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        embeddings = soft_out.embeddings
        soft_tokens = soft_out.soft_tokens

        return HumanizerOutput(
            logits=soft_out.logits,
            soft_tokens=soft_tokens,
            embeddings=embeddings,
            semantic=self.semantic_encoder(embeddings, emb_mask),
            stylometric=self.stylometric(
                embeddings, emb_mask, soft_tokens=soft_tokens
            ),
            human_style=self.human_style_encoder(embeddings, emb_mask),
            diversity=self.diversity_module(
                embeddings, emb_mask, soft_tokens=soft_tokens
            ),
            discriminator_logits=self.discriminator(embeddings, emb_mask),
            temperature=soft_out.temperature,
            extras={
                "backbone_loss": backbone_out.loss,
                "encoder_hidden_states": backbone_out.encoder_hidden_states,
                "decoder_hidden_states": backbone_out.hidden_states,
            },
        )

    def forward(self, *args: Any, **kwargs: Any) -> HumanizerOutput:
        """Default forward is the training (differentiable) path."""
        return self.forward_trainable(*args, **kwargs)

    def soft_embeddings_from_logits(
        self,
        logits: Tensor,
        temperature: Optional[float] = None,
        hard: bool = False,
    ) -> SoftTokenPipelineOutput:
        """Run only Gumbel → DifferentiableEmbedding (for tests / custom loops)."""
        return self.soft_pipeline(logits, temperature=temperature, hard=hard)

    def _embedding_mask(
        self,
        soft_tokens: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        decoder_input_ids: Optional[Tensor],
    ) -> Optional[Tensor]:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if decoder_input_ids is not None and pad_id is not None:
            return (decoder_input_ids != pad_id).long()
        if soft_tokens.size(1) == input_ids.size(1) and attention_mask is not None:
            return attention_mask
        return None

    # ==================================================================
    # INFERENCE PATH — discrete generation (no Gumbel / soft embeddings)
    # ==================================================================

    @torch.no_grad()
    def generate(
        self,
        texts: Union[List[str], str],
        max_new_tokens: Optional[int] = None,
        generation_params: Optional[GenerationParams] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Inference: normal discrete generation → humanized text.

        Does **not** use Gumbel-Softmax or differentiable embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.backend.tokenize(
            texts,
            max_length=self.config.model.max_input_length,
            padding=True,
            truncation=True,
        )
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        params = generation_params or self.backend.generation_params
        if max_new_tokens is not None:
            params = GenerationParams(**{**params.__dict__, "max_new_tokens": max_new_tokens})

        generated = self.backend.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            generation_params=params,
            **kwargs,
        )
        return self.backend.decode(generated)

    def step_gumbel_schedule(self, step: Optional[int] = None) -> float:
        """Advance Gumbel temperature schedule (training loops call this)."""
        return self.gumbel.step_schedule(step)
