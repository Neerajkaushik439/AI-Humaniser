"""Abstract Transformer backbone interface.

All application code talks to ``BaseModelBackend``. Concrete backends
(FLAN-T5, LongT5, Mistral, Llama, …) encapsulate Hugging Face specifics
so loss modules / trainers never import a particular model class.

The interface is intentionally generic: backends may be encoder-decoder or
decoder-only. Callers must not assume identical internal layouts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class GenerationParams:
    """Backend-agnostic generation hyperparameters (from config)."""

    max_new_tokens: int = 128
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    no_repeat_ngram_size: int = 0

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            kwargs["top_k"] = self.top_k
        if self.no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        return kwargs


@dataclass
class BackendOutput:
    """Normalized forward output shared by every backbone."""

    logits: Tensor  # [batch, seq, vocab]
    hidden_states: Optional[Tensor] = None  # [batch, seq, hidden]
    encoder_hidden_states: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    past_key_values: Optional[Any] = None
    attentions: Optional[Any] = None
    extras: Optional[Dict[str, Any]] = field(default_factory=dict)


class BaseModelBackend(ABC, nn.Module):
    """Swappable Transformer backbone for training, inference, and RL."""

    backend_name: str = "base"

    def __init__(self) -> None:
        super().__init__()
        self._generation_params = GenerationParams()

    # ------------------------------------------------------------------
    # Required properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        ...

    @property
    def is_encoder_decoder(self) -> bool:
        """Whether this backbone uses a separate encoder + decoder."""
        return False

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def generation_params(self) -> GenerationParams:
        return self._generation_params

    def set_generation_params(self, params: GenerationParams) -> None:
        self._generation_params = params

    # ------------------------------------------------------------------
    # Embeddings (for DifferentiableEmbedding / soft token path)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_input_embeddings(self) -> nn.Embedding:
        """Token embedding module used to build H_soft = P E."""

    def get_embedding_matrix(self) -> Tensor:
        """Return the embedding weight matrix ``E`` of shape ``[V, D]``."""
        return self.get_input_embeddings().weight

    def get_output_embeddings(self) -> Optional[nn.Module]:
        """LM head / output projection if the backend exposes one."""
        return None

    # ------------------------------------------------------------------
    # Core forward — token logits for Gumbel-Softmax
    # ------------------------------------------------------------------

    @abstractmethod
    def forward_logits(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> BackendOutput:
        """Return vocabulary logits ``[B, T, V]`` (and optional extras).

        This is the training-path entry point that feeds Gumbel-Softmax.
        Backends decide how encoder/decoder (or decoder-only) internals map
        onto this contract.
        """

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> BackendOutput:
        """Alias for ``forward_logits`` (kept for call-site clarity)."""
        return self.forward_logits(input_ids, attention_mask=attention_mask, **kwargs)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> BackendOutput:
        return self.forward_logits(input_ids, attention_mask=attention_mask, **kwargs)

    # ------------------------------------------------------------------
    # Discrete generation (inference path — no Gumbel)
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        generation_params: Optional[GenerationParams] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Autoregressive discrete token generation for inference."""

    # ------------------------------------------------------------------
    # RL helpers (generic; backends may override for efficiency)
    # ------------------------------------------------------------------

    def log_probs_from_logits(self, logits: Tensor, token_ids: Tensor) -> Tensor:
        """Gather per-token log-probs for RL (PPO / REINFORCE).

        Args:
            logits: ``[B, T, V]``
            token_ids: ``[B, T]``
        Returns:
            ``[B, T]`` log probabilities of ``token_ids``.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(
        self,
        texts: Union[str, Sequence[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, Tensor]:
        tok = self.tokenizer
        encoded = tok(
            list(texts) if isinstance(texts, (list, tuple)) else texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return {k: v for k, v in encoded.items()}

    def decode(self, token_ids: Tensor, skip_special_tokens: bool = True) -> List[str]:
        if token_ids.ndim == 1:
            return [self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)]
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Device / checkpoint I/O
    # ------------------------------------------------------------------

    def to_device(self, device: Union[str, torch.device]) -> "BaseModelBackend":
        device = torch.device(device)
        self.to(device)
        return self

    @abstractmethod
    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """Persist model (+ tokenizer when applicable) to ``path``."""

    @abstractmethod
    def load_checkpoint(self, path: Union[str, Path], map_location: Optional[str] = None) -> None:
        """Restore weights from a checkpoint directory or file."""
