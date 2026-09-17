"""FLAN-T5 Transformer backbone backend.

Default initial backbone for AI-Humanizer. All FLAN-T5 / T5 specifics live here;
swap via ``model.backend`` in config without touching the rest of the pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config, T5ForConditionalGeneration

from models.backends.base import BackendOutput, BaseModelBackend, GenerationParams

logger = logging.getLogger("ai_humanizer.backends.flan_t5")


def tiny_t5_config(
    vocab_size: int = 128,
    d_model: int = 32,
    num_layers: int = 1,
    num_heads: int = 4,
    d_ff: int = 64,
) -> T5Config:
    """Tiny T5 config for unit tests (no downloads)."""
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    return T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_kv=d_model // num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=num_heads,
        relative_attention_num_buckets=8,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )


def _default_t5_config(vocab_size: int = 32128) -> T5Config:
    return T5Config(
        vocab_size=vocab_size,
        d_model=512,
        d_kv=64,
        d_ff=1024,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )


class FLANT5Backend(BaseModelBackend):
    """Hugging Face FLAN-T5 / T5 seq2seq backend."""

    backend_name = "flan_t5"

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        tokenizer_name: Optional[str] = None,
        use_peft: bool = False,
        peft_config: Optional[dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        load_weights: bool = True,
        generation_params: Optional[GenerationParams] = None,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self._load_weights = load_weights
        self._using_bootstrap = False

        if generation_params is not None:
            self.set_generation_params(generation_params)

        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = self._load_tokenizer(self.tokenizer_name)

        if model is not None:
            self.model = model
        elif load_weights:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            if use_peft:
                self.model = self._wrap_peft(self.model, peft_config or {})
        else:
            self.model = self._build_untrained_model(model_name)

        self._vocab_size = int(self.model.config.vocab_size)
        self._hidden_size = int(self.model.config.d_model)

        if device is not None:
            self.to_device(device)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_tiny(
        cls,
        vocab_size: int = 128,
        d_model: int = 32,
        device: Optional[Union[str, torch.device]] = None,
        generation_params: Optional[GenerationParams] = None,
    ) -> "FLANT5Backend":
        """Build a tiny randomly-initialized backend for tests (no HF download)."""
        config = tiny_t5_config(vocab_size=vocab_size, d_model=d_model)
        model = T5ForConditionalGeneration(config)
        tokenizer = _BootstrapTokenizer(vocab_size=vocab_size)
        return cls(
            model_name="tiny-t5",
            tokenizer_name="tiny-t5",
            load_weights=False,
            device=device,
            generation_params=generation_params or GenerationParams(max_new_tokens=8),
            model=model,
            tokenizer=tokenizer,
        )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_tokenizer(self, name: str):
        try:
            return AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # noqa: BLE001 — bootstrap must not fail offline
            logger.warning(
                "Could not load tokenizer %s (%s). Using bootstrap tokenizer.",
                name,
                exc,
            )
            return _BootstrapTokenizer(vocab_size=32128)

    def _build_untrained_model(self, model_name: str):
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name)
            return AutoModelForSeq2SeqLM.from_config(config)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not fetch config for %s (%s). Using local T5Config bootstrap.",
                model_name,
                exc,
            )
            self._using_bootstrap = True
            return AutoModelForSeq2SeqLM.from_config(_default_t5_config())

    @staticmethod
    def _wrap_peft(model: nn.Module, peft_cfg: dict) -> nn.Module:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            r=int(peft_cfg.get("r", 8)),
            lora_alpha=int(peft_cfg.get("lora_alpha", 16)),
            lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
            target_modules=peft_cfg.get("target_modules") or ["q", "v"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        return get_peft_model(model, lora)

    # ------------------------------------------------------------------
    # Interface properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def is_encoder_decoder(self) -> bool:
        return True

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return self.model.get_output_embeddings()

    # ------------------------------------------------------------------
    # Training-path logits
    # ------------------------------------------------------------------

    def _decoder_start_ids(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        bos = self.model.config.decoder_start_token_id
        if bos is None:
            bos = getattr(self._tokenizer, "pad_token_id", 0) or 0
        return torch.full((batch_size, 1), fill_value=int(bos), dtype=dtype, device=device)

    def forward_logits(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> BackendOutput:
        """Run FLAN-T5 and return decoder token logits ``[B, T, V]``.

        Provide ``decoder_input_ids`` (or ``labels``) for full-sequence teacher
        forcing during training. If neither is given, a single decoder start
        token is used (useful for smoke tests).
        """
        if decoder_input_ids is None and labels is None:
            decoder_input_ids = self._decoder_start_ids(
                input_ids.size(0), input_ids.device, input_ids.dtype
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        decoder_hidden = None
        if getattr(outputs, "decoder_hidden_states", None) is not None:
            decoder_hidden = outputs.decoder_hidden_states[-1]

        encoder_hidden = None
        if getattr(outputs, "encoder_last_hidden_state", None) is not None:
            encoder_hidden = outputs.encoder_last_hidden_state
        elif getattr(outputs, "encoder_hidden_states", None) is not None:
            encoder_hidden = outputs.encoder_hidden_states[-1]

        return BackendOutput(
            logits=outputs.logits,
            hidden_states=decoder_hidden,
            encoder_hidden_states=encoder_hidden,
            loss=outputs.loss,
            past_key_values=outputs.past_key_values,
            attentions=getattr(outputs, "decoder_attentions", None),
            extras={},
        )

    # ------------------------------------------------------------------
    # Inference-path generation (discrete tokens — no Gumbel)
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        generation_params: Optional[GenerationParams] = None,
        **kwargs: Any,
    ) -> Tensor:
        params = generation_params or self.generation_params
        gen_kwargs = params.to_kwargs()
        gen_kwargs.update(kwargs)
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        # Tokenizer may be a bootstrap stub without save_pretrained.
        if hasattr(self._tokenizer, "save_pretrained"):
            self._tokenizer.save_pretrained(path)
        meta = {
            "backend": self.backend_name,
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "generation_params": self.generation_params.__dict__,
        }
        (path / "ai_humanizer_backend.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    def load_checkpoint(self, path: Union[str, Path], map_location: Optional[str] = None) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        device = map_location or str(self.device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.model.to(device)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception:  # noqa: BLE001
            logger.warning("Tokenizer not found in checkpoint; keeping existing tokenizer.")
        self._vocab_size = int(self.model.config.vocab_size)
        self._hidden_size = int(self.model.config.d_model)
        meta_path = path / "ai_humanizer_backend.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            gp = meta.get("generation_params")
            if isinstance(gp, dict):
                self.set_generation_params(GenerationParams(**gp))


class _BootstrapTokenizer:
    """Tiny stand-in when Hugging Face tokenizer assets are unavailable."""

    def __init__(self, vocab_size: int = 32128) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.bos_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def __call__(
        self,
        texts,
        max_length=None,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ):
        if isinstance(texts, str):
            texts = [texts]
        max_length = max_length or 32
        rows = []
        for text in texts:
            ids = [((ord(ch) % (self.vocab_size - 10)) + 10) for ch in text[: max_length - 1]]
            ids.append(self.eos_token_id)
            if truncation:
                ids = ids[:max_length]
            rows.append(ids)
        if padding:
            width = max(len(r) for r in rows) if rows else max_length
            rows = [r + [self.pad_token_id] * (width - len(r)) for r in rows]
        input_ids = torch.tensor(rows, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        chars = []
        for tid in token_ids:
            if skip_special_tokens and tid in {self.pad_token_id, self.eos_token_id}:
                continue
            chars.append(chr(32 + (int(tid) % 95)))
        return "".join(chars)

    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]
