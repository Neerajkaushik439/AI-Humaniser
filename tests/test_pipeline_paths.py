"""Tests separating training (soft) vs inference (discrete) paths."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import AppConfig, EmbeddingConfig, GumbelConfig, ModelConfig, ModulesConfig
from models.backends.flan_t5 import FLANT5Backend
from models.humanizer import HumanizerModel


def _tiny_humanizer() -> HumanizerModel:
    backend = FLANT5Backend.from_tiny(vocab_size=64, d_model=32)
    # Keep module dims small so Linear layers match hidden_size=32
    modules = ModulesConfig()
    modules.semantic_encoder.hidden_size = 32
    modules.semantic_encoder.output_size = 32
    modules.human_style_encoder.hidden_size = 32
    modules.human_style_encoder.projection_dim = 16
    modules.stylometric.hidden_size = 32
    modules.stylometric.feature_dim = 16
    modules.diversity.hidden_size = 16
    modules.discriminator.hidden_size = 32

    config = AppConfig(
        model=ModelConfig(backend="flan_t5", model_name="tiny-t5", tokenizer_name="tiny-t5"),
        gumbel=GumbelConfig(temperature=0.9, hard=False),
        embedding=EmbeddingConfig(freeze=False, share_weights=True),
        modules=modules,
    )
    return HumanizerModel(config=config, backend=backend)


def test_training_path_uses_soft_tokens_not_argmax():
    model = _tiny_humanizer()
    model.train()
    input_ids = torch.randint(3, 60, (2, 5))
    decoder_input_ids = torch.randint(3, 60, (2, 6))
    out = model.forward_trainable(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        decoder_input_ids=decoder_input_ids,
    )
    assert out.soft_tokens.shape == out.logits.shape
    # Soft distribution: not a hard one-hot (many values in (0,1))
    fractional = ((out.soft_tokens > 0.01) & (out.soft_tokens < 0.99)).any().item()
    assert fractional
    assert out.embeddings.shape == (2, 6, model.backend.hidden_size)
    assert abs(out.temperature - 0.9) < 1e-6


def test_inference_path_does_not_require_gumbel():
    model = _tiny_humanizer()
    model.eval()
    texts = model.generate("hello world", max_new_tokens=4)
    assert isinstance(texts, list) and len(texts) == 1
    assert isinstance(texts[0], str)


def test_end_to_end_gradient_through_humanizer_training_path():
    model = _tiny_humanizer()
    model.train()
    input_ids = torch.randint(3, 60, (2, 4))
    decoder_input_ids = torch.randint(3, 60, (2, 5))
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        decoder_input_ids=decoder_input_ids,
    )
    loss = out.embeddings.pow(2).mean() + out.semantic.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.backend.parameters())
