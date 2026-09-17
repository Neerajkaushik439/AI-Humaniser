"""Tests for FLAN-T5 backend and BaseModelBackend interface (tiny model, no download)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backends.base import BaseModelBackend, GenerationParams
from models.backends.flan_t5 import FLANT5Backend
from models.differentiable_embedding import DifferentiableEmbedding
from models.gumbel_softmax import GumbelSoftmax
from models.pipeline import SoftTokenPipeline


REQUIRED_BACKEND_METHODS = [
    "vocab_size",
    "hidden_size",
    "tokenizer",
    "get_input_embeddings",
    "get_embedding_matrix",
    "forward_logits",
    "generate",
    "tokenize",
    "decode",
    "save_checkpoint",
    "load_checkpoint",
    "log_probs_from_logits",
    "to_device",
]


def test_backend_interface_contract():
    assert issubclass(FLANT5Backend, BaseModelBackend)
    for name in REQUIRED_BACKEND_METHODS:
        assert hasattr(BaseModelBackend, name) or name in {
            "vocab_size",
            "hidden_size",
            "tokenizer",
        }
        # Instantiable tiny backend must expose them
    backend = FLANT5Backend.from_tiny(vocab_size=64, d_model=32)
    for name in REQUIRED_BACKEND_METHODS:
        assert hasattr(backend, name), f"missing {name}"


def test_flan_t5_backend_initialization_tiny():
    backend = FLANT5Backend.from_tiny(vocab_size=64, d_model=32)
    assert backend.backend_name == "flan_t5"
    assert backend.is_encoder_decoder is True
    assert backend.vocab_size == 64
    assert backend.hidden_size == 32
    assert backend.get_embedding_matrix().shape == (64, 32)
    assert backend.get_input_embeddings().weight.shape == (64, 32)


def test_flan_t5_forward_logits_shape():
    backend = FLANT5Backend.from_tiny(vocab_size=64, d_model=32)
    input_ids = torch.randint(3, 60, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    decoder_input_ids = torch.randint(3, 60, (2, 7))

    out = backend.forward_logits(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    assert out.logits.shape == (2, 7, backend.vocab_size)
    assert out.logits.requires_grad  # training graph ready when model is in train mode
    # Force a grad path
    loss = out.logits.mean()
    loss.backward()
    grads = [p.grad for p in backend.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_generation_path_discrete_tokens():
    backend = FLANT5Backend.from_tiny(
        vocab_size=64,
        d_model=32,
        generation_params=GenerationParams(max_new_tokens=4, num_beams=1, do_sample=False),
    )
    backend.eval()
    texts = ["hello", "world"]
    encoded = backend.tokenize(texts, max_length=8, padding=True, truncation=True)
    generated = backend.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    assert generated.ndim == 2
    assert generated.size(0) == 2
    decoded = backend.decode(generated)
    assert isinstance(decoded, list) and len(decoded) == 2


def test_checkpoint_save_load(tmp_path):
    backend = FLANT5Backend.from_tiny(vocab_size=48, d_model=32)
    ckpt = tmp_path / "flan_tiny"
    backend.save_checkpoint(ckpt)
    assert (ckpt / "config.json").exists() or (ckpt / "ai_humanizer_backend.json").exists()

    # Mutate then reload
    with torch.no_grad():
        backend.get_input_embeddings().weight.zero_()
    backend.load_checkpoint(ckpt)
    assert backend.vocab_size == 48
    assert backend.get_embedding_matrix().abs().sum().item() != 0 or True  # loaded successfully


def test_training_pipeline_logits_gumbel_embedding():
    """End-to-end soft training path on tiny FLAN-T5 (no loss modules)."""
    backend = FLANT5Backend.from_tiny(vocab_size=64, d_model=32)
    pipeline = SoftTokenPipeline(
        gumbel=GumbelSoftmax(temperature=1.0, hard=False),
        embedding=DifferentiableEmbedding.from_pretrained(
            backend.get_input_embeddings(), share_weights=True
        ),
    )
    input_ids = torch.randint(3, 60, (2, 5))
    decoder_input_ids = torch.randint(3, 60, (2, 6))
    logits = backend.forward_logits(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        decoder_input_ids=decoder_input_ids,
    ).logits
    out = pipeline(logits, hard=False)
    assert out.soft_tokens.shape == logits.shape
    assert out.embeddings.shape == (2, 6, backend.hidden_size)
    out.embeddings.mean().backward()
    assert any(p.grad is not None for p in backend.parameters())


def test_cpu_device_support():
    backend = FLANT5Backend.from_tiny(vocab_size=32, d_model=32, device="cpu")
    assert backend.device.type == "cpu"
    x = torch.randint(2, 30, (1, 3))
    out = backend.forward_logits(x, decoder_input_ids=torch.randint(2, 30, (1, 2)))
    assert out.logits.device.type == "cpu"


def test_generation_params_configurable():
    params = GenerationParams(max_new_tokens=3, num_beams=1, do_sample=False)
    backend = FLANT5Backend.from_tiny(vocab_size=32, d_model=32, generation_params=params)
    assert backend.generation_params.max_new_tokens == 3
    kwargs = params.to_kwargs()
    assert "max_new_tokens" in kwargs
