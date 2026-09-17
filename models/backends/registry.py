"""Backend registry — resolve ``model.backend`` string to a concrete class.

Register future backends (LongT5, Mistral, Llama) here; config switches them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from models.backends.base import BaseModelBackend, GenerationParams
from models.backends.flan_t5 import FLANT5Backend

BACKEND_REGISTRY: Dict[str, Type[BaseModelBackend]] = {
    "flan_t5": FLANT5Backend,
    "flan-t5": FLANT5Backend,
    "t5": FLANT5Backend,
}


def register_backend(name: str, cls: Type[BaseModelBackend]) -> None:
    BACKEND_REGISTRY[name.lower()] = cls


def get_backend_class(name: str) -> Type[BaseModelBackend]:
    key = name.lower()
    if key not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY))
        raise KeyError(
            f"Unknown model backend {name!r}. Registered: {available}. "
            "Implement BaseModelBackend and call register_backend()."
        )
    return BACKEND_REGISTRY[key]


def build_backend(
    backend: str,
    model_name: str,
    tokenizer_name: str | None = None,
    use_peft: bool = False,
    peft_config: dict | None = None,
    device=None,
    load_weights: bool = True,
    generation_params: Optional[GenerationParams] = None,
    **kwargs: Any,
) -> BaseModelBackend:
    cls = get_backend_class(backend)
    return cls(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        use_peft=use_peft,
        peft_config=peft_config,
        device=device,
        load_weights=load_weights,
        generation_params=generation_params,
        **kwargs,
    )


def build_tiny_backend(backend: str = "flan_t5", **kwargs: Any) -> BaseModelBackend:
    """Construct a tiny randomly-initialized backend for dry-run / tests.

    Requires the registered backend class to expose ``from_tiny(**kwargs)``.
    """
    cls = get_backend_class(backend)
    if not hasattr(cls, "from_tiny"):
        raise TypeError(
            f"Backend {backend!r} ({cls.__name__}) does not implement from_tiny(); "
            "add a tiny factory for dry-run/tests."
        )
    return cls.from_tiny(**kwargs)  # type: ignore[attr-defined]
