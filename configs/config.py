"""Centralized configuration system for AI-Humanizer.

All hyperparameters live here (or in config.yaml). Nothing should hard-code
model names, λ weights, Gumbel settings, or RL hyperparameters elsewhere.

Loss / reward weights are stored as open-ended dictionaries so new terms
(e.g. ``lambda_new_loss``) can be added in YAML without rewriting trainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Union

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _deep_update(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_update(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value
    return base


@dataclass
class ProjectConfig:
    name: str = "AI-Humanizer"
    seed: int = 42
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "outputs/logs"


@dataclass
class PeftConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None


@dataclass
class GenerationConfig:
    """Discrete generation hyperparameters (inference path only)."""

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


@dataclass
class ModelConfig:
    """Backend-agnostic model settings.

    Switch ``backend`` / ``model_name`` to move from FLAN-T5 → LongT5 / Mistral / Llama
    without touching losses, rewards, or trainers.
    """

    backend: str = "flan_t5"
    model_name: str = "google/flan-t5-base"
    tokenizer_name: str = "google/flan-t5-base"
    max_input_length: int = 512
    max_output_length: int = 512
    use_peft: bool = False
    peft: PeftConfig = field(default_factory=PeftConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class GumbelConfig:
    """Gumbel-Softmax settings — temperature MUST be configured here, not hard-coded."""

    temperature: float = 1.0
    hard: bool = False
    # Optional annealing for future training schedules
    schedule_enabled: bool = False
    temperature_end: float = 0.5
    schedule_steps: int = 10000


@dataclass
class EmbeddingConfig:
    freeze: bool = False
    share_weights: bool = True  # tie DifferentiableEmbedding to backbone E


@dataclass
class SemanticEncoderModuleConfig:
    hidden_size: int = 768
    output_size: int = 768


@dataclass
class StylometricModuleConfig:
    feature_dim: int = 64
    hidden_size: int = 256


@dataclass
class HumanStyleEncoderModuleConfig:
    hidden_size: int = 768
    projection_dim: int = 256


@dataclass
class DiversityModuleConfig:
    window_size: int = 16
    hidden_size: int = 128


@dataclass
class DiscriminatorModuleConfig:
    hidden_size: int = 256
    dropout: float = 0.1


@dataclass
class ModulesConfig:
    semantic_encoder: SemanticEncoderModuleConfig = field(default_factory=SemanticEncoderModuleConfig)
    stylometric: StylometricModuleConfig = field(default_factory=StylometricModuleConfig)
    human_style_encoder: HumanStyleEncoderModuleConfig = field(
        default_factory=HumanStyleEncoderModuleConfig
    )
    diversity: DiversityModuleConfig = field(default_factory=DiversityModuleConfig)
    discriminator: DiscriminatorModuleConfig = field(default_factory=DiscriminatorModuleConfig)


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    batch_size: int = 8
    epochs: int = 3
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    warmup_ratio: float = 0.03


@dataclass
class PPOConfig:
    cliprange: float = 0.2
    vf_coef: float = 0.1
    ent_coef: float = 0.01
    gamma: float = 1.0
    lam: float = 0.95
    kl_penalty: float = 0.05


@dataclass
class ReinforceConfig:
    gamma: float = 0.99
    baseline: str = "moving_average"
    entropy_coef: float = 0.01


@dataclass
class RLConfig:
    algorithm: str = "ppo"
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 1
    epochs: int = 1
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reinforce: ReinforceConfig = field(default_factory=ReinforceConfig)


@dataclass
class RepositoryConfig:
    kind: str = "local"
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    experiment_dir: str = "outputs/experiments"


@dataclass
class PairConstructionConfig:
    """Future human→AI rewrite pairing (disabled until you implement it)."""

    enabled: bool = False
    strategy: str = "human_to_ai_rewrite"


@dataclass
class DatasetsConfig:
    """Which corpora are planned for training (see configs/datasets.yaml)."""

    catalog_path: str = "configs/datasets.yaml"
    data_root: str = "data"
    active: List[str] = field(default_factory=list)
    pair_construction: PairConstructionConfig = field(default_factory=PairConstructionConfig)


@dataclass
class AppConfig:
    """Root configuration object injected throughout the system."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    gumbel: GumbelConfig = field(default_factory=GumbelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    modules: ModulesConfig = field(default_factory=ModulesConfig)
    # Extensible weight maps — new λ_* keys are discovered automatically.
    losses: Dict[str, float] = field(
        default_factory=lambda: {
            "lambda_semantic": 1.0,
            "lambda_style": 1.0,
            "lambda_contrastive": 1.0,
            "lambda_adversarial": 0.5,
            "lambda_diversity": 0.5,
            "lambda_ppl": 0.05,
        }
    )
    rewards: Dict[str, float] = field(
        default_factory=lambda: {
            "lambda_semantic": 1.0,
            "lambda_style": 1.0,
            "lambda_human_style": 1.0,
            "lambda_human": 1.0,
            "lambda_discriminator": 1.0,
            "lambda_diversity": 0.5,
            "lambda_ppl": 0.05,
        }
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    repositories: RepositoryConfig = field(default_factory=RepositoryConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)

    # Convenience aliases expected by the brief
    @property
    def model_backend(self) -> str:
        return self.model.backend

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def tokenizer_name(self) -> str:
        return self.model.tokenizer_name

    @property
    def max_input_length(self) -> int:
        return self.model.max_input_length

    @property
    def max_output_length(self) -> int:
        return self.model.max_output_length

    @property
    def learning_rate(self) -> float:
        return self.training.learning_rate

    @property
    def batch_size(self) -> int:
        return self.training.batch_size

    @property
    def epochs(self) -> int:
        return self.training.epochs

    @property
    def gumbel_temperature(self) -> float:
        return self.gumbel.temperature

    @property
    def gumbel_hard(self) -> bool:
        return self.gumbel.hard

    @property
    def rl_algorithm(self) -> str:
        return self.rl.algorithm

    def loss_weight(self, name: str, default: float = 0.0) -> float:
        """Resolve ``lambda_<name>`` or raw ``name`` from the losses map."""
        if name in self.losses:
            return float(self.losses[name])
        key = name if name.startswith("lambda_") else f"lambda_{name}"
        return float(self.losses.get(key, default))

    def reward_weight(self, name: str, default: float = 0.0) -> float:
        if name in self.rewards:
            return float(self.rewards[name])
        key = name if name.startswith("lambda_") else f"lambda_{name}"
        return float(self.rewards.get(key, default))

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain_dict(self)


def _to_plain_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _to_plain_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    return obj


def _from_mapping(cls: type, data: Optional[Mapping[str, Any]]) -> Any:
    if data is None:
        return cls()
    kwargs: Dict[str, Any] = {}
    field_map = {f.name: f for f in fields(cls)}
    for name, f in field_map.items():
        if name not in data:
            continue
        value = data[name]
        nested_type = f.type
        # Handle string annotations for nested dataclasses
        if isinstance(nested_type, str):
            nested_type = eval(nested_type, globals())  # noqa: S307 — controlled local types
        if hasattr(nested_type, "__dataclass_fields__") and isinstance(value, Mapping):
            kwargs[name] = _from_mapping(nested_type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> AppConfig:
    """Load ``config.yaml``, apply optional overrides, return ``AppConfig``."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config at {config_path} must be a mapping.")
            raw = loaded
    if overrides:
        _deep_update(raw, dict(overrides))

    return AppConfig(
        project=_from_mapping(ProjectConfig, raw.get("project")),
        device=str(raw.get("device", "auto")),
        model=_from_mapping(ModelConfig, raw.get("model")),
        gumbel=_from_mapping(GumbelConfig, raw.get("gumbel")),
        embedding=_from_mapping(EmbeddingConfig, raw.get("embedding")),
        modules=_from_mapping(ModulesConfig, raw.get("modules")),
        losses=dict(raw.get("losses") or AppConfig().losses),
        rewards=dict(raw.get("rewards") or AppConfig().rewards),
        training=_from_mapping(TrainingConfig, raw.get("training")),
        rl=_from_mapping(RLConfig, raw.get("rl")),
        repositories=_from_mapping(RepositoryConfig, raw.get("repositories")),
        datasets=_from_mapping(DatasetsConfig, raw.get("datasets")),
    )


def save_config(config: AppConfig, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_to_plain_dict(config), fh, sort_keys=False, default_flow_style=False)
