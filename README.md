# AI-Humanizer

Research codebase for **AI text humanization**: rewrite AI-generated text so it reads more like human writing, using a multi-loss differentiable pipeline plus optional RL (PPO / REINFORCE).

> **Status:** Software architecture and training stack are implemented and tested.  
> **The model has not been trained or evaluated yet.** Dry-run / tests use tiny randomly initialized backends (no claim of quality or detector evasion).

## Final architecture

Authoritative diagram: [`docs/architecture.png`](docs/architecture.png)

```
AI-generated text
  → Tokenizer
  → Transformer backbone (BaseModelBackend; default: FLAN-T5)
  → Gumbel-Softmax soft token distribution
  → Differentiable Embedding (H_soft = P E)
      ├─ Semantic Encoder              → Semantic Loss
      ├─ Stylometric Module            → Style Loss
      ├─ Human Style Encoder           → Contrastive Human Loss
      ├─ Diversity / Burstiness Module → Diversity Loss
      ├─ Human / AI Discriminator      → Adversarial Loss
      └─ RL Reward
            → PPO / REINFORCE
            → Final Humanizer
```

Objective (weights from config only):

```
L = λ1 L_semantic + λ2 L_style + λ3 L_contrastive
  + λ4 L_adversarial + λ5 L_diversity + λ6 L_PPL
```

`λ6` (PPL) stays small — perplexity is a regularizer, not the primary objective.

### Training vs inference

| Path | Flow |
|------|------|
| **Training** | text → logits → Gumbel-Softmax → soft `P` → `H_soft = P E` → losses / rewards |
| **Inference** | text → `backend.generate` → discrete tokens → text |

Inference never routes through Gumbel-Softmax. Training does **not** use `argmax` before differentiable losses (soft mode).

## Project structure

```
AI-Humanizer/
├── configs/           # YAML + typed AppConfig
├── models/
│   ├── backends/      # BaseModelBackend + FLANT5Backend (+ registry)
│   ├── gumbel_softmax.py
│   ├── differentiable_embedding.py
│   ├── pipeline.py    # SoftTokenPipeline (training soft path)
│   ├── humanizer.py   # Final Humanizer wiring
│   └── *_encoder / stylometric / diversity / discriminator
├── losses/            # BaseLoss registry + CompositeHumanizationLoss
├── rewards/           # BaseRewardComponent + CompositeReward
├── training/          # supervised, PPO, REINFORCE, pipeline, checkpoints, metrics, data
├── repositories/      # Dataset / Experiment / Checkpoint interfaces + local/memory
├── utils/             # bootstrap, tiny backend helpers, dry-run
├── tests/
├── data/              # raw / processed (local files)
├── checkpoints/
├── outputs/
├── main.py            # init + --dry-run
├── train.py           # training entry (dry-run by default)
├── inference.py       # discrete generation
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

All hyperparameters live in [`configs/config.yaml`](configs/config.yaml) (loaded by [`configs/config.py`](configs/config.py)):

- model backend / names / lengths / generation
- Gumbel temperature / hard / schedule
- loss λ weights (`lambda_semantic`, …, `lambda_ppl`)
- reward λ weights
- training + RL (PPO / REINFORCE) hyperparameters
- repository kind (`local` | `memory`)

**Do not hard-code λ, τ, or learning rates in trainers.**

## Model switching

FLAN-T5-specific code lives in `models/backends/flan_t5.py` and the registry.

Everything else depends on `BaseModelBackend`:

1. Implement a new backend subclassing `BaseModelBackend`
2. `register_backend("mistral", MistralBackend)` in `models/backends/registry.py`
3. Set in config:

```yaml
model:
  backend: mistral
  model_name: ...
  tokenizer_name: ...
```

Losses, rewards, trainers, RL, repositories, and inference stay unchanged.

## Gumbel-Softmax

`models/gumbel_softmax.py` maps token logits → soft vocabulary distribution `P` with configurable temperature (and optional annealing). Soft mode keeps the path differentiable for training.

## Differentiable embedding

`models/differentiable_embedding.py` computes:

```
H_soft = P @ E
```

so gradients flow: loss → `H_soft` → `P` → logits → backbone.

## Six-loss architecture

Independent modules under `losses/`, composed by `CompositeHumanizationLoss`:

| Name | Class |
|------|--------|
| semantic | `SemanticLoss` |
| style | `StyleLoss` |
| contrastive | `ContrastiveHumanLoss` |
| adversarial | `AdversarialLoss` |
| diversity | `DiversityLoss` |
| ppl | `PPLLoss` |

Extensibility:

```python
@register_loss
class NewLoss(BaseLoss):
    name = "new_loss"
    def forward(self, **batch): ...
```

```yaml
losses:
  lambda_new_loss: 0.1
```

Weight `0` disables a term’s contribution to `total_loss` while still logging the unweighted value. Trainers are not rewritten.

## Reward architecture

`BaseRewardComponent` + `CompositeReward` (config weights):

- `SemanticReward`, `StyleReward`, `HumanStyleReward`
- `DiversityReward`, `DiscriminatorReward` (`lambda_human` alias)
- optional `PPLReward`

PPO / REINFORCE call `CompositeReward` — they do not hard-code the reward formula.

## PPO / REINFORCE

```yaml
rl:
  algorithm: ppo   # or reinforce
```

- `BaseRLTrainer` uses `model.backend: BaseModelBackend` only
- `PPOTrainer` (default) / `ReinforceTrainer`
- TRL may be present; the public API does not require TRL types

## Database abstraction

No MongoDB / PostgreSQL / MySQL dependency.

Interfaces: `DatasetRepository`, `ExperimentRepository`, `CheckpointRepository`  
Initial adapters: local filesystem + in-memory (`repositories/local.py`).

Local datasets: JSON / JSONL / CSV via `DatasetRepository`.

Future `MongoDBRepository` / `PostgresRepository` can plug in without rewriting trainers.

## Datasets (planned — not auto-downloaded)

Configured in [`configs/datasets.yaml`](configs/datasets.yaml) and adapters in [`data_sources/`](data_sources/):

| Need | Dataset |
|------|---------|
| Human style (short) | Blog Authorship Corpus |
| Human style (long) | PG-19 |
| AI vs human | RAID, MAGE |
| Paraphrase / meaning | PAR-3 |
| Rough AI↔human pairs | HC3 |

There is **no** ready-made AI→humanized mapping corpus. A later step (disabled for now) will rewrite human text with AI to build pairs. See [`data/README.md`](data/README.md).

## Dry-run (no training)


Uses a **tiny** randomly initialized backend (no large download):

```bash
python main.py --dry-run
```

Verifies configuration, backend, tokenizer, Gumbel-Softmax, differentiable embedding, all loss/reward modules, composite loss/reward, RL trainer, checkpoint manager, logging, and a short inference generation call.

Also:

```bash
python train.py --dry-run --no-load-weights
```

Config-only init (may fetch HF config if online):

```bash
python main.py
```

## Tests

```bash
pytest
```

Tests use tiny / mock backends. They do **not** download `google/flan-t5-base` weights.

## Future training (not run yet)

1. Place data under `data/raw/` or `data/processed/` (JSON / JSONL / CSV)
2. Set λ / RL hyperparameters in `configs/config.yaml`
3. Run for real (when ready):

```bash
python train.py --train --dataset <name> --stage all --load-weights
```

Expect to iterate on data quality, λ balances, and backbone choice (FLAN-T5 → LongT5 / Mistral / … via config).

## License / research note

This repository is a research scaffolding project. It does not claim to defeat AI detectors or to produce production-ready humanization without further training and evaluation.
