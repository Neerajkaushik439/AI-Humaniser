#!/usr/bin/env python3
"""Training entrypoint — supervised / RL with dry-run or real training.

Examples:
  python train.py --dry-run
  python train.py --train --tiny --stage supervised --dataset smoke_pairs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import SupervisedTrainer, TrainingPipeline, build_rl_trainer
from training.data import HumanizationDataset, build_dataloader
from utils.bootstrap import build_application


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI-Humanizer (or dry-run the pipeline).")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument(
        "--stage",
        choices=["supervised", "rl", "all", "pipeline"],
        default="supervised",
        help="Which training stage to run (default: supervised).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify pipeline on a synthetic batch without training.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually run training (requires dataset unless --tiny synthetic path).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="smoke_pairs",
        help="Dataset name under DatasetRepository / data/processed (default: smoke_pairs).",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use tiny randomly-initialized backend (recommended for smoke training).",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load pretrained backbone weights (not used with --tiny).",
    )
    parser.add_argument(
        "--no-load-weights",
        action="store_true",
        help="Use randomly initialized / config-only backbone.",
    )
    return parser.parse_args()


def _build_tiny_training_app(config_path: str):
    from losses.composite import CompositeHumanizationLoss
    from models.humanizer import HumanizerModel
    from repositories import build_repositories
    from rewards.composite import CompositeReward
    from configs.config import load_config
    from models.backends.registry import build_tiny_backend
    from utils import resolve_device, set_seed, setup_logging
    from utils.bootstrap import Application
    from utils.tiny import tiny_module_config

    config = load_config(config_path)
    # Force tiny-compatible module sizes if smoke config already set them; otherwise shrink.
    if config.model.model_name in {"tiny-backend", "tiny-t5"} or "smoke" in Path(config_path).name:
        pass
    else:
        config.modules = tiny_module_config()
        config.model.model_name = "tiny-backend"
        config.model.tokenizer_name = "tiny-backend"

    setup_logging(config.project.log_dir)
    set_seed(config.project.seed)
    device = resolve_device(config.device if config.device != "auto" else "cpu")

    repositories = build_repositories(
        kind=config.repositories.kind,
        data_dir=config.repositories.data_dir,
        checkpoint_dir=config.repositories.checkpoint_dir,
        experiment_dir=config.repositories.experiment_dir,
    )
    backend = build_tiny_backend(backend=config.model.backend)
    model = HumanizerModel(config=config, backend=backend)
    model.to(device)
    return Application(
        config=config,
        repositories=repositories,
        model=model,
        loss_fn=CompositeHumanizationLoss.from_config(config),
        reward_fn=CompositeReward.from_config(config),
        device=device,
    )


def main() -> int:
    args = parse_args()
    do_train = bool(args.train) and not bool(args.dry_run)
    if not do_train and not args.dry_run and args.stage != "pipeline":
        # Explicit non-dry stage without --train still trains if user asked for supervised/rl/all
        do_train = args.stage in {"supervised", "rl", "all"}

    if args.dry_run or (not do_train and args.stage == "pipeline"):
        from utils.dry_run import run_dry_run

        print("Running training pipeline dry-run (no long training)...")
        report = run_dry_run()
        print(json.dumps({"ok": report.ok, "failed": report.errors}, indent=2))
        print("Training: NOT started.")
        return 0 if report.ok else 1

    config_path = args.config
    if args.tiny and "smoke" not in Path(config_path).name:
        smoke = ROOT / "configs" / "smoke_train.yaml"
        if smoke.exists():
            config_path = str(smoke)
            print(f"Using smoke config: {config_path}")

    if args.tiny:
        app = _build_tiny_training_app(config_path)
        print(f"Tiny backend ready | device={app.device} | params={sum(p.numel() for p in app.model.parameters()):,}")
    else:
        load_weights = bool(args.load_weights) and not bool(args.no_load_weights)
        app = build_application(config_path=config_path, load_weights=load_weights)

    # Ensure smoke dataset is visible to the repository
    dataset_name = args.dataset
    if not app.repositories.datasets.exists(dataset_name):
        processed = ROOT / "data" / "processed" / f"{dataset_name}.json"
        if processed.exists():
            import json as _json

            rows = _json.loads(processed.read_text(encoding="utf-8"))
            app.repositories.datasets.save(dataset_name, rows)
            print(f"Registered dataset '{dataset_name}' from {processed}")
        else:
            print(f"ERROR: dataset '{dataset_name}' not found in repository or {processed}")
            return 1

    ds = HumanizationDataset.from_repository(app.repositories.datasets, dataset_name)
    loader = build_dataloader(
        dataset=ds,
        backend=app.model.backend,
        batch_size=app.config.training.batch_size,
        max_input_length=app.config.model.max_input_length,
        max_output_length=app.config.model.max_output_length,
        shuffle=True,
    )
    print(f"Dataset '{dataset_name}' | examples={len(ds)} | batch_size={app.config.training.batch_size}")
    print(f"Starting training stage={args.stage} | epochs={app.config.training.epochs}")
    print("Interrupt with Ctrl+C anytime to stop mid-run.")

    if args.stage in {"supervised", "all"}:
        trainer = SupervisedTrainer(
            model=app.model,
            config=app.config,
            repositories=app.repositories,
            loss_fn=app.loss_fn,
            dataloader=loader,
        )
        print("=== Supervised training started ===")
        trainer.train()
        print("=== Supervised training finished ===")

    if args.stage in {"rl", "all"}:
        rl_trainer = build_rl_trainer(
            model=app.model,
            config=app.config,
            repositories=app.repositories,
            reward_fn=app.reward_fn,
            loss_fn=app.loss_fn,
            dataloader=loader,
            dry_run=False,
        )
        print(f"=== RL training started ({app.config.rl_algorithm}) ===")
        rl_trainer.train()
        print("=== RL training finished ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
