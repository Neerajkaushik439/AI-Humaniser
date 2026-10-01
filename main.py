#!/usr/bin/env python3
"""AI-Humanizer entrypoint — initialize architecture & optional dry-run (no training)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize AI-Humanizer or run a full-stack dry-run (no training)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Download/load pretrained backbone weights (off by default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Verify config, tiny backend, Gumbel, embeddings, losses, rewards, "
            "RL trainer, checkpoints, and logging without training."
        ),
    )
    return parser.parse_args()


def _print_init_summary(app) -> None:
    cfg = app.config
    health = app.repositories.healthcheck()
    loss_weights = app.loss_fn.active_weights()

    print("=" * 64)
    print(f"  {cfg.project.name} — initialization OK")
    print("=" * 64)
    print("  Objective: L = Σ λ_i L_i  (CompositeHumanizationLoss)")
    print(f"  device            : {app.device}")
    print(f"  model_backend     : {cfg.model_backend}")
    print(f"  model_name        : {cfg.model_name}")
    print(f"  tokenizer_name    : {cfg.tokenizer_name}")
    print(f"  max_input_length  : {cfg.max_input_length}")
    print(f"  max_output_length : {cfg.max_output_length}")
    print(f"  learning_rate     : {cfg.learning_rate}")
    print(f"  batch_size        : {cfg.batch_size}")
    print(f"  epochs            : {cfg.epochs}")
    print(f"  gumbel_temperature: {cfg.gumbel_temperature}")
    print(f"  gumbel_hard       : {cfg.gumbel_hard}")
    print(f"  rl_algorithm      : {cfg.rl_algorithm}")
    print("-" * 64)
    print("  Loss weights (architecture objective):")
    for name, weight in loss_weights.items():
        print(f"    λ_{name:<14} = {weight}")
    print("-" * 64)
    print("  Repository health:")
    for name, ok in health.items():
        print(f"    {name:<14}: {'ok' if ok else 'FAIL'}")
    print("-" * 64)
    print("  Pipeline modules:")
    print("    Tokenizer → Transformer Backbone → Gumbel-Softmax")
    print("    → Differentiable Embedding →")
    print("      Semantic / Stylometric / Human Style / Diversity / Discriminator")
    print("    → RL Reward → PPO/REINFORCE → Final Humanizer")
    print("-" * 64)
    n_params = sum(p.numel() for p in app.model.parameters())
    print(f"  Humanizer parameters: {n_params:,}")
    print("  Training: NOT started.")
    print("=" * 64)


def _print_dry_run(report) -> int:
    print("=" * 64)
    print("  AI-Humanizer — DRY RUN")
    print("=" * 64)
    print("  Uses a tiny randomly-initialized backend (no pretrained download).")
    print("  Training is NOT started.")
    print("-" * 64)
    width = max(len(k) for k in report.checks) if report.checks else 10
    for name, passed in report.checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:<{width}}")
    print("-" * 64)
    if report.details.get("configuration"):
        cfg = report.details["configuration"]
        print(f"  backend        : {cfg.get('backend')}")
        print(f"  gumbel τ       : {cfg.get('gumbel_temperature')}")
        print(f"  rl_algorithm   : {cfg.get('rl_algorithm')}")
    print(f"  overall        : {'OK' if report.ok else 'FAILED'}")
    if report.errors:
        print("  failed checks  :", ", ".join(report.errors))
    print("=" * 64)
    # Machine-readable summary for scripts
    print(json.dumps({"ok": report.ok, "failed": report.errors}, indent=2))
    return 0 if report.ok else 1


def main() -> int:
    args = parse_args()

    if args.dry_run:
        from utils.dry_run import run_dry_run

        report = run_dry_run()
        return _print_dry_run(report)

    from utils.bootstrap import build_application

    app = build_application(config_path=args.config, load_weights=args.load_weights)
    _print_init_summary(app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
