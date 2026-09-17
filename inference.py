#!/usr/bin/env python3
"""Inference entrypoint for the Final Humanizer (discrete generation path)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.bootstrap import build_application


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI-Humanizer inference.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--text", type=str, required=True, help="AI-generated text to humanize.")
    parser.add_argument("--load-weights", action="store_true", help="Load pretrained weights.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = build_application(config_path=args.config, load_weights=args.load_weights)

    if args.checkpoint:
        payload = app.repositories.checkpoints.load(args.checkpoint)
        state = payload.get("model_state", payload.get("model"))
        if state is None:
            raise KeyError(
                f"Checkpoint '{args.checkpoint}' missing model_state/model weights."
            )
        app.model.load_state_dict(state)

    app.model.eval()
    if not args.load_weights and args.checkpoint is None:
        print(
            "Warning: running inference without pretrained/finetuned weights. "
            "Pass --load-weights and/or --checkpoint for real outputs.",
            file=sys.stderr,
        )

    # Discrete generation only — does not use Gumbel-Softmax / soft embeddings.
    outputs = app.model.generate(args.text)
    print(outputs[0] if outputs else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
