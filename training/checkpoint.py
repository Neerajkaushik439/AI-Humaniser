"""Checkpoint manager — filesystem persistence of full training state."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from repositories.base import CheckpointRepository

logger = logging.getLogger("ai_humanizer.training.checkpoint")


class CheckpointManager:
    """Save / load model, optimizer, scheduler, step, epoch, and configs."""

    def __init__(
        self,
        repository: CheckpointRepository,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.repository = repository
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def build_payload(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        step: int = 0,
        epoch: int = 0,
        config: Optional[Any] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg_dict: Dict[str, Any]
        if config is None:
            cfg_dict = {}
        elif hasattr(config, "to_dict"):
            cfg_dict = config.to_dict()
        elif is_dataclass(config):
            cfg_dict = asdict(config)
        elif isinstance(config, dict):
            cfg_dict = dict(config)
        else:
            cfg_dict = {"repr": repr(config)}

        payload: Dict[str, Any] = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
            "step": int(step),
            "epoch": int(epoch),
            "configuration": cfg_dict,
            "loss_weights": dict(loss_weights or cfg_dict.get("losses") or {}),
            "reward_weights": dict(reward_weights or cfg_dict.get("rewards") or {}),
            "rl_configuration": dict(
                rl_config
                or (cfg_dict.get("rl") if isinstance(cfg_dict.get("rl"), dict) else {})
                or {}
            ),
        }
        if extra:
            payload["extra"] = extra
        return payload

    def save(
        self,
        name: str,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        step: int = 0,
        epoch: int = 0,
        config: Optional[Any] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        payload = self.build_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            config=config,
            loss_weights=loss_weights,
            reward_weights=reward_weights,
            rl_config=rl_config,
            extra=extra,
        )
        path = self.repository.save(name, payload)
        # Also write a small JSON sidecar for human inspection when using filesystem dir
        if self.checkpoint_dir is not None:
            meta = {
                "name": name,
                "step": payload["step"],
                "epoch": payload["epoch"],
                "loss_weights": payload["loss_weights"],
                "reward_weights": payload["reward_weights"],
                "rl_configuration": payload["rl_configuration"],
            }
            (self.checkpoint_dir / f"{name}.meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        logger.info("Saved checkpoint '%s' → %s", name, path)
        return Path(path)

    def load(self, name: str) -> Dict[str, Any]:
        payload = self.repository.load(name)
        logger.info("Loaded checkpoint '%s' (step=%s epoch=%s)", name, payload.get("step"), payload.get("epoch"))
        return payload

    def restore(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        map_location: Optional[str] = "cpu",
    ) -> Dict[str, Any]:
        payload = self.load(name)
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state") and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state"])
        return payload

    def exists(self, name: str) -> bool:
        return self.repository.exists(name)

    def list_checkpoints(self):
        return self.repository.list_checkpoints()
