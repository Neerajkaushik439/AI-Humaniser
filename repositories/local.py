"""Local filesystem and in-memory repository implementations."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from repositories.base import (
    CheckpointRepository,
    DatasetRepository,
    ExperimentRepository,
    RepositoryBundle,
)


class LocalDatasetRepository(DatasetRepository):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def healthcheck(self) -> bool:
        return self.root.exists()

    def list_datasets(self) -> List[str]:
        names = {p.stem for p in self.processed_dir.glob("*") if p.is_file()}
        names |= {p.stem for p in self.raw_dir.glob("*") if p.is_file()}
        return sorted(names)

    def _resolve(self, name: str) -> Path:
        for directory in (self.processed_dir, self.raw_dir):
            for suffix in (".pt", ".json", ".jsonl", ".csv", ".txt"):
                candidate = directory / f"{name}{suffix}"
                if candidate.exists():
                    return candidate
        return self.processed_dir / f"{name}.pt"

    def exists(self, name: str) -> bool:
        path = self._resolve(name)
        return path.exists()

    def load(self, name: str) -> Any:
        path = self._resolve(name)
        if not path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found under {self.root}")
        if path.suffix == ".pt":
            return torch.load(path, map_location="cpu", weights_only=False)
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows
        if path.suffix == ".csv":
            import csv
            with path.open("r", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        if path.suffix == ".txt":
            return [
                {"source_text": line.strip()}
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    def save(self, name: str, dataset: Any) -> Path:
        """Save list/dict datasets as JSON by default (portable across repos)."""
        if isinstance(dataset, (list, dict)):
            path = self.processed_dir / f"{name}.json"
            path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
            return path
        path = self.processed_dir / f"{name}.pt"
        torch.save(dataset, path)
        return path


class LocalExperimentRepository(ExperimentRepository):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def healthcheck(self) -> bool:
        return self.root.exists()

    def _path(self, experiment_id: str) -> Path:
        return self.root / f"{experiment_id}.json"

    def create(self, experiment_id: str, metadata: Dict[str, Any]) -> None:
        payload = {"id": experiment_id, "metadata": metadata, "metrics": {}}
        self._path(experiment_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def update_metrics(self, experiment_id: str, metrics: Dict[str, Any]) -> None:
        path = self._path(experiment_id)
        if not path.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' does not exist")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("metrics", {}).update(metrics)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(experiment_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_experiments(self) -> List[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))


class LocalCheckpointRepository(CheckpointRepository):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def healthcheck(self) -> bool:
        return self.root.exists()

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.pt"

    def save(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self._path(name)
        torch.save(payload, path)
        return path

    def load(self, name: str) -> Dict[str, Any]:
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint '{name}' not found at {path}")
        return torch.load(path, map_location="cpu")

    def list_checkpoints(self) -> List[str]:
        return sorted(p.stem for p in self.root.glob("*.pt"))

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def delete(self, name: str) -> None:
        path = self._path(name)
        if path.exists():
            path.unlink()


class InMemoryDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def healthcheck(self) -> bool:
        return True

    def list_datasets(self) -> List[str]:
        return sorted(self._store)

    def load(self, name: str) -> Any:
        if name not in self._store:
            raise FileNotFoundError(name)
        return deepcopy(self._store[name])

    def save(self, name: str, dataset: Any) -> Path:
        self._store[name] = deepcopy(dataset)
        return Path(f"memory://datasets/{name}")

    def exists(self, name: str) -> bool:
        return name in self._store


class InMemoryExperimentRepository(ExperimentRepository):
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def healthcheck(self) -> bool:
        return True

    def create(self, experiment_id: str, metadata: Dict[str, Any]) -> None:
        self._store[experiment_id] = {"id": experiment_id, "metadata": deepcopy(metadata), "metrics": {}}

    def update_metrics(self, experiment_id: str, metrics: Dict[str, Any]) -> None:
        if experiment_id not in self._store:
            raise FileNotFoundError(experiment_id)
        self._store[experiment_id]["metrics"].update(deepcopy(metrics))

    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        payload = self._store.get(experiment_id)
        return deepcopy(payload) if payload else None

    def list_experiments(self) -> List[str]:
        return sorted(self._store)


class InMemoryCheckpointRepository(CheckpointRepository):
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def healthcheck(self) -> bool:
        return True

    def save(self, name: str, payload: Dict[str, Any]) -> Path:
        self._store[name] = deepcopy(payload)
        return Path(f"memory://checkpoints/{name}")

    def load(self, name: str) -> Dict[str, Any]:
        if name not in self._store:
            raise FileNotFoundError(name)
        return deepcopy(self._store[name])

    def list_checkpoints(self) -> List[str]:
        return sorted(self._store)

    def exists(self, name: str) -> bool:
        return name in self._store

    def delete(self, name: str) -> None:
        self._store.pop(name, None)


def build_repositories(kind: str, data_dir: str, checkpoint_dir: str, experiment_dir: str) -> RepositoryBundle:
    """Factory used by the application bootstrap."""
    kind = kind.lower()
    if kind == "memory":
        return RepositoryBundle(
            datasets=InMemoryDatasetRepository(),
            experiments=InMemoryExperimentRepository(),
            checkpoints=InMemoryCheckpointRepository(),
        )
    if kind == "local":
        return RepositoryBundle(
            datasets=LocalDatasetRepository(data_dir),
            experiments=LocalExperimentRepository(experiment_dir),
            checkpoints=LocalCheckpointRepository(checkpoint_dir),
        )
    raise ValueError(f"Unknown repository kind: {kind!r}. Use 'local' or 'memory'.")
