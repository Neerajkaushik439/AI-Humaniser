"""Repository interfaces.

Training and experiment code depend on these abstractions only.
No MongoDB / PostgreSQL / MySQL implementations — swap later without
touching trainers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class BaseRepository(ABC):
    """Root repository interface."""

    @abstractmethod
    def healthcheck(self) -> bool:
        """Return True if the backing store is reachable."""


class DatasetRepository(BaseRepository):
    """Access training / evaluation text corpora."""

    @abstractmethod
    def list_datasets(self) -> List[str]:
        ...

    @abstractmethod
    def load(self, name: str) -> Any:
        ...

    @abstractmethod
    def save(self, name: str, dataset: Any) -> Path:
        ...

    @abstractmethod
    def exists(self, name: str) -> bool:
        ...


class ExperimentRepository(BaseRepository):
    """Persist experiment metadata and metrics."""

    @abstractmethod
    def create(self, experiment_id: str, metadata: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def update_metrics(self, experiment_id: str, metrics: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def list_experiments(self) -> List[str]:
        ...


class CheckpointRepository(BaseRepository):
    """Save / load model checkpoints."""

    @abstractmethod
    def save(self, name: str, payload: Dict[str, Any]) -> Path:
        ...

    @abstractmethod
    def load(self, name: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        ...

    @abstractmethod
    def exists(self, name: str) -> bool:
        ...

    @abstractmethod
    def delete(self, name: str) -> None:
        ...


class RepositoryBundle:
    """Dependency-injection container for all repositories."""

    def __init__(
        self,
        datasets: DatasetRepository,
        experiments: ExperimentRepository,
        checkpoints: CheckpointRepository,
    ) -> None:
        self.datasets = datasets
        self.experiments = experiments
        self.checkpoints = checkpoints

    def healthcheck(self) -> Dict[str, bool]:
        return {
            "datasets": self.datasets.healthcheck(),
            "experiments": self.experiments.healthcheck(),
            "checkpoints": self.checkpoints.healthcheck(),
        }
