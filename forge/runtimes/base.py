from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

from forge.architectures.base import ModelArchitecture
from forge.batch import DenoiseBatch
from forge.parallel.config import ParallelConfig
from forge.parallel.plan import ParallelPlan


class ModelRuntime(ABC):
    """Dynamic model-family runtime."""

    architecture: ModelArchitecture

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return a trainable model."""

    @abstractmethod
    def load_weights(self, model: nn.Module) -> None:
        """Load checkpoint weights and attach runtime state."""

    @abstractmethod
    def canonicalize_batch(self, raw_batch: dict[str, Any]) -> DenoiseBatch:
        """Convert upstream inputs into a canonical DenoiseBatch."""

    @abstractmethod
    def prepare_forward_inputs(self, batch: DenoiseBatch, objective_state: dict[str, Any]) -> dict[str, Any]:
        """Prepare model.forward kwargs."""

    @abstractmethod
    def make_parallel_plan(self, parallel_config: ParallelConfig, batch: DenoiseBatch | None = None) -> ParallelPlan:
        """Build an internal parallel plan for the current model family."""
