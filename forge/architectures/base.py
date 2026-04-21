from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass(frozen=True)
class ConditionSchema:
    required_fields: tuple[str, ...]


@dataclass(frozen=True)
class NativeSequenceParallelSpec:
    supported_algorithms: tuple[str, ...]
    default_algorithm: str
    required_batch_extras: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchitectureParallelSpec:
    native_sequence_parallel: NativeSequenceParallelSpec | None = None


class ModelArchitecture(ABC):
    """Static model-family definition."""

    condition_schema: ConditionSchema
    parallel_spec: ArchitectureParallelSpec

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build an uninitialized model instance."""

    def remap_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return state_dict
