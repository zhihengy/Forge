from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from forge.parallel.plan import ParallelPlan


class ParallelRuntime(ABC):
    @abstractmethod
    def setup(self, plan: ParallelPlan | None = None) -> None:
        """Initialize runtime-local distributed state if needed."""

    @abstractmethod
    def parallelize_model(self, model: Any, plan: ParallelPlan) -> Any:
        """Wrap and return a trainable model."""

    @abstractmethod
    def prepare_optimizer(self, optimizer: Any) -> Any:
        """Return an optimizer compatible with the wrapped model."""

    @abstractmethod
    def redistribute_batch(self, batch: Any, plan: ParallelPlan) -> Any:
        """Redistribute a canonical batch according to the internal plan. For SP use"""

    @abstractmethod
    def backward(self, loss: Any) -> None:
        """Run backward for one step."""

    @abstractmethod
    def step(self, optimizer: Any, scheduler: Any | None = None) -> None:
        """Run the optimizer update and gradient clear."""

    def is_main_process(self) -> bool:
        return True

    def is_distributed(self) -> bool:
        return False

    def barrier(self) -> None:
        return None
