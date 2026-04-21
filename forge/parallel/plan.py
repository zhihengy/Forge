from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StrategySpec:
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallelPlan:
    parameter_degree: int = 1
    sequence_degree: int = 1
    strategies: tuple[StrategySpec, ...] = ()
    required_batch_extras: tuple[str, ...] = ()

    def get_strategy(self, kind: str) -> StrategySpec | None:
        for strategy in self.strategies:
            if strategy.kind == kind:
                return strategy
        return None
