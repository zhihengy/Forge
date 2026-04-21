from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParameterParallelConfig:
    mode: str = "fsdp2"
    degree: int = 1


@dataclass(frozen=True)
class SequenceParallelConfig:
    mode: str = "none"
    algorithm: str = "ulysses"
    degree: int = 1
    attention_backend: str = "native"


@dataclass(frozen=True)
class ParallelConfig:
    backend: str = "torch"
    dp_mode: str = "fsdp2"
    parameter_parallel: ParameterParallelConfig | None = None
    sequence_parallel: SequenceParallelConfig = field(default_factory=SequenceParallelConfig)

    def __post_init__(self) -> None:
        if self.parameter_parallel is None:
            object.__setattr__(self, "parameter_parallel", ParameterParallelConfig(mode=self.dp_mode, degree=1))
