from __future__ import annotations

from dataclasses import dataclass, field


def resolve_context_parallel_degrees(*, algorithm: str, degree: int) -> tuple[int, int]:
    normalized_algorithm = algorithm.lower()
    if degree < 1:
        raise ValueError(f"sequence_parallel.degree must be >= 1, got {degree}")
    if normalized_algorithm != "ulysses":
        raise ValueError(
            f"Unsupported sequence-parallel algorithm: {algorithm}. Forge currently only supports self-owned ulysses."
        )
    return 1, degree


@dataclass(frozen=True)
class ParameterParallelConfig:
    mode: str = "fsdp1"
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
    dp_mode: str = "fsdp1"
    parameter_parallel: ParameterParallelConfig | None = None
    sequence_parallel: SequenceParallelConfig = field(default_factory=SequenceParallelConfig)

    def __post_init__(self) -> None:
        if self.parameter_parallel is None:
            object.__setattr__(self, "parameter_parallel", ParameterParallelConfig(mode=self.dp_mode, degree=1))
