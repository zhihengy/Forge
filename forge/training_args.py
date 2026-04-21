from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from forge.parallel.config import ParallelConfig, SequenceParallelConfig


@dataclass
class TrainingEngineArgs:
    model_name_or_path: str
    output_dir: str = "outputs/qwen_image_fsdp"
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    max_train_steps: int = 1000
    num_train_epochs: int = 1
    seed: int = 42
    dp_mode: str = "fsdp1"
    sequence_parallel_mode: str = "none"
    sequence_parallel_degree: int = 1
    attention_backend: str = "native"

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def parallel_config(self) -> ParallelConfig:
        return ParallelConfig(
            backend="torch",
            dp_mode=self.dp_mode,
            sequence_parallel=SequenceParallelConfig(
                mode=self.sequence_parallel_mode,
                algorithm="ulysses",
                degree=self.sequence_parallel_degree,
                attention_backend=self.attention_backend,
            ),
        )
