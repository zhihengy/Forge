from __future__ import annotations

from dataclasses import dataclass

from forge.objectives.base import TrainingObjective
from forge.objectives.flow_match import FlowMatchObjective
from forge.parallel.base import ParallelRuntime
from forge.parallel.config import ParallelConfig
from forge.parallel.torch_runtime import TorchParallelRuntime
from forge.runtimes.base import ModelRuntime
from forge.runtimes.qwen_image import QwenImageRuntime


@dataclass
class ForgeCore:
    model_runtime: ModelRuntime
    objective: TrainingObjective
    parallel_runtime: ParallelRuntime


def create_core(
    model_name_or_path: str,
    parallel_config: ParallelConfig,
) -> ForgeCore:
    model_runtime: ModelRuntime = QwenImageRuntime(model_name_or_path=model_name_or_path)
    objective: TrainingObjective = FlowMatchObjective()

    if parallel_config.backend != "torch":
        raise ValueError(f"Unsupported parallel backend: {parallel_config.backend}")
    parallel_runtime = TorchParallelRuntime()
    return ForgeCore(
        model_runtime=model_runtime,
        objective=objective,
        parallel_runtime=parallel_runtime,
    )
