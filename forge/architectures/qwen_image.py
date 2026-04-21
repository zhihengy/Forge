from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from forge.architectures.base import (
    ArchitectureParallelSpec,
    ConditionSchema,
    ModelArchitecture,
    NativeSequenceParallelSpec,
)
from forge.model_cores.qwen_image import QwenImageDiTConfig
from forge.model_cores.registry import get_model_core


@dataclass
class QwenImageArchitecture(ModelArchitecture):
    model_config: QwenImageDiTConfig

    def __post_init__(self) -> None:
        self.condition_schema = ConditionSchema(
            required_fields=("latents", "prompt_embeds"),
        )
        self.parallel_spec = ArchitectureParallelSpec(
            native_sequence_parallel=NativeSequenceParallelSpec(
                supported_algorithms=("ulysses",),
                default_algorithm="ulysses",
                required_batch_extras=("encoder_hidden_states_mask", "img_shapes"),
            ),
        )

    def build_model(self) -> nn.Module:
        model_cls = get_model_core("qwen_image")
        return model_cls(**self.model_config.to_init_kwargs())
