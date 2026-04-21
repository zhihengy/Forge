from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from forge.adapters.qwen_image import QwenImageCheckpointAdapter
from forge.architectures.qwen_image import QwenImageArchitecture
from forge.batch import DenoiseBatch
from forge.parallel.config import ParallelConfig
from forge.parallel.plan import ParallelPlan, StrategySpec
from forge.runtimes.base import ModelRuntime


class QwenImageRuntime(ModelRuntime):
    def __init__(self, model_name_or_path: str) -> None:
        self.checkpoint_adapter = QwenImageCheckpointAdapter()
        transformer_config = self.checkpoint_adapter.load_config(model_name_or_path)
        model_config = self.checkpoint_adapter.build_model_config(transformer_config)
        self.model_name_or_path = model_name_or_path
        self.architecture = QwenImageArchitecture(model_config=model_config)
        self.noise_scheduler: FlowMatchEulerDiscreteScheduler | None = None

    def build_model(self) -> nn.Module:
        return self.architecture.build_model()

    def load_weights(self, model: nn.Module) -> None:
        state_dict = self.checkpoint_adapter.load_state_dict(self.model_name_or_path)
        model.load_state_dict(self.checkpoint_adapter.remap_state_dict(state_dict))

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_name_or_path,
            subfolder="scheduler",
        )
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)

    def canonicalize_batch(self, raw_batch: dict[str, Any]) -> DenoiseBatch:
        values = {
            "latents": raw_batch.get("latents", raw_batch.get("latent", raw_batch.get("hidden_states"))),
            "prompt_embeds": raw_batch.get(
                "prompt_embeds",
                raw_batch.get("prompt_embed", raw_batch.get("encoder_hidden_states")),
            ),
        }
        missing = [field for field in self.architecture.condition_schema.required_fields if values.get(field) is None]
        if missing:
            raise ValueError(f"Raw batch is missing required fields: {missing}")

        model_extras = {
            "encoder_hidden_states_mask": raw_batch.get(
                "encoder_hidden_states_mask",
                raw_batch.get("attention_mask"),
            ),
            "img_shapes": raw_batch.get("img_shapes"),
        }
        if raw_batch.get("guidance") is not None:
            model_extras["guidance"] = raw_batch["guidance"]
        if raw_batch.get("attention_kwargs") is not None:
            model_extras["attention_kwargs"] = raw_batch["attention_kwargs"]

        required_extras = ("encoder_hidden_states_mask", "img_shapes")
        missing_extras = [name for name in required_extras if model_extras.get(name) is None]
        if missing_extras:
            raise ValueError(f"Raw batch is missing required QwenImage extras: {missing_extras}")

        sample_ids = raw_batch.get("sample_ids", raw_batch.get("sample_id", []))
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]

        return DenoiseBatch(
            latents=values["latents"],
            prompt_embeds=values["prompt_embeds"],
            timesteps=raw_batch.get("timesteps"),
            noise=raw_batch.get("noise"),
            sample_ids=list(sample_ids),
            model_extras=model_extras,
        )

    def prepare_forward_inputs(self, batch: DenoiseBatch, objective_state: dict[str, Any]) -> dict[str, Any]:
        model_extras = batch.model_extras
        return {
            "hidden_states": objective_state["noisy_latents"],
            "encoder_hidden_states": batch.prompt_embeds,
            "encoder_hidden_states_mask": model_extras["encoder_hidden_states_mask"],
            "timestep": objective_state["timesteps"],
            "img_shapes": model_extras["img_shapes"],
            "guidance": model_extras.get("guidance"),
            "attention_kwargs": model_extras.get("attention_kwargs"),
            "return_dict": False,
        }

    def make_parallel_plan(self, parallel_config: ParallelConfig, batch: DenoiseBatch | None = None) -> ParallelPlan:
        del batch
        spec = self.architecture.parallel_spec
        strategies: list[StrategySpec] = []
        required_batch_extras: set[str] = set()

        parameter_parallel = parallel_config.parameter_parallel
        if parameter_parallel is not None and parameter_parallel.mode == "fsdp2" and parameter_parallel.degree > 1:
            strategies.append(
                StrategySpec(
                    kind="fsdp2",
                    config={"degree": parameter_parallel.degree},
                )
            )

        sequence_parallel = parallel_config.sequence_parallel
        if sequence_parallel.mode in {"auto", "native"}:
            native_spec = spec.native_sequence_parallel
            if native_spec is None:
                raise ValueError("QwenImage architecture does not declare native sequence parallel support.")
            if sequence_parallel.degree > 1:
                algorithm = (sequence_parallel.algorithm or native_spec.default_algorithm).lower()
                if algorithm not in native_spec.supported_algorithms:
                    raise ValueError(
                        f"Unsupported native sequence-parallel algorithm '{algorithm}'. "
                        f"Supported: {native_spec.supported_algorithms}"
                    )
                strategies.insert(
                    0,
                    StrategySpec(
                        kind="native_sequence_parallel",
                        config={
                            "degree": sequence_parallel.degree,
                            "algorithm": algorithm,
                            "attention_backend": sequence_parallel.attention_backend,
                        },
                    ),
                )
                required_batch_extras.update(native_spec.required_batch_extras)
        elif sequence_parallel.mode == "patched":
            raise NotImplementedError("QwenImage patched sequence parallel is not implemented.")

        return ParallelPlan(
            parameter_degree=parameter_parallel.degree if parameter_parallel is not None else 1,
            sequence_degree=sequence_parallel.degree,
            strategies=tuple(strategies),
            required_batch_extras=tuple(sorted(required_batch_extras)),
        )
