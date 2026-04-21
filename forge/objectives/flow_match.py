from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from forge.batch import DenoiseBatch
from forge.objectives.base import TrainingObjective
from forge.runtimes.base import ModelRuntime


class FlowMatchObjective(TrainingObjective):
    def prepare(self, batch: DenoiseBatch, runtime: ModelRuntime, model: Any) -> dict[str, Any]:
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype

        scheduler = runtime.noise_scheduler
        latents = batch.latents.to(device=device, dtype=dtype)
        batch.latents = latents
        if batch.prompt_embeds is not None:
            batch.prompt_embeds = batch.prompt_embeds.to(device=device, dtype=dtype)
        if batch.model_extras:
            batch.model_extras = {
                key: self._move_value(value, device=device, dtype=dtype) for key, value in batch.model_extras.items()
            }

        noise = batch.noise
        if noise is None:
            noise = torch.randn_like(latents)
        else:
            noise = noise.to(device=device, dtype=dtype)

        batch_size = latents.shape[0]
        raw_timesteps = batch.timesteps
        if raw_timesteps is None:
            step_indices = torch.randint(
                0,
                scheduler.timesteps.shape[0],
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            timesteps = scheduler.timesteps.to(device=device)[step_indices]
        else:
            timesteps, step_indices = self._resolve_timesteps(raw_timesteps.view(batch_size), scheduler, device)

        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)[step_indices]
        sigmas = sigmas.view(batch_size, *([1] * (latents.ndim - 1)))

        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        target = noise - latents
        return {
            "timesteps": timesteps,
            "step_indices": step_indices,
            "noise": noise,
            "sigmas": sigmas,
            "noisy_latents": noisy_latents,
            "target": target,
        }

    def compute_loss(
        self,
        model_outputs: Any,
        batch: DenoiseBatch,
        objective_state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        model_pred = model_outputs[0]
        target = objective_state["target"]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        metrics = {"loss": float(loss.detach().item())}
        artifacts = {"sample_ids": batch.sample_ids}
        return loss, metrics, artifacts

    # might need to refactor here
    @staticmethod
    def _resolve_timesteps(
        timesteps: torch.Tensor,
        scheduler: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scheduler_timesteps = scheduler.timesteps.to(device=device)
        raw_timesteps = timesteps.to(device=device)
        indices: list[int] = []
        resolved_timesteps: list[torch.Tensor] = []
        max_index = scheduler_timesteps.shape[0] - 1
        for timestep in raw_timesteps:
            matches = (scheduler_timesteps == timestep).nonzero(as_tuple=False)
            if matches.numel() > 0:
                index = int(matches[0].item())
                indices.append(index)
                resolved_timesteps.append(scheduler_timesteps[index])
                continue

            maybe_index = int(timestep.item())
            if 0 <= maybe_index <= max_index:
                indices.append(maybe_index)
                resolved_timesteps.append(scheduler_timesteps[maybe_index])
                continue

            raise ValueError(f"Provided timestep {float(timestep.item())} is not present in scheduler.timesteps")

        return (
            torch.stack(resolved_timesteps).to(device=device, dtype=scheduler_timesteps.dtype),
            torch.tensor(indices, device=device, dtype=torch.long),
        )

    @staticmethod
    def _move_value(value: Any, *, device: torch.device, dtype: torch.dtype) -> Any:
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() or value.is_complex():
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: FlowMatchObjective._move_value(item, device=device, dtype=dtype) for key, item in value.items()}
        if isinstance(value, list):
            return [FlowMatchObjective._move_value(item, device=device, dtype=dtype) for item in value]
        if isinstance(value, tuple):
            return tuple(FlowMatchObjective._move_value(item, device=device, dtype=dtype) for item in value)
        return value
