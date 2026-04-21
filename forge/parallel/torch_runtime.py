from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from forge.parallel.base import ParallelRuntime
from forge.parallel.plan import ParallelPlan, StrategySpec


class TorchParallelRuntime(ParallelRuntime):
    def __init__(self) -> None:
        self.use_fsdp = False
        self.sequence_parallel_group: dist.ProcessGroup | None = None

    def setup(self, plan: ParallelPlan | None = None) -> None:
        del plan
        return None

    def parallelize_model(self, model: Any, plan: ParallelPlan) -> Any:
        self.use_fsdp = False
        self.sequence_parallel_group = None

        device = self._get_device()
        model.to(device)

        sequence_strategy = plan.get_strategy("native_sequence_parallel")
        if sequence_strategy is not None:
            self._apply_native_sequence_parallel(model, sequence_strategy)

        fsdp_strategy = plan.get_strategy("fsdp2")
        if fsdp_strategy is not None:
            model = self._apply_fsdp(model, fsdp_strategy, device)

        return model

    def prepare_optimizer(self, optimizer: Any) -> Any:
        return optimizer

    def redistribute_batch(self, batch: Any, plan: ParallelPlan) -> Any:
        for extra in plan.required_batch_extras:
            if extra not in getattr(batch, "model_extras", {}):
                raise ValueError(f"Canonical batch is missing required model extra: {extra}")
        return batch

    def backward(self, loss: Any) -> None:
        if self.use_fsdp and self.sequence_parallel_group is not None and self.is_distributed():
            loss = loss * dist.get_world_size(self.sequence_parallel_group)
        loss.backward()

    def step(self, optimizer: Any, scheduler: Any | None = None) -> None:
        self._sync_sequence_parallel_gradients(optimizer)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    def is_main_process(self) -> bool:
        if not self.is_distributed():
            return True
        return dist.get_rank() == 0

    def is_distributed(self) -> bool:
        return bool(dist.is_available() and dist.is_initialized())

    def barrier(self) -> None:
        if self.is_distributed():
            dist.barrier()

    def _apply_native_sequence_parallel(self, model: Any, strategy: StrategySpec) -> None:
        if not self.is_distributed():
            raise RuntimeError("torch.distributed must be initialized before enabling Ulysses sequence parallelism.")

        config = strategy.config
        degree = int(config["degree"])
        if degree != dist.get_world_size():
            raise NotImplementedError(
                "Native sequence parallel currently expects the sequence-parallel degree to match world_size."
            )

        algorithm = str(config["algorithm"])
        if algorithm != "ulysses":
            raise NotImplementedError(
                f"Unsupported native sequence-parallel algorithm: {algorithm}. Forge currently only supports ulysses."
            )

        model.set_attention_backend(str(config["attention_backend"]))
        if not hasattr(model, "enable_ulysses_parallelism"):
            raise TypeError(f"Model {type(model).__name__} does not implement enable_ulysses_parallelism().")
        model.enable_ulysses_parallelism(degree=degree)
        self.sequence_parallel_group = dist.group.WORLD

    def _apply_fsdp(self, model: Any, strategy: StrategySpec, device: torch.device) -> Any:
        world_size = dist.get_world_size() if self.is_distributed() else 1
        degree = int(strategy.config.get("degree", 1))
        if world_size == 1 or degree == 1:
            self.use_fsdp = False
            return model

        if degree != world_size:
            raise NotImplementedError("FSDP currently expects the parameter-parallel degree to match world_size.")

        self.use_fsdp = True
        mesh = init_device_mesh(device.type, (world_size,))
        for block in getattr(model, "transformer_blocks", ()):
            fully_shard(block, mesh=mesh)
        return fully_shard(model, mesh=mesh, reshard_after_forward=False)

    def _sync_sequence_parallel_gradients(self, optimizer: Any) -> None:
        if self.sequence_parallel_group is None or self.use_fsdp or not self.is_distributed():
            return

        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                grad = getattr(param, "grad", None)
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise NotImplementedError("Sparse gradients are not supported with self-owned Ulysses yet.")
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.sequence_parallel_group)

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        return torch.device("cpu")
