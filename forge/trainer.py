from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from forge.core import ForgeCore
from forge.training_args import TrainingEngineArgs


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0


class Trainer:
    def __init__(
        self,
        args: TrainingEngineArgs,
        core: ForgeCore,
        train_loader: Iterable[dict[str, Any]],
        optimizer_factory: Any,
        scheduler_factory: Any | None = None,
    ) -> None:
        self.args = args
        self.core = core
        self.model_runtime = core.model_runtime
        self.objective = core.objective
        self.parallel_runtime = core.parallel_runtime
        self.parallel_config = args.parallel_config()
        self.train_loader = train_loader
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.state = TrainerState()

        self.model = self.model_runtime.build_model()
        self.model_runtime.load_weights(self.model)
        self.parallel_plan = self.model_runtime.make_parallel_plan(self.parallel_config)
        self.parallel_runtime.setup(self.parallel_plan)
        self.model = self.parallel_runtime.parallelize_model(self.model, self.parallel_plan)

        self.optimizer = self.parallel_runtime.prepare_optimizer(self.optimizer_factory(self.model))
        self.scheduler = (
            self.scheduler_factory(self.optimizer) if self.scheduler_factory else None
        )

    def train(self) -> None:
        self.args.output_path().mkdir(parents=True, exist_ok=True)
        self.model.train()
        accumulation = max(1, self.args.gradient_accumulation_steps)
        micro_step = 0

        for epoch in range(self.state.epoch, self.args.num_train_epochs):
            self.state.epoch = epoch
            sampler = getattr(self.train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            for raw_batch in self.train_loader:
                batch = self.model_runtime.canonicalize_batch(raw_batch)
                batch = self.parallel_runtime.redistribute_batch(batch, self.parallel_plan)
                objective_state = self.objective.prepare(batch, self.model_runtime, self.model)
                model_inputs = self.model_runtime.prepare_forward_inputs(batch, objective_state)
                model_outputs = self.model(**model_inputs)
                loss, metrics, _artifacts = self.objective.compute_loss(
                    model_outputs,
                    batch,
                    objective_state,
                )
                loss = loss / accumulation
                self.parallel_runtime.backward(loss)
                micro_step += 1

                if micro_step % accumulation == 0:
                    self.parallel_runtime.step(self.optimizer, self.scheduler)
                    self.state.global_step += 1

                    loss_value = metrics.get("loss")
                    if loss_value is None:
                        loss_value = float(loss.detach().item() * accumulation)
                    if self.parallel_runtime.is_main_process():
                        print(
                            f"[train] step={self.state.global_step} loss={loss_value:.6f}",
                            flush=True,
                        )

                    if self.state.global_step >= self.args.max_train_steps:
                        return
