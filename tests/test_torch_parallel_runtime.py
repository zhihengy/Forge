from __future__ import annotations

import unittest
from unittest import mock

import torch

from forge.parallel.plan import ParallelPlan, StrategySpec
from forge.parallel.torch_runtime import TorchParallelRuntime


class DummySequenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
        self.backend = None
        self.ulysses_degree = None

    def set_attention_backend(self, backend: str) -> None:
        self.backend = backend

    def enable_ulysses_parallelism(self, *, degree: int) -> None:
        self.ulysses_degree = degree


class TorchParallelRuntimeGradientSyncTest(unittest.TestCase):
    def test_step_all_reduces_sequence_parallel_gradients(self) -> None:
        runtime = TorchParallelRuntime()
        runtime.sequence_parallel_group = object()
        runtime.use_fsdp = False

        param = torch.nn.Parameter(torch.tensor([1.0]))
        param.grad = torch.tensor([2.0])
        optimizer = torch.optim.SGD([param], lr=0.1)

        def fake_all_reduce(tensor, *, op, group):
            del op, group
            tensor.mul_(2)

        with (
            mock.patch.object(runtime, "is_distributed", return_value=True),
            mock.patch("forge.parallel.torch_runtime.dist.all_reduce", side_effect=fake_all_reduce) as all_reduce,
        ):
            runtime.step(optimizer)

        all_reduce.assert_called_once()
        grad_arg = all_reduce.call_args.args[0]
        self.assertTrue(torch.equal(grad_arg, torch.tensor([4.0])))
        self.assertEqual(all_reduce.call_args.kwargs["op"], torch.distributed.ReduceOp.SUM)
        self.assertIs(all_reduce.call_args.kwargs["group"], runtime.sequence_parallel_group)
        self.assertTrue(torch.equal(param, torch.tensor([0.6])))

    def test_step_skips_gradient_sync_without_sequence_parallel(self) -> None:
        runtime = TorchParallelRuntime()
        param = torch.nn.Parameter(torch.tensor([1.0]))
        param.grad = torch.tensor([2.0])
        optimizer = torch.optim.SGD([param], lr=0.1)

        with mock.patch.object(runtime, "is_distributed", return_value=True), mock.patch(
            "forge.parallel.torch_runtime.dist.all_reduce"
        ) as all_reduce:
            runtime.step(optimizer)

        all_reduce.assert_not_called()

    def test_backward_scales_loss_for_fsdp_ulysses_combo(self) -> None:
        runtime = TorchParallelRuntime()
        runtime.use_fsdp = True
        runtime.sequence_parallel_group = object()
        param = torch.tensor(1.0, requires_grad=True)
        loss = param * 3.0

        with (
            mock.patch.object(runtime, "is_distributed", return_value=True),
            mock.patch("forge.parallel.torch_runtime.dist.get_world_size", return_value=2),
        ):
            runtime.backward(loss)

        self.assertEqual(float(param.grad.item()), 6.0)

    def test_parallelize_model_applies_ulysses_and_fsdp2_together(self) -> None:
        runtime = TorchParallelRuntime()
        model = DummySequenceModel()
        plan = ParallelPlan(
            parameter_degree=2,
            sequence_degree=2,
            strategies=(
                StrategySpec(kind="native_sequence_parallel", config={"degree": 2, "algorithm": "ulysses", "attention_backend": "native"}),
                StrategySpec(kind="fsdp2", config={"degree": 2}),
            ),
        )

        with (
            mock.patch.object(runtime, "_get_device", return_value=torch.device("cpu")),
            mock.patch.object(runtime, "_apply_native_sequence_parallel") as apply_sp,
            mock.patch.object(runtime, "_apply_fsdp", return_value=model) as apply_fsdp,
        ):
            wrapped = runtime.parallelize_model(model, plan)

        apply_sp.assert_called_once_with(model, plan.strategies[0])
        apply_fsdp.assert_called_once_with(model, plan.strategies[1], torch.device("cpu"))
        self.assertIs(wrapped, model)

    def test_apply_fsdp2_shards_blocks_then_root(self) -> None:
        runtime = TorchParallelRuntime()
        model = DummySequenceModel()
        strategy = StrategySpec(kind="fsdp2", config={"degree": 2})
        mesh = object()

        with (
            mock.patch.object(runtime, "is_distributed", return_value=True),
            mock.patch("forge.parallel.torch_runtime.dist.get_world_size", return_value=2),
            mock.patch("forge.parallel.torch_runtime.init_device_mesh", return_value=mesh) as init_mesh,
            mock.patch("forge.parallel.torch_runtime.fully_shard", side_effect=lambda module, **kwargs: module) as fully_shard,
        ):
            wrapped = runtime._apply_fsdp(model, strategy, torch.device("cuda", 0))

        init_mesh.assert_called_once_with("cuda", (2,))
        self.assertEqual(fully_shard.call_count, 3)
        first_block_call = fully_shard.call_args_list[0]
        second_block_call = fully_shard.call_args_list[1]
        root_call = fully_shard.call_args_list[2]
        self.assertIs(first_block_call.args[0], model.transformer_blocks[0])
        self.assertIs(second_block_call.args[0], model.transformer_blocks[1])
        self.assertIs(root_call.args[0], model)
        self.assertEqual(first_block_call.kwargs, {"mesh": mesh})
        self.assertEqual(second_block_call.kwargs, {"mesh": mesh})
        self.assertEqual(root_call.kwargs, {"mesh": mesh, "reshard_after_forward": False})
        self.assertIs(wrapped, model)


if __name__ == "__main__":
    unittest.main()
