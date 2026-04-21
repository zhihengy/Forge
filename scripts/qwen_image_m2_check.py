from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "m2_qwen_image_check"
VENDORED_DIFFUSERS_SRC = ROOT / "diffusers" / "src"
TRAIN_LR = 1e-4
TRAIN_DTYPE = torch.float32
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if VENDORED_DIFFUSERS_SRC.exists():
    sys.path.insert(0, str(VENDORED_DIFFUSERS_SRC))

import transformers

if not hasattr(transformers, "AutoImageProcessor"):
    class _AutoImageProcessorStub:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise NotImplementedError("AutoImageProcessor is unavailable in this transformers build.")

    transformers.AutoImageProcessor = _AutoImageProcessorStub

from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from forge.core import create_core
from forge.objectives.flow_match import FlowMatchObjective
from forge.parallel.config import ParallelConfig, ParameterParallelConfig, SequenceParallelConfig


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def derive_axes_dims_rope(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 8 != 0:
        raise ValueError(f"attention_head_dim must be divisible by 8, got {head_dim}")
    frame_dim = max(2, head_dim // 8)
    if frame_dim % 2 != 0:
        frame_dim += 1
    spatial_total = head_dim - frame_dim
    height_dim = spatial_total // 2
    if height_dim % 2 != 0:
        height_dim -= 1
    width_dim = head_dim - frame_dim - height_dim
    if min(frame_dim, height_dim, width_dim) <= 0 or width_dim % 2 != 0:
        raise ValueError(
            f"Unable to derive valid rope axes for attention_head_dim={head_dim}: "
            f"{(frame_dim, height_dim, width_dim)}"
        )
    return frame_dim, height_dim, width_dim


def ensure_qwen_image_checkpoint(
    root: Path,
    *,
    patch_size: int,
    latent_channels: int,
    out_channels: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    joint_dim: int,
) -> Path:
    transformer_config = root / "transformer" / "config.json"
    scheduler_config = root / "scheduler" / "scheduler_config.json"
    if transformer_config.exists() and scheduler_config.exists():
        return root

    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = QwenImageTransformer2DModel(
        patch_size=patch_size,
        in_channels=latent_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=joint_dim,
        guidance_embeds=False,
        axes_dims_rope=derive_axes_dims_rope(head_dim),
    )
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    model.save_pretrained(root / "transformer", safe_serialization=False)
    scheduler.save_pretrained(root / "scheduler")
    return root


def inspect_visible_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        used_bytes = total_bytes - free_bytes
        gpus.append(
            {
                "local_rank": index,
                "name": torch.cuda.get_device_name(index),
                "free_gb": round(free_bytes / (1024**3), 2),
                "used_gb": round(used_bytes / (1024**3), 2),
                "total_gb": round(total_bytes / (1024**3), 2),
            }
        )
    return gpus


def run_gpu_preflight(world_size: int, min_free_gb_per_gpu: float) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "status": "error",
            "error": "CUDA is unavailable.",
            "visible_gpus": [],
        }

    visible_gpus = inspect_visible_gpus()
    if len(visible_gpus) < world_size:
        return {
            "status": "error",
            "error": f"Visible CUDA devices ({len(visible_gpus)}) < world_size ({world_size})",
            "visible_gpus": visible_gpus,
        }

    selected = visible_gpus[:world_size]
    too_busy = [gpu for gpu in selected if gpu["free_gb"] < min_free_gb_per_gpu]
    if too_busy:
        ranks = [gpu["local_rank"] for gpu in too_busy]
        return {
            "status": "error",
            "error": (
                f"Visible local ranks {ranks} do not meet min_free_gb_per_gpu={min_free_gb_per_gpu}. "
                f"Use CUDA_VISIBLE_DEVICES to target freer GPUs."
            ),
            "visible_gpus": visible_gpus,
            "selected_gpus": selected,
        }

    return {
        "status": "success",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "min_free_gb_per_gpu": min_free_gb_per_gpu,
        "visible_gpus": visible_gpus,
        "selected_gpus": selected,
    }


def build_sequence_parallel_config(*, world_size: int, attention_backend: str) -> SequenceParallelConfig:
    return SequenceParallelConfig(
        mode="native",
        algorithm="ulysses",
        degree=world_size,
        attention_backend=attention_backend,
    )


def build_parallel_config(
    *,
    world_size: int,
    attention_backend: str,
    use_fsdp: bool,
    use_ulysses: bool,
) -> ParallelConfig:
    return ParallelConfig(
        backend="torch",
        dp_mode="fsdp2" if use_fsdp else "none",
        parameter_parallel=ParameterParallelConfig(mode="fsdp2" if use_fsdp else "none", degree=world_size if use_fsdp else 1),
        sequence_parallel=(
            build_sequence_parallel_config(world_size=world_size, attention_backend=attention_backend)
            if use_ulysses
            else SequenceParallelConfig()
        ),
    )


def summarize_loss_diffs(
    reference_losses: list[float],
    candidate_losses: list[float],
) -> dict[str, Any]:
    per_step_abs_diff = [
        abs(reference - candidate) for reference, candidate in zip(reference_losses, candidate_losses, strict=True)
    ]
    return {
        "reference_losses": reference_losses,
        "candidate_losses": candidate_losses,
        "per_step_abs_diff": per_step_abs_diff,
        "max_abs_diff": max(per_step_abs_diff) if per_step_abs_diff else 0.0,
        "mean_abs_diff": (sum(per_step_abs_diff) / len(per_step_abs_diff)) if per_step_abs_diff else 0.0,
    }


def dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float32:
        return "fp32"
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    return str(dtype).replace("torch.", "")


def build_benchmark_summary(
    *,
    duration_s: float,
    steps: int,
    batch_size: int,
    diffusers_duration_s: float | None = None,
    forge_single_duration_s: float | None = None,
) -> dict[str, float]:
    benchmark = {
        "duration_s": duration_s,
        "steps_per_s": steps / duration_s,
        "samples_per_s": (steps * batch_size) / duration_s,
    }
    if diffusers_duration_s is not None:
        benchmark["speedup_vs_diffusers_single"] = diffusers_duration_s / duration_s
    if forge_single_duration_s is not None:
        benchmark["speedup_vs_forge_single"] = forge_single_duration_s / duration_s
    return benchmark


def extract_rank_errors(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key.startswith("rank_") and key.endswith("_error")}


def make_raw_batches(
    *,
    steps: int,
    batch_size: int,
    height: int,
    width: int,
    prompt_len: int,
    latent_channels: int,
    prompt_dim: int,
    seed: int,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches: list[dict[str, Any]] = []
    seq_len = height * width
    img_shapes = [(1, height, width)] * batch_size

    for step in range(steps):
        latents = torch.randn((batch_size, seq_len, latent_channels), generator=generator, dtype=torch.float32)
        prompt_embeds = torch.randn((batch_size, prompt_len, prompt_dim), generator=generator, dtype=torch.float32)
        noise = torch.randn((batch_size, seq_len, latent_channels), generator=generator, dtype=torch.float32)
        timesteps = torch.randint(0, 1000, (batch_size,), generator=generator, dtype=torch.long)

        mask = torch.ones((batch_size, prompt_len), dtype=torch.bool)
        cutoff = prompt_len - 8 - (step % 4)
        if cutoff > 0:
            mask[0, cutoff:] = False
        if batch_size > 1:
            mask[1, prompt_len - 12 :] = False

        batches.append(
            {
                "latents": latents,
                "prompt_embeds": prompt_embeds,
                "encoder_hidden_states_mask": mask,
                "noise": noise,
                "timesteps": timesteps,
                "img_shapes": list(img_shapes),
                "sample_ids": [f"sample-{step}-{index}" for index in range(batch_size)],
            }
        )
    return batches


def clone_raw_batches(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(batches)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def max_state_dict_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    diff = 0.0
    for key in left:
        if key not in right:
            raise KeyError(f"Missing parameter in reference state dict: {key}")
        diff = max(diff, float((left[key] - right[key]).abs().max().item()))
    return diff


def direct_diffusers_train(
    *,
    model_dir: Path,
    batches: list[dict[str, Any]],
    device: torch.device,
    learning_rate: float,
    attention_backend: str,
) -> dict[str, Any]:
    model = QwenImageTransformer2DModel.from_pretrained(model_dir, subfolder="transformer")
    model.to(device=device)
    model.set_attention_backend(attention_backend)
    model.train()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses: list[float] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for raw_batch in clone_raw_batches(batches):
        latents = raw_batch["latents"].to(device=device)
        prompt_embeds = raw_batch["prompt_embeds"].to(device=device)
        attention_mask = raw_batch["encoder_hidden_states_mask"].to(device=device)
        noise = raw_batch["noise"].to(device=device)
        timesteps, step_indices = FlowMatchObjective._resolve_timesteps(
            raw_batch["timesteps"].to(device=device),
            scheduler,
            device,
        )
        sigmas = scheduler.sigmas.to(device=device, dtype=latents.dtype)[step_indices]
        sigmas = sigmas.view(latents.shape[0], *([1] * (latents.ndim - 1)))

        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        target = noise - latents

        optimizer.zero_grad(set_to_none=True)
        model_pred = model(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=attention_mask,
            timestep=timesteps,
            img_shapes=raw_batch["img_shapes"],
            return_dict=False,
        )[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration_s = time.perf_counter() - started

    return {
        "losses": losses,
        "duration_s": duration_s,
        "state_dict": clone_state_dict(model),
        "precision": dtype_name(TRAIN_DTYPE),
    }


def forge_train(
    *,
    model_dir: Path,
    batches: list[dict[str, Any]],
    device: torch.device,
    learning_rate: float,
    parallel_config: ParallelConfig,
    attention_backend: str,
) -> dict[str, Any]:
    core = create_core(
        model_name_or_path=str(model_dir),
        parallel_config=parallel_config,
    )
    model = core.model_runtime.build_model()
    core.model_runtime.load_weights(model)
    model.to(device=device)
    model.set_attention_backend(attention_backend)
    plan = core.model_runtime.make_parallel_plan(parallel_config)
    core.parallel_runtime.setup(plan)
    model = core.parallel_runtime.parallelize_model(model, plan)
    model.train()

    optimizer = core.parallel_runtime.prepare_optimizer(torch.optim.AdamW(model.parameters(), lr=learning_rate))

    losses: list[float] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if core.parallel_runtime.is_distributed():
        core.parallel_runtime.barrier()
    started = time.perf_counter()

    for raw_batch in clone_raw_batches(batches):
        batch = core.model_runtime.canonicalize_batch(raw_batch)
        batch = core.parallel_runtime.redistribute_batch(batch, plan)
        objective_state = core.objective.prepare(batch, core.model_runtime, model)
        model_inputs = core.model_runtime.prepare_forward_inputs(batch, objective_state)
        outputs = model(**model_inputs)
        loss, metrics, _artifacts = core.objective.compute_loss(outputs, batch, objective_state)
        core.parallel_runtime.backward(loss)
        core.parallel_runtime.step(optimizer)
        losses.append(float(metrics["loss"]))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if core.parallel_runtime.is_distributed():
        core.parallel_runtime.barrier()
    duration_s = time.perf_counter() - started

    result = {
        "losses": losses,
        "duration_s": duration_s,
        "precision": dtype_name(TRAIN_DTYPE),
    }
    if not core.parallel_runtime.is_distributed():
        result["state_dict"] = clone_state_dict(model)
    return result


def run_single_gpu_precision_check(
    *,
    model_dir: Path,
    batches: list[dict[str, Any]],
    device: torch.device,
    attention_backend: str,
) -> dict[str, Any]:
    direct = direct_diffusers_train(
        model_dir=model_dir,
        batches=batches,
        device=device,
        learning_rate=TRAIN_LR,
        attention_backend=attention_backend,
    )
    forge = forge_train(
        model_dir=model_dir,
        batches=batches,
        device=device,
        learning_rate=TRAIN_LR,
        parallel_config=ParallelConfig(
            backend="torch",
            dp_mode="none",
        ),
        attention_backend=attention_backend,
    )

    summary = {
        "status": "success",
        "reference_mode": "diffusers_single_gpu",
        "candidate_mode": "forge_single_gpu",
        "reference_precision": direct["precision"],
        "candidate_precision": forge["precision"],
        "reference_duration_s": direct["duration_s"],
        "candidate_duration_s": forge["duration_s"],
        "max_param_diff": max_state_dict_diff(direct["state_dict"], forge["state_dict"]),
        "benchmark": build_benchmark_summary(
            duration_s=forge["duration_s"],
            steps=len(forge["losses"]),
            batch_size=batches[0]["latents"].shape[0],
            diffusers_duration_s=direct["duration_s"],
        ),
    }
    summary.update(summarize_loss_diffs(direct["losses"], forge["losses"]))
    return summary


def distributed_mode_worker(
    rank: int,
    world_size: int,
    master_port: int,
    model_dir: str,
    steps: int,
    batch_size: int,
    height: int,
    width: int,
    prompt_len: int,
    latent_channels: int,
    prompt_dim: int,
    seed: int,
    attention_backend: str,
    use_fsdp: bool,
    use_ulysses: bool,
    return_dict,
) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

        batches = make_raw_batches(
            steps=steps,
            batch_size=batch_size,
            height=height,
            width=width,
            prompt_len=prompt_len,
            latent_channels=latent_channels,
            prompt_dim=prompt_dim,
            seed=seed,
        )
        result = forge_train(
            model_dir=Path(model_dir),
            batches=batches,
            device=device,
            learning_rate=TRAIN_LR,
            parallel_config=build_parallel_config(
                world_size=world_size,
                attention_backend=attention_backend,
                use_fsdp=use_fsdp,
                use_ulysses=use_ulysses,
            ),
            attention_backend=attention_backend,
        )
        if rank == 0:
            return_dict["status"] = "success"
            return_dict["duration_s"] = result["duration_s"]
            return_dict["losses"] = result["losses"]
            return_dict["precision"] = result["precision"]
    except Exception as error:  # noqa: BLE001
        return_dict[f"rank_{rank}_error"] = repr(error)
        if "status" not in return_dict:
            return_dict["status"] = "error"
            return_dict["error"] = f"rank_{rank}: {repr(error)}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_parallel_precision_check(
    *,
    mode_name: str,
    candidate_mode: str,
    model_dir: Path,
    steps: int,
    batch_size: int,
    height: int,
    width: int,
    prompt_len: int,
    latent_channels: int,
    prompt_dim: int,
    seed: int,
    attention_backend: str,
    world_size: int,
    use_fsdp: bool,
    use_ulysses: bool,
    reference_losses: list[float],
    reference_duration_s: float,
    single_gpu_losses: list[float],
    single_gpu_duration_s: float,
) -> dict[str, Any]:
    manager = mp.Manager()
    return_dict = manager.dict()
    master_port = find_free_port()
    mp.spawn(
        distributed_mode_worker,
        args=(
            world_size,
            master_port,
            str(model_dir),
            steps,
            batch_size,
            height,
            width,
            prompt_len,
            latent_channels,
            prompt_dim,
            seed,
            attention_backend,
            use_fsdp,
            use_ulysses,
            return_dict,
        ),
        nprocs=world_size,
        join=True,
    )
    result = dict(return_dict)
    if result.get("status") != "success":
        summary = {
            "status": result.get("status", "error"),
            "candidate_mode": candidate_mode,
            "world_size": world_size,
            "error": result.get("error"),
        }
        summary.update(extract_rank_errors(result))
        return summary

    candidate_losses = list(result["losses"])
    summary = {
        "status": "success",
        "reference_mode": "diffusers_single_gpu",
        "candidate_mode": candidate_mode,
        "reference_precision": dtype_name(TRAIN_DTYPE),
        "candidate_precision": result["precision"],
        "world_size": world_size,
        "candidate_duration_s": result["duration_s"],
        "benchmark": build_benchmark_summary(
            duration_s=result["duration_s"],
            steps=steps,
            batch_size=batch_size,
            diffusers_duration_s=reference_duration_s,
            forge_single_duration_s=single_gpu_duration_s,
        ),
    }
    summary.update(summarize_loss_diffs(reference_losses, candidate_losses))
    summary["vs_forge_single"] = summarize_loss_diffs(single_gpu_losses, candidate_losses)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="QwenImage Forge precision comparison against diffusers.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--joint-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attention-backend", type=str, default="native")
    parser.add_argument("--world-size", "--ulysses-world-size", dest="world_size", type=int, default=2)
    parser.add_argument("--min-free-gb-per-gpu", type=float, default=60.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_area = args.patch_size * args.patch_size
    if args.latent_channels % patch_area != 0:
        raise ValueError(
            f"latent_channels ({args.latent_channels}) must be divisible by patch_size^2 ({patch_area})"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("This precision check expects at least one visible CUDA device.")

    out_channels = args.latent_channels // patch_area
    model_slug = (
        f"synthetic_qwen_image_p{args.patch_size}_c{args.latent_channels}"
        f"_l{args.num_layers}_h{args.num_heads}x{args.head_dim}_j{args.joint_dim}"
    )
    model_dir = ensure_qwen_image_checkpoint(
        args.output_dir / model_slug,
        patch_size=args.patch_size,
        latent_channels=args.latent_channels,
        out_channels=out_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        joint_dim=args.joint_dim,
    )

    preflight = run_gpu_preflight(args.world_size, args.min_free_gb_per_gpu)
    if preflight["status"] != "success":
        raise RuntimeError(preflight["error"])

    batches = make_raw_batches(
        steps=args.steps,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        prompt_len=args.prompt_len,
        latent_channels=args.latent_channels,
        prompt_dim=args.joint_dim,
        seed=args.seed,
    )
    device = torch.device("cuda", 0)

    single_gpu_summary = run_single_gpu_precision_check(
        model_dir=model_dir,
        batches=batches,
        device=device,
        attention_backend=args.attention_backend,
    )
    fsdp_summary = run_parallel_precision_check(
        model_dir=model_dir,
        mode_name="fsdp",
        candidate_mode="forge_fsdp2",
        steps=args.steps,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        prompt_len=args.prompt_len,
        latent_channels=args.latent_channels,
        prompt_dim=args.joint_dim,
        seed=args.seed,
        attention_backend=args.attention_backend,
        world_size=args.world_size,
        use_fsdp=True,
        use_ulysses=False,
        reference_losses=single_gpu_summary["reference_losses"],
        reference_duration_s=single_gpu_summary["reference_duration_s"],
        single_gpu_losses=single_gpu_summary["candidate_losses"],
        single_gpu_duration_s=single_gpu_summary["candidate_duration_s"],
    )
    ulysses_summary = run_parallel_precision_check(
        model_dir=model_dir,
        mode_name="ulysses",
        candidate_mode="forge_ulysses",
        steps=args.steps,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        prompt_len=args.prompt_len,
        latent_channels=args.latent_channels,
        prompt_dim=args.joint_dim,
        seed=args.seed,
        attention_backend=args.attention_backend,
        world_size=args.world_size,
        use_fsdp=False,
        use_ulysses=True,
        reference_losses=single_gpu_summary["reference_losses"],
        reference_duration_s=single_gpu_summary["reference_duration_s"],
        single_gpu_losses=single_gpu_summary["candidate_losses"],
        single_gpu_duration_s=single_gpu_summary["candidate_duration_s"],
    )
    fsdp_ulysses_summary = run_parallel_precision_check(
        model_dir=model_dir,
        mode_name="fsdp_ulysses",
        candidate_mode="forge_fsdp2_ulysses",
        steps=args.steps,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        prompt_len=args.prompt_len,
        latent_channels=args.latent_channels,
        prompt_dim=args.joint_dim,
        seed=args.seed,
        attention_backend=args.attention_backend,
        world_size=args.world_size,
        use_fsdp=True,
        use_ulysses=True,
        reference_losses=single_gpu_summary["reference_losses"],
        reference_duration_s=single_gpu_summary["reference_duration_s"],
        single_gpu_losses=single_gpu_summary["candidate_losses"],
        single_gpu_duration_s=single_gpu_summary["candidate_duration_s"],
    )

    summary = {
        "status": (
            "success"
            if (
                single_gpu_summary["status"] == "success"
                and fsdp_summary["status"] == "success"
                and ulysses_summary["status"] == "success"
                and fsdp_ulysses_summary["status"] == "success"
            )
            else "error"
        ),
        "model_dir": str(model_dir),
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "height": args.height,
            "width": args.width,
            "prompt_len": args.prompt_len,
            "patch_size": args.patch_size,
            "latent_channels": args.latent_channels,
            "out_channels": out_channels,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "joint_dim": args.joint_dim,
            "attention_backend": args.attention_backend,
            "precision": dtype_name(TRAIN_DTYPE),
            "world_size": args.world_size,
            "min_free_gb_per_gpu": args.min_free_gb_per_gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "preflight": preflight,
        "single_gpu": single_gpu_summary,
        "fsdp": fsdp_summary,
        "ulysses": ulysses_summary,
        "fsdp_ulysses": fsdp_ulysses_summary,
    }

    output_path = args.output_dir / "qwen_image_m2_precision_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"summary_written={output_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
