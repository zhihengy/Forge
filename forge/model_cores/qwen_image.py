from __future__ import annotations

import math
from dataclasses import dataclass
from math import prod
from typing import Any

import torch.distributed as dist

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm
from diffusers.utils import apply_lora_scale, deprecate, logging
from diffusers.utils.torch_utils import lru_cache_unless_export, maybe_allow_in_graph

from forge.model_cores.base_dit import BaseDiT
from forge.parallel.ulysses import (
    UlyssesParallelContext,
    build_ulysses_context,
    gather_tensor,
    shard_tensor,
    ulysses_heads_to_sequence,
    ulysses_sequence_to_heads,
)
from forge.model_cores.registry import register_model_core

try:
    from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
except Exception:  # noqa: BLE001
    class FromOriginalModelMixin:  # type: ignore[no-redef]
        pass

    class PeftAdapterMixin:  # type: ignore[no-redef]
        pass


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass(frozen=True)
class QwenImageDiTConfig:
    patch_size: int = 2
    in_channels: int = 64
    out_channels: int | None = 16
    num_layers: int = 60
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 3584
    guidance_embeds: bool = False
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    zero_cond_t: bool = False
    use_additional_t_cond: bool = False
    use_layer3d_rope: bool = False

    def to_init_kwargs(self) -> dict[str, Any]:
        return {
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "num_layers": self.num_layers,
            "attention_head_dim": self.attention_head_dim,
            "num_attention_heads": self.num_attention_heads,
            "joint_attention_dim": self.joint_attention_dim,
            "guidance_embeds": self.guidance_embeds,
            "axes_dims_rope": self.axes_dims_rope,
            "zero_cond_t": self.zero_cond_t,
            "use_additional_t_cond": self.use_additional_t_cond,
            "use_layer3d_rope": self.use_layer3d_rope,
        }


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None].to(x.device)
        sin = sin[None, None].to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_rotated * freqs_cis.unsqueeze(1)).flatten(3).type_as(x)


def compute_text_seq_len_from_mask(
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor | None,
) -> tuple[int, torch.Tensor | None]:
    batch_size, text_seq_len = encoder_hidden_states.shape[:2]
    if encoder_hidden_states_mask is None:
        return text_seq_len, None

    if encoder_hidden_states_mask.shape[:2] != (batch_size, text_seq_len):
        raise ValueError(
            f"`encoder_hidden_states_mask` shape {encoder_hidden_states_mask.shape} must match "
            f"(batch_size, text_seq_len)=({batch_size}, {text_seq_len})."
        )
    if encoder_hidden_states_mask.dtype != torch.bool:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)
    return text_seq_len, encoder_hidden_states_mask


def build_joint_attention_mask(
    encoder_hidden_states_mask: torch.Tensor,
    *,
    image_seq_len: int,
    ulysses_degree: int = 1,
) -> torch.Tensor:
    batch_size = encoder_hidden_states_mask.shape[0]
    image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=encoder_hidden_states_mask.device)
    if ulysses_degree <= 1:
        joint_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
        return joint_mask[:, None, None, :]

    text_seq_len = encoder_hidden_states_mask.shape[1]
    if text_seq_len % ulysses_degree != 0 or image_seq_len % ulysses_degree != 0:
        raise ValueError(
            "Ulysses attention mask construction expects both text and image sequence lengths "
            f"to be divisible by degree={ulysses_degree}, got text_seq_len={text_seq_len}, image_seq_len={image_seq_len}."
        )

    interleaved_chunks: list[torch.Tensor] = []
    text_chunks = encoder_hidden_states_mask.chunk(ulysses_degree, dim=1)
    image_chunks = image_mask.chunk(ulysses_degree, dim=1)
    for text_chunk, image_chunk in zip(text_chunks, image_chunks, strict=True):
        interleaved_chunks.extend((text_chunk, image_chunk))

    joint_mask = torch.cat(interleaved_chunks, dim=1)
    return joint_mask[:, None, None, :]


def run_ulysses_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    backend: Any,
    context: UlyssesParallelContext,
) -> torch.Tensor:
    query = ulysses_heads_to_sequence(query, context=context)
    key = ulysses_heads_to_sequence(key, context=context)
    value = ulysses_heads_to_sequence(value, context=context)
    output = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        backend=backend,
    )
    return ulysses_sequence_to_heads(output, context=context)


def shard_rotary_embeddings(
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ulysses_context: UlyssesParallelContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    img_freqs, txt_freqs = image_rotary_emb
    return (
        shard_tensor(img_freqs, dim=0, context=ulysses_context),
        shard_tensor(txt_freqs, dim=0, context=ulysses_context),
    )


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, use_additional_t_cond: bool = False) -> None:
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        addition_t_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            conditioning = conditioning + self.addition_t_embedding(addition_t_cond).to(dtype=hidden_states.dtype)
        return conditioning


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.scale_rope = scale_rope

    @staticmethod
    def rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        if dim % 2 != 0:
            raise ValueError(f"RoPE axis dim must be even, got {dim}.")
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        video_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        *,
        txt_seq_lens: list[int] | None = None,
        device: torch.device | None = None,
        max_txt_seq_len: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
                "Please use `max_txt_seq_len` instead.",
                standard_warn=False,
            )
            if max_txt_seq_len is None:
                max_txt_seq_len = max(txt_seq_lens) if isinstance(txt_seq_lens, list) else txt_seq_lens

        if max_txt_seq_len is None:
            raise ValueError("Either `max_txt_seq_len` or `txt_seq_lens` (deprecated) must be provided.")

        if isinstance(video_fhw, list) and len(video_fhw) > 1:
            first_fhw = video_fhw[0]
            if not all(fhw == first_fhw for fhw in video_fhw):
                logger.warning(
                    "Batch inference with variable-sized images is not currently supported in QwenEmbedRope. "
                    "Using the first image shape for RoPE computation."
                )

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            vid_freqs.append(video_freq)
            max_vid_index = max(height // 2, width // 2, max_vid_index) if self.scale_rope else max(
                height, width, max_vid_index
            )

        txt_freqs = self.pos_freqs.to(device)[max_vid_index : max_vid_index + int(max_txt_seq_len), ...]
        return torch.cat(vid_freqs, dim=0), txt_freqs

    @lru_cache_unless_export(maxsize=128)
    def _compute_video_freqs(
        self,
        frame: int,
        height: int,
        width: int,
        idx: int = 0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        pos_freqs = self.pos_freqs.to(device) if device is not None else self.pos_freqs
        neg_freqs = self.neg_freqs.to(device) if device is not None else self.neg_freqs
        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        return torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(frame * height * width, -1)


class QwenEmbedLayer3DRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.scale_rope = scale_rope

    @staticmethod
    def rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        if dim % 2 != 0:
            raise ValueError(f"RoPE axis dim must be even, got {dim}.")
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        video_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        *,
        max_txt_seq_len: int | torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(video_fhw, list) and len(video_fhw) > 1:
            first_entry = video_fhw[0]
            if not all(entry == first_entry for entry in video_fhw):
                logger.warning(
                    "Batch inference with variable-sized images is not currently supported in QwenEmbedLayer3DRope. "
                    "Using the first sample layout for RoPE computation."
                )

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx != layer_num:
                video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            else:
                video_freq = self._compute_condition_freqs(frame, height, width, device)
            vid_freqs.append(video_freq)
            max_vid_index = max(height // 2, width // 2, max_vid_index) if self.scale_rope else max(
                height, width, max_vid_index
            )

        max_vid_index = max(max_vid_index, layer_num)
        txt_freqs = self.pos_freqs.to(device)[max_vid_index : max_vid_index + int(max_txt_seq_len), ...]
        return torch.cat(vid_freqs, dim=0), txt_freqs

    @lru_cache_unless_export(maxsize=None)
    def _compute_video_freqs(
        self,
        frame: int,
        height: int,
        width: int,
        idx: int = 0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        pos_freqs = self.pos_freqs.to(device) if device is not None else self.pos_freqs
        neg_freqs = self.neg_freqs.to(device) if device is not None else self.neg_freqs
        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        return torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(frame * height * width, -1)

    @lru_cache_unless_export(maxsize=None)
    def _compute_condition_freqs(
        self,
        frame: int,
        height: int,
        width: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        pos_freqs = self.pos_freqs.to(device) if device is not None else self.pos_freqs
        neg_freqs = self.neg_freqs.to(device) if device is not None else self.neg_freqs
        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        return torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(frame * height * width, -1)


class QwenDoubleStreamAttnProcessor2_0:
    _attention_backend = None
    _ulysses_context: UlyssesParallelContext | None = None

    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0 or later.")

    def set_ulysses_context(self, context: UlyssesParallelContext | None) -> None:
        self._ulysses_context = context

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream).")

        seq_txt = encoder_hidden_states.shape[1]
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        if self._ulysses_context is not None and self._ulysses_context.degree > 1:
            joint_hidden_states = run_ulysses_attention(
                joint_query,
                joint_key,
                joint_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
                backend=self._attention_backend,
                context=self._ulysses_context,
            )
        else:
            joint_hidden_states = dispatch_attention_fn(
                joint_query,
                joint_key,
                joint_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(dtype=img_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
        return img_attn_output, txt_attn_output


@maybe_allow_in_graph
class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
    ) -> None:
        super().__init__()
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.zero_cond_t = zero_cond_t

    @staticmethod
    def _modulate(
        x: torch.Tensor,
        mod_params: torch.Tensor,
        index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if index is not None:
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]
            index_expanded = index.unsqueeze(-1)
            shift_result = torch.where(index_expanded == 0, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
            scale_result = torch.where(index_expanded == 0, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
            gate_result = torch.where(index_expanded == 0, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)
        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        modulate_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1, modulate_index)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2, modulate_index)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class QwenImageDiT(BaseDiT, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin):
    _no_split_modules = ["QwenImageTransformerBlock"]
    _repeated_blocks = ["QwenImageTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
    ) -> None:
        super().__init__()
        del guidance_embeds
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = (
            QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
            if not use_layer3d_rope
            else QwenEmbedLayer3DRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        )
        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            use_additional_t_cond=use_additional_t_cond,
        )
        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.zero_cond_t = zero_cond_t
        self._ulysses_context: UlyssesParallelContext | None = None

    def enable_ulysses_parallelism(self, *, degree: int, group: dist.ProcessGroup | None = None) -> None:
        context = build_ulysses_context(degree=degree, group=group)
        self._ulysses_context = context
        for module in self.modules():
            if not isinstance(module, Attention):
                continue
            processor = module.processor
            if processor is not None and hasattr(processor, "set_ulysses_context"):
                processor.set_ulysses_context(context)

    def disable_ulysses_parallelism(self) -> None:
        self._ulysses_context = None
        for module in self.modules():
            if not isinstance(module, Attention):
                continue
            processor = module.processor
            if processor is not None and hasattr(processor, "set_ulysses_context"):
                processor.set_ulysses_context(None)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples: list[torch.Tensor] | None = None,
        additional_t_cond: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        if encoder_hidden_states is None:
            raise ValueError("QwenImageDiT requires encoder_hidden_states.")
        if timestep is None:
            raise ValueError("QwenImageDiT requires timestep.")
        if img_shapes is None:
            raise ValueError("QwenImageDiT requires img_shapes.")

        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
                "Please use `encoder_hidden_states_mask` instead.",
                standard_warn=False,
            )

        ulysses_context = self._ulysses_context
        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum(prod(s) for s in sample[1:]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        text_seq_len, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states,
            encoder_hidden_states_mask,
        )

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )
        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            image_seq_len = hidden_states.shape[1]
            block_attention_kwargs["attention_mask"] = build_joint_attention_mask(
                encoder_hidden_states_mask,
                image_seq_len=image_seq_len,
                ulysses_degree=ulysses_context.degree if ulysses_context is not None else 1,
            )

        if ulysses_context is not None and ulysses_context.degree > 1:
            if controlnet_block_samples is not None:
                raise NotImplementedError("ControlNet block samples are not supported with self-owned Ulysses yet.")
            hidden_states = shard_tensor(hidden_states, dim=1, context=ulysses_context)
            encoder_hidden_states = shard_tensor(encoder_hidden_states, dim=1, context=ulysses_context)
            image_rotary_emb = shard_rotary_embeddings(image_rotary_emb, ulysses_context)
            if modulate_index is not None:
                modulate_index = shard_tensor(modulate_index, dim=1, context=ulysses_context)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
            )

            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        output = self.proj_out(self.norm_out(hidden_states, temb))
        if ulysses_context is not None and ulysses_context.degree > 1:
            output = gather_tensor(output, dim=1, context=ulysses_context)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


register_model_core("qwen_image", QwenImageDiT)
