from __future__ import annotations

import unittest

import torch

from forge.adapters.qwen_image import QwenImageCheckpointAdapter
from forge.architectures.qwen_image import QwenImageArchitecture
from forge.model_cores.qwen_image import QwenImageDiT, QwenImageDiTConfig, build_joint_attention_mask
from forge.model_cores.registry import get_model_core
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel


def make_small_config() -> QwenImageDiTConfig:
    return QwenImageDiTConfig(
        patch_size=1,
        in_channels=3,
        out_channels=3,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=10,
        axes_dims_rope=(2, 2, 4),
    )


class QwenImageOwnedDiTTest(unittest.TestCase):
    def test_qwen_image_core_registry_and_architecture_build_owned_model(self) -> None:
        config = make_small_config()
        self.assertIs(get_model_core("qwen_image"), QwenImageDiT)

        architecture = QwenImageArchitecture(model_config=config)
        model = architecture.build_model()
        self.assertIsInstance(model, QwenImageDiT)
        self.assertEqual(architecture.parallel_spec.native_sequence_parallel.supported_algorithms, ("ulysses",))

    def test_qwen_image_checkpoint_adapter_builds_owned_config(self) -> None:
        adapter = QwenImageCheckpointAdapter()
        model_config = adapter.build_model_config(
            {
                "_class_name": "QwenImageTransformer2DModel",
                "patch_size": 1,
                "in_channels": 3,
                "out_channels": 5,
                "num_layers": 4,
                "attention_head_dim": 8,
                "num_attention_heads": 2,
                "joint_attention_dim": 10,
                "axes_dims_rope": [2, 2, 4],
                "unused_key": "ignored",
            }
        )

        self.assertEqual(
            model_config,
            QwenImageDiTConfig(
                patch_size=1,
                in_channels=3,
                out_channels=5,
                num_layers=4,
                attention_head_dim=8,
                num_attention_heads=2,
                joint_attention_dim=10,
                axes_dims_rope=(2, 2, 4),
            ),
        )

    def test_build_joint_attention_mask_interleaves_ulysses_shards(self) -> None:
        text_mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)

        vanilla_mask = build_joint_attention_mask(text_mask, image_seq_len=4)
        self.assertTrue(
            torch.equal(
                vanilla_mask,
                torch.tensor([[[[True, True, False, False, True, True, True, True]]]], dtype=torch.bool),
            )
        )

        ulysses_mask = build_joint_attention_mask(text_mask, image_seq_len=4, ulysses_degree=2)
        self.assertTrue(
            torch.equal(
                ulysses_mask,
                torch.tensor([[[[True, True, True, True, False, False, True, True]]]], dtype=torch.bool),
            )
        )

    @torch.no_grad()
    def test_qwen_image_owned_dit_matches_reference_forward(self) -> None:
        torch.manual_seed(0)
        config = make_small_config()
        reference_model = QwenImageTransformer2DModel(**config.to_init_kwargs()).eval()
        owned_model = QwenImageDiT(**config.to_init_kwargs()).eval()

        adapter = QwenImageCheckpointAdapter()
        owned_model.load_state_dict(adapter.remap_state_dict(reference_model.state_dict()))

        hidden_states = torch.randn(2, 6, config.in_channels)
        encoder_hidden_states = torch.randn(2, 4, config.joint_attention_dim)
        encoder_hidden_states_mask = torch.tensor(
            [[True, True, False, False], [True, False, True, False]],
            dtype=torch.bool,
        )
        timestep = torch.tensor([12, 34], dtype=torch.long)
        img_shapes = [(1, 2, 3), (1, 2, 3)]

        reference_output = reference_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]
        owned_output = owned_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]

        self.assertTrue(torch.allclose(owned_output, reference_output, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
