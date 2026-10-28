# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.multimodal.utils import patchify_image, unpatchify_image
from megatron.core.models.vision.clip_vit_model import (
    CLIPViTModel,
    get_num_image_embeddings,
    get_num_video_embeddings,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestCLIPViTModel:
    """Test CLIP ViT model."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        self.model = CLIPViTModel(
            transformer_config, transformer_layer_spec, img_h=336, img_w=336, patch_dim=14
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.model, CLIPViTModel)

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 174720

    def test_set_input_tensor(self):
        # [s, b, h] expected to the transformer.
        expected_shape = (577, 2, 64)
        input_tensor = torch.zeros(expected_shape)

        self.model.set_input_tensor(input_tensor)

        assert self.model.decoder.input_tensor.shape == torch.Size(expected_shape)

    def test_forward(self):
        self.model.cuda()

        img = torch.zeros((2, 3, 336, 336)).cuda()

        out = self.model.forward(img)
        assert out.shape == torch.Size([2, 577, 64])

    def test_save_load(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)

        self.model.load_state_dict(torch.load(path))


def test_vision_encoder_grad_sync_with_site_local_optimizer_instances():
    """Vision weights follow the same intra- and inter-instance DP path as language weights."""
    if Utils.world_size != 8:
        pytest.skip("Test requires eight ranks")

    try:
        Utils.initialize_model_parallel(num_distributed_optimizer_instances=2)
        parallel_state.initialize_site_parallel(
            Utils.rank // 4, require_complete_optimizer_instances=True
        )
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        vision = CLIPViTModel(
            config,
            get_gpt_layer_with_transformer_engine_spec(),
            img_h=28,
            img_w=28,
            patch_dim=14,
        ).cuda()
        model = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                num_distributed_optimizer_instances=2,
            ),
            vision,
        )
        images = torch.ones(1, 3, 28, 28, device="cuda") * (Utils.rank + 1)
        model(images).float().square().mean().backward()
        model.finish_grad_sync()

        inner_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True, with_gtp_remat=False, partial_data_parallel=True
        )
        outer_group = parallel_state.get_inter_distributed_optimizer_instance_group()
        for buffer in model.buffers:
            shard_size = buffer.grad_data.numel() // inner_group.size()
            start = inner_group.rank() * shard_size
            local_shard = buffer.grad_data[start : start + shard_size]
            assert torch.isfinite(local_shard).all()
            replicas = [torch.empty_like(local_shard) for _ in range(outer_group.size())]
            torch.distributed.all_gather(replicas, local_shard, group=outer_group)
            torch.testing.assert_close(replicas[0], replicas[1])
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(
    "vision_model,pixel_shuffle,tile_tags,expected",
    [
        ("clip", False, False, 1024),
        ("internvit300M", False, False, 1024),
        ("clip", True, False, 256),
        ("internvit300M", True, True, 262),
    ],
)
def test_get_num_image_embeddings(vision_model, pixel_shuffle, tile_tags, expected):
    assert (
        get_num_image_embeddings(
            448, 448, 14, vision_model, True, 1, pixel_shuffle, tile_tags, 0, "nemotron5"
        )
        == expected
    )


@pytest.mark.parametrize("batched", [False, True])
def test_patchify_unpatchify_roundtrip(batched):
    image = torch.arange(3 * 4 * 6).reshape(3, 4, 6)
    if batched:
        image = torch.stack((image, image + image.numel()))

    patches = patchify_image(image, patch_dim=2)

    assert torch.equal(unpatchify_image(patches, img_H=4, img_W=6, patch_dim=2), image)


@pytest.mark.parametrize(
    "pixel_shuffle,attn_pooling,expected_spatial",
    [(False, False, 32), (True, False, 8), (False, True, 4), (True, True, 1)],
)
@pytest.mark.parametrize(
    "disable_vision_class_token,use_tile_tags,expected_non_spatial",
    [(True, False, 0), (False, False, 2), (True, True, 6), (False, True, 8)],
)
def test_get_num_image_embeddings_applies_attention_pooling(
    pixel_shuffle,
    attn_pooling,
    expected_spatial,
    disable_vision_class_token,
    use_tile_tags,
    expected_non_spatial,
):
    # The 4x8 patch grid is reduced by pixel shuffle and then 2x4 image pooling.
    # Class tokens and tile tags must be added without spatial reduction.
    assert (
        get_num_image_embeddings(
            img_h=8,
            img_w=16,
            patch_dim=2,
            vision_model_type="clip",
            disable_vision_class_token=disable_vision_class_token,
            class_token_len=2,
            pixel_shuffle=pixel_shuffle,
            use_tile_tags=use_tile_tags,
            max_num_tiles=1,
            tokenizer_type="nemotron5",
            attn_pooling=attn_pooling,
            attn_pooling_img_h=2,
            attn_pooling_img_w=4,
        )
        == expected_spatial + expected_non_spatial
    )


def test_get_num_video_embeddings_applies_temporal_and_attention_pooling():
    assert (
        get_num_video_embeddings(
            num_frames=4,
            video_temporal_patch_size=2,
            img_h=8,
            img_w=8,
            patch_dim=2,
            vision_model_type="siglip",
            disable_vision_class_token=True,
            class_token_len=0,
            pixel_shuffle=False,
            attn_pooling=True,
            attn_pooling_img_h=1,
            attn_pooling_img_w=1,
            attn_pooling_video_h=2,
            attn_pooling_video_w=2,
        )
        == 8
    )
