# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent torch_dist checkpoints for site-major MIMO ranks."""

import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist

from examples.mimo.training.args import build_module_grid_specs, configure_site_rank_layout
from examples.mimo.training.runtime import wrap_active_modules_with_ddp
from examples.mimo.training.topology import create_topology
from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedTensor
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer import OptimizerConfig
from tests.unit_tests.models.mimo.test_mimo_hetero_runtime import _args as _runtime_args
from tests.unit_tests.models.mimo.test_mimo_hetero_runtime import _build_unwrapped_mimo_model
from tests.unit_tests.models.mimo.test_mimo_site_major import _args as _site_args
from tests.unit_tests.test_utilities import Utils


def _sharded_tensor_leaves(value, path=()):
    if isinstance(value, ShardedTensor):
        if value.data is not None:
            yield path, value.data
    elif isinstance(value, dict):
        # DistOpt creates bucket padding with torch.empty() while materializing
        # sharded state. It is not an optimizer slot and is regenerated on load.
        padding = value.get("padding")
        if padding is True or (
            isinstance(padding, LocalNonpersistentObject) and padding.obj is True
        ):
            return
        for key, child in value.items():
            yield from _sharded_tensor_leaves(child, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _sharded_tensor_leaves(child, path + (index,))


def _build_model_and_optimizer(topology, seed):
    torch.manual_seed(seed)
    model = _build_unwrapped_mimo_model(topology, bf16=True)
    wrap_active_modules_with_ddp(
        _runtime_args(fp32=False),
        model,
        topology,
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True, num_distributed_optimizer_instances=2
        ),
    )
    active = next(iter(model.modality_submodules.values()), None) or model.language_model
    active.broadcast_params()
    optimizer = get_mimo_optimizer(
        model,
        OptimizerConfig(
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            bf16=True,
            use_distributed_optimizer=True,
            clip_grad=0.0,
        ),
    )
    return model, active, optimizer


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs per node")
def test_mimo_site_checkpoint_round_trip_with_distributed_optimizer():
    if Utils.world_size != 8:
        pytest.skip("Test requires eight ranks")

    Utils.initialize_distributed()
    site_ids = (0,) * 4 + (1,) * 4
    args = _site_args()
    args.site_id = site_ids[dist.get_rank()]
    configure_site_rank_layout(args, list(site_ids), "images")
    topology = create_topology(
        build_module_grid_specs(args, 8, "images"),
        site_ids=site_ids,
        num_optimizer_instances=2,
    )
    root = [None]
    try:
        model_a, active_a, optimizer_a = _build_model_and_optimizer(topology, seed=1234)
        optimizer_a.zero_grad()
        loss = sum(p.float().square().sum() for p in active_a.parameters() if p.requires_grad)
        loss.backward()
        active_a.finish_grad_sync()
        assert optimizer_a.step()[0]
        expected = {name: p.detach().clone() for name, p in model_a.named_parameters()}

        if dist.get_rank() == 0:
            root[0] = tempfile.mkdtemp(prefix="mimo_site_ckpt_")
        dist.broadcast_object_list(root, src=0)
        checkpoint_dir = os.path.join(root[0], f"site-{args.site_id}")
        if topology.site_group.rank() == 0:
            os.makedirs(checkpoint_dir)
        dist.barrier(group=topology.site_group)

        metadata = {
            "distrib_optim_sharding_type": "dp_reshardable",
            "dp_cp_group": topology.module_pgs[
                "images" if "images" in model_a.modality_submodules else "language"
            ].dp_cp_gtp_remat,
        }
        state_a = {
            "model": model_a.sharded_state_dict(),
            "optimizer": optimizer_a.sharded_state_dict(
                model_a.sharded_state_dict(), metadata=metadata
            ),
        }
        expected_optimizer = {
            path: tensor.detach().clone()
            for path, tensor in _sharded_tensor_leaves(state_a["optimizer"])
        }
        assert expected_optimizer
        save(
            state_a,
            checkpoint_dir,
            sharded_strategy=FullyParallelSaveStrategyWrapper(
                TorchDistSaveShardedStrategy(process_group=topology.site_group),
                topology.site_group,
            ),
            process_group=topology.site_group,
            preferred_writer_ranks=(),
        )
        dist.barrier(group=topology.site_group)

        model_b, _, optimizer_b = _build_model_and_optimizer(topology, seed=5678)
        state_b = {
            "model": model_b.sharded_state_dict(),
            "optimizer": optimizer_b.sharded_state_dict(
                model_b.sharded_state_dict(), is_loading=True, metadata=metadata
            ),
        }
        loaded = load(
            state_b,
            checkpoint_dir,
            sharded_strategy=FullyParallelLoadStrategyWrapper(
                TorchDistLoadShardedStrategy(), topology.site_group
            ),
            process_group=topology.site_group,
            preferred_reader_ranks=(),
        )
        model_b.load_state_dict(loaded["model"])
        optimizer_b.load_state_dict(loaded["optimizer"])
        for name, parameter in model_b.named_parameters():
            assert torch.equal(parameter, expected[name]), name
        actual_optimizer = dict(
            _sharded_tensor_leaves(
                optimizer_b.sharded_state_dict(model_b.sharded_state_dict(), metadata=metadata)
            )
        )
        assert actual_optimizer.keys() == expected_optimizer.keys()
        for path, tensor in actual_optimizer.items():
            assert torch.equal(tensor, expected_optimizer[path]), path
    finally:
        if root[0] is not None:
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(root[0])
        topology.destroy()
        Utils.destroy_model_parallel()
