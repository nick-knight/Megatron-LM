# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Site-major heterogeneous MIMO process groups and bridge placement."""

import pytest
import torch
import torch.distributed as dist

from examples.mimo.training.args import (
    add_hetero_grid_args,
    build_module_grid_specs,
    configure_site_rank_layout,
)
from examples.mimo.training.runtime import wrap_active_modules_with_ddp
from examples.mimo.training.topology import create_topology
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer import OptimizerConfig
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from tests.unit_tests.models.mimo.test_mimo_hetero_runtime import _args as _runtime_args
from tests.unit_tests.models.mimo.test_mimo_hetero_runtime import _build_unwrapped_mimo_model
from tests.unit_tests.test_utilities import Utils


def _args():
    import argparse

    parser = argparse.ArgumentParser()
    add_hetero_grid_args(parser)
    args = parser.parse_args(
        "--mimo-encoder-tp 2 --mimo-encoder-dp 2 "
        "--mimo-llm-tp 2 --mimo-llm-dp 2 --mimo-llm-ep 1".split()
    )
    args.micro_batch_size = 1
    args.num_experts = 0
    args.site_id = 0
    args.num_distributed_optimizer_instances = 2
    args.sequence_parallel = False
    return args


@pytest.mark.parametrize("expert_parallel_size", [1, 2])
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs per node")
def test_site_major_mimo_keeps_optimizer_shards_and_bridge_local(expert_parallel_size):
    if Utils.world_size != 8:
        pytest.skip("Test requires eight ranks")

    Utils.initialize_distributed()
    site_ids = (0,) * 4 + (1,) * 4
    args = _args()
    args.site_id = site_ids[dist.get_rank()]
    args.mimo_llm_ep = expert_parallel_size
    args.num_experts = expert_parallel_size
    configure_site_rank_layout(args, list(site_ids), "images")
    specs = build_module_grid_specs(args, 8, "images")
    topology = create_topology(
        specs, site_ids=site_ids, num_optimizer_instances=2
    )
    try:
        assert set(dist.get_process_group_ranks(topology.site_group)) == {
            rank for rank, site in enumerate(site_ids) if site == args.site_id
        }
        module = (
            "images"
            if dist.get_rank() in topology.grids["images"].rank_ids
            else MIMO_LANGUAGE_MODULE_KEY
        )
        pgc = topology.module_pgs[module]
        for group in (pgc.intra_dp_cp, pgc.intra_expt_dp, pgc.intra_dist_opt):
            assert {site_ids[rank] for rank in dist.get_process_group_ranks(group)} == {args.site_id}
        assert pgc.inter_dist_opt.size() == 2

        bridge = BridgeCommunicator(topology.grids["images"], topology.grids[MIMO_LANGUAGE_MODULE_KEY])
        for rank, info in bridge.comm_map.items():
            for peer in info.send_to_ranks + info.recv_from_ranks:
                assert site_ids[rank] == site_ids[peer]
    finally:
        BridgeCommunicator.destroy_broadcast_pgs()
        BridgeCommunicator.destroy_bridge_pgs()
        topology.destroy()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs per node")
def test_site_major_mimo_bridge_transfers_activations_and_gradients_locally():
    if Utils.world_size != 8:
        pytest.skip("Test requires eight ranks")

    Utils.initialize_distributed()
    site_ids = (0,) * 4 + (1,) * 4
    args = _args()
    args.site_id = site_ids[dist.get_rank()]
    configure_site_rank_layout(args, list(site_ids), "images")
    topology = create_topology(
        build_module_grid_specs(args, 8, "images"),
        site_ids=site_ids,
        num_optimizer_instances=2,
    )
    try:
        bridge = BridgeCommunicator(
            topology.grids["images"],
            topology.grids[MIMO_LANGUAGE_MODULE_KEY],
            comm_dtype=torch.float32,
            tensor_ndim=2,
        )
        if bridge.is_current_rank_in_grid(bridge.src_grid):
            bridge.send_forward(torch.full((2, 3), args.site_id + 1.0, device="cuda"))
        else:
            received = bridge.recv_forward()
            assert torch.equal(
                received, torch.full((2, 3), args.site_id + 1.0, device="cuda")
            )

        if bridge.is_current_rank_in_grid(bridge.dest_grid):
            bridge.send_backward(torch.full((2, 3), args.site_id + 10.0, device="cuda"))
        else:
            received = bridge.recv_backward()
            assert torch.equal(
                received, torch.full((2, 3), args.site_id + 10.0, device="cuda")
            )
        dist.barrier()
    finally:
        BridgeCommunicator.destroy_broadcast_pgs()
        BridgeCommunicator.destroy_bridge_pgs()
        topology.destroy()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs per node")
def test_site_major_mimo_nonuniform_three_site_placement():
    if Utils.world_size != 12:
        pytest.skip("Test requires twelve ranks")

    Utils.initialize_distributed()
    site_ids = (0,) * 3 + (1,) * 3 + (2,) * 6
    args = _args()
    args.site_id = site_ids[dist.get_rank()]
    args.mimo_encoder_tp = 1
    args.mimo_encoder_dp = 4
    args.mimo_llm_dp = 4
    args.num_distributed_optimizer_instances = 4
    configure_site_rank_layout(args, list(site_ids), "images")
    assert args._mimo_site_instance_counts == (1, 1, 2)
    topology = create_topology(
        build_module_grid_specs(args, 12, "images"),
        site_ids=site_ids,
        num_optimizer_instances=4,
    )
    try:
        module = (
            "images"
            if dist.get_rank() in topology.grids["images"].rank_ids
            else MIMO_LANGUAGE_MODULE_KEY
        )
        pgc = topology.module_pgs[module]
        assert {site_ids[rank] for rank in dist.get_process_group_ranks(pgc.intra_dist_opt)} == {
            args.site_id
        }
        assert pgc.inter_dist_opt.size() == 4
        bridge = BridgeCommunicator(
            topology.grids["images"],
            topology.grids[MIMO_LANGUAGE_MODULE_KEY],
            comm_dtype=torch.float32,
            tensor_ndim=2,
        )
        for rank, info in bridge.comm_map.items():
            for peer in info.send_to_ranks + info.recv_from_ranks:
                assert site_ids[rank] == site_ids[peer]
        if bridge.is_current_rank_in_grid(bridge.src_grid):
            bridge.send_forward(torch.full((2, 3), args.site_id + 1.0, device="cuda"))
        else:
            received = bridge.recv_forward()
            assert torch.equal(
                received, torch.full((2, 3), args.site_id + 1.0, device="cuda")
            )
        dist.barrier()
    finally:
        BridgeCommunicator.destroy_broadcast_pgs()
        BridgeCommunicator.destroy_bridge_pgs()
        topology.destroy()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs per node")
def test_site_major_mimo_optimizer_step_uses_local_shards():
    if Utils.world_size != 8:
        pytest.skip("Test requires eight ranks")

    Utils.initialize_distributed()
    site_ids = (0,) * 4 + (1,) * 4
    args = _args()
    args.site_id = site_ids[dist.get_rank()]
    configure_site_rank_layout(args, list(site_ids), "images")
    topology = create_topology(
        build_module_grid_specs(args, 8, "images"),
        site_ids=site_ids,
        num_optimizer_instances=2,
    )
    try:
        model = _build_unwrapped_mimo_model(topology, bf16=True)
        wrap_active_modules_with_ddp(
            _runtime_args(fp32=False),
            model,
            topology,
            ddp_config=DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                num_distributed_optimizer_instances=2,
            ),
        )
        active = (
            model.modality_submodules["images"]
            if "images" in model.modality_submodules
            else model.language_model
        )
        # DistOpt copies FP32 master parameters at construction, so align model
        # replicas before it captures those parameters.
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
        optimizer.zero_grad()
        loss = sum(
            parameter.float().square().sum()
            for parameter in active.parameters()
            if parameter.requires_grad
        )
        module = "images" if "images" in model.modality_submodules else MIMO_LANGUAGE_MODULE_KEY
        pgc = topology.module_pgs[module]
        (loss * (pgc.dp.rank() + 1)).backward()
        active.finish_grad_sync()
        success, grad_norm, _ = optimizer.step()
        assert success
        assert torch.isfinite(torch.tensor(grad_norm))
        parameter = next(parameter for parameter in active.parameters() if parameter.requires_grad)
        gathered = [torch.empty_like(parameter) for _ in range(pgc.inter_dist_opt.size())]
        dist.all_gather(gathered, parameter.detach(), group=pgc.inter_dist_opt)
        assert all(torch.equal(gathered[0], peer) for peer in gathered[1:])
    finally:
        topology.destroy()
        Utils.destroy_model_parallel()
