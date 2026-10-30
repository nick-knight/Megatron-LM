# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise distinct dense/expert instance partitions through the optimizer factory."""

from copy import deepcopy

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel,
    _build_hybrid_dp_mesh,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

_NUMEL = 1024


class _Model(torch.nn.Module):
    def __init__(self, config, omit_expert):
        super().__init__()
        self.config = config
        self.dense = torch.nn.Parameter(
            torch.ones(_NUMEL, device="cuda", dtype=config.params_dtype)
        )
        if not omit_expert or ps.get_expert_model_parallel_rank() == 0:
            self.expert = torch.nn.Parameter(torch.ones_like(self.dense))
            self.expert.allreduce = False
        for name, parameter in self.named_parameters():
            group = (
                ps.get_gtp_weight_remat_group()
                if name == "dense"
                else ps.get_expert_gtp_weight_remat_group()
            )
            parameter.is_gtp_weight_remat = group.size() > 1

    def sharded_state_dict(self):
        result = {
            "dense": ShardedTensor.from_rank_offsets(
                "dense",
                self.dense,
                (0, ps.get_gtp_weight_remat_rank(), ps.get_gtp_weight_remat_world_size()),
                replica_id=(
                    0,
                    0,
                    ps.get_data_parallel_rank(with_context_parallel=True, with_gtp_remat=False),
                ),
            )
        }
        if hasattr(self, "expert"):
            result["expert"] = ShardedTensor.from_rank_offsets(
                f"expert.{ps.get_expert_model_parallel_rank()}",
                self.expert,
                (
                    0,
                    ps.get_expert_gtp_weight_remat_rank(),
                    ps.get_expert_gtp_weight_remat_world_size(),
                ),
                replica_id=(0, 0, ps.get_expert_data_parallel_rank(with_gtp_remat=False)),
            )
        return result


@pytest.fixture
def independent_instances(request):
    if Utils.world_size != 8:
        pytest.skip("Test requires eight ranks")
    try:
        gtp = getattr(request, "param", False)
        kwargs = (
            dict(context_parallel_size=4, gtp_remat_size=2, expert_gtp_remat_size=2)
            if gtp
            else dict(context_parallel_size=2, expert_model_parallel_size=2, order="tp-cp-dp-ep-pp")
        )
        Utils.initialize_model_parallel(
            **kwargs,
            num_distributed_optimizer_instances=2,
            separate_dense_expert_inter_instance_groups=True,
        )
        yield gtp
    finally:
        Utils.destroy_model_parallel()


def _setup(
    optimizer_type="adam", overlap=False, omit_expert=False, scaled=False, explicit_groups=False
):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        expert_model_parallel_size=ps.get_expert_model_parallel_world_size(),
        num_moe_experts=2,
        context_parallel_size=ps.get_context_parallel_world_size(),
        tensor_parallel_num_weight_shards=ps.get_gtp_weight_remat_world_size(),
        expert_tensor_parallel_num_weight_shards=ps.get_expert_gtp_weight_remat_world_size(),
        params_dtype=torch.bfloat16 if scaled else torch.float32,
        bf16=scaled,
        gradient_accumulation_fusion=False,
    )
    groups = ProcessGroupCollection.use_mpu_process_groups() if explicit_groups else None
    model = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            num_distributed_optimizer_instances=2,
            overlap_grad_reduce=overlap,
            grad_reduce_in_fp32=True,
        ),
        _Model(config, omit_expert),
        pg_collection=groups,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer=optimizer_type,
            use_distributed_optimizer=True,
            params_dtype=config.params_dtype,
            bf16=scaled,
            loss_scale=8.0 if scaled else None,
            lr=0.1,
            weight_decay=0.0,
            clip_grad=1.5,
            log_num_zeros_in_grad=True,
        ),
        [model],
        use_gloo_process_groups=False,
        pg_collection=groups,
    )
    dense, expert = optimizer.chained_optimizers
    gtp = ps.get_gtp_weight_remat_world_size() > 1
    assert dense.distributed_optimizer_instance_id == (
        (Utils.rank % 4) // 2 if gtp else Utils.rank // 4
    )
    assert expert.distributed_optimizer_instance_id == (
        Utils.rank // 4 if gtp else (Utils.rank % 4) // 2
    )
    assert (
        dense.get_grad_stats_parallel_group() is ps.get_intra_distributed_optimizer_instance_group()
    )
    assert (
        expert.get_grad_stats_parallel_group()
        is ps.get_intra_expert_distributed_optimizer_instance_group()
    )
    return model, optimizer


def _gradient(name):
    return torch.arange(_NUMEL, device="cuda", dtype=torch.float32).remainder(
        13 if name == "dense" else 17
    ) - (6 if name == "dense" else 8)


def _prepare(model, optimizer, inf_class=None):
    optimizer.zero_grad()
    model.zero_grad_buffer()
    loss = torch.zeros((), device="cuda")
    for name, parameter in model.module.named_parameters():
        gradient = _gradient(name) * (Utils.rank + 1)
        if name == inf_class and Utils.rank == 0:
            gradient[0] = float("inf")
        loss = loss + (parameter.float() * gradient).sum()
    loss.backward()
    model.finish_grad_sync()


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("omit_expert", [False, True])
@pytest.mark.parametrize("explicit_groups", [False, True])
def test_independent_instance_norm_clipping_and_step(
    independent_instances, overlap, omit_expert, explicit_groups
):
    model, optimizer = _setup(
        overlap=overlap, omit_expert=omit_expert, explicit_groups=explicit_groups
    )
    dense_gradient = _gradient("dense") * 4.5
    expert_gradients = [_gradient("expert") * 1.25]
    if not omit_expert:
        expert_gradients.append(_gradient("expert") * 3.25)
    gradients = [dense_gradient] + expert_gradients
    expected_norm = sum(g.double().square().sum().item() for g in gradients) ** 0.5
    expected_zeros = sum((g == 0).sum().item() for g in gradients)
    _prepare(model, optimizer)
    success, norm, zeros = optimizer.step()
    assert success
    assert norm == pytest.approx(expected_norm, rel=1e-6)
    assert zeros == expected_zeros
    model.start_param_sync(force_sync=True)
    coefficient = 1.5 / (expected_norm + 1e-6)
    reference_params = [torch.nn.Parameter(torch.ones_like(g)) for g in gradients]
    reference = torch.optim.AdamW(reference_params, lr=0.1, weight_decay=0.0)
    for parameter, gradient in zip(reference_params, gradients):
        parameter.grad = gradient * coefficient
    reference.step()
    for name, parameter in model.module.named_parameters():
        expected = (
            reference_params[0]
            if name == "dense"
            else reference_params[1 + ps.get_expert_model_parallel_rank()]
        )
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("inf_class", ["dense", "expert"])
def test_independent_instance_overflow_skips_both_optimizers(independent_instances, inf_class):
    model, optimizer = _setup(scaled=True)
    _prepare(model, optimizer, inf_class)
    success, norm, zeros = optimizer.step()
    assert not success
    assert norm is None and zeros is None
    for parameter in model.parameters():
        torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)


@pytest.mark.parametrize("independent_instances", [True], indirect=True)
def test_gtp_independent_instance_norm(independent_instances):
    model, optimizer = _setup()
    expected = (
        sum(
            (_gradient(name).double() * scale).square().sum().item()
            for name, scale in (("dense", 2.5), ("dense", 6.5), ("expert", 4.0), ("expert", 5.0))
        )
        ** 0.5
    )
    _prepare(model, optimizer)
    assert optimizer.step()[1] == pytest.approx(expected, rel=1e-6)


def _local_states(optimizer):
    return [deepcopy(list(part.optimizer.state.values())) for part in optimizer.chained_optimizers]


def test_custom_collection_requires_expert_statistics(independent_instances):
    model, _ = _setup(explicit_groups=True)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    del groups.intra_expt_dist_opt
    with pytest.raises(ValueError, match="intra_expt_dist_opt must be provided"):
        ProcessGroupCollection.setup_process_groups_for_optimizer(groups, [model], False)


@pytest.fixture
def checkpoint_dir(tmp_path_dist_ckpt, request):
    with TempNamedDir(tmp_path_dist_ckpt / request.node.name, sync=True) as directory:
        yield directory


@pytest.mark.parametrize("sharding_type", ["dp_reshardable", "fully_reshardable"])
@pytest.mark.parametrize("fully_parallel", [False, True])
@pytest.mark.parametrize("independent_instances", [False, True], indirect=True)
def test_independent_instance_checkpoint_resume(
    independent_instances, checkpoint_dir, sharding_type, fully_parallel
):
    model, optimizer = _setup(optimizer_type="adam")
    _prepare(model, optimizer)
    assert optimizer.step()[0]
    model.start_param_sync(force_sync=True)
    metadata = {"distrib_optim_sharding_type": sharding_type}
    checkpoint = checkpoint_dir
    model_state = model.module.sharded_state_dict()
    strategy = TorchDistSaveShardedStrategy()
    if fully_parallel:
        strategy = FullyParallelSaveStrategyWrapper(strategy)
    save(
        {
            "model": model_state,
            "optimizer": optimizer.sharded_state_dict(model_state, metadata=metadata),
        },
        checkpoint,
        sharded_strategy=strategy,
    )
    saved_states = _local_states(optimizer)
    restored_model, restored_optimizer = _setup(optimizer_type="adam")
    restored_model_state = restored_model.module.sharded_state_dict()
    load_strategy = TorchDistLoadShardedStrategy()
    if fully_parallel:
        load_strategy = FullyParallelLoadStrategyWrapper(load_strategy)
    loaded = load(
        {
            "model": restored_model_state,
            "optimizer": restored_optimizer.sharded_state_dict(
                restored_model_state, metadata=metadata, is_loading=True
            ),
        },
        checkpoint,
        sharded_strategy=load_strategy,
    )
    restored_model.module.load_state_dict(loaded["model"])
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restored_model.start_param_sync(force_sync=True)
    for actual, expected in zip(_local_states(restored_optimizer), saved_states):
        assert len(actual) == len(expected)
        for actual_state, expected_state in zip(actual, expected):
            assert actual_state.keys() == expected_state.keys()
            for key in actual_state:
                torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)
    for current_model, current_optimizer in (
        (model, optimizer),
        (restored_model, restored_optimizer),
    ):
        _prepare(current_model, current_optimizer)
        assert current_optimizer.step()[0]
        current_model.start_param_sync(force_sync=True)
    for restored, original in zip(restored_model.parameters(), model.parameters()):
        torch.testing.assert_close(restored, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "inner,outer",
    [
        ("no_shard", "no_shard"),
        ("optim_grads_params", "no_shard"),
        ("optim_grads_params", "optim_grads_params"),
    ],
)
def test_mfsdp_v1_independent_instance_statistics(independent_instances, inner, outer):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        context_parallel_size=2,
        expert_model_parallel_size=2,
        num_moe_experts=2,
    )
    module = _Model(config, omit_expert=False)
    # MFSDP v1 recognizes routed experts by the ".experts." naming convention.
    expert = module.expert
    del module.expert
    module.layer = torch.nn.Module()
    module.layer.experts = torch.nn.Module()
    module.layer.experts.register_parameter("weight", expert)
    model = FullyShardedDataParallel(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=1,
            num_distributed_optimizer_instances=2,
            data_parallel_sharding_strategy=inner,
            outer_dp_sharding_strategy=outer,
            # This toy model has no TransformerLayer from which to infer a unit size.
            suggested_communication_unit_size=2 * _NUMEL,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
        ),
        module=module,
    )
    if outer != "no_shard":
        index = model.megatron_fsdp_dist_index
        for is_expert in (False, True):
            inner_group = index.get_fsdp_group(is_expert)
            outer_group = index.get_outer_fsdp_group(is_expert)
            assert index.get_logical_hybrid_fsdp_rank(is_expert) == (
                inner_group.rank() * outer_group.size() + outer_group.rank()
            )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(use_distributed_optimizer=True, lr=0.01),
        [model],
        use_gloo_process_groups=False,
    )
    dense, expert = optimizer.chained_optimizers
    assert dense.distributed_optimizer_instance_id == Utils.rank // 4
    assert expert.distributed_optimizer_instance_id == (Utils.rank % 4) // 2
    for part, value in (
        (dense, 2.0),
        (expert, 3.0 if ps.get_expert_model_parallel_rank() == 0 else 5.0),
    ):
        for parameter in part.get_parameters():
            parameter.grad = torch.ones_like(parameter) * value
    assert float(optimizer.get_grad_norm()) == pytest.approx(
        (_NUMEL * (4 + 9 + 25)) ** 0.5, rel=1e-6
    )


@pytest.mark.parametrize("independent_instances", [False, True], indirect=True)
def test_mfsdp_v2_hybrid_mesh_axis_order(independent_instances):
    for outer, inner in (
        (
            ps.get_inter_distributed_optimizer_instance_group(),
            ps.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=True),
        ),
        (
            ps.get_inter_expert_distributed_optimizer_instance_group(),
            ps.get_expert_data_parallel_group(partial_expert_data_parallel=True),
        ),
    ):
        mesh = _build_hybrid_dp_mesh(outer, inner, "cuda")
        for name, group in (("dp_outer", outer), ("dp_shard", inner)):
            assert torch.distributed.get_process_group_ranks(mesh.get_group(name)) == (
                torch.distributed.get_process_group_ranks(group)
            )
