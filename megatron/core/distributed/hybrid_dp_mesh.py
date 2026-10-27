# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Device mesh for an outer replicated and inner sharded data-parallel domain."""

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh


def build_hybrid_dp_mesh(outer_group, inner_group, device_type="cuda"):
    """Build a mesh whose rows are intra-instance groups and columns are replicas."""
    if outer_group is None or inner_group is None:
        raise ValueError("Hybrid sharding requires inter- and intra-instance process groups")

    rank = dist.get_rank()
    inner_ranks = dist.get_process_group_ranks(inner_group)
    outer_ranks = dist.get_process_group_ranks(outer_group)
    outer_offsets = torch.tensor(outer_ranks) - rank
    inner_offsets = torch.tensor(inner_ranks) - rank
    layout = rank + outer_offsets[:, None] + inner_offsets[None, :]

    # DeviceMesh.from_group expects the process groups in mesh dimension order.
    expected_layout = layout.flatten().sort().values.reshape(layout.shape)
    expected_inner = expected_layout[outer_group.rank()].tolist()
    expected_outer = expected_layout[:, inner_group.rank()].tolist()
    if inner_ranks != expected_inner:
        raise ValueError(
            f"Hybrid mesh row {expected_inner} does not match the intra-instance "
            f"data-parallel group {inner_ranks}"
        )
    if outer_ranks != expected_outer:
        raise ValueError(
            f"Hybrid mesh column {expected_outer} does not match the inter-instance "
            f"data-parallel group {outer_ranks}"
        )

    return DeviceMesh.from_group(
        [outer_group, inner_group],
        device_type=device_type,
        mesh=layout,
        mesh_dim_names=("dp_outer", "dp_shard"),
    )
