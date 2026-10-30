# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Argument validation for hybrid (dynamic) context parallelism.

A captured graph bakes in static communicators, so it cannot follow the per-microbatch
context-parallel group that hybrid context parallelism selects at runtime.
Site placement is unsupported until the runtime domain can be restricted to a site.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.training.arguments import parse_args, validate_args
from megatron.training.yaml_arguments import validate_yaml

_GUARD = "Hybrid context parallelism not supported with CUDA Graph"


def _hybrid_cp_args(monkeypatch, **overrides):
    monkeypatch.setattr(sys, "argv", ["test_hybrid_context_parallel_args.py"])
    args = parse_args()
    # parse_args reads WORLD_SIZE from the environment; pin it so the test does not depend
    # on how the job was launched.
    args.world_size = 1
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 1024
    args.hybrid_context_parallel = True
    args.max_seqlen_per_dp_cp_rank = 1024
    args.calculate_per_token_loss = True
    args.dataloader_type = "single"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _site_validation_args(monkeypatch, use_yaml, **overrides):
    args = _hybrid_cp_args(
        monkeypatch,
        **{"hybrid_context_parallel": False, "dynamic_context_parallel": False, **overrides},
    )
    if not use_yaml:
        return args, validate_args

    parallel_fields = (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "tp_comm_overlap",
        "sequence_parallel",
        "fp16",
        "bf16",
        "overlap_p2p_comm",
        "dynamic_context_parallel",
        "hybrid_context_parallel",
    )
    language_fields = (
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "ffn_hidden_size",
        "kv_channels",
        "fp32_residual_connection",
        "moe_grouped_gemm",
        "distribute_saved_activations",
        "recompute_granularity",
        "recompute_method",
    )
    args.model_parallel = SimpleNamespace(
        **{key: vars(args).pop(key) for key in parallel_fields},
        params_dtype=vars(args).pop("params_dtype", torch.float32),
    )
    args.language_model = SimpleNamespace(
        **{key: vars(args).pop(key) for key in language_fields},
        activation_func="gelu",
        num_moe_experts=None,
        persist_layer_norm=True,
    )
    return args, validate_yaml


@pytest.mark.parametrize(
    "cuda_graph_impl, prerequisites",
    [
        ("local", {}),
        ("transformer_engine", {}),
        # full_iteration has its own prerequisite, checked before the hybrid-CP guard.
        ("full_iteration", {"check_for_nan_in_loss_and_grad": False}),
    ],
)
def test_hybrid_context_parallel_rejects_cuda_graph_impl(
    monkeypatch, cuda_graph_impl, prerequisites
):
    args = _hybrid_cp_args(monkeypatch, cuda_graph_impl=cuda_graph_impl, **prerequisites)
    with pytest.raises(AssertionError, match=_GUARD):
        validate_args(args)


@pytest.mark.parametrize("deprecated_flag", ["enable_cuda_graph", "external_cuda_graph"])
def test_hybrid_context_parallel_rejects_deprecated_cuda_graph_flags(monkeypatch, deprecated_flag):
    # validate_args translates these into cuda_graph_impl and deletes them before the
    # hybrid-CP checks run, so the guard must read cuda_graph_impl.
    args = _hybrid_cp_args(monkeypatch, **{deprecated_flag: True})
    with pytest.raises(AssertionError, match=_GUARD):
        validate_args(args)


def test_hybrid_context_parallel_without_cuda_graphs_validates(monkeypatch):
    args = _hybrid_cp_args(monkeypatch)
    validate_args(args)
    assert args.cuda_graph_impl == "none"


@pytest.mark.parametrize("use_yaml", [False, True])
@pytest.mark.parametrize("mode", ["hybrid_context_parallel", "dynamic_context_parallel"])
@pytest.mark.parametrize("site_id", [0, 7])
def test_site_placement_rejects_dynamic_context_parallel_before_grid_setup(
    monkeypatch, mode, site_id, use_yaml
):
    args, validator = _site_validation_args(monkeypatch, use_yaml, site_id=site_id, **{mode: True})
    with pytest.raises(ValueError, match="requires site-restricted domains"):
        validator(args)


@pytest.mark.parametrize("use_yaml", [False, True])
def test_site_placement_with_static_context_parallel_validates(monkeypatch, use_yaml):
    args, validator = _site_validation_args(monkeypatch, use_yaml, site_id=0)
    validated = validator(args)
    assert validated.site_id == 0


@pytest.mark.parametrize("use_yaml", [False, True])
def test_site_placement_from_defaults_rejects_dynamic_context_parallel(monkeypatch, use_yaml):
    args, validator = _site_validation_args(monkeypatch, use_yaml, dynamic_context_parallel=True)
    with pytest.raises(ValueError, match="requires site-restricted domains"):
        validator(args, defaults={"site_id": 0})


@pytest.mark.parametrize("use_yaml", [False, True])
@pytest.mark.parametrize("site_id", [-1, True, "0", 0.5])
def test_site_placement_rejects_invalid_ids(monkeypatch, use_yaml, site_id):
    args, validator = _site_validation_args(monkeypatch, use_yaml, site_id=site_id)
    with pytest.raises(ValueError, match="site-id must be a nonnegative integer"):
        validator(args)


@pytest.mark.parametrize("mode", ["hybrid_context_parallel", "dynamic_context_parallel"])
def test_yaml_site_placement_rejects_top_level_dynamic_cp(monkeypatch, mode):
    args, validator = _site_validation_args(monkeypatch, True, site_id=0)
    setattr(args, mode, True)
    with pytest.raises(ValueError, match="requires site-restricted domains"):
        validator(args)
