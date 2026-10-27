<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Training across two sites

Megatron-LM can run one synchronous training job across two sites when model-parallel replicas are site-local and, if enabled, distributed-optimizer instances are site-local. The sites use separate Slurm allocations, but every GPU worker joins the **same** `torch.distributed` world. The distributed optimizer reduces gradients between its instances across sites; `--site-id` supplies site locality to Megatron and validates the placement. It does not launch workers, change their global ranks, or confine every collective to a site.

This guide uses a fixed-size [static `torchrun` rendezvous](https://docs.pytorch.org/docs/stable/elastic/run.html). Each site has one four-GPU node, for eight workers in total. Adapt the node counts, GPU counts, and model arguments to your environment.

## Prerequisites

- Both sites have compatible GPU software environments and the same Megatron-LM revision. Code, data, and checkpoint files need to be accessible to workers **within** each site; a filesystem shared between sites is not required.
- Compute nodes have bidirectional, routable TCP/IP connectivity on the interfaces used for training. The rendezvous hostname or IP address must resolve and be reachable from both sites. Firewall rules must allow the chosen rendezvous port and the peer TCP ports opened by NCCL. A login-node connection alone is insufficient. Use a trusted network or an encrypted overlay when required by your network policy. See [NCCL networking guidance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html).
- The two Slurm allocations overlap in time. A coordinator (which can initially be a person using two shells) reserves both allocations, discovers the rendezvous endpoint, assigns global node ranks, and starts both launchers. Independent `torchrun --standalone` jobs cannot be joined after they start.

## Rank and optimizer placement

For this example, assign node rank 0 and site ID 0 to the first site, then node rank 1 and site ID 1 to the second:

| Site | Node rank | Global worker ranks | `--site-id` | Optimizer instance |
| --- | ---: | --- | ---: | ---: |
| First | 0 | 0–3 | 0 | 0 |
| Second | 1 | 4–7 | 1 | 1 |

With TP=PP=CP=EP=1, the data-parallel size is eight. `--num-distributed-optimizer-instances 2` makes each four-rank instance site-local. Megatron requires each site to occupy one contiguous global-rank block containing complete model-parallel replicas. When using a distributed optimizer, each site must also contain complete optimizer instances. The site IDs need only be distinct nonnegative integers; they do not determine rank order. `--site-id` can also be used without checkpoints or a distributed optimizer; the model-replica placement rule still applies.

With more than one node per site, assign consecutive node ranks to each site. For example, if the first site has `A` nodes and the second has `B`, use node ranks `0..A-1` and `A..A+B-1`, with `--nnodes=$((A+B))` on **every** node. Keep the same `--site-id` on all nodes of a site. Choose TP, PP, CP, EP, rank ordering, and the number of optimizer instances so the complete-replica and complete-instance rules still hold. Megatron checks the placement during initialization.

## Allocate and launch

Reserve one node on each cluster using its own Slurm controller. For example, run this in a separate shell on each cluster, adding your account, partition, time limit, and container options as needed:

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --time=02:00:00
```

On the first cluster, obtain the allocated compute-node name with `scontrol show hostnames "$SLURM_JOB_NODELIST"`. Use a DNS name or IP address for that compute node that is reachable from **both** sites as `MASTER_ADDR`. Choose a free TCP port for `MASTER_PORT`. Set the site and node ranks separately:

```bash
# Run in the first allocation.
export SITE_ID=0 NODE_RANK=0
```

```bash
# Run in the second allocation.
export SITE_ID=1 NODE_RANK=1
```

Export identical rendezvous values on both allocations. Replace the example address with the first site's reachable compute-node address:

```bash
export MASTER_ADDR=compute-0.example.net
export MASTER_PORT=29500
export NCCL_NET=Socket
```

If NCCL selects an interface that is not reachable across sites, set `NCCL_SOCKET_IFNAME` to the appropriate interface name on **each** site. The names may differ. If Gloo process groups use an unreachable interface, set `GLOO_SOCKET_IFNAME` as well. `NCCL_NET=Socket` selects NCCL's TCP/IP network transport; see the [NCCL environment-variable reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) and [PyTorch's interface selection guidance](https://docs.pytorch.org/docs/stable/distributed.html). Container deployments must expose the selected network interface and the site's code and data paths to the workers.

From the Megatron-LM checkout on each site, start the following command in both allocations. Start them close together: each launcher waits for the other before training begins. `srun` starts one task per node; `torch.distributed.run` starts four GPU workers inside that task.

```bash
srun --ntasks=1 --ntasks-per-node=1 \
  uv run python -m torch.distributed.run \
    --nnodes=2 --nproc-per-node=4 \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    pretrain_gpt.py \
      --site-id "$SITE_ID" \
      --use-distributed-optimizer \
      --num-distributed-optimizer-instances 2 \
      --tensor-model-parallel-size 1 \
      --pipeline-model-parallel-size 1 \
      --context-parallel-size 1 \
      --num-layers 2 --hidden-size 128 --num-attention-heads 4 \
      --seq-length 128 --max-position-embeddings 128 \
      --micro-batch-size 1 --global-batch-size 8 \
      --train-iters 10 --lr 1e-4 --min-lr 1e-5 \
      --lr-decay-style constant --bf16 \
      --mock-data --tokenizer-type NullTokenizer --vocab-size 1024 \
      --log-interval 1
```

Use the same model and training configuration on both sites. For real training, replace the mock-data options with your usual data and tokenizer arguments, and use [training examples](training-examples.md) for the remaining configuration. Dataset paths may be site-local, but each worker must receive its intended portion of the global batch.

## Site-local checkpoints

To save one complete checkpoint on each site's filesystem, use synchronous `torch_dist` checkpoints and pass the same `--site-id` values used at launch. For example, append these arguments on **both** sites:

```bash
--ckpt-format torch_dist \
--save /checkpoints/run-1 \
--load /checkpoints/run-1 \
--save-interval 1000
```

The path must resolve to storage shared by workers at a site; it may resolve to a different physical filesystem at the other site. Do not point both sites at the same physical checkpoint directory. The site-local checkpoint groups are used for synchronous `torch_dist` save and load when the caller has not explicitly supplied a checkpoint group. Asynchronous saves with a site-local checkpoint group are currently unsupported. Coordinate retention and restart so that both sites resume from the same training iteration; Megatron does not copy checkpoint files between sites.

## Diagnosing a stalled start

Check that both allocations launched the same total node count, GPU count per node, rendezvous address and port, Megatron configuration, and code revision. Verify that node ranks are unique and that every worker received its site's ID. Then verify compute-node address resolution and TCP reachability, including the rendezvous port and NCCL peer ports. A Python exception on one rank can leave other ranks waiting in a collective, so inspect logs from both sites and find the earliest exception. A failed allocation or worker stops the synchronous job; restart both sites together from coordinated checkpoints.
