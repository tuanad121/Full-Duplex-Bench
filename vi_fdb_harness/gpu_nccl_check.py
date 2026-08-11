#!/usr/bin/env python3
"""Two-GPU NCCL all-reduce smoke test."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
dist.all_reduce(value, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
print(
    json.dumps(
        {
            "rank": rank,
            "local_rank": local_rank,
            "all_reduce_sum": value.item(),
            "passed": value.item() == 3.0,
        }
    ),
    flush=True,
)
dist.destroy_process_group()
