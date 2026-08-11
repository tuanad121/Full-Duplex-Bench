#!/usr/bin/env python3
"""Small CUDA allocation and compute smoke test for every visible GPU."""

from __future__ import annotations

import json

import torch


results = []
for index in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{index}")
    free, total = torch.cuda.mem_get_info(index)
    left = torch.ones((2048, 2048), device=device)
    right = torch.ones((2048, 1), device=device)
    result = float((left @ right).mean().item())
    results.append(
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "free_GiB": round(free / 2**30, 2),
            "total_GiB": round(total / 2**30, 2),
            "matmul_result": result,
            "passed": result == 2048.0,
        }
    )

print(
    json.dumps(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "devices": results,
        },
        indent=2,
    )
)
