#!/usr/bin/env python3
"""
Simple timing benchmark for Wan2.1 multi-GPU WITHOUT profiler overhead.

Usage:
    CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 scripts/benchmark_wan_multi.py
"""

import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# Add Wan2.1 to path
WAN_PATH = Path(__file__).parent.parent.parent / "Wan2.1"
sys.path.insert(0, str(WAN_PATH))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    if rank == 0:
        print(f"=== Wan2.1 Multi-GPU Benchmark ({world_size} GPUs) ===")
        print("NO PROFILER - pure timing measurement")

    # Initialize xFuser for sequence parallelism
    from xfuser.core.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(rank=rank, world_size=world_size)
    initialize_model_parallel(
        sequence_parallel_degree=world_size,
        ring_degree=1,
        ulysses_degree=world_size,
    )

    # Load model with USP enabled
    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    if rank == 0:
        print(f"Loading model from {checkpoint_dir}")

    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=True,
        t5_cpu=True,
    )

    prompt = "Two anthropomorphic cats in boxing gear fight on a spotlighted stage."
    size = "480*832"
    frame_num = 17
    num_steps = 10

    # Warmup
    if rank == 0:
        print("Warmup run...")

    dist.barrier()
    with torch.no_grad():
        _ = wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS[size],
            frame_num=frame_num,
            sampling_steps=num_steps,
            seed=42,
            offload_model=False
        )
    torch.cuda.synchronize()
    dist.barrier()

    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark runs
    num_runs = 3
    times = []

    if rank == 0:
        print(f"\nRunning {num_runs} benchmark iterations...")

    for i in range(num_runs):
        dist.barrier()
        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            video = wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[size],
                frame_num=frame_num,
                sampling_steps=num_steps,
                seed=42 + i,
                offload_model=False
            )

        torch.cuda.synchronize()
        dist.barrier()

        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)

        if rank == 0:
            print(f"  Run {i+1}: {elapsed_ms:.2f} ms ({elapsed_ms/num_steps:.2f} ms/step)")

    if rank == 0:
        mean_time = sum(times) / len(times)
        print(f"\n{'='*50}")
        print(f"RESULTS ({world_size} GPUs with USP):")
        print(f"  Mean inference time: {mean_time:.2f} ms")
        print(f"  Time per step: {mean_time/num_steps:.2f} ms")
        print(f"  Single GPU baseline: ~7534 ms")
        print(f"  Expected speedup: {7534/mean_time:.2f}x")
        print(f"{'='*50}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
