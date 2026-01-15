#!/usr/bin/env python3
"""
Benchmark Wan2.1 with longer sequences (81 frames) to test if USP helps.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=5 python scripts/benchmark_wan_long.py --mode single

    # Multi-GPU USP
    CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 scripts/benchmark_wan_long.py --mode multi
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

WAN_PATH = Path(__file__).parent.parent.parent / "Wan2.1"
sys.path.insert(0, str(WAN_PATH))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS


def run_single_gpu(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"=== Wan2.1 Single GPU - Long Sequence ===")
    print(f"Frames: {args.frame_num}, Size: {args.size}, Steps: {args.num_steps}")
    print("NO PROFILER - pure timing measurement\n")

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    print(f"Loading model...")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )

    prompt = "Two anthropomorphic cats in boxing gear fight on a spotlighted stage."

    # Warmup
    print("Warmup run...")
    with torch.no_grad():
        _ = wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            sampling_steps=args.num_steps,
            seed=42,
            offload_model=False
        )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark
    num_runs = 2
    times = []
    print(f"\nRunning {num_runs} benchmark iterations...")

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            video = wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                sampling_steps=args.num_steps,
                seed=42 + i,
                offload_model=False
            )

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        print(f"  Run {i+1}: {elapsed_ms:.2f} ms ({elapsed_ms/args.num_steps:.2f} ms/step)")

    mean_time = sum(times) / len(times)
    print(f"\n{'='*50}")
    print(f"SINGLE GPU RESULTS ({args.frame_num} frames):")
    print(f"  Mean inference time: {mean_time:.2f} ms")
    print(f"  Time per step: {mean_time/args.num_steps:.2f} ms")
    print(f"{'='*50}")

    return mean_time


def run_multi_gpu(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    if rank == 0:
        print(f"=== Wan2.1 Multi-GPU USP - Long Sequence ({world_size} GPUs) ===")
        print(f"Frames: {args.frame_num}, Size: {args.size}, Steps: {args.num_steps}")
        print("NO PROFILER - pure timing measurement\n")

    # Initialize xFuser
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

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    if rank == 0:
        print(f"Loading model...")

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

    # Warmup
    if rank == 0:
        print("Warmup run...")

    dist.barrier()
    with torch.no_grad():
        _ = wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            sampling_steps=args.num_steps,
            seed=42,
            offload_model=False
        )
    torch.cuda.synchronize()
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark
    num_runs = 2
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
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                sampling_steps=args.num_steps,
                seed=42 + i,
                offload_model=False
            )

        torch.cuda.synchronize()
        dist.barrier()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        if rank == 0:
            print(f"  Run {i+1}: {elapsed_ms:.2f} ms ({elapsed_ms/args.num_steps:.2f} ms/step)")

    if rank == 0:
        mean_time = sum(times) / len(times)
        print(f"\n{'='*50}")
        print(f"MULTI-GPU USP RESULTS ({world_size} GPUs, {args.frame_num} frames):")
        print(f"  Mean inference time: {mean_time:.2f} ms")
        print(f"  Time per step: {mean_time/args.num_steps:.2f} ms")
        print(f"{'='*50}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--frame_num", type=int, default=81, help="Number of frames (4n+1)")
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument("--num_steps", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "single":
        run_single_gpu(args)
    else:
        run_multi_gpu(args)


if __name__ == "__main__":
    main()
