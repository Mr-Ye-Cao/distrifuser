#!/usr/bin/env python3
"""
Benchmark Wan2.1 with DistriFusion tensor parallelism.

Usage:
    # Single GPU baseline
    CUDA_VISIBLE_DEVICES=5 python scripts/benchmark_wan_distrifusion.py --mode single

    # Multi-GPU with DistriFusion TP
    CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 scripts/benchmark_wan_distrifusion.py --mode multi
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# Add paths
SCRIPT_DIR = Path(__file__).parent
DISTRIFUSER_DIR = SCRIPT_DIR.parent
WAN_PATH = DISTRIFUSER_DIR.parent / "Wan2.1"

sys.path.insert(0, str(DISTRIFUSER_DIR))
sys.path.insert(0, str(WAN_PATH))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS


def run_single_gpu(args):
    """Run single GPU baseline without DistriFusion."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"=== Wan2.1 Single GPU Baseline ===")
    print(f"Frames: {args.frame_num}, Size: {args.size}, Steps: {args.num_steps}")
    print("NO DistriFusion - baseline timing\n")

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    print(f"Loading model from {checkpoint_dir}...")
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
    print(f"\n{'='*60}")
    print(f"SINGLE GPU BASELINE ({args.frame_num} frames):")
    print(f"  Mean inference time: {mean_time:.2f} ms")
    print(f"  Time per step: {mean_time/args.num_steps:.2f} ms")
    print(f"{'='*60}")

    return mean_time


def run_multi_gpu_distrifusion(args):
    """Run multi-GPU with DistriFusion tensor parallelism."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"=== Wan2.1 DistriFusion TP ({world_size} GPUs) ===")
        print(f"Frames: {args.frame_num}, Size: {args.size}, Steps: {args.num_steps}")
        print("Using DistriFusion tensor parallelism\n")

    # Import DistriFusion
    from distrifuser.utils_wan import DistriWanConfig
    from distrifuser.pipelines_wan.wan_pipeline import wrap_wan_model

    # Initialize DistriWanConfig (this also initializes the process group)
    distri_config = DistriWanConfig(
        height=SIZE_CONFIGS[args.size][1],
        width=SIZE_CONFIGS[args.size][0],
        frame_num=args.frame_num,
        verbose=(rank == 0),
    )

    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    if rank == 0:
        print(f"Loading model from {checkpoint_dir}...")

    # Load model on this rank
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )

    # Wrap model with DistriFusion TP
    if rank == 0:
        print("Wrapping model with DistriFusion TP...")
    distri_dit = wrap_wan_model(wan_t2v, distri_config)

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

        # Reset counter for DistriFusion
        distri_dit.set_counter(0)

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
        print(f"\n{'='*60}")
        print(f"DISTRIFUSION TP RESULTS ({world_size} GPUs, {args.frame_num} frames):")
        print(f"  Mean inference time: {mean_time:.2f} ms")
        print(f"  Time per step: {mean_time/args.num_steps:.2f} ms")
        print(f"{'='*60}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="single: baseline, multi: DistriFusion TP")
    parser.add_argument("--frame_num", type=int, default=17, help="Number of frames (4n+1)")
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument("--num_steps", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "single":
        run_single_gpu(args)
    else:
        run_multi_gpu_distrifusion(args)


if __name__ == "__main__":
    main()
