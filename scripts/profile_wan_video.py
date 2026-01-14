#!/usr/bin/env python3
"""
Profiling script for Wan2.1 Text-to-Video generation on single and multi-GPU.

This script profiles:
1. Single GPU baseline - compute only
2. Multi-GPU with USP (Ulysses Sequence Parallelism) - communication + compute

Usage:
    # Single GPU profiling
    CUDA_VISIBLE_DEVICES=5 python scripts/profile_wan_video.py --mode single --gpu_id 0

    # Multi-GPU profiling (3 GPUs)
    CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 scripts/profile_wan_video.py --mode multi --ulysses_size 3
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

# Add Wan2.1 to path
WAN_PATH = Path(__file__).parent.parent.parent / "Wan2.1"
sys.path.insert(0, str(WAN_PATH))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS


def setup_logging(rank=0):
    """Setup logging configuration."""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def get_profiler_schedule(warmup=2, active=3, repeat=1):
    """Create profiler schedule."""
    return schedule(
        wait=1,
        warmup=warmup,
        active=active,
        repeat=repeat
    )


def categorize_cuda_events(events):
    """Categorize CUDA events into compute categories."""
    categories = {
        'linear_ffn': 0,      # Linear layers, GEMM
        'attention': 0,        # Attention computations
        'memory_ops': 0,       # Memory operations
        'conv': 0,             # Convolutions (VAE)
        'activation': 0,       # Activation functions
        'normalization': 0,    # Norm layers
        'communication': 0,    # NCCL operations
        'other': 0,
    }

    for evt in events:
        if evt.device_type == 1:  # CUDA
            name = evt.name.lower()
            cuda_time = evt.cuda_time_total / 1000  # to ms

            if 'nccl' in name or 'all_gather' in name or 'all_reduce' in name or 'broadcast' in name:
                categories['communication'] += cuda_time
            elif any(k in name for k in ['gemm', 'mm_', 'linear', 'addmm', 'bmm']):
                categories['linear_ffn'] += cuda_time
            elif any(k in name for k in ['attention', 'flash', 'sdpa', 'sdp']):
                categories['attention'] += cuda_time
            elif any(k in name for k in ['conv', 'cudnn']):
                categories['conv'] += cuda_time
            elif any(k in name for k in ['norm', 'layer_norm', 'rms']):
                categories['normalization'] += cuda_time
            elif any(k in name for k in ['gelu', 'silu', 'relu', 'softmax']):
                categories['activation'] += cuda_time
            elif any(k in name for k in ['copy', 'memcpy', 'memset', 'cat', 'split', 'chunk']):
                categories['memory_ops'] += cuda_time
            else:
                categories['other'] += cuda_time

    return categories


def profile_single_gpu(args):
    """Profile Wan2.1 on single GPU."""
    setup_logging()

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    logging.info(f"=== Single GPU Profiling on GPU {args.gpu_id} ===")
    logging.info(f"Model: Wan2.1-T2V-1.3B")
    logging.info(f"Video size: {args.size}")
    logging.info(f"Frame count: {args.frame_num}")
    logging.info(f"Inference steps: {args.num_steps}")

    # Load model
    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    logging.info(f"Loading model from {checkpoint_dir}")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=args.gpu_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,  # Keep T5 on CPU to save GPU memory
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"wan_single_gpu/{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = "Two anthropomorphic cats in boxing gear fight on a spotlighted stage."

    # Warmup runs
    logging.info("Running warmup...")
    for _ in range(args.warmup_runs):
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

    # Profile runs
    logging.info("Running profiled inference...")

    inference_times = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=get_profiler_schedule(warmup=1, active=args.profile_runs, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(output_dir / "tensorboard")),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for i in range(args.profile_runs + 2):  # +2 for wait and warmup
            start_time = time.perf_counter()

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
            end_time = time.perf_counter()

            if i >= 2:  # Skip wait and warmup
                inference_times.append((end_time - start_time) * 1000)
                logging.info(f"Run {i-1}: {inference_times[-1]:.2f} ms")

            prof.step()

    # Analyze results
    mean_time = sum(inference_times) / len(inference_times) if inference_times else 0
    time_per_step = mean_time / args.num_steps

    # Get CUDA event breakdown
    events = prof.key_averages()
    categories = categorize_cuda_events(events)
    total_cuda_time = sum(categories.values())

    results = {
        "mode": "single_gpu",
        "gpu_id": args.gpu_id,
        "model": "Wan2.1-T2V-1.3B",
        "video_size": args.size,
        "frame_num": args.frame_num,
        "num_steps": args.num_steps,
        "mean_inference_time_ms": mean_time,
        "time_per_step_ms": time_per_step,
        "inference_times_ms": inference_times,
        "cuda_time_breakdown": categories,
        "cuda_time_percentages": {k: v/total_cuda_time*100 if total_cuda_time > 0 else 0
                                   for k, v in categories.items()},
    }

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logging.info("\n" + "="*60)
    logging.info("PROFILING RESULTS - SINGLE GPU")
    logging.info("="*60)
    logging.info(f"Mean inference time: {mean_time:.2f} ms")
    logging.info(f"Time per step: {time_per_step:.2f} ms")
    logging.info(f"\nCUDA Time Breakdown:")
    for cat, time_ms in sorted(categories.items(), key=lambda x: -x[1]):
        pct = time_ms / total_cuda_time * 100 if total_cuda_time > 0 else 0
        logging.info(f"  {cat:20s}: {time_ms:10.2f} ms ({pct:5.1f}%)")
    logging.info(f"\nResults saved to: {output_dir}")

    return results


def profile_multi_gpu(args):
    """Profile Wan2.1 on multiple GPUs with USP (sequence parallelism)."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    setup_logging(rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    if rank == 0:
        logging.info(f"=== Multi-GPU Profiling with {world_size} GPUs ===")
        logging.info(f"Model: Wan2.1-T2V-1.3B")
        logging.info(f"Video size: {args.size}")
        logging.info(f"Frame count: {args.frame_num}")
        logging.info(f"Inference steps: {args.num_steps}")
        logging.info(f"Ulysses size: {args.ulysses_size}")
        logging.info(f"Ring size: {args.ring_size}")

    # Initialize xFuser for sequence parallelism
    from xfuser.core.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(rank=rank, world_size=world_size)
    initialize_model_parallel(
        sequence_parallel_degree=world_size,
        ring_degree=args.ring_size,
        ulysses_degree=args.ulysses_size,
    )

    # Load model with USP enabled
    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    if rank == 0:
        logging.info(f"Loading model from {checkpoint_dir}")

    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=True,  # Enable USP
        t5_cpu=True,
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"wan_multi_gpu/{timestamp}_gpus{world_size}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    prompt = "Two anthropomorphic cats in boxing gear fight on a spotlighted stage."

    # Broadcast seed
    seed = torch.tensor([42], device=f"cuda:{local_rank}")
    dist.broadcast(seed, src=0)
    seed = seed.item()

    # Warmup runs
    if rank == 0:
        logging.info("Running warmup...")

    for i in range(args.warmup_runs):
        with torch.no_grad():
            _ = wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                sampling_steps=args.num_steps,
                seed=seed + i,
                offload_model=False
            )
        dist.barrier()

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Profile runs
    if rank == 0:
        logging.info("Running profiled inference...")

    inference_times = []

    profile_dir = output_dir / f"tensorboard_rank{rank}"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=get_profiler_schedule(warmup=1, active=args.profile_runs, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(profile_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for i in range(args.profile_runs + 2):
            dist.barrier()
            start_time = time.perf_counter()

            with torch.no_grad():
                video = wan_t2v.generate(
                    prompt,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    sampling_steps=args.num_steps,
                    seed=seed + i,
                    offload_model=False
                )

            torch.cuda.synchronize()
            dist.barrier()
            end_time = time.perf_counter()

            if i >= 2:
                inference_times.append((end_time - start_time) * 1000)
                if rank == 0:
                    logging.info(f"Run {i-1}: {inference_times[-1]:.2f} ms")

            prof.step()

    # Analyze results per rank
    events = prof.key_averages()
    categories = categorize_cuda_events(events)
    total_cuda_time = sum(categories.values())

    # Gather results from all ranks
    mean_time = sum(inference_times) / len(inference_times) if inference_times else 0
    time_per_step = mean_time / args.num_steps

    # Communication time analysis
    comm_time = categories['communication']
    compute_time = total_cuda_time - comm_time

    results = {
        "mode": "multi_gpu",
        "world_size": world_size,
        "rank": rank,
        "ulysses_size": args.ulysses_size,
        "ring_size": args.ring_size,
        "model": "Wan2.1-T2V-1.3B",
        "video_size": args.size,
        "frame_num": args.frame_num,
        "num_steps": args.num_steps,
        "mean_inference_time_ms": mean_time,
        "time_per_step_ms": time_per_step,
        "inference_times_ms": inference_times,
        "cuda_time_breakdown": categories,
        "cuda_time_percentages": {k: v/total_cuda_time*100 if total_cuda_time > 0 else 0
                                   for k, v in categories.items()},
        "communication_time_ms": comm_time,
        "compute_time_ms": compute_time,
        "comm_compute_ratio": comm_time / compute_time if compute_time > 0 else 0,
        "communication_percentage": comm_time / total_cuda_time * 100 if total_cuda_time > 0 else 0,
    }

    # Save results per rank
    results_file = output_dir / f"results_rank{rank}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    dist.barrier()

    # Print summary from rank 0
    if rank == 0:
        logging.info("\n" + "="*60)
        logging.info(f"PROFILING RESULTS - {world_size} GPUs with USP")
        logging.info("="*60)
        logging.info(f"Mean inference time: {mean_time:.2f} ms")
        logging.info(f"Time per step: {time_per_step:.2f} ms")
        logging.info(f"\nCommunication Analysis:")
        logging.info(f"  Communication time: {comm_time:.2f} ms")
        logging.info(f"  Compute time: {compute_time:.2f} ms")
        logging.info(f"  Comm/Compute ratio: {results['comm_compute_ratio']:.3f}")
        logging.info(f"  Communication %: {results['communication_percentage']:.1f}%")
        logging.info(f"\nCUDA Time Breakdown (Rank 0):")
        for cat, time_ms in sorted(categories.items(), key=lambda x: -x[1]):
            pct = time_ms / total_cuda_time * 100 if total_cuda_time > 0 else 0
            logging.info(f"  {cat:20s}: {time_ms:10.2f} ms ({pct:5.1f}%)")

        # Calculate potential speedup with DistriFusion-style overlap
        if comm_time > 0 and compute_time > 0:
            potential_speedup = (comm_time + compute_time) / compute_time
            logging.info(f"\n=== DistriFusion Overlap Potential ===")
            logging.info(f"With perfect async overlap: {potential_speedup:.2f}x speedup possible")
            logging.info(f"This would reduce inference time from {mean_time:.2f}ms to ~{mean_time/potential_speedup:.2f}ms")

        logging.info(f"\nResults saved to: {output_dir}")

    dist.barrier()
    dist.destroy_process_group()

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile Wan2.1 video generation")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="single",
                       help="Profiling mode: single GPU or multi GPU")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID for single GPU mode")
    parser.add_argument("--size", type=str, default="480*832",
                       help="Video size (width*height)")
    parser.add_argument("--frame_num", type=int, default=17,
                       help="Number of frames to generate (4n+1)")
    parser.add_argument("--num_steps", type=int, default=10,
                       help="Number of denoising steps")
    parser.add_argument("--warmup_runs", type=int, default=1,
                       help="Number of warmup runs")
    parser.add_argument("--profile_runs", type=int, default=2,
                       help="Number of profiled runs")
    parser.add_argument("--ulysses_size", type=int, default=1,
                       help="Ulysses parallelism size for multi-GPU")
    parser.add_argument("--ring_size", type=int, default=1,
                       help="Ring attention parallelism size for multi-GPU")
    parser.add_argument("--output_dir", type=str,
                       default=str(Path(__file__).parent.parent.parent / "profiling_results"),
                       help="Output directory for profiling results")

    args = parser.parse_args()

    if args.mode == "single":
        profile_single_gpu(args)
    else:
        profile_multi_gpu(args)


if __name__ == "__main__":
    main()
