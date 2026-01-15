#!/usr/bin/env python3
"""
Simple timing benchmark for Wan2.1 single GPU WITHOUT profiler overhead.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/benchmark_wan_single.py
"""

import gc
import sys
import time
from pathlib import Path

import torch

# Add Wan2.1 to path
WAN_PATH = Path(__file__).parent.parent.parent / "Wan2.1"
sys.path.insert(0, str(WAN_PATH))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"=== Wan2.1 Single GPU Benchmark ===")
    print("NO PROFILER - pure timing measurement")

    # Load model
    cfg = WAN_CONFIGS['t2v-1.3B']
    checkpoint_dir = str(WAN_PATH.parent / "Wan2.1-T2V-1.3B")

    print(f"Loading model from {checkpoint_dir}")

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
    size = "480*832"
    frame_num = 17
    num_steps = 10

    # Warmup
    print("Warmup run...")

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

    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark runs
    num_runs = 3
    times = []

    print(f"\nRunning {num_runs} benchmark iterations...")

    for i in range(num_runs):
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

        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)

        print(f"  Run {i+1}: {elapsed_ms:.2f} ms ({elapsed_ms/num_steps:.2f} ms/step)")

    mean_time = sum(times) / len(times)
    print(f"\n{'='*50}")
    print(f"RESULTS (Single GPU):")
    print(f"  Mean inference time: {mean_time:.2f} ms")
    print(f"  Time per step: {mean_time/num_steps:.2f} ms")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
