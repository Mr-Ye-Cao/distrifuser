#!/usr/bin/env python3
"""
Profile NCCL communication patterns for Wan2.1 DiT model.

This script measures:
1. All-gather latency for different tensor sizes (simulating sequence parallelism)
2. All-reduce latency (simulating tensor parallelism)
3. Communication bandwidth between GPUs

Usage:
    CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 scripts/profile_wan_nccl.py
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


def setup_logging(rank=0):
    """Setup logging configuration."""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def benchmark_all_gather(tensor, world_size, num_iters=100, warmup=10):
    """Benchmark all-gather operation."""
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Warmup
    for _ in range(warmup):
        dist.all_gather(gather_list, tensor)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        dist.all_gather(gather_list, tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iters * 1000  # ms


def benchmark_all_reduce(tensor, num_iters=100, warmup=10):
    """Benchmark all-reduce operation."""
    # Warmup
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        dist.all_reduce(tensor.clone())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iters * 1000  # ms


def simulate_dit_forward(model_config, device, world_size, rank):
    """Simulate DiT forward pass communication patterns."""

    # Wan2.1-T2V-1.3B configuration
    dim = model_config.get('dim', 1536)
    num_heads = model_config.get('num_heads', 12)
    num_layers = model_config.get('num_layers', 30)
    seq_len = model_config.get('seq_len', 5120)  # Typical for 480x832, 17 frames
    batch_size = 1

    # In sequence parallelism (USP), sequence is split across GPUs
    local_seq_len = seq_len // world_size

    results = {}

    # 1. Simulate attention KV all-gather (for sequence parallelism)
    # Each GPU needs KV from all other GPUs
    kv_tensor = torch.randn(batch_size, local_seq_len, 2, num_heads, dim // num_heads,
                            device=device, dtype=torch.bfloat16)
    kv_size_mb = kv_tensor.numel() * 2 / 1024 / 1024  # 2 bytes for bf16

    all_gather_time = benchmark_all_gather(kv_tensor, world_size)
    all_gather_bandwidth = kv_size_mb * world_size * 8 / all_gather_time  # Gbps

    results['kv_all_gather'] = {
        'tensor_shape': list(kv_tensor.shape),
        'size_mb': kv_size_mb,
        'time_ms': all_gather_time,
        'bandwidth_gbps': all_gather_bandwidth,
    }

    # 2. Simulate output all-gather after attention
    attn_output = torch.randn(batch_size, local_seq_len, dim,
                               device=device, dtype=torch.bfloat16)
    attn_size_mb = attn_output.numel() * 2 / 1024 / 1024

    attn_gather_time = benchmark_all_gather(attn_output, world_size)
    attn_gather_bandwidth = attn_size_mb * world_size * 8 / attn_gather_time

    results['attn_output_all_gather'] = {
        'tensor_shape': list(attn_output.shape),
        'size_mb': attn_size_mb,
        'time_ms': attn_gather_time,
        'bandwidth_gbps': attn_gather_bandwidth,
    }

    # 3. Simulate tensor parallelism all-reduce (for FFN)
    # In tensor parallelism, each GPU computes part of FFN then reduces
    ffn_output = torch.randn(batch_size, seq_len, dim,
                              device=device, dtype=torch.bfloat16)
    ffn_size_mb = ffn_output.numel() * 2 / 1024 / 1024

    all_reduce_time = benchmark_all_reduce(ffn_output)
    # All-reduce transfers 2*(N-1)/N * data_size effectively
    all_reduce_bandwidth = ffn_size_mb * 2 * (world_size - 1) / world_size * 8 / all_reduce_time

    results['ffn_all_reduce'] = {
        'tensor_shape': list(ffn_output.shape),
        'size_mb': ffn_size_mb,
        'time_ms': all_reduce_time,
        'bandwidth_gbps': all_reduce_bandwidth,
    }

    # 4. Estimate total communication time per layer
    # For sequence parallelism: 2 all-gathers per self-attention layer
    # For tensor parallelism: 2 all-reduces per layer (attention + FFN)
    comm_per_layer_sp = 2 * attn_gather_time  # sequence parallelism
    comm_per_layer_tp = 2 * all_reduce_time    # tensor parallelism

    results['per_layer_comm'] = {
        'sequence_parallel_ms': comm_per_layer_sp,
        'tensor_parallel_ms': comm_per_layer_tp,
        'total_sp_ms': comm_per_layer_sp * num_layers,
        'total_tp_ms': comm_per_layer_tp * num_layers,
    }

    return results


def profile_nccl_communication(args):
    """Profile NCCL communication patterns."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    setup_logging(rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        logging.info(f"=== NCCL Communication Profiling with {world_size} GPUs ===")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"wan_nccl/{timestamp}_gpus{world_size}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    # Wan2.1-T2V-1.3B model config
    model_config = {
        'dim': 1536,
        'num_heads': 12,
        'num_layers': 30,
        'ffn_dim': 8960,
        'seq_len': 5120,  # 480x832, 17 frames with patch_size (1,2,2)
    }

    if rank == 0:
        logging.info(f"\nModel config: {model_config}")
        logging.info(f"\nRunning communication benchmarks...")

    # Profile communication patterns
    comm_results = simulate_dit_forward(model_config, device, world_size, rank)

    if rank == 0:
        logging.info("\n" + "="*60)
        logging.info("NCCL COMMUNICATION BENCHMARK RESULTS")
        logging.info("="*60)

        for name, data in comm_results.items():
            if isinstance(data, dict) and 'time_ms' in data:
                logging.info(f"\n{name}:")
                logging.info(f"  Shape: {data.get('tensor_shape', 'N/A')}")
                logging.info(f"  Size: {data.get('size_mb', 0):.2f} MB")
                logging.info(f"  Time: {data['time_ms']:.4f} ms")
                logging.info(f"  Bandwidth: {data.get('bandwidth_gbps', 0):.2f} Gbps")
            elif isinstance(data, dict):
                logging.info(f"\n{name}:")
                for k, v in data.items():
                    logging.info(f"  {k}: {v:.4f} ms" if isinstance(v, float) else f"  {k}: {v}")

    # Profile with torch.profiler for detailed trace
    if rank == 0:
        logging.info("\n\nGenerating detailed profile trace...")

    profile_dir = output_dir / f"tensorboard_rank{rank}"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(str(profile_dir)),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        # Profile sequence parallelism pattern
        local_seq_len = model_config['seq_len'] // world_size
        dim = model_config['dim']

        for layer_idx in range(min(5, model_config['num_layers'])):  # Profile 5 layers
            # Simulate attention output
            attn_out = torch.randn(1, local_seq_len, dim, device=device, dtype=torch.bfloat16)
            gather_list = [torch.zeros_like(attn_out) for _ in range(world_size)]

            with torch.profiler.record_function(f"layer_{layer_idx}_all_gather"):
                dist.all_gather(gather_list, attn_out)

            # Simulate FFN output (tensor parallel all-reduce)
            ffn_out = torch.randn(1, model_config['seq_len'], dim, device=device, dtype=torch.bfloat16)

            with torch.profiler.record_function(f"layer_{layer_idx}_all_reduce"):
                dist.all_reduce(ffn_out)

    torch.cuda.synchronize()
    dist.barrier()

    # Estimate DistriFusion-style overlap potential
    if rank == 0:
        total_comm_sp = comm_results['per_layer_comm']['total_sp_ms']
        total_comm_tp = comm_results['per_layer_comm']['total_tp_ms']

        # Estimate compute time from single-GPU run (~7534 ms for 10 steps)
        # Compute time per step: ~753 ms, but with parallelism it should be faster
        single_gpu_time_per_step = 753.0  # ms from single-GPU profiling
        # With sequence parallelism, compute should scale ~linearly
        estimated_compute_per_step = single_gpu_time_per_step / world_size

        logging.info("\n" + "="*60)
        logging.info("DISTRIFUSION OVERLAP POTENTIAL ANALYSIS")
        logging.info("="*60)

        logging.info(f"\nSequence Parallelism (USP-style):")
        logging.info(f"  Communication time per step: {total_comm_sp:.2f} ms")
        logging.info(f"  Estimated compute time per step: {estimated_compute_per_step:.2f} ms")
        comm_pct_sp = total_comm_sp / (total_comm_sp + estimated_compute_per_step) * 100
        logging.info(f"  Communication percentage: {comm_pct_sp:.1f}%")
        if estimated_compute_per_step > 0:
            speedup_sp = (total_comm_sp + estimated_compute_per_step) / max(total_comm_sp, estimated_compute_per_step)
            logging.info(f"  Potential overlap speedup: {speedup_sp:.2f}x")

        logging.info(f"\nTensor Parallelism:")
        logging.info(f"  Communication time per step: {total_comm_tp:.2f} ms")
        logging.info(f"  Estimated compute time per step: {estimated_compute_per_step:.2f} ms")
        comm_pct_tp = total_comm_tp / (total_comm_tp + estimated_compute_per_step) * 100
        logging.info(f"  Communication percentage: {comm_pct_tp:.1f}%")
        if estimated_compute_per_step > 0:
            speedup_tp = (total_comm_tp + estimated_compute_per_step) / max(total_comm_tp, estimated_compute_per_step)
            logging.info(f"  Potential overlap speedup: {speedup_tp:.2f}x")

        # Save results
        results = {
            'world_size': world_size,
            'model_config': model_config,
            'communication_benchmarks': comm_results,
            'overlap_analysis': {
                'single_gpu_time_per_step_ms': single_gpu_time_per_step,
                'estimated_compute_per_step_ms': estimated_compute_per_step,
                'sequence_parallel': {
                    'total_comm_ms': total_comm_sp,
                    'comm_percentage': comm_pct_sp,
                    'potential_overlap_speedup': speedup_sp if estimated_compute_per_step > 0 else 0,
                },
                'tensor_parallel': {
                    'total_comm_ms': total_comm_tp,
                    'comm_percentage': comm_pct_tp,
                    'potential_overlap_speedup': speedup_tp if estimated_compute_per_step > 0 else 0,
                },
            }
        }

        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logging.info(f"\nResults saved to: {output_dir}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Profile NCCL communication for Wan2.1")
    parser.add_argument("--output_dir", type=str,
                       default=str(Path(__file__).parent.parent.parent / "profiling_results"),
                       help="Output directory for profiling results")

    args = parser.parse_args()
    profile_nccl_communication(args)


if __name__ == "__main__":
    main()
