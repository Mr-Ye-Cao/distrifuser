# Copyright 2024 DistriFuser Authors. All rights reserved.
# Simplified config for Wan2.1 that supports any world size (not just power of 2)
"""
DistriFusion config for Wan2.1 video generation.

Unlike the original DistriConfig, this doesn't require world_size to be a power of 2,
which is important for video models where we may use 3, 5, 6, etc. GPUs.
"""

import torch
from torch import distributed as dist


class DistriWanConfig:
    """
    Configuration for DistriFusion on Wan2.1.

    This is a simplified version of DistriConfig that:
    - Doesn't require power-of-2 world size
    - Doesn't use batch splitting (video gen is typically batch=1)
    - Doesn't use CUDA graphs (complex video models)
    """

    def __init__(
        self,
        height: int = 720,
        width: int = 1280,
        frame_num: int = 81,
        verbose: bool = False,
    ):
        # Get distributed info
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            try:
                dist.init_process_group("nccl")
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            except Exception as e:
                rank = 0
                world_size = 1
                if verbose:
                    print(f"Running in single-GPU mode: {e}")

        self.world_size = world_size
        self.rank = rank
        self.height = height
        self.width = width
        self.frame_num = frame_num
        self.verbose = verbose

        # For Wan2.1, we use TP without batch splitting
        self.n_device_per_batch = world_size
        self.do_classifier_free_guidance = True  # CFG is handled by model
        self.split_batch = False

        # Set device
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        self.device = device

        # Create process group for all devices (used for all-reduce)
        self.batch_group = None  # Will use default group

        if verbose and rank == 0:
            print(f"DistriWanConfig initialized:")
            print(f"  World size: {world_size}")
            print(f"  Resolution: {width}x{height}")
            print(f"  Frames: {frame_num}")

    def split_idx(self, rank: int = None) -> int:
        """Get the split index for a given rank."""
        if rank is None:
            rank = self.rank
        return rank % self.n_device_per_batch

    def batch_idx(self, rank: int = None) -> int:
        """Get the batch index (always 0 for video gen)."""
        return 0
