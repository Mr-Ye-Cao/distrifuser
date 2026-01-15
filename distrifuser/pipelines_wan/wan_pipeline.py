# Copyright 2024 DistriFuser Authors. All rights reserved.
# Adapted for Wan2.1 DiT video generation models.
"""
DistriFusion wrapper for Wan2.1 T2V pipeline.

Usage:
    from distrifuser.pipelines.wan_pipeline import DistriWanT2VPipeline

    # Initialize distributed config
    distri_config = DistriConfig(parallelism="tensor")

    # Wrap the Wan T2V model
    pipeline = DistriWanT2VPipeline.from_pretrained(distri_config, checkpoint_dir="...")

    # Generate video
    video = pipeline.generate(prompt, ...)
"""

import sys
from pathlib import Path

import torch
import torch.distributed as dist

from ..models.wan.distri_wan_dit import DistriWanDiT
from ..models.wan.distri_wan_dit_async import DistriWanDiTAsync

# Import Wan2.1 modules
WAN_PATH = Path(__file__).parent.parent.parent.parent / "Wan2.1"
if str(WAN_PATH) not in sys.path:
    sys.path.insert(0, str(WAN_PATH))


class DistriWanT2VPipeline:
    """
    DistriFusion wrapper for Wan2.1 text-to-video pipeline.

    This class wraps the WanT2V model and replaces its DiT backbone
    with a tensor-parallel version that distributes computation across GPUs.
    """

    def __init__(self, wan_t2v, distri_config):
        """
        Initialize the distributed pipeline.

        Args:
            wan_t2v: Loaded WanT2V instance
            distri_config: DistriFusion configuration
        """
        self.wan_t2v = wan_t2v
        self.distri_config = distri_config

        # Wrap the DiT model with TP
        self._wrap_model()

    def _wrap_model(self):
        """Wrap the DiT model with DistriFusion TP."""
        distri_config = self.distri_config

        # Wrap the model
        self.distri_dit = DistriWanDiT(self.wan_t2v.model, distri_config)

        # The wrapped modules are already in-place modified in the original model
        # So self.wan_t2v.model now uses TP attention and FFN

        if distri_config.rank == 0 and distri_config.verbose:
            print(f"DistriWanT2VPipeline: Model wrapped with TP (world_size={distri_config.world_size})")

    @staticmethod
    def from_pretrained(distri_config, checkpoint_dir: str, **kwargs):
        """
        Load Wan2.1 model and wrap it with DistriFusion.

        Args:
            distri_config: DistriFusion configuration
            checkpoint_dir: Path to Wan2.1 checkpoint directory
            **kwargs: Additional arguments passed to WanT2V

        Returns:
            DistriWanT2VPipeline instance
        """
        import wan
        from wan.configs import WAN_CONFIGS

        # Get config
        task = kwargs.pop("task", "t2v-1.3B")
        cfg = WAN_CONFIGS[task]

        # Create WanT2V instance
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=checkpoint_dir,
            device_id=distri_config.rank,
            rank=distri_config.rank,
            t5_fsdp=kwargs.pop("t5_fsdp", False),
            dit_fsdp=kwargs.pop("dit_fsdp", False),
            use_usp=False,  # We use our own TP instead of USP
            t5_cpu=kwargs.pop("t5_cpu", True),
        )

        return DistriWanT2VPipeline(wan_t2v, distri_config)

    def generate(self, *args, **kwargs):
        """
        Generate video frames from text prompt.

        All arguments are passed directly to WanT2V.generate().

        Returns:
            Generated video tensor
        """
        # Synchronize before generation
        if dist.is_initialized():
            dist.barrier()

        # Reset counter at start of generation
        self.distri_dit.set_counter(0)

        # Call original generate
        return self.wan_t2v.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Alias for generate()."""
        return self.generate(*args, **kwargs)


def wrap_wan_model(wan_t2v, distri_config) -> DistriWanDiT:
    """
    Utility function to wrap an existing WanT2V model with DistriFusion TP.

    This modifies the model in-place and returns the DistriWanDiT wrapper.

    Args:
        wan_t2v: Loaded WanT2V instance
        distri_config: DistriFusion configuration

    Returns:
        DistriWanDiT wrapper (model is also modified in-place)

    Example:
        import wan
        from wan.configs import WAN_CONFIGS
        from distrifuser.utils import DistriConfig
        from distrifuser.pipelines.wan_pipeline import wrap_wan_model

        # Load model normally
        cfg = WAN_CONFIGS['t2v-1.3B']
        wan_t2v = wan.WanT2V(config=cfg, checkpoint_dir=..., use_usp=False, ...)

        # Initialize distributed config
        distri_config = DistriConfig(parallelism="tensor")

        # Wrap with TP
        distri_dit = wrap_wan_model(wan_t2v, distri_config)

        # Generate (model now uses TP)
        video = wan_t2v.generate(prompt, ...)
    """
    return DistriWanDiT(wan_t2v.model, distri_config)


def wrap_wan_model_async(wan_t2v, distri_config, warmup_steps: int = 4) -> DistriWanDiTAsync:
    """
    Utility function to wrap an existing WanT2V model with async DistriFusion TP.

    This uses the DistriFusion paper's key technique: asynchronous all-reduce
    with activation caching to overlap communication with computation.

    Args:
        wan_t2v: Loaded WanT2V instance
        distri_config: DistriFusion configuration
        warmup_steps: Number of initial steps to use synchronous all-reduce
                     (cache needs to warm up before async can work)

    Returns:
        DistriWanDiTAsync wrapper (model is also modified in-place)

    Example:
        import wan
        from wan.configs import WAN_CONFIGS
        from distrifuser.utils import DistriConfig
        from distrifuser.pipelines.wan_pipeline import wrap_wan_model_async

        # Load model normally
        cfg = WAN_CONFIGS['t2v-1.3B']
        wan_t2v = wan.WanT2V(config=cfg, checkpoint_dir=..., use_usp=False, ...)

        # Initialize distributed config
        distri_config = DistriConfig(parallelism="tensor")

        # Wrap with async TP
        distri_dit = wrap_wan_model_async(wan_t2v, distri_config, warmup_steps=4)

        # Reset caches at start of each generation
        distri_dit.reset_all_caches()
        distri_dit.set_counter(0)

        # Generate (model now uses async TP with communication overlap)
        video = wan_t2v.generate(prompt, ...)

        # Flush any pending async operations
        distri_dit.flush_all_async()
    """
    return DistriWanDiTAsync(wan_t2v.model, distri_config, warmup_steps)
