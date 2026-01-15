# Copyright 2024 DistriFuser Authors. All rights reserved.
# Adapted for Wan2.1 DiT video generation models.
"""
DistriFusion wrapper for Wan2.1 DiT model.

This class wraps the WanModel and replaces attention and FFN modules
with tensor-parallel versions that use all-reduce for communication.
"""

import sys
from pathlib import Path

import torch
import torch.cuda
from torch import nn

from ..base_model import BaseModel
from ...modules.base_module import BaseModule
from ...modules.wan.attention import DistriWanSelfAttentionTP, DistriWanCrossAttentionTP
from ...modules.wan.feed_forward import DistriWanFFNTP
from ...utils import DistriConfig

# Import Wan2.1 modules for isinstance checks
WAN_PATH = Path(__file__).parent.parent.parent.parent.parent / "Wan2.1"
if str(WAN_PATH) not in sys.path:
    sys.path.insert(0, str(WAN_PATH))

try:
    from wan.modules.model import WanSelfAttention, WanT2VCrossAttention, WanI2VCrossAttention, WanAttentionBlock
except ImportError:
    # Fallback for when Wan2.1 is not available
    WanSelfAttention = None
    WanT2VCrossAttention = None
    WanI2VCrossAttention = None
    WanAttentionBlock = None


class DistriWanDiT(BaseModel):
    """
    DistriFusion wrapper for Wan2.1 DiT model.

    Replaces attention and FFN modules with tensor-parallel versions
    that shard computation across GPUs and use all-reduce.
    """

    def __init__(self, model: nn.Module, distri_config: DistriConfig):
        super().__init__(model, distri_config)

        # Wrap attention and FFN modules with TP versions
        self._wrap_modules(model, distri_config)

    def _wrap_modules(self, model: nn.Module, distri_config: DistriConfig):
        """
        Recursively wrap attention and FFN modules with TP versions.
        """
        if WanAttentionBlock is None:
            raise ImportError(
                "Wan2.1 modules not found. Please ensure Wan2.1 is installed."
            )

        wrapped_count = {"self_attn": 0, "cross_attn": 0, "ffn": 0}

        # Iterate through all WanAttentionBlock modules
        for name, module in model.named_modules():
            if isinstance(module, WanAttentionBlock):
                # Wrap self-attention
                if hasattr(module, "self_attn") and isinstance(module.self_attn, WanSelfAttention):
                    wrapped_self_attn = DistriWanSelfAttentionTP(module.self_attn, distri_config)
                    module.self_attn = wrapped_self_attn
                    wrapped_count["self_attn"] += 1

                # Wrap cross-attention
                if hasattr(module, "cross_attn"):
                    if isinstance(module.cross_attn, (WanT2VCrossAttention, WanI2VCrossAttention)):
                        wrapped_cross_attn = DistriWanCrossAttentionTP(module.cross_attn, distri_config)
                        module.cross_attn = wrapped_cross_attn
                        wrapped_count["cross_attn"] += 1

                # Wrap FFN
                if hasattr(module, "ffn") and isinstance(module.ffn, nn.Sequential):
                    wrapped_ffn = DistriWanFFNTP(module.ffn, distri_config)
                    module.ffn = wrapped_ffn
                    wrapped_count["ffn"] += 1

        if distri_config.rank == 0 and distri_config.verbose:
            print(f"DistriWanDiT: Wrapped modules:")
            print(f"  Self-attention: {wrapped_count['self_attn']}")
            print(f"  Cross-attention: {wrapped_count['cross_attn']}")
            print(f"  FFN: {wrapped_count['ffn']}")

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        """
        Forward pass through the wrapped model.

        Args:
            x: List of input video tensors
            t: Diffusion timesteps
            context: Text embeddings
            seq_len: Maximum sequence length
            clip_fea: Optional CLIP features
            y: Optional conditional video inputs

        Returns:
            List of denoised video tensors
        """
        # Reset counter at start of each step
        self.set_counter(self.counter)

        # Call the original model's forward
        output = self.model(x, t, context, seq_len, clip_fea, y)

        # Increment counter
        self.counter += 1

        return output
