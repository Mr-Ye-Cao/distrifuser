# Copyright 2024 DistriFuser Authors. All rights reserved.
# Async DistriFusion wrapper for Wan2.1 DiT model.
"""
Async DistriFusion for Wan2.1 DiT model.

Uses asynchronous all-reduce with activation caching to overlap
communication with computation, achieving speedup over synchronous TP.
"""

import sys
from pathlib import Path

import torch
import torch.cuda
from torch import nn

from ..base_model import BaseModel
from ...modules.base_module import BaseModule
from ...modules.wan.async_attention import DistriWanSelfAttentionAsyncTP, DistriWanCrossAttentionAsyncTP
from ...modules.wan.async_feed_forward import DistriWanFFNAsyncTP

# Import Wan2.1 modules
WAN_PATH = Path(__file__).parent.parent.parent.parent.parent / "Wan2.1"
if str(WAN_PATH) not in sys.path:
    sys.path.insert(0, str(WAN_PATH))

try:
    from wan.modules.model import WanSelfAttention, WanT2VCrossAttention, WanI2VCrossAttention, WanAttentionBlock
except ImportError:
    WanSelfAttention = None
    WanT2VCrossAttention = None
    WanI2VCrossAttention = None
    WanAttentionBlock = None


class DistriWanDiTAsync(BaseModel):
    """
    Async DistriFusion wrapper for Wan2.1 DiT model.

    Key features:
    - Warmup phase: first N steps use synchronous all-reduce
    - After warmup: async all-reduce with activation caching
    - Overlap communication with computation for speedup
    """

    def __init__(self, model: nn.Module, distri_config, warmup_steps: int = 4):
        super().__init__(model, distri_config)

        self.warmup_steps = warmup_steps
        self.async_modules = []

        # Wrap attention and FFN modules with async TP versions
        self._wrap_modules(model, distri_config, warmup_steps)

    def _wrap_modules(self, model: nn.Module, distri_config, warmup_steps: int):
        """
        Wrap attention and FFN modules with async TP versions.
        """
        if WanAttentionBlock is None:
            raise ImportError("Wan2.1 modules not found.")

        wrapped_count = {"self_attn": 0, "cross_attn": 0, "ffn": 0}

        for name, module in model.named_modules():
            if isinstance(module, WanAttentionBlock):
                # Wrap self-attention
                if hasattr(module, "self_attn") and isinstance(module.self_attn, WanSelfAttention):
                    wrapped_self_attn = DistriWanSelfAttentionAsyncTP(
                        module.self_attn, distri_config, warmup_steps
                    )
                    module.self_attn = wrapped_self_attn
                    self.async_modules.append(wrapped_self_attn)
                    wrapped_count["self_attn"] += 1

                # Wrap cross-attention
                if hasattr(module, "cross_attn"):
                    if isinstance(module.cross_attn, (WanT2VCrossAttention, WanI2VCrossAttention)):
                        wrapped_cross_attn = DistriWanCrossAttentionAsyncTP(
                            module.cross_attn, distri_config, warmup_steps
                        )
                        module.cross_attn = wrapped_cross_attn
                        self.async_modules.append(wrapped_cross_attn)
                        wrapped_count["cross_attn"] += 1

                # Wrap FFN
                if hasattr(module, "ffn") and isinstance(module.ffn, nn.Sequential):
                    wrapped_ffn = DistriWanFFNAsyncTP(
                        module.ffn, distri_config, warmup_steps
                    )
                    module.ffn = wrapped_ffn
                    self.async_modules.append(wrapped_ffn)
                    wrapped_count["ffn"] += 1

        if distri_config.rank == 0 and distri_config.verbose:
            print(f"DistriWanDiTAsync: Wrapped modules with async TP:")
            print(f"  Self-attention: {wrapped_count['self_attn']}")
            print(f"  Cross-attention: {wrapped_count['cross_attn']}")
            print(f"  FFN: {wrapped_count['ffn']}")
            print(f"  Warmup steps: {warmup_steps}")

    def reset_all_caches(self):
        """Reset caches in all async modules. Call at start of new generation."""
        for module in self.async_modules:
            module.reset_cache()

    def flush_all_async(self):
        """Wait for all pending async operations to complete."""
        for module in self.async_modules:
            module._wait_for_async()

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        """
        Forward pass through the wrapped model.
        """
        # Call the original model's forward
        output = self.model(x, t, context, seq_len, clip_fea, y)

        # Increment counter
        self.counter += 1

        return output

    def set_counter(self, counter: int = 0):
        """Set counter for all modules."""
        self.counter = counter
        for module in self.async_modules:
            module.set_counter(counter)
