# Copyright 2024 DistriFuser Authors. All rights reserved.
# Async DistriFusion for Wan2.1 FFN with communication overlap.
"""
Async Tensor Parallelism for Wan2.1 feed-forward network.

Same pattern as async_attention.py:
- Warmup: synchronous all-reduce
- After warmup: use cached output, async communicate fresh output
"""

import torch
import torch.cuda
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from ..base_module import BaseModule


class DistriWanFFNAsyncTP(BaseModule):
    """
    Async Tensor Parallel wrapper for Wan2.1 FFN.

    Uses DistriFusion's activation reuse pattern for communication overlap.
    """

    def __init__(self, module: nn.Sequential, distri_config, warmup_steps: int = 4):
        super().__init__(module, distri_config)

        self.warmup_steps = warmup_steps
        self.output_cache = None
        self.async_handle = None
        self.pending_output = None

        # module is nn.Sequential with [Linear, GELU, Linear]
        fc1 = module[0]
        fc2 = module[2]

        ffn_dim = fc1.out_features
        dim = fc1.in_features
        self.original_dim = dim

        assert ffn_dim % distri_config.n_device_per_batch == 0, (
            f"ffn_dim ({ffn_dim}) must be divisible by "
            f"n_device_per_batch ({distri_config.n_device_per_batch})"
        )

        mid_features = ffn_dim // distri_config.n_device_per_batch
        start_idx = distri_config.split_idx() * mid_features
        end_idx = (distri_config.split_idx() + 1) * mid_features

        # Shard fc1 (row-wise)
        sharded_fc1 = nn.Linear(
            dim, mid_features,
            bias=fc1.bias is not None,
            device=fc1.weight.device,
            dtype=fc1.weight.dtype,
        )
        sharded_fc1.weight.data.copy_(fc1.weight.data[start_idx:end_idx])
        if fc1.bias is not None:
            sharded_fc1.bias.data.copy_(fc1.bias.data[start_idx:end_idx])

        # Store original fc2 bias
        self.fc2_bias = fc2.bias.data.clone() if fc2.bias is not None else None

        # Shard fc2 (column-wise, WITHOUT bias)
        sharded_fc2 = nn.Linear(
            mid_features, dim,
            bias=False,
            device=fc2.weight.device,
            dtype=fc2.weight.dtype,
        )
        sharded_fc2.weight.data.copy_(fc2.weight.data[:, start_idx:end_idx])

        old_fc1 = module[0]
        old_fc2 = module[2]
        module[0] = sharded_fc1
        module[2] = sharded_fc2
        del old_fc1, old_fc2
        torch.cuda.empty_cache()

    def _wait_for_async(self):
        """Wait for any pending async communication and update cache."""
        if self.async_handle is not None:
            self.async_handle.wait()
            self.async_handle = None
            if self.pending_output is not None:
                if self.fc2_bias is not None:
                    self.pending_output = self.pending_output + self.fc2_bias.view(1, 1, -1)
                self.output_cache = self.pending_output.clone()
                self.pending_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with async DistriFusion pattern.
        """
        distri_config = self.distri_config
        module = self.module

        self._wait_for_async()

        # Compute local FFN output
        hidden_states = module[0](x)  # fc1
        hidden_states = module[1](hidden_states)  # GELU
        hidden_states = F.linear(hidden_states, module[2].weight, bias=None)  # fc2 without bias

        if self.counter <= self.warmup_steps or self.output_cache is None:
            # Warmup: synchronous all-reduce
            dist.all_reduce(
                hidden_states,
                op=dist.ReduceOp.SUM,
                group=distri_config.batch_group,
                async_op=False
            )
            if self.fc2_bias is not None:
                hidden_states = hidden_states + self.fc2_bias.view(1, 1, -1)
            self.output_cache = hidden_states.clone()
            output = hidden_states
        else:
            # After warmup: use cached, async communicate fresh
            self.pending_output = hidden_states.clone()
            self.async_handle = dist.all_reduce(
                self.pending_output,
                op=dist.ReduceOp.SUM,
                group=distri_config.batch_group,
                async_op=True
            )
            output = self.output_cache

        self.counter += 1
        return output

    def reset_cache(self):
        """Reset cache at start of new generation."""
        self.output_cache = None
        self.async_handle = None
        self.pending_output = None
        self.counter = 0
