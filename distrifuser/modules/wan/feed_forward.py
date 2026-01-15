# Copyright 2024 DistriFuser Authors. All rights reserved.
# Adapted for Wan2.1 DiT video generation models.
"""
Tensor Parallelism wrapper for Wan2.1 feed-forward network.

The Wan2.1 FFN is: Linear(dim -> ffn_dim) -> GELU -> Linear(ffn_dim -> dim)
We shard the intermediate dimension (ffn_dim) across GPUs.
"""

import torch
import torch.cuda
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from ..base_module import BaseModule
from ...utils import DistriConfig


class DistriWanFFNTP(BaseModule):
    """
    Tensor Parallel wrapper for Wan2.1 FFN.

    Shards the intermediate dimension (ffn_dim) across GPUs.
    Each GPU computes a portion of the FFN, then all-reduce
    combines the output.

    FFN structure: Linear(dim -> ffn_dim) -> GELU -> Linear(ffn_dim -> dim)
    """

    def __init__(self, module: nn.Sequential, distri_config: DistriConfig):
        super().__init__(module, distri_config)

        # module is nn.Sequential with [Linear, GELU, Linear]
        fc1 = module[0]  # Linear(dim, ffn_dim)
        gelu = module[1]  # GELU
        fc2 = module[2]  # Linear(ffn_dim, dim)

        ffn_dim = fc1.out_features
        dim = fc1.in_features

        # Ensure ffn_dim is divisible by number of devices
        assert ffn_dim % distri_config.n_device_per_batch == 0, (
            f"ffn_dim ({ffn_dim}) must be divisible by "
            f"n_device_per_batch ({distri_config.n_device_per_batch})"
        )

        mid_features = ffn_dim // distri_config.n_device_per_batch
        start_idx = distri_config.split_idx() * mid_features
        end_idx = (distri_config.split_idx() + 1) * mid_features

        # Shard fc1 (row-wise: output dimension sharding)
        sharded_fc1 = nn.Linear(
            dim, mid_features,
            bias=fc1.bias is not None,
            device=fc1.weight.device,
            dtype=fc1.weight.dtype,
        )
        sharded_fc1.weight.data.copy_(fc1.weight.data[start_idx:end_idx])
        if fc1.bias is not None:
            sharded_fc1.bias.data.copy_(fc1.bias.data[start_idx:end_idx])

        # Store original fc2 bias before sharding (will add after all-reduce)
        self.fc2_bias = fc2.bias.data.clone() if fc2.bias is not None else None

        # Shard fc2 (column-wise: input dimension sharding, WITHOUT bias)
        # Bias will be added after all-reduce to avoid duplication
        sharded_fc2 = nn.Linear(
            mid_features, dim,
            bias=False,  # No bias - will add after all-reduce
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with tensor parallelism.

        Args:
            x: Input tensor [B, L, dim]

        Returns:
            Output tensor [B, L, dim]
        """
        distri_config = self.distri_config
        module = self.module

        # fc1 + GELU
        hidden_states = module[0](x)
        hidden_states = module[1](hidden_states)

        # fc2 (without bias - will add after all-reduce)
        hidden_states = F.linear(hidden_states, module[2].weight, bias=None)

        # All-reduce across devices
        dist.all_reduce(
            hidden_states,
            op=dist.ReduceOp.SUM,
            group=distri_config.batch_group,
            async_op=False
        )

        # Add bias after all-reduce (only once)
        if self.fc2_bias is not None:
            hidden_states = hidden_states + self.fc2_bias.view(1, 1, -1)

        self.counter += 1
        return hidden_states
