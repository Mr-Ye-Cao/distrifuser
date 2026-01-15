# Copyright 2024 DistriFuser Authors. All rights reserved.
# Adapted for Wan2.1 DiT video generation models.
"""
Tensor Parallelism wrappers for Wan2.1 attention modules.

These modules shard attention heads across GPUs and use all-reduce
to combine the outputs, enabling efficient distributed inference.
"""

import torch
import torch.cuda
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from ..base_module import BaseModule
from ...utils import DistriConfig


class DistriWanSelfAttentionTP(BaseModule):
    """
    Tensor Parallel wrapper for WanSelfAttention.

    Shards Q, K, V, O projections by attention heads.
    Each GPU computes attention for a subset of heads,
    then all-reduce combines the output projections.
    """

    def __init__(self, module: nn.Module, distri_config: DistriConfig):
        super().__init__(module, distri_config)

        num_heads = module.num_heads
        head_dim = module.head_dim
        dim = module.dim

        # Calculate head distribution across devices
        sliced_heads = num_heads // distri_config.n_device_per_batch
        remainder_heads = num_heads % distri_config.n_device_per_batch
        if distri_config.split_idx() < remainder_heads:
            sliced_heads += 1
        self.sliced_heads = sliced_heads
        self.head_dim = head_dim

        if sliced_heads > 0:
            # Calculate start/end head indices for this device
            if distri_config.split_idx() < remainder_heads:
                start_head = distri_config.split_idx() * sliced_heads
            else:
                start_head = (
                    remainder_heads * (sliced_heads + 1)
                    + (distri_config.split_idx() - remainder_heads) * sliced_heads
                )
            end_head = start_head + sliced_heads

            # Shard Q projection
            sharded_q = nn.Linear(
                dim, sliced_heads * head_dim,
                bias=module.q.bias is not None,
                device=module.q.weight.device,
                dtype=module.q.weight.dtype,
            )
            sharded_q.weight.data.copy_(
                module.q.weight.data[start_head * head_dim : end_head * head_dim]
            )
            if module.q.bias is not None:
                sharded_q.bias.data.copy_(
                    module.q.bias.data[start_head * head_dim : end_head * head_dim]
                )

            # Shard K projection
            sharded_k = nn.Linear(
                dim, sliced_heads * head_dim,
                bias=module.k.bias is not None,
                device=module.k.weight.device,
                dtype=module.k.weight.dtype,
            )
            sharded_k.weight.data.copy_(
                module.k.weight.data[start_head * head_dim : end_head * head_dim]
            )
            if module.k.bias is not None:
                sharded_k.bias.data.copy_(
                    module.k.bias.data[start_head * head_dim : end_head * head_dim]
                )

            # Shard V projection
            sharded_v = nn.Linear(
                dim, sliced_heads * head_dim,
                bias=module.v.bias is not None,
                device=module.v.weight.device,
                dtype=module.v.weight.dtype,
            )
            sharded_v.weight.data.copy_(
                module.v.weight.data[start_head * head_dim : end_head * head_dim]
            )
            if module.v.bias is not None:
                sharded_v.bias.data.copy_(
                    module.v.bias.data[start_head * head_dim : end_head * head_dim]
                )

            # Store original O bias before sharding (will add after all-reduce)
            self.o_bias = module.o.bias.data.clone() if module.o.bias is not None else None

            # Shard O projection (column-wise sharding, WITHOUT bias)
            # Bias will be added after all-reduce to avoid duplication
            sharded_o = nn.Linear(
                sliced_heads * head_dim, dim,
                bias=False,  # No bias - will add after all-reduce
                device=module.o.weight.device,
                dtype=module.o.weight.dtype,
            )
            sharded_o.weight.data.copy_(
                module.o.weight.data[:, start_head * head_dim : end_head * head_dim]
            )

            # Shard norm_q and norm_k if they exist
            if hasattr(module, 'norm_q') and hasattr(module.norm_q, 'weight'):
                # WanRMSNorm - need to shard the weight
                sharded_norm_q_weight = module.norm_q.weight.data[
                    start_head * head_dim : end_head * head_dim
                ].clone()
            else:
                sharded_norm_q_weight = None

            if hasattr(module, 'norm_k') and hasattr(module.norm_k, 'weight'):
                sharded_norm_k_weight = module.norm_k.weight.data[
                    start_head * head_dim : end_head * head_dim
                ].clone()
            else:
                sharded_norm_k_weight = None

            # Delete old modules
            old_q = module.q
            old_k = module.k
            old_v = module.v
            old_o = module.o

            module.q = sharded_q
            module.k = sharded_k
            module.v = sharded_v
            module.o = sharded_o
            module.num_heads = sliced_heads

            # Update norm weights if they were sharded
            if sharded_norm_q_weight is not None:
                module.norm_q.weight = nn.Parameter(sharded_norm_q_weight)
                module.norm_q.dim = sliced_heads * head_dim
            if sharded_norm_k_weight is not None:
                module.norm_k.weight = nn.Parameter(sharded_norm_k_weight)
                module.norm_k.dim = sliced_heads * head_dim

            del old_q, old_k, old_v, old_o
            torch.cuda.empty_cache()
        else:
            # No heads on this device - still need to store bias for all-reduce
            self.o_bias = module.o.bias.data.clone() if module.o.bias is not None else None

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        Forward pass with tensor parallelism.

        Args:
            x: Input tensor [B, L, C]
            seq_lens: Sequence lengths [B]
            grid_sizes: Grid sizes [B, 3]
            freqs: RoPE frequencies

        Returns:
            Output tensor [B, L, C]
        """
        distri_config = self.distri_config
        module = self.module

        if self.sliced_heads > 0:
            # Call original forward (which now uses sharded weights)
            hidden_states = module.forward(x, seq_lens, grid_sizes, freqs)
        else:
            # This device has no heads - output zeros
            hidden_states = torch.zeros(
                x.shape[0], x.shape[1], module.o.out_features,
                device=x.device,
                dtype=x.dtype,
            )

        # All-reduce across devices
        dist.all_reduce(
            hidden_states,
            op=dist.ReduceOp.SUM,
            group=distri_config.batch_group,
            async_op=False
        )

        # Add bias after all-reduce (only once)
        if self.o_bias is not None:
            hidden_states = hidden_states + self.o_bias.view(1, 1, -1)

        self.counter += 1
        return hidden_states


class DistriWanCrossAttentionTP(BaseModule):
    """
    Tensor Parallel wrapper for WanT2VCrossAttention.

    Similar to DistriWanSelfAttentionTP but for cross-attention.
    """

    def __init__(self, module: nn.Module, distri_config: DistriConfig):
        super().__init__(module, distri_config)

        num_heads = module.num_heads
        head_dim = module.head_dim
        dim = module.dim

        # Calculate head distribution across devices
        sliced_heads = num_heads // distri_config.n_device_per_batch
        remainder_heads = num_heads % distri_config.n_device_per_batch
        if distri_config.split_idx() < remainder_heads:
            sliced_heads += 1
        self.sliced_heads = sliced_heads
        self.head_dim = head_dim

        if sliced_heads > 0:
            # Calculate start/end head indices for this device
            if distri_config.split_idx() < remainder_heads:
                start_head = distri_config.split_idx() * sliced_heads
            else:
                start_head = (
                    remainder_heads * (sliced_heads + 1)
                    + (distri_config.split_idx() - remainder_heads) * sliced_heads
                )
            end_head = start_head + sliced_heads

            # Shard Q projection
            sharded_q = nn.Linear(
                dim, sliced_heads * head_dim,
                bias=module.q.bias is not None,
                device=module.q.weight.device,
                dtype=module.q.weight.dtype,
            )
            sharded_q.weight.data.copy_(
                module.q.weight.data[start_head * head_dim : end_head * head_dim]
            )
            if module.q.bias is not None:
                sharded_q.bias.data.copy_(
                    module.q.bias.data[start_head * head_dim : end_head * head_dim]
                )

            # Shard K projection
            sharded_k = nn.Linear(
                dim, sliced_heads * head_dim,
                bias=module.k.bias is not None,
                device=module.k.weight.device,
                dtype=module.k.weight.dtype,
            )
            sharded_k.weight.data.copy_(
                module.k.weight.data[start_head * head_dim : end_head * head_dim]
            )
            if module.k.bias is not None:
                sharded_k.bias.data.copy_(
                    module.k.bias.data[start_head * head_dim : end_head * head_dim]
                )

            # Shard V projection
            sharded_v = nn.Linear(
                dim, sliced_heads * head_dim,
                bias=module.v.bias is not None,
                device=module.v.weight.device,
                dtype=module.v.weight.dtype,
            )
            sharded_v.weight.data.copy_(
                module.v.weight.data[start_head * head_dim : end_head * head_dim]
            )
            if module.v.bias is not None:
                sharded_v.bias.data.copy_(
                    module.v.bias.data[start_head * head_dim : end_head * head_dim]
                )

            # Store original O bias before sharding (will add after all-reduce)
            self.o_bias = module.o.bias.data.clone() if module.o.bias is not None else None

            # Shard O projection (column-wise sharding, WITHOUT bias)
            # Bias will be added after all-reduce to avoid duplication
            sharded_o = nn.Linear(
                sliced_heads * head_dim, dim,
                bias=False,  # No bias - will add after all-reduce
                device=module.o.weight.device,
                dtype=module.o.weight.dtype,
            )
            sharded_o.weight.data.copy_(
                module.o.weight.data[:, start_head * head_dim : end_head * head_dim]
            )

            # Shard norm weights
            if hasattr(module, 'norm_q') and hasattr(module.norm_q, 'weight'):
                sharded_norm_q_weight = module.norm_q.weight.data[
                    start_head * head_dim : end_head * head_dim
                ].clone()
            else:
                sharded_norm_q_weight = None

            if hasattr(module, 'norm_k') and hasattr(module.norm_k, 'weight'):
                sharded_norm_k_weight = module.norm_k.weight.data[
                    start_head * head_dim : end_head * head_dim
                ].clone()
            else:
                sharded_norm_k_weight = None

            # Delete old modules
            old_q = module.q
            old_k = module.k
            old_v = module.v
            old_o = module.o

            module.q = sharded_q
            module.k = sharded_k
            module.v = sharded_v
            module.o = sharded_o
            module.num_heads = sliced_heads

            # Update norm weights
            if sharded_norm_q_weight is not None:
                module.norm_q.weight = nn.Parameter(sharded_norm_q_weight)
                module.norm_q.dim = sliced_heads * head_dim
            if sharded_norm_k_weight is not None:
                module.norm_k.weight = nn.Parameter(sharded_norm_k_weight)
                module.norm_k.dim = sliced_heads * head_dim

            del old_q, old_k, old_v, old_o
            torch.cuda.empty_cache()
        else:
            # No heads on this device - still need to store bias for all-reduce
            self.o_bias = module.o.bias.data.clone() if module.o.bias is not None else None

    def forward(self, x, context, context_lens):
        """
        Forward pass with tensor parallelism.

        Args:
            x: Query tensor [B, L1, C]
            context: Key/Value tensor [B, L2, C]
            context_lens: Context lengths [B]

        Returns:
            Output tensor [B, L1, C]
        """
        distri_config = self.distri_config
        module = self.module

        if self.sliced_heads > 0:
            # Call original forward (which now uses sharded weights)
            hidden_states = module.forward(x, context, context_lens)
        else:
            # This device has no heads - output zeros
            hidden_states = torch.zeros(
                x.shape[0], x.shape[1], module.o.out_features,
                device=x.device,
                dtype=x.dtype,
            )

        # All-reduce across devices
        dist.all_reduce(
            hidden_states,
            op=dist.ReduceOp.SUM,
            group=distri_config.batch_group,
            async_op=False
        )

        # Add bias after all-reduce (only once)
        if self.o_bias is not None:
            hidden_states = hidden_states + self.o_bias.view(1, 1, -1)

        self.counter += 1
        return hidden_states
