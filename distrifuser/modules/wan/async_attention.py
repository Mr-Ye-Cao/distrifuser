# Copyright 2024 DistriFuser Authors. All rights reserved.
# Async DistriFusion for Wan2.1 attention with communication overlap.
"""
Async Tensor Parallelism for Wan2.1 attention.

Key insight from DistriFusion paper:
- Consecutive diffusion timesteps produce similar outputs (temporal similarity)
- We can use "stale" activations from previous timestep while computing fresh ones
- This hides communication latency behind computation

Pattern:
1. Warmup phase (first N steps): Use synchronous all-reduce
2. After warmup:
   - Use cached output from previous timestep for computation
   - Start async all-reduce for current output
   - Cache results for next timestep
"""

import torch
import torch.cuda
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from ..base_module import BaseModule


class DistriWanSelfAttentionAsyncTP(BaseModule):
    """
    Async Tensor Parallel wrapper for WanSelfAttention.

    Uses DistriFusion's activation reuse pattern:
    - Cache outputs between timesteps
    - Use stale outputs while async communicating fresh ones
    """

    def __init__(self, module: nn.Module, distri_config, warmup_steps: int = 4):
        super().__init__(module, distri_config)

        self.warmup_steps = warmup_steps

        # Cache for async communication
        self.output_cache = None  # Cached output from previous timestep
        self.async_handle = None  # Handle for async all-reduce
        self.pending_output = None  # Output waiting for async completion

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
        self.original_dim = dim

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

            # Store original O bias before sharding
            self.o_bias = module.o.bias.data.clone() if module.o.bias is not None else None

            # Shard O projection (WITHOUT bias - will add after all-reduce)
            sharded_o = nn.Linear(
                sliced_heads * head_dim, dim,
                bias=False,
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
                module.norm_q.weight = nn.Parameter(sharded_norm_q_weight)
                module.norm_q.dim = sliced_heads * head_dim

            if hasattr(module, 'norm_k') and hasattr(module.norm_k, 'weight'):
                sharded_norm_k_weight = module.norm_k.weight.data[
                    start_head * head_dim : end_head * head_dim
                ].clone()
                module.norm_k.weight = nn.Parameter(sharded_norm_k_weight)
                module.norm_k.dim = sliced_heads * head_dim

            # Replace original modules
            old_q, old_k, old_v, old_o = module.q, module.k, module.v, module.o
            module.q = sharded_q
            module.k = sharded_k
            module.v = sharded_v
            module.o = sharded_o
            module.num_heads = sliced_heads
            del old_q, old_k, old_v, old_o
            torch.cuda.empty_cache()
        else:
            self.o_bias = module.o.bias.data.clone() if module.o.bias is not None else None

    def _wait_for_async(self):
        """Wait for any pending async communication and update cache."""
        if self.async_handle is not None:
            self.async_handle.wait()
            self.async_handle = None
            # Update cache with fresh output
            if self.pending_output is not None:
                # Add bias after all-reduce
                if self.o_bias is not None:
                    self.pending_output = self.pending_output + self.o_bias.view(1, 1, -1)
                self.output_cache = self.pending_output.clone()
                self.pending_output = None

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        Forward pass with async DistriFusion pattern.
        """
        distri_config = self.distri_config
        module = self.module

        # Wait for any pending async from previous layer call
        self._wait_for_async()

        # Compute local attention output
        if self.sliced_heads > 0:
            hidden_states = module.forward(x, seq_lens, grid_sizes, freqs)
        else:
            hidden_states = torch.zeros(
                x.shape[0], x.shape[1], self.original_dim,
                device=x.device,
                dtype=x.dtype,
            )

        # Decide sync vs async based on counter
        if self.counter <= self.warmup_steps or self.output_cache is None:
            # Warmup: synchronous all-reduce
            dist.all_reduce(
                hidden_states,
                op=dist.ReduceOp.SUM,
                group=distri_config.batch_group,
                async_op=False
            )
            # Add bias
            if self.o_bias is not None:
                hidden_states = hidden_states + self.o_bias.view(1, 1, -1)
            # Cache for next timestep
            self.output_cache = hidden_states.clone()
            output = hidden_states
        else:
            # After warmup: use cached output, async communicate fresh output
            # Start async all-reduce for fresh output
            self.pending_output = hidden_states.clone()
            self.async_handle = dist.all_reduce(
                self.pending_output,
                op=dist.ReduceOp.SUM,
                group=distri_config.batch_group,
                async_op=True
            )
            # Return cached output from previous timestep
            output = self.output_cache

        self.counter += 1
        return output

    def reset_cache(self):
        """Reset cache at start of new generation."""
        self.output_cache = None
        self.async_handle = None
        self.pending_output = None
        self.counter = 0


class DistriWanCrossAttentionAsyncTP(BaseModule):
    """
    Async Tensor Parallel wrapper for WanT2VCrossAttention.
    Same pattern as DistriWanSelfAttentionAsyncTP.
    """

    def __init__(self, module: nn.Module, distri_config, warmup_steps: int = 4):
        super().__init__(module, distri_config)

        self.warmup_steps = warmup_steps
        self.output_cache = None
        self.async_handle = None
        self.pending_output = None

        num_heads = module.num_heads
        head_dim = module.head_dim
        dim = module.dim

        sliced_heads = num_heads // distri_config.n_device_per_batch
        remainder_heads = num_heads % distri_config.n_device_per_batch
        if distri_config.split_idx() < remainder_heads:
            sliced_heads += 1
        self.sliced_heads = sliced_heads
        self.head_dim = head_dim
        self.original_dim = dim

        if sliced_heads > 0:
            if distri_config.split_idx() < remainder_heads:
                start_head = distri_config.split_idx() * sliced_heads
            else:
                start_head = (
                    remainder_heads * (sliced_heads + 1)
                    + (distri_config.split_idx() - remainder_heads) * sliced_heads
                )
            end_head = start_head + sliced_heads

            # Shard Q, K, V projections
            sharded_q = nn.Linear(dim, sliced_heads * head_dim, bias=module.q.bias is not None,
                                  device=module.q.weight.device, dtype=module.q.weight.dtype)
            sharded_q.weight.data.copy_(module.q.weight.data[start_head * head_dim : end_head * head_dim])
            if module.q.bias is not None:
                sharded_q.bias.data.copy_(module.q.bias.data[start_head * head_dim : end_head * head_dim])

            sharded_k = nn.Linear(dim, sliced_heads * head_dim, bias=module.k.bias is not None,
                                  device=module.k.weight.device, dtype=module.k.weight.dtype)
            sharded_k.weight.data.copy_(module.k.weight.data[start_head * head_dim : end_head * head_dim])
            if module.k.bias is not None:
                sharded_k.bias.data.copy_(module.k.bias.data[start_head * head_dim : end_head * head_dim])

            sharded_v = nn.Linear(dim, sliced_heads * head_dim, bias=module.v.bias is not None,
                                  device=module.v.weight.device, dtype=module.v.weight.dtype)
            sharded_v.weight.data.copy_(module.v.weight.data[start_head * head_dim : end_head * head_dim])
            if module.v.bias is not None:
                sharded_v.bias.data.copy_(module.v.bias.data[start_head * head_dim : end_head * head_dim])

            self.o_bias = module.o.bias.data.clone() if module.o.bias is not None else None

            sharded_o = nn.Linear(sliced_heads * head_dim, dim, bias=False,
                                  device=module.o.weight.device, dtype=module.o.weight.dtype)
            sharded_o.weight.data.copy_(module.o.weight.data[:, start_head * head_dim : end_head * head_dim])

            # Shard norms
            if hasattr(module, 'norm_q') and hasattr(module.norm_q, 'weight'):
                sharded_norm_q_weight = module.norm_q.weight.data[start_head * head_dim : end_head * head_dim].clone()
                module.norm_q.weight = nn.Parameter(sharded_norm_q_weight)
                module.norm_q.dim = sliced_heads * head_dim
            if hasattr(module, 'norm_k') and hasattr(module.norm_k, 'weight'):
                sharded_norm_k_weight = module.norm_k.weight.data[start_head * head_dim : end_head * head_dim].clone()
                module.norm_k.weight = nn.Parameter(sharded_norm_k_weight)
                module.norm_k.dim = sliced_heads * head_dim

            old_q, old_k, old_v, old_o = module.q, module.k, module.v, module.o
            module.q = sharded_q
            module.k = sharded_k
            module.v = sharded_v
            module.o = sharded_o
            module.num_heads = sliced_heads
            del old_q, old_k, old_v, old_o
            torch.cuda.empty_cache()
        else:
            self.o_bias = module.o.bias.data.clone() if module.o.bias is not None else None

    def _wait_for_async(self):
        if self.async_handle is not None:
            self.async_handle.wait()
            self.async_handle = None
            if self.pending_output is not None:
                if self.o_bias is not None:
                    self.pending_output = self.pending_output + self.o_bias.view(1, 1, -1)
                self.output_cache = self.pending_output.clone()
                self.pending_output = None

    def forward(self, x, context, context_lens):
        distri_config = self.distri_config
        module = self.module

        self._wait_for_async()

        if self.sliced_heads > 0:
            hidden_states = module.forward(x, context, context_lens)
        else:
            hidden_states = torch.zeros(
                x.shape[0], x.shape[1], self.original_dim,
                device=x.device, dtype=x.dtype,
            )

        if self.counter <= self.warmup_steps or self.output_cache is None:
            dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM,
                          group=distri_config.batch_group, async_op=False)
            if self.o_bias is not None:
                hidden_states = hidden_states + self.o_bias.view(1, 1, -1)
            self.output_cache = hidden_states.clone()
            output = hidden_states
        else:
            self.pending_output = hidden_states.clone()
            self.async_handle = dist.all_reduce(
                self.pending_output, op=dist.ReduceOp.SUM,
                group=distri_config.batch_group, async_op=True
            )
            output = self.output_cache

        self.counter += 1
        return output

    def reset_cache(self):
        self.output_cache = None
        self.async_handle = None
        self.pending_output = None
        self.counter = 0
