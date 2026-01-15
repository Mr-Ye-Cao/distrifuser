# Copyright 2024 DistriFuser Authors. All rights reserved.

from .utils import DistriConfig

__all__ = [
    "DistriConfig",
]

# All pipelines use lazy import to avoid dependency issues
def __getattr__(name):
    # UNet pipelines (existing)
    if name == "DistriSDXLPipeline":
        from .pipelines import DistriSDXLPipeline
        return DistriSDXLPipeline
    elif name == "DistriSDPipeline":
        from .pipelines import DistriSDPipeline
        return DistriSDPipeline
    # Wan2.1 DiT support (new)
    elif name == "DistriWanT2VPipeline":
        from .pipelines_wan.wan_pipeline import DistriWanT2VPipeline
        return DistriWanT2VPipeline
    elif name == "DistriWanDiT":
        from .models.wan.distri_wan_dit import DistriWanDiT
        return DistriWanDiT
    elif name == "wrap_wan_model":
        from .pipelines_wan.wan_pipeline import wrap_wan_model
        return wrap_wan_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
