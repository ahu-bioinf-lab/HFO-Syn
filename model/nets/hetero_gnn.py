"""兼容入口：实际实现见 hetero_stack_enc.py。"""
from .hetero_stack_enc import (
    DualFusedHeteroGraphEncoder,
    HeteroGraphEncoder,
    build_hetero_encoder_if_enabled,
    resolve_entity_keys,
)

__all__ = [
    "DualFusedHeteroGraphEncoder",
    "HeteroGraphEncoder",
    "build_hetero_encoder_if_enabled",
    "resolve_entity_keys",
]
