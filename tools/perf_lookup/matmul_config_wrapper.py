"""
Matmul Program Config Wrapper
------------------------------
A clean Python API on top of matmul_config.py.

Instead of passing CLI args, you call predict_config() directly with
plain Python values.

Usage:
    from matmul_config_wrapper import predict_config, TensorSpec, MemSpec

    cfg = predict_config(
        a=TensorSpec(batch=1, rows=512, cols=256),
        b=TensorSpec(batch=1, rows=256, cols=512),
    )
    print(cfg)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from matmul_config import (
    MemoryLayout, BufferType, ShardOrientation, DataType,
    ShardSpec, MemoryConfig, TensorInfo,
    generate_matmul_program_config,
)
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# User-facing dataclasses  (simpler than the internal ones)
# ─────────────────────────────────────────────────────────────────

@dataclass
class ShardInfo:
    """Describe how a tensor is sharded across cores."""
    grid_x:      int                        # number of cores along X
    grid_y:      int                        # number of cores along Y
    shape_h:     int                        # shard height in elements
    shape_w:     int                        # shard width  in elements
    orientation: str = "ROW_MAJOR"          # "ROW_MAJOR" or "COL_MAJOR"


@dataclass
class MemSpec:
    """Memory placement of a tensor."""
    layout:      str = "INTERLEAVED"        # INTERLEAVED / HEIGHT_SHARDED /
                                            # WIDTH_SHARDED / BLOCK_SHARDED
    buffer_type: str = "DRAM"              # DRAM / L1
    shard:       Optional[ShardInfo] = None


@dataclass
class TensorSpec:
    """
    Describes one tensor operand.

    For A  →  shape is  (batch, M, K)
    For B  →  shape is  (batch, K, N)

    batch = product of all dims except the last two.
    So a 4D tensor (2, 4, 128, 256) → batch=8, rows=128, cols=256
    """
    rows:      int                          # M for A,  K for B
    cols:      int                          # K for A,  N for B
    batch:     int       = 1
    mem:       MemSpec   = field(default_factory=MemSpec)
    transpose: bool      = False


@dataclass
class DeviceSpec:
    """Compute grid of the device."""
    core_x: int = 8
    core_y: int = 8


# ─────────────────────────────────────────────────────────────────
# Internal conversion helpers
# ─────────────────────────────────────────────────────────────────

def _to_shard_spec(s: Optional[ShardInfo]) -> Optional[ShardSpec]:
    if s is None:
        return None
    return ShardSpec(
        grid_x=s.grid_x,
        grid_y=s.grid_y,
        shape_h=s.shape_h,
        shape_w=s.shape_w,
        orientation=ShardOrientation(s.orientation),
    )


def _to_mem_config(m: MemSpec) -> MemoryConfig:
    return MemoryConfig(
        layout=MemoryLayout(m.layout),
        buffer_type=BufferType(m.buffer_type),
        shard=_to_shard_spec(m.shard),
    )


def _to_tensor_info(t: TensorSpec) -> TensorInfo:
    return TensorInfo(
        batch=t.batch,
        rows=t.rows,
        cols=t.cols,
        mem=_to_mem_config(t.mem),
    )


# ─────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────

def predict_config(
    a: TensorSpec,
    b: TensorSpec,
    output_mem:       Optional[MemSpec]   = None,
    device:           Optional[DeviceSpec] = None,
    user_core_x:      Optional[int]       = None,
    user_core_y:      Optional[int]       = None,
    fp32_dest_acc:    bool                = False,
    user_run_batched: bool                = False,
) -> dict:
    """
    Predict the MatmulProgramConfig for a given set of inputs.

    Parameters
    ----------
    a                : TensorSpec for input A  (batch, M, K)
    b                : TensorSpec for input B  (batch, K, N)
    output_mem       : MemSpec for output tensor  (default: DRAM INTERLEAVED)
    device           : DeviceSpec  (default: 8x8 grid)
    user_core_x/y    : optional override for compute grid
    fp32_dest_acc    : enable FP32 destination accumulation
    user_run_batched : treat as batched BMM (not matmul)

    Returns
    -------
    dict with 'config_type' and all config parameters
    """
    if output_mem is None:
        output_mem = MemSpec()
    if device is None:
        device = DeviceSpec()

    tensor_a    = _to_tensor_info(a)
    tensor_b    = _to_tensor_info(b)
    output_mem_ = _to_mem_config(output_mem)

    return generate_matmul_program_config(
        tensor_a, tensor_b,
        a.transpose, b.transpose,
        output_mem_,
        fp32_dest_acc,
        user_core_x, user_core_y,
        user_run_batched,
        device.core_x, device.core_y,
    )


def pretty_print(cfg: dict):
    """Print a config dict in a readable format."""
    print("\n" + "=" * 55)
    print(f"  CONFIG : {cfg['config_type']}")
    print("=" * 55)
    for k, v in cfg.items():
        if k == "config_type":
            continue
        print(f"  {k:<24} = {v}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────
# Example usage  (run this file directly to see all demos)
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n─── Example 1: Simple interleaved DRAM matmul ───")
    cfg = predict_config(
        a=TensorSpec(batch=1, rows=512, cols=256),
        b=TensorSpec(batch=1, rows=256, cols=512),
    )
    pretty_print(cfg)

    print("─── Example 2: 4D tensor A, B is 2D (fuse_batch) ───")
    # A is (2, 4, 128, 256)  →  batch=8, rows=128, cols=256
    cfg = predict_config(
        a=TensorSpec(batch=8, rows=128, cols=256),
        b=TensorSpec(batch=1, rows=256, cols=512),
    )
    pretty_print(cfg)

    print("─── Example 3: Both A and B batched ───")
    cfg = predict_config(
        a=TensorSpec(batch=8, rows=128, cols=256),
        b=TensorSpec(batch=8, rows=256, cols=512),
    )
    pretty_print(cfg)

    print("─── Example 4: Narrow/tall shape → 1D config ───")
    cfg = predict_config(
        a=TensorSpec(batch=1, rows=4096, cols=256),
        b=TensorSpec(batch=1, rows=256,  cols=256),
    )
    pretty_print(cfg)

    print("─── Example 5: Height sharded A on L1 ───")
    cfg = predict_config(
        a=TensorSpec(
            batch=1, rows=512, cols=256,
            mem=MemSpec(
                layout="HEIGHT_SHARDED",
                buffer_type="L1",
                shard=ShardInfo(grid_x=8, grid_y=1, shape_h=64, shape_w=256),
            ),
        ),
        b=TensorSpec(batch=1, rows=256, cols=512),
    )
    pretty_print(cfg)

    print("─── Example 6: Block sharded A → 2D mcast ───")
    cfg = predict_config(
        a=TensorSpec(
            batch=1, rows=512, cols=256,
            mem=MemSpec(
                layout="BLOCK_SHARDED",
                buffer_type="L1",
                shard=ShardInfo(grid_x=4, grid_y=4, shape_h=128, shape_w=64),
            ),
        ),
        b=TensorSpec(batch=1, rows=256, cols=512),
    )
    pretty_print(cfg)

    print("─── Example 7: User override core grid ───")
    cfg = predict_config(
        a=TensorSpec(batch=1, rows=512, cols=256),
        b=TensorSpec(batch=1, rows=256, cols=512),
        user_core_x=4,
        user_core_y=4,
    )
    pretty_print(cfg)

    print("─── Example 8: Width sharded A → 1D mcast_in0=True ───")
    cfg = predict_config(
        a=TensorSpec(
            batch=1, rows=256, cols=512,
            mem=MemSpec(
                layout="WIDTH_SHARDED",
                buffer_type="L1",
                shard=ShardInfo(grid_x=8, grid_y=1, shape_h=256, shape_w=64),
            ),
        ),
        b=TensorSpec(batch=1, rows=512, cols=512),
    )
    pretty_print(cfg)

    print("─── Example 9: FP32 dest accumulation ───")
    cfg = predict_config(
        a=TensorSpec(batch=1, rows=512, cols=256),
        b=TensorSpec(batch=1, rows=256, cols=512),
        fp32_dest_acc=True,
    )
    pretty_print(cfg)

    print("─── Example 10: BMM (user_run_batched=True) ───")
    cfg = predict_config(
        a=TensorSpec(
            batch=4, rows=128, cols=256,
            mem=MemSpec(
                layout="HEIGHT_SHARDED",
                buffer_type="L1",
                shard=ShardInfo(grid_x=4, grid_y=1, shape_h=128, shape_w=256),
            ),
        ),
        b=TensorSpec(batch=1, rows=256, cols=512),
        user_run_batched=True,
    )
    pretty_print(cfg)