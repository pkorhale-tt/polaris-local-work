"""
Matmul Program Config Predictor
Ported from ttnn/operations/matmul/device/config/matmul_program_config.cpp

Usage:
    python matmul_config.py --help
"""

import argparse
import math
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TILE_SIZE = 32

# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────
class MemoryLayout(Enum):
    INTERLEAVED    = "INTERLEAVED"
    HEIGHT_SHARDED = "HEIGHT_SHARDED"
    WIDTH_SHARDED  = "WIDTH_SHARDED"
    BLOCK_SHARDED  = "BLOCK_SHARDED"

class BufferType(Enum):
    DRAM = "DRAM"
    L1   = "L1"

class ShardOrientation(Enum):
    ROW_MAJOR = "ROW_MAJOR"
    COL_MAJOR = "COL_MAJOR"

class DataType(Enum):
    BFLOAT16 = "BFLOAT16"
    FLOAT32  = "FLOAT32"

# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class ShardSpec:
    grid_x:      int
    grid_y:      int
    shape_h:     int              # shard height in elements
    shape_w:     int              # shard width  in elements
    orientation: ShardOrientation

@dataclass
class MemoryConfig:
    layout:      MemoryLayout
    buffer_type: BufferType
    shard:       Optional[ShardSpec] = None

    def is_sharded(self):
        return self.layout != MemoryLayout.INTERLEAVED

@dataclass
class TensorInfo:
    batch: int
    rows:  int                    # M for A, K for B (after transpose logic is applied)
    cols:  int                    # K for A, N for B
    mem:   MemoryConfig
    dtype: DataType = DataType.BFLOAT16

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
SUBBLOCK_HW_CHOICES = [
    (4, 2), (2, 4), (8, 1), (1, 8),
    (7, 1), (1, 7),
    (3, 2), (2, 3), (6, 1), (1, 6),
    (5, 1), (1, 5),
    (2, 2), (4, 1), (1, 4),
    (3, 1), (1, 3),
    (2, 1), (1, 2),
    (1, 1),
]

def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b

def is_narrow_shape(height: int, width: int, all_dram: bool) -> bool:
    NARROW_RATIO = 8
    ratio = height // width if height > width else width // height
    if ratio > NARROW_RATIO:
        return True
    if all_dram:
        return height <= TILE_SIZE or width <= TILE_SIZE
    return False

def get_subblock_sizes(
    m_tiles: int, n_tiles: int, fp32_dest_acc: bool
) -> Tuple[int, int]:
    for (out_h, out_w) in SUBBLOCK_HW_CHOICES:
        if (out_h * out_w) <= 4 or not fp32_dest_acc:
            if m_tiles % out_h == 0 and n_tiles % out_w == 0:
                return (out_h, out_w)
    return (1, 1)

def get_matmul_subblock_params(
    per_core_M: int,
    per_core_N: int,
    m_eq_h_constraint: bool,
    n_eq_w_constraint: bool,
    fp32_dest_acc: bool,
) -> Tuple[int, int]:
    for (out_h, out_w) in SUBBLOCK_HW_CHOICES:
        if fp32_dest_acc and (out_h * out_w) > 4:
            continue
        if n_eq_w_constraint:
            if out_w != per_core_N or out_h != 1:
                continue
        if m_eq_h_constraint:
            if out_h != per_core_M or out_w != 1:
                continue
        if per_core_M % out_h == 0 and per_core_N % out_w == 0:
            return (out_h, out_w)
    return (1, 1)

# ─────────────────────────────────────────────
# Config result classes (plain dicts for readability)
# ─────────────────────────────────────────────
def make_config(name: str, **kwargs) -> dict:
    return {"config_type": name, **kwargs}

# ─────────────────────────────────────────────
# Core logic (ported from C++)
# ─────────────────────────────────────────────

def create_matmul_1d_systolic_array_program_config(
    batch_a: int,
    M: int, K: int, N: int,
    core_x: int, core_y: int,
    layout_a: MemoryLayout,
    fp32_dest_acc: bool,
) -> dict:
    batch_and_m_tiles = (batch_a * M) // TILE_SIZE
    k_tiles           = K // TILE_SIZE
    n_tiles           = N // TILE_SIZE
    num_cores         = core_x * core_y

    is_tall = batch_and_m_tiles > n_tiles
    if layout_a == MemoryLayout.HEIGHT_SHARDED:
        is_tall = True
    elif layout_a == MemoryLayout.WIDTH_SHARDED:
        is_tall = False

    if is_tall:
        per_core_M = div_up(batch_and_m_tiles, num_cores)
        k_tiles_pc = div_up(k_tiles, num_cores)
        per_core_N = n_tiles
    else:
        per_core_M = batch_and_m_tiles
        k_tiles_pc = div_up(k_tiles, num_cores)
        per_core_N = div_up(n_tiles, num_cores)

    while k_tiles % k_tiles_pc != 0:
        k_tiles_pc -= 1

    out_h, out_w = get_subblock_sizes(per_core_M, per_core_N, fp32_dest_acc)

    return make_config(
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        grid=(core_x, core_y),
        in0_block_w=k_tiles_pc,
        out_subblock_h=out_h,
        out_subblock_w=out_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        mcast_in0=not is_tall,
        fuse_batch=True,
    )


def get_mcast_1d_config(
    M: int, K: int, N: int,
    mcast_in0: bool,
    grid_x: int, grid_y: int,
    fp32_dest_acc: bool,
    out_sharded: bool,
    in0_tile_h: int = TILE_SIZE,
    in0_tile_w: int = TILE_SIZE,
    in1_tile_w: int = TILE_SIZE,
) -> dict:
    if mcast_in0:
        per_core_M = M // in0_tile_h
        per_core_N = div_up(div_up(N, grid_x * grid_y), in1_tile_w)
    else:
        per_core_M = div_up(div_up(M, grid_x * grid_y), in0_tile_h)
        per_core_N = N // in1_tile_w

    in0_block_w = 2 if (K // in0_tile_w) % 2 == 0 else 1

    per_core_N_eq_w = out_sharded and not mcast_in0
    per_core_M_eq_h = out_sharded and mcast_in0

    out_h, out_w = get_matmul_subblock_params(
        per_core_M, per_core_N, per_core_M_eq_h, per_core_N_eq_w, fp32_dest_acc
    )

    return make_config(
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        grid=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_h,
        out_subblock_w=out_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        mcast_in0=mcast_in0,
        fuse_batch=False,
    )


def create_matmul_program_config(
    tensor_a: TensorInfo,
    tensor_b: TensorInfo,
    transpose_a: bool,
    transpose_b: bool,
    core_x: int, core_y: int,
    fp32_dest_acc: bool,
    output_mem: MemoryConfig,
) -> dict:
    batch_a = tensor_a.batch
    batch_b = tensor_b.batch
    M = tensor_a.rows
    K = tensor_a.cols
    N = tensor_b.cols

    input_b_is_batched   = batch_b > 1
    any_size_within_tile = K <= TILE_SIZE or M <= TILE_SIZE or N <= TILE_SIZE
    a_is_sharded         = tensor_a.mem.is_sharded()
    a_layout             = tensor_a.mem.layout
    a_is_block_sharded   = a_layout == MemoryLayout.BLOCK_SHARDED

    # ── batched B path ──────────────────────────────────────────────
    if input_b_is_batched:
        if not a_is_sharded and not tensor_b.mem.is_sharded():
            m_tiles = div_up(M, TILE_SIZE)
            n_tiles = div_up(N, TILE_SIZE)
            k_tiles = 1
        elif a_is_sharded:
            shard = tensor_a.mem.shard
            m_tiles = shard.shape_h // TILE_SIZE
            n_tiles = tensor_b.cols // TILE_SIZE
            k_tiles = shard.shape_w // TILE_SIZE
        else:
            shard   = tensor_b.mem.shard
            m_tiles = div_up(M, TILE_SIZE)
            n_tiles = shard.shape_w // TILE_SIZE
            k_tiles = 1

        out_h, out_w = get_subblock_sizes(m_tiles, n_tiles, fp32_dest_acc)
        return make_config(
            "MatmulMultiCoreReuseProgramConfig",
            grid=(core_x, core_y),
            in0_block_w=k_tiles,
            out_subblock_h=out_h,
            out_subblock_w=out_w,
            per_core_M=m_tiles,
            per_core_N=n_tiles,
        )

    # ── narrow / small shape → 1D systolic ─────────────────────────
    height = batch_a * M
    width  = N
    if is_narrow_shape(height, width, False) or any_size_within_tile:
        if not a_is_block_sharded:
            return create_matmul_1d_systolic_array_program_config(
                batch_a, M, K, N, core_x, core_y, a_layout, fp32_dest_acc
            )

    # ── interleaved A ───────────────────────────────────────────────
    if not a_is_sharded:
        m_tiles = math.ceil(((batch_a * M) / TILE_SIZE) / core_y)
        n_tiles = math.ceil((N / TILE_SIZE) / core_x)
        k_tiles = 4
        while (K // TILE_SIZE) % k_tiles != 0:
            k_tiles -= 1
    else:
        if not a_is_block_sharded:
            return create_matmul_1d_systolic_array_program_config(
                batch_a, M, K, N, core_x, core_y, a_layout, fp32_dest_acc
            )
        shard  = tensor_a.mem.shard
        k      = K // TILE_SIZE
        n      = N // TILE_SIZE
        m_tiles = shard.shape_h // TILE_SIZE
        n_tiles = (n * shard.shape_w) // (k * TILE_SIZE)
        k_tiles = math.gcd(shard.shape_w // TILE_SIZE, k)

    n_tiles = max(n_tiles, 1)

    out_h, out_w = get_subblock_sizes(m_tiles, n_tiles, fp32_dest_acc)
    transpose_mcast = (
        a_is_block_sharded
        and tensor_a.mem.shard is not None
        and tensor_a.mem.shard.orientation == ShardOrientation.COL_MAJOR
    )
    if out_w != n_tiles:
        out_h = 1

    return make_config(
        "MatmulMultiCoreReuseMultiCastProgramConfig",
        grid=(core_x, core_y),
        in0_block_w=k_tiles,
        out_subblock_h=out_h,
        out_subblock_w=out_w,
        per_core_M=m_tiles,
        per_core_N=n_tiles,
        transpose_mcast=transpose_mcast,
    )


def create_simple_matmul_program_config(
    tensor_a: TensorInfo,
    tensor_b: TensorInfo,
    transpose_a: bool,
    transpose_b: bool,
    output_mem: MemoryConfig,
    fp32_dest_acc: bool,
    core_x: int,
    core_y: int,
) -> dict:
    M  = tensor_a.rows
    K  = tensor_a.cols
    N  = tensor_b.cols
    Mt = M // TILE_SIZE
    Kt = K // TILE_SIZE
    Nt = N // TILE_SIZE

    mem_a = tensor_a.mem
    mem_b = tensor_b.mem

    all_interleaved = (
        mem_a.layout == MemoryLayout.INTERLEAVED
        and mem_b.layout == MemoryLayout.INTERLEAVED
        and output_mem.layout == MemoryLayout.INTERLEAVED
    )
    all_dram = (
        mem_a.buffer_type == BufferType.DRAM
        and mem_b.buffer_type == BufferType.DRAM
        and output_mem.buffer_type == BufferType.DRAM
    )
    all_dram_interleaved = all_dram and all_interleaved

    height = tensor_a.rows  # padded
    width  = tensor_b.cols
    narrow = is_narrow_shape(height, width, all_dram)

    is_wide = is_tall = False
    if all_interleaved and narrow:
        is_wide = width > height
        is_tall = not is_wide

    per_core_M = per_core_N = 8  # simplified (real code checks L1)
    in0_block_w = 2

    num_blocks_y = div_up(Mt, per_core_M)
    num_blocks_x = div_up(Nt, per_core_N)

    if (output_mem.is_sharded() or num_blocks_y > 1 or num_blocks_x > 1) and Kt % in0_block_w != 0:
        in0_block_w = 1

    # ── detect 1D grid for BLOCK_SHARDED output ─────────────────────
    block_1d_col = block_1d_row = False
    if (
        output_mem.is_sharded()
        and output_mem.layout == MemoryLayout.BLOCK_SHARDED
        and output_mem.shard is not None
    ):
        shard = output_mem.shard
        is_single = shard.grid_x == 1 and shard.grid_y == 1
        block_1d_col = not is_single and shard.grid_x == 1
        block_1d_row = not is_single and shard.grid_y == 1

        batch_b = tensor_b.batch
        if (block_1d_col or block_1d_row) and batch_b > 1:
            raise ValueError(
                "BLOCK_SHARDED on 1D grid not supported when both tensors are batched."
            )

    num_blocks_x_fits = num_blocks_x <= core_x
    num_blocks_y_fits = num_blocks_y <= core_y
    # Mirrors the C++ get_core_range lambda:
    # returns (0,0) when both block counts are 1 (single-block case
    # handled by all_dram_interleaved / use_mcast_2d path instead)
    if (num_blocks_y == 1 and num_blocks_x == 1) or \
       not num_blocks_y_fits or not num_blocks_x_fits:
        core_range_y = 0
        core_range_x = 0
    else:
        core_range_y = num_blocks_y
        core_range_x = num_blocks_x

    fits = num_blocks_x * num_blocks_y <= core_x * core_y and Kt % in0_block_w == 0

    if all_dram_interleaved or fits:
        use_1d_in0 = is_wide or (
            core_range_y == 0
            and output_mem.is_sharded()
            and (
                output_mem.layout == MemoryLayout.WIDTH_SHARDED
                or block_1d_row
            )
        )
        use_1d_in1 = is_tall or (
            core_range_y == 0
            and output_mem.is_sharded()
            and (
                output_mem.layout == MemoryLayout.HEIGHT_SHARDED
                or block_1d_col
            )
        )
        use_2d = all_dram_interleaved or (
            core_range_y == 0
            and output_mem.is_sharded()
            and output_mem.layout == MemoryLayout.BLOCK_SHARDED
            and not block_1d_col
            and not block_1d_row
        )

        if core_range_y == 1 or use_1d_in0:
            return get_mcast_1d_config(M, K, N, mcast_in0=True,
                                       grid_x=core_x, grid_y=core_y,
                                       fp32_dest_acc=fp32_dest_acc,
                                       out_sharded=False)
        if core_range_x == 1 or use_1d_in1:
            return get_mcast_1d_config(M, K, N, mcast_in0=False,
                                       grid_x=core_x, grid_y=core_y,
                                       fp32_dest_acc=fp32_dest_acc,
                                       out_sharded=False)
        if (core_range_y > 0 and num_blocks_x_fits and num_blocks_y_fits) or use_2d:
            out_subblock_h = 4
            out_subblock_w = 2
            if out_subblock_w != per_core_N:
                out_subblock_h = 1
            transpose_mcast = (
                mem_a.layout == MemoryLayout.BLOCK_SHARDED
                and mem_a.shard is not None
                and mem_a.shard.orientation == ShardOrientation.COL_MAJOR
            )
            if all_dram_interleaved:
                per_core_M = div_up(Mt, core_y)
                per_core_N = div_up(Nt, core_x)
                in0_block_w = Kt // core_x if Kt % core_x == 0 else 1
                out_subblock_h, out_subblock_w = get_matmul_subblock_params(
                    per_core_M, per_core_N, False, False, fp32_dest_acc
                )
            return make_config(
                "MatmulMultiCoreReuseMultiCastProgramConfig",
                grid=(core_x, core_y),
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
                transpose_mcast=transpose_mcast,
            )

    return make_config("MatmulMultiCoreProgramConfig")


def generate_matmul_program_config(
    tensor_a: TensorInfo,
    tensor_b: TensorInfo,
    transpose_a: bool,
    transpose_b: bool,
    output_mem: MemoryConfig,
    fp32_dest_acc: bool,
    user_core_x: Optional[int],
    user_core_y: Optional[int],
    user_run_batched: bool,
    device_core_x: int,
    device_core_y: int,
) -> dict:
    has_user_grid = user_core_x is not None and user_core_y is not None
    core_x = user_core_x if has_user_grid else device_core_x
    core_y = user_core_y if has_user_grid else device_core_y

    if has_user_grid or not tensor_a.mem.is_sharded():
        if has_user_grid:
            return create_matmul_program_config(
                tensor_a, tensor_b, transpose_a, transpose_b,
                core_x, core_y, fp32_dest_acc, output_mem
            )
        return create_simple_matmul_program_config(
            tensor_a, tensor_b, transpose_a, transpose_b,
            output_mem, fp32_dest_acc, device_core_x, device_core_y
        )

    # sharded A, no user grid
    bmm = user_run_batched

    # get_matmul_program_config logic (matmul=True path only, simplified)
    a_layout = tensor_a.mem.layout
    shard_a  = tensor_a.mem.shard
    grid_x   = shard_a.grid_x if shard_a else device_core_x
    grid_y   = shard_a.grid_y if shard_a else device_core_y

    M  = tensor_a.batch * tensor_a.rows
    K  = tensor_a.cols
    N  = tensor_b.cols
    Mt = M // TILE_SIZE
    Kt = K // TILE_SIZE
    Nt = N // TILE_SIZE

    if not bmm:
        # matmul = True
        if a_layout in (MemoryLayout.WIDTH_SHARDED, MemoryLayout.HEIGHT_SHARDED):
            mcast_in0 = a_layout == MemoryLayout.WIDTH_SHARDED
            if mcast_in0:
                per_core_M = Mt
                per_core_N = div_up(Nt, shard_a.grid_x * shard_a.grid_y)
                in0_block_w = math.gcd(shard_a.shape_w // TILE_SIZE, Kt)
            else:
                per_core_M = shard_a.shape_h // TILE_SIZE
                per_core_N = Nt
                in0_block_w = Kt
            per_core_N_eq_w = output_mem.is_sharded()
            out_h, out_w = get_matmul_subblock_params(
                per_core_M, per_core_N, False, per_core_N_eq_w, fp32_dest_acc
            )
            return make_config(
                "MatmulMultiCoreReuseMultiCast1DProgramConfig",
                grid=(grid_x, grid_y),
                in0_block_w=in0_block_w,
                out_subblock_h=out_h,
                out_subblock_w=out_w,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
                mcast_in0=mcast_in0,
                fuse_batch=True,
            )

        if a_layout == MemoryLayout.BLOCK_SHARDED:
            transpose_mcast = shard_a.orientation == ShardOrientation.COL_MAJOR if shard_a else False
            vx = grid_y if transpose_mcast else grid_x
            vy = grid_x if transpose_mcast else grid_y
            per_core_M = div_up(Mt, vy)
            per_core_N = div_up(Nt, vx)
            in0_block_w = math.gcd(shard_a.shape_w // TILE_SIZE, Kt) if shard_a else 1
            per_core_N_eq_w = output_mem.is_sharded()
            out_h, out_w = get_matmul_subblock_params(
                per_core_M, per_core_N, False, per_core_N_eq_w, fp32_dest_acc
            )
            return make_config(
                "MatmulMultiCoreReuseMultiCastProgramConfig",
                grid=(grid_x, grid_y),
                in0_block_w=in0_block_w,
                out_subblock_h=out_h,
                out_subblock_w=out_w,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
                transpose_mcast=transpose_mcast,
            )
    else:
        # bmm path
        per_core_M = shard_a.shape_h // TILE_SIZE if shard_a else div_up(Mt, grid_y)
        per_core_N = Nt
        in0_block_w = shard_a.shape_w // TILE_SIZE if shard_a else Kt
        per_core_N_eq_w = output_mem.is_sharded()
        out_h, out_w = get_matmul_subblock_params(
            per_core_M, per_core_N, False, per_core_N_eq_w, fp32_dest_acc
        )
        return make_config(
            "MatmulMultiCoreReuseProgramConfig",
            grid=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_h,
            out_subblock_w=out_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
        )

    # fallback
    return create_matmul_program_config(
        tensor_a, tensor_b, transpose_a, transpose_b,
        core_x, core_y, fp32_dest_acc, output_mem
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Predict TT-NN Matmul Program Config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Tensor A ──────────────────────────────
    g_a = p.add_argument_group("Tensor A  (input)")
    g_a.add_argument("--a-batch",   type=int, default=1,   help="Batch size of A")
    g_a.add_argument("--a-M",       type=int, required=True, help="Row dim of A (M)")
    g_a.add_argument("--a-K",       type=int, required=True, help="Col dim of A (K)")
    g_a.add_argument("--a-layout",  choices=[e.value for e in MemoryLayout],
                                    default="INTERLEAVED", help="Memory layout of A")
    g_a.add_argument("--a-buffer",  choices=[e.value for e in BufferType],
                                    default="DRAM",        help="Buffer type of A")
    g_a.add_argument("--a-shard-h",   type=int, default=None, help="Shard height  (if sharded)")
    g_a.add_argument("--a-shard-w",   type=int, default=None, help="Shard width   (if sharded)")
    g_a.add_argument("--a-shard-gx",  type=int, default=None, help="Shard grid X  (if sharded)")
    g_a.add_argument("--a-shard-gy",  type=int, default=None, help="Shard grid Y  (if sharded)")
    g_a.add_argument("--a-orient",  choices=[e.value for e in ShardOrientation],
                                    default="ROW_MAJOR",  help="Shard orientation of A")
    g_a.add_argument("--transpose-a", action="store_true", help="Transpose A before matmul")

    # ── Tensor B ──────────────────────────────
    g_b = p.add_argument_group("Tensor B  (weights)")
    g_b.add_argument("--b-batch",   type=int, default=1,   help="Batch size of B")
    g_b.add_argument("--b-K",       type=int, required=True, help="Row dim of B (K, must equal A-K)")
    g_b.add_argument("--b-N",       type=int, required=True, help="Col dim of B (N)")
    g_b.add_argument("--b-layout",  choices=[e.value for e in MemoryLayout],
                                    default="INTERLEAVED", help="Memory layout of B")
    g_b.add_argument("--b-buffer",  choices=[e.value for e in BufferType],
                                    default="DRAM",        help="Buffer type of B")
    g_b.add_argument("--b-shard-h",   type=int, default=None)
    g_b.add_argument("--b-shard-w",   type=int, default=None)
    g_b.add_argument("--b-shard-gx",  type=int, default=None)
    g_b.add_argument("--b-shard-gy",  type=int, default=None)
    g_b.add_argument("--transpose-b", action="store_true", help="Transpose B before matmul")

    # ── Output memory ─────────────────────────
    g_o = p.add_argument_group("Output memory config")
    g_o.add_argument("--out-layout", choices=[e.value for e in MemoryLayout],
                                     default="INTERLEAVED")
    g_o.add_argument("--out-buffer", choices=[e.value for e in BufferType],
                                     default="DRAM")
    g_o.add_argument("--out-shard-h",  type=int, default=None)
    g_o.add_argument("--out-shard-w",  type=int, default=None)
    g_o.add_argument("--out-shard-gx", type=int, default=None)
    g_o.add_argument("--out-shard-gy", type=int, default=None)

    # ── Device / compute ──────────────────────
    g_d = p.add_argument_group("Device and compute")
    g_d.add_argument("--device-core-x", type=int, default=8,  help="Device compute grid X")
    g_d.add_argument("--device-core-y", type=int, default=8,  help="Device compute grid Y")
    g_d.add_argument("--user-core-x",   type=int, default=None, help="Override compute grid X")
    g_d.add_argument("--user-core-y",   type=int, default=None, help="Override compute grid Y")
    g_d.add_argument("--fp32-dest-acc",  action="store_true",   help="Enable FP32 dest accumulation")
    g_d.add_argument("--run-batched",    action="store_true",   help="Run as batched BMM")
    g_d.add_argument("--output-dtype",   choices=[e.value for e in DataType],
                                         default="BFLOAT16")

    return p.parse_args()


def build_shard(h, w, gx, gy, orient_str) -> Optional[ShardSpec]:
    if h is None:
        return None
    return ShardSpec(
        grid_x=gx, grid_y=gy,
        shape_h=h, shape_w=w,
        orientation=ShardOrientation(orient_str),
    )


def print_config(cfg: dict):
    print("\n" + "=" * 52)
    print(f"  CONFIG TYPE : {cfg['config_type']}")
    print("=" * 52)
    for k, v in cfg.items():
        if k == "config_type":
            continue
        print(f"  {k:<22} = {v}")
    print("=" * 52)


def main():
    args = parse_args()

    shard_a = build_shard(
        args.a_shard_h, args.a_shard_w,
        args.a_shard_gx, args.a_shard_gy,
        args.a_orient,
    )
    shard_b = build_shard(
        args.b_shard_h, args.b_shard_w,
        args.b_shard_gx, args.b_shard_gy,
        "ROW_MAJOR",
    )
    shard_out = build_shard(
        args.out_shard_h, args.out_shard_w,
        args.out_shard_gx, args.out_shard_gy,
        "ROW_MAJOR",
    )

    tensor_a = TensorInfo(
        batch=args.a_batch, rows=args.a_M, cols=args.a_K,
        mem=MemoryConfig(MemoryLayout(args.a_layout), BufferType(args.a_buffer), shard_a),
    )
    tensor_b = TensorInfo(
        batch=args.b_batch, rows=args.b_K, cols=args.b_N,
        mem=MemoryConfig(MemoryLayout(args.b_layout), BufferType(args.b_buffer), shard_b),
    )
    output_mem = MemoryConfig(
        MemoryLayout(args.out_layout), BufferType(args.out_buffer), shard_out
    )

    print(f"\nInput Summary")
    print(f"  A : batch={tensor_a.batch}  M={tensor_a.rows}  K={tensor_a.cols}"
          f"  layout={tensor_a.mem.layout.value}  buf={tensor_a.mem.buffer_type.value}"
          f"  transpose={args.transpose_a}")
    print(f"  B : batch={tensor_b.batch}  K={tensor_b.rows}  N={tensor_b.cols}"
          f"  layout={tensor_b.mem.layout.value}  buf={tensor_b.mem.buffer_type.value}"
          f"  transpose={args.transpose_b}")
    print(f"  Out layout={output_mem.layout.value}  buf={output_mem.buffer_type.value}")
    print(f"  Device grid=({args.device_core_x},{args.device_core_y})"
          f"  fp32_dest_acc={args.fp32_dest_acc}  batched={args.run_batched}")

    try:
        cfg = generate_matmul_program_config(
            tensor_a, tensor_b,
            args.transpose_a, args.transpose_b,
            output_mem,
            args.fp32_dest_acc,
            args.user_core_x, args.user_core_y,
            args.run_batched,
            args.device_core_x, args.device_core_y,
        )
        print_config(cfg)
    except Exception as e:
        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()