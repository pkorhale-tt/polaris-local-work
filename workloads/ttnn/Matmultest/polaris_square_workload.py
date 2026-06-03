# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time

sys.path.append(".")

IS_POLARIS = os.getenv("IRD_ARCH_NAME", "") == ""

from loguru import logger

if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
    import ttsim.front.ttnn.minitorch_shim as torch
    from ttsim.front.ttnn.device import set_default_device
    from ttsim.front.ttnn.tensor import ttnn_random

    torch_random = ttnn_random
else:
    import torch
    import ttnn

    def torch_random(shape, low=-1.0, high=1.0, dtype=torch.bfloat16):
        return (low + (high - low) * torch.rand(shape)).to(dtype)


DTYPE_MAP = {
    "BFLOAT16": ttnn.bfloat16,
    "BFLOAT8_B": ttnn.bfloat8_b,
    "FLOAT32": ttnn.float32,
}
if hasattr(ttnn, "bfloat4_b"):
    DTYPE_MAP["BFLOAT4_B"] = ttnn.bfloat4_b

MATH_FIDELITY_MAP = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


def get_dtype(dtype_str: str):
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype on this backend: {dtype_str}")
    return DTYPE_MAP[dtype_str]


def make_mem_config(layout: str, buffer_type: str):
    buf = ttnn.BufferType.L1 if buffer_type == "L1" else ttnn.BufferType.DRAM

    layout_map = {
        "INTERLEAVED": ttnn.TensorMemoryLayout.INTERLEAVED,
        "HEIGHT_SHARDED": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "WIDTH_SHARDED": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "BLOCK_SHARDED": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }

    mem_layout = layout_map.get(layout, ttnn.TensorMemoryLayout.INTERLEAVED)

    return ttnn.MemoryConfig(
        memory_layout=mem_layout,
        buffer_type=buf,
    )


def make_compute_kernel_config(compute: dict):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=MATH_FIDELITY_MAP.get(compute.get("math_fidelity", "HiFi2"), ttnn.MathFidelity.HiFi2),
        math_approx_mode=compute.get("math_approx_mode", False),
        fp32_dest_acc_en=compute.get("fp32_dest_acc_en", False),
        packer_l1_acc=compute.get("packer_l1_acc", True),
    )


def make_tensor(device, shape_4d, dtype_str, mem_layout, buf_type, verbose=False):
    dtype = get_dtype(dtype_str)
    mem_cfg = make_mem_config(mem_layout, buf_type)

    if verbose:
        logger.info(f"Creating tensor shape={shape_4d} dtype={dtype_str} mem={mem_layout}/{buf_type}")

    host_t = torch_random(shape_4d, -1.0, 1.0, dtype=torch.bfloat16)
    tt_t = ttnn.from_torch(host_t, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tt_t = ttnn.to_device(tt_t, device, memory_config=mem_cfg)
    return tt_t


def make_op(op_id: int, batch: int, size: int, fid: str):
    return {
        "op_id": op_id,
        "description": f"Matmul M={size} K={size} N={size} batch={batch} fid={fid}",
        "tensor_a": {
            "batch": batch,
            "M": size,
            "K": size,
            "dtype": "BFLOAT16",
            "memory_layout": "INTERLEAVED",
            "buffer_type": "DRAM",
            "transpose": False,
        },
        "tensor_b": {
            "batch": 1,
            "K": size,
            "N": size,
            "dtype": "BFLOAT16",
            "memory_layout": "INTERLEAVED",
            "buffer_type": "DRAM",
            "transpose": False,
        },
        "output": {
            "dtype": "BFLOAT16",
            "memory_layout": "INTERLEAVED",
            "buffer_type": "DRAM",
        },
        "compute": {
            "math_fidelity": fid,
            "fp32_dest_acc_en": False,
            "packer_l1_acc": True,
            "math_approx_mode": False,
        },
        "flags": {
            "user_run_batched": False,
        },
    }


def build_square_sweep_ops(sizes, batches, fids):
    ops = []
    op_id = 1
    for batch in batches:
        for size in sizes:
            for fid in fids:
                ops.append(make_op(op_id, batch, size, fid))
                op_id += 1
    return ops


def parse_csv_ints(s: str):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_csv_strs(s: str):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def run_op(device, op: dict, warmup: int, runs: int, verbose: bool = False) -> dict:
    op_id = op["op_id"]
    a_cfg = op["tensor_a"]
    b_cfg = op["tensor_b"]
    o_cfg = op["output"]
    comp = op["compute"]

    if a_cfg.get("transpose", False) or b_cfg.get("transpose", False):
        raise NotImplementedError("transpose=true is not supported in this runner")

    batch_a = int(a_cfg["batch"])
    m_dim = int(a_cfg["M"])
    k_dim = int(a_cfg["K"])
    batch_b = int(b_cfg["batch"])
    n_dim = int(b_cfg["N"])

    shape_a = (1, batch_a, m_dim, k_dim)
    shape_b = (1, batch_b, k_dim, n_dim)

    compute_kernel_cfg = make_compute_kernel_config(comp)
    output_mem_cfg = make_mem_config(o_cfg["memory_layout"], o_cfg["buffer_type"])
    output_dtype = get_dtype(o_cfg["dtype"])

    logger.info(f"{'─'*70}")
    logger.info(f"Op {op_id}: {op['description']}")
    logger.info(f"Shape A: {shape_a}")
    logger.info(f"Shape B: {shape_b}")

    tt_a = None
    tt_b = None
    out = None

    try:
        tt_a = make_tensor(device, shape_a, a_cfg["dtype"], a_cfg["memory_layout"], a_cfg["buffer_type"], verbose)
        tt_b = make_tensor(device, shape_b, b_cfg["dtype"], b_cfg["memory_layout"], b_cfg["buffer_type"], verbose)

        for _ in range(warmup):
            out = ttnn.matmul(
                tt_a,
                tt_b,
                memory_config=output_mem_cfg,
                dtype=output_dtype,
                compute_kernel_config=compute_kernel_cfg,
            )
            ttnn.deallocate(out)
            out = None

        durations_us = []
        out_shape = None

        for _ in range(runs):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()

            out = ttnn.matmul(
                tt_a,
                tt_b,
                memory_config=output_mem_cfg,
                dtype=output_dtype,
                compute_kernel_config=compute_kernel_cfg,
            )

            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            durations_us.append((t1 - t0) * 1e6)

            torch_out = ttnn.to_torch(out)
            out_shape = list(torch_out.shape)
            expected_shape = [1, batch_a, m_dim, n_dim]
            if out_shape != expected_shape:
                raise RuntimeError(f"Expected output shape {expected_shape}, got {out_shape}")

            ttnn.deallocate(out)
            out = None

        durations_sorted = sorted(durations_us)
        mean_us = sum(durations_us) / len(durations_us)
        std_us = (sum((x - mean_us) ** 2 for x in durations_us) / len(durations_us)) ** 0.5

        return {
            "op_id": op_id,
            "description": op["description"],
            "shape_a": list(shape_a),
            "shape_b": list(shape_b),
            "output_shape": out_shape,
            "math_fidelity": comp.get("math_fidelity", "HiFi2"),
            "runs": runs,
            "min_us": round(min(durations_us), 2),
            "max_us": round(max(durations_us), 2),
            "median_us": round(durations_sorted[len(durations_sorted) // 2], 2),
            "mean_us": round(mean_us, 2),
            "std_us": round(std_us, 2),
        }

    finally:
        if out is not None:
            try:
                ttnn.deallocate(out)
            except Exception:
                pass
        if tt_a is not None:
            try:
                ttnn.deallocate(tt_a)
            except Exception:
                pass
        if tt_b is not None:
            try:
                ttnn.deallocate(tt_b)
            except Exception:
                pass


def run_square_workload(wlname: str, device, cfg: dict):
    if IS_POLARIS:
        set_default_device(device)

    _ = cfg.get("bs", 1)

    warmup = int(cfg.get("warmup", 0))
    runs = int(cfg.get("runs", 1))
    verbose = bool(cfg.get("verbose", False))

    sizes = parse_csv_ints(cfg.get("sizes", "128,256,512"))
    batches = parse_csv_ints(cfg.get("batches", "1,8"))
    fids = parse_csv_strs(cfg.get("fids", "LoFi,HiFi2,HiFi4"))

    ops = build_square_sweep_ops(sizes, batches, fids)

    logger.info(f"Running workload '{wlname}'")
    logger.info(f"sizes={sizes}")
    logger.info(f"batches={batches}")
    logger.info(f"fids={fids}")
    logger.info(f"total_ops={len(ops)}")

    results = []
    errors = []

    for op in ops:
        try:
            results.append(run_op(device, op, warmup=warmup, runs=runs, verbose=verbose))
        except Exception as e:
            logger.error(f"Op {op['op_id']} failed: {e}")
            errors.append({
                "op_id": op["op_id"],
                "description": op["description"],
                "error": str(e),
            })

    logger.info(f"Completed workload '{wlname}' success={len(results)} failed={len(errors)}")
    return {"results": results, "errors": errors}