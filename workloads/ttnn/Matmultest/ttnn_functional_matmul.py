# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

# -------- Helpers to map strings to ttnn enums / dtypes --------

def parse_dtype(dtype_str: str):
    """
    Map a string like 'bfloat8_b', 'bfp8', 'bf16', 'float32' to ttnn dtype.
    Adjust symbols to match your TTNN build if needed.
    """
    s = str(dtype_str).lower()

    if s in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if s in ("fp32", "float32"):
        return ttnn.float32

    if s in ("bfp8", "bfloat8_b", "bfp8_b", "bf8"):
        # Change to ttnn.bfloat8 or ttnn.bfp8_b if your build uses that
        return ttnn.bfloat8_b

    raise ValueError(f"Unsupported dtype string for TTNN matmul workload: {dtype_str}")

def parse_math_fidelity(fidelity_str: str):
    """
    Map a string like 'LoFi', 'HiFi2', 'HiFi3', 'HiFi4' to ttnn.MathFidelity.
    """
    s = str(fidelity_str).lower()
    if s == "lofi":
        return ttnn.MathFidelity.LoFi
    if s == "hifi2":
        return ttnn.MathFidelity.HiFi2
    if s == "hifi3":
        return ttnn.MathFidelity.HiFi3
    if s == "hifi4":
        return ttnn.MathFidelity.HiFi4

    raise ValueError(f"Unsupported math fidelity for TTNN matmul workload: {fidelity_str}")

# -------- Polaris TTNN workload entry point --------

def run_matmul_test(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Polaris TTNN workload entry for a simple MxK @ KxN matmul.

    Supports:
      - single size (M,K,N), or
      - sweep of square sizes via cfg["sizes"] = "32,64,128,...,4096".

    Args:
        wlname: Workload identifier from YAML (e.g., "Matmul2Tensors").
        device: TTNN device managed by Polaris (do NOT open/close here).
        cfg:    Merged configuration dict from YAML + CLI overrides.

    Relevant cfg keys:
        bs        : int, batch size (required by Polaris infra, but unused here; default 1)
        M,K,N     : ints for single size (default 1024 each, if "sizes" not given)
        sizes     : str, comma-separated list of square sizes, e.g. "32,64,..."
        dtype     : str, device dtype ("bfloat8_b", "bf16", "float32", ...) (default "bfloat8_b")
        fidelity  : str, math fidelity ("HiFi4", "HiFi3", "HiFi2", "LoFi") (default "HiFi4")
        num_runs  : int, how many times to run matmul per size (default 1)
    """

    # 1) Read config
    batch_size = int(cfg.get("bs", 1))

    # Sweep mode if "sizes" is provided; otherwise fall back to single M,K,N
    sizes_str = cfg.get("sizes", None)
    if sizes_str is None:
        # Single size
        M = int(cfg.get("M", cfg.get("m", 1024)))
        K = int(cfg.get("K", cfg.get("k", 1024)))
        N = int(cfg.get("N", cfg.get("n", 1024)))
        size_list = [(M, K, N)]
    else:
        # Sweep: e.g. "32,64,128,256,512,1024,2048,4096"
        size_list = []
        for token in str(sizes_str).split(","):
            x = int(token.strip())
            if x <= 0:
                raise ValueError(f"Invalid matrix size in sizes list: {x}")
            size_list.append((x, x, x))

    num_runs = int(cfg.get("num_runs", cfg.get("runs", 1)))  # default 1
    dtype_str = cfg.get("dtype", "bfloat8_b")
    fidelity_str = cfg.get("fidelity", "HiFi4")

    dtype = parse_dtype(dtype_str)
    math_fidelity = parse_math_fidelity(fidelity_str)

    logger.info(f"=== TTNN Polaris Matmul Workload: {wlname} ===")
    logger.info(f"Batch size (unused here) : {batch_size}")
    logger.info(f"Data type      : {dtype_str}")
    logger.info(f"Math fidelity  : {fidelity_str}")
    logger.info(f"Num runs per size: {num_runs}")
    logger.info(f"Size list      : {size_list}")

    # 2) Compute kernel config
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    logger.info(f"Requested math_fidelity: {compute_config.math_fidelity}")

    # 3) Sweep over all sizes
    output_tensor = None

    for (M, K, N) in size_list:
        logger.info(f"--- Running matmul for M={M}, K={K}, N={N} ---")

        logger.info(
            f"Creating TTNN tensors A[{M}, {K}] and B[{K}, {N}] on device "
            f"with dtype={dtype_str}..."
        )

        a_tt = ttnn._rand(
            (M, K),
            dtype=dtype,
            device=device,
        )

        b_tt = ttnn._rand(
            (K, N),
            dtype=dtype,
            device=device,
        )

        for i in range(num_runs):
            logger.info(
                f"Run {i + 1}/{num_runs}: matmul ({dtype_str}, {fidelity_str}) "
                f"size=({M},{K},{N})"
            )
            output_tensor = ttnn.matmul(
                a_tt,
                b_tt,
                compute_kernel_config=compute_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

    # 4) Optional: log result shape/precision for sanity
    try:
        out_shape = output_tensor.shape
    except Exception:
        out_shape = "unknown"

    logger.info(f"Matmul sweep completed; final output shape (TTNN): {out_shape}")
    logger.info("Returning output tensor for Polaris graph capture.")

    return output_tensor