from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

def parse_dtype(dtype_str: str):
    s = str(dtype_str).lower()
    if s in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if s in ("fp32", "float32"):
        return ttnn.float32
    if s in ("bfp8", "bfloat8_b", "bfp8_b", "bf8"):
        return ttnn.bfloat8_b
    raise ValueError(f"Unsupported dtype string for TTNN matmul workload: {dtype_str}")

def parse_math_fidelity(fidelity_str: str):
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

def run_matmul_test(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Square matmul sweep: MxK @ KxN, with M=K=N for each size.

    Control sizes in 2 ways:

      1) Explicit list:
         sizes: "32,64,128,256,512,1024,2048,4096"

      2) Range (multiples of 32):
         min_size: 32
         max_size: 4096
         step: 32   (default if not provided)

    If both 'sizes' and (min_size/max_size) are given, 'sizes' wins.
    """

    batch_size = int(cfg.get("bs", 1))

    # Decide size list
    sizes_str = cfg.get("sizes", None)
    if sizes_str is not None:
        size_list = []
        for token in str(sizes_str).split(","):
            x = int(token.strip())
            if x <= 0:
                raise ValueError(f"Invalid matrix size in sizes list: {x}")
            size_list.append((x, x, x))
    else:
        min_size = int(cfg.get("min_size", 32))
        max_size = int(cfg.get("max_size", 4096))
        step = int(cfg.get("step", 32))

        if min_size <= 0 or max_size <= 0 or step <= 0 or min_size > max_size:
            raise ValueError(
                f"Invalid range: min_size={min_size}, max_size={max_size}, step={step}"
            )

        size_list = []
        x = min_size
        while x <= max_size:
            size_list.append((x, x, x))
            x += step

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
    logger.info(f"Size list (square): {size_list}")

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    output_tensor = None
    for (M, K, N) in size_list:
        logger.info(f"--- Running matmul for M={M}, K={K}, N={N} ---")

        a_tt = ttnn._rand((M, K), dtype=dtype, device=device)
        b_tt = ttnn._rand((K, N), dtype=dtype, device=device)

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

    try:
        out_shape = output_tensor.shape
    except Exception:
        out_shape = "unknown"

    logger.info(f"Matmul sweep completed; final output shape (TTNN): {out_shape}")
    logger.info("Returning output tensor for Polaris graph capture.")
    return output_tensor