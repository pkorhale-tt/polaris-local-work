"""
step2_predict_config.py
-----------------------
Reads ops_raw.yaml (from step1), runs matmul_config + core predictor
for each op, and writes ops_configured.yaml.

Each output entry adds:
    config_type:   MatmulMultiCoreReuseMultiCast1DProgramConfig | ...
    per_core_M:    int
    per_core_N:    int
    in0_block_w:   int
    mcast_in0:     bool | null
    fuse_batch:    bool
    predicted_cores: int

Usage:
    python step2_predict_config.py ops_raw.yaml --out ops_configured.yaml

Note: matmul_config.py and matmul_config_wrapper.py must be importable
      (i.e. on PYTHONPATH or in the same directory).
"""

import sys
import os
import math
import yaml
import argparse
from pathlib import Path

# ── Allow importing matmul_config_wrapper from the kernel_config dir ──────────
# Adjust this path if your files live elsewhere.
KERNEL_CONFIG_DIR = os.environ.get(
    "KERNEL_CONFIG_DIR",
    os.path.join(os.path.dirname(__file__), "..")   # default: one level up
)
sys.path.insert(0, KERNEL_CONFIG_DIR)
sys.path.insert(0, os.path.dirname(__file__))

try:
    from matmul_config_wrapper import predict_config, TensorSpec, MemSpec, DeviceSpec
    HAS_CONFIG_PREDICTOR = True
except ImportError:
    HAS_CONFIG_PREDICTOR = False
    print("[WARN] matmul_config_wrapper not found — using heuristic fallback.")

# ── Constants ─────────────────────────────────────────────────────────────────
TILE         = 32
DEVICE_X     = 8
DEVICE_Y     = 8
DEVICE_CORES = DEVICE_X * DEVICE_Y

# ── Heuristic fallback (mirrors create_simple_matmul_program_config logic) ────

def div_up(a, b):
    return (a + b - 1) // b

def heuristic_config(batch, M, K, N):
    """
    Simplified version of the C++ config selection used when the full
    Python port (matmul_config.py) is not on the path.
    Covers the two most common ViT cases:
      - narrow/tall → 1D mcast
      - square/wide DRAM interleaved → 2D mcast
    """
    Mt = M // TILE
    Kt = K // TILE
    Nt = N // TILE

    height = batch * M
    width  = N
    ratio  = max(height, width) // max(1, min(height, width))
    narrow = ratio > 8 or height <= TILE or width <= TILE

    if narrow:
        # 1D systolic
        is_tall = height > width
        num_cores = DEVICE_CORES
        if is_tall:
            per_core_M = div_up(batch * Mt, num_cores)
            per_core_N = Nt
            mcast_in0  = False
        else:
            per_core_M = batch * Mt
            per_core_N = div_up(Nt, num_cores)
            mcast_in0  = True
        in0_block_w  = 2 if Kt % 2 == 0 else 1
        predicted_cores = (
            Nt // per_core_N if mcast_in0 else (batch * Mt) // per_core_M
        )
        return dict(
            config_type="MatmulMultiCoreReuseMultiCast1DProgramConfig",
            per_core_M=per_core_M, per_core_N=per_core_N,
            in0_block_w=in0_block_w, mcast_in0=mcast_in0,
            fuse_batch=True, predicted_cores=predicted_cores,
        )
    else:
        # 2D mcast (all DRAM interleaved path)
        per_core_M = div_up(Mt, DEVICE_Y)
        per_core_N = div_up(Nt, DEVICE_X)
        in0_block_w = Kt // DEVICE_X if Kt % DEVICE_X == 0 else 1
        cores_m = Mt // per_core_M
        cores_n = Nt // per_core_N
        return dict(
            config_type="MatmulMultiCoreReuseMultiCastProgramConfig",
            per_core_M=per_core_M, per_core_N=per_core_N,
            in0_block_w=in0_block_w, mcast_in0=None,
            fuse_batch=False, predicted_cores=cores_m * cores_n,
        )

# ── Core count formula (mirrors predict_cores.py) ─────────────────────────────

def predict_cores_from_cfg(cfg_type, batch, M, K, N, per_core_M, per_core_N,
                            mcast_in0, fuse_batch):
    Mt_fused = (batch * M) // TILE
    Mt_plain = M // TILE
    Nt       = N // TILE

    is_1d = cfg_type is not None and "1D" in cfg_type
    if is_1d:
        if mcast_in0:
            return Nt // per_core_N
        else:
            Mt = Mt_fused if fuse_batch else Mt_plain
            return Mt // per_core_M
    else:
        # 2D
        return (Mt_plain // per_core_M) * (Nt // per_core_N)

# ── Per-op prediction ──────────────────────────────────────────────────────────

def predict_op(op: dict) -> dict:
    result = dict(op)   # copy all original fields

    M = op.get("M")
    K = op.get("K")
    N = op.get("N")
    if not all([M, K, N]):
        result.update(config_type="UNKNOWN", predicted_cores=0,
                      per_core_M=None, per_core_N=None,
                      in0_block_w=None, mcast_in0=None, fuse_batch=False,
                      config_error="missing shape")
        return result

    batch = op.get("batch", 1)

    if HAS_CONFIG_PREDICTOR:
        try:
            a = TensorSpec(batch=batch, rows=M, cols=K,
                           mem=MemSpec(layout=op.get("layout_a", "INTERLEAVED"),
                                       buffer_type=op.get("buffer_a", "DRAM")))
            b = TensorSpec(batch=op.get("batch_b", 1), rows=K, cols=N,
                           mem=MemSpec(layout=op.get("layout_b", "INTERLEAVED"),
                                       buffer_type=op.get("buffer_b", "DRAM")))
            out_mem = MemSpec(layout=op.get("layout_out", "INTERLEAVED"),
                              buffer_type=op.get("buffer_out", "DRAM"))
            cfg = predict_config(a=a, b=b, output_mem=out_mem,
                                 device=DeviceSpec(core_x=DEVICE_X, core_y=DEVICE_Y))

            cfg_type    = cfg["config_type"]
            per_core_M  = cfg.get("per_core_M", 1)
            per_core_N  = cfg.get("per_core_N", 1)
            in0_block_w = cfg.get("in0_block_w", 1)
            mcast_in0   = cfg.get("mcast_in0", None)
            fuse_batch  = cfg.get("fuse_batch", False)

        except Exception as e:
            cfg_type = "ERROR"
            per_core_M = per_core_N = in0_block_w = 1
            mcast_in0 = None
            fuse_batch = False
            result["config_error"] = str(e)
    else:
        fc = heuristic_config(batch, M, K, N)
        cfg_type    = fc["config_type"]
        per_core_M  = fc["per_core_M"]
        per_core_N  = fc["per_core_N"]
        in0_block_w = fc["in0_block_w"]
        mcast_in0   = fc["mcast_in0"]
        fuse_batch  = fc["fuse_batch"]

    predicted_cores = predict_cores_from_cfg(
        cfg_type, batch, M, K, N, per_core_M, per_core_N, mcast_in0, fuse_batch
    )
    predicted_cores = max(1, min(predicted_cores, DEVICE_CORES))

    result.update(
        config_type=cfg_type,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        in0_block_w=in0_block_w,
        mcast_in0=mcast_in0,
        fuse_batch=fuse_batch,
        predicted_cores=predicted_cores,
    )
    return result

# ── YAML I/O ──────────────────────────────────────────────────────────────────

class _Dumper(yaml.Dumper):
    pass
_Dumper.add_representer(
    bool,
    lambda d, v: d.represent_scalar("tag:yaml.org,2002:bool", "true" if v else "false")
)
_Dumper.add_representer(
    type(None),
    lambda d, v: d.represent_scalar("tag:yaml.org,2002:null", "null")
)

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def write_yaml(records, path):
    with open(path, "w") as f:
        yaml.dump(records, f, Dumper=_Dumper,
                  default_flow_style=False, sort_keys=False)

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Step 2 — predict config + cores for each op")
    p.add_argument("raw_yaml",  help="Input: ops_raw.yaml from step1")
    p.add_argument("--out",     default="ops_configured.yaml")
    p.add_argument("--kernel-config-dir", default=None,
                   help="Path to directory containing matmul_config_wrapper.py")
    args = p.parse_args()

    if args.kernel_config_dir:
        sys.path.insert(0, args.kernel_config_dir)
        try:
            from matmul_config_wrapper import predict_config, TensorSpec, MemSpec, DeviceSpec
            globals()["HAS_CONFIG_PREDICTOR"] = True
            globals()["predict_config"] = predict_config
            globals()["TensorSpec"] = TensorSpec
            globals()["MemSpec"] = MemSpec
            globals()["DeviceSpec"] = DeviceSpec
        except ImportError:
            pass

    ops = load_yaml(args.raw_yaml)
    out = []
    for op in ops:
        out.append(predict_op(op))

    write_yaml(out, args.out)
    print(f"Processed {len(out)} ops  →  {args.out}")
    print(f"\n{'ID':>4}  {'Config':>8}  {'Cores':>5}  {'pcM':>4}  {'pcN':>4}  Description")
    print("─" * 80)
    for o in out:
        short = {
            "MatmulMultiCoreReuseMultiCast1DProgramConfig": "1D",
            "MatmulMultiCoreReuseMultiCastProgramConfig":   "2D",
            "MatmulMultiCoreReuseProgramConfig":            "REUSE",
            "MatmulMultiCoreProgramConfig":                 "SIMPLE",
            "UNKNOWN": "???",
            "ERROR":   "ERR",
        }.get(o.get("config_type", ""), o.get("config_type", "")[:6])
        print(f"{o['op_id']:>4}  {short:>8}  "
              f"{str(o.get('predicted_cores','')):>5}  "
              f"{str(o.get('per_core_M','')):>4}  "
              f"{str(o.get('per_core_N','')):>4}  "
              f"{o['description'][:38]}")

if __name__ == "__main__":
    main()