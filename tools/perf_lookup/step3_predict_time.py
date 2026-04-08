"""
step3_predict_time.py
---------------------
Reads ops_configured.yaml (from step2) and predicts kernel device time
for every op using the pipeline model from pipeline_model_v3.py.

Output: ops_timed.yaml  (all fields from step2 + timing predictions)

New fields added per op:
    predicted_ns:        estimated kernel time in nanoseconds
    compute_cyc_per_k:   compute cycles per K-step
    reader_cyc_per_k:    reader (bandwidth) cycles per K-step
    bottleneck:          "COMPUTE" or "BANDWIDTH"
    total_bytes_per_core: total DRAM bytes per core
    eff_bw_GBs:          effective bandwidth used (GB/s)

Usage:
    python step3_predict_time.py ops_configured.yaml --out ops_timed.yaml
"""

import sys
import os
import yaml
import math
import argparse
from pathlib import Path

# ── Pipeline model constants (mirrors pipeline_model_v3.py) ───────────────────

TILE       = 32
CLOCK_HZ   = 1e9
BYTES_BF16 = 2

FPU_CYCLES           = 8
OVERHEAD_PER_TILE    = 40    # cycles: dst acquire + pack + release
BIG_OP_FACTOR        = 2.0   # slowdown for huge 2D square ops

FIDELITY = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}

# Per-config effective BW constants (GB/s)
# Tune these vs measured data; mirrors calibration from pipeline_model_v3
BW_TABLE = {
    "1D":   {"small": 1.2e9, "medium": 1.9e9, "large": 4.0e9},
    "2D":   {"small": 1.0e9, "medium": 1.6e9, "large": 3.5e9},
    "REUSE":{"small": 1.0e9, "medium": 1.5e9, "large": 3.0e9},
    "DEFAULT":{"small": 1.0e9, "medium": 1.5e9, "large": 3.0e9},
}
BW_THRESHOLDS = [500_000, 5_000_000]   # bytes/core breakpoints

def get_eff_bw(total_bytes, cfg_type_short):
    bw = BW_TABLE.get(cfg_type_short, BW_TABLE["DEFAULT"])
    if total_bytes < BW_THRESHOLDS[0]:
        return bw["small"]
    elif total_bytes < BW_THRESHOLDS[1]:
        return bw["medium"]
    return bw["large"]

def cfg_short(cfg_type):
    if "1D" in str(cfg_type):    return "1D"
    if "ReuseProg" in str(cfg_type): return "REUSE"
    if "MultiCast" in str(cfg_type): return "2D"
    return "DEFAULT"

# ── DRAM tile accounting (mirrors dram_tiles_per_core from pipeline_model_v3) ─

def dram_tiles_per_core(Mt, Kt, Nt, per_core_M, per_core_N,
                         cfg_short_, mcast_in0, num_cores):
    tiles_A = per_core_M * Kt
    tiles_B = per_core_N * Kt
    tiles_C = per_core_M * per_core_N

    if cfg_short_ == "1D":
        if mcast_in0 is True:
            tiles_A /= max(1, num_cores)
        elif mcast_in0 is False:
            tiles_B /= max(1, num_cores)
        else:
            tiles_A /= max(1, num_cores)
            tiles_B /= max(1, num_cores)
    elif cfg_short_ == "2D":
        cores_m = max(1, Mt // per_core_M)
        cores_n = max(1, Nt // per_core_N)
        tiles_A /= cores_n
        tiles_B /= cores_m
        if cores_m == cores_n and Mt == Nt and per_core_M == per_core_N:
            tiles_A *= 0.8
            tiles_B *= 0.8
    # REUSE / DEFAULT: no sharing, each core reads its own tiles

    return tiles_A, tiles_B, tiles_C

# ── Main prediction logic ──────────────────────────────────────────────────────

def predict_time(op: dict) -> dict:
    result = dict(op)

    M = op.get("M")
    K = op.get("K")
    N = op.get("N")
    if not all([M, K, N]):
        result.update(predicted_ns=None, bottleneck=None,
                      compute_cyc_per_k=None, reader_cyc_per_k=None,
                      total_bytes_per_core=None, eff_bw_GBs=None,
                      time_error="missing shape")
        return result

    batch       = op.get("batch", 1)
    batch_b     = op.get("batch_b", 1)
    per_core_M  = op.get("per_core_M")
    per_core_N  = op.get("per_core_N")
    num_cores   = op.get("predicted_cores", 64)
    cfg_type    = op.get("config_type", "UNKNOWN")
    mcast_in0   = op.get("mcast_in0", None)
    fuse_batch  = op.get("fuse_batch", False)
    buf_out     = op.get("buffer_out", "DRAM")
    fidelity    = op.get("math_fidelity", "HiFi2")

    if per_core_M is None or per_core_N is None:
        result.update(predicted_ns=None, bottleneck="UNKNOWN",
                      time_error="missing per_core params")
        return result

    Mt = M // TILE
    Kt = K // TILE
    Nt = N // TILE

    # fuse_batch folds batch into M for compute loop count
    bl = 1 if fuse_batch else batch

    tb_AB = TILE * TILE * BYTES_BF16
    tb_C  = tb_AB  # BF16 output (extend here for BF8 if needed)

    fm = FIDELITY.get(fidelity, 2)
    compute_cyc = FPU_CYCLES * fm

    cs = cfg_short(cfg_type)

    # Big-op slowdown for huge symmetric 2D mcast
    if cs == "2D" and M >= 1024 and K >= 1024 and N >= 1024:
        compute_cyc *= BIG_OP_FACTOR

    tiles_A, tiles_B, tiles_C = dram_tiles_per_core(
        Mt, Kt, Nt, per_core_M, per_core_N, cs, mcast_in0, num_cores
    )

    A_bytes = bl * tiles_A * tb_AB
    B_bytes = bl * tiles_B * tb_AB
    C_bytes = bl * tiles_C * tb_C if buf_out == "DRAM" else 0.0
    total   = A_bytes + B_bytes + C_bytes

    eff_bw     = get_eff_bw(total, cs)
    tiles_C_pc = per_core_M * per_core_N
    denom      = bl * tiles_C_pc * Kt
    bpk        = total / denom if denom > 0 else total

    reader_cyc    = (bpk / eff_bw) * CLOCK_HZ
    cyc_per_k     = max(compute_cyc, reader_cyc)
    bottleneck    = "COMPUTE" if compute_cyc >= reader_cyc else "BANDWIDTH"

    cycles_per_tile = Kt * cyc_per_k + OVERHEAD_PER_TILE
    total_cycles    = bl * tiles_C_pc * cycles_per_tile
    pred_ns         = total_cycles / CLOCK_HZ * 1e9

    result.update(
        predicted_ns=int(round(pred_ns)),
        compute_cyc_per_k=round(compute_cyc, 1),
        reader_cyc_per_k=round(reader_cyc, 1),
        bottleneck=bottleneck,
        total_bytes_per_core=int(round(total)),
        eff_bw_GBs=round(eff_bw / 1e9, 1),
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
    p = argparse.ArgumentParser(description="Step 3 — predict kernel time per op")
    p.add_argument("configured_yaml", help="Input: ops_configured.yaml from step2")
    p.add_argument("--out", default="ops_timed.yaml")
    p.add_argument("--fidelity", default="HiFi2",
                   choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"],
                   help="Math fidelity override (default: HiFi2)")
    args = p.parse_args()

    ops = load_yaml(args.configured_yaml)
    out = []
    for op in ops:
        if "math_fidelity" not in op:
            op["math_fidelity"] = args.fidelity
        out.append(predict_time(op))

    write_yaml(out, args.out)
    print(f"Timed {len(out)} ops  →  {args.out}")

    total_ns = sum(o.get("predicted_ns") or 0 for o in out)
    print(f"\n{'ID':>4}  {'ns':>10}  {'Bot':>9}  {'BW':>5}  Description")
    print("─" * 78)
    for o in out:
        ns  = o.get("predicted_ns")
        bot = (o.get("bottleneck") or "")[:3]
        bw  = o.get("eff_bw_GBs", "")
        print(f"{o['op_id']:>4}  {str(ns or 'N/A'):>10}  {bot:>9}  {str(bw):>5}  {o['description'][:40]}")
    print("─" * 78)
    print(f"{'TOTAL':>4}  {total_ns:>10,}  ns  ({total_ns/1e6:.2f} ms)")

if __name__ == "__main__":
    main()