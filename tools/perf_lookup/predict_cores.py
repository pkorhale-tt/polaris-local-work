"""
Core Count Predictor for Matmul Ops
--------------------------------------
Reads perf CSV, predicts number of cores used for each matmul row,
compares against actual CORE COUNT column, writes output CSV.

Usage:
    python predict_cores.py <input.csv> <output.csv>

Logic:
    For 1D config (mcast_in0=True):
        num_cores = Nt / per_core_N          (N is split, M is broadcast)

    For 1D config (mcast_in0=False):
        num_cores = Mt / per_core_M          (M is split, N is broadcast)
        Mt = batch*M/TILE if fuse_batch else M/TILE

    For 2D config:
        num_cores = (Mt / per_core_M) * (Nt / per_core_N)
        Mt = M/TILE (fuse_batch is always False for 2D)
"""

import sys
import os
import re
import pandas as pd

TILE = 32

# ─────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────

def parse_dim(val) -> int:
    """Parse '224[224]' → 224"""
    if pd.isna(val):
        return 1
    m = re.match(r'^(\d+)', str(val).strip())
    return int(m.group(1)) if m else 1


def parse_attrs(attrs_str: str) -> dict:
    """Extract all fields needed for core count from ATTRIBUTES."""
    result = {
        "per_core_M":  None,
        "per_core_N":  None,
        "fuse_batch":  False,
        "mcast_in0":   None,       # None means 2D config
        "config_type": None,
    }
    if pd.isna(attrs_str):
        return result

    s = str(attrs_str)

    m = re.search(r'per_core_M=(\d+)', s);  result["per_core_M"] = int(m.group(1)) if m else None
    m = re.search(r'per_core_N=(\d+)', s);  result["per_core_N"] = int(m.group(1)) if m else None
    m = re.search(r'fuse_batch=(\d)', s);   result["fuse_batch"] = m.group(1) == '1' if m else False
    m = re.search(r'mcast_in0=(\d)', s);    result["mcast_in0"]  = (m.group(1) == '1') if m else None
    m = re.search(r"'program_config': '([A-Za-z0-9_]+)", s)
    result["config_type"] = m.group(1) if m else None

    return result


# ─────────────────────────────────────────────
# Core count formula
# ─────────────────────────────────────────────

def predict_cores(row) -> tuple:
    """
    Returns (predicted_cores, actual_cores, match, reason)
    """
    try:
        attrs = parse_attrs(row.get("ATTRIBUTES", ""))

        per_core_M  = attrs["per_core_M"]
        per_core_N  = attrs["per_core_N"]
        fuse_batch  = attrs["fuse_batch"]
        mcast_in0   = attrs["mcast_in0"]
        config_type = attrs["config_type"]

        if per_core_M is None or per_core_N is None:
            return None, int(row["CORE COUNT"]), False, "missing per_core params"

        # Shapes
        W     = parse_dim(row.get("INPUT_0_W_PAD[LOGICAL]"))
        Z     = parse_dim(row.get("INPUT_0_Z_PAD[LOGICAL]"))
        M     = parse_dim(row.get("INPUT_0_Y_PAD[LOGICAL]"))
        N     = parse_dim(row.get("INPUT_1_X_PAD[LOGICAL]"))
        batch = W * Z

        Mt_fused = (batch * M) // TILE      # batch folded into M
        Mt_plain = M // TILE                # just M
        Nt       = N // TILE

        is_1d = config_type is not None and "1D" in config_type

        if is_1d:
            if mcast_in0:
                # N is split across all cores, M is broadcast
                predicted = Nt // per_core_N
                reason = f"1D mcast_in0=True: Nt({Nt}) / per_core_N({per_core_N})"
            else:
                # M is split across all cores, N is broadcast
                Mt = Mt_fused if fuse_batch else Mt_plain
                predicted = Mt // per_core_M
                reason = f"1D mcast_in0=False: Mt({Mt}) / per_core_M({per_core_M}) fuse={fuse_batch}"
        else:
            # 2D: both M and N dimensions are split
            Mt = Mt_plain   # 2D never fuses batch
            cores_y = Mt // per_core_M
            cores_x = Nt // per_core_N
            predicted = cores_y * cores_x
            reason = f"2D: (Mt({Mt})/pcM({per_core_M})) * (Nt({Nt})/pcN({per_core_N})) = {cores_y}*{cores_x}"

        actual = int(row["CORE COUNT"])
        match  = predicted == actual

        return predicted, actual, match, reason

    except Exception as e:
        return None, None, False, f"ERROR: {e}"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_cores.py <input.csv> <output.csv>")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]

    print(f"\nReading: {in_path}")
    df = pd.read_csv(in_path)
    print(f"Total rows     : {len(df)}")

    matmul_mask = df["OP CODE"].str.contains("matmul|Matmul", case=False, na=False)
    print(f"Matmul rows    : {matmul_mask.sum()}")

    pred_col   = [""] * len(df)
    actual_col = [""] * len(df)
    match_col  = [""] * len(df)
    reason_col = [""] * len(df)

    correct = 0
    errors  = 0
    total   = 0

    for idx in df[matmul_mask].index:
        row = df.loc[idx]
        predicted, actual, match, reason = predict_cores(row)

        pred_col[idx]   = predicted if predicted is not None else "ERROR"
        actual_col[idx] = actual
        match_col[idx]  = match
        reason_col[idx] = reason

        total += 1
        if "ERROR" in str(reason):
            errors += 1
        elif match:
            correct += 1

    df["PREDICTED_CORES"] = pred_col
    df["ACTUAL_CORES"]    = actual_col
    df["CORES_MATCH"]     = match_col
    df["CORES_REASON"]    = reason_col

    df.to_csv(out_path, index=False)
    print(f"Output written : {out_path}")

    print(f"\n{'─'*42}")
    print(f"  Matmul rows processed  : {total}")
    print(f"  Correct predictions    : {correct}  ({100*correct//total if total else 0}%)")
    print(f"  Wrong predictions      : {total - correct - errors}")
    print(f"  Errors                 : {errors}")
    print(f"{'─'*42}")

    # Show mismatches
    mat_df = df[matmul_mask].copy()
    wrong  = mat_df[mat_df["CORES_MATCH"] == False]
    if len(wrong) > 0:
        print(f"\nMismatches ({len(wrong)}):")
        for _, r in wrong.head(10).iterrows():
            print(f"  predicted={r['PREDICTED_CORES']}  actual={r['ACTUAL_CORES']}")
            print(f"  reason   ={r['CORES_REASON']}")
            print()


if __name__ == "__main__":
    main()
