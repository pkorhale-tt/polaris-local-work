import pandas as pd
import re
import sys
import argparse

TILE = 32
def pad_tile(x): return ((x + TILE - 1) // TILE) * TILE

def parse_polaris_shape(s):
    parts = str(s).split(';')
    shapes = []
    for p in parts:
        m = re.search(r'\[([0-9x]+)\]', p)
        if m:
            dims = [int(x) for x in m.group(1).split('x')]
            shapes.append(dims)
    if len(shapes) < 2: return None,None,None,None
    a, b = shapes[0], shapes[1]
    M = a[-2] if len(a) >= 2 else 1
    K = a[-1]; N = b[-1]
    batch = 1
    for d in a[:-2]: batch *= d
    return batch, M, K, N

def parse_dim(v):
    m = re.match(r'^(\d+)', str(v).strip())
    return int(m.group(1)) if m else 0

def main():
    p = argparse.ArgumentParser(
        description="Compare HW kernel time vs Polaris predicted time")
    p.add_argument("polaris_csv",
                   help="Polaris opstats CSV  (n150-TTNN-...opstats.csv)")
    p.add_argument("hw_csv",
                   help="Hardware perf CSV   (ops_perf_results_...csv)")
    p.add_argument("--out", default="comparison.csv",
                   help="Output CSV path (default: comparison.csv)")
    args = p.parse_args()

    polaris = pd.read_csv(args.polaris_csv)
    hw      = pd.read_csv(args.hw_csv)

    # ── Polaris ───────────────────────────────────────────────────────────────
    pol_mm = polaris[polaris['optype']=='MatMul'].copy()
    pol_rows = []
    for _, r in pol_mm.iterrows():
        b, M, K, N = parse_polaris_shape(r['input_tensors'])
        if M is None: continue
        pol_rows.append({
            'batch'         : b,
            'M'             : pad_tile(M),
            'K'             : pad_tile(K),
            'N'             : pad_tile(N),
            'polaris_ns'    : r['msecs'] * 1e6,
            'uses_predictor': r['uses_perf_lookup']
        })
    pol_df  = pd.DataFrame(pol_rows)
    pol_agg = pol_df.groupby(['batch','M','K','N']).agg(
        polaris_ns     = ('polaris_ns',    'mean'),
        uses_predictor = ('uses_predictor','first')
    ).reset_index()

    # ── Hardware ──────────────────────────────────────────────────────────────
    hw_mm = hw[hw['OP CODE'].str.contains(
        'matmul|Matmul|MatMul', case=False, na=False)].copy()
    hw_rows = []
    for _, r in hw_mm.iterrows():
        W = parse_dim(r['INPUT_0_W_PAD[LOGICAL]'])
        Z = parse_dim(r['INPUT_0_Z_PAD[LOGICAL]'])
        M = parse_dim(r['INPUT_0_Y_PAD[LOGICAL]'])
        K = parse_dim(r['INPUT_0_X_PAD[LOGICAL]'])
        N = parse_dim(r['INPUT_1_X_PAD[LOGICAL]'])
        hw_rows.append({
            'batch': W * Z, 'M': M, 'K': K, 'N': N,
            'hw_ns': r['DEVICE KERNEL DURATION [ns]']
        })
    hw_df  = pd.DataFrame(hw_rows)
    hw_agg = hw_df.groupby(['batch','M','K','N']).agg(
        hw_ns = ('hw_ns','mean')
    ).reset_index()

    # ── Merge and compare ─────────────────────────────────────────────────────
    merged = pd.merge(hw_agg, pol_agg,
                      on=['batch','M','K','N'], how='outer')
    merged['ratio']     = (merged['polaris_ns'] / merged['hw_ns']).round(3)
    merged['error_pct'] = (
        (merged['polaris_ns'] - merged['hw_ns']) / merged['hw_ns'] * 100
    ).round(1)
    merged = merged.sort_values(['batch','M','K','N']).reset_index(drop=True)
    merged['hw_ns']      = merged['hw_ns'].round(0)
    merged['polaris_ns'] = merged['polaris_ns'].round(0)

    out = merged[['batch','M','K','N','hw_ns','polaris_ns',
                  'uses_predictor','ratio','error_pct']]

    # ── Output ────────────────────────────────────────────────────────────────
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print()

    matched = merged.dropna(subset=['hw_ns','polaris_ns'])
    matched = matched[matched['hw_ns'] > 0]
    print(f"Matched shapes:   {len(matched)}")
    print(f"Mean abs error %: {matched['error_pct'].abs().mean():.1f}%")
    print(f"Mean ratio:       {matched['ratio'].mean():.3f}")
    print(f"Saved to:         {args.out}")

if __name__ == "__main__":
    main()
