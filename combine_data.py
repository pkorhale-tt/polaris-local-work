import sys
import pandas as pd

def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python combine_data.py "
            "<lofi_csv> <hifi2_csv> <hifi4_csv> <output_csv>"
        )
        sys.exit(1)

    lofi_csv, hifi2_csv, hifi4_csv, output_csv = sys.argv[1:5]

    df_lofi = pd.read_csv(lofi_csv)
    df_hifi2 = pd.read_csv(hifi2_csv)
    df_hifi4 = pd.read_csv(hifi4_csv)

    # Basic sanity check: same number of rows
    n1, n2, n3 = len(df_lofi), len(df_hifi2), len(df_hifi4)
    if not (n1 == n2 == n3):
        print(f"Error: row counts differ: LoFi={n1}, HiFi2={n2}, HiFi4={n3}")
        sys.exit(1)

    # Add fidelity tags
    df_lofi  = df_lofi.copy()
    df_hifi2 = df_hifi2.copy()
    df_hifi4 = df_hifi4.copy()

    df_lofi["fidelity"]  = "LoFi"
    df_hifi2["fidelity"] = "HiFi2"
    df_hifi4["fidelity"] = "HiFi4"

    rows = []
    for i in range(n1):
        # For dimension i: first LoFi row, then HiFi2, then HiFi4
        rows.append(df_lofi.iloc[i])
        rows.append(df_hifi2.iloc[i])
        rows.append(df_hifi4.iloc[i])

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved merged file to: {output_csv}")

if __name__ == "__main__":
    main()