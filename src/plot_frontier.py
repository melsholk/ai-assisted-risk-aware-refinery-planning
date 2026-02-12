from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/frontier.csv", help="Path to frontier CSV")
    ap.add_argument("--out_dir", default="outputs", help="Where to save images")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    # Identify risk metric columns
    var_col = [c for c in df.columns if c.startswith("var_profit_a")]
    cvar_col = [c for c in df.columns if c.startswith("cvar_profit_a")]
    if not var_col or not cvar_col:
        raise ValueError("Could not find var_profit_a* or cvar_profit_a* columns in CSV.")
    var_col = var_col[0]
    cvar_col = cvar_col[0]

    # 1) Efficient frontier: Expected vs CVaR
    plt.figure()
    plt.plot(df[cvar_col], df["expected_profit"], marker="o")
    plt.xlabel(f"{cvar_col} (downside tail mean profit)")
    plt.ylabel("expected_profit")
    plt.title("Efficient Frontier: Expected Profit vs CVaR Profit")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "frontier_expected_vs_cvar.png"), dpi=200)
    plt.close()

    # 2) Efficient frontier: Expected vs VaR
    plt.figure()
    plt.plot(df[var_col], df["expected_profit"], marker="o")
    plt.xlabel(f"{var_col} (downside quantile profit)")
    plt.ylabel("expected_profit")
    plt.title("Efficient Frontier: Expected Profit vs VaR Profit")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "frontier_expected_vs_var.png"), dpi=200)
    plt.close()

    # 3) Policy levers vs lambda
    cols = ["T_FCC_D", "T_FCC_G", "T_HC", "T_VR", "T_REF", "imp_fs_total"]
    cols = [c for c in cols if c in df.columns]

    for c in cols:
        plt.figure()
        plt.plot(df["lam"], df[c], marker="o")
        plt.xlabel("lam")
        plt.ylabel(c)
        plt.title(f"First-stage decision vs lambda: {c}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"policy_{c}.png"), dpi=200)
        plt.close()

    # 4) Spot risk diagnostic
    if "spot_bind_freq" in df.columns:
        plt.figure()
        plt.plot(df["lam"], df["spot_bind_freq"], marker="o")
        plt.xlabel("lam")
        plt.ylabel("spot_bind_freq")
        plt.title("Spot cap binding frequency vs lambda")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "spot_bind_freq.png"), dpi=200)
        plt.close()

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
