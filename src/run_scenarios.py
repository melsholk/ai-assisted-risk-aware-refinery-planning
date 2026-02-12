from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from src.model import load_data, default_params, build_and_solve_lp
from src.scenarios import generate_scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--import_cap_risk", action="store_true")
    args = ap.parse_args()

    data = load_data(args.data_dir)
    base = default_params(data)

    # Base deterministic
    base_res = build_and_solve_lp(data, params=base, verbose=False)
    base_profit = float(base_res["economics"].loc[base_res["economics"]["item"] == "Profit", "USDk_per_day"].values[0])
    print(f"\nBase deterministic profit (USDk/day): {base_profit:,.2f}")

    scenarios = generate_scenarios(
        base,
        n=args.n,
        seed=args.seed,
        enable_import_cap_risk=args.import_cap_risk,
    )

    rows = []
    for sc in scenarios:
        res = build_and_solve_lp(data, params=sc.params, verbose=False)
        econ = res["economics"].set_index("item")["USDk_per_day"]
        profit = float(econ["Profit"])
        crude = res["crude"].set_index("crude")["run_kbpd"].to_dict()
        h2_buy = float(res["hydrogen"]["H2_purchased"].iloc[0])

        rows.append(
            {
                "scenario": sc.name,
                "prob": sc.prob,
                "profit": profit,
                "crude_LightSweet": crude.get("LightSweet", 0.0),
                "crude_MedSour": crude.get("MedSour", 0.0),
                "crude_HeavySour": crude.get("HeavySour", 0.0),
                "h2_buy": h2_buy,
            }
        )

    df = pd.DataFrame(rows)

    # Probability-weighted stats
    p = df["prob"].values
    p = p / p.sum()
    x = df["profit"].values
    mean = float(np.sum(p * x))
    var = float(np.sum(p * (x - mean) ** 2))
    sd = float(np.sqrt(var))

    # Quantiles (unweighted for now—fine for sanity; later do weighted)
    q05, q50, q95 = np.quantile(x, [0.05, 0.50, 0.95])

    print("\n=== Scenario Profit Summary (USDk/day) ===")
    print(f"Mean (prob-weighted): {mean:,.2f}")
    print(f"Std  (prob-weighted): {sd:,.2f}")
    print(f"P05/P50/P95 (unweighted): {q05:,.2f} / {q50:,.2f} / {q95:,.2f}")

    print("\n=== Worst 5 scenarios (by profit) ===")
    print(df.sort_values("profit").head(5).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\n=== Best 5 scenarios (by profit) ===")
    print(df.sort_values("profit", ascending=False).head(5).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
