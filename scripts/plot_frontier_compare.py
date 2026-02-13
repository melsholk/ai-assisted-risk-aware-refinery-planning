from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.model import load_data, default_params
from src.scenarios import generate_scenarios
from src.oracle import solve_oracle, default_first_stage_vars
from src.ml.features import extract_distribution_features
from src.ml.projection import project_first_stage
from src.ml.restore import restore_first_stage
from scipy.optimize import linprog
from src.model import build_lp_matrices


def weighted_var_cvar_profit(profits, probs, alpha):
    order = np.argsort(profits)
    P = profits[order]
    w = probs[order]
    cdf = np.cumsum(w)
    k = int(np.searchsorted(cdf, alpha, side="left"))
    k = min(max(k, 0), len(P) - 1)
    var = float(P[k])
    tail = profits <= var + 1e-12
    tail_prob = float(np.sum(probs[tail]))
    cvar = float(np.sum(probs[tail] * profits[tail]) / max(tail_prob, 1e-12))
    return var, cvar


def eval_policy_on_scenarios(data, scenarios, alpha, policy_fs):
    probs = np.array([s.prob for s in scenarios], dtype=float)
    probs = probs / probs.sum()

    profits = []
    for s in scenarios:
        lp = build_lp_matrices(data, params=s.params)
        idx = lp["idx"]
        bounds = list(lp["bounds"])

        for nm, vv in policy_fs.items():
            if nm in idx:
                bounds[idx[nm]] = (float(vv), float(vv))

        res = linprog(
            c=lp["c"],
            A_ub=lp["A_ub"],
            b_ub=lp["b_ub"],
            A_eq=lp["A_eq"],
            b_eq=lp["b_eq"],
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            profits.append(-1e9)
        else:
            profits.append(-float(res.fun))  # profit = -loss

    profits = np.array(profits, dtype=float)
    exp_profit = float(np.sum(probs * profits))
    var_profit, cvar_profit = weighted_var_cvar_profit(profits, probs, alpha)
    return {"expected_profit": exp_profit, "var_profit": var_profit, "cvar_profit": cvar_profit}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n_scenarios", type=int, default=75)
    p.add_argument("--alpha_tail", type=float, default=0.10)
    p.add_argument("--import_cap_risk", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.90)

    p.add_argument("--model_path", default="outputs/models/policy_enet.joblib")
    p.add_argument("--out_dir", default="outputs/figures")

    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0],
    )
    args = p.parse_args()

    data = load_data(args.data_dir)
    base = default_params(data)
    first_stage_vars = default_first_stage_vars(data)

    scenarios = generate_scenarios(
        base=base,
        n=args.n_scenarios,
        seed=args.seed,
        alpha_tail=args.alpha_tail,
        enable_import_cap_risk=bool(args.import_cap_risk),
    )
    probs = np.array([s.prob for s in scenarios], dtype=float)
    probs = probs / probs.sum()

    # Load model bundle
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    meta = bundle.get("meta", {})

    # Distribution features for this scenario set
    Xdist = extract_distribution_features(base, scenarios)

    rows = []
    for lam in args.lambdas:
        # ---- Oracle ----
        t0 = time.time()
        oracle = solve_oracle(
            data=data,
            base_params=base,
            scenarios=scenarios,
            alpha=args.alpha,
            lam=float(lam),
            first_stage_vars=first_stage_vars,
        )
        t_oracle = time.time() - t0

        # ---- ML policy + restore + recourse evaluation ----
        # Start with zeros for all features to avoid NaNs
        xrow = {c: 0.0 for c in feature_cols}

        # Fill distribution features we computed
        for k, v in Xdist.items():
            if k in xrow:
                xrow[k] = float(v)

        # Fill standard meta features if they exist in feature_cols
        xrow["lam"] = float(lam)
        if "seed" in xrow:
            xrow["seed"] = float(args.seed)
        if "n_scenarios" in xrow:
            xrow["n_scenarios"] = float(args.n_scenarios)
        if "alpha" in xrow:
            xrow["alpha"] = float(args.alpha)
        if "alpha_tail" in xrow:
            xrow["alpha_tail"] = float(args.alpha_tail)
        if "import_cap_risk" in xrow:
            xrow["import_cap_risk"] = float(args.import_cap_risk)
        if "scenario_set_id" in xrow:
            xrow["scenario_set_id"] = 0.0  # not meaningful here

        X = pd.DataFrame([xrow], columns=feature_cols)


        t1 = time.time()
        yhat = model.predict(X)[0]
        yhat_policy = {first_stage_vars[i]: float(yhat[i]) for i in range(len(first_stage_vars))}
        yhat_policy = project_first_stage(yhat_policy, data, base)
        policy = restore_first_stage(
            data=data,
            params=base,
            scenarios_params=scenarios[0].params,
            first_stage_vars=first_stage_vars,
            yhat=yhat_policy,
            w=1.0,
        )
        econ = eval_policy_on_scenarios(data, scenarios, alpha=args.alpha, policy_fs=policy)
        t_ml = time.time() - t1

        rows.append(
            {
                "lam": float(lam),
                "oracle_expected_profit": oracle["metrics"]["expected_profit"],
                "oracle_cvar_profit": oracle["metrics"]["cvar_profit"],
                "ml_expected_profit": econ["expected_profit"],
                "ml_cvar_profit": econ["cvar_profit"],
                "gap_expected_profit": econ["expected_profit"] - oracle["metrics"]["expected_profit"],
                "gap_cvar_profit": econ["cvar_profit"] - oracle["metrics"]["cvar_profit"],
                "time_oracle_s": t_oracle,
                "time_ml_s": t_ml,
            }
        )

    df = pd.DataFrame(rows)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_dir) / f"frontier_compare_seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)

    # ---- Plot 1: Expected profit vs lambda ----
    plt.figure()
    plt.plot(df["lam"], df["oracle_expected_profit"], marker="o", label="Oracle (CVaR LP)")
    plt.plot(df["lam"], df["ml_expected_profit"], marker="o", label="ML policy + restore")
    plt.xlabel("lambda")
    plt.ylabel("Expected profit")
    plt.title(f"Expected profit vs lambda (seed={args.seed})")
    plt.legend()
    f1 = Path(args.out_dir) / f"expected_profit_vs_lambda_seed{args.seed}.png"
    plt.savefig(f1, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: CVaR profit vs lambda ----
    plt.figure()
    plt.plot(df["lam"], df["oracle_cvar_profit"], marker="o", label="Oracle (CVaR LP)")
    plt.plot(df["lam"], df["ml_cvar_profit"], marker="o", label="ML policy + restore")
    plt.xlabel("lambda")
    plt.ylabel(f"CVaR_{args.alpha} profit")
    plt.title(f"CVaR profit vs lambda (seed={args.seed})")
    plt.legend()
    f2 = Path(args.out_dir) / f"cvar_profit_vs_lambda_seed{args.seed}.png"
    plt.savefig(f2, bbox_inches="tight")
    plt.close()

    # ---- Plot 3: Frontier (Expected vs CVaR) ----
    plt.figure()
    plt.plot(df["oracle_cvar_profit"], df["oracle_expected_profit"], marker="o", label="Oracle (CVaR LP)")
    plt.plot(df["ml_cvar_profit"], df["ml_expected_profit"], marker="o", label="ML policy + restore")
    plt.xlabel(f"CVaR_{args.alpha} profit")
    plt.ylabel("Expected profit")
    plt.title(f"Efficient frontier comparison (seed={args.seed})")
    plt.legend()
    f3 = Path(args.out_dir) / f"frontier_oracle_vs_ml_seed{args.seed}.png"
    plt.savefig(f3, bbox_inches="tight")
    plt.close()

    print("Wrote:", out_csv)
    print("Wrote:", f1)
    print("Wrote:", f2)
    print("Wrote:", f3)
    print("\nTiming summary (mean over lambdas):")
    print(df[["time_oracle_s", "time_ml_s"]].mean())


if __name__ == "__main__":
    main()
