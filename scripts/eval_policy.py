from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model import load_data, default_params
from src.scenarios import generate_scenarios
from src.ml.features import extract_distribution_features
from src.ml.projection import project_first_stage
from scipy.optimize import linprog
from src.model import build_lp_matrices
from src.ml.restore import restore_first_stage




def weighted_var_cvar_profit(profits, probs, alpha):
    order = np.argsort(profits)  # worst first
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


def eval_one_instance(data, scenarios, alpha, policy_fs):
    """
    Fix first-stage vars to policy_fs and solve each scenario recourse LP.
    Mirrors the diagnostic logic in src/run_frontier.py. 
    """
    probs = np.array([s.prob for s in scenarios], dtype=float)
    probs = probs / probs.sum()

    profits = []
    for s in scenarios:
        lp = build_lp_matrices(data, params=s.params)
        idx = lp["idx"]
        bounds = list(lp["bounds"])

        # Fix first-stage vars by tightening bounds
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
            continue

        # Your LP is a minimization of loss = -profit, so profit = -obj
        profit = -float(res.fun)
        profits.append(profit)

    profits = np.array(profits, dtype=float)
    exp_profit = float(np.sum(probs * profits))
    var_profit, cvar_profit = weighted_var_cvar_profit(profits, probs, alpha)

    return {
        "expected_profit": exp_profit,
        "var_profit": var_profit,
        "cvar_profit": cvar_profit,
    }

def split_by_scenario_set(df, test_frac = 0.25, seed = 0):
    rng = np.random.default_rng(seed)
    ids = df["scenario_set_id"].unique()
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_frac))
    test_ids = set(ids[:n_test])
    test = df[df["scenario_set_id"].isin(test_ids)].copy()
    train = df[~df["scenario_set_id"].isin(test_ids)].copy()
    return train, test


def main(
    model_path = "outputs/models/policy_enet.joblib",
    data_path = "data/ml/tiny.parquet",
    meta_path = "data/ml/meta_tiny.json",
    data_dir = "data",
    alpha = 0.90,
    n_scenarios = 30,
    alpha_tail = 0.10,
    enable_import_cap_risk = True,
):
    bundle = joblib.load(model_path)
    model = bundle["model"]

    df = pd.read_parquet(data_path)
    meta = json.loads(Path(meta_path).read_text())
    feature_cols = meta["feature_cols"]
    label_cols = meta["label_cols"]
    first_stage_vars = meta["first_stage_vars"]


    train_df, test_df = split_by_scenario_set(df, test_frac=0.25, seed=0)
    print(f"Eval rows: {len(test_df)} (test only), Train rows: {len(train_df)}")
    df = test_df

    data = load_data(data_dir)
    base = default_params(data)

    results = []
    for idx, row in df.iterrows():
        # regenerate scenarios from the stored seed / settings (for tiny dataset)
        seed = int(row["seed"])
        scenarios = generate_scenarios(
            base=base,
            n=int(row["n_scenarios"]),
            seed=int(row["seed"]),
            alpha_tail=float(row["alpha_tail"]),
            enable_import_cap_risk=bool(row["import_cap_risk"]),
        )


        # Build features (must match training)
        Xdist = extract_distribution_features(base, scenarios)
        xrow = {c: row[c] for c in feature_cols if c in row.index}  # includes lam + ids
        # Replace any dist features from dataset with freshly computed ones (robustness)
        for k, v in Xdist.items():
            if k in xrow:
                xrow[k] = v

        X = pd.DataFrame([xrow], columns=feature_cols)
        yhat = model.predict(X)[0]

        yhat_policy = {first_stage_vars[i]: float(yhat[i]) for i in range(len(first_stage_vars))}

        # First do cheap projection (bounds/caps)
        yhat_policy = project_first_stage(yhat_policy, data, base)

        # Then do feasibility restoration using a representative params for this scenario set:
        # use the first scenario's params (or later: probability-weighted mean params)
        pred_policy = restore_first_stage(
            data=data,
            params=base,
            scenarios_params=scenarios[0].params,
            first_stage_vars=first_stage_vars,
            yhat=yhat_policy,
            w=1.0,
        )
        

        # Evaluate economics under recourse
        econ = eval_one_instance(data, scenarios, alpha=float(row["alpha"]), policy_fs=pred_policy)

        results.append({
            "scenario_set_id": int(row["scenario_set_id"]),
            "seed": seed,
            "lam": float(row["lam"]),
            "pred_expected_profit": econ["expected_profit"],
            "pred_cvar_profit": econ["cvar_profit"],
            "oracle_expected_profit": float(row["oracle__expected_profit"]),
            "oracle_cvar_profit": float(row["oracle__cvar_profit"]),
            "gap_expected_profit": econ["expected_profit"] - float(row["oracle__expected_profit"]),
            "gap_cvar_profit": econ["cvar_profit"] - float(row["oracle__cvar_profit"]),
        })

    out = pd.DataFrame(results)
    bad = (out["gap_expected_profit"] < -1e6).mean()
    print(f"Fraction of cases with gap < -1e6: {bad:.3f}")

    print(out.groupby("lam")[["gap_expected_profit", "gap_cvar_profit"]].mean())
    print("\nOverall:")
    print(out[["gap_expected_profit", "gap_cvar_profit"]].describe())

    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    out.to_csv("outputs/reports/eval_policy.csv", index=False)
    print("\nWrote outputs/reports/eval_policy.csv")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="outputs/models/policy_enet.joblib")
    p.add_argument("--data_path", default="data/ml/train.parquet")
    p.add_argument("--meta_path", default="data/ml/meta.json")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--alpha", type=float, default=0.90)
    args = p.parse_args()
    main(model_path=args.model_path, data_path=args.data_path, meta_path=args.meta_path, data_dir=args.data_dir, alpha=args.alpha)

