from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


def split_by_scenario_set(df, test_frac = 0.25, seed = 0):
    rng = np.random.default_rng(seed)
    ids = df["scenario_set_id"].unique()
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_frac))
    test_ids = set(ids[:n_test])
    test = df[df["scenario_set_id"].isin(test_ids)].copy()
    train = df[~df["scenario_set_id"].isin(test_ids)].copy()
    return train, test


def train_and_report(name, model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)

    # Aggregate metrics
    mae = mean_absolute_error(Y_test, pred)
    rmse = np.sqrt(mean_squared_error(Y_test, pred))

    # Per-target MAE (useful diagnostics)
    per_mae = np.mean(np.abs(Y_test - pred), axis=0)

    print(f"\n=== {name} ===")
    print(f"MAE  (avg over all outputs): {mae:.4f}")
    print(f"RMSE (avg over all outputs): {rmse:.4f}")
    print("Top-10 worst targets by MAE:")
    worst_idx = np.argsort(-per_mae)[:10]
    for i in worst_idx:
        print(f"  {Y_test.columns[i]}  MAE={per_mae[i]:.4f}")

    return model


def main(
    data_path = "data/ml/tiny.parquet",
    meta_path = "data/ml/meta_tiny.json",
    out_dir = "outputs/models",
    test_frac = 0.25,
    seed = 0,
):
    df = pd.read_parquet(data_path)
    meta = json.loads(Path(meta_path).read_text())

    feature_cols = meta["feature_cols"]
    label_cols = meta["label_cols"]

    # Basic sanity: ensure lam is in features
    if "lam" not in feature_cols:
        raise RuntimeError("Expected 'lam' to be in feature_cols. Check meta json.")

    # Split by scenario_set_id to avoid leakage
    train_df, test_df = split_by_scenario_set(df, test_frac=test_frac, seed=seed)

    X_train = train_df[feature_cols]
    Y_train = train_df[label_cols]
    X_test = test_df[feature_cols]
    Y_test = test_df[label_cols]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Model 1: ElasticNet (interpretable baseline) ----
    enet = MultiOutputRegressor(
        Pipeline(
            steps=[
                ("x_scaler", StandardScaler(with_mean=True, with_std=True)),
                ("reg", ElasticNet(alpha=5e-3, l1_ratio=0.3, max_iter=20000, random_state=seed)),
            ]
        )
    )

    enet = train_and_report("MultiOutput ElasticNet", enet, X_train, Y_train, X_test, Y_test)
    joblib.dump(
        {"model": enet, "feature_cols": feature_cols, "label_cols": label_cols, "meta": meta},
        Path(out_dir) / "policy_enet.joblib",
    )

    # ---- Model 2: Gradient boosting (strong tabular baseline) ----
    # HistGradientBoostingRegressor is single-output; wrap in MultiOutputRegressor
    hgb = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.06,
            max_iter=500,
            random_state=seed,
        )
    )

    hgb = train_and_report("MultiOutput HistGradientBoosting", hgb, X_train, Y_train, X_test, Y_test)
    joblib.dump(
        {"model": hgb, "feature_cols": feature_cols, "label_cols": label_cols, "meta": meta},
        Path(out_dir) / "policy_hgb.joblib",
    )

    print(f"\nSaved models to: {out_dir}")
    print("Next: add projection + economic evaluation (expected profit + CVaR gap).")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/ml/train.parquet")
    p.add_argument("--meta_path", default="data/ml/meta.json")
    p.add_argument("--out_dir", default="outputs/models")
    p.add_argument("--test_frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(**vars(args))

