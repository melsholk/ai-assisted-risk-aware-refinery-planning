from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.model import load_data, default_params
from src.scenarios import generate_scenarios
from src.oracle import solve_oracle, default_first_stage_vars
from src.ml.features import extract_distribution_features


def main(
    data_dir = "data",
    out_path = "data/ml/train.parquet",
    meta_path = "data/ml/meta.json",
    n_sets = 200,
    n_scenarios = 75,
    seed0 = 1,
    alpha_tail = 0.10,
    enable_import_cap_risk = True,
    alpha = 0.90,
    lambdas = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0),
):
    data = load_data(data_dir)
    base = default_params(data)
    first_stage_vars = default_first_stage_vars(data)

    rows = []
    for k in range(n_sets):
        seed = seed0 + k
        scenarios = generate_scenarios(
            base=base,
            n=n_scenarios,
            seed=seed,
            alpha_tail=alpha_tail,
            enable_import_cap_risk=enable_import_cap_risk,
        )
        X = extract_distribution_features(base, scenarios)

        for lam in lambdas:
            out = solve_oracle(
                data=data,
                base_params=base,
                scenarios=scenarios,
                alpha=alpha,
                lam=lam,
                first_stage_vars=first_stage_vars,
            )
            if not out["success"]:
                print(f"[WARN] set={k} lam={lam}: {out['message']}")
                continue

            row = {
                "scenario_set_id": k,
                "seed": seed,
                "n_scenarios": n_scenarios,
                "alpha": alpha,
                "alpha_tail": alpha_tail,
                "import_cap_risk": int(enable_import_cap_risk),
                "lam": lam,
            }
            row.update(X)

            # labels (first-stage policy)
            for v in first_stage_vars:
                row[f"y__{v}"] = float(out["policy"].get(v, 0.0))

            # store oracle metrics too (useful for evaluation later)
            for mk, mv in out["metrics"].items():
                row[f"oracle__{mk}"] = float(mv)

            rows.append(row)

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    meta = {
        "feature_cols": [c for c in df.columns if not c.startswith("y__") and not c.startswith("oracle__")],
        "label_cols": [c for c in df.columns if c.startswith("y__")],
        "metric_cols": [c for c in df.columns if c.startswith("oracle__")],
        "first_stage_vars": first_stage_vars,
    }
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    Path(meta_path).write_text(json.dumps(meta, indent=2))

    print(f"Wrote {len(df)} rows -> {out_path}")
    print(df.head(3).T)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()

    # Core dataset controls
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_path", default="data/ml/train.parquet")
    p.add_argument("--meta_path", default="data/ml/meta.json")

    p.add_argument("--n_sets", type=int, default=200)
    p.add_argument("--n_scenarios", type=int, default=75)
    p.add_argument("--seed0", type=int, default=1)

    p.add_argument("--alpha_tail", type=float, default=0.10)
    p.add_argument("--enable_import_cap_risk", type=int, default=1)

    p.add_argument("--alpha", type=float, default=0.90)

    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0],
    )

    args = p.parse_args()

    main(
        data_dir=args.data_dir,
        out_path=args.out_path,
        meta_path=args.meta_path,
        n_sets=args.n_sets,
        n_scenarios=args.n_scenarios,
        seed0=args.seed0,
        alpha_tail=args.alpha_tail,
        enable_import_cap_risk=bool(args.enable_import_cap_risk),
        alpha=args.alpha,
        lambdas=tuple(args.lambdas),
    )

