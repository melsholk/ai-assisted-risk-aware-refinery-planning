from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from src.model import load_data, default_params, build_lp_matrices
from src.scenarios import generate_scenarios, Scenario
from src.cvar import solve_cvar_extensive_form


def weighted_quantile(values, weights, q):
    """Weighted quantile at q in [0,1]."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    if len(v) == 0:
        return float("nan")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1]")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be nonnegative and sum to > 0")

    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w) / w.sum()
    return float(np.interp(q, cw, v))


def weighted_cvar_lower_tail(values, weights, tail_mass):
    """
    Weighted CVaR of the LOWER tail of `values`, with tail mass = tail_mass.
    Example: tail_mass = 0.10 => mean of worst 10% outcomes.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) == 0:
        return float("nan")
    if w.sum() <= 0:
        return float("nan")
    if not (0.0 < tail_mass <= 1.0):
        raise ValueError("tail_mass must be in (0,1]")

    q = tail_mass  # cutoff quantile for worst tail
    var = weighted_quantile(v, w, q)
    mask = v <= var + 1e-12
    if not np.any(mask):
        return float(var)

    v_tail = v[mask]
    w_tail = w[mask]
    if w_tail.sum() <= 0:
        return float(var)

    return float(np.sum(w_tail * v_tail) / np.sum(w_tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--lams", default="0,0.1,0.3,1,3,10", help="comma-separated lambda values")
    ap.add_argument("--import_cap_risk", action="store_true")
    ap.add_argument("--include_base", action="store_true", help="Include deterministic base as scenario 'base'")
    ap.add_argument("--out_csv", default="", help="Optional path to save results CSV (e.g., outputs/frontier.csv)")
    args = ap.parse_args()

    if not (0.0 < args.alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    data = load_data(args.data_dir)
    base = default_params(data)

    scenarios = generate_scenarios(
        base,
        n=args.n,
        seed=args.seed,
        enable_import_cap_risk=args.import_cap_risk,
    )

    if args.include_base:
        scenarios = [Scenario(name="base", prob=1.0, params=base)] + scenarios
        total = sum(s.prob for s in scenarios)
        scenarios = [Scenario(name=s.name, prob=s.prob / total, params=s.params) for s in scenarios]

    # Normalize probabilities defensively
    p = np.array([s.prob for s in scenarios], dtype=float)
    p = p / p.sum()

    lam_list = [float(x.strip()) for x in args.lams.split(",") if x.strip() != ""]
    rows = []

    # First-stage vars
    lp0 = build_lp_matrices(data, params=scenarios[0].params)
    first_stage_vars = [
        v for v in lp0["var_names"]
        if v.startswith("x[") or v.startswith("T[") or v.startswith("imp_fs[")
    ]

    tail_mass = 1.0 - args.alpha  # e.g., alpha=0.90 => worst 10%

    for lam in lam_list:
        r = solve_cvar_extensive_form(
            data=data,
            scenarios=scenarios,
            alpha=args.alpha,
            lam=lam,
            first_stage_vars=first_stage_vars,
        )

        if not r.success:
            rows.append({"lam": lam, "status": "FAIL", "msg": r.message})
            continue

        # Expected profit (exact)
        expected_profit = -r.expected_loss

        # Scenario profits (exact at optimum)
        profits = -np.array([r.scenario_loss[s.name] for s in scenarios], dtype=float)

        # Textbook downside metrics in PROFIT SPACE:
        # VaR_alpha(profit) = (1-alpha)-quantile of profit  (10th percentile if alpha=0.90)
        # CVaR_alpha(profit) = mean of worst (1-alpha) profits
        if lam <= 1e-12:
            var_profit = weighted_quantile(profits, p, tail_mass)
            cvar_profit = weighted_cvar_lower_tail(profits, p, tail_mass)
        else:
            # LP solves VaR/CVaR in LOSS space:
            # z = VaR_alpha(loss), CVaR_loss = z + sum p*s/(1-alpha)
            # Convert to profit space:
            var_profit = -r.z
            cvar_profit = -r.cvar_loss

        # Contracted import total (first-stage)
        imp_fs_total = sum(float(v) for k, v in r.first_stage.items() if k.startswith("imp_fs["))

        # Spot binding diagnostic (re-solve recourse with first-stage fixed)
        spot_bind_count = 0
        spot_used_total = 0.0
        fs_values = {k: float(v) for k, v in r.first_stage.items()}
        from scipy.optimize import linprog

        for sc in scenarios:
            lp = build_lp_matrices(data, params=sc.params)
            idx = lp["idx"]
            bounds = list(lp["bounds"])

            # Fix first-stage vars
            for nm, vv in fs_values.items():
                if nm in idx:
                    bounds[idx[nm]] = (vv, vv)

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
                continue

            x = res.x
            tol = 1e-6
            bound_hit = False

            for (pm, capv) in sc.params.imp_spot_max.items():
                p_, m_ = pm
                nm = f"imp_spot[{p_},{m_}]"
                if nm in idx:
                    val = float(x[idx[nm]])
                    spot_used_total += val
                    if float(capv) > 0 and val >= float(capv) - tol:
                        bound_hit = True

            if bound_hit:
                spot_bind_count += 1

        spot_bind_freq = spot_bind_count / max(len(scenarios), 1)
        spot_used_avg = spot_used_total / max(len(scenarios), 1)

        rows.append(
            {
                "lam": lam,
                "expected_profit": expected_profit,
                f"var_profit_a{args.alpha:.2f}": var_profit,
                f"cvar_profit_a{args.alpha:.2f}": cvar_profit,
                "imp_fs_total": imp_fs_total,
                "spot_bind_freq": spot_bind_freq,
                "spot_used_avg": spot_used_avg,
                "T_FCC_D": r.first_stage.get("T[FCC_D]", np.nan),
                "T_FCC_G": r.first_stage.get("T[FCC_G]", np.nan),
                "T_HC": r.first_stage.get("T[HC]", np.nan),
                "T_VR": r.first_stage.get("T[VR]", np.nan),
                "T_REF": r.first_stage.get("T[REF]", np.nan),
            }
        )

    df = pd.DataFrame(rows)

    if args.out_csv:
        import os
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df.to_csv(args.out_csv, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 80)

    print("\n=== Efficient Frontier (textbook VaR/CVaR: downside profit quantiles) ===")

    def _fmt(v):
        if isinstance(v, (int, float, np.floating)):
            return f"{v:,.2f}"
        return str(v)

    print(df.to_string(index=False, formatters={c: _fmt for c in df.columns}))


if __name__ == "__main__":
    main()
