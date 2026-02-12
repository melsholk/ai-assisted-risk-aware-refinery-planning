from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from src.model import load_data, default_params, build_lp_matrices
from src.scenarios import generate_scenarios, Scenario
from src.cvar import solve_cvar_extensive_form


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Weighted quantile of values at probability q in [0,1].
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1]")
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) == 0:
        return float("nan")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    if w.sum() <= 0:
        raise ValueError("sum(weights) must be positive")

    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w) / w.sum()
    return float(np.interp(q, cw, v))


def weighted_cvar_tail(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Weighted CVaR of values in the LOWER tail with mass (1-q).
    Here values are profits, so this returns average of worst profits.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) == 0:
        return float("nan")
    if w.sum() <= 0:
        return float("nan")

    # VaR threshold in profit space
    var_q = weighted_quantile(v, w, q)

    # Tail set: v <= var_q
    mask = v <= var_q + 1e-12
    if not np.any(mask):
        return float(var_q)

    v_tail = v[mask]
    w_tail = w[mask]
    if w_tail.sum() <= 0:
        return float(var_q)

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
    args = ap.parse_args()

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

    # Build first-stage var list once (stable ordering required)
    lp0 = build_lp_matrices(data, params=scenarios[0].params)
    first_stage_vars = [
        v for v in lp0["var_names"]
        if v.startswith("x[") or v.startswith("T[") or v.startswith("imp_fs[")
    ]

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

        # Expected profit is always exact from scenario losses (via r.expected_loss)
        expected_profit = -r.expected_loss

        # Compute VaR/CVaR in profit space:
        # - if lam == 0, z/s are not incentivized -> compute post-solve from scenario profits
        # - if lam > 0, use LP-exact z and cvar_loss
        profits = -np.array([r.scenario_loss[s.name] for s in scenarios], dtype=float)

        if lam <= 1e-12:
            var_profit = weighted_quantile(profits, p, args.alpha)
            cvar_profit = weighted_cvar_tail(profits, p, args.alpha)
        else:
            var_profit = -r.z
            cvar_profit = -r.cvar_loss

        # Import contracting diagnostics (first-stage totals)
        imp_fs_total = 0.0
        for k, v in r.first_stage.items():
            if k.startswith("imp_fs["):
                imp_fs_total += float(v)

        # Spot is second-stage (scenario-dependent). But we can still diagnose:
        #  - spot used in each scenario (from solution losses we don't have volumes)
        # So: compute spot-cap binding freq using scenario params and scenario-wise LP solve
        # (Fast method: rebuild each scenario LP and solve with first-stage fixed would be heavy.)
        # Instead, we approximate binding freq by checking if spot is "needed" via cap tightness:
        # We'll do a light diagnostic: for each scenario, solve *its own* deterministic LP with
        # first-stage decisions fixed (non-anticipativity), then see if any imp_spot hits cap.
        #
        # This is still LP and OK for n~100; if it becomes slow, we can sample scenarios.

        # Fix first-stage decisions in a per-scenario LP and re-solve
        spot_bind_count = 0
        spot_used_total = 0.0
        spot_cap_total = 0.0

        # Build list of first-stage values to fix
        fs_values = {k: float(v) for k, v in r.first_stage.items()}

        for sc in scenarios:
            lp = build_lp_matrices(data, params=sc.params)
            var_names = lp["var_names"]
            idx = lp["idx"]
            bounds = list(lp["bounds"])

            # Fix first-stage vars by setting bounds [val, val]
            for nm, vv in fs_values.items():
                if nm in idx:
                    bounds[idx[nm]] = (vv, vv)

            # Solve scenario recourse LP
            from scipy.optimize import linprog
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
                # if recourse infeasible, count as bind-like stress (optional)
                continue

            x = res.x
            # Check any spot import variable hitting its spot cap (within tolerance)
            tol = 1e-6
            for (pm, capv) in sc.params.imp_spot_max.items():
                p_, m_ = pm
                nm = f"imp_spot[{p_},{m_}]"
                if nm in idx:
                    val = float(x[idx[nm]])
                    spot_used_total += val
                    spot_cap_total += float(capv)
                    if float(capv) > 0 and val >= float(capv) - tol:
                        spot_bind_count += 1
                        break  # count scenario once

        spot_bind_freq = spot_bind_count / max(len(scenarios), 1)

        rows.append(
            {
                "lam": lam,
                "expected_profit": expected_profit,
                f"cvar_profit_a{args.alpha:.2f}": cvar_profit,
                f"var_profit_a{args.alpha:.2f}": var_profit,
                "imp_fs_total": imp_fs_total,
                "spot_bind_freq": spot_bind_freq,
                "spot_used_avg": spot_used_total / max(len(scenarios), 1),
                "T_FCC_D": r.first_stage.get("T[FCC_D]", np.nan),
                "T_FCC_G": r.first_stage.get("T[FCC_G]", np.nan),
                "T_HC": r.first_stage.get("T[HC]", np.nan),
                "T_VR": r.first_stage.get("T[VR]", np.nan),
                "T_REF": r.first_stage.get("T[REF]", np.nan),
            }
        )

    df = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 80)

    print("\n=== Efficient Frontier (VaR/CVaR correct at lam=0; import contracting diagnostics) ===")

    def _fmt(v):
        if isinstance(v, (int, float, np.floating)):
            return f"{v:,.2f}"
        return str(v)

    print(df.to_string(index=False, formatters={c: _fmt for c in df.columns}))


if __name__ == "__main__":
    main()
