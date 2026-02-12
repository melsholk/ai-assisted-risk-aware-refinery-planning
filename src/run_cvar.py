from __future__ import annotations

import argparse
import numpy as np

from src.model import load_data, default_params, build_and_solve_lp, build_lp_matrices
from src.scenarios import generate_scenarios, Scenario
from src.cvar import solve_cvar_extensive_form


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--import_cap_risk", action="store_true")
    ap.add_argument("--include_base", action="store_true", help="Include deterministic base as scenario 'base'")
    ap.add_argument("--base_only", action="store_true", help="Solve CVaR with ONLY the deterministic base scenario")
    args = ap.parse_args()

    data = load_data(args.data_dir)
    base = default_params(data)

    if args.base_only:
        scenarios = [Scenario(name="base", prob=1.0, params=base)]
    else:
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

    # First-stage vars: crude + unit throughputs + contracted imports
    lp0 = build_lp_matrices(data, params=scenarios[0].params)
    first_stage_vars = [
        v for v in lp0["var_names"]
        if v.startswith("x[") or v.startswith("T[") or v.startswith("imp_fs[")
    ]

    r = solve_cvar_extensive_form(
        data=data,
        scenarios=scenarios,
        alpha=args.alpha,
        lam=args.lam,
        first_stage_vars=first_stage_vars,
    )

    if not r.success:
        print("\nCVaR solve FAILED")
        print(r.message)
        return

    print("\n=== CVaR EXTENSIVE FORM SOLUTION ===")
    print(f"alpha: {r.alpha:.3f}   lambda: {r.lam:.3f}")
    print(f"Objective (expected loss + lambda*CVaR): {r.objective:,.4f}")

    print("\n--- First-stage decisions (shared across scenarios) ---")
    for k in sorted(r.first_stage.keys()):
        print(f"{k:18s}  {r.first_stage[k]:,.4f}")

    losses = np.array([r.scenario_loss[s.name] for s in scenarios], dtype=float)
    print("\n--- Scenario loss summary (loss = -profit) ---")
    print(f"min/median/max loss: {losses.min():,.2f} / {np.median(losses):,.2f} / {losses.max():,.2f}")

    print("\n--- Risk metrics (profit space) ---")
    print(f"Expected profit (USDk/day): {-r.expected_loss:,.2f}")
    print(f"VaR_{r.alpha:.2f} profit (USDk/day): {-r.z:,.2f}")
    print(f"CVaR_{r.alpha:.2f} profit (USDk/day): {-r.cvar_loss:,.2f}")

    if any(s.name == "base" for s in scenarios):
        det = build_and_solve_lp(data, params=base, verbose=False)
        det_profit = float(det["economics"].set_index("item").loc["Profit", "USDk_per_day"])
        print(f"\nDeterministic base profit (USDk/day): {det_profit:,.2f}")


if __name__ == "__main__":
    main()
