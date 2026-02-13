from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.model import Data, LPParams, load_data, default_params
from src.scenarios import Scenario, generate_scenarios
from src.cvar import solve_cvar_extensive_form


def _weighted_var_cvar_profit(profits, probs, alpha):
    """
    VaR/CVaR in PROFIT space (lower tail):
      VaR_alpha(profit) = alpha-quantile of profit
      CVaR_alpha(profit) = E[profit | profit <= VaR_alpha(profit)] (discrete)
    probs must sum to 1.
    """
    order = np.argsort(profits)  # ascending profit = worst first
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


def default_first_stage_vars(data):
    # Keep consistent with src/cvar.py default + include contracted imports (imp_fs) as first-stage.
    # Your cvar.py default does crude x + unit throughputs only; we extend it.
    params = default_params(data)
    crudes = sorted(params.crude_cost.keys())
    fs = [f"x[{c}]" for c in crudes] + ["T[FCC_G]", "T[FCC_D]", "T[VR]", "T[REF]", "T[HC]", "T[HDT]"]

    # Contracted imports: imp_fs[p,m] for all keys that exist in params.imp_max
    # (Assumes model creates variables imp_fs[...] for each (p,m) in imp_max)
    for (p, m) in sorted(params.imp_max.keys()):
        fs.append(f"imp_fs[{p},{m}]")

    return fs


def solve_oracle(
    data,
    base_params,
    scenarios,
    alpha,
    lam,
    first_stage_vars = None,
):
    """
    OR 'truth' oracle:
      (scenario set, lam) -> optimal first-stage policy + profit/risk metrics.
    """
    if first_stage_vars is None:
        first_stage_vars = default_first_stage_vars(data)

    res = solve_cvar_extensive_form(
        data=data,
        scenarios=scenarios,
        alpha=alpha,
        lam=lam,
        first_stage_vars=first_stage_vars,
    )
    if not res.success:
        return {
            "success": False,
            "message": res.message,
            "alpha": alpha,
            "lam": lam,
        }

    # Convert loss = -profit
    probs = np.array([sc.prob for sc in scenarios], dtype=float)
    probs = probs / probs.sum()

    losses = np.array([res.scenario_loss[sc.name] for sc in scenarios], dtype=float)
    profits = -losses

    exp_profit = float(np.sum(probs * profits))
    var_profit, cvar_profit = _weighted_var_cvar_profit(profits, probs, alpha)

    return {
        "success": True,
        "message": res.message,
        "alpha": alpha,
        "lam": lam,
        "policy": dict(res.first_stage),          # var_name -> value
        "scenario_loss": dict(res.scenario_loss), # scenario_name -> loss
        "metrics": {
            "expected_profit": exp_profit,
            "var_profit": var_profit,
            "cvar_profit": cvar_profit,
            "expected_loss": float(res.expected_loss),
            "var_loss": float(res.z),
            "cvar_loss": float(res.cvar_loss),
        },
        "meta": {
            "first_stage_vars": list(first_stage_vars),
        },
    }


def quick_oracle_smoke_test(
    data_dir = "data",
    n = 50,
    seed = 1,
    alpha_tail = 0.10,
    enable_import_cap_risk = False,
    alpha = 0.90,
    lam = 1.0,
):
    """
    Convenience entry point for a fast sanity run.
    """
    data = load_data(data_dir)
    base = default_params(data)
    scenarios = generate_scenarios(
        base=base,
        n=n,
        seed=seed,
        alpha_tail=alpha_tail,
        enable_import_cap_risk=enable_import_cap_risk,
    )
    out = solve_oracle(data, base, scenarios, alpha=alpha, lam=lam)
    return out
