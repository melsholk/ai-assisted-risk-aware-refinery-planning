from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

from src.model import LPParams


@dataclass(frozen=True)
class Scenario:
    name: str
    prob: float
    params: LPParams


def _copy_with(base, **overrides):
    d = dict(base.__dict__)
    d.update(overrides)
    return LPParams(**d)


def generate_scenarios(
    base,
    n = 50,
    seed = 1,
    alpha_tail = 0.10,
    enable_import_cap_risk = False,
):
    """
    Scenario generator with:
      - 2-factor netback shocks (market level M, spread tilt S)
      - crude diff shocks (correlated with M)
      - hydrogen shocks
      - OPTIONAL: SPOT import disruptions via imp_spot_max (not total imp_max)
    """
    rng = np.random.default_rng(seed)

    # --- Products present in the model ---
    prods_present = sorted({p for (p, m) in base.netback.keys()})
    canonical = ["G", "U", "J", "L", "F"]
    prods = [p for p in canonical if p in prods_present] + [p for p in prods_present if p not in canonical]

    # ---------- Netback factor model ----------
    sigma_M = 0.12
    sigma_S = 0.20  # keep your updated spread volatility

    sigma_eps = {p: 0.08 for p in prods}
    sigma_eps.update({"G": 0.10, "U": 0.09, "J": 0.07, "L": 0.12, "F": 0.14})

    a = {p: 0.9 for p in prods}
    b = {p: 0.0 for p in prods}
    if "G" in prods: b["G"] = +0.8
    if "U" in prods: b["U"] = -0.8
    if "J" in prods: b["J"] = -0.2
    if "L" in prods: b["L"] = +0.1
    if "F" in prods: b["F"] = -0.1

    # crude correlation with market level
    rho_crude_M = 0.35
    crudes = sorted(base.crude_cost.keys())
    crude_rel_sigma = 0.08

    # Hydrogen
    h2_avail_sigma = max(0.10 * base.h2_avail, 0.5)
    h2_price_rel_sigma = 0.20

    # Mean correction for lognormal multipliers
    var_log = {}
    for p in prods:
        var_log[p] = (a[p] ** 2) * (sigma_M ** 2) + (b[p] ** 2) * (sigma_S ** 2) + (sigma_eps[p] ** 2)

    raw_params = []
    stress_scores = []

    for i in range(n):
        M = rng.normal(0.0, sigma_M)
        S = rng.normal(0.0, sigma_S)

        mult_prod: Dict[str, float] = {}
        for p in prods:
            eps = rng.normal(0.0, sigma_eps[p])
            log_mult = a[p] * M + b[p] * S + eps
            mult_prod[p] = float(np.exp(log_mult - 0.5 * var_log[p]))

        nb = dict(base.netback)
        for (p, m), v in base.netback.items():
            if p in mult_prod:
                nb[(p, m)] = float(v * mult_prod[p])

        # Crude costs with correlation to M
        cc = dict(base.crude_cost)
        M_std = (M / sigma_M) if sigma_M > 0 else 0.0
        for c in crudes:
            z_ind = rng.normal(0.0, crude_rel_sigma)
            z_corr = rho_crude_M * (M_std * crude_rel_sigma) + np.sqrt(max(1.0 - rho_crude_M ** 2, 0.0)) * z_ind
            cc[c] = float(base.crude_cost[c] * np.exp(z_corr))

        # Hydrogen
        h2_av = float(max(0.0, base.h2_avail + rng.normal(0.0, h2_avail_sigma)))
        h2_buy_cost = float(base.h2_buy_cost * np.exp(rng.normal(0.0, h2_price_rel_sigma)))

        # TOTAL import caps are stable (contracts exist), BUT SPOT availability can disrupt
        imp_spot_max = dict(base.imp_spot_max)

        if enable_import_cap_risk and len(imp_spot_max) > 0:
            # With some probability, spot market tightens for one (p,m)
            if rng.random() < 0.20:
                key = list(imp_spot_max.keys())[rng.integers(0, len(imp_spot_max))]
                # severe spot tightening: 70-95% cut
                imp_spot_max[key] = float(imp_spot_max[key] * rng.uniform(0.05, 0.30))

        sp = _copy_with(
            base,
            netback=nb,
            crude_cost=cc,
            h2_avail=h2_av,
            h2_buy_cost=h2_buy_cost,
            imp_spot_max=imp_spot_max,
        )
        raw_params.append(sp)

        # Stress score for probability tilt
        avg_mult = float(np.mean([mult_prod[p] for p in prods])) if len(prods) else 1.0
        crude_mult = float(np.mean([cc[c] / base.crude_cost[c] for c in crudes])) if len(crudes) else 1.0
        h2_mult = float((h2_av + 1e-9) / (base.h2_avail + 1e-9)) if base.h2_avail > 0 else 1.0
        stress = (1.0 / max(avg_mult, 1e-6)) + crude_mult + (1.0 / max(h2_mult, 1e-6))
        stress_scores.append(stress)

    # Probability tilt toward tail stress scenarios
    stress_scores = np.array(stress_scores, dtype=float)
    thresh = float(np.quantile(stress_scores, 1.0 - alpha_tail))
    w = np.ones(n, dtype=float)
    w[stress_scores >= thresh] = 2.0
    probs = w / w.sum()

    scenarios = []
    for i in range(n):
        scenarios.append(Scenario(name=f"s{i:03d}", prob=float(probs[i]), params=raw_params[i]))

    return scenarios
