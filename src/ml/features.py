from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from src.model import LPParams
from src.scenarios import Scenario


def _weighted_mean(x, w):
    return float(np.sum(w * x))


def _weighted_std(x, w):
    mu = _weighted_mean(x, w)
    var = float(np.sum(w * (x - mu) ** 2))
    return float(np.sqrt(max(var, 0.0)))


def _weighted_quantile(x, w, q):
    """Weighted quantile for q in [0,1]."""
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws)
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(max(idx, 0), len(xs) - 1)
    return float(xs[idx])


def _weighted_corr(x, y, w):
    mx = _weighted_mean(x, w)
    my = _weighted_mean(y, w)
    cov = float(np.sum(w * (x - mx) * (y - my)))
    sx = _weighted_std(x, w)
    sy = _weighted_std(y, w)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(cov / (sx * sy))


def extract_distribution_features(base, scenarios):
    """
    Scenario-set (distribution) features.
    Output is a flat dict suitable for ML.
    """
    probs = np.array([s.prob for s in scenarios], dtype=float)
    probs = probs / probs.sum()

    feats: Dict[str, float] = {}
    n = len(scenarios)
    feats["n_scenarios"] = float(n)

    # --- Product netback multipliers (by product, aggregated across markets) ---
    # base.netback and scenario.params.netback are keyed by (p, m)
    products = sorted({p for (p, m) in base.netback.keys()})
    markets_by_p: Dict[str, List[str]] = {}
    for (p, m) in base.netback.keys():
        markets_by_p.setdefault(p, []).append(m)

    # Build multiplier matrix: scen x product
    for p in products:
        ms = markets_by_p[p]
        # base average netback for this product
        base_vals = np.array([base.netback[(p, m)] for m in ms], dtype=float)
        base_avg = float(np.mean(base_vals)) if len(base_vals) else 1.0

        mult = []
        for s in scenarios:
            vals = np.array([s.params.netback[(p, m)] for m in ms], dtype=float)
            avg = float(np.mean(vals)) if len(vals) else base_avg
            mult.append(avg / base_avg if abs(base_avg) > 1e-12 else 1.0)
        mult = np.array(mult, dtype=float)

        feats[f"nb_{p}_mean"] = _weighted_mean(mult, probs)
        feats[f"nb_{p}_std"]  = _weighted_std(mult, probs)
        feats[f"nb_{p}_p10"]  = _weighted_quantile(mult, probs, 0.10)
        feats[f"nb_{p}_p05"]  = _weighted_quantile(mult, probs, 0.05)

    # A few correlation features among canonical products if present
    canon = ["G", "U", "J", "L", "F"]
    present = [p for p in canon if p in products]
    # precompute multipliers again for these to avoid recomputation
    mult_by_p: Dict[str, np.ndarray] = {}
    for p in present:
        ms = markets_by_p[p]
        base_avg = float(np.mean([base.netback[(p, m)] for m in ms]))
        arr = []
        for s in scenarios:
            avg = float(np.mean([s.params.netback[(p, m)] for m in ms]))
            arr.append(avg / base_avg if abs(base_avg) > 1e-12 else 1.0)
        mult_by_p[p] = np.array(arr, dtype=float)

    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            p1, p2 = present[i], present[j]
            feats[f"nb_corr_{p1}_{p2}"] = _weighted_corr(mult_by_p[p1], mult_by_p[p2], probs)

    # --- Crude multipliers ---
    crudes = sorted(base.crude_cost.keys())
    for c in crudes:
        base_c = float(base.crude_cost[c])
        mult = np.array([s.params.crude_cost[c] / base_c if abs(base_c) > 1e-12 else 1.0 for s in scenarios], dtype=float)
        feats[f"crude_{c}_mean"] = _weighted_mean(mult, probs)
        feats[f"crude_{c}_std"]  = _weighted_std(mult, probs)
        feats[f"crude_{c}_p90"]  = _weighted_quantile(mult, probs, 0.90)

    # --- Hydrogen ---
    h2_av = np.array([s.params.h2_avail for s in scenarios], dtype=float)
    feats["h2_avail_mean"] = _weighted_mean(h2_av, probs)
    feats["h2_avail_std"]  = _weighted_std(h2_av, probs)
    feats["h2_avail_p10"]  = _weighted_quantile(h2_av, probs, 0.10)

    base_h2_cost = float(base.h2_buy_cost)
    h2_cost_mult = np.array([s.params.h2_buy_cost / base_h2_cost if abs(base_h2_cost) > 1e-12 else 1.0 for s in scenarios], dtype=float)
    feats["h2_cost_mult_mean"] = _weighted_mean(h2_cost_mult, probs)
    feats["h2_cost_mult_std"]  = _weighted_std(h2_cost_mult, probs)
    feats["h2_cost_mult_p90"]  = _weighted_quantile(h2_cost_mult, probs, 0.90)

    # --- Spot import disruption features (imp_spot_max changes) ---
    # Compare scenario spot cap vs base to infer disruptions
    if hasattr(base, "imp_spot_max") and len(base.imp_spot_max) > 0:
        keys = list(base.imp_spot_max.keys())
        base_caps = np.array([base.imp_spot_max[k] for k in keys], dtype=float)
        cap_mult = []
        disrupted = []
        for s in scenarios:
            caps = np.array([s.params.imp_spot_max[k] for k in keys], dtype=float)
            m = np.minimum(caps / np.maximum(base_caps, 1e-12), 10.0)
            cap_mult.append(float(np.mean(m)))
            disrupted.append(float(np.any(m < 0.999)))
        cap_mult = np.array(cap_mult, dtype=float)
        disrupted = np.array(disrupted, dtype=float)

        feats["spot_cap_mult_mean"] = _weighted_mean(cap_mult, probs)
        feats["spot_cap_mult_p10"]  = _weighted_quantile(cap_mult, probs, 0.10)
        feats["spot_disruption_rate"] = _weighted_mean(disrupted, probs)

    return feats
